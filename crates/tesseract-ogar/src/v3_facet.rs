//! V3-substrate <-> Python-SDK parity probe support.
//!
//! Walks a loaded Tesseract `Network` byte stream (the REAL `/tmp/eng.lstm`),
//! builds one [`FacetCascade`] per node via the Core's proven
//! [`tesseract_core::network::NetworkHeader::to_facet`] (never hand-rolled
//! byte packing), and provides the canonical TSV line formatters + Python
//! decode harness both `examples/v3_facet_probe.rs` and
//! `tests/v3_facet_parity.rs` share — one source of truth, so the two call
//! sites can never drift against each other.
//!
//! # Why this module re-walks the header stream instead of reusing
//! `tesseract_ocr::Network`'s tree directly
//!
//! `tesseract_ocr::Network::from_le_bytes` (B1, `E-OCR-NETWORK-FORWARD-1`) is
//! the byte-parity-proven loader for the RUNNABLE compute tree, but its
//! `Node` enum only retains the fields the *forward pass* needs (`ntype`
//! implicitly, via the variant; `ni`/`no` recoverable via `Node::num_outputs`)
//! — it discards each non-root node's [`NetworkHeader`] (`training` /
//! `needs_backprop` / `network_flags` / cumulative `num_weights`) once the
//! subclass payload is parsed (see that crate's `network.rs::read_node`,
//! which binds the header to `_` for every non-root child). Those fields ARE
//! the V3 facet's tiers 2-5 ([`NetworkHeader::to_facet`]'s documented
//! per-tier projection), so a faithful per-node facet needs the full header
//! at EVERY node, not just the root.
//!
//! `tesseract-ogar` may not edit `tesseract-ocr` (a sibling agent owns that
//! crate), so [`collect_facets`] re-derives the SAME recursive dispatch
//! shape (documented in `tesseract_ocr::network`'s own module-level wire-
//! format doc comment) using ONLY already-proven, already-public primitives:
//! [`NetworkHeader::from_le_bytes`] for every header (Core,
//! `E-OCR-NETWORK-SINK-1`), [`Lstm::from_le_bytes`] /
//! [`WeightMatrix::from_le_bytes_prefix`] for the two variable-length leaf
//! payloads (recognizer Leaf 5 / Leaf 2, byte-parity proven), and fixed-size
//! skips for the three 2-`i32`-payload node kinds (Convolve/Maxpool/Reconfig)
//! plus the 5-`i32` `Input` `StaticShape` — all documented in
//! `tesseract_ocr::network`'s own module doc comment, so nothing here is
//! invented. [`collect_facets`] returns the total bytes consumed so its
//! caller can cross-check against `tesseract_ocr::Network::from_le_bytes`
//! (the SAME public API `network_dump` uses) — see `examples/v3_facet_probe.rs`.

use lance_graph_contract::facet::FacetCascade;
use lance_graph_contract::ogar_codebook::classid_canon;
use tesseract_core::network::{NetworkError, NetworkHeader, NetworkType, NETWORK_LAYER};
use tesseract_recognizer::{Lstm, RecognizerError, WeightMatrix};

/// `NF_LAYER_SPECIFIC_LR` (`network.h` `NetworkFlags`) — mirrors the private
/// constant of the same name in `tesseract_ocr::network` (that module cannot
/// export it without an edit this crate is not permitted to make this
/// session). Same documented flag value; used only to skip the trailing
/// `learning_rates_` vector a `Plumbing` node carries when the flag is set.
const NF_LAYER_SPECIFIC_LR: i32 = 64;

/// A failure walking the header stream.
#[derive(Debug)]
pub enum FacetWalkError {
    /// A base [`NetworkHeader`] failed to parse.
    Header(NetworkError),
    /// An [`Lstm`] payload failed to parse.
    Lstm(RecognizerError),
    /// A [`WeightMatrix`] payload failed to parse.
    WeightMatrix(RecognizerError),
    /// The byte stream ended before a fixed-size or count-prefixed field.
    UnexpectedEof,
    /// A node type this walk does not expect in a supported inference model
    /// (2-D `Par2dLstm`, `LstmSoftmax*`, `None`, `TensorFlow`), or a
    /// `Reversed` wrapper with a child count other than 1.
    Unsupported(&'static str),
}

impl std::fmt::Display for FacetWalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header(e) => write!(f, "network header: {e:?}"),
            Self::Lstm(e) => write!(f, "LSTM payload: {e}"),
            Self::WeightMatrix(e) => write!(f, "weight matrix payload: {e}"),
            Self::UnexpectedEof => write!(f, "byte stream ended mid-field"),
            Self::Unsupported(what) => write!(f, "unsupported node kind: {what}"),
        }
    }
}

impl std::error::Error for FacetWalkError {}

/// One walked network node: its [`NetworkType`] (diagnostic only — not part
/// of the facet payload itself, though it IS baked into
/// [`facet`](Self::facet)'s `facet_classid` custom-low half) and the
/// [`FacetCascade`] [`NetworkHeader::to_facet`] derived from the REAL header
/// this walk read for that node.
#[derive(Debug, Clone, Copy)]
pub struct NodeFacet {
    /// The node's layer type.
    pub ntype: NetworkType,
    /// The node's V3 content-blind facet.
    pub facet: FacetCascade,
}

/// A forward-only cursor over the header stream — mirrors the small private
/// `Cursor`/`ByteReader` idiom already used in `tesseract_ocr::network` and
/// `lance_graph_contract::network` (each Core-first module gets its own
/// minimal "just enough" byte reader; this is the `v3_facet`-scoped one).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.pos..]
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], FacetWalkError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(FacetWalkError::UnexpectedEof)?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or(FacetWalkError::UnexpectedEof)?;
        self.pos = end;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32, FacetWalkError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }
}

/// Walk the base-header stream of a serialized `Network` (the head of an
/// `.lstm` component), pre-order (a node's own facet is pushed before its
/// children are walked — matching `network_dump`'s `describe()` print
/// order), returning one [`NodeFacet`] per node and the total number of
/// bytes consumed by the whole tree.
///
/// # Errors
///
/// [`FacetWalkError`] on truncation or an unsupported node kind.
pub fn collect_facets(bytes: &[u8]) -> Result<(Vec<NodeFacet>, usize), FacetWalkError> {
    let mut cur = Cursor::new(bytes);
    let mut out = Vec::new();
    walk(&mut cur, &mut out)?;
    Ok((out, cur.pos))
}

fn walk(cur: &mut Cursor<'_>, out: &mut Vec<NodeFacet>) -> Result<(), FacetWalkError> {
    let (hdr, used) =
        NetworkHeader::from_le_bytes(cur.remaining()).map_err(FacetWalkError::Header)?;
    cur.pos += used;
    let ntype = hdr.ntype;
    let flags = hdr.network_flags;
    out.push(NodeFacet {
        ntype,
        facet: hdr.to_facet(),
    });

    match ntype {
        NetworkType::Input => {
            cur.take(20)?; // StaticShape: 5 x i32 (batch,height,width,depth,loss_type)
        }
        NetworkType::Series
        | NetworkType::Parallel
        | NetworkType::Replicated
        | NetworkType::ParRlLstm
        | NetworkType::ParUdLstm => {
            let count = cur.read_u32()?;
            for _ in 0..count {
                walk(cur, out)?;
            }
            skip_layer_lr(cur, flags)?;
        }
        NetworkType::XReversed | NetworkType::YReversed | NetworkType::XyTranspose => {
            let count = cur.read_u32()?;
            if count != 1 {
                return Err(FacetWalkError::Unsupported(
                    "Reversed node with child count != 1",
                ));
            }
            walk(cur, out)?;
            skip_layer_lr(cur, flags)?;
        }
        NetworkType::Convolve | NetworkType::Maxpool | NetworkType::Reconfig => {
            cur.take(8)?; // 2 x i32 (x_scale/half_x, y_scale/half_y)
        }
        NetworkType::Lstm | NetworkType::LstmSummary => {
            let (lstm, used) =
                Lstm::from_le_bytes(cur.remaining()).map_err(FacetWalkError::Lstm)?;
            cur.pos += used;
            // Mirror the main loader (network.rs): Lstm::from_le_bytes reads
            // only the 1-D four-gate prefix. A 2-D LSTM carries a 5th (GFS)
            // gate this walk cannot consume, so `used` would be short and the
            // walk would mis-parse the stream. Reject it here instead of
            // advancing a bogus offset (codex P2 on #15).
            let (ni, ns) = (lstm.num_inputs(), lstm.state_size());
            if hdr.ni as usize + 2 * ns == ni + ns {
                return Err(FacetWalkError::Unsupported("2-D LSTM (GFS gate)"));
            }
        }
        NetworkType::Logistic
        | NetworkType::PosClip
        | NetworkType::SymClip
        | NetworkType::Tanh
        | NetworkType::Relu
        | NetworkType::Linear
        | NetworkType::Softmax
        | NetworkType::SoftmaxNoCtc => {
            let (_, used) = WeightMatrix::from_le_bytes_prefix(cur.remaining())
                .map_err(FacetWalkError::WeightMatrix)?;
            cur.pos += used;
        }
        NetworkType::Par2dLstm
        | NetworkType::LstmSoftmax
        | NetworkType::LstmSoftmaxEncoded
        | NetworkType::None
        | NetworkType::TensorFlow => {
            return Err(FacetWalkError::Unsupported(ntype.type_name()));
        }
    }
    Ok(())
}

/// `Plumbing::DeSerialize` reads a trailing `learning_rates_` (`Vec<f32>`) —
/// `u32 count` + `count x f32` — after its children WHEN
/// `NF_LAYER_SPECIFIC_LR` is set. Inference models still carry the flag; the
/// rates are training-only, so read past them (mirrors
/// `tesseract_ocr::network::skip_layer_lr`).
fn skip_layer_lr(cur: &mut Cursor<'_>, flags: i32) -> Result<(), FacetWalkError> {
    if flags & NF_LAYER_SPECIFIC_LR != 0 {
        let count = cur.read_u32()?;
        cur.take(count as usize * 4)?;
    }
    Ok(())
}

/// Every walked facet's CANON (hi-u16) half must equal [`NETWORK_LAYER`]
/// (`0x0804`, the `network_layer` OGAR concept every network node's classid
/// is minted under — `NetworkType::classid`,
/// `compose_classid(NETWORK_LAYER, ntype as u16)`). Returns the first
/// offending `(index, classid)` pair, if any — `None` means every facet's
/// concept half is correct. The Rust-side half of the bonus check (see the
/// module docs); [`crate::v3_facet::DECODE_SCRIPT`] asserts the Python-side
/// half inline.
#[must_use]
pub fn first_wrong_concept(facets: &[NodeFacet]) -> Option<(usize, u32)> {
    facets
        .iter()
        .enumerate()
        .find(|(_, nf)| classid_canon(nf.facet.facet_classid) != NETWORK_LAYER)
        .map(|(i, nf)| (i, nf.facet.facet_classid))
}

/// The `facet\t<i>\t<classid %08X>\t<12 payload bytes as 24 hex>` line.
#[must_use]
pub fn facet_line(i: usize, facet: FacetCascade) -> String {
    let b = facet.to_bytes();
    let hex: String = b[4..16].iter().map(|x| format!("{x:02x}")).collect();
    format!("facet\t{i}\t{:08X}\t{hex}", facet.facet_classid)
}

/// The `fields\t<i>\t<classid>\t<lo,hi pairs...>` line — the Rust-decoded
/// tier bytes, tiers 0..6 in order, each printed `lo,hi`. MUST byte-match the
/// line [`DECODE_SCRIPT`] prints for the SAME 16 bytes via the OGAR-generated
/// Python `Facet.from_bytes`: Python's `is_a_chain[t]` decodes byte `4+2t`
/// (== this function's `tiers[t].lo`) and `part_of_chain[t]` decodes byte
/// `5+2t` (== `tiers[t].hi`) — see `lance_graph_contract::facet::FacetCascade
/// ::from_bytes` and `ogar_capability_surface.Facet.from_bytes`, which are
/// byte-for-byte the same decode.
#[must_use]
pub fn fields_line(i: usize, facet: FacetCascade) -> String {
    let pairs: Vec<String> = facet
        .tiers
        .iter()
        .map(|t| format!("{:02X},{:02X}", t.lo, t.hi))
        .collect();
    format!(
        "fields\t{i}\t{:08X}\t{}",
        facet.facet_classid,
        pairs.join(",")
    )
}

/// The Python decode-and-print script both the example and the integration
/// test run against the SAME facet-bytes file this module's callers write —
/// one canonical source, so the two call sites can never drift against each
/// other. Reads 16-byte chunks from the path given as `argv[1]`, decodes each
/// with the OGAR-generated SDK's `Facet.from_bytes` (must be importable on
/// `PYTHONPATH`), prints one `fields\t<i>\t<classid>\t<lo,hi pairs>` line per
/// facet in the SAME format [`fields_line`] produces, and asserts the BONUS
/// concept check (every facet's CANON half == `CLASS_IDS["network_layer"]`)
/// inline — a failed assertion exits non-zero, which
/// [`run_python_decode`]'s caller observes as a process failure. On success
/// it reports `BONUS_CONCEPT_CHECK_OK\t<n>` on stderr.
pub const DECODE_SCRIPT: &str = r#"import sys

import ogar_capability_surface as m

with open(sys.argv[1], "rb") as f:
    data = f.read()
assert len(data) % 16 == 0, f"facet file length {len(data)} is not a multiple of 16"
n = len(data) // 16

network_layer = m.CLASS_IDS["network_layer"]
for i in range(n):
    fac = m.Facet.from_bytes(data[i * 16 : (i + 1) * 16])
    concept = m.concept_of(fac.classid)
    assert concept == network_layer, (
        f"facet {i}: concept {concept:04X} != network_layer {network_layer:04X}"
    )
    pairs = ",".join(
        f"{lo:02X},{hi:02X}" for lo, hi in zip(fac.is_a_chain, fac.part_of_chain)
    )
    print(f"fields\t{i}\t{fac.classid:08X}\t{pairs}")

print(f"BONUS_CONCEPT_CHECK_OK\t{n}", file=sys.stderr)
"#;

/// Write [`DECODE_SCRIPT`] to `<dir>/decode_facets.py`, returning its path.
///
/// # Errors
///
/// Any [`std::io::Error`] writing the file.
pub fn write_decode_script(dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let path = dir.join("decode_facets.py");
    std::fs::write(&path, DECODE_SCRIPT)?;
    Ok(path)
}

/// Run `python3 <script> <bin_path>` with `PYTHONPATH=<pythonpath_dir>` — the
/// exact invocation both the example and the integration test use to decode
/// [`facet_line`]-written bytes through the OGAR-generated SDK.
///
/// # Errors
///
/// Any [`std::io::Error`] spawning `python3`.
pub fn run_python_decode(
    script: &std::path::Path,
    bin_path: &std::path::Path,
    pythonpath_dir: &std::path::Path,
) -> std::io::Result<std::process::Output> {
    std::process::Command::new("python3")
        .env("PYTHONPATH", pythonpath_dir)
        .arg(script)
        .arg(bin_path)
        .output()
}

/// The result of comparing Rust-computed `fields` lines ([`fields_line`])
/// against the `fields` lines [`DECODE_SCRIPT`] printed to stdout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldsDiff {
    /// Every line matched; carries the count.
    Match(usize),
    /// The two sides produced a different number of `fields` lines.
    LineCountMismatch {
        /// Lines Rust computed.
        rust: usize,
        /// `fields` lines Python's stdout carried.
        python: usize,
    },
    /// The first line (0-based index) where the two sides diverge.
    FirstMismatch {
        /// The 0-based line index.
        index: usize,
        /// Rust's line.
        rust: String,
        /// Python's line.
        python: String,
    },
}

/// Compare Rust's `fields` lines (from [`fields_line`]) against Python
/// stdout's `fields` lines (from [`DECODE_SCRIPT`], via
/// [`run_python_decode`]). Only lines starting `"fields\t"` in Python's
/// stdout are considered (robust to future non-`fields` diagnostic lines,
/// though [`DECODE_SCRIPT`] currently prints only `fields` lines to stdout).
#[must_use]
pub fn diff_fields_lines(rust_lines: &[String], python_stdout: &str) -> FieldsDiff {
    let python_lines: Vec<&str> = python_stdout
        .lines()
        .filter(|l| l.starts_with("fields\t"))
        .collect();
    if rust_lines.len() != python_lines.len() {
        return FieldsDiff::LineCountMismatch {
            rust: rust_lines.len(),
            python: python_lines.len(),
        };
    }
    for (i, (r, p)) in rust_lines.iter().zip(python_lines.iter()).enumerate() {
        if r != p {
            return FieldsDiff::FirstMismatch {
                index: i,
                rust: r.clone(),
                python: (*p).to_string(),
            };
        }
    }
    FieldsDiff::Match(rust_lines.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the base header bytes a `Network::Serialize` would write for one
    /// node — mirrors `tesseract_core::network::tests::header_bytes` /
    /// `tesseract_ocr::network::tests::header` (same wire format, third
    /// independent copy for this module's own unit coverage).
    fn header_bytes(type_name: &str, ni: i32, no: i32, num_weights: i32, name: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.push(0u8); // tag = NT_NONE
        b.extend_from_slice(&(type_name.len() as u32).to_le_bytes());
        b.extend_from_slice(type_name.as_bytes());
        b.push(0u8); // training = TS_DISABLED
        b.push(0u8); // needs_backprop = false
        b.extend_from_slice(&0i32.to_le_bytes()); // network_flags
        b.extend_from_slice(&ni.to_le_bytes());
        b.extend_from_slice(&no.to_le_bytes());
        b.extend_from_slice(&num_weights.to_le_bytes());
        b.extend_from_slice(&(name.len() as u32).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b
    }

    #[test]
    fn walks_a_minimal_series_tree_and_matches_tesseract_ocr_shape() {
        // Series[ Input[1,4,0,1], Maxpool[2,2] ] — the same tree
        // `tesseract_ocr::network::tests::loads_a_minimal_series_tree` builds,
        // so this proves the two independent walkers agree on byte layout.
        let mut b = header_bytes("Series", 1, 1, 0, "root");
        b.extend_from_slice(&2_u32.to_le_bytes()); // child count
        b.extend(header_bytes("Input", 1, 1, 0, "in"));
        for v in [1_i32, 4, 0, 1, 0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend(header_bytes("Maxpool", 1, 1, 0, "mp"));
        for v in [2_i32, 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }

        let (facets, consumed) = collect_facets(&b).expect("walk");
        assert_eq!(consumed, b.len(), "consumes exactly the tree");
        assert_eq!(facets.len(), 3, "root + Input + Maxpool");
        assert_eq!(facets[0].ntype, NetworkType::Series);
        assert_eq!(facets[1].ntype, NetworkType::Input);
        assert_eq!(facets[2].ntype, NetworkType::Maxpool);

        // Every facet's CANON half is NETWORK_LAYER — the bonus check, on
        // synthetic data.
        assert_eq!(first_wrong_concept(&facets), None);
    }

    #[test]
    fn facet_line_and_fields_line_are_stable_and_tab_separated() {
        // A `Series` header needs its trailing `u32` child count on the wire
        // even when empty — append `0` for a valid single-node (root only)
        // tree, mirroring the real eng.lstm root's ni/no/num_weights.
        let mut b = header_bytes("Series", 36, 111, 385807, "root");
        b.extend_from_slice(&0_u32.to_le_bytes());
        let (facets, consumed) = collect_facets(&b).expect("walk single empty-Series node");
        assert_eq!(consumed, b.len());
        let facet = facets[0].facet;

        let fl = facet_line(0, facet);
        assert!(fl.starts_with("facet\t0\t"));
        let parts: Vec<&str> = fl.split('\t').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[2], format!("{:08X}", facet.facet_classid));
        assert_eq!(parts[3].len(), 24, "12 payload bytes as 24 hex chars");

        let gl = fields_line(0, facet);
        assert!(gl.starts_with("fields\t0\t"));
        let gparts: Vec<&str> = gl.split('\t').collect();
        assert_eq!(gparts.len(), 4);
        // 6 tiers x "lo,hi" = 12 comma-separated hex bytes.
        assert_eq!(gparts[3].split(',').count(), 12);
    }

    #[test]
    fn diff_fields_lines_detects_match_count_and_first_mismatch() {
        let rust = vec!["fields\t0\tAABBCCDD\t01,02".to_string()];
        assert_eq!(
            diff_fields_lines(&rust, "fields\t0\tAABBCCDD\t01,02\n"),
            FieldsDiff::Match(1)
        );
        assert_eq!(
            diff_fields_lines(&rust, ""),
            FieldsDiff::LineCountMismatch { rust: 1, python: 0 }
        );
        assert_eq!(
            diff_fields_lines(&rust, "fields\t0\tAABBCCDD\t01,03\n"),
            FieldsDiff::FirstMismatch {
                index: 0,
                rust: "fields\t0\tAABBCCDD\t01,02".to_string(),
                python: "fields\t0\tAABBCCDD\t01,03".to_string(),
            }
        );
    }

    #[test]
    fn rejects_reversed_with_wrong_child_count_and_unsupported_types() {
        let mut b = header_bytes("RTLReversed", 1, 1, 0, "r");
        b.extend_from_slice(&2_u32.to_le_bytes()); // child count != 1
        assert!(matches!(
            collect_facets(&b),
            Err(FacetWalkError::Unsupported(_))
        ));

        let b = header_bytes("TensorFlow", 1, 1, 0, "tf");
        assert!(matches!(
            collect_facets(&b),
            Err(FacetWalkError::Unsupported(_))
        ));
    }
}
