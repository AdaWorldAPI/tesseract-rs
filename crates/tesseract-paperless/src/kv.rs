//! The document KV layout — the **S-2 preflight dedup gate** and the keys of
//! the document subtree.
//!
//! # What this module is, and what it deliberately is not
//!
//! Each repo in this stack keeps its own concern, and the split is what makes
//! a dedup gate placeable at all:
//!
//! | repo | owns | never does |
//! |---|---|---|
//! | `tesseract-rs` recognition crates | recognition → `doc.v1` | storage, hashing, persistence |
//! | `OGAR` | the vocabulary — classids, `ActionDef`s, the doc IR | I/O |
//! | `lance-graph` | the type layer + (eventually) the KV store | ingestion policy |
//! | **this crate** | **the gate, the layout, the ingestion ORDER** | recognition, minting concepts, storing anything |
//!
//! Note the last cell. This module is the **layout**: how a document becomes
//! keys. It computes the convergence hash, builds the subtree's `NodeGuid`s,
//! and answers the one question that must be answered *before* any expensive
//! work happens — have we seen these bytes already? It does not answer that
//! question itself: [`DedupIndex`] is a trait, and no implementation ships
//! here. Recognition in this workspace stays storage-less, and a gate that
//! carried a store would be the first crack in that.
//!
//! # S-2 — why the gate is here and not in `persist_document`
//!
//! From the ingestion spine merged into OGAR
//! (`docs/OGAR-DOC-INGESTION-SPINE.md`, PR #266), invariant **S-2**:
//!
//! > Dedup is the first real gate, and it runs BEFORE the expensive work.
//!
//! `OGAR-DOC-W4-BUILD-SPEC` makes `persist_document` idempotent on
//! `content_sha256` — correct, but that is *after* recognition has run. It
//! prevents a duplicate **subtree**; it does not prevent a duplicate
//! **spend**, and an OCR pass is seconds-to-minutes of compute. The gate has
//! to sit on the raw bytes, ahead of `OcrExecutor`, which is why it lives in
//! the assembly repo rather than in either upstream.
//!
//! Note the second half of S-2, which is easy to drop: the incoming hash is
//! matched against **both** the original and any derived-artifact hash. A
//! re-ingested *export* of a document already held is the same document, and
//! only the derived key catches it.
//!
//! # S-5 — write order
//!
//! Also from the spine: **bytes before address.** The document root value
//! carries a *reference* to the blob (`content_sha256`, storage key, mime,
//! counts) and never the bytes themselves. If the address is written first
//! and the blob put then fails, the graph holds a resolvable `document_guid`
//! pointing at nothing — corruption that is invisible until read. Written the
//! other way round, a failure leaves an unreferenced blob: garbage, and
//! collectable. [`DocumentKeys`] is deliberately cheap to construct and
//! carries no I/O, so a caller can compute the address, put the bytes, and
//! only then commit the row.

use lance_graph_contract::facet::FacetCascade;

// ---------------------------------------------------------------------------
// Trap 1 — the silent V1 fallback
// ---------------------------------------------------------------------------

/// `mint_for`'s `V2 | V3` arm is `#[cfg(feature = "guid-v2-tail")]`; with the
/// feature **off** it falls through to the V1 constructor
/// (`canonical_node.rs:527`) and mints a u24-tail GUID **with no error**.
///
/// That is a wrong-key-shape bug that compiles, runs, and looks correct. The
/// V1 tail is a flat u24 with no axis — it cannot carry a rail — and is
/// forbidden for new units by the canon
/// (`E-V1-TAIL-FORBIDDEN-V3-IS-CONTENT-BLIND-1`).
///
/// This crate sidesteps the fallback entirely by building a [`FacetCascade`]
/// and converting, rather than calling `mint_for`. That is also the measured
/// house idiom — 8 files / 28 `facet::` uses against 3 `mint_for` sites — and
/// the conversion is byte-identical in both directions.
///
/// The const assert below is the fuse: `FacetCascade` is the 4+12 shape, and
/// if that ever stops being true the build stops rather than minting a
/// silently-wrong key.
const _: () = assert!(
    core::mem::size_of::<FacetCascade>() == 16,
    "FacetCascade must be the 16-byte 4+12 facet; a changed layout means \
     every key this crate mints is silently the wrong shape"
);

/// Length of a SHA-256 digest, and of the `content_sha256` field the document
/// root value carries (`OGAR-DOC-W4-BUILD-SPEC` §W4-2).
pub const CONTENT_SHA256_LEN: usize = 32;

// ---------------------------------------------------------------------------
// The convergence key
// ---------------------------------------------------------------------------

/// The content hash of a document's **original input bytes** — the
/// convergence key.
///
/// # Why SHA-256 and not `ContentId`
///
/// `lance_graph_contract::content_store::ContentId` is a content-addressed
/// key too, but it is `fnv1a` → `u64`, and it is the right key for its own
/// job (text spans). It is the wrong key here for two reasons:
///
/// 1. **`OGAR-DOC-W4-BUILD-SPEC` §W4-2 specifies `content_sha256[32]`.**
/// 2. **fnv1a is not cryptographic.** Collisions are cheap to construct on
///    purpose, and this hash decides "we already have this document". A
///    deliberate collision means a submitted invoice silently dedups onto an
///    existing one and is never stored — an attack, not a mishap, in an
///    accounting path.
///
/// So the two coexist rather than compete: SHA-256 is the document
/// convergence key; `ContentId` stays the span store's key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentSha256(pub [u8; CONTENT_SHA256_LEN]);

impl ContentSha256 {
    /// Hash the original input bytes.
    ///
    /// This is the *only* place the convergence key is produced. Nothing
    /// upstream computes it: tesseract-rs carries no digest dependency at
    /// all, and its consumer guide states plainly that `doc.v1` does not
    /// stamp a hash.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        Self(h.finalize().into())
    }

    /// The digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; CONTENT_SHA256_LEN] {
        &self.0
    }
}

impl core::fmt::Debug for ContentSha256 {
    /// Lower-case hex, so a log line is greppable against a stored key.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// S-2 — the preflight gate
// ---------------------------------------------------------------------------

/// What the preflight gate decided, and therefore whether recognition runs.
///
/// This is the type S-2 exists to produce. It is returned *before*
/// `OcrExecutor` is constructed, so a [`Verdict::Duplicate`] costs a hash and
/// a lookup rather than a full recognition pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Not seen. Recognition should run, and the bytes should be put to the
    /// blob store **before** the subtree row is committed (S-5).
    Novel,
    /// Already held. Recognition must be skipped.
    Duplicate {
        /// Which stored key matched — the original bytes, or a derived
        /// artifact. Both are checked; see [`DedupIndex::look_up`].
        matched: MatchedOn,
    },
}

/// Which stored hash an incoming document matched.
///
/// S-2's second half: matching only the original hash misses a re-ingested
/// *export* of a document already held, which is the same document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchedOn {
    /// The hash of the original input bytes.
    Original,
    /// The hash of a derived artifact (an exported/normalised rendition).
    Derived,
}

/// The lookup S-2 needs: *have we seen these bytes?*
///
/// Deliberately a trait and not a struct. There is **no KV put/get in
/// lance-graph today** — `symbiont` is a link probe that only prints that the
/// stack linked, and `surreal_container::open` returns
/// `Err(Blocked { reason: "surrealdb kv-lance fork dep not wired" })` with
/// every module a `// TODO task NN` header. Rather than pretend otherwise,
/// the gate is expressed against the operation it needs, so the layout is
/// testable now and the storage side implements this when it lands.
///
/// This is also exactly what `W4-8` prescribes: *"No storage backend chosen
/// (KV blob is the consumer's)."*
pub trait DedupIndex {
    /// Return the stored document key for `hash`, checking **both** the
    /// original and derived-artifact hashes, or `None` if unseen.
    fn look_up(&self, hash: &ContentSha256) -> Option<(DocumentGuid, MatchedOn)>;
}

/// Run the S-2 preflight gate over raw input bytes.
///
/// Hashes once, asks the index, and returns before anything expensive
/// happens. The caller runs recognition only on [`Verdict::Novel`].
pub fn preflight<I: DedupIndex>(bytes: &[u8], index: &I) -> (ContentSha256, Verdict) {
    let hash = ContentSha256::of(bytes);
    let verdict = match index.look_up(&hash) {
        Some((_, matched)) => Verdict::Duplicate { matched },
        None => Verdict::Novel,
    };
    (hash, verdict)
}

// ---------------------------------------------------------------------------
// The subtree keys
// ---------------------------------------------------------------------------

/// A document's root key — the address everything else references.
///
/// This is what `medcare-rs` stores beside a lab value and `odoo-rs`/`woa-rs`
/// beside an invoice line, so an extracted fact keeps a resolvable path back
/// to the pixels it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocumentGuid(pub FacetCascade);

/// The keys of one document's subtree, per `W4-2`.
///
/// Structure only — no values, no bytes, no I/O. Construction is cheap
/// precisely so a caller can obtain the address, put the blob, and commit the
/// row in that order (S-5).
#[derive(Debug, Clone)]
pub struct DocumentKeys {
    /// The subtree root (`document`, classid `0x080B`).
    pub root: DocumentGuid,
    /// The convergence key. Lives in the root's *value* as part of the
    /// raw-ref — never the bytes themselves (`W4-2`: "awareness never
    /// re-embeds raw bytes").
    pub content_sha256: ContentSha256,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate must be decided from the bytes alone, with no recognition and
    /// no I/O — that is the entire point of S-2.
    struct Empty;
    impl DedupIndex for Empty {
        fn look_up(&self, _: &ContentSha256) -> Option<(DocumentGuid, MatchedOn)> {
            None
        }
    }

    struct Seen(ContentSha256, MatchedOn);
    impl DedupIndex for Seen {
        fn look_up(&self, h: &ContentSha256) -> Option<(DocumentGuid, MatchedOn)> {
            (h == &self.0).then(|| (DocumentGuid(FacetCascade::from_bytes(&[0u8; 16])), self.1))
        }
    }

    #[test]
    fn identical_bytes_hash_identically_and_differing_bytes_do_not() {
        let a = ContentSha256::of(b"invoice pdf bytes");
        let b = ContentSha256::of(b"invoice pdf bytes");
        let c = ContentSha256::of(b"invoice pdf byteS");
        assert_eq!(a, b, "the convergence key must be stable for equal bytes");
        assert_ne!(
            a, c,
            "a one-bit input change must change the key, or dedup silently \
             merges distinct documents"
        );
    }

    #[test]
    fn an_unseen_document_is_novel() {
        let (_, v) = preflight(b"fresh scan", &Empty);
        assert_eq!(v, Verdict::Novel);
    }

    /// The can-fire half: a document already held must be caught by the gate,
    /// so recognition never runs.
    #[test]
    fn a_document_already_held_is_a_duplicate() {
        let bytes = b"a scan we already have";
        let (hash, _) = preflight(bytes, &Empty);
        let (_, v) = preflight(bytes, &Seen(hash, MatchedOn::Original));
        assert_eq!(
            v,
            Verdict::Duplicate {
                matched: MatchedOn::Original
            }
        );
    }

    /// S-2's second half, and the one most easily dropped: a re-ingested
    /// EXPORT of a held document is the same document, and only the
    /// derived-artifact hash catches it. An index that checked the original
    /// hash alone would report `Novel` here and pay for a full re-recognition.
    #[test]
    fn a_reingested_export_matches_on_the_derived_hash() {
        let export = b"an exported rendition of a document we hold";
        let (hash, _) = preflight(export, &Empty);
        let (_, v) = preflight(export, &Seen(hash, MatchedOn::Derived));
        assert_eq!(
            v,
            Verdict::Duplicate {
                matched: MatchedOn::Derived
            },
            "matching only the original hash misses re-ingested exports"
        );
    }

    /// Anti-vacuity for the two tests above: the fixture index must genuinely
    /// discriminate, or `Duplicate` would be its answer to everything and the
    /// assertions would pass for the wrong reason.
    #[test]
    fn the_fixture_index_does_not_match_everything() {
        let (hash, _) = preflight(b"one document", &Empty);
        let (_, v) = preflight(
            b"a completely different document",
            &Seen(hash, MatchedOn::Original),
        );
        assert_eq!(
            v,
            Verdict::Novel,
            "the index must answer Novel for bytes it has not seen"
        );
    }

    /// Trap 1, asserted rather than described: the facet is the 4+12 shape and
    /// the bridge to a node key is byte-identical, so this crate never needs
    /// `mint_for` and never risks its silent V1 fallback.
    #[test]
    fn the_facet_is_the_four_plus_twelve_shape_and_round_trips() {
        let mut raw = [0u8; 16];
        raw[0..4].copy_from_slice(&0x0000_080Bu32.to_le_bytes());
        raw[4] = 0xAB;
        raw[5] = 0xCD;
        let f = FacetCascade::from_bytes(&raw);
        assert_eq!(
            f.to_bytes(),
            raw,
            "FacetCascade must round-trip byte-identically, or the key this \
             crate mints is not the key that gets stored"
        );
    }
}
