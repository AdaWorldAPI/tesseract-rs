//! V3-substrate <-> Python-SDK parity probe.
//!
//! Loads the REAL `/tmp/eng.lstm` via the SAME public API `network_dump`
//! uses ([`tesseract_ocr::Network::from_le_bytes`]), walks it into one
//! content-blind V3 [`FacetCascade`](lance_graph_contract::facet::FacetCascade)
//! per node via [`tesseract_ogar::v3_facet::collect_facets`] (which reuses
//! the Core's proven `NetworkHeader::to_facet` — never hand-rolled byte
//! packing), writes the concatenated facet bytes + a TSV dump, then emits
//! the OGAR-generated Python SDK fresh and proves its `Facet.from_bytes`
//! decodes those SAME bytes byte-identically to the Rust side.
//!
//! ```sh
//! cargo run -p tesseract-ogar --example v3_facet_probe
//! ```
//!
//! Requires `/tmp/eng.lstm` (see `tesseract-rs/CLAUDE.md` "the proven
//! method") and the `AdaWorldAPI/OGAR` sibling checkout at
//! `/home/user/OGAR` (for the `ogar-adapter-python` SDK emitter) — both
//! early-return with a stderr note when absent, so this example is a no-op
//! (not a failure) outside a fully staged environment.

#![allow(clippy::print_stdout, reason = "dump CLI")]

use std::path::Path;

use tesseract_ogar::v3_facet::{
    collect_facets, diff_fields_lines, facet_line, fields_line, first_wrong_concept,
    run_python_decode, write_decode_script, FieldsDiff,
};

fn main() {
    let lstm_path = Path::new("/tmp/eng.lstm");
    if !lstm_path.exists() {
        eprintln!("v3_facet_probe: /tmp/eng.lstm not present in this environment — skipping");
        return;
    }
    let bytes = std::fs::read(lstm_path).expect("read /tmp/eng.lstm");

    // Load via the SAME public API `network_dump` uses — cross-checked below
    // against this probe's own header walk (which needs per-node headers
    // `Network::from_le_bytes`'s tree does not retain; see the `v3_facet`
    // module docs for why).
    let (net, consumed_via_network) =
        tesseract_ocr::Network::from_le_bytes(&bytes).expect("load network via public API");

    let (facets, consumed_via_walk) = collect_facets(&bytes).expect("walk network headers");
    assert_eq!(
        consumed_via_walk, consumed_via_network,
        "facet walk consumed a different byte count than Network::from_le_bytes"
    );
    let root = facets[0].facet;
    assert_eq!(
        root.tiers[0].as_u16(),
        net.ni as u16,
        "root ni tier mismatch"
    );
    assert_eq!(
        root.tiers[1].as_u16(),
        net.no as u16,
        "root no tier mismatch"
    );
    let root_nw = u32::from(root.tiers[3].as_u16()) | (u32::from(root.tiers[4].as_u16()) << 16);
    assert_eq!(
        root_nw, net.num_weights as u32,
        "root num_weights tiers mismatch"
    );

    eprintln!(
        "v3_facet_probe: walked {} facets ({} bytes; matches Network::from_le_bytes's {} bytes)",
        facets.len(),
        consumed_via_walk,
        consumed_via_network
    );

    // Bonus check, Rust side: every facet's CANON half == NETWORK_LAYER.
    if let Some((i, classid)) = first_wrong_concept(&facets) {
        panic!(
            "facet {i} has classid {classid:08X} whose CANON half is not NETWORK_LAYER (0x0804)"
        );
    }
    eprintln!(
        "v3_facet_probe: bonus check OK (Rust) — all {} facets' CANON half == NETWORK_LAYER (0x0804)",
        facets.len()
    );

    // Artifact 1: concatenated 16-byte facets. Artifact 2: stdout TSV.
    let bin_path = Path::new("/tmp/v3_facets.bin");
    let mut bin = Vec::with_capacity(facets.len() * 16);
    let mut rust_fields = Vec::with_capacity(facets.len());
    for (i, nf) in facets.iter().enumerate() {
        bin.extend_from_slice(&nf.facet.to_bytes());
        println!("{}", facet_line(i, nf.facet));
        let gl = fields_line(i, nf.facet);
        println!("{gl}");
        rust_fields.push(gl);
    }
    std::fs::write(bin_path, &bin).expect("write /tmp/v3_facets.bin");
    eprintln!(
        "v3_facet_probe: wrote {} bytes ({} facets x 16B) to {}",
        bin.len(),
        facets.len(),
        bin_path.display()
    );

    // Python side: emit the OGAR-generated SDK fresh, decode the SAME bin
    // file with its `Facet.from_bytes`, diff its `fields` lines against
    // Rust's.
    let ogar_python_crate = Path::new("/home/user/OGAR/crates/ogar-adapter-python");
    if !ogar_python_crate.exists() {
        eprintln!(
            "v3_facet_probe: OGAR sibling (ogar-adapter-python) not present at {} — skipping Python parity",
            ogar_python_crate.display()
        );
        return;
    }
    let dir = std::env::temp_dir().join(format!("v3-facet-probe-example-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let module_path = dir.join("ogar_capability_surface.py");
    std::fs::write(
        &module_path,
        ogar_adapter_python::emit_python("ogar_capability_surface"),
    )
    .expect("write emitted OGAR Python SDK");
    let script_path = write_decode_script(&dir).expect("write decode script");

    eprintln!(
        "v3_facet_probe: PYTHONPATH={} python3 {} {}",
        dir.display(),
        script_path.display(),
        bin_path.display()
    );
    let out =
        run_python_decode(&script_path, bin_path, &dir).expect("spawn python3 decode_facets.py");
    if !out.status.success() {
        panic!(
            "python3 decode failed (exit {:?}):\nstdout: {}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    eprintln!(
        "v3_facet_probe: python3 stderr: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();

    match diff_fields_lines(&rust_fields, &stdout) {
        FieldsDiff::Match(n) => {
            eprintln!("v3_facet_probe: GREEN — {n} `fields` lines byte-identical (Rust vs Python)");
        }
        FieldsDiff::LineCountMismatch { rust, python } => {
            panic!("v3_facet_probe: MISMATCH — line count differs (rust {rust}, python {python})");
        }
        FieldsDiff::FirstMismatch {
            index,
            rust,
            python,
        } => {
            panic!("v3_facet_probe: MISMATCH at line {index}\nrust:   {rust}\npython: {python}");
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}
