//! Integration test: the V3-substrate <-> Python-SDK parity probe, automated.
//!
//! Loads the REAL `/tmp/eng.lstm`, walks it into
//! [`tesseract_ogar::v3_facet`] facets, emits the OGAR-generated Python SDK
//! (`ogar-adapter-python`, dev-dependency) fresh, decodes the SAME facet
//! bytes through its `Facet.from_bytes`, and asserts the two sides' `fields`
//! lines are byte-identical. Early-returns (with an explanation on stderr)
//! when `/tmp/eng.lstm`, the OGAR sibling checkout, or `python3` are not
//! present in this environment — this test proves the ALREADY-staged
//! environment stays green, it does not stage the environment itself
//! (mirrors the `smoke_recognize_line_matches_proven_regression` unit test's
//! early-return convention in `src/lib.rs`, and `ogar-adapter-python`'s own
//! `tests/parity.rs` compile+run pattern).

use std::path::Path;

use tesseract_ogar::v3_facet::{
    collect_facets, diff_fields_lines, fields_line, first_wrong_concept, run_python_decode,
    write_decode_script, FieldsDiff,
};

/// A unique temp dir under `/tmp` (or `$TMPDIR`) for one test run.
fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let dir = std::env::temp_dir().join(format!(
        "tesseract-ogar-{tag}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir under /tmp");
    dir
}

#[test]
fn v3_facets_decode_identically_in_python() {
    let lstm_path = Path::new("/tmp/eng.lstm");
    if !lstm_path.exists() {
        eprintln!("v3_facets_decode_identically_in_python: skipping — /tmp/eng.lstm not present");
        return;
    }
    let ogar_python_crate = Path::new("/home/user/OGAR/crates/ogar-adapter-python");
    if !ogar_python_crate.exists() {
        eprintln!(
            "v3_facets_decode_identically_in_python: skipping — OGAR sibling checkout not present at {}",
            ogar_python_crate.display()
        );
        return;
    }
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("v3_facets_decode_identically_in_python: skipping — python3 not on PATH");
        return;
    }

    let bytes = std::fs::read(lstm_path).expect("read /tmp/eng.lstm");

    // Load via the SAME public API `network_dump` uses, as a cross-check that
    // this crate's own header walk parses the identical tree (see the
    // `v3_facet` module docs for why a separate walk is needed).
    let (net, consumed_via_network) =
        tesseract_ocr::Network::from_le_bytes(&bytes).expect("load network via public API");
    let (facets, consumed_via_walk) = collect_facets(&bytes).expect("walk network headers");
    assert_eq!(
        consumed_via_walk, consumed_via_network,
        "facet walk consumed a different byte count than Network::from_le_bytes"
    );
    assert!(!facets.is_empty(), "eng.lstm must yield at least one facet");
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

    // Bonus check, Rust side: every facet's CANON half == NETWORK_LAYER.
    assert_eq!(
        first_wrong_concept(&facets),
        None,
        "every eng.lstm facet's CANON half must equal NETWORK_LAYER (0x0804)"
    );

    let dir = unique_tmp_dir("v3-facet-parity");

    let bin_path = dir.join("v3_facets.bin");
    let mut bin = Vec::with_capacity(facets.len() * 16);
    let mut rust_fields = Vec::with_capacity(facets.len());
    for (i, nf) in facets.iter().enumerate() {
        bin.extend_from_slice(&nf.facet.to_bytes());
        rust_fields.push(fields_line(i, nf.facet));
    }
    std::fs::write(&bin_path, &bin).expect("write facet bin");

    // Emit the OGAR-generated Python SDK fresh (dev-dependency; never a
    // hand-maintained copy) and decode the SAME bytes through it.
    let module_path = dir.join("ogar_capability_surface.py");
    std::fs::write(
        &module_path,
        ogar_adapter_python::emit_python("ogar_capability_surface"),
    )
    .expect("write emitted OGAR Python SDK");
    let script_path = write_decode_script(&dir).expect("write decode script");

    let out = run_python_decode(&script_path, &bin_path, &dir).expect("spawn python3");
    assert!(
        out.status.success(),
        "python3 decode failed (exit {:?}):\nstdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // The decode script's own inline assertion is the Python-side half of
    // the bonus check (every facet's concept == CLASS_IDS["network_layer"]);
    // a successful exit plus this stderr marker proves it held for all
    // `facets.len()` facets.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&format!("BONUS_CONCEPT_CHECK_OK\t{}", facets.len())),
        "python decode script did not report its bonus concept check for all facets:\n{stderr}"
    );

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    match diff_fields_lines(&rust_fields, &stdout) {
        FieldsDiff::Match(n) => {
            assert_eq!(n, facets.len(), "matched line count must equal facet count");
        }
        FieldsDiff::LineCountMismatch { rust, python } => {
            panic!(
                "fields line count differs: rust={rust} python={python}\npython stdout:\n{stdout}"
            );
        }
        FieldsDiff::FirstMismatch {
            index,
            rust,
            python,
        } => {
            panic!("fields line {index} differs:\nrust:   {rust}\npython: {python}");
        }
    }

    eprintln!(
        "v3_facets_decode_identically_in_python: GREEN — {} facets, Rust and Python `fields` lines byte-identical",
        facets.len()
    );

    let _ = std::fs::remove_dir_all(&dir);
}
