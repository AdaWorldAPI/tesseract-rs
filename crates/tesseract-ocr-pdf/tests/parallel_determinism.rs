//! Determinism gate for the `parallel` feature's page-chunk OCR
//! ([`tesseract_ocr_pdf::OcrPipeline::ocr_pages_parallel`]): the parallel
//! path must be byte-identical to the serial twin
//! ([`tesseract_ocr_pdf::OcrPipeline::ocr_pages_serial`]), and out-of-order
//! job submission must still come back sorted by `(doc_id, page_no)`.
//!
//! See `crates/tesseract-ocr-pdf/tests/golden_pdfs.rs` (sibling test) for
//! this workspace's corpus-path convention. Hermetic: a missing corpus fixture
//! is a test failure, not a skip.

#![cfg(feature = "parallel")]

use std::fs;
use std::path::{Path, PathBuf};

use tesseract_ocr_pdf::{OcrPipeline, PageJob};

/// The workspace's `corpus/` root, a sibling of `crates/`.
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Build the pipeline from `corpus/model`'s real `eng.lstm` components,
/// mirroring `tesseract-ocr-pdf/src/main.rs`'s `load_pipeline`.
fn load_pipeline() -> OcrPipeline {
    let data_dir = corpus_dir().join("model");
    let lstm = data_dir.join("eng.lstm");
    let unicharset = data_dir.join("eng.lstm-unicharset");
    let recoder = data_dir.join("eng.lstm-recoder");
    let word_dawg = data_dir.join("eng.lstm-word-dawg");
    let punc_dawg = data_dir.join("eng.lstm-punc-dawg");
    let number_dawg = data_dir.join("eng.lstm-number-dawg");
    OcrPipeline::from_data_paths(
        &lstm,
        &unicharset,
        &recoder,
        Some(word_dawg.as_path()),
        Some(punc_dawg.as_path()),
        Some(number_dawg.as_path()),
    )
    .unwrap_or_else(|e| panic!("loading pipeline from {}: {e}", data_dir.display()))
}

/// Read + parse `corpus/pages/page_<nn>.pgm` into `(grey, width, height)`.
fn load_page(nn: &str) -> (Vec<u8>, usize, usize) {
    let path = corpus_dir().join("pages").join(format!("page_{nn}.pgm"));
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    tesseract_ocr::parse_pgm(&bytes).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

#[test]
fn parallel_equals_serial_byte_identical() {
    let pipeline = load_pipeline();

    let pages: Vec<(Vec<u8>, usize, usize)> = ["01", "02", "03", "04"]
        .iter()
        .map(|nn| load_page(nn))
        .collect();

    let make_jobs = || -> Vec<PageJob> {
        pages
            .iter()
            .enumerate()
            .map(|(i, (grey, w, h))| PageJob {
                doc_id: 0,
                page_no: i + 1,
                grey: grey.clone(),
                width: *w,
                height: *h,
            })
            .collect()
    };

    let serial = pipeline
        .ocr_pages_serial(make_jobs())
        .unwrap_or_else(|e| panic!("ocr_pages_serial: {e}"));
    let parallel = pipeline
        .ocr_pages_parallel(make_jobs())
        .unwrap_or_else(|e| panic!("ocr_pages_parallel: {e}"));

    assert_eq!(parallel.len(), serial.len(), "job count must be preserved");
    for (i, (p, s)) in parallel.iter().zip(serial.iter()).enumerate() {
        assert_eq!(p.doc_id, s.doc_id, "page index {i}: doc_id mismatch");
        assert_eq!(p.page_no, s.page_no, "page index {i}: page_no mismatch");
        assert_eq!(
            p.text, s.text,
            "page index {i} (page_no {}): parallel text must be byte-identical to serial",
            p.page_no
        );
    }
}

#[test]
fn out_of_order_jobs_are_resorted() {
    let pipeline = load_pipeline();

    let (grey1, w1, h1) = load_page("01");
    let (grey2, w2, h2) = load_page("02");

    // Serial references, computed independently via the in-order serial twin,
    // for the two distinct pages used below.
    let reference = pipeline
        .ocr_pages_serial(vec![
            PageJob {
                doc_id: 0,
                page_no: 1,
                grey: grey1.clone(),
                width: w1,
                height: h1,
            },
            PageJob {
                doc_id: 0,
                page_no: 2,
                grey: grey2.clone(),
                width: w2,
                height: h2,
            },
        ])
        .unwrap_or_else(|e| panic!("ocr_pages_serial: {e}"));
    let text1 = reference[0].text.clone();
    let text2 = reference[1].text.clone();

    // Submit out of (doc_id, page_no) order, and interleave a second doc_id
    // (doc_id 1 reuses page_01's buffer).
    let jobs = vec![
        PageJob {
            doc_id: 0,
            page_no: 2,
            grey: grey2.clone(),
            width: w2,
            height: h2,
        },
        PageJob {
            doc_id: 1,
            page_no: 1,
            grey: grey1.clone(),
            width: w1,
            height: h1,
        },
        PageJob {
            doc_id: 0,
            page_no: 1,
            grey: grey1.clone(),
            width: w1,
            height: h1,
        },
    ];

    let results = pipeline
        .ocr_pages_parallel(jobs)
        .unwrap_or_else(|e| panic!("ocr_pages_parallel: {e}"));

    let keys: Vec<(usize, usize)> = results.iter().map(|r| (r.doc_id, r.page_no)).collect();
    let mut sorted_keys = keys.clone();
    sorted_keys.sort_unstable();
    assert_eq!(
        keys, sorted_keys,
        "results must come back sorted by (doc_id, page_no)"
    );

    assert_eq!(keys, vec![(0, 1), (0, 2), (1, 1)]);
    assert_eq!(
        results[0].text, text1,
        "(doc 0, page 1) must match page_01 serial text"
    );
    assert_eq!(
        results[1].text, text2,
        "(doc 0, page 2) must match page_02 serial text"
    );
    assert_eq!(
        results[2].text, text1,
        "(doc 1, page 1) must match page_01 serial text"
    );
}
