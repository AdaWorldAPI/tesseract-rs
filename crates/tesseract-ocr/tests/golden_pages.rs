//! Golden integration tests (P6 D6.2): page-level recognition through the
//! real makerow line finder ([`LstmRecognizer::recognize_page_makerow`]).
//!
//! See `golden_lines.rs` (same crate) for the corpus-path contract and the
//! `UPDATE_GOLDEN=1` regeneration mode this file follows identically. These
//! are hermetic gates: a missing corpus or golden file is a test failure,
//! not a skip.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tesseract_core::DictLite;
use tesseract_ocr::{parse_pgm, LstmRecognizer};

/// The workspace's `corpus/` root, a sibling of `crates/`.
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

fn update_golden() -> bool {
    std::env::var_os("UPDATE_GOLDEN").is_some()
}

/// Compare `actual` against the golden file at `golden_path`; with
/// `UPDATE_GOLDEN=1` set, write `actual` as the new golden instead (creating
/// the golden directory if needed). A missing golden file (outside
/// `UPDATE_GOLDEN` mode) is a hard test failure, not a skip.
fn check_golden(golden_path: &Path, actual: &str, fixture: &str) {
    if update_golden() {
        if let Some(parent) = golden_path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("creating golden dir {}: {e}", parent.display()));
        }
        fs::write(golden_path, actual)
            .unwrap_or_else(|e| panic!("writing golden {}: {e}", golden_path.display()));
        return;
    }
    let expected = fs::read_to_string(golden_path).unwrap_or_else(|e| {
        panic!(
            "reading golden {} for fixture {fixture:?}: {e} \
             (goldens regenerate via UPDATE_GOLDEN=1)",
            golden_path.display()
        )
    });
    assert_eq!(
        actual,
        expected,
        "golden mismatch for fixture {fixture:?} at {} (goldens regenerate via UPDATE_GOLDEN=1)",
        golden_path.display()
    );
}

static RECOGNIZER: OnceLock<LstmRecognizer> = OnceLock::new();
static DICT: OnceLock<DictLite> = OnceLock::new();

/// The shared recognizer, assembled once per test-binary run from
/// `corpus/model/`.
fn recognizer() -> &'static LstmRecognizer {
    RECOGNIZER.get_or_init(|| {
        let model = corpus_dir().join("model");
        let lstm = fs::read(model.join("eng.lstm")).expect("read eng.lstm");
        let uni = fs::read_to_string(model.join("eng.lstm-unicharset"))
            .expect("read eng.lstm-unicharset");
        let rec = fs::read(model.join("eng.lstm-recoder")).expect("read eng.lstm-recoder");
        LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer")
    })
}

/// The shared dictionary, assembled once per test-binary run from
/// `corpus/model/`. Returns an owned clone; `recognize_page_makerow` takes
/// `Option<&DictLite>`, so callers borrow their own local copy.
fn dict() -> DictLite {
    DICT.get_or_init(|| {
        let model = corpus_dir().join("model");
        let word = fs::read(model.join("eng.lstm-word-dawg")).expect("read eng.lstm-word-dawg");
        let punc = fs::read(model.join("eng.lstm-punc-dawg")).expect("read eng.lstm-punc-dawg");
        let number =
            fs::read(model.join("eng.lstm-number-dawg")).expect("read eng.lstm-number-dawg");
        DictLite::from_components(&word, &punc, &number).expect("load dict")
    })
    .clone()
}

/// Recognize `corpus/pages/page_<nn>.pgm` (`nn` a zero-padded two-digit
/// number, e.g. `"01"`) through the real makerow line finder.
fn recognize_page(nn: &str) -> String {
    let img_path = corpus_dir().join("pages").join(format!("page_{nn}.pgm"));
    let bytes = fs::read(&img_path).unwrap_or_else(|e| panic!("read {}: {e}", img_path.display()));
    let (grey, w, h) = parse_pgm(&bytes).unwrap_or_else(|e| panic!("parse_pgm(page_{nn}): {e}"));
    let d = dict();
    recognizer()
        .recognize_page_makerow(&grey, w, h, Some(&d))
        .unwrap_or_else(|e| panic!("recognize_page_makerow(page_{nn}): {e}"))
}

#[test]
fn golden_page_text() {
    for n in 1..=10 {
        let nn = format!("{n:02}");
        let text = recognize_page(&nn);

        // Sanity floor asserted unconditionally, in BOTH normal and
        // UPDATE_GOLDEN mode: the fixture pages have 6-10 lines each, so a
        // real recognition must clear at least 2 non-empty lines. This
        // guards UPDATE_GOLDEN mode from silently baking in an
        // empty/near-empty golden from a broken pipeline.
        let non_empty_lines = text.split('\n').filter(|l| !l.is_empty()).count();
        assert!(
            !text.is_empty(),
            "page_{nn}: recognized text must not be empty"
        );
        assert!(
            non_empty_lines >= 2,
            "page_{nn}: expected >= 2 non-empty lines, got {non_empty_lines} (text={text:?})"
        );

        let golden_path = corpus_dir()
            .join("golden")
            .join("pages")
            .join(format!("page_{nn}.txt"));
        check_golden(&golden_path, &format!("{text}\n"), &format!("page_{nn}"));
    }
}

#[test]
fn golden_pages_deterministic() {
    let a = recognize_page("01");
    let b = recognize_page("01");
    assert_eq!(
        a, b,
        "recognize_page_makerow(page_01) must be deterministic"
    );
}
