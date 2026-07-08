//! Golden integration tests (P6 D6.2): line-image recognition end to end.
//!
//! Reads the fixed corpus contract under `<workspace root>/corpus/` (a
//! sibling of `crates/`) and recognizes each line fixture through the
//! proven byte-parity pipeline ([`LstmRecognizer::recognize_image_file`] /
//! [`LstmRecognizer::recognize_image_file_with_dict`] /
//! [`LstmRecognizer::recognize_image_file_words`]), comparing the output
//! against a golden file under `corpus/golden/lines/`.
//!
//! These are hermetic gates: a missing corpus or golden file is a test
//! failure, not a skip. Set `UPDATE_GOLDEN=1` to (re)write the golden files
//! from the current recognizer output instead of asserting against them:
//!
//! ```sh
//! UPDATE_GOLDEN=1 cargo test -p tesseract-ocr --test golden_lines
//! cargo test -p tesseract-ocr --test golden_lines
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tesseract_core::DictLite;
use tesseract_ocr::{parse_pgm, render_tsv, LineWords, LstmRecognizer};

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
/// `corpus/model/`. Returns an owned clone since every consuming API here
/// (`recognize_image_file_with_dict`, `recognize_image_file_words`) takes
/// `DictLite` by value.
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

/// The non-dict line fixtures. `page_roomy`/`page_tight` (also under
/// `corpus/lines/`) are NOT included here -- they belong to the makerow
/// page-path unit test in `src/lstm_recognizer.rs`.
const LINE_FIXTURES: [&str; 7] = [
    "img_8", "img_16", "img_24", "img_40", "img_64", "img_100", "line36",
];

fn line_path(stem: &str) -> PathBuf {
    corpus_dir().join("lines").join(format!("{stem}.pgm"))
}

#[test]
fn golden_line_text() {
    for stem in LINE_FIXTURES {
        let (_ids, text) = recognizer()
            .recognize_image_file(&line_path(stem))
            .unwrap_or_else(|e| panic!("recognize_image_file({stem}): {e}"));
        let golden_path = corpus_dir()
            .join("golden")
            .join("lines")
            .join(format!("{stem}.txt"));
        check_golden(&golden_path, &format!("{text}\n"), stem);
    }
}

#[test]
fn golden_line36_dict_text() {
    let (_ids, text) = recognizer()
        .recognize_image_file_with_dict(&line_path("line36"), dict())
        .unwrap_or_else(|e| panic!("recognize_image_file_with_dict(line36): {e}"));
    let golden_path = corpus_dir()
        .join("golden")
        .join("lines")
        .join("line36.dict.txt");
    check_golden(&golden_path, &format!("{text}\n"), "line36.dict");
}

/// TSV mirroring note: the brief pointed at
/// `examples/recognize_image_dict_dump.rs` for "how `Vec<WordResult>` turns
/// into TSV text", but that example (read in full) only prints `uids`/
/// `text` via `println!` -- it never constructs a `LineWords` or calls
/// `render_tsv`. The actual (and only) existing
/// `Vec<WordResult>` -> `LineWords` -> `render_tsv` composition in this
/// crate is `examples/render_page_tsv.rs`; this test mirrors THAT example's
/// pattern instead (the dict-loading half matches
/// `recognize_image_dict_dump.rs` verbatim, and matches `render_page_tsv.rs`
/// too, which loads its dict identically).
///
/// `render_page_tsv.rs` derives `page_w`/`page_h` as
/// `box_r.max(0) as u32`/`box_t.max(0) as u32`. Here `line_box = (0, 0, w as
/// i32, h as i32)` per this test's spec, so `box_r == w as i32` and
/// `box_t == h as i32`; since `w`/`h` (from `parse_pgm`) are already
/// non-negative `usize`, `w as u32`/`h as u32` is the same value without the
/// redundant `.max(0)` guard (that guard exists in the example because its
/// box coordinates are free-form CLI-supplied `i32`s, not derived from an
/// unsigned image dimension).
///
/// The golden is the raw `render_tsv` return value, unmodified: in
/// `render_page_tsv.rs`, `print!("{tsv}")` writes exactly that string to
/// stdout (the plain-text rendering goes to stderr separately via a
/// different call), so there are no extra non-TSV lines to strip here.
#[test]
fn golden_line36_dict_tsv() {
    let img_path = line_path("line36");
    let bytes = fs::read(&img_path).unwrap_or_else(|e| panic!("read {}: {e}", img_path.display()));
    let (_grey, w, h) = parse_pgm(&bytes).unwrap_or_else(|e| panic!("parse_pgm(line36): {e}"));
    let line_box = (0, 0, w as i32, h as i32);

    let words = recognizer()
        .recognize_image_file_words(&img_path, Some(dict()), line_box, 1.0)
        .unwrap_or_else(|e| panic!("recognize_image_file_words(line36, dict): {e}"));
    let line = LineWords { words, line_box };

    let tsv = render_tsv(&[line], &recognizer().charset, w as u32, h as u32);

    let golden_path = corpus_dir()
        .join("golden")
        .join("lines")
        .join("line36.dict.tsv");
    check_golden(&golden_path, &tsv, "line36.dict.tsv");
}

#[test]
fn golden_lines_deterministic() {
    let path = line_path("img_24");
    let (ids1, text1) = recognizer()
        .recognize_image_file(&path)
        .unwrap_or_else(|e| panic!("recognize_image_file(img_24) run 1: {e}"));
    let (ids2, text2) = recognizer()
        .recognize_image_file(&path)
        .unwrap_or_else(|e| panic!("recognize_image_file(img_24) run 2: {e}"));
    assert_eq!(
        text1, text2,
        "recognize_image_file(img_24) text must be deterministic"
    );
    assert_eq!(
        ids1, ids2,
        "recognize_image_file(img_24) unichar ids must be deterministic"
    );
}
