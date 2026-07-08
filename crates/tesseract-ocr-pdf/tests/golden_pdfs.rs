//! Golden integration tests (P6 D6.2): PDF text-layer extraction
//! ([`tesseract_ocr_pdf::extract_text_layer`]).
//!
//! See `tesseract-ocr/tests/golden_lines.rs` (sibling crate) for the
//! corpus-path contract and the `UPDATE_GOLDEN=1` regeneration mode this
//! file follows identically. These are hermetic gates: a missing corpus or
//! golden file is a test failure, not a skip.

use std::fs;
use std::path::{Path, PathBuf};

use tesseract_ocr_pdf::extract_text_layer;

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

/// Encode `extract_text_layer`'s per-page `Option<String>` output into one
/// document string: a page with no text layer becomes the literal sentinel
/// `<NO-TEXT-LAYER>`, pages are joined with `"\f\n"`, and the whole document
/// gets one trailing `"\n"`.
fn encode_pages(pages: &[Option<String>]) -> String {
    let joined = pages
        .iter()
        .map(|p| p.as_deref().unwrap_or("<NO-TEXT-LAYER>"))
        .collect::<Vec<&str>>()
        .join("\u{c}\n");
    format!("{joined}\n")
}

/// Extract + encode `corpus/pdfs/doc_<nn>.pdf` (`nn` a zero-padded
/// two-digit number, e.g. `"01"`).
fn extract_doc(nn: &str) -> String {
    let pdf_path = corpus_dir().join("pdfs").join(format!("doc_{nn}.pdf"));
    let bytes = fs::read(&pdf_path).unwrap_or_else(|e| panic!("read {}: {e}", pdf_path.display()));
    let pages =
        extract_text_layer(&bytes).unwrap_or_else(|e| panic!("extract_text_layer(doc_{nn}): {e}"));
    encode_pages(&pages)
}

#[test]
fn golden_pdf_text_layer() {
    for n in 1..=5 {
        let nn = format!("{n:02}");
        let encoded = extract_doc(&nn);
        let golden_path = corpus_dir()
            .join("golden")
            .join("pdfs")
            .join(format!("doc_{nn}.txt"));
        check_golden(&golden_path, &encoded, &format!("doc_{nn}"));
    }
}

#[test]
fn golden_pdf_deterministic() {
    let a = extract_doc("01");
    let b = extract_doc("01");
    assert_eq!(a, b, "extract_text_layer(doc_01) must be deterministic");
}
