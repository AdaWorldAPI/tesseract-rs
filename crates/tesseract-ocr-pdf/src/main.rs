//! `tesseract-ocr-pdf` — D5.3 orchestrator binary.
//!
//! Per page: does it have an extractable text layer (D5.1)? Print it, no
//! OCR. Otherwise (image-only page): extract the largest image XObject
//! (D5.2, pragmatic image-XObject variant — see
//! [`tesseract_ocr_pdf::extract_page_image`]) and OCR it. A page with
//! neither a text layer nor a supported image XObject (e.g. vector-only
//! content, which needs a full page rasterizer — out of scope, see
//! `.claude/plans/pdf-to-text-ocr-v1.md` Phase 5, D5.2-full) reports the
//! specific reason to stderr and is skipped.
//!
//! ```sh
//! # PDF mode (text-layer fast path, then image-XObject OCR fallback):
//! tesseract-ocr-pdf document.pdf --data-dir /tmp
//!
//! # Direct-image mode (bypasses PDF entirely, demos the OCR arm E2E):
//! tesseract-ocr-pdf /tmp/line36.pgm --data-dir /tmp
//! ```
//!
//! `--data-dir DIR` looks for `eng.lstm`, `eng.lstm-unicharset`,
//! `eng.lstm-recoder`, and (optionally, for the dict-beam path)
//! `eng.lstm-word-dawg` / `eng.lstm-punc-dawg` / `eng.lstm-number-dawg`.
#![allow(
    clippy::print_stdout,
    reason = "the orchestrator's job is to print results"
)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use tesseract_ocr_pdf::{extract_page_image, extract_text_layer, OcrPipeline};

fn parse_args() -> (PathBuf, PathBuf) {
    let mut input = None;
    let mut data_dir = PathBuf::from("/tmp");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--data-dir" {
            if let Some(dir) = args.next() {
                data_dir = PathBuf::from(dir);
            }
        } else if input.is_none() {
            input = Some(PathBuf::from(arg));
        }
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("usage: tesseract-ocr-pdf <pdf-or-pgm> [--data-dir DIR]");
        std::process::exit(2);
    });
    (input, data_dir)
}

fn load_pipeline(data_dir: &Path) -> Result<OcrPipeline, Box<dyn std::error::Error>> {
    let lstm = data_dir.join("eng.lstm");
    let unicharset = data_dir.join("eng.lstm-unicharset");
    let recoder = data_dir.join("eng.lstm-recoder");
    let word_dawg = data_dir.join("eng.lstm-word-dawg");
    let punc_dawg = data_dir.join("eng.lstm-punc-dawg");
    let number_dawg = data_dir.join("eng.lstm-number-dawg");
    let have_dict = word_dawg.exists() && punc_dawg.exists() && number_dawg.exists();
    let pipeline = OcrPipeline::from_data_paths(
        &lstm,
        &unicharset,
        &recoder,
        have_dict.then_some(word_dawg.as_path()),
        have_dict.then_some(punc_dawg.as_path()),
        have_dict.then_some(number_dawg.as_path()),
    )?;
    Ok(pipeline)
}

/// Direct `.pgm` input: bypasses PDF handling entirely, feeds the grey image
/// straight into the OCR arm. Demonstrates D5.3's OCR half is already wired
/// for raw image input, independent of the PDF/raster front end.
fn run_pgm(path: &Path, data_dir: &Path) -> ExitCode {
    let pipeline = match load_pipeline(data_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: loading OCR data from {}: {e}", data_dir.display());
            return ExitCode::FAILURE;
        }
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let (grey, w, h) = match tesseract_ocr::parse_pgm(&bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: parsing {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match pipeline.ocr_grey_page(&grey, w, h) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: OCR failed on {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

/// PDF input: D5.1 text-layer fast path per page; image-only pages fall back
/// to D5.2's image-XObject extraction + OCR. A page that is neither (no
/// text layer AND no supported image XObject — e.g. genuinely vector-only
/// content, or an image encoding D5.2 doesn't cover) reports the specific
/// reason to stderr and is skipped.
fn run_pdf(path: &Path, data_dir: &Path) -> ExitCode {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let pages = match extract_text_layer(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: parsing PDF {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    // The OCR pipeline is only needed (and only loaded) if at least one page
    // lacks a text layer.
    let mut pipeline: Option<Result<OcrPipeline, Box<dyn std::error::Error>>> = None;

    let mut any_output = false;
    for (i, page) in pages.iter().enumerate() {
        let page_number = u32::try_from(i + 1).expect("page index fits u32");
        match page {
            Some(text) => {
                any_output = true;
                println!("--- page {page_number} (text layer) ---");
                println!("{text}");
            }
            None => match extract_page_image(&bytes, page_number) {
                Ok(Some(image)) => {
                    let pipeline = pipeline.get_or_insert_with(|| load_pipeline(data_dir));
                    match pipeline {
                        Ok(pipeline) => match pipeline.ocr_grey_page(&image.data, image.w, image.h)
                        {
                            Ok(text) => {
                                any_output = true;
                                println!("--- page {page_number} (OCR) ---");
                                println!("{text}");
                            }
                            Err(e) => {
                                eprintln!("page {page_number}: OCR failed: {e}");
                            }
                        },
                        Err(e) => {
                            eprintln!(
                                "page {page_number}: image-only, but loading OCR data from {} \
                                 failed: {e}",
                                data_dir.display()
                            );
                        }
                    }
                }
                Ok(None) => {
                    eprintln!(
                        "page {page_number}: image-only (no text layer) — no supported image \
                         XObject on this page either (likely vector-only content); a full \
                         page rasterizer is NOT IMPLEMENTED (D5.2-full pending); see \
                         .claude/plans/pdf-to-text-ocr-v1.md Phase 5"
                    );
                }
                Err(e) => {
                    eprintln!(
                        "page {page_number}: image-only (no text layer) — found an image \
                         XObject but could not decode it: {e}"
                    );
                }
            },
        }
    }
    if any_output {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "warning: no page in {} yielded text or a recognized image",
            path.display()
        );
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let (input, data_dir) = parse_args();
    let is_pgm = input
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("pgm"));
    if is_pgm {
        run_pgm(&input, &data_dir)
    } else {
        run_pdf(&input, &data_dir)
    }
}
