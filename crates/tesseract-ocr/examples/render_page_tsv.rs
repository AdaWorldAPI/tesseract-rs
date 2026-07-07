//! P4a end-to-end demo: recognize a line image into [`tesseract_ocr::LineWords`],
//! then render it both ways — plain text ([`tesseract_ocr::render_text`]) and
//! Tesseract TSV ([`tesseract_ocr::render_tsv`]). See `crates/tesseract-ocr/src/renderer.rs`
//! for the exact C++ source lines each formatting decision transcodes.
//!
//! This wraps a single recognized line as a one-line "page" (`block_num =
//! par_num = 1`, `line_num = 1` — the APPROX placeholders documented in the
//! renderer module, since this crate has no textord layout stage).
//!
//! ```sh
//! cargo run -p tesseract-ocr --example render_page_tsv -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder \
//!   /tmp/eng.lstm-word-dawg /tmp/eng.lstm-punc-dawg /tmp/eng.lstm-number-dawg \
//!   /tmp/line36.pgm 0 0 1000 36 1.0 > /tmp/rust_page.tsv
//! # render_text goes to stderr, render_tsv (the byte-parity surface for a
//! # future TSV oracle) goes to stdout.
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;

use tesseract_core::DictLite;
use tesseract_ocr::{render_text, render_tsv, LineWords, LstmRecognizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm_path = args.get(1).map_or("/tmp/eng.lstm", String::as_str);
    let uni_path = args
        .get(2)
        .map_or("/tmp/eng.lstm-unicharset", String::as_str);
    let rec_path = args.get(3).map_or("/tmp/eng.lstm-recoder", String::as_str);
    let word_dawg_path = args
        .get(4)
        .map_or("/tmp/eng.lstm-word-dawg", String::as_str);
    let punc_dawg_path = args
        .get(5)
        .map_or("/tmp/eng.lstm-punc-dawg", String::as_str);
    let number_dawg_path = args
        .get(6)
        .map_or("/tmp/eng.lstm-number-dawg", String::as_str);
    let img_path = args.get(7).map_or("/tmp/line36.pgm", String::as_str);
    let box_l: i32 = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let box_b: i32 = args.get(9).and_then(|s| s.parse().ok()).unwrap_or(0);
    let box_r: i32 = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let box_t: i32 = args.get(11).and_then(|s| s.parse().ok()).unwrap_or(36);
    let scale_factor: f32 = args.get(12).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let lstm = std::fs::read(lstm_path).expect("read eng.lstm");
    let uni = std::fs::read_to_string(uni_path).expect("read unicharset");
    let rec = std::fs::read(rec_path).expect("read recoder");
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer");

    let word = std::fs::read(word_dawg_path).expect("read word dawg");
    let punc = std::fs::read(punc_dawg_path).expect("read punc dawg");
    let number = std::fs::read(number_dawg_path).expect("read number dawg");
    let dict = DictLite::from_components(&word, &punc, &number).expect("load dict");

    let line_box = (box_l, box_b, box_r, box_t);
    let words = r
        .recognize_image_file_words(Path::new(img_path), Some(dict), line_box, scale_factor)
        .expect("recognize image words");

    // page_w/page_h — the TSV box conversion's `pix_height`/`pix_width`
    // (`renderer.rs::to_image_rect`). This is a single-line "page", so we
    // use the line_box's own extent (box_r as width, box_t as the bottom-up
    // height — the same convention `recognize_image_file_words` was given).
    let page_w = box_r.max(0) as u32;
    let page_h = box_t.max(0) as u32;

    let line = LineWords { words, line_box };
    let charset = &r.charset;

    let text = render_text(std::slice::from_ref(&line), charset);
    eprint!("{text}");

    let tsv = render_tsv(std::slice::from_ref(&line), charset, page_w, page_h);
    print!("{tsv}");
}
