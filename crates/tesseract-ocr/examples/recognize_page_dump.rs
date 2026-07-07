//! D3.0 (Batch 3-alt) end-to-end demo: a full GREY page (multiple stacked
//! text-line images) → segmented lines → text, via
//! [`LstmRecognizer::recognize_page`].
//!
//! **APPROXIMATION — not a Tesseract transcode; replaced by the textord
//! batches (plan §P3).** This example proves the composition works, not
//! byte-parity vs libtesseract (there is no libtesseract "recognize an
//! arbitrary multi-line page with no layout analysis" API to diff against —
//! that's exactly the gap D3.0 unblocks).
//!
//! ```sh
//! cargo run -p tesseract-ocr --features seg-approx --example recognize_page_dump -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/page_test.pgm
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;

use tesseract_ocr::{parse_pgm, LstmRecognizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm_path = args.get(1).map_or("/tmp/eng.lstm", String::as_str);
    let uni_path = args
        .get(2)
        .map_or("/tmp/eng.lstm-unicharset", String::as_str);
    let rec_path = args.get(3).map_or("/tmp/eng.lstm-recoder", String::as_str);
    let page_path = args.get(4).map_or("/tmp/page_test.pgm", String::as_str);

    let lstm = std::fs::read(lstm_path).expect("read eng.lstm");
    let uni = std::fs::read_to_string(uni_path).expect("read unicharset");
    let rec = std::fs::read(rec_path).expect("read recoder");
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer");

    let bytes = std::fs::read(Path::new(page_path)).expect("read page pgm");
    let (grey, w, h) = parse_pgm(&bytes).expect("parse page pgm");

    let text = r.recognize_page(&grey, w, h, None).expect("recognize page");
    println!("{text}");
}
