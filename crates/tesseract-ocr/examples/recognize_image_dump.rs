//! A6b byte-parity surface: recognize an image FILE on disk → text, the full
//! pure-Rust `image → text` path (P5 PGM decode → pre-scale → A6a `from_grey_pix`
//! → B3-core `recognize_grid`). The oracle (`/tmp/image_text_oracle.cpp`) reads
//! the SAME PGM via leptonica `pixRead` → `Input::PreparePixInput` → the REAL
//! `net->Forward` + beam + extract + id→text.
//!
//! Byte-parity holds for an image at the model input height (identity scale).
//!
//! ```sh
//! cargo run -p tesseract-ocr --example recognize_image_dump -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/line36.pgm > /tmp/rust_img.tsv
//! /tmp/image_text_oracle /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/line36.pgm > /tmp/oracle_img.tsv
//! diff /tmp/oracle_img.tsv /tmp/rust_img.tsv   # byte-identical => image→text green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;

use tesseract_ocr::LstmRecognizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm_path = args.get(1).map_or("/tmp/eng.lstm", String::as_str);
    let uni_path = args
        .get(2)
        .map_or("/tmp/eng.lstm-unicharset", String::as_str);
    let rec_path = args.get(3).map_or("/tmp/eng.lstm-recoder", String::as_str);
    let img_path = args.get(4).map_or("/tmp/line36.pgm", String::as_str);

    let lstm = std::fs::read(lstm_path).expect("read eng.lstm");
    let uni = std::fs::read_to_string(uni_path).expect("read unicharset");
    let rec = std::fs::read(rec_path).expect("read recoder");
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer");

    let (uids, text) = r
        .recognize_image_file(Path::new(img_path))
        .expect("recognize image");

    print!("uids");
    for u in &uids {
        print!("\t{u}");
    }
    println!();
    println!("text\t{text}");
}
