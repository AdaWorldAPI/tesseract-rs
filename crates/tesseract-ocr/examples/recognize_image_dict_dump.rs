//! D1.3 byte-parity surface: recognize an image FILE on disk → text **with the
//! dictionary active** — the dict-path counterpart of `recognize_image_dump`.
//! Same P5 PGM decode → pre-scale → A6a `from_grey_pix` pipeline, but decodes
//! via [`LstmRecognizer::recognize_image_file_with_dict`] (`RecodeBeamSearch::
//! new_with_dict` + `decode_with_dict`, the production `kDictRatio = 2.25`,
//! `kCertOffset = -0.085`, `worst_dict_cert = kWorstDictCertainty /
//! kCertaintyScale = -25.0/7.0`).
//!
//! ```sh
//! cargo run -p tesseract-ocr --example recognize_image_dict_dump -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder \
//!   /tmp/eng.lstm-word-dawg /tmp/eng.lstm-punc-dawg /tmp/eng.lstm-number-dawg \
//!   /tmp/line36.pgm > /tmp/rust_img_dict.tsv
//! /tmp/image_text_dict_oracle /tmp/eng.lstm /tmp/eng.lstm-unicharset \
//!   /tmp/eng.lstm-recoder /tmp/line36.pgm > /tmp/oracle_img_dict.tsv
//! diff /tmp/oracle_img_dict.tsv /tmp/rust_img_dict.tsv   # byte-identical => dict-path green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

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

    let lstm = std::fs::read(lstm_path).expect("read eng.lstm");
    let uni = std::fs::read_to_string(uni_path).expect("read unicharset");
    let rec = std::fs::read(rec_path).expect("read recoder");
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer");

    let word = std::fs::read(word_dawg_path).expect("read word dawg");
    let punc = std::fs::read(punc_dawg_path).expect("read punc dawg");
    let number = std::fs::read(number_dawg_path).expect("read number dawg");
    let dict = DictLite::from_components(&word, &punc, &number).expect("load dict");

    let (uids, text) = r
        .recognize_image_file_with_dict(Path::new(img_path), dict)
        .expect("recognize image with dict");

    print!("uids");
    for u in &uids {
        print!("\t{u}");
    }
    println!();
    println!("text\t{text}");
}
