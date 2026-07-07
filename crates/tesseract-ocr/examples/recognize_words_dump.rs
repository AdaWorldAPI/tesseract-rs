//! Byte-parity surface for the word/box output leaf:
//! `RecodeBeamSearch::ExtractBestPathAsWords` (`recodebeam.cpp:239-322`) — the
//! word-splitting counterpart of `recognize_image_dict_dump`'s flat unichar-id
//! run. The oracle (`/tmp/words_oracle.cpp`) is VERBATIM through the beam
//! decode (real network forward + real dict, production `kDictRatio =
//! 2.25`/`kCertOffset = -0.085`/`worst_dict_cert = -25.0/7.0`), then dumps the
//! per-word / per-character-cell fields this example reproduces.
//!
//! Dump format (tab-separated; matches `/tmp/words_oracle_format.txt`).
//! **stdout is the byte-parity surface** — only `w`/`c` lines; the
//! `num_words=<N>` self-description goes to stderr, mirroring the oracle:
//!
//! ```text
//! w\t<i>\t<leading_space 0|1>\t<permuter %d>\t<space_certainty %08x>
//! c\t<unichar_id %d>\t<left %d>\t<bottom %d>\t<right %d>\t<top %d>\t<cert %08x>\t<rating %08x>
//! ```
//!
//! ```sh
//! cargo run -p tesseract-ocr --example recognize_words_dump -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder \
//!   /tmp/eng.lstm-word-dawg /tmp/eng.lstm-punc-dawg /tmp/eng.lstm-number-dawg \
//!   /tmp/line36.pgm 0 0 1000 36 1.0 > /tmp/rust_words.tsv
//! /tmp/words_oracle /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder \
//!   /tmp/line36.pgm 0 0 1000 36 1.0 > /tmp/oracle_words.tsv 2> /tmp/oracle_words.stderr
//! grep -E '^(w|c)\t' /tmp/oracle_words.tsv > /tmp/oracle_words.filtered.tsv
//! tail -n +2 /tmp/rust_words.tsv > /tmp/rust_words.filtered.tsv
//! diff /tmp/oracle_words.filtered.tsv /tmp/rust_words.filtered.tsv   # byte-identical => words-path green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

fn print_f32_hex(v: f32) {
    print!("{:08x}", v.to_bits());
}

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

    let words = r
        .recognize_image_file_words(
            Path::new(img_path),
            Some(dict),
            (box_l, box_b, box_r, box_t),
            scale_factor,
        )
        .expect("recognize image words");

    eprintln!("num_words={}", words.len());
    for (i, word) in words.iter().enumerate() {
        print!(
            "w\t{i}\t{}\t{}\t",
            i32::from(word.leading_space),
            word.permuter as i32
        );
        print_f32_hex(word.space_certainty);
        println!();

        for (col, &unichar_id) in word.unichar_ids.iter().enumerate() {
            let (left, bottom, right, top) = word
                .char_boxes
                .get(col)
                .copied()
                .unwrap_or((-1, -1, -1, -1));
            print!("c\t{unichar_id}\t{left}\t{bottom}\t{right}\t{top}\t");
            print_f32_hex(word.certs[col]);
            print!("\t");
            print_f32_hex(word.ratings[col]);
            println!();
        }
    }
}
