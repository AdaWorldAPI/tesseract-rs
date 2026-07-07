//! 3F₂ E2E: page image → text through the REAL makerow line finder
//! (`recognize_page_makerow` — Otsu → conn_comp_areas → filter_blobs →
//! make_rows → compute_block_xheight → per-row band recognition).
//!
//! ```sh
//! cargo run -p tesseract-ocr --example recognize_page_makerow_dump -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/page_test.pgm \
//!   [word_dawg punc_dawg number_dawg]
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_core::DictLite;
use tesseract_ocr::{parse_pgm, LstmRecognizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm = std::fs::read(args.get(1).map_or("/tmp/eng.lstm", String::as_str)).expect("lstm");
    let uni = std::fs::read_to_string(
        args.get(2)
            .map_or("/tmp/eng.lstm-unicharset", String::as_str),
    )
    .expect("unicharset");
    let rec = std::fs::read(args.get(3).map_or("/tmp/eng.lstm-recoder", String::as_str))
        .expect("recoder");
    let img = std::fs::read(args.get(4).map_or("/tmp/page_test.pgm", String::as_str)).expect("pgm");

    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble");
    let (grey, w, h) = parse_pgm(&img).expect("parse pgm");

    let dict = match (args.get(5), args.get(6), args.get(7)) {
        (Some(wd), Some(pd), Some(nd)) => {
            let word = std::fs::read(wd).expect("word dawg");
            let punc = std::fs::read(pd).expect("punc dawg");
            let number = std::fs::read(nd).expect("number dawg");
            Some(DictLite::from_components(&word, &punc, &number).expect("dict"))
        }
        _ => None,
    };

    let text = r
        .recognize_page_makerow(&grey, w, h, dict.as_ref())
        .expect("recognize page (makerow)");
    println!("{text}");
}
