//! B2 byte-parity surface: assemble an [`LstmRecognizer`] from the three split
//! traineddata components (`eng.lstm` + `eng.lstm-unicharset` +
//! `eng.lstm-recoder`) and dump the trailing scalar fields the C++
//! `LSTMRecognizer::DeSerialize` reads after the network. The oracle
//! (`/tmp/lstm_recognizer_oracle.cpp`) reads the SAME fields off the SAME
//! `eng.lstm` via libtesseract's `TFile` + `Network::CreateFromFile`.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example lstm_recognizer_dump -- \
//!     /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder > /tmp/rust_lstmrec.tsv
//! /tmp/lstm_recognizer_oracle /tmp/eng.lstm > /tmp/oracle_lstmrec.tsv
//! # the 8 trailing-parse lines are byte-identical => B2 green
//! diff <(grep -E '^(netstr|tflags|titer|siter|null|abeta|lrate|moment)' /tmp/oracle_lstmrec.tsv) \
//!      <(grep -E '^(netstr|tflags|titer|siter|null|abeta|lrate|moment)' /tmp/rust_lstmrec.tsv)
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_ocr::LstmRecognizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm_path = args.get(1).map_or("/tmp/eng.lstm", String::as_str);
    let uni_path = args
        .get(2)
        .map_or("/tmp/eng.lstm-unicharset", String::as_str);
    let rec_path = args.get(3).map_or("/tmp/eng.lstm-recoder", String::as_str);

    let lstm = std::fs::read(lstm_path).expect("read eng.lstm");
    let uni = std::fs::read_to_string(uni_path).expect("read unicharset");
    let rec = std::fs::read(rec_path).expect("read recoder");

    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer");

    // The 8 trailing-parse fields — byte-identical to the oracle's dump.
    println!("netstr\t{}", r.network_str);
    println!("tflags\t{}", r.training_flags);
    println!("titer\t{}", r.training_iteration);
    println!("siter\t{}", r.sample_iteration);
    println!("null\t{}", r.null_char);
    println!("abeta\t{:08x}", r.adam_beta.to_bits());
    println!("lrate\t{:08x}", r.learning_rate.to_bits());
    println!("moment\t{:08x}", r.momentum.to_bits());

    // Assembly cross-checks (distinct prefixes — not part of the parity diff;
    // the components are each already byte-parity-proven, E-CPP-PARITY-1..7).
    println!("xnw\t{}", r.network.num_weights);
    println!("xcharset\t{}", r.charset.size());
    println!("xcoderange\t{}", r.recoder.code_range());
    println!("xrecoding\t{}", r.is_recoding());
    println!("xintmode\t{}", r.is_int_mode());
}
