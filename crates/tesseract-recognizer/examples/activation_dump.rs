//! Dump `tanh(x)` / `logistic(x)` over a deterministic x sweep as f32
//! bit-patterns — for byte-parity diff against the libtesseract oracle
//! (`/tmp/activation_oracle.cpp`, which dumps `tesseract::Tanh`/`Logistic`). A
//! green diff also proves this crate's regenerated LUTs match libtesseract's
//! baked `TanhTable`/`LogisticTable`.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example activation_dump -- 4096 > /tmp/rust_act.tsv
//! g++ -std=c++17 -DFAST_FLOAT /tmp/activation_oracle.cpp \
//!   -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/ccutil \
//!   $(pkg-config --cflags tesseract) -o /tmp/activation_oracle \
//!   $(pkg-config --libs tesseract) $(pkg-config --libs lept)
//! /tmp/activation_oracle 4096 > /tmp/oracle_act.tsv
//! diff /tmp/oracle_act.tsv /tmp/rust_act.tsv   # byte-identical => Leaf 3 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_recognizer::activation::{logistic, tanh};

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    for k in 0..n {
        // x sweep in [-n/256, +n/256) — same integer arithmetic as the oracle.
        let x = (k - n / 2) as f32 / 128.0;
        println!(
            "{k}\t{:08x}\t{:08x}",
            tanh(x).to_bits(),
            logistic(x).to_bits()
        );
    }
}
