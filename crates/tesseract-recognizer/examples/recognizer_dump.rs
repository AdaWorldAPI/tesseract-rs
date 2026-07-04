//! Dump the int8 `MatrixDotVector` combined-integer output on a DETERMINISTIC
//! synthetic case (scales = 1.0), for byte-parity diff against the libtesseract
//! oracle (`/tmp/matdotvec_oracle.cpp`). With scales = 1.0 the output is the
//! exact integer `Σ_j w(i,j)·u[j] + w(i,num_in)·127`, so the diff is on integers
//! and is independent of the lib's `TFloat` width.
//!
//! ```sh
//! g++ -std=c++17 /tmp/matdotvec_oracle.cpp \
//!   -I/tmp/tesseract/src/arch -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
//!   $(pkg-config --cflags tesseract) -o /tmp/matdotvec_oracle \
//!   $(pkg-config --libs tesseract) $(pkg-config --libs lept)
//! /tmp/matdotvec_oracle 48 49 > /tmp/oracle_matdotvec.tsv
//! cargo run -p tesseract-recognizer --example recognizer_dump -- 48 49 > /tmp/rust_matdotvec.tsv
//! diff /tmp/oracle_matdotvec.tsv /tmp/rust_matdotvec.tsv   # byte-identical => Leaf 1 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use ndarray::Array2;
use tesseract_recognizer::matrix_dot_vector;

// Deterministic synthetic int8 — MUST match the C++ oracle's wv/uv.
fn wv(i: usize, j: usize) -> i8 {
    (((i as i64 * 7 + j as i64 * 3) % 251) - 125) as i8
}
fn uv(j: usize) -> i8 {
    (((j as i64 * 5 + 2) % 251) - 125) as i8
}

fn main() {
    let arg = |n: usize, default: usize| {
        std::env::args()
            .nth(n)
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let num_out = arg(1, 48);
    let num_in = arg(2, 49);

    let mut w = Array2::<i8>::zeros((num_out, num_in + 1));
    for i in 0..num_out {
        for j in 0..=num_in {
            w[[i, j]] = wv(i, j);
        }
    }
    let u: Vec<i8> = (0..num_in).map(uv).collect();
    let scales = vec![1.0_f64; num_out]; // scales = 1.0 -> exact integer combined

    match matrix_dot_vector(w.view(), &scales, &u) {
        Ok(v) => {
            for (i, &val) in v.iter().enumerate() {
                // val is the exact integer combined value (scales = 1.0).
                println!("{i}\t{}", val as i64);
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
