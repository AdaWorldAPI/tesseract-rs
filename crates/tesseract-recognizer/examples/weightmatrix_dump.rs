//! Generate a seeded int-mode `WeightMatrix` serialization to a file, then dump
//! this crate's `forward()` output as f32 bit-patterns — for byte-parity diff
//! against the libtesseract oracle (`/tmp/weightmatrix_oracle.cpp`), which reads
//! the SAME file and dumps its own `DeSerialize` + `MatrixDotVector`. Because
//! libtesseract parses the bytes with the REAL format, a wrong wire layout here
//! would make its output diverge — so the diff is an independent proof.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example weightmatrix_dump -- /tmp/wm.bin 8 5 > /tmp/rust_wm.tsv
//! g++ -std=c++17 -DFAST_FLOAT /tmp/weightmatrix_oracle.cpp \
//!   -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/arch \
//!   -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
//!   $(pkg-config --cflags tesseract) -o /tmp/weightmatrix_oracle \
//!   $(pkg-config --libs tesseract) $(pkg-config --libs lept)
//! /tmp/weightmatrix_oracle /tmp/wm.bin 5 > /tmp/oracle_wm.tsv
//! diff /tmp/oracle_wm.tsv /tmp/rust_wm.tsv   # byte-identical => Leaf 2 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;

use tesseract_recognizer::WeightMatrix;

// Deterministic synthetic values — the WRITER side; the oracle only READS the file.
fn wv(i: usize, j: usize) -> i8 {
    (((i as i64 * 7 + j as i64 * 3) % 251) - 125) as i8
}
fn sv(i: usize) -> f64 {
    ((i % 7) + 1) as f64 * 0.03
}
fn uv(j: usize) -> i8 {
    (((j as i64 * 5 + 2) % 251) - 125) as i8
}

const K_INT8_FLAG: u8 = 1;
const K_DOUBLE_FLAG: u8 = 128;
const INT8_MAX_F64: f64 = 127.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "/tmp/wm.bin".into());
    let num_out: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let num_in: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let dim2 = num_in + 1;

    // Build the int-mode WeightMatrix wire bytes (the exact Serialize layout).
    let mut b = Vec::new();
    b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
    b.extend_from_slice(&(num_out as u32).to_le_bytes());
    b.extend_from_slice(&(dim2 as u32).to_le_bytes());
    b.push(0); // empty_
    for i in 0..num_out {
        for j in 0..dim2 {
            b.push(wv(i, j) as u8);
        }
    }
    b.extend_from_slice(&(num_out as u32).to_le_bytes());
    for i in 0..num_out {
        b.extend_from_slice(&(sv(i) * INT8_MAX_F64).to_le_bytes());
    }
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&b))
        .expect("write wm.bin");

    // Load it back and dump this crate's forward() as f32 bit-patterns.
    let wm = WeightMatrix::from_le_bytes(&b).expect("valid wm");
    let u: Vec<i8> = (0..num_in).map(uv).collect();
    let v = wm.forward(&u).expect("forward");
    for (i, &val) in v.iter().enumerate() {
        println!("{i}\t{:08x}", val.to_bits());
    }
}
