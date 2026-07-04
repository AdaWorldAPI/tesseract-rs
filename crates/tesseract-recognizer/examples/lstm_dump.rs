//! Generate a seeded 1-D `LSTM` payload (`i32 na_` + 4 gate `WeightMatrix`es) to
//! a file, then dump this crate's `Lstm::forward()` output over a deterministic
//! int8 sequence as f32 bit-patterns — for byte-parity diff against the
//! libtesseract oracle (`/tmp/lstm_oracle.cpp`), which reads the SAME file via
//! the REAL `WeightMatrix::DeSerialize` and runs the REAL `MatrixDotVector` +
//! `FuncInplace<GFunc/FFunc>` + `MultiplyVectorsInPlace`/`MultiplyAccumulate`/
//! `ClipVector`/`FuncMultiply<HFunc>` — the exact per-timestep body of
//! `LSTM::Forward`.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example lstm_dump -- /tmp/lstm.bin 8 5 3 > /tmp/rust_lstm.tsv
//! g++ -std=c++17 -DFAST_FLOAT /tmp/lstm_oracle.cpp \
//!   -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/arch \
//!   -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
//!   $(pkg-config --cflags tesseract) -o /tmp/lstm_oracle \
//!   $(pkg-config --libs tesseract) $(pkg-config --libs lept)
//! /tmp/lstm_oracle /tmp/lstm.bin 5 3 > /tmp/oracle_lstm.tsv   # ns=8 ni=5 timesteps=3
//! diff /tmp/oracle_lstm.tsv /tmp/rust_lstm.tsv   # byte-identical => Leaf 5 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;

use tesseract_recognizer::Lstm;

// Deterministic synthetic values — the WRITER side; the oracle only READS the file.
fn wv(i: usize, j: usize, seed: i64) -> i8 {
    (((i as i64 * 7 + j as i64 * 3 + seed) % 251) - 125) as i8
}
fn sv(i: usize) -> f64 {
    ((i % 7) + 1) as f64 * 0.05
}
// A per-timestep int8 input (length ni), varied by t so the sequence is non-trivial.
fn iv(t: usize, j: usize) -> i8 {
    (((t as i64 * 13 + j as i64 * 5 + 2) % 251) - 125) as i8
}

const K_INT8_FLAG: u8 = 1;
const K_DOUBLE_FLAG: u8 = 128;
const INT8_MAX_F64: f64 = 127.0;

// Append one int-mode WeightMatrix (num_out=ns × (num_in=na + 1)) to `b`.
fn push_wm(b: &mut Vec<u8>, ns: usize, na: usize, seed: i64) {
    let dim2 = na + 1;
    b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
    b.extend_from_slice(&(ns as u32).to_le_bytes());
    b.extend_from_slice(&(dim2 as u32).to_le_bytes());
    b.push(0); // empty_
    for i in 0..ns {
        for j in 0..dim2 {
            b.push(wv(i, j, seed) as u8);
        }
    }
    b.extend_from_slice(&(ns as u32).to_le_bytes());
    for i in 0..ns {
        b.extend_from_slice(&(sv(i) * INT8_MAX_F64).to_le_bytes());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/lstm.bin".into());
    let ns: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let ni: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let steps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let na = ni + ns;

    // LSTM payload: i32 na_, then the 4 gate matrices (CI, GI, GF1, GO).
    let mut b = Vec::new();
    b.extend_from_slice(&(na as i32).to_le_bytes());
    for (g, seed) in [10_i64, 20, 30, 40].into_iter().enumerate() {
        let _ = g;
        push_wm(&mut b, ns, na, seed);
    }
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&b))
        .expect("write lstm.bin");

    // Load it back and run the recurrence over the deterministic int8 sequence.
    let (lstm, _consumed) = Lstm::from_le_bytes(&b).expect("valid lstm");
    let seq: Vec<Vec<i8>> = (0..steps)
        .map(|t| (0..ni).map(|j| iv(t, j)).collect())
        .collect();
    let refs: Vec<&[i8]> = seq.iter().map(Vec::as_slice).collect();
    let out = lstm.forward(&refs).expect("forward");

    // Dump per-timestep output as f32 bit-patterns: "<t>\t<i>\t<bits>".
    for (t, line) in out.iter().enumerate() {
        for (i, &val) in line.iter().enumerate() {
            println!("{t}\t{i}\t{:08x}", val.to_bits());
        }
    }
}
