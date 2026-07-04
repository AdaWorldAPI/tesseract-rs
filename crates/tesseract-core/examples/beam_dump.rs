//! Generate a synthetic softmax matrix, decode it with the Rust
//! [`RecodeBeamSearch`](tesseract_core::RecodeBeamSearch), write that SAME matrix
//! to a `.bin` for the C++ oracle to read (so the INPUT is byte-identical), and
//! print `label`/`xcoord` lines — the byte-parity surface for recognizer Leaf 7b.
//!
//! ```sh
//! # C++ oracle (recodebeam_oracle.cpp) reads the .bin this writes and runs the
//! # REAL RecodeBeamSearch::Decode + ExtractBestPathAsLabels:
//! cargo run -p tesseract-core --example beam_dump -- /tmp/eng.lstm-recoder /tmp/beam_probs.bin 110 0 > /tmp/rust_beam.tsv
//! /tmp/recodebeam_oracle /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/beam_probs.bin 110 0 > /tmp/oracle_beam.tsv
//! diff /tmp/oracle_beam.tsv /tmp/rust_beam.tsv   # byte-identical => Leaf 7b green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;
use std::path::Path;

use tesseract_core::{RecodeBeamSearch, UnicharCompress};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let recoder_path = args.get(1).map_or("/tmp/eng.lstm-recoder", String::as_str);
    let bin_path = args.get(2).map_or("/tmp/beam_probs.bin", String::as_str);
    let null_char: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(110);
    let simple: bool = args.get(4).is_some_and(|s| s == "1");

    let recoder = UnicharCompress::load_from_file(Path::new(recoder_path)).expect("load recoder");
    let n = recoder.code_range() as usize; // 111 for eng.lstm (the Fc111 output width)

    // Winner pattern (T=8): a code held across steps (CTC-folds) with nulls
    // between, to exercise folding + null-dropping.
    let winners: [i32; 8] = [5, 5, null_char, 7, 7, 7, null_char, 9];

    // Distinct (tie-free) probabilities: a tiny per-code base breaks ties, the
    // winner gets +0.8, the null +0.1 (unless it is the winner), then normalize.
    let rows: Vec<Vec<f32>> = winners
        .iter()
        .map(|&w| {
            let mut row = vec![0.0_f32; n];
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = 0.001 + (c as f32) * 1e-5;
            }
            row[w as usize] += 0.8;
            if w != null_char {
                row[null_char as usize] += 0.1;
            }
            let sum: f32 = row.iter().sum();
            for slot in &mut row {
                *slot /= sum;
            }
            row
        })
        .collect();

    // Write the shared .bin: i32 T, i32 N, then T·N f32 LE.
    let mut buf = Vec::new();
    buf.extend_from_slice(&(rows.len() as i32).to_le_bytes());
    buf.extend_from_slice(&(n as i32).to_le_bytes());
    for row in &rows {
        for &v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(bin_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write probs.bin");

    // Decode with the Rust beam and dump labels + xcoords.
    let mut beam = RecodeBeamSearch::new(&recoder, null_char, simple);
    let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    beam.decode(&refs, 1.0, 0.0);
    let (labels, xcoords) = beam.extract_best_path_as_labels();
    for label in &labels {
        println!("label\t{label}");
    }
    for xcoord in &xcoords {
        println!("xcoord\t{xcoord}");
    }
}
