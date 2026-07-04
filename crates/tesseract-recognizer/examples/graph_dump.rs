//! Generate a `Series[LSTM, FC(tanh)]` (LSTM payload + one FC `WeightMatrix`) to
//! a file, then dump `Layer::forward()` over a deterministic int8 sequence as f32
//! bit-patterns — the byte-parity surface for the graph-walk composition (Leaf
//! 6). The libtesseract oracle (`/tmp/graph_oracle.cpp`) reads the SAME file and
//! runs the REAL LSTM per-timestep body → the REAL `WriteTimeStep` int8 requant →
//! the REAL `MatrixDotVector`+`FuncInplace<GFunc>` — proving the inter-layer
//! requant + chaining order.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example graph_dump -- /tmp/graph.bin 8 5 6 4 > /tmp/rust_graph.tsv
//! # (ns=8 ni=5 fc_no=6 steps=4)
//! /tmp/graph_oracle /tmp/graph.bin 4 > /tmp/oracle_graph.tsv
//! diff /tmp/oracle_graph.tsv /tmp/rust_graph.tsv   # byte-identical => Leaf 6 green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;

use tesseract_recognizer::{FcActivation, Layer, Lstm, WeightMatrix};

const K_INT8_FLAG: u8 = 1;
const K_DOUBLE_FLAG: u8 = 128;
const INT8_MAX_F64: f64 = 127.0;

fn wv(i: usize, j: usize, seed: i64) -> i8 {
    (((i as i64 * 7 + j as i64 * 3 + seed) % 251) - 125) as i8
}
fn sv(i: usize) -> f64 {
    ((i % 7) + 1) as f64 * 0.05
}
fn iv(t: usize, j: usize) -> i8 {
    (((t as i64 * 13 + j as i64 * 5 + 2) % 251) - 125) as i8
}

fn push_wm(b: &mut Vec<u8>, num_out: usize, num_in: usize, seed: i64) {
    let dim2 = num_in + 1;
    b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
    b.extend_from_slice(&(num_out as u32).to_le_bytes());
    b.extend_from_slice(&(dim2 as u32).to_le_bytes());
    b.push(0);
    for i in 0..num_out {
        for j in 0..dim2 {
            b.push(wv(i, j, seed) as u8);
        }
    }
    b.extend_from_slice(&(num_out as u32).to_le_bytes());
    for i in 0..num_out {
        b.extend_from_slice(&(sv(i) * INT8_MAX_F64).to_le_bytes());
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "/tmp/graph.bin".into());
    let ns: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let ni: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let fc_no: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    let steps: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
    let na = ni + ns;

    // File = LSTM payload (i32 na_ + 4 gates) then the FC WeightMatrix (fc_no × ns+1).
    let mut b = Vec::new();
    b.extend_from_slice(&(na as i32).to_le_bytes());
    for seed in [10_i64, 20, 30, 40] {
        push_wm(&mut b, ns, na, seed);
    }
    push_wm(&mut b, fc_no, ns, 50); // FC: takes the LSTM's ns outputs
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&b))
        .expect("write graph.bin");

    // Build Series[LSTM, FC(tanh)] and run it.
    let (lstm, consumed) = Lstm::from_le_bytes(&b).expect("lstm");
    let fc_weights = WeightMatrix::from_le_bytes(&b[consumed..]).expect("fc");
    let net = Layer::Series(vec![
        Layer::Lstm(Box::new(lstm)),
        Layer::FullyConnected {
            weights: fc_weights,
            activation: FcActivation::Tanh,
        },
    ]);
    let s: Vec<Vec<i8>> = (0..steps)
        .map(|t| (0..ni).map(|j| iv(t, j)).collect())
        .collect();
    let refs: Vec<&[i8]> = s.iter().map(Vec::as_slice).collect();
    let out = net.forward(&refs).expect("forward");
    for (t, line) in out.iter().enumerate() {
        for (i, &val) in line.iter().enumerate() {
            println!("{t}\t{i}\t{:08x}", val.to_bits());
        }
    }
}
