//! B3-core byte-parity surface: load the full recognizer (B2) + build a
//! synthetic int8 feature grid (the SAME construction as B1's `network_dump`,
//! written to a shared `.bin`), run [`LstmRecognizer::recognize_grid`], dump the
//! best-path unichar ids + text. The oracle (`/tmp/recognize_grid_oracle.cpp`)
//! loads the same eng.lstm network + recoder + charset via libtesseract, builds
//! the same grid, runs the REAL `network->Forward` + `RecodeBeamSearch::Decode`
//! + `ExtractBestPathAsUnicharIds` + charset id→text.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example recognize_grid_dump -- \
//!   /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/rg_input.bin 24 > /tmp/rust_rg.tsv
//! /tmp/recognize_grid_oracle /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder /tmp/rg_input.bin > /tmp/oracle_rg.tsv
//! diff /tmp/oracle_rg.tsv /tmp/rust_rg.tsv   # byte-identical => B3-core green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;

use tesseract_ocr::LstmRecognizer;
use tesseract_recognizer::{NetworkIo, StrideMap, TRand};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lstm_path = args.get(1).map_or("/tmp/eng.lstm", String::as_str);
    let uni_path = args
        .get(2)
        .map_or("/tmp/eng.lstm-unicharset", String::as_str);
    let rec_path = args.get(3).map_or("/tmp/eng.lstm-recoder", String::as_str);
    let bin_path = args.get(4).map_or("/tmp/rg_input.bin", String::as_str);
    let width: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(24);

    let lstm = std::fs::read(lstm_path).expect("read eng.lstm");
    let uni = std::fs::read_to_string(uni_path).expect("read unicharset");
    let rec = std::fs::read(rec_path).expect("read recoder");
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).expect("assemble recognizer");

    // Synthetic input grid — identical construction to B1's network_dump (the
    // network's Input shape height/depth, chosen width; deterministic f32 →
    // write_time_step int8 quant), written to the shared .bin for the oracle.
    let shape = r.network.input_shape.expect("Input node");
    let (height, depth) = (shape.height, shape.depth);
    let mut map = StrideMap::default();
    map.set_stride(&[(height, width)]);
    let mut input = NetworkIo::default();
    input.resize_to_map(true, &map, depth as usize);

    let mut buf = Vec::new();
    for v in [1_i32, height, width, depth] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let walk = map.clone();
    let mut idx = walk.index_first();
    loop {
        let t = idx.t() as usize;
        let vals: Vec<f32> = (0..depth)
            .map(|f| ((t as i32 * 17 + f * 5) % 200 - 100) as f32 / 100.0)
            .collect();
        for &v in &vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        input.write_time_step(t, &vals);
        if !idx.increment() {
            break;
        }
    }
    std::fs::File::create(bin_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write rg_input.bin");

    let mut rng = TRand::default();
    rng.set_seed(1);
    let (uids, text) = r.recognize_grid(&input, &mut rng).expect("recognize");

    print!("uids");
    for u in &uids {
        print!("\t{u}");
    }
    println!();
    println!("text\t{text}");
}
