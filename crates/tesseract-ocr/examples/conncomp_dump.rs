//! Batch-3B byte-parity surface: connected-component bounding boxes.
//!
//! Two modes:
//! - default: self-generates the session-standard synthetic grey image
//!   `((x*37 + y*11) ^ (x*y)) % 256`, Otsu-binarizes it (full rect), writes
//!   `/tmp/conncomp_input.bin` for the oracle, and dumps the boxes.
//! - `--pgm <path>`: reads a real PGM, Otsu-binarizes, connectivity 8.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example conncomp_dump -- 24 36 4
//! /tmp/conncomp_oracle /tmp/conncomp_input.bin
//! # diff the two outputs => byte-parity
//! ```
#![allow(clippy::print_stdout, reason = "dump CLI")]

use tesseract_ocr::{conn_comp_bb, otsu_threshold_gray, threshold_rect_to_binary};

fn synthetic_grey(w: usize, h: usize) -> Vec<u8> {
    let mut grey = vec![0_u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 37 + y * 11) ^ (x * y)) % 256;
            grey[y * w + x] = u8::try_from(v).expect("mod 256 fits u8");
        }
    }
    grey
}

fn binarize(grey: &[u8], w: usize, h: usize) -> Vec<u8> {
    let otsu = otsu_threshold_gray(grey, w, 0, 0, w, h);
    threshold_rect_to_binary(grey, w, 0, 0, w, h, otsu)
}

fn dump(binary: &[u8], w: usize, h: usize, connectivity: u32) {
    // Write the oracle's input file: i32 w, i32 h, i32 connectivity, then
    // w*h raw bytes.
    let mut buf = Vec::with_capacity(12 + binary.len());
    buf.extend_from_slice(&i32::try_from(w).expect("w fits i32").to_le_bytes());
    buf.extend_from_slice(&i32::try_from(h).expect("h fits i32").to_le_bytes());
    buf.extend_from_slice(&(connectivity as i32).to_le_bytes());
    buf.extend_from_slice(binary);
    std::fs::write("/tmp/conncomp_input.bin", &buf).expect("write oracle input");

    let boxes = conn_comp_bb(binary, w, h, connectivity);
    println!("n\t{}", boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        println!("bb\t{i}\t{}\t{}\t{}\t{}", b.x, b.y, b.w, b.h);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(String::as_str) == Some("--pgm") {
        let path = args.get(2).map_or("/tmp/line36.pgm", String::as_str);
        let bytes = std::fs::read(path).expect("read pgm");
        let (grey, w, h) = tesseract_ocr::parse_pgm(&bytes).expect("parse pgm");
        let binary = binarize(&grey, w, h);
        dump(&binary, w, h, 8);
        return;
    }

    let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);
    let connectivity: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let grey = synthetic_grey(w, h);
    let binary = binarize(&grey, w, h);
    dump(&binary, w, h, connectivity);
}
