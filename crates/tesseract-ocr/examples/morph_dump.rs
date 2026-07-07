//! Batch-3C byte-parity surface: binary brick morphology.
//!
//! Self-generates the session-standard synthetic grey image
//! `((x*37 + y*11) ^ (x*y)) % 256`, Otsu-binarizes it (full rect), writes
//! `/tmp/morph_input.bin` for the oracle, and dumps the morphed result.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example morph_dump -- 24 36 dilate 3 3
//! /tmp/morph_oracle /tmp/morph_input.bin
//! # diff the two outputs => byte-parity
//! ```
#![allow(clippy::print_stdout, reason = "dump CLI")]

use tesseract_ocr::{
    close_brick, dilate_brick, erode_brick, open_brick, otsu_threshold_gray,
    threshold_rect_to_binary,
};

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

fn op_code(op: &str) -> i32 {
    match op {
        "dilate" => 0,
        "erode" => 1,
        "open" => 2,
        "close" => 3,
        other => panic!("unknown op {other}"),
    }
}

fn dump(binary: &[u8], w: usize, h: usize, op: &str, hsize: usize, vsize: usize) {
    // Write the oracle's input file: i32 w, i32 h, i32 op, i32 hsize,
    // i32 vsize, then w*h raw bytes.
    let mut buf = Vec::with_capacity(20 + binary.len());
    buf.extend_from_slice(&i32::try_from(w).expect("w fits i32").to_le_bytes());
    buf.extend_from_slice(&i32::try_from(h).expect("h fits i32").to_le_bytes());
    buf.extend_from_slice(&op_code(op).to_le_bytes());
    buf.extend_from_slice(&i32::try_from(hsize).expect("hsize fits i32").to_le_bytes());
    buf.extend_from_slice(&i32::try_from(vsize).expect("vsize fits i32").to_le_bytes());
    buf.extend_from_slice(binary);
    std::fs::write("/tmp/morph_input.bin", &buf).expect("write oracle input");

    let out = match op {
        "dilate" => dilate_brick(binary, w, h, hsize, vsize),
        "erode" => erode_brick(binary, w, h, hsize, vsize),
        "open" => open_brick(binary, w, h, hsize, vsize),
        "close" => close_brick(binary, w, h, hsize, vsize),
        other => panic!("unknown op {other}"),
    };

    for y in 0..h {
        print!("m\t{y}");
        for x in 0..w {
            print!("\t{}", out[y * w + x]);
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);
    let op: String = args.get(3).cloned().unwrap_or_else(|| "dilate".to_string());
    let hsize: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let vsize: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);

    let grey = synthetic_grey(w, h);
    let binary = binarize(&grey, w, h);
    dump(&binary, w, h, &op, hsize, vsize);
}
