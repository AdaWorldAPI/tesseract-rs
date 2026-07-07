//! pixScale sub-leaf byte-parity surface: read a PGM, scale it to
//! `target_height` via [`scale_gray_li`] (the `pixScaleGrayLI` transcode), dump
//! the scaled grey bytes. The oracle (`/tmp/scale_li_oracle.cpp`) reads the SAME
//! PGM (`pixRead`), calls the REAL `pixScaleGrayLI(pix, f, f)`, dumps the same.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example scale_li_dump -- /tmp/img50.pgm 36 > /tmp/rust_scaleli.tsv
//! /tmp/scale_li_oracle /tmp/img50.pgm 36 > /tmp/oracle_scaleli.tsv
//! diff /tmp/oracle_scaleli.tsv /tmp/rust_scaleli.tsv   # byte-identical => sub-leaf green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_ocr::image_input::{parse_pgm, scale_gray_li};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pgm = args.get(1).map_or("/tmp/img50.pgm", String::as_str);
    let target_h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);

    let bytes = std::fs::read(pgm).expect("read pgm");
    let (grey, w, h) = parse_pgm(&bytes).expect("parse pgm");

    // f = target/height (f32, exactly as ImageData::PreScale). pixScaleGrayLI's
    // own dims: wd = round(f·ws), hd = round(f·hs).
    let f = target_h as f32 / h as f32;
    let wd = (f * w as f32 + 0.5) as usize;
    let hd = (f * h as f32 + 0.5) as usize;
    let out = scale_gray_li(&grey, w, h, wd, hd);

    println!("dim\t{wd}\t{hd}");
    for y in 0..hd {
        print!("r\t{y}");
        for &b in &out[y * wd..(y + 1) * wd] {
            print!("\t{b}");
        }
        println!();
    }
}
