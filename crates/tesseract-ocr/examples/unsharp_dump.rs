//! pixScale sub-leaf byte-parity surface: read a PGM, apply
//! [`unsharp_mask_gray_2d`] (the `pixUnsharpMaskingGray2D` transcode) at
//! `halfwidth`/`fract`, dump the sharpened grey. The oracle
//! (`/tmp/unsharp_oracle.cpp`) reads the SAME PGM and calls the REAL public
//! `pixUnsharpMasking(pix, halfwidth, fract)` — which, for 8-bit grey +
//! `halfwidth ∈ {1,2}`, routes to `pixUnsharpMaskingGray2D`.
//!
//! `fract` is parsed as `f64` then narrowed to `f32` — matching how `pixScale`'s
//! `0.2`/`0.4` double literals narrow to its `l_float32 sharpfract`, and how the
//! oracle's `(float)atof(...)` narrows — so both sides use the identical `f32`.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example unsharp_dump -- /tmp/img50.pgm 2 0.4 > /tmp/rust_unsharp.tsv
//! /tmp/unsharp_oracle /tmp/img50.pgm 2 0.4 > /tmp/oracle_unsharp.tsv
//! diff /tmp/oracle_unsharp.tsv /tmp/rust_unsharp.tsv
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_ocr::image_input::{parse_pgm, unsharp_mask_gray_2d};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pgm = args.get(1).map_or("/tmp/img50.pgm", String::as_str);
    let halfwidth: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    // Parse as f64 then narrow → identical f32 to the oracle's (float)atof.
    let fract: f32 = args
        .get(3)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.4) as f32;

    let bytes = std::fs::read(pgm).expect("read pgm");
    let (grey, w, h) = parse_pgm(&bytes).expect("parse pgm");
    let out = unsharp_mask_gray_2d(&grey, w, h, halfwidth, fract);

    println!("dim\t{w}\t{h}");
    for y in 0..h {
        print!("r\t{y}");
        for &b in &out[y * w..(y + 1) * w] {
            print!("\t{b}");
        }
        println!();
    }
}
