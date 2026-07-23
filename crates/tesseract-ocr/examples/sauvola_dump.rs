//! Dump `tesseract_ocr::binarize::sauvola_binarize` output for byte-parity against
//! `.claude/harvest/oracles/sauvola_oracle.cpp` (leptonica `pixSauvolaBinarize`).
//!
//! Emits one line per pixel: `"<idx>\t<threshold>\t<binary 0|1>"` — identical to
//! the oracle. `addborder = 1` is implied (the transcode's document path).
//!
//! ```sh
//! g++ -std=c++17 .claude/harvest/oracles/sauvola_oracle.cpp -I/usr/include/leptonica \
//!     -lleptonica -o /tmp/sauvola_oracle
//! /tmp/sauvola_oracle /tmp/sauvola_in.pgm 8 0.34 > /tmp/o_sauvola.tsv
//! cargo run -q -p tesseract-ocr --example sauvola_dump -- /tmp/sauvola_in.pgm 8 0.34 > /tmp/r_sauvola.tsv
//! diff /tmp/o_sauvola.tsv /tmp/r_sauvola.tsv   # byte-identical => Sauvola parity holds
//! ```

use std::path::Path;

use tesseract_ocr::binarize::sauvola_binarize;
use tesseract_ocr::image_input::parse_pgm;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: sauvola_dump <grey.pgm> <whsize> <factor>");
        std::process::exit(2);
    }
    let bytes = std::fs::read(Path::new(&args[1])).expect("read pgm");
    let whsize: usize = args[2].parse().expect("whsize");
    let factor: f32 = args[3].parse().expect("factor");

    let (grey, w, h) = parse_pgm(&bytes).expect("parse pgm");
    let s = sauvola_binarize(&grey, w, h, whsize, factor);

    let mut out = String::with_capacity(w * h * 12);
    for idx in 0..w * h {
        out.push_str(&idx.to_string());
        out.push('\t');
        out.push_str(&s.threshold[idx].to_string());
        out.push('\t');
        out.push_str(&s.binary[idx].to_string());
        out.push('\n');
    }
    print!("{out}");
}
