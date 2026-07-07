//! pixScale (grey, f>=0.7) byte-parity surface: read a PGM, scale to
//! `target_height` via `pix_scale_grey` (the composed pixScaleGrayLI +
//! pixUnsharpMasking dispatch), dump. The oracle calls the REAL leptonica
//! `pixScale(pix, f, f)` — the whole point: byte-parity vs pixScale itself.
#![allow(clippy::print_stdout, reason = "dump CLI")]
use tesseract_ocr::image_input::{parse_pgm, pix_scale_grey};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pgm = args.get(1).map_or("/tmp/img50.pgm", String::as_str);
    let target_h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);
    let bytes = std::fs::read(pgm).expect("read pgm");
    let (grey, w, h) = parse_pgm(&bytes).expect("parse pgm");
    let f = target_h as f32 / h as f32;
    let (out, wd, hd) = pix_scale_grey(&grey, w, h, f);
    println!("dim\t{wd}\t{hd}");
    for y in 0..hd {
        print!("r\t{y}");
        for &b in &out[y * wd..(y + 1) * wd] {
            print!("\t{b}");
        }
        println!();
    }
}
