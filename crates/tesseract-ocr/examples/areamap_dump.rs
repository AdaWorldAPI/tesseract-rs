//! pixScaleAreaMap byte-parity surface: read a PGM, area-map downscale to
//! target height via scale_gray_area_map, dump. Oracle calls the REAL
//! leptonica pixScaleAreaMap(pix, f, f). Use a NON-power-of-2 f (else
//! pixScaleAreaMap routes to pixScaleAreaMap2, a different kernel).
#![allow(clippy::print_stdout, reason = "dump CLI")]
use tesseract_ocr::image_input::{parse_pgm, scale_gray_area_map};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pgm = args.get(1).map_or("/tmp/a60.pgm", String::as_str);
    let target_h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);
    let bytes = std::fs::read(pgm).expect("read pgm");
    let (grey, w, h) = parse_pgm(&bytes).expect("parse pgm");
    let f = target_h as f32 / h as f32;
    let wd = ((f * w as f32) + 0.5) as usize;
    let hd = ((f * h as f32) + 0.5) as usize;
    let out = scale_gray_area_map(&grey, w, h, wd, hd);
    println!("dim\t{wd}\t{hd}");
    for y in 0..hd {
        print!("r\t{y}");
        for &b in &out[y * wd..(y + 1) * wd] {
            print!("\t{b}");
        }
        println!();
    }
}
