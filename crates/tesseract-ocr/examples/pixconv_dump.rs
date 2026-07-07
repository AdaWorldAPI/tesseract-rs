//! `pixConvertRGBToGray` byte-parity surface: generate the SESSION-STANDARD
//! synthetic RGB grid, run `rgb_to_gray`, dump the grey result AND write the
//! RGB input to `/tmp/pixconv_input.bin` for a libtesseract/leptonica oracle
//! (`pixConvertRGBToGray` via the REAL leptonica `pixconv.c`) to consume.
//!
//! Usage: `pixconv_dump <w> <h> [rwt gwt bwt]` (weights default to 0,0,0 =
//! the default perceptual weights / `pixConvertRGBToLuminance`).
//!
//! Input binary format (`/tmp/pixconv_input.bin`): `i32 w`, `i32 h`, then
//! `w*h*3` RGB bytes, row-major (R,G,B per pixel).
#![allow(clippy::print_stdout, reason = "dump CLI")]
use std::io::Write as _;
use tesseract_ocr::image_input::rgb_to_gray;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);
    let rwt: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let gwt: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let bwt: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    // SESSION-STANDARD synthetic RGB.
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 37 + y * 11) % 256) as u8;
            let g = ((x * 7 + y * 13) % 256) as u8;
            let b = (((x * 3) ^ (y * 5)) % 256) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }

    // Write the shared input for the C++ oracle.
    let mut buf = Vec::with_capacity(8 + rgb.len());
    buf.extend_from_slice(&(w as i32).to_le_bytes());
    buf.extend_from_slice(&(h as i32).to_le_bytes());
    buf.extend_from_slice(&rgb);
    std::fs::File::create("/tmp/pixconv_input.bin")
        .and_then(|mut f| f.write_all(&buf))
        .expect("write pixconv_input.bin");

    let grey = rgb_to_gray(&rgb, w, h, rwt, gwt, bwt);
    println!("dim\t{w}\t{h}");
    for y in 0..h {
        print!("r\t{y}");
        for &g in &grey[y * w..(y + 1) * w] {
            print!("\t{g}");
        }
        println!();
    }
}
