//! Otsu-threshold byte-parity surface: generate the session-standard
//! synthetic grey image, run [`otsu_threshold_gray`] + [`threshold_rect_to_binary`]
//! over the full rect, dump the threshold/hi_value decision and the binary
//! rows. An oracle can reproduce the same input via
//! `/tmp/otsu_input.bin` (`i32 w`, `i32 h`, `w*h` raw grey bytes) and the
//! REAL `tesseract::OtsuThreshold` + `ImageThresholder::ThresholdRectToPix`.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example otsu_dump -- 24 36 > /tmp/rust_otsu_24x36.tsv
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use tesseract_ocr::threshold::{otsu_threshold_gray, threshold_rect_to_binary};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(36);

    // Session-standard synthetic grey generator: ((x*37+y*11)^(x*y)) % 256.
    let mut grey = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 37 + y * 11) ^ (x * y)) % 256;
            grey[y * w + x] = v as u8;
        }
    }

    // Bank the input for an external oracle: i32 w, i32 h, then w*h raw bytes.
    let mut input_blob = Vec::with_capacity(8 + w * h);
    input_blob.extend_from_slice(&(w as i32).to_le_bytes());
    input_blob.extend_from_slice(&(h as i32).to_le_bytes());
    input_blob.extend_from_slice(&grey);
    if let Err(err) = std::fs::write("/tmp/otsu_input.bin", &input_blob) {
        eprintln!("warning: could not write /tmp/otsu_input.bin: {err}");
    }

    let otsu = otsu_threshold_gray(&grey, w, 0, 0, w, h);
    let binary = threshold_rect_to_binary(&grey, w, 0, 0, w, h, otsu);

    println!("otsu\t{}\t{}", otsu.threshold, otsu.hi_value);
    for y in 0..h {
        print!("b\t{y}");
        for &v in &binary[y * w..(y + 1) * w] {
            print!("\t{v}");
        }
        println!();
    }
}
