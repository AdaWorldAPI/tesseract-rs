//! A6a byte-parity surface: build a synthetic 8-bit grey image, run
//! [`from_grey_pix`] (the `NetworkIO::FromPix` transcode), dump the int8 grid.
//! The oracle (`/tmp/frompix_oracle.cpp`) builds the SAME image (read from the
//! shared `.bin`) into a leptonica `Pix` and runs the REAL `NetworkIO::FromPix`.
//!
//! ```sh
//! cargo run -p tesseract-recognizer --example from_pix_dump -- /tmp/frompix_input.bin 24 > /tmp/rust_frompix.tsv
//! /tmp/frompix_oracle /tmp/frompix_input.bin > /tmp/oracle_frompix.tsv
//! diff /tmp/oracle_frompix.tsv /tmp/rust_frompix.tsv   # byte-identical => A6a green
//! ```
#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::io::Write;

use tesseract_recognizer::{from_grey_pix, FlexDim, TRand};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bin_path = args.get(1).map_or("/tmp/frompix_input.bin", String::as_str);
    let width: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let height: usize = 36; // eng's input height.

    // Synthetic grey image, deterministic per (y,x) — a gradient with structure
    // so ComputeBlackWhite finds real local extrema on the middle row.
    let grey: Vec<u8> = (0..height * width)
        .map(|i| {
            let (y, x) = (i / width, i % width);
            (((x * 37 + y * 11) ^ (x * y)) % 256) as u8
        })
        .collect();

    // Write the shared .bin: i32 width, i32 height, then row-major u8 pixels.
    let mut buf = Vec::new();
    buf.extend_from_slice(&(width as i32).to_le_bytes());
    buf.extend_from_slice(&(height as i32).to_le_bytes());
    buf.extend_from_slice(&grey);
    std::fs::File::create(bin_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write frompix_input.bin");

    let mut rng = TRand::default();
    rng.set_seed(1);
    // eng shape: height=36, width=0 (variable).
    let io = from_grey_pix(&grey, width, height, 36, 0, &mut rng);

    println!(
        "shape\t{}\t{}\t{}\t{}",
        io.stride_map().size(FlexDim::Height),
        io.stride_map().size(FlexDim::Width),
        io.num_features(),
        io.width()
    );
    for t in 0..io.width() {
        print!("i\t{t}");
        for &v in io.i(t) {
            print!("\t{v}");
        }
        println!();
    }
}
