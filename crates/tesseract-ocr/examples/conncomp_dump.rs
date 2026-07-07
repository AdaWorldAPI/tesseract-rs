//! Batch-3B byte-parity surface: connected-component bounding boxes. Batch
//! 3F₂ leaf 1 adds an `--areas` mode: per-component ink pixel count
//! (`conn_comp_areas`, the `BLOBNBOX::enclosed_area()` source).
//!
//! Modes:
//! - default: self-generates the session-standard synthetic grey image
//!   `((x*37 + y*11) ^ (x*y)) % 256`, Otsu-binarizes it (full rect), writes
//!   `/tmp/conncomp_input.bin` for the oracle, and dumps the boxes.
//! - `--pgm <path>`: reads a real PGM, Otsu-binarizes, connectivity 8.
//! - `--areas`: combine with either of the above to dump `conn_comp_areas`
//!   (box + pixel_count) instead of plain boxes; pairs with the oracle's
//!   own `--areas` flag.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example conncomp_dump -- 24 36 4
//! /tmp/conncomp_oracle /tmp/conncomp_input.bin
//! # diff the two outputs => byte-parity
//!
//! cargo run -p tesseract-ocr --example conncomp_dump -- --areas 24 36 4
//! /tmp/conncomp_oracle --areas /tmp/conncomp_input.bin
//! # diff the two outputs => byte-parity (boxes + pixel counts)
//! ```
#![allow(clippy::print_stdout, reason = "dump CLI")]

use tesseract_ocr::{conn_comp_areas, conn_comp_bb, otsu_threshold_gray, threshold_rect_to_binary};

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

fn write_oracle_input(binary: &[u8], w: usize, h: usize, connectivity: u32) {
    // Write the oracle's input file: i32 w, i32 h, i32 connectivity, then
    // w*h raw bytes.
    let mut buf = Vec::with_capacity(12 + binary.len());
    buf.extend_from_slice(&i32::try_from(w).expect("w fits i32").to_le_bytes());
    buf.extend_from_slice(&i32::try_from(h).expect("h fits i32").to_le_bytes());
    buf.extend_from_slice(&(connectivity as i32).to_le_bytes());
    buf.extend_from_slice(binary);
    std::fs::write("/tmp/conncomp_input.bin", &buf).expect("write oracle input");
}

fn dump(binary: &[u8], w: usize, h: usize, connectivity: u32) {
    write_oracle_input(binary, w, h, connectivity);

    let boxes = conn_comp_bb(binary, w, h, connectivity);
    println!("n\t{}", boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        println!("bb\t{i}\t{}\t{}\t{}\t{}", b.x, b.y, b.w, b.h);
    }
}

/// Batch 3F₂ leaf 1: dump `conn_comp_areas` (box + ink pixel count) instead
/// of plain boxes, matching the oracle's `--areas` `cc` lines.
fn dump_areas(binary: &[u8], w: usize, h: usize, connectivity: u32) {
    write_oracle_input(binary, w, h, connectivity);

    let comps = conn_comp_areas(binary, w, h, connectivity);
    println!("n\t{}", comps.len());
    for (i, c) in comps.iter().enumerate() {
        println!(
            "cc\t{i}\t{}\t{}\t{}\t{}\t{}",
            c.bb.x, c.bb.y, c.bb.w, c.bb.h, c.pixel_count
        );
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let areas_mode = if let Some(pos) = args.iter().position(|a| a == "--areas") {
        args.remove(pos);
        true
    } else {
        false
    };

    if args.first().map(String::as_str) == Some("--pgm") {
        let path = args.get(1).map_or("/tmp/line36.pgm", String::as_str);
        let bytes = std::fs::read(path).expect("read pgm");
        let (grey, w, h) = tesseract_ocr::parse_pgm(&bytes).expect("parse pgm");
        let binary = binarize(&grey, w, h);
        if areas_mode {
            dump_areas(&binary, w, h, 8);
        } else {
            dump(&binary, w, h, 8);
        }
        return;
    }

    let w: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(24);
    let h: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(36);
    let connectivity: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let grey = synthetic_grey(w, h);
    let binary = binarize(&grey, w, h);
    if areas_mode {
        dump_areas(&binary, w, h, connectivity);
    } else {
        dump(&binary, w, h, connectivity);
    }
}
