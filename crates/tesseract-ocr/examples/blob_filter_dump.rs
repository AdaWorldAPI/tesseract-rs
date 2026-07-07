//! Batch 3F₂ leaf 2 byte-parity dump: reads a fixture produced by
//! `.claude/harvest/oracles/gen_blob_filter_fixtures.py` (format documented
//! there) and runs [`tesseract_ocr::filter_blobs`], printing the same
//! canonical `BLOBS`/`NOISE`/`SMALL`/`LARGE`/`SCALARS` lines as
//! `.claude/harvest/oracles/blob_filter_oracle.cpp` so the two outputs can
//! be `diff`ed byte-for-byte.
//!
//! Per `filter_blobs`'s module doc, list **membership** (not raw
//! traversal-order sequences) is the byte-parity contract: each of the four
//! output lists is dumped as its SORTED (ascending) set of original-fixture
//! indices, sidestepping the real Tesseract `BLOBNBOX_IT` splice-order
//! subtlety this port deliberately does not replicate (see the module doc
//! for the full justification — no PARITY-PIN is needed here since
//! `filter_noise_blobs` has no `sort`/`nth_element`, only per-element
//! threshold tests).
//!
//! ```sh
//! cargo run -p tesseract-ocr --example blob_filter_dump -- /tmp/blob_filter_input_seed1_clean.bin > /tmp/rust_bf_seed1_clean.txt
//! /tmp/blob_filter_oracle /tmp/blob_filter_input_seed1_clean.bin > /tmp/cpp_bf_seed1_clean.txt
//! diff /tmp/rust_bf_seed1_clean.txt /tmp/cpp_bf_seed1_clean.txt
//! ```
#![allow(
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    reason = "a dump CLI example writes to stdout by design; fixture sizes are small"
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use tesseract_ocr::{filter_blobs, ConnComp, ConnCompBox};

struct Cursor {
    buf: Vec<u8>,
    pos: usize,
}

impl Cursor {
    fn new(buf: Vec<u8>) -> Self {
        Cursor { buf, pos: 0 }
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
}

fn hex32(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

/// Map each output `(left,bottom,right,top)` tuple back to its original
/// fixture index for the canonical membership dump. `filter_blobs` never
/// merges or splits components (each output tuple corresponds to exactly
/// one input `ConnComp`), and the fixture generator produces pairwise
/// distinct boxes, so a box -> index lookup is unambiguous.
fn dump_indices(
    tag: &str,
    tuples: &[(i32, i32, i32, i32)],
    index_of: &HashMap<(i32, i32, i32, i32), usize>,
) {
    let mut idx: Vec<usize> = tuples
        .iter()
        .map(|t| {
            *index_of
                .get(t)
                .unwrap_or_else(|| panic!("output tuple {t:?} not found in original fixture"))
        })
        .collect();
    idx.sort_unstable();
    print!("{tag}\t{}", idx.len());
    for i in idx {
        print!("\t{i}");
    }
    println!();
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: blob_filter_dump <fixture.bin>");
    let mut buf = Vec::new();
    File::open(&path)
        .unwrap_or_else(|_| panic!("cannot open fixture {path}"))
        .read_to_end(&mut buf)
        .unwrap();
    let mut c = Cursor::new(buf);

    let n_blobs = c.u32();
    let mut components: Vec<ConnComp> = Vec::with_capacity(n_blobs as usize);
    let mut index_of: HashMap<(i32, i32, i32, i32), usize> =
        HashMap::with_capacity(n_blobs as usize);
    for i in 0..n_blobs {
        let left = c.i32();
        let bottom = c.i32();
        let right = c.i32();
        let top = c.i32();
        let pixel_count = c.i32();
        // ConnComp stores a raster-space box (x, y, w, h); filter_blobs's
        // box_tuple() maps it back to (left, bottom, right, top) via
        // (x, y, x+w, y+h) -- so constructing bb this way round-trips
        // exactly through box_tuple(), keeping the index lookup exact.
        let comp = ConnComp {
            bb: ConnCompBox {
                x: left,
                y: bottom,
                w: right - left,
                h: top - bottom,
            },
            pixel_count,
        };
        let tuple = (left, bottom, right, top);
        let prev = index_of.insert(tuple, i as usize);
        assert!(
            prev.is_none(),
            "fixture must have pairwise-distinct (left,bottom,right,top) boxes"
        );
        components.push(comp);
    }

    println!("FIXTURE n_blobs={n_blobs}");

    let out = filter_blobs(&components);

    dump_indices("BLOBS", &out.blobs, &index_of);
    dump_indices("NOISE", &out.noise, &index_of);
    dump_indices("SMALL", &out.small, &index_of);
    dump_indices("LARGE", &out.large, &index_of);
    println!(
        "SCALARS line_size_hex={} line_spacing_hex={} max_blob_size_hex={}",
        hex32(out.line_size),
        hex32(out.line_spacing),
        hex32(out.max_blob_size)
    );
}
