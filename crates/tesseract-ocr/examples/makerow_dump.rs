//! Batch 3E wave-2 byte-parity dump: reads a fixture produced by
//! `/tmp/gen_makerow_fixtures.py` (format documented there) and runs the
//! full `makerow.cpp` row-assignment + cleanup chain stage by stage,
//! printing the exact same `STAGEn ...` lines as `/tmp/makerow_oracle.cpp`
//! so the two outputs can be `diff`ed byte-for-byte.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example makerow_dump -- /tmp/makerow_input_seed1_clean.bin > /tmp/rust_makerow_seed1_clean.txt
//! /tmp/makerow_oracle /tmp/makerow_input_seed1_clean.bin > /tmp/cpp_makerow_seed1_clean.txt
//! diff /tmp/rust_makerow_seed1_clean.txt /tmp/cpp_makerow_seed1_clean.txt
//! ```
#![allow(
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    reason = "a dump CLI example writes to stdout by design; fixture sizes are small"
)]

use std::fs::File;
use std::io::Read;

use tesseract_ocr::{
    assign_blobs_to_rows, compute_page_skew, delete_non_dropout_rows, expand_rows,
    fit_parallel_rows, make_initial_textrows, ToBlockCtx, ToRow,
};

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
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
}

fn hex32(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

fn dump_row_blobs(row: &ToRow) -> String {
    row.blobs
        .iter()
        .map(|(l, b, r, t)| format!("{l},{b},{r},{t}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: makerow_dump <fixture.bin>");
    let mut buf = Vec::new();
    File::open(&path)
        .unwrap_or_else(|_| panic!("cannot open fixture {path}"))
        .read_to_end(&mut buf)
        .unwrap();
    let mut c = Cursor::new(buf);

    let n_blobs = c.u32();
    let mut blobs = Vec::with_capacity(n_blobs as usize);
    for _ in 0..n_blobs {
        blobs.push((c.i32(), c.i32(), c.i32(), c.i32()));
    }
    let line_spacing = c.f32();
    let line_size = c.f32();
    let max_blob_size = c.f32();
    let block_left = c.i32();

    println!("FIXTURE n_blobs={n_blobs}");

    let mut block = ToBlockCtx {
        blobs,
        block_left,
        line_spacing,
        line_size,
        max_blob_size,
        ..Default::default()
    };

    // ---- Stage 1: make_initial_textrows (assign pass 0 + fit_lms_line) ----
    make_initial_textrows(&mut block);
    println!("STAGE1_ROWS {}", block.rows.len());
    for (i, row) in block.rows.iter().enumerate() {
        println!(
            "STAGE1_ROW[{i}] min_hex={} max_hex={} m_hex={} c_hex={} err_hex={} blobs={}",
            hex32(row.min_y()),
            hex32(row.max_y()),
            hex32(row.line_m()),
            hex32(row.line_c()),
            hex32(row.line_error()),
            dump_row_blobs(row)
        );
    }

    // ---- Stage 2: compute_page_skew (single block) ----
    let row_slices: Vec<&[ToRow]> = vec![block.rows.as_slice()];
    let (page_m, page_err) = compute_page_skew(&row_slices);
    println!(
        "STAGE2_SKEW page_m_hex={} page_err_hex={}",
        hex32(page_m),
        hex32(page_err)
    );

    // ---- Stage 3: fit_parallel_rows ----
    fit_parallel_rows(&mut block, page_m);
    println!("STAGE3_ROWS {}", block.rows.len());
    for (i, row) in block.rows.iter().enumerate() {
        println!(
            "STAGE3_ROW[{i}] parc_hex={} intercept_hex={} believ_hex={} nblobs={}",
            hex32(row.parallel_c()),
            hex32(row.intercept()),
            hex32(row.believability()),
            row.blobs.len()
        );
    }

    // ---- Stage 4: delete_non_dropout_rows ----
    delete_non_dropout_rows(&mut block, page_m);
    println!(
        "STAGE4_ROWS {} pool={}",
        block.rows.len(),
        block.blobs.len()
    );
    for (i, row) in block.rows.iter().enumerate() {
        println!(
            "STAGE4_ROW[{i}] min_hex={} max_hex={} intercept_hex={}",
            hex32(row.min_y()),
            hex32(row.max_y()),
            hex32(row.intercept())
        );
    }

    // ---- Stage 5: expand_rows ----
    expand_rows(&mut block, page_m);
    println!(
        "STAGE5_ROWS {} line_spacing_hex={} line_size_hex={} max_blob_size_hex={} baseline_offset_hex={}",
        block.rows.len(),
        hex32(block.line_spacing),
        hex32(block.line_size),
        hex32(block.max_blob_size),
        hex32(block.baseline_offset)
    );
    for (i, row) in block.rows.iter().enumerate() {
        println!(
            "STAGE5_ROW[{i}] min_hex={} max_hex={}",
            hex32(row.min_y()),
            hex32(row.max_y())
        );
    }

    // ---- Stage 6: reconsolidate + the three assign_blobs_to_rows passes
    // (the rest of cleanup_rows_making, called via the sub-functions so the
    // dump can still snapshot stage 5 in between). ----
    {
        let pool = &mut block.blobs;
        for row in &mut block.rows {
            pool.append(&mut row.blobs);
        }
    }
    assign_blobs_to_rows(&mut block, Some(page_m), false, false); // pass 1
    assign_blobs_to_rows(&mut block, Some(page_m), true, true); // pass 2
    assign_blobs_to_rows(&mut block, Some(page_m), false, false); // pass 3

    let total_assigned: usize = block.rows.iter().map(|r| r.blobs.len()).sum();
    println!(
        "STAGE6_ROWS {} pool={} total_assigned={}",
        block.rows.len(),
        block.blobs.len(),
        total_assigned
    );
    for (i, row) in block.rows.iter().enumerate() {
        println!("STAGE6_ROW[{i}] blobs={}", dump_row_blobs(row));
    }
}
