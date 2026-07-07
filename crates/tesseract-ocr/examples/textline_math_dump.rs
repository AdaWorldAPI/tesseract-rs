//! Batch 3E wave-1 byte-parity dump: reads the shared fixture
//! `/tmp/textline_math_input.bin` (format documented in
//! `/tmp/gen_textline_fixtures.py`'s header and mirrored below) and runs the
//! Rust [`Stats`]/`textline` leaves, printing the exact same
//! `SECTIONn ...` lines as `/tmp/textline_math_oracle.cpp` so the two
//! outputs can be `diff`ed byte-for-byte.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example textline_math_dump > /tmp/rust_textline_math.txt
//! /tmp/textline_math_oracle > /tmp/cpp_textline_math.txt
//! diff /tmp/rust_textline_math.txt /tmp/cpp_textline_math.txt
//! ```
#![allow(
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    reason = "a dump CLI example writes to stdout by design; fixture sizes are small"
)]

use std::fs::File;
use std::io::Read;

use tesseract_ocr::stats::Stats;
use tesseract_ocr::textline::{
    compute_dropout_distances, compute_height_modes, compute_line_occupation,
    compute_occupation_threshold, fill_heights, DetLineFit, FCoord, ICoord,
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
    fn u8(&mut self) -> u8 {
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn f64(&mut self) -> f64 {
        let v = f64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
}

fn hex64(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}
fn hex32(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

fn main() {
    let mut buf = Vec::new();
    File::open("/tmp/textline_math_input.bin")
        .expect("run /tmp/gen_textline_fixtures.py first")
        .read_to_end(&mut buf)
        .unwrap();
    let mut c = Cursor::new(buf);

    // ---- Section 1: STATS ----
    {
        let rmin = c.i32();
        let rmax = c.i32();
        let mut s = Stats::new(rmin, rmax);
        let n = c.u32();
        for _ in 0..n {
            let v = c.i32();
            let cnt = c.i32();
            s.add(v, cnt);
        }
        println!("SECTION1 total={} mode={}", s.get_total(), s.mode());
        println!("SECTION1 mean_hex={}", hex64(s.mean()));
        println!("SECTION1 sd_hex={}", hex64(s.sd()));
        println!("SECTION1 median_hex={}", hex64(s.median()));
        println!(
            "SECTION1 min_bucket={} max_bucket={}",
            s.min_bucket(),
            s.max_bucket()
        );
        let nf = c.u32();
        for _ in 0..nf {
            let frac = c.f64();
            println!("SECTION1 ile[{frac:.4}]_hex={}", hex64(s.ile(frac)));
        }
    }

    // ---- Section 2: occupation/threshold/dropout ----
    {
        let line_count = c.u32();
        let mut occupation = vec![0i32; line_count as usize];
        for o in &mut occupation {
            *o = c.i32();
        }
        let low_window = c.i32();
        let high_window = c.i32();
        let occ_thresh = c.f64();
        let thresholds = compute_occupation_threshold(
            low_window,
            high_window,
            line_count as i32,
            &occupation,
            occ_thresh,
        );
        print!("SECTION2 thresholds=");
        for t in &thresholds {
            print!("{t},");
        }
        println!();
        let mut dropout = thresholds.clone();
        compute_dropout_distances(&occupation, &mut dropout, line_count as i32);
        print!("SECTION2 dropout=");
        for d in &dropout {
            print!("{d},");
        }
        println!();
    }

    // ---- Section 3: height modes ----
    {
        let rmin = c.i32();
        let rmax = c.i32();
        let mut heights = Stats::new(rmin, rmax);
        let n = c.u32();
        for _ in 0..n {
            let v = c.i32();
            let cnt = c.i32();
            heights.add(v, cnt);
        }
        let min_height = c.i32();
        let max_height = c.i32();
        let maxmodes = c.i32();
        let modes = compute_height_modes(&heights, min_height, max_height, maxmodes);
        print!("SECTION3 count={} modes=", modes.len());
        for m in &modes {
            print!("{m},");
        }
        println!();
    }

    // ---- Section 4: fill_heights ----
    {
        let n = c.u32();
        let mut boxes = Vec::with_capacity(n as usize);
        for _ in 0..n {
            boxes.push((c.i32(), c.i32(), c.i32(), c.i32()));
        }
        let gradient = c.f32();
        let parallel_c = c.f32();
        let min_height = c.i32();
        let max_height = c.i32();
        let min_blob_height_fraction = c.f32();
        let (heights, floating) = fill_heights(
            &boxes,
            gradient,
            parallel_c,
            min_height,
            max_height,
            min_blob_height_fraction,
        );
        println!(
            "SECTION4 heights_total={} floating_total={} heights_mode={}",
            heights.get_total(),
            floating.get_total(),
            heights.mode()
        );
        println!("SECTION4 heights_mean_hex={}", hex64(heights.mean()));
    }

    // ---- Section 5: compute_line_occupation ----
    {
        let n = c.u32();
        let mut blobs = Vec::with_capacity(n as usize);
        for _ in 0..n {
            blobs.push((c.i32(), c.i32(), c.i32(), c.i32()));
        }
        let gradient = c.f32();
        let min_y = c.i32();
        let max_y = c.i32();
        let (occupation, deltas) = compute_line_occupation(&blobs, gradient, min_y, max_y);
        print!("SECTION5 occupation=");
        for v in &occupation {
            print!("{v},");
        }
        println!();
        print!("SECTION5 deltas=");
        for v in &deltas {
            print!("{v},");
        }
        println!();
    }

    // ---- Section 6: DetLineFit ----
    {
        let n_configs = c.u32();
        for ci in 0..n_configs {
            let kind = c.u8();
            let n_pts = c.u32();
            let mut lms = DetLineFit::default();
            for _ in 0..n_pts {
                let x = c.i32();
                let y = c.i32();
                let hw = c.i32();
                if hw != 0 {
                    lms.add_with_halfwidth(ICoord::new(x, y), hw);
                } else {
                    lms.add(ICoord::new(x, y));
                }
            }
            match kind {
                0 => {
                    let (m, cc, error) = lms.fit_mc();
                    println!(
                        "SECTION6[{ci}] kind=fit m_hex={} c_hex={} error_hex={}",
                        hex32(m),
                        hex32(cc),
                        hex64(error)
                    );
                }
                1 => {
                    let dx = c.f32();
                    let dy = c.f32();
                    let mind = c.f64();
                    let maxd = c.f64();
                    let (line_pt, error) = lms.constrained_fit(FCoord::new(dx, dy), mind, maxd);
                    println!(
                        "SECTION6[{ci}] kind=constrained_dir pt=({},{}) error_hex={}",
                        line_pt.x,
                        line_pt.y,
                        hex64(error)
                    );
                }
                2 => {
                    let m = c.f64();
                    let (cc, error) = lms.constrained_fit_mc(m);
                    println!(
                        "SECTION6[{ci}] kind=constrained_m c_hex={} error_hex={}",
                        hex32(cc),
                        hex64(error)
                    );
                }
                _ => unreachable!("unknown DetLineFit fixture kind"),
            }
        }
    }
}
