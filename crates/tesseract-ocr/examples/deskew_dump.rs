//! Byte-parity dump CLI for the deskew wave's three leaves shipped in
//! `tesseract_ocr::deskew` (D1 `rotate90_grey`, D2
//! `find_differential_square_sum`, D5 `rotate_am_gray`/`rotate_am_gray_corner`).
//! Output mirrors `.claude/harvest/oracles/skew_oracle.cpp`'s own
//! `print_f32`/`dump_pix` helpers exactly, so it is directly `diff`-able
//! against that oracle's stdout.
//!
//! ## Arms
//! ```text
//!   deskew_dump rot90           <pgm> <direction>
//!   deskew_dump dss             <pgm> <thresh> <angle_deg>
//!   deskew_dump rotamgray       <pgm> <angle_deg> <grayval>
//!   deskew_dump rotamgraycorner <pgm> <angle_deg> <grayval>
//! ```
//!
//! ## `rot90` — oracle arm added, diff is green
//! `skew_oracle.cpp` gained a matching `rot90` arm calling the real
//! `pixRotate90` through its own `dump_pix` helper. **Verified byte-identical
//! both directions on a full 512×720 page.**
//!
//! ## `dss` — a real scope seam, read before trusting a diff
//! The oracle's `dss` arm does MORE than this crate's `find_differential_
//! square_sum`: it binarizes, applies a **single vertical shear**
//! (`pixVShearCorner` / `pixVShearCenter`, pivot-selected — leaf D3, NOT
//! shipped here), THEN scores the sheared result. This arm only binarizes +
//! scores, so it is comparable to the oracle ONLY where that shear is a
//! no-op: `angle_deg = 0.0`, at which **both pivots return the identical sum
//! (verified: 258022 on `page_01`)**, confirming the shear contributes
//! nothing there. **Verified byte-identical at angle 0.**
//!
//! To exercise D2 at other effective angles, pre-rotate the INPUT PGM
//! externally and pass `angle_deg = 0.0` to BOTH sides — `deg` is pure
//! pass-through metadata here, never a rotation trigger. Do not read a `dss`
//! diff at nonzero `angle_deg` as D2 parity evidence until D3 lands and this
//! arm is extended to call it.
//!
//! (The oracle originally used `pixRotateShearCenter` here — a COMPOSED
//! two/three-shear rotation, not the single shear the sweep performs. A
//! correct D3 would have FAILED that arm while a wrong one could have
//! passed; corrected per codex P1 on PR #58.)
//!
//! ## `rotamgray` / `rotamgraycorner` — the degrees→radians seam
//! `rotate_am_gray`/`rotate_am_gray_corner` take RADIANS (matching
//! `pixRotateAMGray`'s real signature). The oracle's CLI accepts DEGREES for
//! convenience and converts internally with its OWN formula — `deg *
//! 3.14159265358979323846f / 180.0f`, entirely in **f32** — which is a
//! DIFFERENT literal/precision than `skew.c`'s own `deg2rad`
//! (`(3.1415926535/180.0)` computed in f64 then narrowed once). This CLI
//! transcribes the ORACLE's conversion verbatim (not `skew.c`'s, and not
//! `f32::to_radians()`, which is neither) so the SAME `<angle_deg>` CLI
//! argument reaches `rotate_am_gray`/`rotate_am_gray_corner` as the
//! identical radians bit pattern on both sides — a mismatched conversion
//! here would look like a D5 kernel bug but would actually be conversion
//! drift.
//!
//! ## Floats
//! Every float is dumped as `<label>\t0x<8-hex-digit bits>\t<decimal>`,
//! matching the oracle's `print_f32`. The hex bits are the parity subject.
//! The decimal column is `%.9g`-compatible (see [`format_g9`]) so the two
//! sides agree character for character and the diff can be a plain whole-file
//! `diff` — the oracle's original "never compare the decimal" caveat is no
//! longer needed, and a mismatched formatter would otherwise have flagged
//! every float line as a parity failure.
//!
//! ```sh
//! g++ -std=c++17 .claude/harvest/oracles/skew_oracle.cpp \
//!     -I/usr/include/leptonica -lleptonica -o /tmp/skew_oracle
//!
//! /tmp/skew_oracle rotamgray in.pgm 2.5 255 > /tmp/o_ramg.tsv
//! cargo run -q -p tesseract-ocr --example deskew_dump -- rotamgray in.pgm 2.5 255 > /tmp/r_ramg.tsv
//! diff /tmp/o_ramg.tsv /tmp/r_ramg.tsv
//!
//! /tmp/skew_oracle dss in.pgm 128 0 > /tmp/o_dss.tsv
//! cargo run -q -p tesseract-ocr --example deskew_dump -- dss in.pgm 128 0 > /tmp/r_dss.tsv
//! diff /tmp/o_dss.tsv /tmp/r_dss.tsv
//! ```
#![allow(clippy::print_stdout, reason = "dump CLI")]

use std::path::Path;

use tesseract_ocr::deskew::{
    find_differential_square_sum, rotate90_grey, rotate_am_gray, rotate_am_gray_corner,
};
use tesseract_ocr::image_input::parse_pgm;

/// The oracle's own degrees→radians conversion (`skew_oracle.cpp`'s
/// `rotamgray` / `rotamgraycorner` arms): `deg * 3.14159265358979323846f /
/// 180.0f`, computed **entirely in f32**, so a CLI-supplied degree value
/// reaches the leaf under test as the exact same radians bit pattern the
/// oracle produced for the same arm.
///
/// Two things this is deliberately NOT:
///
/// * **Not `skew.c`'s `deg2rad`** — that one is an f64 division narrowed to
///   f32 on assignment (manifest audit §1), a different value. It belongs to
///   the sweep, not to this harness.
/// * **Not `f32::to_radians()`** — that computes `self * (PI / 180.0)`,
///   folding the division into a constant BEFORE the multiply. Floating-point
///   multiplication is not associative, so `deg * (PI/180)` and
///   `(deg * PI) / 180` can differ in the last bit. The operation ORDER is
///   what matters here, not the constant.
///
/// `PI` itself is written as `std::f32::consts::PI` rather than the C++
/// literal only because clippy's `approx_constant` rejects the literal — and
/// the substitution is provably free: `3.14159265358979323846` rounded to f32
/// and `f32::consts::PI` are the SAME bit pattern, `0x40490fdb` (verified,
/// not assumed). The C++ side's `3.14159265358979323846f` is likewise a float
/// literal, so it rounds to that same value at compile time.
fn oracle_deg_to_rad(deg: f32) -> f32 {
    deg * core::f32::consts::PI / 180.0_f32
}

/// Mirrors the oracle's `print_f32`: `"<label>\t0x<8-hex-digit bits>\t<decimal>"`.
/// The hex bits are the parity subject; the decimal is for a human and is
/// never compared.
fn print_f32(label: &str, v: f32) {
    println!(
        "{label}\t0x{:08x}\t{}",
        v.to_bits(),
        format_g9(f64::from(v))
    );
}

/// C's `printf("%.9g", x)` — 9 significant digits, `%e` or `%f` whichever is
/// shorter, trailing zeros and a trailing `.` stripped.
///
/// Rust has no `%g`, and the obvious substitutes are both wrong for a diff:
/// `{:.9e}` always uses exponent form (`2.580220000e5` where C prints
/// `258022`), and `{}` uses Rust's shortest-round-trip algorithm, which is a
/// different rule again. Since the harness diffs the oracle's stdout against
/// this one LINE BY LINE, a formatting mismatch reads as a parity failure on
/// every float line — noise that buries the real signal. Reproducing `%g`
/// exactly is cheaper than teaching every reader to ignore a column.
///
/// The hex bits remain the actual parity subject; this is the human-readable
/// companion, and it now agrees with the oracle character for character.
fn format_g9(x: f64) -> String {
    const P: i32 = 9;
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        // C prints "inf"/"-inf"/"nan"; Rust's Display agrees on all three.
        return format!("{x}");
    }
    // C chooses the form from the exponent of the value AFTER rounding to P
    // significant digits — round first, or a value like 9.999e-5 picks the
    // wrong branch.
    let exp = {
        let e = x.abs().log10().floor() as i32;
        let scaled = x.abs() / 10f64.powi(e);
        // Rounding to P sig digits can carry into the next decade (9.9995 -> 10.00).
        if (scaled * 10f64.powi(P - 1)).round() >= 10f64.powi(P) {
            e + 1
        } else {
            e
        }
    };
    // C's rule verbatim: exponent form when `exp < -4 || exp >= P`.
    let mut s = if !(-4..P).contains(&exp) {
        let m = format!("{:.*e}", (P - 1) as usize, x);
        // Rust renders the exponent as `e5` / `e-5`; C as `e+05` / `e-05`.
        let (mant, e) = m.split_once('e').unwrap_or((m.as_str(), "0"));
        let ev: i32 = e.parse().unwrap_or(0);
        format!(
            "{}e{}{:02}",
            strip(mant),
            if ev < 0 { '-' } else { '+' },
            ev.abs()
        )
    } else {
        strip(&format!("{:.*}", (P - 1 - exp).max(0) as usize, x))
    };
    if s.is_empty() {
        s.push('0');
    }
    s
}

/// Drop trailing zeros in a fractional part, then a bare trailing `.`.
/// Integral renderings (no `.`) are returned untouched.
fn strip(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Mirrors the oracle's `dump_pix`: dimension header lines
/// (`<tag>_w`/`<tag>_h`/`<tag>_d`), then one `"<tag>\t<idx>\t<value>"` line
/// per row-major pixel. `<tag>_d` is always `8` here (this CLI only ever
/// handles 8bpp grey buffers).
fn dump_pix(tag: &str, buf: &[u8], w: usize, h: usize) {
    println!("{tag}_w\t{w}");
    println!("{tag}_h\t{h}");
    println!("{tag}_d\t8");
    for (idx, &v) in buf.iter().enumerate() {
        println!("{tag}\t{idx}\t{v}");
    }
}

/// `pixThresholdToBinary` (fixed threshold, ON where `grey < thresh`) — this
/// is explicitly SKIP'd as a `deskew.rs` leaf per the manifest (v1 scope),
/// but the `dss` arm still needs SOME way to turn a grey PGM fixture into
/// the binary buffer `find_differential_square_sum` expects, so it lives
/// here as harness-only plumbing (matching `skew_oracle.cpp`'s own
/// `to_binary`). Leptonica's ON=1 becomes THIS CRATE's ON=0 at this exact
/// boundary — the conversion the module doc for `deskew.rs` insists on
/// making explicit rather than silently assuming.
fn threshold_to_binary(grey: &[u8], thresh: i32) -> Vec<u8> {
    grey.iter()
        .map(|&g| if i32::from(g) < thresh { 0u8 } else { 255u8 })
        .collect()
}

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  deskew_dump rot90           <pgm> <direction>");
    eprintln!("  deskew_dump dss             <pgm> <thresh> <angle_deg>");
    eprintln!("  deskew_dump rotamgray       <pgm> <angle_deg> <grayval>");
    eprintln!("  deskew_dump rotamgraycorner <pgm> <angle_deg> <grayval>");
    std::process::exit(2);
}

fn read_grey(path: &str) -> (Vec<u8>, usize, usize) {
    let bytes = std::fs::read(Path::new(path)).expect("read pgm");
    parse_pgm(&bytes).expect("parse pgm")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }
    let arm = args[1].as_str();

    match arm {
        "rot90" => {
            if args.len() < 4 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let direction: i32 = args[3].parse().expect("direction");
            let (out, ow, oh) = rotate90_grey(&grey, w, h, direction);
            dump_pix("rot90", &out, ow, oh);
        }
        "dss" => {
            if args.len() < 5 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let thresh: i32 = args[3].parse().expect("thresh");
            let deg: f32 = args[4].parse().expect("angle_deg");
            let binary = threshold_to_binary(&grey, thresh);
            let sum = find_differential_square_sum(&binary, w, h);
            println!("rc\t0");
            print_f32("deg", deg);
            print_f32("sum", sum);
        }
        "rotamgray" => {
            if args.len() < 5 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let deg: f32 = args[3].parse().expect("angle_deg");
            let grayval: u8 = args[4].parse().expect("grayval");
            let rad = oracle_deg_to_rad(deg);
            let out = rotate_am_gray(&grey, w, h, rad, grayval);
            print_f32("deg", deg);
            dump_pix("rot", &out, w, h);
        }
        "rotamgraycorner" => {
            if args.len() < 5 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let deg: f32 = args[3].parse().expect("angle_deg");
            let grayval: u8 = args[4].parse().expect("grayval");
            let rad = oracle_deg_to_rad(deg);
            let out = rotate_am_gray_corner(&grey, w, h, rad, grayval);
            print_f32("deg", deg);
            dump_pix("rot", &out, w, h);
        }
        _ => usage(),
    }
}
