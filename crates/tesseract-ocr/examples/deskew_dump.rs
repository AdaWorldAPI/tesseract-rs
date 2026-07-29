//! Byte-parity dump CLI for the deskew wave's leaves shipped in
//! `tesseract_ocr::deskew` (D1 `rotate90_grey`, D2
//! `find_differential_square_sum`, D3 `v_shear_corner`/`v_shear_center`, D4
//! `find_skew_sweep_and_search_score_pivot`, D5
//! `rotate_am_gray`/`rotate_am_gray_corner`, D6 `deskew_general`, D7
//! `deskew_both`). Output
//! mirrors `.claude/harvest/oracles/skew_oracle.cpp`'s own `print_f32`/
//! `dump_pix` helpers exactly, so it is directly `diff`-able against that
//! oracle's stdout.
//!
//! ## Arms
//! ```text
//!   deskew_dump rot90           <pgm> <direction>
//!   deskew_dump dss             <pgm> <thresh> <angle_deg> <pivot 1=corner|2=center>
//!   deskew_dump vshear          <pgm> <thresh> <angle_deg> <pivot 1=corner|2=center>
//!   deskew_dump sweep           <pgm> <thresh> <redsweep> <redsearch> <sweepcenter>
//!                               <sweeprange> <sweepdelta> <minbsdelta> <pivot 1=corner|2=center>
//!   deskew_dump rotamgray       <pgm> <angle_deg> <grayval>
//!   deskew_dump rotamgraycorner <pgm> <angle_deg> <grayval>
//!   deskew_dump deskew          <pgm> <redsearch>
//!   deskew_dump deskewboth      <pgm> <redsearch>
//! ```
//!
//! ## `rot90` — oracle arm added, diff is green
//! `skew_oracle.cpp` gained a matching `rot90` arm calling the real
//! `pixRotate90` through its own `dump_pix` helper. **Verified byte-identical
//! both directions on a full 512×720 page.**
//!
//! ## `dss` — now the REAL D2+D3 end-to-end falsifier, at any angle
//! D3 (`v_shear_corner`/`v_shear_center`) is shipped, so this arm now
//! reproduces the oracle's `dss` arm EXACTLY: binarize, apply the SINGLE
//! vertical shear the sweep actually performs (`v_shear_corner`/
//! `v_shear_center`, pivot-selected, using [`skew_deg_to_rad`] —
//! `skew.c`'s OWN `deg2rad`, NOT [`oracle_deg_to_rad`]), THEN score with
//! `find_differential_square_sum`. Previously this arm only binarized +
//! scored (comparable to the oracle ONLY at `angle_deg = 0.0`, where the
//! shear is a no-op) — that limitation is closed now that D3 exists, and the
//! signature grew a REQUIRED 4th `<pivot>` argument to match the oracle's own
//! `dss` signature exactly (`<pgm> <thresh> <angle_deg> <pivot
//! 1=corner|2=center>`).
//!
//! ## `vshear` — D3's raw sheared-buffer dump (diagnostic; no oracle arm yet)
//! Dumps the FULL sheared 1bpp buffer via [`dump_pix`], polarity-flipped to
//! leptonica's native convention (`1 = ON`, vs this crate's `0 = ON` — see
//! [`threshold_to_binary`]'s own boundary-flip precedent). **No oracle-side
//! counterpart exists for this arm yet** (`skew_oracle.cpp` has no matching
//! per-pixel `vshear` dump — only the scalar-scoring `dss` arm exercises
//! `pixVShearCorner`/`pixVShearCenter`), so there is nothing to `diff` this
//! against mechanically today. It exists for manual/visual inspection and as
//! the natural extension point for a future oracle arm. **The real
//! byte-parity evidence for D3 is the updated `dss` arm above** — it already
//! reproduces the oracle's OWN `pixVShearCorner`/`pixVShearCenter` +
//! `pixFindDifferentialSquareSum` composition end-to-end, at any angle, and
//! IS diffable today.
//!
//! ## `sweep` — the D4 detector, matching the oracle's `sweep` arm exactly
//! Same positional signature as `skew_oracle.cpp`'s `sweep` arm (which itself
//! calls the REAL `pixFindSkewSweepAndSearchScorePivot`, not the unrelated
//! `pixFindSkewSweep`). Always prints `rc`/`angle`/`conf`/`endscore` in that
//! order, even on `rc = 1` (matching the C's own contract of initializing all
//! three outputs to `0.0` before any validation) — the Rust
//! [`find_skew_sweep_and_search_score_pivot`] returning `None` maps to
//! `rc = 1` with all three fields printed as `0.0`, never an omitted line.
//!
//! (The oracle originally used `pixRotateShearCenter` in its `dss` arm — a
//! COMPOSED two/three-shear rotation, not the single shear the sweep
//! performs. A correct D3 would have FAILED that arm while a wrong one could
//! have passed; corrected per codex P1 on PR #58.)
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
//! ## `deskew` — D6, matching the oracle's own bare `dump_pix` output
//! `skew_oracle.cpp`'s `deskew` arm calls `pixDeskew`, which discards
//! angle/conf via `NULL` — so its ENTIRE stdout is one `dump_pix("deskew",
//! pixd)` call, no `rc` line, no angle/conf lines. This arm reproduces that
//! exact shape by calling [`tesseract_ocr::deskew::deskew_general`] with
//! `pixDeskew`'s own literal default arguments (`redsweep=0, sweeprange=0.0,
//! sweepdelta=0.0, thresh=0`; only `redsearch` comes from argv) and dumping
//! ONLY the resulting buffer — `deskew_general`'s own richer `DeskewResult`
//! (which DOES carry angle/conf, unlike `pixDeskew`) is available to any
//! OTHER caller, but this CLI arm deliberately narrows back down to what the
//! oracle can actually produce, so the two sides stay byte-for-byte
//! diffable. On `None` (invalid `redsearch`), this prints nothing to stdout
//! and exits `1`, matching the oracle's own `pixDeskew failed` branch.
//!
//! ## `deskewboth` — D7, oracle arm supplied, diff is green
//! This arm did NOT exist when the D6/D7 transcode landed: `skew_oracle.cpp`
//! had no `deskewboth` counterpart, and inventing an un-diffable dump format
//! was already the wrong move once on this same file (see `vshear` above,
//! which exists only for manual inspection and says so). **Declining to
//! invent one was the right call — the fix was to supply the missing oracle
//! side, not to lower the evidence bar.** `skew_oracle.cpp` gained a
//! `deskewboth` arm calling the real `pixDeskewBoth`, and this arm now
//! mirrors it: same bare-`dump_pix` shape (no `rc`/angle/conf, since
//! `pixDeskewBoth` returns only a PIX). **Verified byte-identical 10/10** —
//! 5 fixtures × `redsearch` ∈ {2, 4}, both sides asserted non-empty.
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
//! /tmp/skew_oracle dss in.pgm 128 2.5 1 > /tmp/o_dss.tsv
//! cargo run -q -p tesseract-ocr --example deskew_dump -- dss in.pgm 128 2.5 1 > /tmp/r_dss.tsv
//! diff /tmp/o_dss.tsv /tmp/r_dss.tsv
//!
//! /tmp/skew_oracle sweep in.pgm 128 4 2 0.0 7.0 1.0 0.01 1 > /tmp/o_sweep.tsv
//! cargo run -q -p tesseract-ocr --example deskew_dump -- sweep in.pgm 128 4 2 0.0 7.0 1.0 0.01 1 > /tmp/r_sweep.tsv
//! diff /tmp/o_sweep.tsv /tmp/r_sweep.tsv
//!
//! /tmp/skew_oracle deskew in.pgm 2 > /tmp/o_deskew.tsv
//! cargo run -q -p tesseract-ocr --example deskew_dump -- deskew in.pgm 2 > /tmp/r_deskew.tsv
//! diff /tmp/o_deskew.tsv /tmp/r_deskew.tsv
//! ```
#![allow(clippy::print_stdout, reason = "dump CLI")]

use std::path::Path;

use tesseract_ocr::deskew::{
    deskew_both, deskew_general, find_differential_square_sum,
    find_skew_sweep_and_search_score_pivot, rotate90_grey, rotate_am_gray, rotate_am_gray_corner,
    v_shear_center, v_shear_corner, ShearPivot,
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

/// `skew.c`'s OWN degrees→radians conversion (`skew.c:708`,
/// `pixFindSkewSweepAndSearchScorePivot`'s `deg2rad = 3.1415926535 / 180.`):
/// an f64 division narrowed to f32 ONCE, then a plain f32 multiply against
/// the degree value — reproduced identically by `skew_oracle.cpp`'s `dss`
/// arm (`l_float32 deg2rad = static_cast<l_float32>(3.1415926535 / 180.);`).
/// Textually distinct from [`oracle_deg_to_rad`] (used only by the
/// `rotamgray`/`rotamgraycorner` arms, a DIFFERENT literal/precision
/// belonging to `rotateam.c`'s own conversion) — do not merge the two. The
/// `dss`, `vshear`, and `sweep` arms below all use THIS conversion, matching
/// what `v_shear_corner`/`v_shear_center`/`find_skew_sweep_and_search_score_pivot`
/// are actually driven by inside `skew.c`.
///
/// `3.1415926535` is NOT `f64::consts::PI` (`0x400921fb54411744` vs
/// `0x400921fb54442d18`) — but the `as f32` discards the difference, so both
/// spellings yield the identical f32 (verified by bit comparison). Written as
/// the constant because clippy's `approx_constant` rejects the literal and
/// this repo forbids `#[allow]`; the equality holds ONLY while this stays
/// narrowed to f32. Same reasoning as `deskew.rs`'s two sites — see
/// `.claude/plans/deskew-wave-v1.md` § "The pi-literal rule".
fn skew_deg_to_rad(deg: f32) -> f32 {
    let deg2rad = (core::f64::consts::PI / 180.0_f64) as f32;
    deg * deg2rad
}

/// Parses a `<pivot 1=corner|2=center>` CLI argument, matching the oracle's
/// own `L_SHEAR_ABOUT_CORNER`/`L_SHEAR_ABOUT_CENTER` numbering exactly.
fn parse_pivot(s: &str) -> ShearPivot {
    match s.parse::<i32>().expect("pivot") {
        1 => ShearPivot::Corner,
        2 => ShearPivot::Center,
        other => panic!("pivot must be 1 (corner) or 2 (center), got {other}"),
    }
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
    eprintln!("  deskew_dump dss             <pgm> <thresh> <angle_deg> <pivot 1=corner|2=center>");
    eprintln!("  deskew_dump vshear          <pgm> <thresh> <angle_deg> <pivot 1=corner|2=center>");
    eprintln!(
        "  deskew_dump sweep           <pgm> <thresh> <redsweep> <redsearch> <sweepcenter> \
         <sweeprange> <sweepdelta> <minbsdelta> <pivot 1=corner|2=center>"
    );
    eprintln!("  deskew_dump rotamgray       <pgm> <angle_deg> <grayval>");
    eprintln!("  deskew_dump rotamgraycorner <pgm> <angle_deg> <grayval>");
    eprintln!("  deskew_dump deskew          <pgm> <redsearch>");
    eprintln!("  deskew_dump deskewboth      <pgm> <redsearch>");
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
            if args.len() < 6 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let thresh: i32 = args[3].parse().expect("thresh");
            let deg: f32 = args[4].parse().expect("angle_deg");
            let pivot = parse_pivot(&args[5]);
            let binary = threshold_to_binary(&grey, thresh);
            let rad = skew_deg_to_rad(deg);
            let sheared = match pivot {
                ShearPivot::Corner => v_shear_corner(&binary, w, h, rad, 255),
                ShearPivot::Center => v_shear_center(&binary, w, h, rad, 255),
            };
            let sum = find_differential_square_sum(&sheared, w, h);
            println!("rc\t0");
            print_f32("deg", deg);
            print_f32("sum", sum);
        }
        "vshear" => {
            if args.len() < 6 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let thresh: i32 = args[3].parse().expect("thresh");
            let deg: f32 = args[4].parse().expect("angle_deg");
            let pivot = parse_pivot(&args[5]);
            let binary = threshold_to_binary(&grey, thresh);
            let rad = skew_deg_to_rad(deg);
            let sheared = match pivot {
                ShearPivot::Corner => v_shear_corner(&binary, w, h, rad, 255),
                ShearPivot::Center => v_shear_center(&binary, w, h, rad, 255),
            };
            print_f32("deg", deg);
            // Polarity flip for the dump ONLY: this crate's convention is
            // 0=ON; leptonica's native 1bpp (what a `pixGetPixel` dump would
            // show) is 1=ON. Convert at this output boundary -- see the
            // module doc's `vshear` section and `threshold_to_binary`'s own
            // boundary-flip precedent -- do NOT dump this crate's raw bytes
            // as though they were leptonica bit values.
            let lept_bits: Vec<u8> = sheared
                .iter()
                .map(|&b| if b == 0 { 1 } else { 0 })
                .collect();
            dump_pix("vshear", &lept_bits, w, h);
        }
        "sweep" => {
            if args.len() < 11 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let thresh: i32 = args[3].parse().expect("thresh");
            let redsweep: u32 = args[4].parse().expect("redsweep");
            let redsearch: u32 = args[5].parse().expect("redsearch");
            let sweepcenter: f32 = args[6].parse().expect("sweepcenter");
            let sweeprange: f32 = args[7].parse().expect("sweeprange");
            let sweepdelta: f32 = args[8].parse().expect("sweepdelta");
            let minbsdelta: f32 = args[9].parse().expect("minbsdelta");
            let pivot = parse_pivot(&args[10]);
            let binary = threshold_to_binary(&grey, thresh);
            match find_skew_sweep_and_search_score_pivot(
                &binary,
                w,
                h,
                redsweep,
                redsearch,
                sweepcenter,
                sweeprange,
                sweepdelta,
                minbsdelta,
                pivot,
            ) {
                Some(result) => {
                    println!("rc\t0");
                    print_f32("angle", result.angle);
                    print_f32("conf", result.conf);
                    print_f32("endscore", result.endscore);
                }
                None => {
                    // The C initializes *pangle=*pconf=*pendscore=0.0 BEFORE
                    // any validation and never rewrites them on an error
                    // path -- reproduce that exact output shape rather than
                    // omitting the three lines.
                    println!("rc\t1");
                    print_f32("angle", 0.0);
                    print_f32("conf", 0.0);
                    print_f32("endscore", 0.0);
                }
            }
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
        "deskew" => {
            if args.len() < 4 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let redsearch: u32 = args[3].parse().expect("redsearch");
            // pixDeskew(pixg, redsearch) == pixDeskewGeneral(pixg, 0, 0.0,
            // 0.0, redsearch, 0, NULL, NULL) -- skew.c:222. Matches the
            // oracle's own `deskew` arm call exactly.
            match deskew_general(&grey, w, h, 0, 0.0, 0.0, redsearch, 0) {
                Some(result) => {
                    // The oracle's `deskew` arm discards angle/conf (NULL)
                    // and prints ONLY dump_pix's own output -- no `rc` line,
                    // no angle/conf lines. Match that exactly so the two
                    // sides stay byte-for-byte diffable.
                    dump_pix("deskew", &result.grey, result.w, result.h);
                }
                None => {
                    eprintln!("deskew_general failed (returns None on invalid redsweep/redsearch)");
                    std::process::exit(1);
                }
            }
        }
        // D7 — `pixDeskewBoth`. The oracle GAINED a matching `deskewboth` arm
        // (added at the orchestrator level after this file was written), so
        // this leaf now has real byte-parity evidence rather than unit tests
        // alone. Same bare-`dump_pix` shape as the `deskew` arm above:
        // `pixDeskewBoth` returns only a PIX, so there is no `rc`/angle/conf
        // line on either side.
        "deskewboth" => {
            if args.len() < 4 {
                usage();
            }
            let (grey, w, h) = read_grey(&args[2]);
            let redsearch: u32 = args[3].parse().expect("redsearch");
            match deskew_both(&grey, w, h, redsearch) {
                Some((out, ow, oh)) => dump_pix("deskewboth", &out, ow, oh),
                None => {
                    eprintln!("deskew_both failed (returns None on invalid redsearch)");
                    std::process::exit(1);
                }
            }
        }
        _ => usage(),
    }
}
