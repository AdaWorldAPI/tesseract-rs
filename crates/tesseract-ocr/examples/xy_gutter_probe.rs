//! Diagnostic: report the TOP-LEVEL vertical (column) cut decision for a page —
//! the page width, the `min_gap_frac`-derived absolute gutter threshold, and
//! every interior whitespace run with its width, so a merged-column result can
//! be attributed to a specific gutter/threshold margin rather than guessed at.
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_ocr::xy_cut::binarize_page_with;
use tesseract_ocr::{BinarizeMode, XyCutParams};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("read pgm");
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).expect("parse pgm");
    let params = XyCutParams::default();
    let binary = binarize_page_with(&grey, w, h, BinarizeMode::Otsu);

    // Top-level column ink profile: a column x is inked if ANY row has ink.
    let inked: Vec<bool> = (0..w)
        .map(|x| (0..h).any(|y| binary[y * w + x] == 0))
        .collect();

    let gap_min = (params.min_gap_frac * w as f32).ceil().max(1.0) as usize;
    println!("page {w}x{h}");
    println!(
        "min_gap_frac={} -> gap_min = ceil({} * {w}) = {gap_min} px required",
        params.min_gap_frac, params.min_gap_frac
    );

    let first = inked.iter().position(|&b| b).unwrap_or(0);
    let last = inked.iter().rposition(|&b| b).unwrap_or(0);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = first;
    while i <= last {
        if inked[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i <= last && !inked[i] {
            i += 1;
        }
        runs.push((s, i));
    }
    let inked_cols = inked.iter().filter(|&&b| b).count();
    let n_cols_guess = runs.len() + 1;
    println!(
        "interior whitespace runs: {} (=> {} column bands), inked cols {inked_cols}",
        runs.len(),
        n_cols_guess
    );
    let mut passing = 0;
    for (s, e) in &runs {
        let width = e - s;
        let ok = width >= gap_min;
        if ok {
            passing += 1;
        }
        println!(
            "  gutter x={s:5}..{e:5}  width={width:4}  {}",
            if ok { "PASS" } else { "fail" }
        );
    }
    println!("\n{passing}/{} gutters clear the threshold", runs.len());
    let avg_col = inked_cols.checked_div(n_cols_guess).unwrap_or(0);
    println!("mean inked column band width ~= {avg_col} px");
    if avg_col > 0 {
        println!(
            "gap_min as a fraction of ONE column band: {:.1}%",
            100.0 * gap_min as f32 / avg_col as f32
        );
    }
    let leaves = tesseract_ocr::xy_cut(&grey, w, h, &params);
    println!("xy_cut leaves: {}", leaves.len());
}
