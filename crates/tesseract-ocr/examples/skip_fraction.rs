//! **Instrumentation ONLY** — the go/no-go gate for the skip/escalate cascade
//! (operator-hardened: "First task is instrumentation only. Abort cascade work
//! if skipable fraction < 30%. Proceed if >= 60%.").
//!
//! This example RECOGNIZES NOTHING and touches no recognizer/GEMM code. It only
//! measures how much of the work the LSTM forward would process is trivially
//! **empty** — the upper bound on what a blank-run skip gate could avoid. The
//! LSTM stays the truth oracle; this only asks whether there is enough
//! skippable volume to justify building a gate in front of it.
//!
//! Metric (per the pinned decision — forward = 99.4% of time, so the target is
//! forward VOLUME): for each committed page, find the text-line bands
//! (`find_text_lines`, the projection-profile finder — adequate for *counting*,
//! we are not recognizing), and within each band's INK SPAN (leading/trailing
//! blank columns are trimmed — the typographic crop would drop them, so they are
//! not LSTM work), count how many interior columns are **empty** (no ink pixel
//! in the band's row range) vs **ink**. The blank-column fraction is preserved
//! under the front-end's 2x maxpool downsampling, so it proxies the fraction of
//! LSTM timesteps that are pure-blank (the safely-skippable floor).
//!
//! What this does NOT measure (the SECOND probe, gated on this one passing):
//! the "trivial/high-confidence repeated token" middle category — that needs the
//! Morton/CAM-PQ descriptor, which is not built and must not be, per the
//! instrumentation-only rule. This probe reports the EMPTY floor only; the
//! trivial-repeat category would only raise the skippable fraction, so the
//! empty floor alone is a sound lower bound for the go/no-go gate.
//!
//! ```sh
//! cargo run --release -p tesseract-ocr --features seg-approx --example skip_fraction
//! ```
#![cfg(feature = "seg-approx")]

use std::path::{Path, PathBuf};

use tesseract_ocr::{find_text_lines, otsu_threshold_gray, parse_pgm, threshold_rect_to_binary};

fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// One page's tally: interior (ink-span) columns across all its line bands, and
/// how many of those are empty (no ink in the band's row range).
struct PageTally {
    name: String,
    bands: usize,
    interior_cols: usize,
    empty_cols: usize,
}

/// Count, within a band spanning raster rows `[top, bottom)` of a `w`-wide grey
/// image, the interior (between first and last ink column) total vs empty
/// columns. `is_ink[y*w + x]` is precomputed page-wide (grey < otsu).
fn tally_band(is_ink: &[bool], w: usize, top: usize, bottom: usize) -> (usize, usize) {
    let col_has_ink = |x: usize| (top..bottom).any(|y| is_ink[y * w + x]);
    let first = (0..w).find(|&x| col_has_ink(x));
    let last = (0..w).rev().find(|&x| col_has_ink(x));
    let (Some(first), Some(last)) = (first, last) else {
        return (0, 0); // band with no ink at all — contributes nothing
    };
    let interior = last - first + 1;
    let empty = (first..=last).filter(|&x| !col_has_ink(x)).count();
    (interior, empty)
}

fn main() {
    let pages_dir = corpus_root().join("pages");
    let mut tallies: Vec<PageTally> = Vec::new();

    for nn in 1..=10 {
        let name = format!("page_{nn:02}.pgm");
        let path = pages_dir.join(&name);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                continue;
            }
        };
        let (grey, w, h) = parse_pgm(&bytes).unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
        // Exactly the real path's binarization (recognize_page_makerow): Otsu +
        // threshold_rect_to_binary, foreground (ink) == 0 per the crate's
        // grey-image convention.
        let otsu = otsu_threshold_gray(&grey, w, 0, 0, w, h);
        let binary = threshold_rect_to_binary(&grey, w, 0, 0, w, h, otsu);
        let is_ink: Vec<bool> = binary.iter().map(|&b| b == 0).collect();

        let bands = find_text_lines(&grey, w, h);
        let mut interior = 0usize;
        let mut empty = 0usize;
        for band in &bands {
            let (i, e) = tally_band(&is_ink, w, band.top, band.bottom);
            interior += i;
            empty += e;
        }
        tallies.push(PageTally {
            name,
            bands: bands.len(),
            interior_cols: interior,
            empty_cols: empty,
        });
    }

    // ---- report ----
    println!("# skip-fraction instrumentation (LSTM-invocation gate go/no-go)\n");
    println!("Empty-column fraction within line-band ink spans — the safely-skippable");
    println!("floor for a blank-run skip gate in front of `network.forward`. RECOGNIZES");
    println!("NOTHING; the LSTM stays the truth oracle. This measures whether enough");
    println!("skippable volume exists to justify the cascade.\n");
    println!("| page | bands | interior cols | empty cols | empty % |");
    println!("|---|---|---|---|---|");
    let mut tot_interior = 0usize;
    let mut tot_empty = 0usize;
    for t in &tallies {
        let pct = if t.interior_cols == 0 {
            0.0
        } else {
            100.0 * t.empty_cols as f64 / t.interior_cols as f64
        };
        println!(
            "| {} | {} | {} | {} | {:.1}% |",
            t.name, t.bands, t.interior_cols, t.empty_cols, pct
        );
        tot_interior += t.interior_cols;
        tot_empty += t.empty_cols;
    }
    let agg = if tot_interior == 0 {
        0.0
    } else {
        100.0 * tot_empty as f64 / tot_interior as f64
    };
    println!("\n**Aggregate empty-column fraction: {agg:.1}%** ({tot_empty} / {tot_interior})\n");

    let verdict = if agg < 30.0 {
        "ABORT — skippable fraction < 30%. The blank-run skip gate cannot pay for \
         itself; the real lever is BATCHING the forward (fill the starved VNNI \
         tiles), not skipping it. Do NOT build the cascade."
    } else if agg >= 60.0 {
        "PROCEED — skippable fraction >= 60%. A blank-run skip gate in front of \
         network.forward is worth building (always escalating ink columns to the \
         exact LSTM preserves byte parity). Next: the CAM-PQ trivial-repeat probe \
         on HARD glyphs (touching/kerning/broken/italic/multilingual), not clean \
         isolated glyphs."
    } else {
        "MARGINAL (30–60%) — a blank-run skip gate would help modestly but is not \
         a clear win; weigh it against the batching lever, which attacks the same \
         99.4% without the skip machinery. Prefer batching first."
    };
    println!("## Verdict\n\n{verdict}");
    println!(
        "\n_Note: this is the EMPTY floor only. The trivial-repeat category (gated on \
         a Morton/CAM-PQ descriptor, not built) could raise the skippable fraction, \
         so a MARGINAL/ABORT here is a firm 'the easy skip is not there'; a PROCEED \
         here is already sufficient on empties alone._"
    );
}
