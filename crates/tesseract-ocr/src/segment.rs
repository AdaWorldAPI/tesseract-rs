//! The shared row-segmentation prefix: binarize the page and find connected
//! components — extracted verbatim (no behaviour change) from
//! `lstm_recognizer.rs`'s `makerow_row_crops`, so every consumer of the line
//! finder shares ONE binarize+conn-comp pass instead of drifting.
//!
//! This is orchestration, not a single Tesseract leaf: it composes already
//! byte-parity-proven pieces (`threshold`, `conncomp`, `blob_filter`,
//! `textline`) the same way `Tesseract::FindLines`'s early stages do, but is
//! not itself claimed as a byte-for-byte transcode of any one function.
//!
//! Two entry points, because `makerow.cpp`'s real pipeline computes TWO
//! different things from the same blobs, and only one of them survives to
//! the end:
//!
//! - [`segment_rows`] runs the FULL line finder (`make_rows`, wave 1-3 +
//!   x-height) — the one [`crate::lstm_recognizer::LstmRecognizer`]'s row
//!   crops use. Its final `ToRow::line_m()` is the SAME value for every row
//!   in the block: `make_rows` → `cleanup_rows_making` →
//!   `fit_parallel_rows(block, page_m)` deliberately forces every row onto
//!   one shared page-wide gradient (real Tesseract's actual assumption: a
//!   rotated-but-flat page has all its lines parallel). This is correct and
//!   intentional — do not "fix" it.
//! - [`segment_rows_independent`] stops ONE STEP EARLIER, at
//!   `make_initial_textrows` (`makerow.cpp:254-289`) — before the parallel
//!   constraint is applied, each row still carries its OWN independent LMS
//!   line fit (`fit_lms_line` per row). [`crate::rectify`] needs exactly
//!   this: a page-wide-constant slope can only ever measure ROTATION; a
//!   trapezoid/keystone page's row-to-row slope VARIATION only exists in
//!   this pre-parallel-fit signal, because `fit_parallel_rows` erases it a
//!   few lines later in the real pipeline. The row/blob grouping here is the
//!   cruder pass-0 assignment (`assign_blobs_to_rows(block, None, ...)`, not
//!   yet refined by the three gradient-aware passes `cleanup_rows_making`
//!   runs) — adequate for "how much does this row tilt", not claimed
//!   adequate for anything needing exact blob membership.

use crate::blob_filter::filter_blobs;
use crate::conncomp::conn_comp_areas;
use crate::textline::{compute_block_xheight, make_initial_textrows, make_rows, ToBlockCtx};
use crate::threshold::{otsu_threshold_gray, threshold_rect_to_binary};

/// Binarize (Otsu) + 8-connected components + the noise/size partition —
/// shared by both [`segment_rows`] and [`segment_rows_independent`]. Returns
/// a `ToBlockCtx` seeded with the surviving blob pool and `filter_blobs`'
/// line-size estimate, ready for either `make_rows` or `make_initial_textrows`.
fn seed_block(grey: &[u8], w: usize, h: usize) -> ToBlockCtx {
    // P2: binarize the whole page (foreground = 0 per the crate's
    // grey-image convention).
    let otsu = otsu_threshold_gray(grey, w, 0, 0, w, h);
    let binary = threshold_rect_to_binary(grey, w, 0, 0, w, h, otsu);

    // 3B + 3F₂ leaf 1: components with ink pixel counts (8-connectivity,
    // matching the real pipeline's blob granularity most closely).
    let mut components = conn_comp_areas(&binary, w, h, 8);
    // Raster space → Tesseract y-UP page space for the makerow math.
    for c in &mut components {
        c.bb.y = h as i32 - (c.bb.y + c.bb.h);
    }

    // 3F₂ leaf 2: noise partition + the line-size seed.
    let filtered = filter_blobs(&components);

    ToBlockCtx {
        blobs: filtered.blobs,
        block_left: 0,
        line_spacing: filtered.line_spacing,
        line_size: filtered.line_size,
        max_blob_size: filtered.max_blob_size,
        ..Default::default()
    }
}

/// The FULL line finder: binarize → components → the real wave-1..3
/// `make_rows` (parallel-forced) → x-height. Everything
/// [`crate::lstm_recognizer::LstmRecognizer`]'s row crops need. Every row's
/// `line_m()` is the SAME page-wide value — see the module docs.
#[must_use]
pub(crate) fn segment_rows(grey: &[u8], w: usize, h: usize) -> ToBlockCtx {
    let mut blocks = [seed_block(grey, w, h)];
    let page_m = make_rows(&mut blocks);
    let [mut block] = blocks;
    compute_block_xheight(&mut block, page_m, 0.0);
    block
}

/// Stops at `make_initial_textrows` — each row keeps its OWN independent LMS
/// line fit (`ToRow::line_m()`/`parallel_c()` genuinely vary row-to-row), not
/// yet forced parallel. See the module docs for why [`crate::rectify`] needs
/// this instead of [`segment_rows`].
#[must_use]
pub(crate) fn segment_rows_independent(grey: &[u8], w: usize, h: usize) -> ToBlockCtx {
    let mut block = seed_block(grey, w, h);
    make_initial_textrows(&mut block);
    block
}
