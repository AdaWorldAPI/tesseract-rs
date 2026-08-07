//! Integration falsifier for `recognize_page_blocks_words` — the
//! multi-column reading-order composition (consumer-side, NOT a Tesseract
//! transcode; see its doc comment).
//!
//! Built from the real repro shape: side-by-side copies of REAL paragraph
//! text with a wide white gutter. The whole-page makerow surface merges each
//! visual row ACROSS the gutter into one full-width line (an 8-column
//! resolution test sheet read as 26 full-width lines where ~176 per-column
//! lines exist); the block-aware surface must read column by column.
//!
//! Lives in `tests/` (not the lib test module) because it runs full-page
//! recognition several times — paragraph-scale content is the point (the
//! degenerate tiny-fixture case is covered by the method's own
//! no-content-loss fallback), and that costs real seconds.

use std::path::Path;

use tesseract_core::DictLite;
use tesseract_ocr::{render_text, LstmRecognizer};

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

fn load() -> Option<(LstmRecognizer, Option<DictLite>)> {
    let c = corpus();
    if !c.join("model/eng.lstm").exists() {
        eprintln!("skipping: corpus model absent");
        return None;
    }
    let lstm = std::fs::read(c.join("model/eng.lstm")).unwrap();
    let uni = std::fs::read_to_string(c.join("model/eng.lstm-unicharset")).unwrap();
    let rec = std::fs::read(c.join("model/eng.lstm-recoder")).unwrap();
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
    let dict = match (
        std::fs::read(c.join("model/eng.lstm-word-dawg")),
        std::fs::read(c.join("model/eng.lstm-punc-dawg")),
        std::fs::read(c.join("model/eng.lstm-number-dawg")),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };
    Some((r, dict))
}

/// A horizontal band of `page_01.pgm` (real rendered paragraphs) — tall
/// enough to carry several text lines, short enough to keep the test's
/// recognition cost in seconds rather than minutes.
fn page_band() -> Option<(Vec<u8>, usize, usize)> {
    let p = corpus().join("pages/page_01.pgm");
    if !p.exists() {
        eprintln!("skipping: corpus pages absent");
        return None;
    }
    let bytes = std::fs::read(&p).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let (top, bottom) = (40usize, 240usize.min(h));
    let bh = bottom - top;
    let band = grey[top * w..bottom * w].to_vec();
    Some((band, w, bh))
}

#[test]
fn blocks_surface_reads_two_columns_in_column_order() {
    let Some((r, dict)) = load() else { return };
    let Some((band, w, h)) = page_band() else {
        return;
    };

    // The single column's own reading — the per-column ground truth.
    let single = r
        .recognize_page_makerow_words(&band, w, h, dict.as_ref())
        .unwrap();
    let single_text = render_text(&single, &r.charset);
    assert!(
        single.len() >= 3,
        "band must carry several real lines, got {}",
        single.len()
    );

    // Compose: [col][gutter][col], gutter = half a column, white.
    let gutter = w / 2;
    let w2 = w * 2 + gutter;
    let mut two_col = vec![255u8; w2 * h];
    for y in 0..h {
        two_col[y * w2..y * w2 + w].copy_from_slice(&band[y * w..(y + 1) * w]);
        let right0 = w + gutter;
        two_col[y * w2 + right0..y * w2 + right0 + w].copy_from_slice(&band[y * w..(y + 1) * w]);
    }

    let merged = r
        .recognize_page_makerow_words(&two_col, w2, h, dict.as_ref())
        .unwrap();
    let blocked = r
        .recognize_page_blocks_words(&two_col, w2, h, dict.as_ref())
        .unwrap();

    // (a) More lines than the across-the-gutter merged reading.
    assert!(
        blocked.len() > merged.len(),
        "blocked ({}) must out-line the merged reading ({})",
        blocked.len(),
        merged.len()
    );

    // (b) Every blocked line stays inside one column (never spans the gutter
    // midpoint); (c) all left-column lines precede any right-column line.
    let mid = (w + gutter / 2) as i32;
    let mut seen_right = false;
    for line in &blocked {
        let (l, _, rt, _) = line.line_box;
        assert!(
            rt <= mid || l >= mid,
            "line [{l},{rt}] spans the gutter midpoint {mid}"
        );
        if l >= mid {
            seen_right = true;
        } else {
            assert!(
                !seen_right,
                "left-column line after a right-column line: reading order broken"
            );
        }
    }

    // (d) Each column reproduces the single band's own text.
    let left: Vec<_> = blocked
        .iter()
        .filter(|l| l.line_box.2 <= mid)
        .cloned()
        .collect();
    let right: Vec<_> = blocked
        .iter()
        .filter(|l| l.line_box.0 >= mid)
        .cloned()
        .collect();
    assert_eq!(
        render_text(&left, &r.charset),
        single_text,
        "left column must read as the single band"
    );
    assert_eq!(
        render_text(&right, &r.charset),
        single_text,
        "right column must read as the single band"
    );

    // (e) Line metrics survive the crop→page translation: every blocked line
    // carries metrics whose baseline sits inside its own line box (item-1
    // integration — the metrics drive renderer font size + baseline).
    for line in &blocked {
        let m = line.metrics.expect("makerow lines carry metrics");
        let (_, b, _, t) = line.line_box;
        assert!(
            m.baseline >= b as f32 - 1.0 && m.baseline <= t as f32 + 1.0,
            "baseline {} outside line box [{b},{t}]",
            m.baseline
        );
        assert!(m.xheight > 0.0, "xheight must be measured");
    }
}

/// The no-content-loss guard: a page whose XY-cut over-splits into
/// unrecognizable micro-blocks (here: `page_roomy.pgm` doubled — 24×36
/// single-glyph leaves) must fall back to the whole-page reading rather than
/// silently dropping text.
#[test]
fn blocks_surface_never_loses_content_to_over_splitting() {
    let Some((r, _)) = load() else { return };
    let p = corpus().join("lines/page_roomy.pgm");
    if !p.exists() {
        eprintln!("skipping: corpus lines absent");
        return;
    }
    let bytes = std::fs::read(&p).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();

    let gutter = w / 2;
    let w2 = w * 2 + gutter;
    let mut two_col = vec![255u8; w2 * h];
    for y in 0..h {
        two_col[y * w2..y * w2 + w].copy_from_slice(&grey[y * w..(y + 1) * w]);
        let right0 = w + gutter;
        two_col[y * w2 + right0..y * w2 + right0 + w].copy_from_slice(&grey[y * w..(y + 1) * w]);
    }

    let whole = r
        .recognize_page_makerow_words(&two_col, w2, h, None)
        .unwrap();
    let blocked = r
        .recognize_page_blocks_words(&two_col, w2, h, None)
        .unwrap();

    let words = |ls: &[tesseract_ocr::LineWords]| ls.iter().map(|l| l.words.len()).sum::<usize>();
    assert!(
        words(&blocked) >= words(&whole),
        "blocked reading ({} words) must never lose content vs whole-page ({} words)",
        words(&blocked),
        words(&whole)
    );
}

// ---------------------------------------------------------------------------
// Grid inheritance (#50/#53): the integration falsifier the CORPUS cannot give.
//
// `examples/raster_probe.rs` measured the committed corpus: 23 fixtures, the
// raster changes the block list on ZERO. That is not a defect — it is the
// P-50 result (on `resgrid.pgm` the raster detects the 8-column lattice and
// correctly splits nothing, because that page cuts vertically only, so both
// bands already live in the same columns and there is no merged band to
// inherit geometry FOR). The failure mode the feature exists to fix simply is
// not expressed anywhere in the corpus — the same gap `gen_faded_contrast.py`
// was built to close for Wolf. So it is built here.
// ---------------------------------------------------------------------------

/// Draw a hollow glyph-sized mark (density ~30-40%, so `filter_blobs`'
/// `>= h*w*0.7` "too dense to be text" heuristic does not reject it — the
/// documented `rectify.rs` fixture trap).
fn mark(page: &mut [u8], w: usize, x: usize, y: usize, mw: usize, mh: usize) {
    for dy in 0..mh {
        for dx in 0..mw {
            let edge = dy == 0 || dy == mh - 1 || dx == 0 || dx == mw - 1;
            if edge {
                page[(y + dy) * w + (x + dx)] = 0;
            }
        }
    }
}

/// A SOLID bar. Deliberately not [`mark`]: a hollow rectangle draws only its
/// edges, so a "bridge" built from one would leave its own white interior as a
/// corridor and `xy_cut` would cut straight through it — measured, that was
/// this fixture's second wrong version (widths [184, 280, 280, 280], still no
/// merged band). A bridge must be solid; the density heuristic that motivates
/// hollow glyphs lives in `filter_blobs`, which is downstream of `xy_cut`'s
/// projection profile and irrelevant here.
fn bar(page: &mut [u8], w: usize, x: usize, y: usize, bw: usize, bh: usize) {
    for dy in 0..bh {
        for dx in 0..bw {
            page[(y + dy) * w + (x + dx)] = 0;
        }
    }
}

/// A 2-band x 4-column page whose TOP band has clean gutters (so `xy_cut`
/// splits it into 4) and whose BOTTOM band has ink bridging every gutter (so
/// `xy_cut` cannot split it and emits ONE full-width block). The top band's
/// geometry is exactly the evidence the bottom band lacks.
fn two_band_page(bridge_bottom: bool) -> (Vec<u8>, usize, usize) {
    // Geometry chosen so `xy_cut`'s axis choice is UNAMBIGUOUS. It picks the
    // axis with the THICKEST valid valley (tie -> vertical), so the band gap
    // must clearly beat the column gutters or the page cuts into full-height
    // columns and never separates the bands at all — measured, that is what a
    // 130 px band gap against 92 px gutters did (every block spanned
    // y 40..510, both bands, no horizontal cut anywhere).
    //
    //   column pitch 250, content 184  => gutter  66 px
    //   band gap                        => 212 px   (3.2x the gutter)
    //
    // Content is 7*24 + 16 = 184, NOT 8*24 = 192: eight marks at 24 px pitch
    // put the LAST mark's left edge at 7*24. Getting that wrong left a 20 px
    // sliver between the content edge and the bridge bar, which cleared
    // `gap_min` (15 at this extent) and split the band anyway — the bridge
    // must start exactly at the content edge.
    //
    // The text-ROW geometry matters just as much, and cost three wrong
    // versions to see: `axis_cuts` confirms a valley only if the gap to the
    // ADJACENT CANDIDATE valley is >= `min_region_px` (24) — not the gap to
    // the rect edge. With 20 px rows at 10 px leading, every inter-row gap
    // clears `gap_min` (ceil(0.015 * 640) = 10) and so becomes a candidate,
    // leaving the band gap with a 20 px neighbour and getting it REJECTED.
    // Rows are therefore 24 px tall at 30 px pitch => 6 px leading, BELOW
    // `gap_min`, so the band gap is the only horizontal candidate.
    //
    // ...and 24 px was still wrong, for the OPPOSITE reason: `gap_min` is
    // relative to the CURRENT rect, so once the page is cut into bands and
    // columns the extent collapses (640 -> 178) and a 6 px leading clears the
    // now-tiny bar (ceil(0.015 * 178) = 3), decomposing every column into one
    // block PER LINE (measured: 48 blocks). Rows are 28 px at 30 px pitch =>
    // 2 px leading, below the threshold at EVERY recursion depth. The marks
    // are likewise 16 px at 24 px pitch, so the 8 px inter-mark gaps are
    // rejected by the `min_region_px` confirm filter (a 16 px mark between
    // two candidates is < 24) rather than by thickness alone.
    let (w, h) = (1100usize, 640usize);
    let mut page = vec![255u8; w * h];
    let cols = [40usize, 290, 540, 790];
    for band_y in [40usize, 430] {
        for &cx in &cols {
            for row in 0..6 {
                for i in 0..8 {
                    mark(&mut page, w, cx + i * 24, band_y + row * 30, 16, 28);
                }
            }
        }
    }
    if bridge_bottom {
        // Solid bars covering each bottom-band gutter COMPLETELY:
        // content ends at cx+184, the next column starts at cx+250, so the
        // bar spans cx+184 .. cx+254 and overlaps both sides.
        for &cx in &cols[..3] {
            for row in 0..6 {
                bar(&mut page, w, cx + 184, 430 + row * 30, 70, 28);
            }
        }
    }
    (page, w, h)
}

#[test]
fn a_merged_band_inherits_the_clean_band_s_column_lattice() {
    let params = tesseract_ocr::xy_cut::XyCutParams::default();

    // Control: with no bridging, xy_cut alone already finds every column, so
    // the raster has nothing to add. Proves the fixture is a real 4-column
    // layout and not an artifact.
    let (clean, w, h) = two_band_page(false);
    let clean_blocks = tesseract_ocr::xy_cut::xy_cut(&clean, w, h, &params);
    assert!(
        clean_blocks.len() >= 4,
        "un-bridged fixture must expose the 4-column layout, got {}",
        clean_blocks.len()
    );

    // The real case: bridging the bottom band's gutters merges it.
    let (page, w, h) = two_band_page(true);
    let blocks = tesseract_ocr::xy_cut::xy_cut(&page, w, h, &params);
    let widths: Vec<usize> = blocks.iter().map(|b| b.right - b.left).collect();
    let merged = widths.iter().filter(|&&x| x > w / 2).count();
    assert!(
        merged >= 1,
        "fixture must produce at least one merged full-width band \
         (widths {widths:?}) or this test measures nothing"
    );

    // Detection + split — what `apply_grid_raster` composes at all three
    // wired call sites.
    let raster = tesseract_ocr::grid_raster::detect_column_raster(&blocks)
        .expect("the clean band establishes a lattice");
    let after = tesseract_ocr::grid_raster::split_nonconforming(&blocks, &raster);

    assert!(
        after.len() > blocks.len(),
        "the merged band must be re-segmented: {} -> {}",
        blocks.len(),
        after.len()
    );
    let still_merged = after.iter().filter(|b| b.right - b.left > w / 2).count();
    assert!(
        still_merged < merged,
        "at least one merged band must be gone: {merged} -> {still_merged}"
    );
}

/// The WIRING falsifier — the sibling above proves the FEATURE fires but calls
/// `detect_column_raster`/`split_nonconforming` directly, so it passes
/// identically with `apply_grid_raster` unhooked (measured: it did). This one
/// goes through the wired public entry and asserts what a reader actually
/// sees: with the merged band read as ONE block, every recognized line spans
/// all four columns — text read straight across the gutters, the exact defect
/// `recognize_page_blocks_words` exists to prevent one level up.
#[test]
fn the_wired_path_does_not_read_a_merged_band_across_its_gutters() {
    let Some((r, dict)) = load() else { return };
    let (page, w, h) = two_band_page(true);
    let lines = r
        .recognize_page_blocks_words(&page, w, h, dict.as_ref())
        .expect("recognize");
    assert!(!lines.is_empty(), "fixture must recognize something");

    // A line whose box spans more than half the page crossed a gutter.
    let wide = lines
        .iter()
        .filter(|l| (l.line_box.2 - l.line_box.0) as usize > w / 2)
        .count();
    assert_eq!(
        wide,
        0,
        "no line may span the gutters; {wide} of {} do (boxes {:?})",
        lines.len(),
        lines
            .iter()
            .map(|l| (l.line_box.0, l.line_box.2))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// KNOWN DEFECT, pinned two-sided (found via tesseract-ocr/test phototest.tif).
//
// `gap_min = ceil(min_gap_frac * extent)` is PAGE-RELATIVE, so on a SMALL page
// it falls below ordinary word spacing and `xy_cut` cuts at every word — the
// exact mirror of the wide-multi-column bug the gutter fallback fixes, where
// the same threshold was too STRICT. Measured on upstream's own canonical
// `phototest.tif` (640x480, 12 pt text, 2 paragraphs, 8 lines):
// **19 blocks, most of them single words.**
//
// The recognized TEXT is saved only by `recognize_page_blocks_words`'
// no-content-loss fallback (the whole-page reading wins on word count), so
// this is invisible in the goldens — but `recognize_document`'s REGIONS are
// word fragments, and any consumer reading `doc.v1`'s region structure on a
// small page gets nonsense.
//
// Not fixed here: the principled correction is the same shape as the gutter
// fallback — judge a valley against its NEIGHBOURS, not the page — and that
// is a change to `xy_cut`'s cut decision with golden implications. Pinned so
// a fix FAILS this test and forces a deliberate re-pin.
// ---------------------------------------------------------------------------

/// A small page with ordinary word spacing — the regime where the
/// page-relative threshold under-shoots. 640 px wide => `gap_min` = 10, and
/// the word gaps are 10 px, so every one of them is a cut candidate; the
/// words are 40 px wide, comfortably over `min_region_px` (24), so the
/// confirm filter passes them too.
fn small_page_ordinary_word_spacing() -> (Vec<u8>, usize, usize) {
    let (w, h) = (640usize, 200usize);
    let mut page = vec![255u8; w * h];
    // 4 lines x 6 words; word = 4 marks at 8 px advance (ink 31), word gap 10.
    for row in 0..4 {
        let y = 40 + row * 34;
        for word in 0..6 {
            let wx = 40 + word * 41;
            for i in 0..4 {
                mark(&mut page, w, wx + i * 8, y, 7, 22);
            }
        }
    }
    (page, w, h)
}

#[test]
fn small_page_word_spacing_is_cut_as_if_it_were_column_gutters() {
    let (page, w, h) = small_page_ordinary_word_spacing();
    let params = tesseract_ocr::xy_cut::XyCutParams::default();
    let blocks = tesseract_ocr::xy_cut::xy_cut(&page, w, h, &params);

    // Anti-vacuity: the fixture must genuinely be the small-page regime —
    // gap_min at or below the drawn word gap, or it measures nothing.
    let gap_min = (0.015_f32 * w as f32).ceil() as usize;
    assert!(
        gap_min <= 10,
        "fixture must sit in the regime where gap_min ({gap_min}) <= the 10 px \
         word gap, or the over-split cannot happen"
    );

    // THE DEFECT, pinned by its SIGNATURE rather than a count: a 4-line page
    // with no columns should yield ONE full-width block. Measured, it yields
    // 6 — one per word position, each spanning all four lines, exactly as if
    // the word gaps were column gutters. So no block is page-wide.
    let widest = blocks.iter().map(|b| b.right - b.left).max().unwrap_or(0);
    assert!(
        widest < w / 2,
        "PINNED DEFECT: ordinary word spacing is cut as column gutters, so no \
         block should span the page — got {} blocks, widest {widest} px of \
         {w}. If a block is now page-wide the defect is FIXED and this pin \
         must be re-measured, not deleted",
        blocks.len()
    );
}
