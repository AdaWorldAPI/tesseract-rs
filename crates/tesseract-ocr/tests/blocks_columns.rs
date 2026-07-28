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
