//! **ClassView placement — the same synthesize-the-border algorithm, rotated.**
//!
//! A page IS a table with one big cell: `Main`, with `Header` above and
//! `Footer` below. The whitespace separating them is the same signal as the
//! whitespace between table columns — a corridor — just on the other axis.
//! So the gutter-derivation that recovered 4 table columns should recover the
//! page's furniture bands, and this measures whether it does, against `xy_cut`
//! (what the pipeline uses today) on a fixture with KNOWN band positions.
//!
//! Named ClassView per the OGAR concept: one register, projected into a typed
//! reading. `ogar_doc_ir::RegionKind` already carries `Header`/`Main`/`Footer`.

use std::path::Path;
use tesseract_ocr::{xy_cut, XyCutParams};

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Horizontal ink projection → whitespace bands, median-filtered so ordinary
/// inter-LINE leading does not read as a region boundary. Exactly the table
/// algorithm's gutter filter, on the other axis: band gaps are the outliers,
/// line leading is the median.
fn classview_bands(grey: &[u8], w: usize, h: usize) -> Vec<(usize, usize)> {
    let inked: Vec<bool> = (0..h)
        .map(|y| (0..w).any(|x| grey[y * w + x] < 128))
        .collect();
    let top = inked.iter().position(|&b| b).unwrap_or(0);
    let bot = inked.iter().rposition(|&b| b).unwrap_or(h - 1);

    let mut gaps = Vec::new();
    let mut run: Option<usize> = None;
    for (y, &ink) in inked.iter().enumerate().take(bot + 1).skip(top) {
        if ink {
            if let Some(s) = run.take() {
                gaps.push((s, y));
            }
        } else if run.is_none() {
            run = Some(y);
        }
    }
    let mut widths: Vec<usize> = gaps.iter().map(|(s, e)| e - s).collect();
    widths.sort_unstable();
    let median = widths.get(widths.len() / 2).copied().unwrap_or(0);
    let splits: Vec<(usize, usize)> = gaps
        .into_iter()
        .filter(|(s, e)| e - s >= (median * 2).max(1))
        .collect();

    // Bands are what the wide gaps separate.
    let mut bands = Vec::new();
    let mut cur = top;
    for (s, e) in &splits {
        bands.push((cur, *s));
        cur = *e;
    }
    bands.push((cur, bot + 1));
    bands
}

#[test]
fn classview_vs_xy_cut_placement_exactness() {
    let p = corpus().join("pages/page_furniture.pgm");
    if !p.exists() {
        eprintln!("skipping: furniture fixture absent");
        return;
    }
    let bytes = std::fs::read(&p).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let gt: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(corpus().join("pages/page_furniture.gt.json")).unwrap(),
    )
    .unwrap();

    let want = [
        ("header", &gt["header"]),
        ("main", &gt["main"]),
        ("footer", &gt["footer"]),
    ];

    // ── what the pipeline uses today ──
    let regions = xy_cut(&grey, w, h, &XyCutParams::default());
    eprintln!("xy_cut produced {} regions:", regions.len());
    for r in &regions {
        eprintln!("   top={} bottom={}", r.top, r.bottom);
    }

    // ── the ClassView band projection ──
    let bands = classview_bands(&grey, w, h);
    eprintln!("classview produced {} bands:", bands.len());
    for (t, b) in &bands {
        eprintln!("   top={t} bottom={b}");
    }

    eprintln!("\nplacement error vs ground truth (px, top/bottom):");
    for (name, g) in &want {
        let (gt_t, gt_b) = (g["top"].as_i64().unwrap(), g["bottom"].as_i64().unwrap());
        let best_xy = regions
            .iter()
            .map(|r| {
                (
                    (r.top as i64 - gt_t).abs() + (r.bottom as i64 - gt_b).abs(),
                    r.top,
                    r.bottom,
                )
            })
            .min();
        let best_cv = bands
            .iter()
            .map(|&(t, b)| ((t as i64 - gt_t).abs() + (b as i64 - gt_b).abs(), t, b))
            .min();
        eprintln!(
            "  {name:7} gt=[{gt_t},{gt_b}]  xy_cut={:?}  classview={:?}",
            best_xy.map(|(e, t, b)| format!("[{t},{b}] err={e}")),
            best_cv.map(|(e, t, b)| format!("[{t},{b}] err={e}")),
        );
    }
}
