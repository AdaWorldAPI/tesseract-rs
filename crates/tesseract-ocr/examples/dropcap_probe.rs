//! Where does a DROP CAP go? Dump `filter_blobs`' four buckets for a real
//! page, with the thresholds that decided each one.
//!
//! Diagnostic only. The open defect (CLAUDE.md, "the drop-cap initial is
//! dropped"): a page whose first paragraph opens with a large ornamental
//! initial recognizes as "ice" rather than "Alice". Two candidate
//! mechanisms were named but never separated:
//!
//! 1. `filter_blobs` classifies it `large` (`height > max_y` or
//!    `width > max_x`) and NOTHING consumes that bucket — the exact
//!    structural shape the period bug had with `noise`, at the opposite end
//!    of the size scale.
//! 2. `make_rows` sees it but assigns it to no row (it spans several line
//!    heights, so no single row band claims it).
//!
//! Those have different fixes, and the recognized text alone cannot tell
//! them apart. This prints the bucket membership directly, so the answer is
//! read rather than inferred.
//!
//! ```sh
//! cargo run -p tesseract-ocr --release --example dropcap_probe -- /tmp/alice_page.pgm
//! ```
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_ocr::{conn_comp_areas, filter_blobs};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pgm = &a[1];
    let bytes = std::fs::read(pgm).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    println!("== {pgm}  {w}x{h} ==");

    // Reproduce `segment::seed_block`'s chain: the DEFAULT (Otsu) binarize,
    // 8-connectivity components, raster -> y-UP flip, then filter_blobs.
    // `binarize_page_with` is the public Otsu entry; segmentation's own
    // wrapper differs only in the local-adaptive arms, which Otsu does not
    // take, so the default path is reproduced exactly here.
    let binary =
        tesseract_ocr::xy_cut::binarize_page_with(&grey, w, h, tesseract_ocr::BinarizeMode::Otsu);
    let mut components = conn_comp_areas(&binary, w, h, 8);
    for c in &mut components {
        c.bb.y = h as i32 - (c.bb.y + c.bb.h);
    }
    let f = filter_blobs(&components);

    println!(
        "components={}  ->  blobs={} noise={} small={} large={}",
        components.len(),
        f.blobs.len(),
        f.noise.len(),
        f.small.len(),
        f.large.len()
    );
    println!(
        "line_size={:.3} line_spacing={:.3} max_blob_size={:.3}",
        f.line_size, f.line_spacing, f.max_blob_size
    );

    // `large` is the bucket under suspicion. Print every member, tallest
    // first, with the height ratio against the ordinary-pool median so a
    // drop cap (a few line-heights tall, ONE glyph wide) is distinguishable
    // from a rule, a figure, or a page-wide smear.
    let mut pool_h: Vec<i32> = f.blobs.iter().map(|b| (b.3 - b.1).abs()).collect();
    pool_h.sort_unstable();
    let med_pool_h = pool_h.get(pool_h.len() / 2).copied().unwrap_or(1).max(1);
    println!("\nmedian ORDINARY blob height = {med_pool_h}");

    let mut larges: Vec<(i32, i32, i32, i32)> = f.large.clone();
    larges.sort_by_key(|b| -(b.3 - b.1).abs());
    println!(
        "\n--- LARGE bucket ({} members, tallest first) ---",
        larges.len()
    );
    println!("  (nothing in this crate consumes `.large`; every one of these is DROPPED)");
    for b in larges.iter().take(25) {
        let (l, bot, r, top) = *b;
        let hh = (top - bot).abs();
        let ww = (r - l).abs();
        println!(
            "  x={l:5}..{r:<5} y={bot:5}..{top:<5}  {ww:4}x{hh:<4}  \
             h/median={:.2}x  aspect={:.2}",
            hh as f32 / med_pool_h as f32,
            ww as f32 / hh.max(1) as f32
        );
    }
    if larges.len() > 25 {
        println!("  ... {} more", larges.len() - 25);
    }

    // A drop cap is TALL but roughly glyph-shaped (aspect ~0.4-1.2) and sits
    // near the left margin. Rules and smears are extreme-aspect; figures are
    // large in both axes. Naming the shape makes the finding checkable
    // rather than a claim about "the big one".
    println!("\n--- drop-cap-SHAPED members of `large` ---");
    println!("  (tall: h >= 1.4x median ordinary height; glyph-ish: 0.3 <= w/h <= 1.5)");
    let mut hits = 0;
    for b in &larges {
        let (l, bot, r, top) = *b;
        let hh = (top - bot).abs();
        let ww = (r - l).abs();
        let aspect = ww as f32 / hh.max(1) as f32;
        if hh as f32 >= 1.4 * med_pool_h as f32 && (0.3..=1.5).contains(&aspect) {
            hits += 1;
            println!(
                "  x={l:5}..{r:<5} y={bot:5}..{top:<5}  {ww:4}x{hh:<4}  \
                 h/median={:.2}x  aspect={aspect:.2}",
                hh as f32 / med_pool_h as f32
            );
        }
    }
    if hits == 0 {
        println!("  NONE — so the drop cap is NOT in `large`, and mechanism (1) is");
        println!("  ruled out. Look at row assignment (mechanism 2) instead.");
        return;
    }

    // ---- recognition experiment on the tallest large-bucket member --------
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let Ok(lstm) = std::fs::read(c.join("eng.lstm")) else {
        println!("\n(no corpus model — skipping the recognition arm)");
        return;
    };
    let r = tesseract_ocr::LstmRecognizer::from_components(
        &lstm,
        &std::fs::read_to_string(c.join("eng.lstm-unicharset")).unwrap(),
        &std::fs::read(c.join("eng.lstm-recoder")).unwrap(),
    )
    .unwrap();
    let dict = match (
        std::fs::read(c.join("eng.lstm-word-dawg")),
        std::fs::read(c.join("eng.lstm-punc-dawg")),
        std::fs::read(c.join("eng.lstm-number-dawg")),
    ) {
        (Ok(a), Ok(b2), Ok(n)) => tesseract_core::DictLite::from_components(&a, &b2, &n).ok(),
        _ => None,
    };

    let (l, bot, rr, top) = larges[0];
    // y-UP page space back to raster rows.
    let rt = (h as i32 - top).max(0) as usize;
    let rb = (h as i32 - bot).max(0) as usize;
    let (l, rr) = (l.max(0) as usize, rr.max(0) as usize);
    println!("\n--- recognition experiment (raster x={l}..{rr} y={rt}..{rb}) ---");
    let pad = 6usize;
    recognize_crop(
        &r,
        dict.as_ref(),
        &grey,
        w,
        h,
        (
            l.saturating_sub(pad),
            rt.saturating_sub(pad),
            (rr + pad).min(w),
            (rb + pad).min(h),
        ),
        "the glyph ALONE (its own unit)",
    );
    // For contrast: the same vertical band INCLUDING the text to its right,
    // i.e. what widening a row crop leftward would produce.
    recognize_crop(
        &r,
        dict.as_ref(),
        &grey,
        w,
        h,
        (
            l.saturating_sub(pad),
            rt.saturating_sub(pad),
            (rr + 900).min(w),
            (rb + pad).min(h),
        ),
        "glyph + text to its right (a widened row crop)",
    );
    // And the text WITHOUT the glyph, the current behaviour, as the baseline.
    recognize_crop(
        &r,
        dict.as_ref(),
        &grey,
        w,
        h,
        (
            rr + pad,
            rt.saturating_sub(pad),
            (rr + 900).min(w),
            (rb + pad).min(h),
        ),
        "text only, glyph excluded (current behaviour)",
    );
}

// ---------------------------------------------------------------------------
// Experiment: CAN the dropped glyph be recognized at all, and how?
// ---------------------------------------------------------------------------
//
// A drop cap spans SEVERAL text lines by construction, so no single-line crop
// contains the whole glyph — which is why widening a row's crop (the fix that
// worked for line-final periods) cannot work here, and plausibly why real
// libtesseract only manages "Aitice": it crams a multi-line-tall glyph into a
// one-line band. This arm tests the alternative directly: recognize the blob
// ALONE, as its own unit, and print what comes back.
fn recognize_crop(
    r: &tesseract_ocr::LstmRecognizer,
    dict: Option<&tesseract_core::DictLite>,
    grey: &[u8],
    w: usize,
    h: usize,
    rect: (usize, usize, usize, usize), // raster l, t, r, b
    label: &str,
) {
    let (l, t, rr, b) = rect;
    let (cw, ch) = (rr.saturating_sub(l), b.saturating_sub(t));
    if cw == 0 || ch == 0 || rr > w || b > h {
        println!("  {label}: (crop out of range)");
        return;
    }
    let mut crop = vec![255u8; cw * ch];
    for y in 0..ch {
        crop[y * cw..(y + 1) * cw].copy_from_slice(&grey[(t + y) * w + l..(t + y) * w + l + cw]);
    }
    match r.recognize_page_makerow(&crop, cw, ch, dict) {
        Ok(text) => println!("  {label}: {:?}", text.trim()),
        Err(e) => println!("  {label}: ERROR {e:?}"),
    }
}
