//! Does SHRINKING an aggrandized paragraph-initial and gluing it to the line
//! beat reading it separately?
//!
//! Diagnostic. Three arms on the SAME line:
//!   A. today  — line without the cap (the cap's ink is discarded upstream)
//!   B. glue   — cap downscaled to the body band, butted against the line,
//!               ONE recognition pass over normal geometry
//!   C. split  — cap recognized as its own unit, spliced textually
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_core::DictLite;
use tesseract_ocr::{conn_comp_areas, filter_blobs, LstmRecognizer};

/// Nearest-neighbour box downscale — deliberately the crudest resampler, so a
/// PASS here is a lower bound on what a real filter would achieve.
fn downscale(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![255u8; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            out[y * dw + x] = src[(y * sh / dh).min(sh - 1) * sw + (x * sw / dw).min(sw - 1)];
        }
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let r = LstmRecognizer::from_components(
        &std::fs::read(c.join("eng.lstm")).unwrap(),
        &std::fs::read_to_string(c.join("eng.lstm-unicharset")).unwrap(),
        &std::fs::read(c.join("eng.lstm-recoder")).unwrap(),
    )
    .unwrap();
    let dict = match (
        std::fs::read(c.join("eng.lstm-word-dawg")),
        std::fs::read(c.join("eng.lstm-punc-dawg")),
        std::fs::read(c.join("eng.lstm-number-dawg")),
    ) {
        (Ok(x), Ok(p), Ok(n)) => DictLite::from_components(&x, &p, &n).ok(),
        _ => None,
    };

    // ── DETECT: the "chaoda" outlier — how many sigma is the cap? ──────────
    let binary =
        tesseract_ocr::xy_cut::binarize_page_with(&grey, w, h, tesseract_ocr::BinarizeMode::Otsu);
    let mut comps = conn_comp_areas(&binary, w, h, 8);
    for cc in &mut comps {
        cc.bb.y = h as i32 - (cc.bb.y + cc.bb.h);
    }
    let f = filter_blobs(&comps);
    let hs: Vec<f32> = f.blobs.iter().map(|b| (b.3 - b.1).abs() as f32).collect();
    let mean = hs.iter().sum::<f32>() / hs.len() as f32;
    let sd = (hs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hs.len() as f32).sqrt();
    let cap = *f
        .large
        .iter()
        .max_by_key(|b| (b.3 - b.1).abs())
        .expect("a large blob");
    let (cl, cbot, cr, ctop) = cap;
    let caph = (ctop - cbot).abs() as f32;
    println!(
        "ordinary glyph height: mean={mean:.2} sd={sd:.2}  (n={})",
        hs.len()
    );
    println!(
        "drop cap {}x{} -> {:.2} SD above the mean   <-- the detector\n",
        (cr - cl).abs(),
        caph as i32,
        (caph - mean) / sd
    );

    // raster coords
    let (rt, rb) = (
        (h as i32 - ctop).max(0) as usize,
        (h as i32 - cbot).max(0) as usize,
    );
    let (cl, cr) = (cl.max(0) as usize, cr.max(0) as usize);
    // The cap spans ~3 line-heights, so its OWN y-range is the right window:
    // makerow finds the rows inside it. A hardcoded band slices through the
    // cap and line 1 and measures nothing (the first version of this probe
    // did exactly that and its control failed to reproduce the baseline).
    let band = rb - rt;
    let line_r = (cr + 900).min(w);

    let crop = |l: usize, t: usize, rr: usize, b: usize| {
        let (cw, ch) = (rr - l, b - t);
        let mut v = vec![255u8; cw * ch];
        for y in 0..ch {
            v[y * cw..(y + 1) * cw].copy_from_slice(&grey[(t + y) * w + l..(t + y) * w + l + cw]);
        }
        (v, cw, ch)
    };
    let say = |label: &str, (v, cw, ch): (Vec<u8>, usize, usize)| match r.recognize_page_makerow(
        &v,
        cw,
        ch,
        dict.as_ref(),
    ) {
        Ok(t) => println!("  {label:<34} {:?}", t.trim()),
        Err(e) => println!("  {label:<34} ERROR {e:?}"),
    };

    // ── A. today, swept over the seam offset ─────────────────────────────
    println!("--- A. today (cap discarded), seam sweep ---");
    for off in [-8i32, -4, 0, 6, 12] {
        let l = (cr as i32 + off).max(0) as usize;
        say(
            &format!("line from cr{off:+}"),
            crop(l, rt, line_r, rt + band),
        );
    }

    // ── C. split ──────────────────────────────────────────────────────────
    println!("--- C. split (cap as its own unit) ---");
    say("cap alone", crop(cl, rt, cr + 6, rb));

    // ── B. glue: shrink the cap into the body band, butt it against the line
    println!("--- B. glue (cap shrunk to the body band, ONE pass) ---");
    let (capimg, capw, caph2) = crop(cl, rt, cr, rb);
    // Shrink to ONE body line, not to the whole 3-line window.
    let target_h = (mean * 1.4) as usize;
    let target_w = (capw * target_h / caph2).max(1);
    let small = downscale(&capimg, capw, caph2, target_w, target_h);
    let (line, lw, lh) = crop(cr - 8, rt, line_r, rt + band);
    let gap = 6usize;
    let (gw, gh) = (target_w + gap + lw, lh);
    let mut glued = vec![255u8; gw * gh];
    // baseline-align: the shrunk cap sits ON the body baseline.
    // Sit the shrunk cap on the FIRST line's baseline (top of the window),
    // not the bottom of the 3-line block.
    let cap_top = 2usize;
    for y in 0..target_h.min(gh - cap_top) {
        glued[(cap_top + y) * gw..(cap_top + y) * gw + target_w]
            .copy_from_slice(&small[y * target_w..(y + 1) * target_w]);
    }
    for y in 0..lh {
        let dst = y * gw + target_w + gap;
        glued[dst..dst + lw].copy_from_slice(&line[y * lw..(y + 1) * lw]);
    }
    say("glued", (glued, gw, gh));
}
