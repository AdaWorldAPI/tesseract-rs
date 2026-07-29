//! Dump doc.v1 structure + per-line metrics for an arbitrary grey PGM.
//! Diagnostic only: reports regions, table grids, and the font-size spread
//! that drives renderer placement.
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (pgm, model) = (&a[1], a.get(2).map_or("eng", |s| s.as_str()));
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let r = LstmRecognizer::from_components(
        &std::fs::read(c.join(format!("{model}.lstm"))).unwrap(),
        &std::fs::read_to_string(c.join(format!("{model}.lstm-unicharset"))).unwrap(),
        &std::fs::read(c.join(format!("{model}.lstm-recoder"))).unwrap(),
    )
    .unwrap();
    let dict = match (
        std::fs::read(c.join(format!("{model}.lstm-word-dawg"))),
        std::fs::read(c.join(format!("{model}.lstm-punc-dawg"))),
        std::fs::read(c.join(format!("{model}.lstm-number-dawg"))),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };
    let bytes = std::fs::read(pgm).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    // ── char-box height reality check ───────────────────────────────────
    let lines = r
        .recognize_page_blocks_words(&grey, w, h, dict.as_ref())
        .unwrap();
    // INK-EXTENT measurement: for each char box's x-span, find the real
    // top/bottom of ink in the binarized page. This is the height char_boxes
    // do NOT carry.
    let otsu = tesseract_ocr::otsu_threshold_gray(&grey, w, 0, 0, w, h);
    let bin = tesseract_ocr::threshold_rect_to_binary(&grey, w, 0, 0, w, h, otsu);
    let ink_h = |x0: i32, x1: i32, ytop: i32, ybot: i32| -> Option<i32> {
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        for y in ytop.max(0)..ybot.min(h as i32) {
            for x in x0.max(0)..x1.min(w as i32) {
                if bin[y as usize * w + x as usize] == 0 {
                    lo = lo.min(y);
                    hi = hi.max(y);
                }
            }
        }
        (lo <= hi).then_some(hi - lo + 1)
    };
    let mut multi = 0usize;
    for (li, l) in lines.iter().enumerate().take(24) {
        let mut hs: Vec<i32> = Vec::new();
        let mut txt = String::new();
        for wd in &l.words {
            let ids: Vec<u32> = wd.unichar_ids.iter().map(|&i| i as u32).collect();
            txt.push_str(&tesseract_core::ids_to_text(&r.charset, &ids));
            txt.push(' ');
            for &(_, b, _, t) in &wd.char_boxes {
                hs.push((t - b).abs());
            }
        }
        let distinct: std::collections::BTreeSet<i32> = hs.iter().copied().collect();
        if distinct.len() > 1 {
            multi += 1;
        }
        // measured ink heights per char box, top-down page coords
        let mut inks: Vec<i32> = Vec::new();
        for wd in &l.words {
            for &(cl, _, cr, _) in &wd.char_boxes {
                let top = h as i32 - l.line_box.3;
                let bot = h as i32 - l.line_box.1;
                if let Some(ih) = ink_h(cl, cr, top, bot) {
                    inks.push(ih);
                }
            }
        }
        inks.sort_unstable();
        let pct = |q: f64| -> i32 {
            if inks.is_empty() {
                return 0;
            }
            inks[(((inks.len() - 1) as f64) * q).round() as usize]
        };
        println!(
            "   n={:3} med={:3} p75={:3} p90={:3} max={:3}",
            inks.len(),
            pct(0.5),
            pct(0.75),
            pct(0.90),
            inks.last().copied().unwrap_or(0)
        );
        println!(
            "line {li}: line_box_h={} | {} char_boxes, {} DISTINCT heights {:?} | {}",
            (l.line_box.3 - l.line_box.1).abs(),
            hs.len(),
            distinct.len(),
            distinct.iter().take(6).collect::<Vec<_>>(),
            txt.chars().take(30).collect::<String>()
        );
    }
    println!(
        "lines with >1 distinct char-box height: {}/{}\n",
        multi,
        lines.len().min(8)
    );

    let doc = r
        .recognize_document(&grey, w, h, dict.as_ref(), None)
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();

    let mut fonts: Vec<(f64, String)> = Vec::new();
    for page in v["pages"].as_array().unwrap() {
        println!("page {}x{}", page["width"], page["height"]);
        for reg in page["regions"].as_array().unwrap() {
            let kind = reg["type"].as_str().unwrap_or("?");
            let n = reg["lines"].as_array().map_or(0, Vec::len);
            print!("  region {kind} lines={n}");
            if let (Some(rr), Some(cc)) = (reg["rows"].as_u64(), reg["cols"].as_u64()) {
                print!(
                    "  GRID {rr}x{cc} cells={}",
                    reg["cells"].as_array().map_or(0, Vec::len)
                );
            }
            println!();
            for line in reg["lines"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                let t = line["text"].as_str().unwrap_or("");
                let words: String = line["words"]
                    .as_array()
                    .map(|ws| {
                        ws.iter()
                            .filter_map(|w| w["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_else(|| t.to_string());
                if let Some(gp) = line["glyph_px"].as_f64() {
                    println!(
                        "   GLYPH_PX {gp:6.1}   {}",
                        words.chars().take(34).collect::<String>()
                    );
                }
                if let Some(xh) = line["xheight"].as_f64() {
                    let (asc, desc) = (
                        line["ascrise"].as_f64().unwrap_or(0.0),
                        line["descdrop"].as_f64().unwrap_or(0.0),
                    );
                    let measured = xh + asc - desc;
                    // the OTHER formula the renderer uses when metrics are None
                    let bb = line["bbox"].as_array().unwrap();
                    let box_h = bb[3].as_f64().unwrap() - bb[1].as_f64().unwrap();
                    let guessed = box_h * 0.5;
                    fonts.push((
                        measured,
                        format!(
                            "guess={guessed:5.1} ratio={:.2}x  {}",
                            guessed / measured,
                            words.chars().take(30).collect::<String>()
                        ),
                    ));
                }
            }
        }
    }
    fonts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let med = fonts[fonts.len() / 2].0;
    println!(
        "\nper-line font_px (xheight+ascrise-descdrop), median={med:.1}, n={}",
        fonts.len()
    );
    for (f, t) in fonts.iter().take(3) {
        println!("   MIN {f:6.1}  ({:.2}x median)  {t}", f / med);
    }
    for (f, t) in fonts.iter().rev().take(3) {
        println!("   MAX {f:6.1}  ({:.2}x median)  {t}", f / med);
    }
}
