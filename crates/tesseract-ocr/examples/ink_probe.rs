//! Proves the line-final-period loss end to end: the ink IS present, the
//! recognition crop cuts it off, and widening the crop recovers it.
//!
//! For each recognized line: scan right of the last recognized word (inside
//! the line band) for leftover ink; when found, re-recognize the line from a
//! crop widened just past that ink and report whether a period appears.
//!
//! ## Root cause this measures (`blob_filter.rs`)
//!
//! A period at book scale is **5-6 px tall**. `filter_blobs` pass 1 sends any
//! component with `height < TEXTORD_MAX_NOISE_SIZE` (**7**) to
//! `FilteredBlobs::noise` — and **nothing in this crate consumes `noise`**; it
//! is populated and dropped. So `make_rows` never sees the period, the row ink
//! bbox stops at the last full-height glyph, and `makerow_row_crops` (that
//! bbox + `kImagePadding = 4`) slices the period in half. Commas descend below
//! the baseline, clear the `h >= 7` bar, and land in `blobs` — which is why
//! commas survive at full count while periods vanish. Real libtesseract has the
//! same constant but RETAINS `TO_BLOCK::noise_blobs` and re-inserts them later;
//! this transcode ported the classification but not the re-insertion.
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]
use tesseract_core::DictLite;
use tesseract_ocr::{BinarizeMode, LstmRecognizer};
fn main() {
    let a: Vec<String> = std::env::args().collect();
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
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };
    let bytes = std::fs::read(&a[1]).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let otsu = tesseract_ocr::otsu_threshold_gray(&grey, w, 0, 0, w, h);
    let bin = tesseract_ocr::threshold_rect_to_binary(&grey, w, 0, 0, w, h, otsu);
    let doc = r
        .recognize_document_with_mode(&grey, w, h, dict.as_ref(), None, BinarizeMode::Otsu)
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();
    let (mut with_trailing_ink, mut total, mut recovered) = (0usize, 0usize, 0usize);
    for page in v["pages"].as_array().unwrap() {
        for reg in page["regions"].as_array().unwrap() {
            for line in reg["lines"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                let lb = line["bbox"].as_array().unwrap();
                let (lt, lb_) = (lb[1].as_i64().unwrap(), lb[3].as_i64().unwrap());
                let words = line["words"].as_array().unwrap();
                let Some(last) = words.last() else { continue };
                let wb = last["bbox"].as_array().unwrap();
                let wr = wb[2].as_i64().unwrap();
                total += 1;
                // scan a 40px window right of the last word, inside the band
                let (x0, x1) = (wr.max(0) as usize, ((wr + 40).max(0) as usize).min(w));
                let (y0, y1) = (lt.max(0) as usize, (lb_.max(0) as usize).min(h));
                let mut ink = 0usize;
                let (mut minx, mut maxx, mut miny, mut maxy) = (usize::MAX, 0, usize::MAX, 0);
                for y in y0..y1 {
                    for x in x0..x1 {
                        if bin[y * w + x] == 0 {
                            ink += 1;
                            minx = minx.min(x);
                            maxx = maxx.max(x);
                            miny = miny.min(y);
                            maxy = maxy.max(y);
                        }
                    }
                }
                if ink > 0 {
                    with_trailing_ink += 1;
                    // Re-recognize this line from a crop widened just past the
                    // leftover ink. If a period appears, the glyph was present
                    // all along and the CROP was the loss.
                    let ll = lb[0].as_i64().unwrap().max(0) as usize;
                    let (cx1, cy0, cy1) = ((maxx + 4).min(w), y0, y1);
                    let (cw, ch) = (cx1.saturating_sub(ll), cy1 - cy0);
                    let mut recovered_txt = String::new();
                    if cw > 0 && ch > 0 {
                        let mut crop = vec![255u8; cw * ch];
                        for yy in 0..ch {
                            crop[yy * cw..(yy + 1) * cw]
                                .copy_from_slice(&grey[(cy0 + yy) * w + ll..(cy0 + yy) * w + cx1]);
                        }
                        if let Ok((_, t)) = r.recognize_grey_line(&crop, cw, ch, dict.clone()) {
                            recovered_txt = t;
                        }
                    }
                    let gained = recovered_txt.trim_end().ends_with('.');
                    if gained {
                        recovered += 1;
                    }
                    if with_trailing_ink <= 8 {
                        let txt: String = words
                            .iter()
                            .filter_map(|x| x["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(" ");
                        let tail: String = txt
                            .chars()
                            .rev()
                            .take(16)
                            .collect::<String>()
                            .chars()
                            .rev()
                            .collect();
                        println!("  y={lt}..{lb_} last_word_right={wr}  ink {ink}px at x={minx}..{maxx} ({}x{})  ...{tail:?}  widened={}",
                            maxx-minx+1, maxy-miny+1, if gained {"PERIOD RECOVERED"} else {"no period"});
                    }
                }
            }
        }
    }
    println!(
        "\nlines={total}  lines_with_unrecognized_ink_right_of_last_word={with_trailing_ink}  \
of which a WIDENED crop recovers a period={recovered}"
    );
}
