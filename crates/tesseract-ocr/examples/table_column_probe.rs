//! Why does `extract_table_grid` merge two columns? Dump the whitespace
//! rivers it actually sees, against the bar it actually applies.
//!
//! Diagnostic only. The question this exists to answer, before anyone tunes a
//! threshold: is the missing column boundary a river that EXISTS and merely
//! failed the width bar (a threshold problem, tunable), or a river that is
//! genuinely ABSENT because recognition merged words across it (a recognition
//! problem, where no threshold helps)? Those two have opposite fixes, and the
//! grid shape alone cannot distinguish them.
//!
//! Reproduces `extract_table_grid`'s river computation — same `gap_rows`
//! occupancy scan, same `support` majority, same `gap_min` bar — and prints
//! every candidate with its measured width so the binding constraint is
//! visible rather than inferred.
//!
//! **It deliberately shows the RAW, UN-BRIDGED rivers**, i.e. the state
//! before `extract_table_grid`'s fragment-bridging step. That is the whole
//! diagnostic value: the fragmentation is exactly what the raw view exposes
//! and the final grid hides. So the probe's own `=> N columns` line reports
//! the pre-bridging count and will read LOWER than the `region ... cols=`
//! line above it (which is the real emitted grid) wherever bridging fired —
//! that divergence is the signal, not a bug.
//!
//! ```sh
//! cargo run -p tesseract-ocr --release --example table_column_probe -- \
//!     corpus/lab/lab_table_ruled.pgm [strip]
//! ```
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_core::DictLite;
use tesseract_ocr::{DocumentOptions, LstmRecognizer};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pgm = &a[1];
    let strip = a.iter().any(|s| s == "strip");

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

    let bytes = std::fs::read(pgm).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let doc = r
        .recognize_document_with_options(
            &grey,
            w,
            h,
            dict.as_ref(),
            None,
            DocumentOptions {
                strip_borders: strip,
                ..DocumentOptions::default()
            },
        )
        .unwrap();

    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();
    println!("== {pgm}  strip_borders={strip} ==");

    for page in v["pages"].as_array().unwrap() {
        for (ri, region) in page["regions"].as_array().unwrap().iter().enumerate() {
            if region["type"].as_str() != Some("table") {
                continue;
            }
            let lines = region["lines"].as_array().unwrap();
            println!(
                "\nregion {ri}: type=table rows={} cols={} lines={}",
                region["rows"],
                region["cols"],
                lines.len()
            );

            // --- replicate extract_table_grid's river scan exactly -------
            let mut words: Vec<(i32, i32, i32, i32, String)> = Vec::new();
            for line in lines {
                for wd in line["words"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                    let b = wd["bbox"].as_array().unwrap();
                    words.push((
                        b[0].as_i64().unwrap() as i32,
                        b[1].as_i64().unwrap() as i32,
                        b[2].as_i64().unwrap() as i32,
                        b[3].as_i64().unwrap() as i32,
                        wd["text"].as_str().unwrap_or("").to_string(),
                    ));
                }
            }
            if words.is_empty() {
                println!("  (no words)");
                continue;
            }
            let x0 = words.iter().map(|t| t.0).min().unwrap();
            let x1 = words.iter().map(|t| t.2).max().unwrap();
            let mut heights: Vec<i32> = words.iter().map(|t| (t.3 - t.1).max(1)).collect();
            heights.sort_unstable();
            let med_h = heights[heights.len() / 2].max(1) as usize;
            let gap_min = 2 * med_h;
            let width = (x1 - x0).max(1) as usize;
            let n_rows = lines.len() as u32;
            let support = if n_rows <= 1 { 1 } else { n_rows / 2 + 1 };

            println!(
                "  x0={x0} x1={x1} width={width}  med_word_h={med_h}  \
                 gap_min=2*med_h={gap_min}  rows={n_rows} support>={support}"
            );

            let mut gap_rows = vec![0u32; width];
            for line in lines {
                let mut covered = vec![false; width];
                for wd in line["words"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                    let b = wd["bbox"].as_array().unwrap();
                    let a0 = (b[0].as_i64().unwrap() as i32 - x0).clamp(0, x1 - x0) as usize;
                    let b0 = (b[2].as_i64().unwrap() as i32 - x0).clamp(0, x1 - x0) as usize;
                    for cv in covered.iter_mut().take(b0).skip(a0) {
                        *cv = true;
                    }
                }
                for (x, &cv) in covered.iter().enumerate() {
                    if !cv {
                        gap_rows[x] += 1;
                    }
                }
            }

            // Every maximal run where >= support rows are whitespace.
            println!("  --- candidate rivers (>= {support} rows whitespace) ---");
            let mut start: Option<usize> = None;
            let mut accepted = 0usize;
            let mut rejected: Vec<(usize, usize)> = Vec::new();
            for (x, &g) in gap_rows.iter().enumerate() {
                if g < support {
                    if let Some(s) = start.take() {
                        let wdt = x - s;
                        let ok = wdt >= gap_min;
                        if ok {
                            accepted += 1;
                        } else {
                            rejected.push((s, wdt));
                        }
                        // The river width is the INTERSECTION of the rows'
                        // gaps. Also measure, at this river's midpoint, how
                        // wide the gap is IN EACH ROW that is blank there --
                        // a ragged (e.g. right-aligned numeric) column has
                        // per-row gaps far wider than their common core, so
                        // the two measures can disagree sharply. Which one a
                        // 2*med_h bar is calibrated for is the open question.
                        let mid = (s + x) / 2;
                        let mut per_row: Vec<usize> = Vec::new();
                        for line in lines {
                            let mut lo = 0usize;
                            let mut hi = width;
                            let mut blank = true;
                            for wd in line["words"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                                let b = wd["bbox"].as_array().unwrap();
                                let a0 =
                                    (b[0].as_i64().unwrap() as i32 - x0).clamp(0, x1 - x0) as usize;
                                let b0 =
                                    (b[2].as_i64().unwrap() as i32 - x0).clamp(0, x1 - x0) as usize;
                                if a0 <= mid && mid < b0 {
                                    blank = false;
                                    break;
                                }
                                if b0 <= mid {
                                    lo = lo.max(b0);
                                }
                                if a0 > mid {
                                    hi = hi.min(a0);
                                }
                            }
                            if blank {
                                per_row.push(hi.saturating_sub(lo));
                            }
                        }
                        per_row.sort_unstable();
                        let med_row_gap = per_row.get(per_row.len() / 2).copied().unwrap_or(0);
                        println!(
                            "    river x={:5}..{:<5} width={:4}  {}  (needs {gap_min})  \
                             per-row gaps n={} median={}  [{}]",
                            s + x0 as usize,
                            x + x0 as usize,
                            wdt,
                            if ok { "ACCEPT" } else { "reject" },
                            per_row.len(),
                            med_row_gap,
                            per_row
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                    }
                } else if start.is_none() {
                    start = Some(x);
                }
            }
            if let Some(s) = start {
                let wdt = width - s;
                println!(
                    "    river x={:5}..{:<5} width={:4}  (trailing, not a cut)",
                    s + x0 as usize,
                    width + x0 as usize,
                    wdt
                );
            }
            println!(
                "  => {accepted} interior cuts accepted -> {} columns",
                accepted + 1
            );
            if !rejected.is_empty() {
                let widest = rejected.iter().map(|t| t.1).max().unwrap();
                println!(
                    "  NOTE {} river(s) rejected on width; widest rejected = {widest} \
                     ({:.2}x the {gap_min} bar). If a real column boundary is among \
                     these, the bar is the binding constraint. If NOT, the boundary \
                     has no river at all and recognition merged across it.",
                    rejected.len(),
                    widest as f32 / gap_min as f32
                );
            }

            // Per-row word spans: shows directly whether any single word
            // straddles a printed column boundary.
            println!("  --- per-row word x-spans ---");
            for (li, line) in lines.iter().enumerate() {
                let spans: Vec<String> = line["words"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                    .iter()
                    .map(|wd| {
                        let b = wd["bbox"].as_array().unwrap();
                        format!(
                            "[{}..{}]{:?}",
                            b[0].as_i64().unwrap(),
                            b[2].as_i64().unwrap(),
                            wd["text"].as_str().unwrap_or("")
                        )
                    })
                    .collect();
                println!("    row {li}: {}", spans.join(" "));
            }
        }
    }
}
