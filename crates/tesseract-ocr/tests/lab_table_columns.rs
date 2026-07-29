//! **Defect-2 regression: a four-column table collapses to one column.**
//!
//! Companion to `lab_table_grid.rs`, which covers defect 1 (a *borderless*
//! table is not classified a table at all). This one covers the more dangerous
//! defect, because **it looks like success**: a ruled table IS classified,
//! cells ARE produced, confidence reads high — and every row is one
//! undifferentiated blob a consumer would parse happily into garbage.
//!
//! # Why this needs its own fixture
//!
//! `lab_table_grid.rs` draws glyph-shaped ink marks. That is exactly right for
//! `decide_if_table`, a morphology leaf that only ever sees ink. But those
//! marks do not RECOGNIZE — they yield no words, and
//! `structured::extract_table_grid` splits columns by the whitespace gaps
//! *between recognized words*. Measuring the splitter therefore needs a
//! fixture the LSTM can actually read, which is what
//! `corpus/gen/gen_lab_table.py` renders (real DejaVu text, four columns,
//! ruled so it is classified in the first place).
//!
//! The gutters in that fixture are ~90 px against a ~26 px glyph height —
//! several times the "one median word-height" the splitter looks for. That is
//! deliberate: it removes the "the columns were just too close together"
//! explanation, so a collapse is unambiguously the splitter's behaviour and
//! not the fixture's geometry.
//!
//! # What the measurement actually found — NOT what was first reported
//!
//! The defect was first described as "a ruled table collapses to ONE column".
//! On this fixture it does not: it recovers **3 of 4**, and the mechanism is
//! different and more specific than a broken splitter.
//!
//! **The vertical rules are recognized as glyphs.** Measured cell text:
//!
//! ```text
//! row 0: printed [Parameter | Ergebnis  | Einheit | Referenz]
//!        got     [Parameter | = ~—~—~*'| Ergebnis ==—|:«&Ejiinheit | …]
//! row 1: printed [Haemoglobin | 14.2    | g/dl    | 13.5 - 17.5]
//!        got     [Haemoglobin |   | 142  —   | g/dl | | 13.5 -17.5]
//! ```
//!
//! Those `|`, `=`, `—`, `‘` are the printed rules being read as characters.
//! They do two things: corrupt the cell text, and — the load-bearing part —
//! **fill the inter-column gutters with "words"**. `extract_table_grid` splits
//! on whitespace gaps between recognized words, so once a rule occupies the
//! gutter there is no gap left to split on. The splitter is not broken; its
//! input is polluted.
//!
//! Column 0 recovers cleanly (7/7 parameter names) and is asserted below as
//! the control: recognition itself is fine, so the fault is specific to
//! rule-adjacent columns.
//!
//! **The actionable consequence:** `decide_if_table` ALREADY computes the
//! horizontal- and vertical-line masks (that is how it scores `nhb`/`nvb`), so
//! the rules are already located. They are simply never subtracted before
//! recognition. Rule-stripping is the fix, and the masks for it already exist.
//!
//! # This test pins a DEFECT, and is two-sided on purpose
//!
//! The assertions below record what the pipeline *currently does*, not what it
//! should do. If the column splitter is fixed, this test **fails** — and that
//! failure is the intended signal: come here, re-pin to the correct value, and
//! tell the table-extraction consumers (medcare-rs lab import, odoo-rs /
//! woa-rs invoice lines) that `region["cells"]` became usable. A one-sided
//! `assert!(cols >= 1)` would silently keep passing through the fix and tell
//! nobody.

use std::path::Path;

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Load the eng recognizer + dict, or `None` when the corpus model is absent
/// (the skip pattern every corpus-dependent test in this repo uses).
fn load() -> Option<(LstmRecognizer, Option<DictLite>)> {
    let c = corpus();
    if !c.join("model/eng.lstm").exists() {
        eprintln!("skipping: corpus model absent");
        return None;
    }
    let lstm = std::fs::read(c.join("model/eng.lstm")).ok()?;
    let uni = std::fs::read_to_string(c.join("model/eng.lstm-unicharset")).ok()?;
    let rec = std::fs::read(c.join("model/eng.lstm-recoder")).ok()?;
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).ok()?;
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

/// Recognize the fixture and return its `doc.v1` as parsed JSON.
fn recognize_fixture() -> Option<serde_json::Value> {
    let (r, dict) = load()?;
    let p = corpus().join("lab/lab_table_ruled.pgm");
    if !p.exists() {
        eprintln!("skipping: lab fixture absent (run corpus/gen/gen_lab_table.py)");
        return None;
    }
    let bytes = std::fs::read(&p).ok()?;
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).ok()?;
    let doc = r
        .recognize_document(&grey, w, h, dict.as_ref(), None)
        .ok()?;
    serde_json::from_str(&doc.json).ok()
}

/// The measurement. Reports the full grid shape before asserting, so a
/// failure message carries the numbers rather than only the verdict.
#[test]
fn ruled_table_column_split_is_measured_and_pinned() {
    let Some(v) = recognize_fixture() else { return };

    let gt: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(corpus().join("lab/lab_table_ruled.gt.json")).unwrap(),
    )
    .unwrap();
    let gt_rows = gt["rows"].as_u64().unwrap();
    let gt_cols = gt["cols"].as_u64().unwrap();

    let mut tables = Vec::new();
    for page in v["pages"].as_array().unwrap() {
        for region in page["regions"].as_array().unwrap() {
            if region["type"].as_str() == Some("table") {
                let rows = region["rows"].as_u64().unwrap_or(0);
                let cols = region["cols"].as_u64().unwrap_or(0);
                let cells = region["cells"].as_array().map_or(0, Vec::len);
                tables.push((rows, cols, cells));
            }
        }
    }

    eprintln!(
        "ground truth: {gt_rows} rows x {gt_cols} cols; \
         detected table regions: {}",
        tables.len()
    );
    for (i, (rows, cols, cells)) in tables.iter().enumerate() {
        eprintln!("  table {i}: rows={rows} cols={cols} cells={cells}");
    }

    // Dump the recovered grid against the printed one. A bare column COUNT
    // says a boundary was lost; this says WHICH — the difference between
    // "the splitter is broken" and "these two specific columns merge", which
    // is the difference between a vague finding and an actionable one.
    for page in v["pages"].as_array().unwrap() {
        for region in page["regions"].as_array().unwrap() {
            if region["type"].as_str() != Some("table") {
                continue;
            }
            let mut grid: Vec<Vec<String>> = Vec::new();
            for cell in region["cells"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                let r = cell["row"].as_u64().unwrap_or(0) as usize;
                let c = cell["col"].as_u64().unwrap_or(0) as usize;
                while grid.len() <= r {
                    grid.push(Vec::new());
                }
                while grid[r].len() <= c {
                    grid[r].push(String::new());
                }
                grid[r][c] = cell["text"].as_str().unwrap_or_default().to_string();
            }
            for (r, row) in grid.iter().enumerate() {
                let printed = gt["cells"][r]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|s| s.as_str().unwrap_or_default())
                            .collect::<Vec<_>>()
                            .join(" | ")
                    })
                    .unwrap_or_default();
                eprintln!("  row {r}: printed [{printed}]");
                eprintln!("           got     [{}]", row.join(" | "));
            }
        }
    }

    assert!(
        !tables.is_empty(),
        "the RULED fixture must be classified a table at all — if this fails \
         the defect is upstream in decide_if_table, not in the column \
         splitter, and lab_table_grid.rs is the test that covers it"
    );

    let (rows, cols, cells) = tables[0];
    assert!(
        cells > 0,
        "a classified table must produce cells; got a grid of {rows}x{cols} \
         with none"
    );

    // ── the pinned defect ────────────────────────────────────────────────
    // Measured: 3 recovered columns against 4 printed. Pinned two-sided — see
    // the module docs. When rule-stripping lands, this fails; re-pin to
    // `assert_eq!(cols, gt_cols)` and notify the extraction consumers.
    assert!(
        cols < gt_cols,
        "column recovery appears FIXED: measured {cols} columns against \
         {gt_cols} printed. This test pinned a known defect; if that is now \
         corrected, re-pin to assert_eq!(cols, {gt_cols}) and tell the \
         table-extraction consumers (medcare-rs lab import, odoo-rs / woa-rs \
         invoice lines) that region[\"cells\"] became usable."
    );

    // ── the CONTROL that localizes the defect ────────────────────────────
    // Column 0 is bounded on its right by a vertical rule but has a wide,
    // word-dominated span; it recovers CLEANLY (7/7 parameter names). That is
    // what proves recognition itself is fine and the fault is specific to
    // rule-adjacent columns — without it, "3 of 4 columns" is equally
    // consistent with the recognizer simply being bad on this fixture.
    let col0: Vec<String> = v["pages"][0]["regions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["type"].as_str() == Some("table"))
        .flat_map(|r| r["cells"].as_array().map(Vec::as_slice).unwrap_or(&[]))
        .filter(|c| c["col"].as_u64() == Some(0))
        .map(|c| c["text"].as_str().unwrap_or_default().trim().to_string())
        .collect();
    let printed_col0: Vec<String> = gt["cells"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row[0].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        col0, printed_col0,
        "column 0 must recover cleanly — it is the control proving the \
         recognizer works on this fixture, so the lost boundary is a \
         rule-adjacency problem and not general recognition failure"
    );
}

/// The honesty check that makes the defect *dangerous* rather than merely
/// wrong: the collapsed grid still reports high per-cell confidence. A
/// consumer gating on confidence alone would import the garbage.
///
/// This is the same shape as the `mean_conf 99.47` / `CER 0.6154` pair
/// measured elsewhere in this repo — confident, structured, and wrong. It is
/// asserted here so the pairing is recorded as a property of the system, not
/// an anecdote in a commit message.
#[test]
fn collapsed_cells_still_report_high_confidence() {
    let Some(v) = recognize_fixture() else { return };

    let mut confs = Vec::new();
    for page in v["pages"].as_array().unwrap() {
        for region in page["regions"].as_array().unwrap() {
            if region["type"].as_str() != Some("table") {
                continue;
            }
            for cell in region["cells"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(c) = cell["conf"].as_f64() {
                    confs.push(c);
                }
            }
        }
    }

    if confs.is_empty() {
        eprintln!("no table cells with confidence — nothing to assert");
        return;
    }
    let min = confs.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = confs.iter().sum::<f64>() / confs.len() as f64;
    eprintln!(
        "cell confidence: n={} min={min:.2} mean={mean:.2}",
        confs.len()
    );

    assert!(
        mean > 50.0,
        "the point of this test is that a WRONG grid still looks trustworthy \
         (measured mean {mean:.2}); if confidence has dropped to reflect the \
         collapse, that is a real improvement — re-pin this deliberately"
    );
}
