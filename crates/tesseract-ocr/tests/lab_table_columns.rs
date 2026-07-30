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
//! **The vertical borders are recognized as glyphs.** Measured cell text:
//!
//! ```text
//! row 0: printed [Parameter | Ergebnis  | Einheit | Referenz]
//!        got     [Parameter | = ~—~—~*'| Ergebnis ==—|:«&Ejiinheit | …]
//! row 1: printed [Haemoglobin | 14.2    | g/dl    | 13.5 - 17.5]
//!        got     [Haemoglobin |   | 142  —   | g/dl | | 13.5 -17.5]
//! ```
//!
//! Those `|`, `=`, `—`, `‘` are the printed borders being read as characters.
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
//! the printed borders are already located. They are simply never subtracted before
//! recognition. Border stripping is the fix, and the masks for it already exist.
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

/// Recognize the fixture and return its `doc.v1` as parsed JSON. When
/// `strip` is set the grey page goes through
/// [`tesseract_ocr::strip_borders_page`] first — the opt-in pre-processing
/// step that paints printed borders out to background before the recognizer
/// ever sees them.
fn recognize_fixture_opt(strip: bool) -> Option<serde_json::Value> {
    let (r, dict) = load()?;
    let p = corpus().join("lab/lab_table_ruled.pgm");
    if !p.exists() {
        eprintln!("skipping: lab fixture absent (run corpus/gen/gen_lab_table.py)");
        return None;
    }
    let bytes = std::fs::read(&p).ok()?;
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).ok()?;
    let doc = r
        .recognize_document_with_options(
            &grey,
            w,
            h,
            dict.as_ref(),
            None,
            tesseract_ocr::DocumentOptions {
                strip_borders: strip,
                ..tesseract_ocr::DocumentOptions::default()
            },
        )
        .ok()?;
    serde_json::from_str(&doc.json).ok()
}

/// The NAIVE strip — applied to the grey page BEFORE calling in, which is
/// how `rectify` is wired and therefore the obvious first thing to try.
/// Kept because it is the falsifier for the coupling finding, not because it
/// is a path anyone should use.
fn recognize_fixture_naive_strip() -> Option<serde_json::Value> {
    let (r, dict) = load()?;
    let p = corpus().join("lab/lab_table_ruled.pgm");
    if !p.exists() {
        return None;
    }
    let bytes = std::fs::read(&p).ok()?;
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).ok()?;
    let grey = tesseract_ocr::strip_borders_page(&grey, w, h, tesseract_ocr::BinarizeMode::Otsu)
        .expect("strip_borders_page on the ruled fixture");
    let doc = r
        .recognize_document(&grey, w, h, dict.as_ref(), None)
        .ok()?;
    serde_json::from_str(&doc.json).ok()
}

/// The default (un-stripped) path — what every other test here measures.
fn recognize_fixture() -> Option<serde_json::Value> {
    recognize_fixture_opt(false)
}

/// Collect `(rows, cols, cells)` for every `table` region in a doc.v1 value.
fn table_shapes(v: &serde_json::Value) -> Vec<(u64, u64, usize)> {
    let mut out = Vec::new();
    for page in v["pages"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        for region in page["regions"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            if region["type"].as_str() == Some("table") {
                out.push((
                    region["rows"].as_u64().unwrap_or(0),
                    region["cols"].as_u64().unwrap_or(0),
                    region["cells"].as_array().map_or(0, Vec::len),
                ));
            }
        }
    }
    out
}

/// Dump a doc.v1's table grid to stderr, cell text included — so a failure
/// message carries WHICH boundary moved, not just a count.
fn dump_grid(label: &str, v: &serde_json::Value) {
    for page in v["pages"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        for region in page["regions"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
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
                eprintln!("  [{label}] row {r}: {}", row.join(" | "));
            }
        }
    }
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
    // the module docs. When border stripping lands, this fails; re-pin to
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

/// **The coupling finding, pinned — survived a real detour, restored to its
/// original form, and the detour is worth keeping on record.**
///
/// Measured: stripping the borders from the page BEFORE calling the
/// recognizer destroys table detection outright (3 recovered columns become
/// zero table regions). The cause is structural: two of `decide_if_table`'s
/// four score conditions (`nhb > 1`, `nvb > 2`) COUNT BORDERS, so a page
/// with its rules already removed can only score on the whitespace pair —
/// defect 1's own borderless regime.
///
/// **The printed borders are simultaneously what ruins the columns and what
/// proves it is a table.** That is why `DocumentOptions::strip_borders`
/// exists on the RECOGNIZER rather than as a caller pre-processing step:
/// only inside can detection read the original page while recognition reads
/// the stripped one.
///
/// # The detour (2026-07-30), recorded because the lesson generalizes
///
/// The ingredient-3 fix (`xy_cut_table_aware`) initially made classification
/// table-aware UNCONDITIONALLY — every `recognize_document` call, not just
/// `strip_borders`-opted-in ones. Under that design, naive pre-stripping
/// (this test's own `naive` variant, which calls PLAIN `recognize_document`)
/// briefly ALSO stopped destroying detection: classification stayed whole
/// instead of pre-fragmenting the table into narrow per-column blocks before
/// `decide_if_table` ever ran on any of them, and a full-width crop has far
/// more internal whitespace for `nvw` to count than a narrow column does —
/// enough to clear the threshold via `nvw` alone even with the rules gone.
///
/// **That "improvement" was a symptom of a real bug, not a fix.** The SAME
/// unconditional change broke `quality_resolution_grid.rs`'s 8+7+0 CER
/// fence: an ordinary 8-column TEXT grid (no table anywhere) ALSO produces
/// enough long whitespace corridors to clear `decide_if_table`'s borderless
/// path — the exact fragility `lab_table_grid.rs` already knew about,
/// firing on a case with no border signal to fall back on. Measured: 16
/// cells' worth of per-cell lines (~48) merged into 6 full-width readings —
/// the precise failure mode `recognize_page_blocks_words_with_mode` exists
/// to prevent.
///
/// **The fix: gate `xy_cut_table_aware` behind `opts.strip_borders`.** A
/// caller who has not opted into table-specific handling gets the plain,
/// proven `xy_cut` unchanged — protecting every existing
/// `recognize_document`/`recognize_document_with_mode` caller, including
/// this test's `naive` variant, which restores this test's ORIGINAL finding
/// exactly. **The general lesson:** `decide_if_table`'s borderless (`nvw`
/// only) path is not reliable enough to be trusted as an UNCONDITIONAL veto
/// against splitting ANY layout — it discriminates tables from wide
/// multi-column prose only about as well as the two ARE actually different
/// in whitespace-corridor count, which is not well. It is safe only when a
/// caller has already signalled "I specifically care about tables here."
///
/// This test exists so that reasoning stays falsifiable rather than asserted
/// in a doc comment. If naive pre-strip ever stops destroying detection
/// again, that is exactly the signal to re-check whether the same
/// unconditional-veto mistake has been reintroduced — re-pin only after
/// confirming the OTHER quality fences (`quality_resolution_grid.rs`
/// especially) are still green, not before.
#[test]
fn naive_pre_strip_destroys_table_detection() {
    let Some(plain) = recognize_fixture_opt(false) else {
        return;
    };
    let Some(naive) = recognize_fixture_naive_strip() else {
        return;
    };
    let plain_shapes = table_shapes(&plain);
    let naive_shapes = table_shapes(&naive);
    eprintln!("plain tables:      {plain_shapes:?}");
    eprintln!("naive-strip tables: {naive_shapes:?}");

    assert!(
        !plain_shapes.is_empty(),
        "precondition: the ruled fixture must be detected as a table at all \
         on the unmodified page, or this test measures nothing"
    );
    assert!(
        naive_shapes.is_empty(),
        "pre-stripping is EXPECTED to destroy detection ({naive_shapes:?}); \
         if a rule-independent detection path now exists AND \
         quality_resolution_grid.rs's 8+7+0 fence still passes, re-pin this \
         test deliberately — but if the fence broke too, this is the same \
         unconditional-veto mistake the detour above already made once"
    );
}

/// **`strip_borders` fixes the TEXT, preserves detection, AND — since
/// `xy_cut_table_aware` — keeps the table as ONE recognized region instead
/// of column-major-fragmenting it.** Measured, pinned, re-pinned honestly
/// after the ingredient-3 fix landed rather than silently edited.
///
/// **Ingredient 1 (border-glyph pollution) — fixed, measured as a relative
/// reduction, not an exact string.** An exact substring match on the OLD
/// measurement (`"13.5-17.5"`, no space) turned out to be sensitive to
/// incidental tokenization — recognizing the table as one full-width block
/// (this fix) vs the OLD per-column narrow crops changes exactly where
/// word boundaries land, so the SAME underlying improvement can read as
/// `"13.5 -17.5"` (with a space) instead. That is not a regression in the
/// property that actually matters: the literal rule-glyph characters
/// (`|`/`=`/`—`) that used to stand in for every printed border are what
/// this test measures, by COUNT, comparing stripped against plain on the
/// SAME fixture — robust to exactly where a boundary lands.
///
/// **Ingredient 2 (detection surviving) — fixed**, matches
/// `naive_pre_strip_no_longer_destroys_detection_but_relies_on_nvw_alone`'s
/// finding: `xy_cut_table_aware` fixed classification's own robustness as a
/// side effect, so this held even before the recognition-side fix landed.
///
/// **Ingredient 3 (full-width row recognition) — STRUCTURALLY fixed, with
/// one separate, narrower gap remaining.** The table no longer
/// column-fragments into `28×1`; it recognizes as one region spanning every
/// column, `xy_cut_table_aware`'s whole point. But the recovered grid is
/// `7×3`, not the printed `7×4` — `structured::extract_table_grid`'s OWN
/// whitespace-gap column splitter (a median-word-height heuristic, applied
/// to the WORDS the recognizer actually produced) still merges two of the
/// four real columns on this fixture. That is a DIFFERENT, narrower problem
/// than the one this test was built to catch — the fragmentation-before-
/// recognition defect this test pins is gone; a gap-threshold tuning
/// question in a different function is what is left, and belongs in its own
/// investigation rather than blocking this re-pin.
#[test]
fn stripping_borders_keeps_the_table_as_one_region_and_reduces_border_glyphs() {
    let Some(plain) = recognize_fixture_opt(false) else {
        return;
    };
    let Some(stripped) = recognize_fixture_opt(true) else {
        return;
    };

    let plain_shapes = table_shapes(&plain);
    let stripped_shapes = table_shapes(&stripped);
    eprintln!("plain    tables: {plain_shapes:?}");
    eprintln!("stripped tables: {stripped_shapes:?}");
    dump_grid("plain", &plain);
    dump_grid("stripped", &stripped);

    // Ingredient 2 held: detection survives stripping when it happens inside.
    assert!(
        !stripped_shapes.is_empty(),
        "stripping INSIDE the recognizer must keep the table detected — that \
         is the whole reason it is a DocumentOptions flag and not a caller \
         pre-processing step"
    );

    let table_text = |doc: &serde_json::Value| -> String {
        doc["pages"][0]["regions"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["type"].as_str() == Some("table"))
            .flat_map(|r| r["cells"].as_array().map(Vec::as_slice).unwrap_or(&[]))
            .filter_map(|c| c["text"].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let plain_text = table_text(&plain);
    let stripped_text = table_text(&stripped);
    let pipe_count = |s: &str| s.matches('|').count();
    eprintln!(
        "border-glyph '|' count: plain={} stripped={}",
        pipe_count(&plain_text),
        pipe_count(&stripped_text)
    );

    // Ingredient 1, measured as a relative reduction — robust to exactly
    // where the crop-shape-sensitive word boundaries land (see the doc
    // comment above for why an exact substring match was the wrong test).
    assert!(
        pipe_count(&stripped_text) < pipe_count(&plain_text),
        "stripping must measurably REDUCE the literal rule-glyph ('|') \
         count vs the un-stripped read; plain={:?} stripped={:?}",
        plain_text,
        stripped_text
    );
    assert!(
        stripped_text.contains("Referenz"),
        "the header must read cleanly once the printed borders are gone; got \
         {stripped_text:?}"
    );

    // Ingredient 3, STRUCTURAL half: the table stays ONE region (any column
    // count > 1), not 28×1 column-major fragmentation. This is the part
    // `xy_cut_table_aware` fixes and the part this test exists to guard.
    let stripped_cols = stripped_shapes.first().map_or(0, |s| s.1);
    assert!(
        stripped_cols > 1,
        "stripping must no longer leave a column-major 1-column reading \
         (xy_cut_table_aware regression); measured {stripped_cols} column(s): \
         {stripped_shapes:?}"
    );

    // Ingredient 3, REMAINING half — pinned two-sided, same discipline as
    // before: the grid is not yet the full printed 4 columns. When
    // `extract_table_grid`'s gap heuristic is tuned to separate the last
    // two columns, THIS assertion fails — that is the signal to re-pin to 4
    // and tell the extraction consumers (medcare-rs lab import, odoo-rs /
    // woa-rs invoice lines) that region["cells"] is fully usable, not a
    // signal that something broke.
    assert_eq!(
        stripped_cols, 3,
        "measured 3 of the printed 4 columns recovered (extract_table_grid's \
         gap heuristic still merges two columns on this fixture); if this \
         is now 4, the remaining gap has closed — re-pin deliberately, \
         do not just widen this assertion"
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
