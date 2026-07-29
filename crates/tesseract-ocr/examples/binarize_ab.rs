//! **The falsifying A/B probe** — the whole local-adaptive ladder (Otsu →
//! Sauvola → Wolf-Jolion → Singh) under TWO INDEPENDENT degradations, the
//! failure modes a local mean/std is supposed to survive and a single
//! global Otsu threshold cannot (see `binarize.rs`'s module docs and the
//! `xy_cut::BinarizeMode` doc comments):
//!
//! - **Uneven illumination** (`uneven_*.pgm`, `gen_uneven_light.py`) — a
//!   multiplicative field that shifts brightness but preserves LOCAL
//!   contrast. Sauvola's home turf.
//! - **Faded contrast** (`faded_*.pgm`, `gen_faded_contrast.py`, ADDED
//!   2026-07-29) — a uniform compression of the whole page's dynamic range
//!   toward mid-grey, which lowers local std EVERYWHERE rather than in one
//!   region. This is what Wolf-Jolion's own claim is actually about: Sauvola's
//!   FIXED `R = 128` denominator collapses when `s << 128` uniformly, and
//!   Wolf's `s/max_s` normalization is built specifically to survive that.
//!   The first run of this probe (uneven-only) found Wolf byte-identical to
//!   Sauvola on every fixture — because uneven illumination never exercises
//!   this failure mode at all. See `.claude/harvest/binarization-roadmap.md`
//!   for the full account of that gap and why `faded_*` closes it.
//!
//! The existing rendered corpus (`corpus/pages/*.pgm`,
//! `corpus/quality/resgrid.pgm`) is cleanly and evenly lit at full contrast,
//! so neither degradation is visible there; both generators start from the
//! same clean corpus page so `uneven_clean.pgm`'s Otsu recognition remains
//! the one CER reference for every fixture in both families.
//!
//! ```sh
//! python3 corpus/gen/gen_uneven_light.py     # writes corpus/quality/uneven_*.pgm
//! python3 corpus/gen/gen_faded_contrast.py   # writes corpus/quality/faded_*.pgm
//! cargo run -p tesseract-ocr --example binarize_ab
//! ```
//!
//! ## History — the gap this probe found, and its fix
//!
//! The first run of this probe found that
//! [`LstmRecognizer::recognize_document_with_mode`]'s `binarize_mode`
//! parameter reached ONLY the region/table classification pass
//! (`region_figures` / `block_is_table` / the layout `xy_cut` call feeding
//! `build_regions`) — **not** the actual text-line finding/recognition
//! (`recognize_page_blocks_words`, which ran its own separate, always-Otsu
//! binarization via `segment::segment_rows`, untouched by this parameter).
//! `Document::word_count`, `Document::line_count`, and
//! `Document::mean_confidence` were therefore computed from a `page` value
//! built BEFORE `binarize_mode` was ever consulted — provably
//! mode-independent for any input, by construction. That result is recorded
//! in `.claude/harvest/sauvola-vs-otsu-probe.md`: every `word_count` /
//! `mean_conf` / `cer` cell read IDENTICAL for Otsu vs Sauvola on every
//! fixture, which was a measured falsifying result, not a bug in the probe
//! — it meant the wiring at the time could not make Sauvola improve
//! recognized TEXT quality on an unevenly-lit page, only how such a page's
//! regions got CLASSIFIED.
//!
//! **That gap is now closed.** `binarize_mode` is threaded through
//! `crate::segment::segment_rows_with_mode` /
//! `segment_rows_independent_with_mode` and up through
//! `recognize_page_blocks_words_with_mode` /
//! `recognize_page_makerow_words_with_mode`, so
//! `recognize_document_with_mode`'s mode now governs the SAME binarization
//! the line finder and word/line recognizer use — not just region/table
//! classification. The `word_count` / `mean_conf` / `cer` columns below can
//! therefore legitimately differ between Otsu and Sauvola now. **Whether
//! they actually do, and by how much, is exactly what this run of the probe
//! measures — this file does not assert an expected outcome; it prints the
//! numbers and lets them decide.**
//!
//! ## What this probe can and cannot settle for Wolf and Singh
//!
//! Sauvola is a byte-parity leaf; Wolf and Singh are not, and no oracle
//! exists for either (leptonica implements neither). **This probe is the
//! only evidence they have** — which makes it worth being precise about what
//! it establishes. It measures whether each method, at its own documented
//! default `k`, recovers text a global threshold destroys on THESE fixtures.
//! It does NOT establish that either implementation matches its reference; a
//! transcription error that happened to still binarize sensibly would show up
//! here as a merely-mediocre row, not as a failure. Read the numbers as
//! "does this rung help on this degradation", never as "is this rung
//! correct".
//!
//! The `mode_delta_cer` column makes the direct, per-fixture comparison
//! explicit: it is the CER between a fixture's OWN Otsu-mode text and its
//! OWN text under each later mode (Otsu is `MODES[0]`, so it has no prior
//! mode on a fixture to diff against and reports `-`). This isolates "did switching
//! binarization mode change what was recognized on this input" from "how
//! much did the degradation itself cost", which is what the pre-existing
//! `cer` column measures (against the clean-page/Otsu reference).
//!
//! The `ink_frac` column (computed directly from
//! [`binarize_page_with`](tesseract_ocr::xy_cut::binarize_page_with),
//! entirely independent of the OCR pipeline) remains the lowest-level
//! signal: the fraction of page pixels classified as ink, which a global
//! Otsu split inflates sharply once the dimmed region's background drops
//! toward — or past — the point the WHOLE page's histogram places its one
//! threshold, while Sauvola holds near the clean page's own fraction
//! throughout.

use std::fs;
use std::path::{Path, PathBuf};

use tesseract_core::DictLite;
use tesseract_ocr::xy_cut::binarize_page_with;
use tesseract_ocr::{parse_pgm, BinarizeMode, LstmRecognizer};

/// The corpus root, resolved relative to this crate's manifest dir — the
/// same convention `golden_bench.rs` / `golden_report.rs` use.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Prints a diagnostic to stderr and exits 1 — a measurement that silently
/// produced a partial/misleading number would be worse than one that stops
/// (mirroring `golden_bench.rs::fail`).
fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    std::process::exit(1);
}

/// The fixtures this probe measures, in report order: (label, filename under
/// `corpus/quality/`, from `gen_uneven_light.py` or `gen_faded_contrast.py`
/// — see the module docs for which generator owns which fixture family).
/// `clean` MUST be first — its own Otsu recognition is the CER reference
/// text for every row, in BOTH families (per the brief: "the clean page's
/// Otsu reading is the reference text — that is the ground truth here, not
/// an external transcript").
const FIXTURES: &[(&str, &str)] = &[
    ("clean", "uneven_clean.pgm"),
    ("linear_060", "uneven_linear_060.pgm"),
    ("linear_085", "uneven_linear_085.pgm"),
    ("vignette_060", "uneven_vignette_060.pgm"),
    ("vignette_085", "uneven_vignette_085.pgm"),
    // Faded contrast (gen_faded_contrast.py) — a DIFFERENT axis of
    // degradation from everything above: uniform dynamic-range compression
    // rather than a spatial illumination field. This is the fixture family
    // that can actually exercise Wolf-Jolion's claim (see the module docs).
    // `_060`/`_085` mean the same "fraction of range lost" as the uneven_*
    // tags, for direct magnitude comparison across the two families.
    ("faded_060", "faded_060.pgm"),
    ("faded_085", "faded_085.pgm"),
];

/// The modes under test — the whole Niblack ladder this crate implements.
/// The window (`whsize = 16`) comfortably fits the 512x720 source page (well
/// above its `2*whsize+3` minimum) and is held CONSTANT across modes so the
/// comparison isolates the closing formula, which is the only thing that
/// differs between them.
///
/// **`k` is deliberately NOT constant, and must not be made so.** It is not
/// transferable between these methods — the reference implementation warns
/// about exactly this, and Niblack famously needs a *negative* `k` where the
/// others need a positive one. Each value below is that method's own
/// documented default: Sauvola `0.34` (this crate's production constant),
/// Wolf `0.5` (the reference implementation's default), Singh `0.06` (the
/// paper's own document-binarization figure; its `[0,1]` range is the one
/// attached to the equation itself — see `singh_binarize`'s docs for why a
/// different figure in the same paper cannot be taken at face value).
///
/// Consequence for reading the table: a mode's row is that method **at its
/// own default**, not that method at its best. A mode losing here has not
/// been shown to be worse — it has been shown not to win untuned. Sweeping
/// `k` per method is a separate measurement.
const MODES: &[(&str, BinarizeMode)] = &[
    ("otsu", BinarizeMode::Otsu),
    (
        "sauvola",
        BinarizeMode::Sauvola {
            whsize: 16,
            k: 0.34,
        },
    ),
    ("wolf", BinarizeMode::Wolf { whsize: 16, k: 0.5 }),
    (
        "singh",
        BinarizeMode::Singh {
            whsize: 16,
            k: 0.06,
        },
    ),
];

/// Levenshtein edit distance (two-row DP) — the CER numerator, copied from
/// `tests/quality_resolution_grid.rs`'s helper of the same name so this
/// probe scores CER exactly the way the CI quality fence does.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Concatenate every recognized word's text, in doc.v1 traversal order
/// (`pages -> regions -> lines -> words`) — the same traversal
/// `tests/quality_resolution_grid.rs` uses to bucket words per cell,
/// simplified here to a flat space-joined string for CER scoring.
fn extract_text(doc_json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(doc_json).expect("parse doc.v1 JSON");
    let mut words = Vec::new();
    for page in v["pages"].as_array().map(|x| x.as_slice()).unwrap_or(&[]) {
        for region in page["regions"]
            .as_array()
            .map(|x| x.as_slice())
            .unwrap_or(&[])
        {
            for line in region["lines"]
                .as_array()
                .map(|x| x.as_slice())
                .unwrap_or(&[])
            {
                for word in line["words"]
                    .as_array()
                    .map(|x| x.as_slice())
                    .unwrap_or(&[])
                {
                    if let Some(t) = word["text"].as_str() {
                        words.push(t.to_string());
                    }
                }
            }
        }
    }
    words.join(" ")
}

/// Fraction of page pixels classified as ink (`0`, this crate's foreground
/// convention) under `mode` — computed directly from
/// [`binarize_page_with`], entirely independent of the OCR pipeline (see
/// the module docs' finding on why this is the column that actually moves).
fn ink_fraction(grey: &[u8], w: usize, h: usize, mode: BinarizeMode) -> f64 {
    let binary = binarize_page_with(grey, w, h, mode);
    let ink = binary.iter().filter(|&&p| p == 0).count();
    ink as f64 / binary.len() as f64
}

fn main() {
    let root = corpus_root();
    let model_dir = root.join("model");
    let quality_dir = root.join("quality");

    let lstm_path = model_dir.join("eng.lstm");
    let uni_path = model_dir.join("eng.lstm-unicharset");
    let rec_path = model_dir.join("eng.lstm-recoder");

    let lstm =
        fs::read(&lstm_path).unwrap_or_else(|e| fail(&format!("read {}", lstm_path.display()), e));
    let uni = fs::read_to_string(&uni_path)
        .unwrap_or_else(|e| fail(&format!("read {}", uni_path.display()), e));
    let rec =
        fs::read(&rec_path).unwrap_or_else(|e| fail(&format!("read {}", rec_path.display()), e));
    let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
        .unwrap_or_else(|e| fail("LstmRecognizer::from_components", e));

    // Dict is optional -- a missing/corrupt DAWG degrades gracefully to
    // None, the same rule `state.rs` (tesseract-ocr-web) and the quality
    // fence test both use.
    let dict = match (
        fs::read(model_dir.join("eng.lstm-word-dawg")),
        fs::read(model_dir.join("eng.lstm-punc-dawg")),
        fs::read(model_dir.join("eng.lstm-number-dawg")),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };

    // Load every fixture up front; fail loudly (with a hint) if either
    // generator hasn't been run yet. Both write into the same
    // corpus/quality/ dir, and both are committed, so this only fires on a
    // local checkout that skipped the fixtures.
    let pages: Vec<(&str, Vec<u8>, usize, usize)> = FIXTURES
        .iter()
        .map(|&(label, filename)| {
            let path = quality_dir.join(filename);
            let bytes = fs::read(&path).unwrap_or_else(|e| {
                fail(
                    &format!(
                        "read {} -- run `python3 corpus/gen/gen_uneven_light.py` and \
                         `python3 corpus/gen/gen_faded_contrast.py` first",
                        path.display()
                    ),
                    e,
                )
            });
            let (grey, w, h) = parse_pgm(&bytes)
                .unwrap_or_else(|e| fail(&format!("parse_pgm {}", path.display()), e));
            (label, grey, w, h)
        })
        .collect();

    // The CER reference: the clean page's OWN Otsu recognition (per the
    // brief -- ground truth is measured here, not an external transcript).
    let (clean_label, clean_grey, clean_w, clean_h) = &pages[0];
    assert_eq!(
        *clean_label, "clean",
        "FIXTURES[0] must be the clean baseline"
    );
    // mode_delta_cer (below) diffs each fixture's Sauvola text against ITS
    // OWN Otsu text from earlier in the same inner loop -- that only means
    // what it says if Otsu really does run first every time.
    assert_eq!(
        MODES[0].0, "otsu",
        "MODES[0] must be Otsu for mode_delta_cer's self-relative comparison"
    );
    let clean_doc = recognizer
        .recognize_document_with_mode(
            clean_grey,
            *clean_w,
            *clean_h,
            dict.as_ref(),
            None,
            BinarizeMode::Otsu,
        )
        .unwrap_or_else(|e| fail("recognize_document_with_mode(clean, otsu)", e));
    let reference_text = extract_text(&clean_doc.json);
    let reference_len = reference_text.chars().count().max(1); // guards a div-by-zero

    println!("# binarize_ab — Sauvola vs Otsu under uneven illumination\n");
    println!("reference (clean, otsu) text: {reference_text:?}\n");
    println!("| fixture | mode | word_count | mean_conf | cer | ink_frac | mode_delta_cer |");
    println!("|---|---|---|---|---|---|---|");

    // Per-mode running sums for the closing summary line.
    let mut cer_sum = vec![0.0f64; MODES.len()];
    let mut ink_sum = vec![0.0f64; MODES.len()];
    let mut n_rows = vec![0usize; MODES.len()];

    for (label, grey, w, h) in &pages {
        // Otsu's recognized text for THIS fixture, captured on the mi==0
        // pass so the Sauvola pass (mi==1) can diff against it directly --
        // the per-fixture, mode-vs-mode comparison `mode_delta_cer` reports,
        // as distinct from `cer` (each mode vs the clean/Otsu reference).
        let mut fixture_otsu_text: Option<String> = None;
        for (mi, &(mode_label, mode)) in MODES.iter().enumerate() {
            let doc = recognizer
                .recognize_document_with_mode(grey, *w, *h, dict.as_ref(), None, mode)
                .unwrap_or_else(|e| {
                    fail(
                        &format!("recognize_document_with_mode({label}, {mode_label})"),
                        e,
                    )
                });
            let text = extract_text(&doc.json);
            let cer = levenshtein(&reference_text, &text) as f64 / reference_len as f64;
            let ink = ink_fraction(grey, *w, *h, mode);
            let conf = doc
                .mean_confidence
                .map_or_else(|| "n/a".to_string(), |c| format!("{c:.2}"));
            // How much did switching binarization mode change the RECOGNIZED
            // TEXT for this SAME fixture -- the direct measurement of
            // whether binarize_mode now reaches word/line recognition,
            // isolated from how much the degradation itself hurt
            // recognition (that's what `cer`, against the clean/Otsu
            // reference, measures). Otsu (mi==0) has no prior mode on this
            // fixture to diff against.
            let mode_delta = match &fixture_otsu_text {
                None => "-".to_string(),
                Some(otsu_text) => {
                    let len = otsu_text.chars().count().max(1);
                    let d = levenshtein(otsu_text, &text) as f64 / len as f64;
                    format!("{d:.4}")
                }
            };
            if mi == 0 {
                fixture_otsu_text = Some(text);
            }
            println!(
                "| {label} | {mode_label} | {} | {conf} | {cer:.4} | {ink:.4} | {mode_delta} |",
                doc.word_count
            );
            cer_sum[mi] += cer;
            ink_sum[mi] += ink;
            n_rows[mi] += 1;
        }
    }

    println!();
    for (mi, &(mode_label, _)) in MODES.iter().enumerate() {
        let n = n_rows[mi].max(1) as f64;
        println!(
            "summary[{mode_label}]: mean_cer={:.4} mean_ink_frac={:.4} (n={})",
            cer_sum[mi] / n,
            ink_sum[mi] / n,
            n_rows[mi]
        );
    }
}
