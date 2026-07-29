//! **Where does a page's time actually go?** — the profile that decides
//! whether any pixel loop in this crate is worth vectorizing through
//! `ndarray::simd`.
//!
//! # Why this exists
//!
//! `CLAUDE.md` records a deferral: "no measurement says binarization is hot;
//! Sauvola on a 512×720 page is ~370k pixels of integral-image arithmetic
//! against an LSTM forward that dominates the per-page cost. Vectorize after
//! profiling, through the polyfill, never before."
//!
//! That was an **assumption**, and this repo's own standing lesson is that a
//! result you did not trace is a claim about the measurement apparatus, not
//! about the system. So: trace it.
//!
//! # What is timed, and why these boundaries
//!
//! The candidates for vectorization are all **per-pixel sweeps** over a whole
//! page. They are timed against the recognition they compete with:
//!
//! - `otsu` — `xy_cut::binarize_page_with(Otsu)`, the default. One histogram
//!   pass plus one threshold pass.
//! - `sauvola` / `wolf` / `singh` — the local-adaptive rungs. These carry the
//!   expensive part: two integral images, a windowed mean, a windowed
//!   mean-square, a per-pixel `sqrt` (Sauvola/Wolf), and for Wolf two global
//!   reductions. If ANY pixel loop here is hot, it is one of these.
//! - `strip_borders` — the morphology chain (two openings, two seedfills, a
//!   subtract) added for table borders.
//! - `recognize_document` — the whole pipeline: layout, region and table
//!   classification, line finding, the int8 LSTM forward per line, the CTC
//!   beam, and doc.v1 assembly.
//!
//! The ratio of the first four to the last is the entire decision. A stage
//! that is 1% of the page cannot be worth a dependency edge and an
//! architecture argument; a stage that is 20% can.
//!
//! # Reading the output honestly — and a correction worth keeping
//!
//! An earlier version of this comment asserted that a debug profile is
//! "biased in favour of the pixel loops looking hot", on the reasoning that
//! debug penalizes tight scalar loops while the LSTM sits inside `ndarray`'s
//! already-SIMD `matmul_i8_to_i32`. **Measured, that is backwards.**
//!
//! | stage | debug share | release share | debug→release speedup |
//! |---|---|---|---|
//! | `binarize[sauvola]` | 0.10% | 0.50% | 11.4× |
//! | `strip_borders` | 2.73% | 9.65% | 15.8× |
//! | `recognize_document` | 100% | 100% | **55.7×** |
//!
//! The LSTM path speeds up **55×** from debug to release while the scalar
//! pixel loops speed up only 11-16×, so the pixel loops are a LARGER share of
//! release than of debug. The reason is the opposite of the assumption:
//! `ndarray`'s SIMD is a stack of `#[inline(always)]` wrappers over
//! intrinsics, and in debug nothing inlines — every lane operation becomes a
//! real function call, which is far more punishing than what debug does to a
//! plain `for` loop over a slice. Being *already SIMD* is what makes a debug
//! build slow, not what protects it.
//!
//! **So always cite the release row.** A debug profile understates every
//! pixel stage here, and a decision made on it would be made on a number
//! biased the wrong way.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example stage_timing            # debug
//! cargo run -p tesseract-ocr --release --example stage_timing  # release
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tesseract_core::DictLite;
use tesseract_ocr::xy_cut::binarize_page_with;
use tesseract_ocr::{parse_pgm, BinarizeMode, LstmRecognizer};

fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    std::process::exit(1);
}

/// Run `f` `n` times and return the BEST wall time, not the mean. A minimum
/// is the right statistic for "how long does this work take": it is the run
/// least contaminated by scheduler noise and cold cache, and unlike a mean it
/// cannot be dragged up by one unlucky sample. The spread is printed too, so
/// a suspiciously wide one is visible rather than hidden.
fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> (std::time::Duration, std::time::Duration) {
    let mut best = std::time::Duration::MAX;
    let mut worst = std::time::Duration::ZERO;
    for _ in 0..n {
        let t = Instant::now();
        let out = f();
        let d = t.elapsed();
        // Keep the result alive so the optimizer cannot delete the work.
        std::hint::black_box(&out);
        best = best.min(d);
        worst = worst.max(d);
    }
    (best, worst)
}

fn main() {
    let root = corpus_root();
    let model_dir = root.join("model");

    let page_path = root.join("pages/page_01.pgm");
    let bytes =
        fs::read(&page_path).unwrap_or_else(|e| fail(&format!("read {}", page_path.display()), e));
    let (grey, w, h) = parse_pgm(&bytes)
        .unwrap_or_else(|e| fail(&format!("parse_pgm {}", page_path.display()), e));

    let lstm = fs::read(model_dir.join("eng.lstm")).unwrap_or_else(|e| fail("read eng.lstm", e));
    let uni = fs::read_to_string(model_dir.join("eng.lstm-unicharset"))
        .unwrap_or_else(|e| fail("read eng.lstm-unicharset", e));
    let rec =
        fs::read(model_dir.join("eng.lstm-recoder")).unwrap_or_else(|e| fail("read recoder", e));
    let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
        .unwrap_or_else(|e| fail("LstmRecognizer::from_components", e));

    let dict = match (
        fs::read(model_dir.join("eng.lstm-word-dawg")),
        fs::read(model_dir.join("eng.lstm-punc-dawg")),
        fs::read(model_dir.join("eng.lstm-number-dawg")),
    ) {
        (Ok(a), Ok(b), Ok(c)) => DictLite::from_components(&a, &b, &c).ok(),
        _ => None,
    };

    let profile = if cfg!(debug_assertions) {
        "debug — UNDERSTATES every pixel stage, do not decide on this; see the module docs"
    } else {
        "release — the row to cite"
    };
    println!("# stage_timing — page {w}x{h} = {} px, {profile}\n", w * h);

    // ── the pixel-loop candidates ────────────────────────────────────────
    const WH: usize = 16;
    let stages: [(&str, BinarizeMode); 4] = [
        ("otsu", BinarizeMode::Otsu),
        (
            "sauvola",
            BinarizeMode::Sauvola {
                whsize: WH,
                k: 0.34,
            },
        ),
        ("wolf", BinarizeMode::Wolf { whsize: WH, k: 0.5 }),
        (
            "singh",
            BinarizeMode::Singh {
                whsize: WH,
                k: 0.06,
            },
        ),
    ];

    let mut rows: Vec<(String, f64)> = Vec::new();
    for (label, mode) in stages {
        let (best, worst) = best_of(5, || binarize_page_with(&grey, w, h, mode));
        println!(
            "  binarize[{label:<8}] {:>9.2} ms   (worst {:>9.2} ms)",
            best.as_secs_f64() * 1e3,
            worst.as_secs_f64() * 1e3
        );
        rows.push((format!("binarize[{label}]"), best.as_secs_f64()));
    }

    let binary = binarize_page_with(&grey, w, h, BinarizeMode::Otsu);
    let (best, worst) = best_of(5, || {
        tesseract_ocr::strip_borders_grey(&grey, &binary, w, h)
    });
    println!(
        "  strip_borders        {:>9.2} ms   (worst {:>9.2} ms)",
        best.as_secs_f64() * 1e3,
        worst.as_secs_f64() * 1e3
    );
    rows.push(("strip_borders".to_string(), best.as_secs_f64()));

    // ── what they compete with ───────────────────────────────────────────
    // Only 2 iterations: this is seconds per run, and the point is the
    // ORDER OF MAGNITUDE against the millisecond stages above, not a
    // precise figure.
    let (doc_best, doc_worst) = best_of(2, || {
        recognizer
            .recognize_document(&grey, w, h, dict.as_ref(), None)
            .unwrap_or_else(|e| fail("recognize_document", e))
    });
    println!(
        "\n  recognize_document   {:>9.2} ms   (worst {:>9.2} ms)",
        doc_best.as_secs_f64() * 1e3,
        doc_worst.as_secs_f64() * 1e3
    );

    // ── the decision ─────────────────────────────────────────────────────
    let total = doc_best.as_secs_f64();
    println!("\n## share of one full recognize_document\n");
    println!("| stage | ms | % of page |");
    println!("|---|---|---|");
    for (label, secs) in &rows {
        println!(
            "| {label} | {:.2} | {:.2}% |",
            secs * 1e3,
            secs / total * 100.0
        );
    }
    println!("| recognize_document | {:.2} | 100% |", total * 1e3);

    let worst_stage = rows
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("stages is non-empty");
    let share = worst_stage.1 / total * 100.0;
    println!(
        "\nhottest pixel stage: {} at {:.2}% of a page.",
        worst_stage.0, share
    );
    println!(
        "Amdahl ceiling if it were made INFINITELY fast: {:.2}% off the page.",
        share
    );
}
