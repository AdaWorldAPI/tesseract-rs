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
//! # ⚠ It also settles a FOUNDING claim, and settles it against
//!
//! Before the transcode existed, the case for doing one at all — rather than
//! binding to libtesseract — included: *a pure-Rust pipeline can route its hot
//! pixel work through `ndarray::simd`, and **the image resizing is the
//! target***. A binding is stuck with leptonica's kernels; a transcode is not.
//!
//! **Measured, resizing is the SMALLEST pixel stage here** — 1.06 ms for every
//! line on the page, **0.12%**, less than plain Otsu binarization. The line
//! crops are simply tiny: 7 lines × ~512×28 px ≈ 100k pixels against 368,640
//! for one whole-page binarize, at a gentle f ≈ 1.3 upscale.
//!
//! And that is not an artifact of this page having few lines. `prescale` runs
//! **per line** and the LSTM forward runs **per line**, so both sides of the
//! ratio scale together — ~0.15 ms of scaling against ~110 ms of recognition
//! per line. A 176-line page multiplies both by 25. If anything the ratio gets
//! *worse* for the scaler on a dense page, because the fixed whole-page
//! overhead (layout, region classification) amortizes away and leaves the LSTM
//! an even larger share.
//!
//! **The transcode was still worth doing — the reason was just mis-attributed.**
//! Its payoff is what this repo actually demonstrates: zero C at runtime (the
//! web demo is a glibc binary plus a ~4 MB model), byte-parity *provability*
//! (you cannot diff a binding against itself), WASM/Docker deployability, and
//! the whole `doc.v1` surface a binding could never have produced. None of
//! that is a SIMD argument, and none of it needed one.
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
//! # Two isolated-timing traps this probe fell into, both now fixed
//!
//! ## Trap 1 — an isolated call is NOT an Amdahl fraction
//!
//! The first version divided one isolated `binarize_page_with` call by one
//! `recognize_document` and called the quotient an Amdahl ceiling. That is
//! invalid twice over, and a codex review caught both:
//!
//! - The adaptive rows time Sauvola/Wolf/Singh while the denominator runs
//!   **Otsu** — so the numerator was a stage the denominator never executed.
//! - Even for Otsu, the pipeline binarizes **several times per page**: once in
//!   `recognize_document_with_options`, again inside
//!   `recognize_page_blocks_words_with_mode`'s `xy_cut`, again per block in
//!   makerow segmentation, and again for the layout `xy_cut`. One invocation
//!   understates the optimizable fraction by however many times it actually
//!   runs.
//!
//! The fix does not require counting invocations at all: **time the whole
//! pipeline under each mode.** `recognize_document_with_mode(Sauvola)` minus
//! `recognize_document_with_mode(Otsu)` is the true in-pipeline cost delta of
//! the binarizer, counting every call site automatically. That is the number
//! this probe now reports, and the only one a decision should rest on.
//!
//! The isolated per-call timings are kept, because "what does one call cost"
//! is a genuinely useful number — but they are labelled ONE INVOCATION and
//! carry no page-share column.
//!
//! ## Trap 2 — a real measurement with an invented mechanism
//!
//! Debug and release disagree sharply about how large the pixel stages look:
//!
//! | stage | debug share | release share | debug→release speedup |
//! |---|---|---|---|
//! | `binarize[sauvola]` | 0.10% | 0.50% | 11.4× |
//! | `strip_borders` | 2.73% | 9.65% | 15.8× |
//! | `recognize_document` | 100% | 100% | **55.7×** |
//!
//! The **ratios are measured** and the practical guidance follows from them:
//! a debug profile understates every pixel stage here, so **always cite the
//! release row**.
//!
//! An earlier version of this comment then explained the gap by asserting
//! that "in debug nothing inlines, so every lane operation becomes a real
//! function call". **That explanation is false** — rustc honours
//! `#[inline(always)]` at `-C opt-level=0` (LLVM's AlwaysInliner runs at O0),
//! so the ndarray SIMD wrappers *are* inlined in debug. The speedup gap is
//! real; the cause attached to it was invented rather than verified, and a
//! codex review flagged it.
//!
//! **The mechanism is therefore UNVERIFIED and deliberately not guessed at
//! again here.** A plausible candidate — that O0 skips the optimization
//! *after* inlining, leaving bounds checks, no LICM and no register
//! allocation across a SIMD-heavy body — is exactly the kind of story that
//! sounded convincing last time. Verifying it means reading the generated
//! code (`cargo asm` / `--emit=llvm-ir` on the real ndarray path), not
//! reasoning about it. Until someone does that, the ratios stand alone.
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

    // ── the per-LINE scale, which is the one that was nearly missed ──────
    //
    // `LstmRecognizer::prepare_grid` calls `prescale_grey_to_height` ONCE PER
    // RECOGNIZED LINE, so unlike every stage above this one does not run once
    // per page -- it runs ~30x. Timing it on a single crop would understate
    // it by that factor, which is exactly how a hot stage hides inside a
    // whole-page profile.
    //
    // This also matters more than the binarizers for a second reason: the
    // scale chain (`scale_gray_li` bilinear, `scale_gray_area_map`,
    // `unsharp_mask_gray_2d`) is f32/f64 arithmetic per pixel, which IS what
    // `ndarray::simd_ops` covers (`add_f32` / `mul_f32` / `scale_f32` /
    // `add_mul_f32`). The binary morphology of `strip_borders` is not.
    //
    // Line geometry comes from a real doc.v1 run rather than being invented,
    // so the crop widths, heights and therefore the scale FACTORS are the
    // ones the recognizer actually hits -- the factor decides which kernel
    // runs (LI vs area-map vs exact 2^-n), so a made-up factor could time the
    // wrong code path entirely.
    let probe_doc = recognizer
        .recognize_document(&grey, w, h, dict.as_ref(), None)
        .unwrap_or_else(|e| fail("recognize_document (line-geometry probe)", e));
    let doc_json: serde_json::Value =
        serde_json::from_str(&probe_doc.json).unwrap_or_else(|e| fail("parse doc.v1", e));
    let mut line_boxes: Vec<(usize, usize)> = Vec::new();
    for page in doc_json["pages"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        for region in page["regions"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            for line in region["lines"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                let b = &line["bbox"];
                let (l, t, r, bo) = (
                    b[0].as_i64().unwrap_or(0),
                    b[1].as_i64().unwrap_or(0),
                    b[2].as_i64().unwrap_or(0),
                    b[3].as_i64().unwrap_or(0),
                );
                let (cw, ch) = ((r - l).max(1) as usize, (bo - t).max(1) as usize);
                line_boxes.push((cw.min(w), ch.min(h)));
            }
        }
    }
    if line_boxes.is_empty() {
        println!("  prescale/page          (no lines recognized -- nothing to time)");
    } else {
        // The model input height every line is scaled TO.
        const TARGET_H: usize = 36;
        let crops: Vec<Vec<u8>> = line_boxes
            .iter()
            .map(|&(cw, ch)| grey[..cw * ch].to_vec())
            .collect();
        let (best, worst) = best_of(5, || {
            let mut acc = 0usize;
            for (crop, &(cw, ch)) in crops.iter().zip(&line_boxes) {
                let (out, ow) = tesseract_ocr::prescale_grey_to_height(crop, cw, ch, TARGET_H);
                acc += out.len() + ow;
            }
            acc
        });
        let heights: Vec<usize> = line_boxes.iter().map(|&(_, ch)| ch).collect();
        let hmin = heights.iter().copied().min().unwrap_or(0);
        let hmax = heights.iter().copied().max().unwrap_or(0);
        println!(
            "  prescale x{:<3} lines  {:>9.2} ms   (worst {:>9.2} ms)  \
             [line h {hmin}..{hmax} -> {TARGET_H}, f {:.2}..{:.2}]",
            line_boxes.len(),
            best.as_secs_f64() * 1e3,
            worst.as_secs_f64() * 1e3,
            TARGET_H as f64 / hmax.max(1) as f64,
            TARGET_H as f64 / hmin.max(1) as f64,
        );
        rows.push((
            format!("prescale (all {} lines)", line_boxes.len()),
            best.as_secs_f64(),
        ));
    }

    // ── THE VALID MEASUREMENT: whole pipeline, per mode ──────────────────
    //
    // This replaces an invalid one. Dividing an isolated `binarize_page_with`
    // call by one `recognize_document` is NOT an Amdahl fraction: the
    // pipeline binarizes several times per page (initial binarize, the
    // `xy_cut` inside `recognize_page_blocks_words_with_mode`, per-block
    // makerow segmentation, and the layout `xy_cut` again), and for the
    // adaptive modes the denominator was running Otsu and so never executed
    // the numerator's stage at all.
    //
    // Timing the WHOLE pipeline under each mode sidesteps counting call sites
    // entirely: `with_mode(X) - with_mode(Otsu)` is the true end-to-end cost
    // of choosing X, with every invocation included automatically. It also
    // stays correct if a call site is added or removed later, which a
    // hand-counted multiplier would not.
    println!("\n## whole pipeline per mode — the number a decision rests on\n");
    let mut page_ms: Vec<(String, f64)> = Vec::new();
    for (label, mode) in stages {
        let (best, worst) = best_of(2, || {
            recognizer
                .recognize_document_with_mode(&grey, w, h, dict.as_ref(), None, mode)
                .unwrap_or_else(|e| fail("recognize_document_with_mode", e))
        });
        println!(
            "  recognize_document[{label:<8}] {:>9.2} ms   (worst {:>9.2} ms)",
            best.as_secs_f64() * 1e3,
            worst.as_secs_f64() * 1e3
        );
        page_ms.push((label.to_string(), best.as_secs_f64()));
    }
    let otsu_total = page_ms
        .iter()
        .find(|(l, _)| l == "otsu")
        .map(|(_, s)| *s)
        .unwrap_or(f64::NAN);
    let doc_best = std::time::Duration::from_secs_f64(otsu_total);

    // ── the decision ─────────────────────────────────────────────────────
    let total = doc_best.as_secs_f64();
    // ── the in-pipeline cost of choosing a mode ──────────────────────────
    // This IS a valid optimizable fraction: it is measured end-to-end on the
    // same workload, so every call site of the binarizer is included and no
    // multiplier has to be guessed.
    println!("\n## in-pipeline cost of each mode (vs Otsu, whole page)\n");
    println!("| mode | page ms | delta vs otsu | delta as % of page |");
    println!("|---|---|---|---|");
    for (label, secs) in &page_ms {
        let delta = secs - otsu_total;
        println!(
            "| {label} | {:.2} | {:+.2} ms | {:+.2}% |",
            secs * 1e3,
            delta * 1e3,
            delta / otsu_total * 100.0
        );
    }

    // ── isolated per-call costs, NOT page shares ─────────────────────────
    // Deliberately reported without a "% of page" column. One invocation over
    // one page is not an Amdahl fraction (see the module docs, Trap 1) and
    // presenting it as one is what this probe got wrong first time round.
    println!("\n## isolated cost of ONE invocation (not a page share)\n");
    println!("| stage | ms for one call |");
    println!("|---|---|");
    for (label, secs) in &rows {
        println!("| {label} | {:.2} |", secs * 1e3);
    }
    println!(
        "\nreference: one whole recognize_document (otsu) = {:.2} ms",
        total * 1e3
    );

    let adaptive_worst = page_ms
        .iter()
        .filter(|(l, _)| l != "otsu")
        .map(|(_, s)| s - otsu_total)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\nWorst adaptive binarizer costs {:+.2}% of a page end-to-end. That is\n\
         the ceiling any binarizer optimization could recover, and it counts\n\
         every call site — not one isolated invocation.",
        adaptive_worst / otsu_total * 100.0
    );
}
