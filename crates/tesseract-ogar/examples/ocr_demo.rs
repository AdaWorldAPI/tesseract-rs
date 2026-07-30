//! **Consumer test-drive of the OGAR OCR executor surface.**
//!
//! Shows exactly how a consumer (woa-rs, medcare-rs, smb-office-rs, …) invokes
//! the OCR capabilities that `ogar_vocab::ocr_actions` declares — the same
//! typed request/response surface an external API would advertise, but every
//! call is an in-process function call (no serialization). Run it:
//!
//! ```sh
//! # default: recognize the bundled corpus page
//! cargo run -p tesseract-ogar --example ocr_demo
//! # or point it at your own P5 (binary) PGM:
//! cargo run -p tesseract-ogar --example ocr_demo -- path/to/page.pgm
//! ```
//!
//! It (1) prints the OGAR-declared capability table + the exhaustiveness fuse,
//! then (2) loads the executor from the bundled `eng` model and runs the
//! `recognize_document` one-shot (→ `doc.v1` JSON + typed invoice fields +
//! confidence) and the plain-text `recognize_page` path on a real image.
//!
//! Image input is P5 PGM only (kept dependency-light — no `image` crate in this
//! crate); the web demo (`tesseract-ocr-web`) is the PNG/JPEG/WebP front door.

use std::path::PathBuf;

use tesseract_ogar::{OcrExecutor, OcrRequest, OcrResponse, COVERED_CAPABILITIES};

fn main() {
    // ── 1. The OGAR capability surface this executor implements ──────────────
    println!("== OGAR OCR capability table (the authoritative ogar_vocab::ocr_actions) ==");
    for spec in ogar_vocab::ocr_actions::ocr_actions() {
        let params: Vec<String> = spec
            .params
            .iter()
            .map(|p| {
                if p.mandatory {
                    p.name.to_string()
                } else {
                    format!("[{}]", p.name) // optional
                }
            })
            .collect();
        println!(
            "  {:<24} ({}) -> {}",
            spec.def.predicate,
            params.join(", "),
            spec.produces.join(", ")
        );
    }
    println!(
        "  fuse: {} capabilities declared by OGAR == {} covered by tesseract-ogar\n",
        ogar_vocab::ocr_actions::OCR_ACTION_NAMES.len(),
        COVERED_CAPABILITIES.len()
    );

    // ── 2. Load the executor from the bundled eng model ──────────────────────
    // MODEL_DIR env override (same convention as tesseract-ocr-web's server —
    // the Docker runtime image sets this), else the CARGO_MANIFEST_DIR-relative
    // bundled corpus/model (unchanged local `cargo run` default).
    let model = std::env::var("MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model"));
    if !model.join("eng.lstm").exists() {
        println!(
            "(model not present at {} — skipping the live run. The table above IS \
             the surface a consumer calls; wire your model dir to see it execute.)",
            model.display()
        );
        return;
    }
    let dawg = |name: &str| {
        let p = model.join(name);
        p.exists().then_some(p)
    };
    let executor = OcrExecutor::from_data_paths(
        &model.join("eng.lstm"),
        &model.join("eng.lstm-unicharset"),
        &model.join("eng.lstm-recoder"),
        dawg("eng.lstm-word-dawg").as_deref(),
        dawg("eng.lstm-punc-dawg").as_deref(),
        dawg("eng.lstm-number-dawg").as_deref(),
    )
    .expect("load the eng recognizer + dictionary from corpus/model");

    // ── 3. The image (CLI arg, DEMO_IMAGE env override, or a bundled page) ───
    let img = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var("DEMO_IMAGE").ok().map(PathBuf::from))
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm")
        });
    let bytes = std::fs::read(&img).expect("read the image file");
    let (grey, w, h) = tesseract_ocr::parse_pgm(&bytes)
        .expect("parse a P5 (binary) PGM — convert other formats to PGM first");

    // ── 4. recognize_document — the one-shot a consumer wants ────────────────
    println!("== recognize_document on {} ({w}x{h}) ==", img.display());
    match executor
        .execute(OcrRequest::RecognizeDocument {
            grey: &grey,
            width: w,
            height: h,
            binarize: tesseract_ogar::BinarizeMode::default(),
            with_dict: true,
            harvest_profile: Some("german_invoice"),
        })
        .expect("execute recognize_document")
    {
        OcrResponse::DocumentOut { doc_json, fields } => {
            println!("doc.v1 JSON (structure + classified regions + quality):");
            println!("{doc_json}");
            println!("\nharvested fields ({}):", fields.len());
            for f in &fields {
                println!("  {:<16} = {:<14} checks={:?}", f.key, f.value, f.checks);
            }
            if fields.is_empty() {
                println!(
                    "  (none — this fixture is not a German invoice; the harvest ran, \
                          found no labelled amounts/IBAN, and returned empty)"
                );
            }
        }
        other => println!("unexpected response: {other:?}"),
    }

    // ── 5. recognize_page — the simple plain-text path ───────────────────────
    if let OcrResponse::PageText { text, .. } = executor
        .execute(OcrRequest::RecognizePage {
            grey: &grey,
            width: w,
            height: h,
            with_dict: true,
        })
        .expect("execute recognize_page")
    {
        println!("\n== recognize_page (plain text) ==\n{text}");
    }

    println!(
        "\n(Tip: the doc.v1 JSON above carries \"quality\":{{\"mean_conf\",\"low_confidence\"}} — \
         a low score flags handwriting / low-res / non-printed input instead of returning it silently.)"
    );

    // ── 6. reasoning layer — sentence assembly + deepnsm SPO + NarsTruth ─────
    // Not part of the OGAR-declared capability table above (see
    // `reasoning.rs`'s module docs for why): a plain post-processing step a
    // caller reaches for AFTER `recognize_page_words`, using the typed
    // `DocPage` surface rather than `recognize_document`'s JSON string.
    let words = match executor
        .execute(OcrRequest::RecognizePageWords {
            grey: &grey,
            width: w,
            height: h,
            with_dict: true,
        })
        .expect("execute recognize_page_words")
    {
        OcrResponse::LineWordsOut(lines) => lines,
        other => panic!("unexpected response: {other:?}"),
    };
    let page =
        tesseract_ocr::DocPage::from_line_words(&words, executor.charset(), w as u32, h as u32);
    let sentences = tesseract_ogar::sentences::assemble_sentences(&page);
    println!(
        "\n== reasoning: {} sentence(s) assembled from {} recognized line(s) ==",
        sentences.len(),
        page.lines.len()
    );

    // DEEPNSM_VOCAB_DIR lets the Docker runtime point at its own copy of
    // deepnsm's word_frequency/ CSVs (the sibling source path below only
    // exists at BUILD time, inside the builder stage's /src layout).
    let vocab_dir = std::env::var("DEEPNSM_VOCAB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../lance-graph/crates/deepnsm/word_frequency")
        });
    if !vocab_dir.join("word_rank_lookup.csv").exists() {
        println!(
            "(deepnsm vocabulary not present at {} — skipping SPO extraction. \
             Sentence text + bbox + mean_conf are still shown below.)",
            vocab_dir.display()
        );
        for s in &sentences {
            println!("  [{:>5.1}] {}", s.mean_conf, s.text);
        }
        return;
    }
    let reasoner = tesseract_ogar::reasoning::SentenceReasoner::from_vocab_dir(&vocab_dir)
        .expect("load the deepnsm vocabulary");
    for belief in reasoner.analyze(sentences) {
        println!(
            "  [{:>5.1} conf, {:.0}% coverage] {}",
            belief.sentence.mean_conf,
            belief.coverage * 100.0,
            belief.sentence.text
        );
        for t in &belief.triples {
            match &t.object {
                Some(obj) => println!("      SPO: {} -> {} -> {}", t.subject, t.predicate, obj),
                None => println!("      SP:  {} -> {}", t.subject, t.predicate),
            }
        }
        println!(
            "      belief: frequency={:.2} confidence={:.2}",
            belief.truth.frequency, belief.truth.confidence
        );
    }
}
