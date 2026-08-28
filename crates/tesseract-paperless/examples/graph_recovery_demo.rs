//! A document reaches lance-graph the way the KJV does, and the muscle-
//! memory/consistency-recovery pass runs over it — real numbers, on a real
//! recognized page.
//!
//! ```sh
//! DEEPNSM_V2_CAM96_DIR=/path/to/cam96/data \
//!   cargo run -p tesseract-paperless --release --features ocr,token \
//!   --example graph_recovery_demo
//! ```

use std::path::PathBuf;

use tesseract_ogar::{OcrExecutor, OcrRequest, OcrResponse};
use tesseract_paperless::consistency::{GraphEngine, Role};

// A telemetry percentage over small token counts — same justification as
// `consistency.rs`'s own module-scoped allow. `too_many_lines`: this is a
// linear demo trace (setup, recognize, print several report sections) —
// splitting it into helpers would scatter one narrative across several
// functions for no reader benefit, unlike library code.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    let model = std::env::var("MODEL_DIR").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model"),
        PathBuf::from,
    );
    if !model.join("eng.lstm").exists() {
        println!("model not present at {} — skipping", model.display());
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
    .expect("load the eng recognizer");

    let Ok(cam_dir) = std::env::var("DEEPNSM_V2_CAM96_DIR") else {
        println!("DEEPNSM_V2_CAM96_DIR not set — skipping (see data/README.md in deepnsm-v2)");
        return;
    };
    let cam_dir = PathBuf::from(cam_dir);
    let vocab_dir = std::env::var("DEEPNSM_VOCAB_DIR").map_or_else(
        |_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../lance-graph/crates/deepnsm/word_frequency")
        },
        PathBuf::from,
    );

    let engine = GraphEngine::from_paths(
        &vocab_dir,
        &cam_dir.join("bible_vocab.txt"),
        &cam_dir.join("cam96_codebook.bin"),
        &cam_dir.join("cam96_codes.bin"),
    )
    .expect("load real vocab + cam96 assets");

    let img = std::env::args().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm"),
        PathBuf::from,
    );
    let bytes = std::fs::read(&img).expect("read image");
    let (grey, w, h) = tesseract_ogar::parse_pgm(&bytes).expect("parse P5 pgm");

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
    let page = tesseract_ogar::DocPage::from_line_words(
        &words,
        executor.charset(),
        u32::try_from(w).expect("page width fits u32"),
        u32::try_from(h).expect("page height fits u32"),
    );

    println!(
        "== {} — {} recognized line(s) ==\n",
        img.display(),
        page.lines.len()
    );

    let (sentences, corrections) = engine.recover(&page, 100.0); // threshold 100 = every role considered, for the demo

    let mut total_triples = 0usize;
    let mut total_tokens = 0usize;
    let mut total_in_vocab = 0usize;
    for (i, gs) in sentences.iter().enumerate() {
        total_triples += gs.triples.len();
        total_tokens += gs.tokens_total;
        total_in_vocab += gs.tokens_in_vocab;
        println!(
            "[{i}] \"{}\"  (tokens {}/{} in-vocab, aligned={}, mean_conf={:.1})",
            gs.sentence.text,
            gs.tokens_in_vocab,
            gs.tokens_total,
            gs.well_aligned,
            gs.sentence.mean_conf
        );
        for t in &gs.triples {
            println!(
                "     SPO({}, {}, {:?})  conf(S/P/O)=({:.1}/{:.1}/{:?})  truth=freq:{:.3} conf:{:.3}",
                t.subject, t.predicate, t.object, t.subject_conf, t.predicate_conf, t.object_conf,
                t.truth.frequency, t.truth.confidence
            );
        }
    }

    println!(
        "\n== Totals: {} sentences, {} triples, vocab coverage {}/{} ({:.0}%) ==",
        sentences.len(),
        total_triples,
        total_in_vocab,
        total_tokens,
        100.0 * (total_in_vocab as f32) / (total_tokens.max(1) as f32)
    );

    println!(
        "\n== Consistency corrections considered (threshold=100, i.e. every role): {} ==",
        corrections.len()
    );
    for c in corrections.iter().filter(|c| c.lexical_candidate.is_some()) {
        let role = match c.role {
            Role::Subject => "S",
            Role::Predicate => "P",
            Role::Object => "O",
        };
        println!(
            "  [{role}] {:?} conf={:.1} -> lexical={:?}  sim(orig)={:?} sim(cand)={:?}  endorsed={}",
            c.original, c.original_conf, c.lexical_candidate,
            c.context_similarity_original, c.context_similarity_candidate, c.endorsed
        );
    }

    // ── Does the KJV-trained semantic space carry usable signal on
    //    NON-BIBLICAL modern prose at all? The decisive question this
    //    whole mechanism depends on — checked directly, not assumed.
    //    Every word pair here is real, measured in-vocab (page_01.gt.txt).
    println!("\n== Does the trained semantic space discriminate on THIS text's own vocabulary? ==");
    let pairs: &[(&str, &str, &str)] = &[
        ("topically related?", "birds", "sang"),
        ("topically UNrelated control", "birds", "door"),
        ("topically related?", "dawn", "morning"),
        ("topically UNrelated control", "dawn", "door"),
        ("topically related?", "garden", "grass"),
        ("topically UNrelated control", "garden", "night"),
        ("topically related?", "wind", "moved"),
        ("topically UNrelated control", "wind", "smelled"),
    ];
    for (label, a, b) in pairs {
        match engine.word_similarity(a, b) {
            Some(s) => println!("  {label:32} sim({a:?},{b:?}) = {s:.4}"),
            None => println!("  {label:32} sim({a:?},{b:?}) = NONE (no code)"),
        }
    }

    // ── A real simulated OCR corruption, end to end. "grass" (real, in-
    //    vocab, sentence [4] "The garden smelled of cut grass.") corrupted
    //    by one substitution, as a genuine OCR confusion would produce.
    //    Shows exactly what correction.rs's own suggest() proposes and
    //    whether the sentence's own context word ("garden") corroborates it
    //    over the ORIGINAL corrupted string — the real end-to-end decision
    //    the pipeline makes, not a hand-picked favorable case.
    println!("\n== A real simulated OCR corruption: \"grass\" -> \"grass\" typo variants ==");
    for corrupted in ["gtass", "grasz", "grazs", "grase"] {
        let lexical = engine.suggest_correction(corrupted);
        let sim_corrupted = engine.word_similarity(corrupted, "garden");
        let sim_candidate = lexical
            .as_ref()
            .and_then(|(c, _)| engine.word_similarity(c, "garden"));
        println!(
            "  {corrupted:?} -> lexical={lexical:?}  sim(corrupted,\"garden\")={sim_corrupted:?}  sim(candidate,\"garden\")={sim_candidate:?}"
        );
    }
}
