//! Baseline for the v1 COCA `d` behaviour, pinned BEFORE any fix.
//!
//! In the committed COCA `lemmas_5k.csv`, PoS letter `d` marks determiners
//! (`this`, `that`, `which`, `all`, `some`, …); modals such as `can` and
//! `would` are tagged `v`. deepnsm v1 maps `d` to `PoS::Modal`
//! (`deepnsm/src/pos.rs`), and its parser treats `Modal` like a verb. This
//! file records what that does today through the path this crate ships
//! (`SentenceReasoner::analyze`, the source of the web binary's `spo_json`),
//! so a later correction is a measured change rather than a surprise.
//!
//! These pins are a record, not an endorsement. A deliberate fix to `d`
//! should turn them red and be re-pinned in the same change, with the
//! difference reported.

use std::path::{Path, PathBuf};

use tesseract_ogar::reasoning::SentenceReasoner;
use tesseract_ogar::sentences::AssembledSentence;
use tesseract_ogar::PoS;

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The committed v1 vocabulary in the sibling lance-graph checkout. Required:
/// a missing directory is a broken checkout, so this fails rather than skips.
fn vocab_dir() -> PathBuf {
    let dir = workspace().join("../lance-graph/crates/deepnsm/word_frequency");
    assert!(
        dir.join("lemmas_5k.csv").exists(),
        "v1 vocabulary missing at {} — the lance-graph sibling checkout is required",
        dir.display()
    );
    dir
}

/// Lowercased lemmas whose COCA PoS letter is `d`, in file order, deduplicated.
fn d_lemmas(dir: &Path) -> Vec<String> {
    let csv = std::fs::read_to_string(dir.join("lemmas_5k.csv")).expect("read lemmas_5k.csv");
    let mut out: Vec<String> = Vec::new();
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.get(2) == Some(&"d") {
            let w = f[1].to_lowercase();
            if !out.contains(&w) {
                out.push(w);
            }
        }
    }
    out
}

fn sentence(text: &str) -> AssembledSentence {
    AssembledSentence {
        text: text.to_string(),
        bbox: (0, 0, 0, 0),
        line_indices: vec![0],
        mean_conf: 100.0,
    }
}

/// The `d` census: 34 distinct `d` lemmas, 24 of which reach the tagger as
/// `Modal`. The other 10 have an earlier row with another PoS, and v1's first
/// row wins; they are pinned by name below. `this`, `that`, `all` and `both`
/// are among them (tagged `Adverb`), which is why the corpus ground truth,
/// whose only `d` lemmas are `all` and `this`, has no `Modal` token.
#[test]
fn every_coca_d_lemma_is_tagged_modal_today() {
    let dir = vocab_dir();
    let reasoner = SentenceReasoner::from_vocab_dir(&dir).expect("load v1 vocab");
    let lemmas = d_lemmas(&dir);
    let modal: Vec<&String> = lemmas
        .iter()
        .filter(|w| {
            reasoner
                .vocab()
                .tokenize(w)
                .first()
                .is_some_and(|t| t.pos == PoS::Modal)
        })
        .collect();
    assert_eq!(
        (lemmas.len(), modal.len()),
        (34, 24),
        "d lemmas / tagged Modal: {lemmas:?}"
    );
    let not_modal: Vec<&str> = lemmas
        .iter()
        .filter(|w| !modal.contains(w))
        .map(String::as_str)
        .collect();
    assert_eq!(
        not_modal,
        ["this", "that", "all", "much", "own", "another", "such", "each", "both", "half"],
        "d lemmas an earlier row tags as something other than Modal"
    );
    // Anti-vacuity: the census must contain the determiners named above, so a
    // file or parse change cannot empty it unnoticed.
    for w in ["this", "that", "which", "all", "some"] {
        assert!(
            lemmas.iter().any(|l| l == w),
            "{w} missing from the d census"
        );
    }
}

/// Exact sentences through `SentenceReasoner::analyze`: the triples and the
/// coverage that land in `spo_json` today.
#[test]
fn determiner_sentences_through_analyze_are_pinned() {
    let reasoner = SentenceReasoner::from_vocab_dir(&vocab_dir()).expect("load v1 vocab");
    let texts = [
        "This dog saw all the men.",
        "Some people found the record.",
        "Each man took any bread.",
        "The dog bit the man.",
    ];
    let got: Vec<String> = reasoner
        .analyze(texts.iter().map(|t| sentence(t)).collect())
        .iter()
        .map(|b| {
            let triples: Vec<String> = b
                .triples
                .iter()
                .map(|t| {
                    format!(
                        "({},{},{})",
                        t.subject,
                        t.predicate,
                        t.object.as_deref().unwrap_or("-")
                    )
                })
                .collect();
            format!("cov={:.3} {}", b.coverage, triples.join(" "))
        })
        .collect();
    // `Each man took any bread`: `any` is COCA `d`, arrives as `Modal`, and
    // the parser reads it as a second predicate — the bogus `(man,any,bread)`.
    let want: [&str; 4] = [
        "cov=1.000 (dog,see,man)",
        "cov=1.000 (people,found,record)",
        "cov=1.000 (man,take,-) (man,any,bread)",
        "cov=1.000 (dog,bite,man)",
    ];
    assert_eq!(got, want);
}

/// A compact statistic over the committed corpus ground truth: how many
/// tokens are `d` lemmas tagged `Modal`, and how many triples `analyze`
/// extracts in total.
///
/// Measured: the ground truth's only `d` lemmas are `all` (twice) and `this`
/// (once), and v1 tags both `Adverb`, so no `d`-lemma token is tagged `Modal`.
/// A fix that only remaps `Modal` therefore does not move this test; a fix that
/// also changes which COCA row wins for `all`/`this` would. It guards the
/// corpus goldens against collateral change; the exact cases above carry the
/// teeth.
#[test]
fn corpus_ground_truth_d_statistics_are_pinned() {
    let dir = vocab_dir();
    let reasoner = SentenceReasoner::from_vocab_dir(&dir).expect("load v1 vocab");
    let lemmas = d_lemmas(&dir);
    let pages = workspace().join("corpus/pages");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&pages)
        .expect("read corpus/pages")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".gt.txt"))
        .collect();
    files.sort();

    let (mut sentences, mut tokens, mut d_modal, mut triples) = (0usize, 0usize, 0usize, 0usize);
    for f in &files {
        let text = std::fs::read_to_string(f).expect("read ground truth");
        let batch: Vec<AssembledSentence> = text
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(sentence)
            .collect();
        sentences += batch.len();
        for s in &batch {
            for t in reasoner.vocab().tokenize(&s.text) {
                tokens += 1;
                if t.pos == PoS::Modal && lemmas.contains(&t.surface) {
                    d_modal += 1;
                }
            }
        }
        triples += reasoner
            .analyze(batch)
            .iter()
            .map(|b| b.triples.len())
            .sum::<usize>();
    }
    assert_eq!(
        (files.len(), sentences, tokens, d_modal, triples),
        (10, 70, 413, 0, 29),
        "files, sentences, tokens, d-lemma tokens tagged Modal, triples"
    );
}
