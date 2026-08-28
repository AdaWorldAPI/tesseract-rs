//! The reasoning layer over assembled sentences: `deepnsm`'s SPO extraction
//! plus a [`NarsTruth`] belief per sentence.
//!
//! **Not a 15th OGAR capability.** `OcrRequest`/`OcrResponse`'s
//! exhaustiveness fuse (see the crate root docs) ties `COVERED_CAPABILITIES`
//! 1:1 to `ogar_vocab::ocr_actions::OCR_ACTION_NAMES` — reasoning over
//! already-recognized text is explicitly OUTSIDE that declared OCR action
//! table (`tesseract-rs/CLAUDE.md`'s own framing: "the OPTIONAL seed a
//! consumer feeds via OGAR", never one of the 14 canonical OCR actions).
//! This module is a plain post-processing library a caller reaches for
//! AFTER getting a [`DocPage`](tesseract_ocr::DocPage) — no request/response
//! variant, no capability mint, no exhaustiveness-fuse change.
//!
//! ## What this wires, and what it deliberately does not
//!
//! Per `tesseract-rs/CLAUDE.md`'s "AS-IS BOUNDARY" analysis, the cost of
//! reaching lance-graph's reasoning surface splits into three pieces, only
//! two of which are cheap:
//!
//! - **[`NarsTruth`]** — zero-dep contract crate, already a
//!   `tesseract-ogar` dependency. Wired here.
//! - **Per-sentence SPO** (`deepnsm`'s 6-state PoS FSM → triples) — path
//!   deps `ndarray` + `lance-graph-contract`, both already satisfied
//!   transitively in this workspace. Wired here, via the LOW-level
//!   `Vocabulary` + `parser` API (NOT `DeepNsmEngine::load`, which also
//!   needs a `codebook_pq.bin`/`cam_codes.bin` pair this repo does not
//!   ship — those only feed the VSA/distance-matrix half of the pipeline,
//!   which SPO extraction does not need).
//! - **NARS *reasoning*** (belief arena, revision, the 5 tactics) — lives in
//!   `lance-graph-planner`, which pulls `serde`/`tokio`/`tracing` — outside
//!   this crate's lean dependency set. NOT wired here; a caller that needs
//!   revision-over-time across multiple recognized documents reaches for
//!   `lance-graph-planner` directly, downstream of this module's output.
//!
//! ## Known limitation: `deepnsm`'s vocabulary is context-blind on
//! ## noun/verb homographs — measured, not a wiring bug
//!
//! `Vocabulary::tokenize` assigns exactly one PoS per surface form, chosen
//! by that form's own COCA corpus frequency, with no sentence context.
//! Common English noun/verb homographs (`bite(s)`, `run(s)`, `sleep(s)`,
//! `walk(s)`, …) resolve to whichever sense is more frequent OVERALL — often
//! the noun sense, even mid-verb-phrase. Measured directly against the real
//! `word_frequency/` data while wiring this module: "the dog bites the man"
//! tags `bites` as `Noun` (its `word_forms.csv` wordFreq 5275 beats the verb
//! sense's 1559), so [`SentenceReasoner::analyze`] returns ZERO triples for
//! that sentence — not because the FSM parser is wrong (a hand-built token
//! sequence with `bites` forced to `Verb` correctly yields
//! `SPO(dog, bites, man)`), but because the upstream PoS tag was already
//! wrong before the parser ever saw it. This is a real, structural
//! limitation of context-free frequency-based tagging, not a quick fix —
//! disambiguating "bites" would need the surrounding tokens, which is a PoS
//! tagger in its own right. Out of scope for this wiring pass; noted here
//! so a caller doesn't mistake an empty `triples` list for a wiring failure.
//! [`SentenceBelief::coverage`] is unaffected by this (the word still counts
//! as "resolved," just under the wrong PoS), so it stays a useful signal
//! even when `triples` comes back empty.

use std::path::Path;

use deepnsm::parser::Parser;
use deepnsm::spo::SpoTriple;
use deepnsm::vocabulary::Vocabulary;

pub use lance_graph_contract::exploration::NarsTruth;

use crate::sentences::AssembledSentence;

/// A failure loading the [`SentenceReasoner`]'s vocabulary.
#[derive(Debug)]
pub struct ReasoningError(String);

impl std::fmt::Display for ReasoningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "deepnsm vocabulary: {}", self.0)
    }
}

impl std::error::Error for ReasoningError {}

/// One SPO triple resolved back to its lemma text (from `deepnsm`'s 12-bit
/// vocabulary ranks — see [`SpoTriple`]). `object` is `None` for an
/// intransitive triple ([`SpoTriple::has_object`] false), never the literal
/// sentinel word at rank [`deepnsm::spo::NO_ROLE`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedTriple {
    /// The subject's lemma.
    pub subject: String,
    /// The predicate's lemma.
    pub predicate: String,
    /// The object's lemma, or `None` for an intransitive triple.
    pub object: Option<String>,
}

/// One assembled sentence plus what the reasoning layer could extract from
/// it: `deepnsm`'s SPO triples (resolved to lemma text), the FSM's token
/// coverage, and a [`NarsTruth`] belief blending OCR confidence with that
/// coverage — see [`sentence_nars_truth`].
///
/// No `PartialEq`: [`NarsTruth`] itself doesn't derive it (an upstream,
/// zero-dep contract type this crate doesn't own).
#[derive(Clone, Debug)]
pub struct SentenceBelief {
    /// The source sentence (text, bbox, contributing lines, OCR mean_conf).
    pub sentence: AssembledSentence,
    /// SPO triples the FSM parser resolved from this sentence's tokens.
    pub triples: Vec<ResolvedTriple>,
    /// FSM token coverage ∈ [0, 1] — classified tokens / total tokens
    /// (`deepnsm::parser::ParseResult::coverage`).
    pub coverage: f32,
    /// The belief this module attaches to the sentence — see
    /// [`sentence_nars_truth`].
    pub truth: NarsTruth,
}

/// Maps OCR confidence + parse coverage to a [`NarsTruth`] belief about a
/// recognized-and-parsed sentence's reliability.
///
/// **This module's own construction, not a transcode of any Tesseract or
/// NARS-canonical formula** — same footing as `structured.rs`'s `doc.v1` /
/// `rectify.rs`'s heuristics: consumer-side synthesis over a proven
/// substrate, documented as such rather than asserted as ground truth.
///
/// - `frequency` — the plain mean of two independent [0,1] "is this
///   trustworthy" signals: OCR mean word confidence (`mean_word_conf/100`)
///   and FSM parse coverage. A simple average keeps neither signal
///   dominating; a caller who wants to weight them differently should
///   compute frequency itself and use [`NarsTruth::new`] directly.
/// - `confidence` (the NARS evidence-weight sense — see
///   [`NarsTruth::revision`]'s own use of `confidence/(1-confidence)` as a
///   weight) — the standard NARS evidence discount `w/(w+1)` where `w` is
///   the token count: more tokens observed in agreement is more evidence,
///   asymptotically approaching but never reaching 1 (`NarsTruth::new`
///   itself clamps confidence to `[0, 0.99]`).
#[must_use]
pub fn sentence_nars_truth(mean_word_conf: f32, coverage: f32, token_count: usize) -> NarsTruth {
    let freq = ((mean_word_conf / 100.0).clamp(0.0, 1.0) + coverage.clamp(0.0, 1.0)) / 2.0;
    let w = token_count as f32;
    let conf = w / (w + 1.0);
    NarsTruth::new(freq, conf)
}

/// A loaded `deepnsm` vocabulary + FSM parser, ready to extract SPO triples
/// from [`AssembledSentence`]s.
pub struct SentenceReasoner {
    vocab: Vocabulary,
    parser: Parser,
}

impl SentenceReasoner {
    /// Load the `deepnsm` 4,096-word COCA vocabulary from `dir` (the
    /// `word_frequency/` directory shipped in the `deepnsm` crate — see
    /// [`deepnsm::vocabulary::Vocabulary::load`]'s module docs for the two
    /// CSVs it reads). Uses [`deepnsm::parser::DEFAULT_COVERAGE_THRESHOLD`]
    /// (0.85).
    ///
    /// # Errors
    ///
    /// [`ReasoningError`] if the vocabulary CSVs can't be read/parsed.
    pub fn from_vocab_dir(dir: &Path) -> Result<Self, ReasoningError> {
        let vocab = Vocabulary::load(dir).map_err(ReasoningError)?;
        Ok(Self {
            vocab,
            parser: Parser::new(),
        })
    }

    /// The loaded vocabulary — the SAME instance [`Self::analyze`] tokenizes
    /// against. Exposed so a caller building a SECOND reasoning pipeline over
    /// the same COCA tags (e.g. feeding a different PoS-aware FSM) reuses
    /// this loaded vocabulary instead of re-parsing `word_rank_lookup.csv` +
    /// `word_forms.csv` a second time. Read-only: nothing outside this module
    /// mutates a `SentenceReasoner`'s vocabulary once loaded.
    #[must_use]
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Resolve one [`SpoTriple`]'s vocabulary ranks to lemma text.
    fn resolve_triple(&self, t: &SpoTriple) -> ResolvedTriple {
        ResolvedTriple {
            subject: self.vocab.word(t.subject()).to_string(),
            predicate: self.vocab.word(t.predicate()).to_string(),
            object: t
                .has_object()
                .then(|| self.vocab.word(t.object()).to_string()),
        }
    }

    /// Run SPO extraction + [`sentence_nars_truth`] over every assembled
    /// sentence, in order. A sentence with zero resolvable tokens (e.g. an
    /// empty or all-OOV line) still produces a [`SentenceBelief`] — with an
    /// empty `triples` list and `coverage: 0.0` — never silently dropped.
    #[must_use]
    pub fn analyze(&self, sentences: Vec<AssembledSentence>) -> Vec<SentenceBelief> {
        sentences
            .into_iter()
            .map(|sentence| {
                let tokens = self.vocab.tokenize(&sentence.text);
                let result = self.parser.parse_with_coverage(&tokens);
                let triples = result
                    .structure
                    .triples
                    .iter()
                    .map(|t| self.resolve_triple(t))
                    .collect();
                let truth = sentence_nars_truth(sentence.mean_conf, result.coverage, tokens.len());
                SentenceBelief {
                    sentence,
                    triples,
                    coverage: result.coverage,
                    truth,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn sentence_nars_truth_blends_ocr_conf_and_coverage() {
        let t = sentence_nars_truth(100.0, 1.0, 3);
        assert!(
            (t.frequency - 1.0).abs() < 1e-6,
            "mean of two 1.0 signals must be 1.0, got {}",
            t.frequency
        );
        assert!(
            (t.confidence - 0.75).abs() < 1e-6,
            "3 tokens -> w/(w+1) = 0.75, got {}",
            t.confidence
        );
    }

    #[test]
    fn sentence_nars_truth_confidence_increases_with_more_tokens() {
        let few = sentence_nars_truth(80.0, 0.8, 1);
        let many = sentence_nars_truth(80.0, 0.8, 10);
        assert!(
            many.confidence > few.confidence,
            "more tokens must mean more evidence: {} vs {}",
            few.confidence,
            many.confidence
        );
    }

    #[test]
    fn sentence_nars_truth_frequency_increases_with_word_conf() {
        let low = sentence_nars_truth(40.0, 0.5, 5);
        let high = sentence_nars_truth(95.0, 0.5, 5);
        assert!(
            high.frequency > low.frequency,
            "higher OCR confidence must raise frequency: {} vs {}",
            low.frequency,
            high.frequency
        );
    }

    #[test]
    fn sentence_nars_truth_zero_tokens_is_defined_not_nan() {
        let t = sentence_nars_truth(0.0, 0.0, 0);
        assert_eq!(t.confidence, 0.0, "0/(0+1) = 0, not NaN");
        assert_eq!(t.frequency, 0.0);
    }

    /// Sibling-repo path to `deepnsm`'s bundled `word_frequency/` data —
    /// mirrors this crate's own `../../../lance-graph/...` path-dep
    /// convention (see `Cargo.toml`). Graceful-skip when absent, matching
    /// the established pattern for real-data tests in this workspace
    /// (`lib.rs`'s `smoke_recognize_line_matches_proven_regression`).
    fn vocab_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../lance-graph/crates/deepnsm/word_frequency")
    }

    #[test]
    fn analyze_extracts_a_real_spo_triple_from_a_simple_sentence() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "analyze_extracts_a_real_spo_triple_from_a_simple_sentence: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        // NOT "The dog bites the man." — measured during this wiring (see
        // this fn's doc comment): deepnsm's vocabulary resolves "bites" to
        // its NOUN sense (word_forms.csv's noun-lemma row for "bites" has
        // wordFreq 5275 vs the verb-lemma row's 1559 — a real corpus-
        // frequency fact, not a lookup bug), so the FSM never sees a verb
        // token to anchor a triple on. "sees" has no such competing noun
        // sense and resolves correctly — verified against the REAL
        // tokenize() path, not hand-built tokens, so this is an honest
        // end-to-end proof of the wiring in this module.
        let sentence = AssembledSentence {
            text: "The dog sees the cat.".to_string(),
            bbox: (0, 0, 100, 10),
            line_indices: vec![0],
            mean_conf: 95.0,
        };
        let beliefs = reasoner.analyze(vec![sentence]);
        assert_eq!(beliefs.len(), 1);
        let belief = &beliefs[0];
        assert!(
            belief.coverage > 0.0,
            "a real English sentence over the real COCA vocabulary must resolve \
             SOME tokens, not report zero coverage"
        );
        assert!(
            !belief.triples.is_empty(),
            "'The dog sees the cat.' is a canonical SVO sentence with an \
             unambiguous verb — the FSM must extract at least one SPO triple from it"
        );
        let triple = &belief.triples[0];
        assert!(
            !triple.subject.is_empty() && !triple.predicate.is_empty(),
            "resolved triple must carry real lemma text, not empty strings: {triple:?}"
        );
    }

    #[test]
    fn analyze_never_drops_a_sentence_even_with_zero_coverage() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "analyze_never_drops_a_sentence_even_with_zero_coverage: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        // Gibberish, unlikely to resolve against the COCA vocabulary.
        let sentence = AssembledSentence {
            text: "Zxqvblorptfizzniknok wobbledoop.".to_string(),
            bbox: (0, 0, 100, 10),
            line_indices: vec![0],
            mean_conf: 40.0,
        };
        let beliefs = reasoner.analyze(vec![sentence]);
        assert_eq!(
            beliefs.len(),
            1,
            "a low/zero-coverage sentence must still produce a SentenceBelief, \
             never be silently dropped"
        );
    }
}
