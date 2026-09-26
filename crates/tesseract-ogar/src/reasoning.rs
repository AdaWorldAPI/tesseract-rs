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
//! ## Narrowed limitation: `deepnsm`'s vocabulary is context-blind on
//! ## noun/verb homographs — a targeted rule now covers the common case
//!
//! `Vocabulary::tokenize` assigns exactly one PoS per surface form, chosen
//! by that form's own COCA corpus frequency, with no sentence context.
//! Common English noun/verb homographs (`bite(s)`, `run(s)`, `sleep(s)`,
//! `walk(s)`, …) resolve to whichever sense is more frequent OVERALL — often
//! the noun sense, even mid-verb-phrase. Measured directly against the real
//! `word_frequency/` data while wiring this module: "the dog bites the man"
//! tagged `bites` as `Noun` (its `word_forms.csv` wordFreq 5275 beats the
//! verb sense's 1559), so [`SentenceReasoner::analyze`] returned ZERO
//! triples for that sentence — not because the FSM parser was wrong (a
//! hand-built token sequence with `bites` forced to `Verb` correctly yields
//! `SPO(dog, bites, man)`), but because the upstream PoS tag was already
//! wrong before the parser ever saw it. Context-free frequency-based
//! tagging picks one reading, and it is not always the right one.
//!
//! [`SentenceReasoner::disambiguate`] closes the common case of this,
//! running between tokenize and parse: for a surface form with BOTH a noun
//! and a verb reading (built from the same `word_forms.csv`/
//! `word_rank_lookup.csv` data at load time, see the `homographs` table), it
//! looks at the *previous* token (skipping over adverbs) and picks Noun
//! after a determiner/adjective/preposition (the filler slot of a noun
//! phrase: "the bites", "big runs", "of bites") or Verb after a nominal
//! (noun/pronoun) or modal subject ("dog bites", "he runs", "will run").
//! Anything else — including sentence-initial position, where there is no
//! previous token to judge by — is left unchanged.
//!
//! This is still a narrow, structural limitation, not a general PoS
//! tagger: it covers exactly the noun/verb-after-a-determiner-or-subject
//! pattern. Sentence-initial homographs ("Plan the trip carefully."), gerunds, other
//! PoS pairs (noun/adjective, etc.), and deeper syntax (relative clauses,
//! coordination) are all still unhandled — disambiguating those in general
//! would need the surrounding tokens at a scale closer to a real PoS
//! tagger. [`SentenceBelief::coverage`] is unaffected either way (the word
//! still counts as "resolved," under whichever PoS was chosen), so it stays
//! a useful signal even when `triples` comes back empty for an unhandled
//! case.

use std::collections::HashMap;
use std::path::Path;

use deepnsm::parser::Parser;
use deepnsm::pos::PoS;
use deepnsm::spo::SpoTriple;
use deepnsm::vocabulary::{Token, Vocabulary};

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

/// A surface form that has BOTH a noun reading and a verb reading in the
/// loaded `deepnsm` vocabulary — e.g. "bites" (noun index 3923, "the
/// bites"; verb index 2942, "he bites"). Both indexes are 0-based
/// `deepnsm` vocabulary ranks, the same space [`deepnsm::vocabulary::Token::rank`]
/// lives in. Built once at load time by [`SentenceReasoner::from_vocab_dir`]
/// — see [`load_homographs`].
#[derive(Clone, Copy, Debug)]
struct NounVerb {
    /// 0-based vocabulary rank for this surface form's NOUN reading.
    noun: u16,
    /// 0-based vocabulary rank for this surface form's VERB reading.
    verb: u16,
}

/// Scan `word_forms.csv` and `word_rank_lookup.csv` in `dir` for surface
/// forms that resolve to a DIFFERENT vocabulary rank depending on whether
/// they are read as a noun or a verb, and record the first rank seen for
/// each PoS per surface form (lowercased). A form is kept only if BOTH a
/// noun and a verb rank were found for it.
///
/// If either CSV is missing or unreadable, this simply produces a smaller
/// (possibly empty) table — never an error beyond what
/// [`Vocabulary::load`]'s own read of the same files already reports.
fn load_homographs(dir: &Path) -> HashMap<String, NounVerb> {
    let mut nouns: HashMap<String, u16> = HashMap::new();
    let mut verbs: HashMap<String, u16> = HashMap::new();

    fn record(
        surface: &str,
        pos_letter: &str,
        rank_1based: &str,
        nouns: &mut HashMap<String, u16>,
        verbs: &mut HashMap<String, u16>,
    ) {
        let Ok(rank) = rank_1based.parse::<usize>() else {
            return;
        };
        let idx = rank.saturating_sub(1);
        if idx >= deepnsm::vocabulary::VOCAB_SIZE {
            return;
        }
        // Safe: idx < VOCAB_SIZE (4096), well within u16's range.
        #[allow(clippy::cast_possible_truncation)]
        let idx = idx as u16;
        let surface = surface.to_lowercase();
        match pos_letter {
            "n" => {
                nouns.entry(surface).or_insert(idx);
            }
            "v" => {
                verbs.entry(surface).or_insert(idx);
            }
            _ => {}
        }
    }

    // word_forms.csv: lemRank,lemma,PoS,lemFreq,wordFreq,word — surface
    // form is column 5, matching Vocabulary::load's own reading of it.
    if let Ok(content) = std::fs::read_to_string(dir.join("word_forms.csv")) {
        for line in content.lines().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 6 {
                continue;
            }
            record(fields[5], fields[2], fields[0], &mut nouns, &mut verbs);
        }
    }

    // word_rank_lookup.csv: rank,word,pos,freq
    if let Ok(content) = std::fs::read_to_string(dir.join("word_rank_lookup.csv")) {
        for line in content.lines().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 4 {
                continue;
            }
            record(fields[1], fields[2], fields[0], &mut nouns, &mut verbs);
        }
    }

    nouns
        .into_iter()
        .filter_map(|(surface, noun)| {
            verbs
                .get(&surface)
                .map(|&verb| (surface, NounVerb { noun, verb }))
        })
        .collect()
}

/// A loaded `deepnsm` vocabulary + FSM parser, ready to extract SPO triples
/// from [`AssembledSentence`]s.
pub struct SentenceReasoner {
    vocab: Vocabulary,
    parser: Parser,
    /// Surface forms with both a noun and a verb reading — see
    /// [`load_homographs`] and [`Self::disambiguate`].
    homographs: HashMap<String, NounVerb>,
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
        let homographs = load_homographs(dir);
        Ok(Self {
            vocab,
            parser: Parser::new(),
            homographs,
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

    /// Resolve the noun/verb PoS ambiguity for tokens whose surface form
    /// has both readings (see [`Self::homographs`], built once at load
    /// time by [`load_homographs`]). Runs between tokenize and parse — see
    /// the module docs' "Narrowed limitation" section.
    ///
    /// Processes left to right: for each ambiguous token, the *previous*
    /// token's PoS (skipping over any [`PoS::Adverb`]s in between, so
    /// "he quickly runs" still reads a nominal subject two tokens back)
    /// decides the reading. A determiner/adjective/preposition before it
    /// means Noun (the filler of a noun phrase — "the bites", "big runs",
    /// "of bites"); a nominal ([`PoS::is_nominal`]) or [`PoS::Modal`]
    /// subject means Verb ("dog bites", "he runs", "will run"). Anything
    /// else — including no previous token at all, at sentence start — is
    /// left unchanged. Each decision is made against the PREVIOUS token's
    /// pos as already decided by this same left-to-right pass, which is
    /// intended (an ambiguous token never needs to look more than one slot
    /// back).
    fn disambiguate(&self, tokens: &mut [Token]) {
        for i in 0..tokens.len() {
            let Some(&NounVerb { noun, verb }) = self.homographs.get(&tokens[i].surface) else {
                continue;
            };

            let mut prev_pos = None;
            let mut j = i;
            while j > 0 {
                j -= 1;
                if tokens[j].pos != PoS::Adverb {
                    prev_pos = Some(tokens[j].pos);
                    break;
                }
            }

            let Some(prev_pos) = prev_pos else {
                // Sentence-initial (or only adverbs precede it) — no
                // previous-token signal to judge by, leave unchanged.
                continue;
            };

            if matches!(prev_pos, PoS::Article | PoS::Adjective | PoS::Preposition) {
                tokens[i].pos = PoS::Noun;
                tokens[i].rank = Some(noun);
            } else if prev_pos.is_nominal() || prev_pos == PoS::Modal {
                tokens[i].pos = PoS::Verb;
                tokens[i].rank = Some(verb);
            }
        }
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
                let mut tokens = self.vocab.tokenize(&sentence.text);
                self.disambiguate(&mut tokens);
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

    #[test]
    fn homograph_after_a_subject_is_read_as_a_verb() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "homograph_after_a_subject_is_read_as_a_verb: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        // Anti-vacuity: prove the bug this fixture exercises is real on this
        // input BEFORE disambiguation — without it, "bites" resolves to its
        // Noun sense (see the module docs' "Narrowed limitation" section).
        let raw_tokens = reasoner.vocab().tokenize("The dog bites the man.");
        let raw_bites = raw_tokens
            .iter()
            .find(|t| t.surface == "bites")
            .expect("'bites' must tokenize to a real token");
        assert_eq!(
            raw_bites.pos,
            PoS::Noun,
            "the context-free tagger must still pick the Noun sense for \
             'bites' on this input — otherwise this fixture proves nothing \
             about the disambiguation pass"
        );

        let sentence = AssembledSentence {
            text: "The dog bites the man.".to_string(),
            bbox: (0, 0, 100, 10),
            line_indices: vec![0],
            mean_conf: 95.0,
        };
        let beliefs = reasoner.analyze(vec![sentence]);
        assert_eq!(beliefs.len(), 1);
        let belief = &beliefs[0];
        assert!(
            !belief.triples.is_empty(),
            "disambiguation should recover a verb reading for 'bites', \
             yielding at least one SPO triple"
        );
        let has_expected = belief
            .triples
            .iter()
            .any(|t| t.subject == "dog" && t.predicate == "bite");
        assert!(
            has_expected,
            "expected an SPO(dog, bite, ...) triple, got {:?}",
            belief.triples
        );
    }

    #[test]
    fn homograph_after_a_determiner_stays_a_noun() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "homograph_after_a_determiner_stays_a_noun: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        let mut tokens = reasoner.vocab().tokenize("The bites were painful.");
        reasoner.disambiguate(&mut tokens);
        let bites = tokens
            .iter()
            .find(|t| t.surface == "bites")
            .expect("'bites' must tokenize to a real token");
        assert_eq!(
            bites.pos,
            PoS::Noun,
            "'bites' after the determiner 'the' must stay a Noun reading"
        );
    }

    /// The noun branch's can-fire test. `homograph_after_a_determiner_stays_a_noun`
    /// cannot fail on its own: the context-free tagger already reads `bites`
    /// as a noun there, so removing the noun branch changes nothing. `plan`
    /// is one of the homographs the tagger reads as a VERB (measured: also
    /// `study`, `answer`, `attack`), so only the noun branch can turn it
    /// into a noun after `the`.
    #[test]
    fn homograph_after_a_determiner_is_read_as_a_noun() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "homograph_after_a_determiner_is_read_as_a_noun: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        let raw = reasoner.vocab().tokenize("The plan was simple.");
        let raw_plan = raw
            .iter()
            .find(|t| t.surface == "plan")
            .expect("'plan' must tokenize to a real token");
        assert_eq!(
            raw_plan.pos,
            PoS::Verb,
            "the context-free tagger must read 'plan' as a verb here, or this \
             fixture cannot tell the noun branch from no branch at all"
        );
        assert!(
            reasoner.homographs.contains_key("plan"),
            "'plan' must be in the noun/verb side table"
        );

        let mut tokens = raw.clone();
        reasoner.disambiguate(&mut tokens);
        let plan = tokens
            .iter()
            .find(|t| t.surface == "plan")
            .expect("'plan' survives disambiguation");
        assert_eq!(
            plan.pos,
            PoS::Noun,
            "'plan' after the determiner 'the' must be read as a noun"
        );
        assert_eq!(
            plan.rank,
            Some(reasoner.homographs["plan"].noun),
            "the rank must move to the noun lemma together with the PoS"
        );
    }

    #[test]
    fn homograph_at_sentence_start_is_left_unchanged() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "homograph_at_sentence_start_is_left_unchanged: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        let tokens = reasoner.vocab().tokenize("Plan the trip carefully.");
        let original_pos = tokens[0].pos;
        let mut disambiguated = tokens.clone();
        reasoner.disambiguate(&mut disambiguated);
        assert_eq!(
            disambiguated[0].pos, original_pos,
            "a sentence-initial homograph has no previous token to judge by \
             and must be left exactly as the tokenizer produced it"
        );
    }

    #[test]
    fn the_side_table_holds_real_noun_verb_pairs() {
        let dir = vocab_dir();
        if !dir.join("word_rank_lookup.csv").exists() {
            eprintln!(
                "the_side_table_holds_real_noun_verb_pairs: skipping — \
                 {} not present in this environment",
                dir.display()
            );
            return;
        }
        let reasoner =
            SentenceReasoner::from_vocab_dir(&dir).expect("load the real deepnsm vocabulary");

        let bites = reasoner
            .homographs
            .get("bites")
            .expect("'bites' must be a recorded homograph");
        assert_eq!(bites.noun, 3923, "bites' noun rank (0-based)");
        assert_eq!(bites.verb, 2942, "bites' verb rank (0-based)");

        let runs = reasoner
            .homographs
            .get("runs")
            .expect("'runs' must be a recorded homograph");
        assert_eq!(runs.noun, 1224, "runs' noun rank (0-based)");
        assert_eq!(runs.verb, 201, "runs' verb rank (0-based)");
    }
}
