//! `consistency` — a document reaches lance-graph the way the KJV does, and
//! then gets read back through it: real per-sentence SPO extraction over
//! deepnsm-v2's FSM and trained CAM-PQ 96 semantic space, with OCR word
//! confidence as the "muscle memory" anchor and every correction reported
//! with provenance, never silent.
//!
//! ## The three layers, and what each one actually is
//!
//! - **LSTM = mechanical muscle memory.** The recognizer's per-word
//!   confidence (`DocWord::conf`) is never re-derived here — it is read as
//!   given, and HIGH-confidence words are the anchors this whole module
//!   depends on and never itself corrects. Fast, first-pass, load-bearing.
//! - **Graph/grammar consistency recovery.** deepnsm-v2's REAL
//!   [`deepnsm_v2::fsm::parse_to_spo`] (not v1's) runs on every assembled
//!   sentence, producing role-typed (subject/predicate/object) triples
//!   addressed to `(page, line_indices, bbox)` — a document's own natural
//!   tree in place of the KJV's book:chapter:verse. **Honest scope, stated
//!   once and not glossed over below:** the PoS TAGGING step is NOT new —
//!   it reuses v1's SAME context-free COCA tagger (`SentenceReasoner::vocab`,
//!   re-exported for exactly this reuse), so it carries v1's SAME
//!   noun/verb-homograph weakness (documented in `reasoning.rs`). What v2
//!   contributes that v1 structurally cannot is the FSM's clause machinery
//!   (relative clauses, subject-carry chaining) and, above all, a TRAINED
//!   distributional semantic space (`Nsm::word_similarity`) — v1 has no
//!   notion of meaning distance at all. "Consistency recovery" here means:
//!   a low-confidence role-filler's Levenshtein candidate (from
//!   [`tesseract_ogar::correction`]) is endorsed only when it is
//!   MEANING-CLOSER to the sentence's own other high-confidence content
//!   words than the original recognition was — topical/semantic coherence,
//!   not syntactic selectional restriction, and declines outright wherever
//!   either word lacks a trained code (never fabricates a verdict from an
//!   absent signal).
//! - **Token recovery.** `DocWord::text`/bbox are already byte-exact by
//!   construction (`tesseract-rs/CLAUDE.md`, `E-ONE-RECEIPT-MANY-BORROWED-
//!   CONSUMERS-1`) — nothing here invents a new addressing scheme. Every
//!   [`GraphTriple`] and [`ConsistencyCorrection`] carries the ORIGINAL text
//!   and its exact `line_indices`/bbox alongside any endorsed candidate, so
//!   a caller can always recover what was actually printed regardless of
//!   what this module concluded about it.
//!
//! ## Vocabulary coverage — measured, not assumed
//!
//! `bible_vocab.txt` is the KJV's OWN 12,543-word vocabulary (per
//! `deepnsm-v2`'s own crate docs), not a general-English list. A modern
//! document's everyday nouns/verbs WILL be partially out-of-vocabulary —
//! measured on this repo's own `corpus/pages/page_01.gt.txt`: 30/38 (79%)
//! unique words in-vocab, but content words like "clock"/"coffee"/"boots"/
//! "hike"/"rack"/"ticked"/"cooled" are OOV, and an OOV role-filler cannot be
//! semantically judged (no [`deepnsm_v2::Cam96`] code exists for it) —
//! [`GraphSentence::tokens_in_vocab`]/`tokens_total` report this per
//! sentence so a caller sees the real coverage rather than a silent gap.

// Every count/ratio computed here (token counts, occurrence tallies,
// coverage fractions) is a small telemetry number bound by a sentence's or
// page's own token count — never realistically large enough for f32's
// 23-bit mantissa to matter. Scoped to this module rather than argued at
// each of the several call sites.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::Path;

use deepnsm_v2::codebook::{load_cam96_codes, load_cam96_space, CodebookError};
use deepnsm_v2::fsm::{parse_to_spo, Pos as V2Pos, Tagged};
use deepnsm_v2::vocab::WordId;
use deepnsm_v2::{Nsm, PaletteVocab};

pub use lance_graph_contract::exploration::NarsTruth;

use tesseract_ogar::correction::{suggest, CorrectionPolicy, Lexicon};
use tesseract_ogar::reasoning::{ReasoningError, SentenceReasoner};
use tesseract_ogar::sentences::{assemble_sentences, AssembledSentence};
use tesseract_ogar::{DocPage, PoS as V1Pos};

/// Which SPO role a word occupies — the address a [`ConsistencyCorrection`]
/// reports itself against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The clause's subject.
    Subject,
    /// The clause's verb.
    Predicate,
    /// The clause's object (transitive triples only).
    Object,
}

/// One SPO triple extracted from a recognized sentence, addressed to the
/// document's own `(page, line_indices, bbox)` tree — the sentence-level
/// generalization of KJV's book:chapter:verse address, over a document that
/// has no such pre-existing structure.
#[derive(Clone, Debug)]
pub struct GraphTriple {
    /// The sentence's own address (line indices + top-down bbox).
    pub line_indices: Vec<usize>,
    /// Top-down image bbox (the sentence's own).
    pub bbox: (i32, i32, i32, i32),
    /// The resolved subject lemma.
    pub subject: String,
    /// The resolved predicate lemma.
    pub predicate: String,
    /// The resolved object lemma, `None` for an intransitive triple.
    pub object: Option<String>,
    /// Mean OCR confidence (0-100) over every occurrence of the subject word
    /// id in this sentence.
    pub subject_conf: f32,
    /// Mean OCR confidence over every occurrence of the predicate word id.
    pub predicate_conf: f32,
    /// `None` only when the triple is intransitive (`object` is also
    /// `None`) — never a missing-but-expected value.
    pub object_conf: Option<f32>,
    /// This triple's belief, per [`triple_nars_truth`].
    pub truth: NarsTruth,
}

impl GraphTriple {
    /// The (role, text, confidence) triples this triple's roles resolve to,
    /// object included only when the triple is transitive.
    fn roles(&self) -> Vec<(Role, &str, f32)> {
        let mut r = vec![
            (Role::Subject, self.subject.as_str(), self.subject_conf),
            (
                Role::Predicate,
                self.predicate.as_str(),
                self.predicate_conf,
            ),
        ];
        if let (Some(obj), Some(conf)) = (&self.object, self.object_conf) {
            r.push((Role::Object, obj.as_str(), conf));
        }
        r
    }
}

/// One assembled sentence plus everything the graph layer extracted from
/// it — never dropped, even at zero coverage (mirrors
/// `SentenceReasoner::analyze`'s own guarantee).
#[derive(Clone, Debug)]
pub struct GraphSentence {
    /// The source sentence (text, bbox, contributing lines, OCR `mean_conf`).
    pub sentence: AssembledSentence,
    /// SPO triples the v2 FSM resolved from this sentence's tokens.
    pub triples: Vec<GraphTriple>,
    /// Tokens the v1 tagger produced for this sentence.
    pub tokens_total: usize,
    /// Of those, how many resolved to a `WordId` in v2's trained vocabulary
    /// (and so were visible to the FSM at all — an OOV token is structurally
    /// invisible, not merely uncertain).
    pub tokens_in_vocab: usize,
    /// Whether per-word confidence alignment succeeded for this sentence
    /// (token surfaces matched the flattened `DocWord` sequence 1:1). When
    /// `false`, every role's confidence in this sentence's triples falls
    /// back UNIFORMLY to `sentence.mean_conf` — declared, never guessed.
    pub well_aligned: bool,
}

/// One endorsed or declined correction candidate, reported whether or not
/// it was applied — "every change reported" extended to "every candidate
/// considered", matching `tesseract_ogar::correction`'s own doctrine.
#[derive(Clone, Debug)]
pub struct ConsistencyCorrection {
    /// The source sentence's line indices.
    pub line_indices: Vec<usize>,
    /// Which SPO role this correction concerns.
    pub role: Role,
    /// The word as originally recognized.
    pub original: String,
    /// The original word's OCR confidence (0-100).
    pub original_conf: f32,
    /// `None` when `tesseract_ogar::correction::suggest` itself declined
    /// (a digit, a known word, below the length floor, or nothing in
    /// budget) — the graph layer never proposes where the lexical layer
    /// found nothing.
    pub lexical_candidate: Option<String>,
    /// Mean [`Nsm::word_similarity`] between the ORIGINAL word and the
    /// triple's OTHER high-confidence content roles. `None` when no other
    /// role has a trained code to compare against (never fabricated).
    pub context_similarity_original: Option<f32>,
    /// Same, for `lexical_candidate`. `None` for the same reason, or if
    /// there is no candidate.
    pub context_similarity_candidate: Option<f32>,
    /// `true` only when a candidate exists AND its context similarity
    /// strictly exceeds the original's by [`GraphEngine::ENDORSE_MARGIN`] —
    /// the graph layer's OWN verdict, independent of whether the lexical
    /// layer proposed anything.
    pub endorsed: bool,
}

/// Failure loading a [`GraphEngine`]'s data assets.
#[derive(Debug)]
pub enum GraphEngineError {
    /// Loading the v1 COCA vocabulary failed.
    Reasoning(ReasoningError),
    /// Building the correction lexicon from the same vocab dir failed.
    Lexicon(String),
    /// Loading the trained CAM-PQ 96 codebook failed.
    Codebook(CodebookError),
    /// Reading one of the asset files failed.
    Io(std::io::Error),
}

impl std::fmt::Display for GraphEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reasoning(e) => write!(f, "v1 vocabulary: {e}"),
            Self::Lexicon(e) => write!(f, "correction lexicon: {e}"),
            Self::Codebook(e) => write!(f, "cam96 codebook: {e:?}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for GraphEngineError {}

/// Confidence threshold (0-100) below which a role-filler is a candidate for
/// consistency recovery. Policy pin, not a measurement — same footing as
/// `tesseract-ocr`'s own documented policy constants.
pub const LOW_CONFIDENCE_THRESHOLD: f32 = 70.0;

/// A document's real path into lance-graph: v1's tagger (reused, not
/// reimplemented) feeding v2's real FSM and trained CAM-PQ 96 space.
pub struct GraphEngine {
    reasoner: SentenceReasoner,
    lexicon: Lexicon,
    policy: CorrectionPolicy,
    nsm: Nsm,
}

/// [`GraphEngine::tag_sentence`]'s result — its own struct purely for
/// readability (clippy's `type_complexity` on the equivalent 5-tuple);
/// every field is read at the one call site via destructuring.
struct TaggedSentence {
    tagged: Vec<Tagged>,
    conf_by_id: HashMap<WordId, (f32, u32)>,
    tokens_total: usize,
    tokens_in_vocab: usize,
    well_aligned: bool,
}

impl GraphEngine {
    /// A candidate must beat the original's context similarity by at least
    /// this much to be endorsed. Policy pin: large enough that float noise
    /// on an already-close pair cannot flip a verdict, small enough not to
    /// bury genuine recoveries. Re-measure against real corpora before
    /// treating this as tuned.
    pub const ENDORSE_MARGIN: f32 = 0.02;

    /// The absolute bar a lexical candidate's context similarity must clear
    /// to be endorsed when the ORIGINAL recognized text has no code (the
    /// common OCR-garbage case — see `recover`'s endorse logic). **A
    /// PROVISIONAL pin, not a tuned threshold**: measured on exactly 4 real
    /// word pairs from `corpus/pages/page_01.gt.txt` against the real
    /// KJV-trained codebook (`examples/graph_recovery_demo.rs`) — genuinely
    /// topically related noun pairs scored 0.61-0.69 ("dawn"/"morning",
    /// "garden"/"grass"); unrelated controls and noun-verb pairs scored
    /// 0.24-0.36. `0.5` sits between those two measured clusters, roughly
    /// centered. n=4 is not evidence of a general threshold — re-measure
    /// against a real corpus of confirmed OCR corrections before trusting
    /// this value in production, and note the SAME n=4 check found
    /// noun-verb similarity (the Predicate role) noticeably weaker than
    /// noun-noun (Subject/Object) — this bar may need to differ by role.
    pub const ABSOLUTE_ENDORSE_THRESHOLD: f32 = 0.5;

    /// Load the v1 COCA tagger (`vocab_dir`, the `deepnsm` `word_frequency/`
    /// directory) and v2's trained CAM-PQ 96 space (`bible_vocab_txt`,
    /// `cam96_codebook_bin`, `cam96_codes_bin` — the `v0.1.0-cam96-data`
    /// release assets; see `deepnsm-v2/data/README.md`).
    ///
    /// # Errors
    ///
    /// [`GraphEngineError`] if any asset can't be read or parsed.
    pub fn from_paths(
        vocab_dir: &Path,
        bible_vocab_txt: &Path,
        cam96_codebook_bin: &Path,
        cam96_codes_bin: &Path,
    ) -> Result<Self, GraphEngineError> {
        let reasoner =
            SentenceReasoner::from_vocab_dir(vocab_dir).map_err(GraphEngineError::Reasoning)?;
        let lexicon =
            Lexicon::from_deepnsm_vocab_dir(vocab_dir).map_err(GraphEngineError::Lexicon)?;

        let vocab_text = std::fs::read_to_string(bible_vocab_txt).map_err(GraphEngineError::Io)?;
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(vocab_text.lines());

        let codebook_bytes = std::fs::read(cam96_codebook_bin).map_err(GraphEngineError::Io)?;
        let space = load_cam96_space(&codebook_bytes).map_err(GraphEngineError::Codebook)?;
        let codes_bytes = std::fs::read(cam96_codes_bin).map_err(GraphEngineError::Io)?;
        let codes = load_cam96_codes(&codes_bytes).map_err(GraphEngineError::Codebook)?;

        let nsm = Nsm::with_codes(vocab, space, codes);

        Ok(Self {
            reasoner,
            lexicon,
            policy: CorrectionPolicy::default(),
            nsm,
        })
    }

    /// The trained [`Nsm::word_similarity`] this engine's correction pass
    /// depends on — exposed so a caller (or a falsifier) can check what the
    /// semantic space actually says about two words directly, independent
    /// of any sentence context.
    #[must_use]
    pub fn word_similarity(&self, a: &str, b: &str) -> Option<f32> {
        self.nsm.word_similarity(a, b)
    }

    /// Whether a candidate correction is endorsed, given the ORIGINAL
    /// recognized text's context similarity and the CANDIDATE's — the pure
    /// decision [`Self::recover`] applies per role-filler, factored out for
    /// direct testing.
    ///
    /// Two routes, not one — found by running this against a real simulated
    /// corruption (`examples/graph_recovery_demo.rs`) and measuring the
    /// FIRST version's behaviour: it required BOTH similarities to exist,
    /// which structurally excludes the single strongest evidence case —
    /// genuinely garbled non-word OCR output (no code, so
    /// `original` is always `None`) where the lexical layer already found a
    /// real, contextually-fitting candidate. That is the common "confident
    /// and wrong" OCR failure mode this repo's own findings document
    /// repeatedly, and the original rule could never confirm it.
    #[must_use]
    fn decide_endorse(original: Option<f32>, candidate: Option<f32>) -> bool {
        match (original, candidate) {
            // Both real words with codes: OCR confused one real word for
            // another (e.g. "hen"/"ten") — endorse only on a STRICT
            // comparative win.
            (Some(o), Some(c)) => c > o + Self::ENDORSE_MARGIN,
            // The recognized text is not itself a real word (the common
            // case for genuine OCR garbage) but the lexical candidate IS,
            // and clears an ABSOLUTE plausibility bar against the
            // sentence's own context — endorse on the candidate's own
            // strength, since there is no original-side signal to compare
            // against.
            (None, Some(c)) => c >= Self::ABSOLUTE_ENDORSE_THRESHOLD,
            // No signal at all either direction: decline.
            _ => false,
        }
    }

    /// The plain Levenshtein/frequency correction [`Self::recover`] uses as
    /// its lexical candidate, exposed for the same reason as
    /// [`Self::word_similarity`] — a caller checking what one component of
    /// the pipeline says, independent of the rest.
    #[must_use]
    pub fn suggest_correction(&self, word: &str) -> Option<(String, usize)> {
        suggest(word, &self.lexicon, &self.policy)
    }

    /// Map v1's context-free COCA tag onto v2's FSM tag. Not a lossless
    /// mapping — v1's `is_negated` flag has no v2 counterpart and is
    /// dropped here, a genuine (small) capability loss, documented rather
    /// than hidden. `that`/`which`/`who`/`whom`/`whose` are promoted to
    /// [`V2Pos::Rel`] regardless of their v1 tag (Pronoun or Conjunction —
    /// v1's tag set does not distinguish a relativizer from either), since
    /// v2's relative-clause machinery is exactly what those words feed.
    fn map_pos(pos: V1Pos, surface: &str) -> V2Pos {
        if matches!(surface, "that" | "which" | "who" | "whom" | "whose") {
            return V2Pos::Rel;
        }
        match pos {
            V1Pos::Article => V2Pos::Det,
            V1Pos::Adjective => V2Pos::Adj,
            V1Pos::Verb => V2Pos::Verb,
            V1Pos::Noun | V1Pos::Pronoun => V2Pos::Noun,
            V1Pos::Adverb
            | V1Pos::Preposition
            | V1Pos::Conjunction
            | V1Pos::Modal
            | V1Pos::Interjection
            | V1Pos::Particle
            | V1Pos::Negation
            | V1Pos::Existential => V2Pos::Other,
        }
    }

    /// Strip leading/trailing non-alphanumeric characters and lowercase —
    /// the normalization used to align a v1 token's `surface` against a
    /// flattened `DocWord.text`.
    fn normalize(s: &str) -> String {
        s.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    }

    /// Flatten the `DocWord`s of `sentence`'s contributing lines, in order.
    fn flatten_words<'a>(page: &'a DocPage, sentence: &AssembledSentence) -> Vec<&'a str> {
        let mut out = Vec::new();
        for &li in &sentence.line_indices {
            let Some(line) = page.lines.get(li) else {
                continue;
            };
            for w in &line.words {
                out.push(w.text.as_str());
            }
        }
        out
    }

    fn flatten_confs(page: &DocPage, sentence: &AssembledSentence) -> Vec<f32> {
        let mut out = Vec::new();
        for &li in &sentence.line_indices {
            let Some(line) = page.lines.get(li) else {
                continue;
            };
            for w in &line.words {
                out.push(w.conf);
            }
        }
        out
    }

    /// Tag one sentence and build the `Tagged` stream v2's FSM consumes,
    /// plus a `WordId -> mean confidence` table for the roles it can
    /// resolve. Returns a [`TaggedSentence`].
    fn tag_sentence(&self, page: &DocPage, sentence: &AssembledSentence) -> TaggedSentence {
        let tokens = self.reasoner.vocab().tokenize(&sentence.text);
        let flat_words = Self::flatten_words(page, sentence);
        let flat_confs = Self::flatten_confs(page, sentence);

        let well_aligned = tokens.len() == flat_words.len()
            && tokens
                .iter()
                .zip(flat_words.iter())
                .all(|(t, w)| Self::normalize(&t.surface) == Self::normalize(w));

        let mut tagged = Vec::with_capacity(tokens.len() + 1);
        let mut conf_by_id: HashMap<WordId, (f32, u32)> = HashMap::new();
        let mut tokens_in_vocab = 0usize;

        for (i, tok) in tokens.iter().enumerate() {
            let Some(id) = self.nsm.vocab.id(&tok.surface) else {
                continue;
            };
            tokens_in_vocab += 1;
            let conf = if well_aligned {
                flat_confs[i]
            } else {
                sentence.mean_conf
            };
            let entry = conf_by_id.entry(id).or_insert((0.0, 0));
            entry.0 += conf;
            entry.1 += 1;
            tagged.push(Tagged::new(id, Self::map_pos(tok.pos, &tok.surface)));
        }
        tagged.push(Tagged::new(0, V2Pos::Stop));

        TaggedSentence {
            tagged,
            conf_by_id,
            tokens_total: tokens.len(),
            tokens_in_vocab,
            well_aligned,
        }
    }

    /// Real per-sentence SPO extraction over deepnsm-v2's FSM. Never drops a
    /// sentence, mirroring `SentenceReasoner::analyze`.
    #[must_use]
    pub fn analyze(&self, page: &DocPage) -> Vec<GraphSentence> {
        assemble_sentences(page)
            .into_iter()
            .map(|sentence| {
                let TaggedSentence {
                    tagged,
                    conf_by_id,
                    tokens_total,
                    tokens_in_vocab,
                    well_aligned,
                } = self.tag_sentence(page, &sentence);
                let spos = parse_to_spo(&tagged);

                let conf_of = |id: WordId| -> f32 {
                    conf_by_id
                        .get(&id)
                        .map_or(sentence.mean_conf, |(sum, n)| sum / (*n as f32))
                };

                let triples: Vec<GraphTriple> = spos
                    .into_iter()
                    .map(|spo| {
                        let subject = self.nsm.vocab.word(spo.subject).unwrap_or("").to_string();
                        let predicate =
                            self.nsm.vocab.word(spo.predicate).unwrap_or("").to_string();
                        let has_object = spo.object != 0 || subject.is_empty();
                        // Spo's intransitive sentinel is checked the same way
                        // v1's SpoTriple::has_object reads: an object id of 0
                        // (the vocab's own rank-0 slot, the highest-frequency
                        // word) would be a false negative for a genuinely
                        // transitive triple whose object IS that word — a
                        // known, narrow edge this module inherits rather than
                        // resolves (v2's `Spo` has no dedicated intransitive
                        // sentinel distinct from a real WordId 0).
                        let object = has_object
                            .then(|| self.nsm.vocab.word(spo.object).unwrap_or("").to_string());
                        let subject_conf = conf_of(spo.subject);
                        let predicate_conf = conf_of(spo.predicate);
                        let object_conf = object.as_ref().map(|_| conf_of(spo.object));

                        let mut role_confs = vec![subject_conf, predicate_conf];
                        if let Some(c) = object_conf {
                            role_confs.push(c);
                        }
                        let truth = triple_nars_truth(&role_confs, tokens_in_vocab, tokens_total);

                        GraphTriple {
                            line_indices: sentence.line_indices.clone(),
                            bbox: sentence.bbox,
                            subject,
                            predicate,
                            object,
                            subject_conf,
                            predicate_conf,
                            object_conf,
                            truth,
                        }
                    })
                    .collect();

                GraphSentence {
                    sentence,
                    triples,
                    tokens_total,
                    tokens_in_vocab,
                    well_aligned,
                }
            })
            .collect()
    }

    /// [`Self::analyze`] plus grammar-consistency correction for every
    /// role-filler below `low_conf_threshold`. Returns the sentence-level
    /// results unchanged (corrections are reported ALONGSIDE, never applied
    /// in place) plus the flat correction list, endorsed and declined alike.
    #[must_use]
    pub fn recover(
        &self,
        page: &DocPage,
        low_conf_threshold: f32,
    ) -> (Vec<GraphSentence>, Vec<ConsistencyCorrection>) {
        let sentences = self.analyze(page);
        let mut corrections = Vec::new();

        for gs in &sentences {
            for triple in &gs.triples {
                let roles = triple.roles();
                for &(role, text, conf) in &roles {
                    if conf >= low_conf_threshold {
                        continue;
                    }
                    let context: Vec<&str> = roles
                        .iter()
                        .filter(|(r, _, c)| *r != role && *c >= low_conf_threshold)
                        .map(|(_, t, _)| *t)
                        .collect();

                    let lexical_candidate =
                        suggest(text, &self.lexicon, &self.policy).map(|(c, _dist)| c);

                    let sim_against = |word: &str| -> Option<f32> {
                        if context.is_empty() {
                            return None;
                        }
                        let mut total = 0.0f32;
                        let mut n = 0u32;
                        for ctx in &context {
                            if let Some(s) = self.nsm.word_similarity(word, ctx) {
                                total += s;
                                n += 1;
                            }
                        }
                        (n > 0).then_some(total / n as f32)
                    };

                    let context_similarity_original = sim_against(text);
                    let context_similarity_candidate =
                        lexical_candidate.as_deref().and_then(sim_against);

                    let endorsed = Self::decide_endorse(
                        context_similarity_original,
                        context_similarity_candidate,
                    );

                    corrections.push(ConsistencyCorrection {
                        line_indices: triple.line_indices.clone(),
                        role,
                        original: text.to_string(),
                        original_conf: conf,
                        lexical_candidate,
                        context_similarity_original,
                        context_similarity_candidate,
                        endorsed,
                    });
                }
            }
        }

        (sentences, corrections)
    }
}

/// Maps role-filler OCR confidences to a [`NarsTruth`] belief about one SPO
/// triple's reliability — the same construction as
/// `tesseract_ogar::reasoning::sentence_nars_truth`, at TRIPLE (not
/// sentence) granularity: `frequency` blends mean role confidence with the
/// sentence's OWN vocabulary coverage (`tokens_in_vocab/tokens_total`);
/// `confidence` uses `tokens_in_vocab` (not the raw token count) as the
/// evidence weight — deliberately, since an OOV token is structurally
/// invisible to the FSM and contributes zero evidence toward any triple it
/// might otherwise have anchored.
#[must_use]
pub fn triple_nars_truth(
    role_confs: &[f32],
    tokens_in_vocab: usize,
    tokens_total: usize,
) -> NarsTruth {
    let mean_role_conf = if role_confs.is_empty() {
        0.0
    } else {
        role_confs.iter().sum::<f32>() / role_confs.len() as f32
    };
    let coverage = if tokens_total == 0 {
        0.0
    } else {
        tokens_in_vocab as f32 / tokens_total as f32
    };
    let freq = f32::midpoint(
        (mean_role_conf / 100.0).clamp(0.0, 1.0),
        coverage.clamp(0.0, 1.0),
    );
    let w = tokens_in_vocab as f32;
    let conf = w / (w + 1.0);
    NarsTruth::new(freq, conf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "checking for EXACTLY 0.0 (not NaN) from 0.0/(0.0+1.0), which is exact division, is the whole point of this test"
    )]
    fn triple_nars_truth_zero_is_defined_not_nan() {
        let t = triple_nars_truth(&[], 0, 0);
        assert_eq!(t.frequency, 0.0);
        assert_eq!(t.confidence, 0.0);
    }

    #[test]
    fn triple_nars_truth_blends_conf_and_coverage() {
        let t = triple_nars_truth(&[100.0, 100.0], 4, 4);
        assert!((t.frequency - 1.0).abs() < 1e-6);
        assert!(
            (t.confidence - 0.8).abs() < 1e-6,
            "4/(4+1)=0.8, got {}",
            t.confidence
        );
    }

    #[test]
    fn map_pos_promotes_relativizers_regardless_of_v1_tag() {
        assert_eq!(GraphEngine::map_pos(V1Pos::Pronoun, "that"), V2Pos::Rel);
        assert_eq!(
            GraphEngine::map_pos(V1Pos::Conjunction, "which"),
            V2Pos::Rel
        );
        assert_eq!(GraphEngine::map_pos(V1Pos::Pronoun, "he"), V2Pos::Noun);
    }

    // ── decide_endorse: two routes, plus the gap the second route closes ──
    //
    // Disable table (each verified red-then-green by hand while writing
    // these): reverting to the single `(Some(o), Some(c)) if c > o +
    // MARGIN` arm (deleting the `(None, Some(c))` route) fails
    // `endorses_a_garbled_original_on_absolute_candidate_strength` — it is
    // exactly the case that route exists for. Lowering
    // `ABSOLUTE_ENDORSE_THRESHOLD` to `0.0` fails
    // `declines_a_garbled_original_below_the_absolute_bar` (a weak, barely-
    // related candidate would wrongly endorse). Deleting the whole function
    // (hardcoding `true`) fails `declines_when_neither_side_has_a_code`.

    #[test]
    fn endorses_a_comparative_win_when_both_sides_have_codes() {
        assert!(GraphEngine::decide_endorse(
            Some(0.30),
            Some(0.30 + GraphEngine::ENDORSE_MARGIN + 0.001)
        ));
    }

    #[test]
    fn declines_a_comparative_non_win_when_both_sides_have_codes() {
        // Candidate is HIGHER but not by more than the margin — must not
        // flip on noise-scale differences.
        assert!(!GraphEngine::decide_endorse(
            Some(0.30),
            Some(0.30 + GraphEngine::ENDORSE_MARGIN - 0.001)
        ));
        // Candidate is actually WORSE than the original.
        assert!(!GraphEngine::decide_endorse(Some(0.50), Some(0.20)));
    }

    #[test]
    fn endorses_a_garbled_original_on_absolute_candidate_strength() {
        // The route this session's own real-corruption demo found missing:
        // "grasz" (garbage, no code) -> lexical candidate "grass" (real
        // word, sim 0.688 to "garden" in the same sentence, measured on the
        // real trained codebook) must be endorsed even though the original
        // has no comparable score at all.
        assert!(GraphEngine::decide_endorse(
            None,
            Some(GraphEngine::ABSOLUTE_ENDORSE_THRESHOLD + 0.05)
        ));
    }

    #[test]
    fn declines_a_garbled_original_below_the_absolute_bar() {
        assert!(!GraphEngine::decide_endorse(
            None,
            Some(GraphEngine::ABSOLUTE_ENDORSE_THRESHOLD - 0.05)
        ));
    }

    #[test]
    fn declines_when_neither_side_has_a_code() {
        assert!(!GraphEngine::decide_endorse(None, None));
        // Original coincidentally has a code but the lexical layer proposed
        // nothing (correction.rs itself declined) — never endorse from a
        // one-sided original-only signal.
        assert!(!GraphEngine::decide_endorse(Some(0.9), None));
    }

    #[test]
    fn map_pos_drops_negation_to_other_not_a_core_slot() {
        assert_eq!(GraphEngine::map_pos(V1Pos::Negation, "not"), V2Pos::Other);
    }

    #[test]
    fn normalize_strips_punctuation_and_case() {
        assert_eq!(GraphEngine::normalize("Cat."), "cat");
        assert_eq!(GraphEngine::normalize("\"door\""), "door");
    }

    /// Real vocab/codebook asset paths, or `None` if any is missing —
    /// graceful-skip, matching `reasoning.rs`'s own established pattern for
    /// real-data tests in this workspace.
    fn data_paths() -> Option<(
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )> {
        let vocab_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../lance-graph/crates/deepnsm/word_frequency");
        let cam_dir = std::env::var("DEEPNSM_V2_CAM96_DIR").ok()?;
        let cam_dir = std::path::PathBuf::from(cam_dir);
        let bible_vocab = cam_dir.join("bible_vocab.txt");
        let codebook = cam_dir.join("cam96_codebook.bin");
        let codes = cam_dir.join("cam96_codes.bin");
        if !vocab_dir.join("word_rank_lookup.csv").exists()
            || !bible_vocab.exists()
            || !codebook.exists()
            || !codes.exists()
        {
            return None;
        }
        Some((vocab_dir, bible_vocab, codebook, codes))
    }

    /// A real `DocPage` for `corpus/pages/page_01.pgm` via the sanctioned
    /// executor path (`OcrExecutor::execute(RecognizePageWords)` →
    /// `DocPage::from_line_words`) — `DocPage` cannot be hand-built from raw
    /// strings (`from_line_words` needs a real `CharSet` and
    /// `WordResult`-shaped `LineWords`), so this is the one way an external
    /// caller reaches one, mirroring `ocr_demo.rs`'s own composition.
    fn real_page() -> Option<DocPage> {
        let model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
        if !model.join("eng.lstm").exists() {
            eprintln!("real_page: skipping — {} not present", model.display());
            return None;
        }
        let dawg = |name: &str| {
            let p = model.join(name);
            p.exists().then_some(p)
        };
        let executor = tesseract_ogar::OcrExecutor::from_data_paths(
            &model.join("eng.lstm"),
            &model.join("eng.lstm-unicharset"),
            &model.join("eng.lstm-recoder"),
            dawg("eng.lstm-word-dawg").as_deref(),
            dawg("eng.lstm-punc-dawg").as_deref(),
            dawg("eng.lstm-number-dawg").as_deref(),
        )
        .expect("load the eng recognizer");

        let img = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../corpus/pages/page_01.pgm");
        let bytes = std::fs::read(&img).expect("read page_01.pgm");
        let (grey, w, h) = tesseract_ogar::parse_pgm(&bytes).expect("parse P5 pgm");

        let words = match executor
            .execute(tesseract_ogar::OcrRequest::RecognizePageWords {
                grey: &grey,
                width: w,
                height: h,
                with_dict: true,
            })
            .expect("execute recognize_page_words")
        {
            tesseract_ogar::OcrResponse::LineWordsOut(lines) => lines,
            other => panic!("unexpected response: {other:?}"),
        };
        Some(DocPage::from_line_words(
            &words,
            executor.charset(),
            u32::try_from(w).expect("page width fits u32"),
            u32::try_from(h).expect("page height fits u32"),
        ))
    }

    #[test]
    fn analyze_extracts_real_triples_from_a_recognized_page() {
        let Some((vocab_dir, bible_vocab, codebook, codes)) = data_paths() else {
            eprintln!("analyze_extracts_real_triples_from_a_recognized_page: skipping — real cam96 data assets not present (set DEEPNSM_V2_CAM96_DIR)");
            return;
        };
        let Some(page) = real_page() else {
            return;
        };
        let engine = GraphEngine::from_paths(&vocab_dir, &bible_vocab, &codebook, &codes)
            .expect("load real assets");

        let results = engine.analyze(&page);
        assert!(
            !results.is_empty(),
            "page_01 has 7 sentences; must not be dropped"
        );

        let total_triples: usize = results.iter().map(|g| g.triples.len()).sum();
        assert!(
            total_triples > 0,
            "page_01's simple SVO sentences (measured 79% vocabulary coverage \
             against the real KJV-trained codebook) must yield at least one \
             real triple through v2's actual FSM — zero would mean the \
             tagging/lookup wiring is broken, not that the text is unparseable"
        );

        // Anti-vacuity: at least one sentence must show LESS than full
        // vocabulary coverage — proving the OOV accounting is real, not a
        // silent always-100% pass (measured: "clock"/"ticked"/"cooled"/
        // "coffee"/"boots"/"hike"/"rack" are OOV against bible_vocab.txt).
        assert!(
            results.iter().any(|g| g.tokens_in_vocab < g.tokens_total),
            "page_01 measurably contains OOV words against the KJV vocabulary; \
             a coverage report showing 100% everywhere would be wrong"
        );
    }
}
