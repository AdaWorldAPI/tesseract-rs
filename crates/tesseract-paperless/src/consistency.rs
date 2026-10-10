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
//!   multi-reading FSM (not v1's parser) runs on every assembled
//!   sentence, producing role-typed (subject/predicate/object) triples
//!   addressed to `(page, line_indices, bbox)` — a document's own natural
//!   tree in place of the KJV's book:chapter:verse. **Honest scope:** v1
//!   (`SentenceReasoner::vocab`) only splits the sentence into surfaces; the
//!   readings come from v2's own lexical layer — the COCA lemma table first
//!   (F9), then every `word_forms.csv` reading, folded by
//!   [`deepnsm_v2::coca`] — and v2's multi-reading FSM
//!   ([`deepnsm_v2::fsm::parse_readings`]) keeps the readings the structure
//!   cannot separate. Triples found on every surviving reading are
//!   [`GraphSentence::triples`]; the rest are only counted
//!   ([`GraphSentence::alternative_triples`]). What v2 contributes beyond
//!   that is the clause machinery (relative clauses, subject-carry chaining)
//!   and a TRAINED distributional semantic space (`Nsm::word_similarity`). "Consistency recovery" here means:
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

use deepnsm_v2::coca::{fsm_pos_tag, reading_set};
use deepnsm_v2::codebook::{load_cam96_codes, load_cam96_space, CodebookError};
use deepnsm_v2::fsm::{parse_readings, Pos as V2Pos, PosSet, Reading};
#[cfg(test)]
use deepnsm_v2::fsm::{parse_to_spo, Tagged};
use deepnsm_v2::lexical::{load_word_forms_csv, LexicalEvidence};
use deepnsm_v2::vocab::WordId;
use deepnsm_v2::{Nsm, PaletteVocab};

pub use lance_graph_contract::exploration::NarsTruth;

use tesseract_ogar::correction::{suggest, CorrectionPolicy, Lexicon};
use tesseract_ogar::reasoning::{ReasoningError, SentenceReasoner};
use tesseract_ogar::sentences::{assemble_sentences, AssembledSentence};
#[cfg(test)]
use tesseract_ogar::PoS as V1Pos;
use tesseract_ogar::{DocPage, Token as V1Token};

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
    /// SPO triples the v2 FSM resolved from this sentence's tokens: those
    /// found on every reading the structure could not rule out.
    pub triples: Vec<GraphTriple>,
    /// Triples found on only some readings (a homograph the structure could
    /// not separate). Counted, not built: no confidence or truth is attached
    /// to a reading that may not hold.
    pub alternative_triples: usize,
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
    /// Loading v2's lexical evidence (`word_forms.csv`) failed.
    Evidence(deepnsm_v2::lexical::EvidenceError),
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
            Self::Evidence(e) => write!(f, "lexical evidence: {e}"),
            Self::Codebook(e) => write!(f, "cam96 codebook: {e:?}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for GraphEngineError {}

/// v2's lexical layer for `vocab`, from the COCA tables in `vocab_dir`: every
/// `word_forms.csv` reading of each in-vocabulary surface, and the lemma
/// table's first-row tag per lemma, both folded by [`deepnsm_v2::coca`].
fn load_lexical(
    vocab_dir: &Path,
    vocab: &PaletteVocab,
) -> Result<(LexicalEvidence, HashMap<String, V2Pos>), GraphEngineError> {
    let read = |name: &str| std::fs::read_to_string(vocab_dir.join(name));
    let lemmas_csv = read("lemmas_5k.csv").map_err(GraphEngineError::Io)?;
    let forms_csv = read("word_forms.csv").map_err(GraphEngineError::Io)?;
    let mut lemma_tags = HashMap::new();
    for line in lemmas_csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if let (Some(lemma), Some(pos)) = (f.get(1), f.get(2)) {
            lemma_tags
                .entry(lemma.to_lowercase())
                .or_insert_with(|| fsm_pos_tag(pos));
        }
    }
    let (evidence, _report) = load_word_forms_csv(&lowercase_word_column(&forms_csv), vocab)
        .map_err(GraphEngineError::Evidence)?;
    Ok((evidence, lemma_tags))
}

/// Lowercase the `word` column of `word_forms.csv`, leaving the header and the
/// other fields as they are: `load_word_forms_csv` matches surfaces exactly,
/// and the tokenizer lowercases. Same rule as deepnsm-v2's `bible_wave`.
fn lowercase_word_column(forms_csv: &str) -> String {
    let mut out = String::with_capacity(forms_csv.len());
    for (i, line) in forms_csv.lines().enumerate() {
        match line.rsplit_once(',') {
            Some((head, word)) if i > 0 => {
                out.push_str(head);
                out.push(',');
                out.push_str(&word.to_lowercase());
            }
            _ => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

/// Confidence threshold (0-100) below which a role-filler is a candidate for
/// consistency recovery, AND below which it is excluded from the "trusted
/// context" set used to judge OTHER roles in the same triple (`recover`
/// applies the identical comparison both ways — a role at or above this bar
/// is deemed simultaneously "doesn't need correction" and "good enough to
/// anchor a correction elsewhere").
///
/// **Measured, not a bare policy pin** (`examples/conf_cliff_probe.rs`, the
/// per-cell mean/min/max `DocWord::conf` on `corpus/quality/resgrid.pgm`'s
/// 16-cell resolution ladder, the same fixture `quality_resolution_grid.rs`
/// pins by CER). The confidence axis cliffs well before the CER axis does:
///
/// | cell | `mean_conf` | `min_conf` | CER |
/// |---|---|---|---|
/// | 0-12 (clean) | 99.45-99.54 | 99.32-99.47 | 0.000 |
/// | 13 (wobbling, still correct) | 98.11 | 93.09 | 0.000 |
/// | 14 (last legible rung) | 96.53 | 89.03 | 0.023 |
/// | 15 (confident garbage) | 90.04 | 80.01 | 0.814 |
///
/// The previous value (`70.0`) sat BELOW cell 15's mean (90.04) — the
/// pipeline's own "confident garbage" cell would have supplied trusted
/// context for correcting other roles. `95.0` sits strictly between cell
/// 12's floor (99.32, so genuinely clean text is never flagged) and cell
/// 15's mean (90.04, so confident-garbage-grade text reliably is).
///
/// **Scope, stated honestly:** this is ONE degradation axis (image
/// resolution/blur) on ONE fixture. It has not been checked against
/// illumination, faded-contrast, or skew degradation (this crate's siblings
/// have fixtures for those — `corpus/quality/{uneven,faded}_*.pgm` — but
/// `conf_cliff_probe.rs` has not been run against them). Re-measure before
/// defending this number outside the resolution axis.
pub const LOW_CONFIDENCE_THRESHOLD: f32 = 95.0;

/// A document's real path into lance-graph: v1's tagger (reused, not
/// reimplemented) feeding v2's real FSM and trained CAM-PQ 96 space.
pub struct GraphEngine {
    reasoner: SentenceReasoner,
    lexicon: Lexicon,
    policy: CorrectionPolicy,
    nsm: Nsm,
    /// v2's lexical layer over the same routing vocabulary: every COCA
    /// reading of every in-vocabulary surface (`word_forms.csv`).
    evidence: LexicalEvidence,
    /// The COCA lemma table (`lemmas_5k.csv`), first row per lemma, folded by
    /// [`deepnsm_v2::coca`] — consulted before the forms readings (F9).
    lemma_tags: HashMap<String, V2Pos>,
}

/// [`GraphEngine::tag_sentence`]'s result — its own struct purely for
/// readability (clippy's `type_complexity` on the equivalent 5-tuple);
/// every field is read at the one call site via destructuring.
struct TaggedSentence {
    readings: Vec<Reading>,
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

        let (evidence, lemma_tags) = load_lexical(vocab_dir, &nsm.vocab)?;

        Ok(Self {
            reasoner,
            lexicon,
            policy: CorrectionPolicy::default(),
            nsm,
            evidence,
            lemma_tags,
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
    #[cfg(test)]
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

    /// The v1 → v2 seam: each v1 token whose surface is in v2's routing
    /// vocabulary becomes one v2 [`Tagged`], tagged by [`Self::map_pos`].
    /// Tokens outside the vocabulary are dropped here (the FSM never sees
    /// them); the returned index is the token's position in `tokens`, so a
    /// caller can line it up with per-token data such as OCR confidence.
    ///
    /// Pure and asset-free, so the seam can be tested without the cam96
    /// release data.
    #[cfg(test)]
    fn seam_tags(vocab: &PaletteVocab, tokens: &[V1Token]) -> Vec<(usize, Tagged)> {
        tokens
            .iter()
            .enumerate()
            .filter_map(|(i, tok)| {
                let id = vocab.id(&tok.surface)?;
                Some((i, Tagged::new(id, Self::map_pos(tok.pos, &tok.surface))))
            })
            .collect()
    }

    /// The v2 lexical seam: each v1 token whose surface is in v2's routing
    /// vocabulary becomes one v2 [`Reading`] carrying EVERY reading v2's
    /// lexical layer admits. v1 contributes only the surface split; its tags
    /// are not read.
    ///
    /// Readings, in order: a relativizer surface (`that`, `which`, `who`,
    /// `whom`, `whose`) is [`V2Pos::Rel`]; else the lemma table's tag (F9);
    /// else every [`LexicalEvidence`] reading, folded by [`deepnsm_v2::coca`];
    /// else [`PosSet::EMPTY`] (unknown — never a guessed reading).
    fn seam_readings(
        vocab: &PaletteVocab,
        evidence: &LexicalEvidence,
        lemma_tags: &HashMap<String, V2Pos>,
        tokens: &[V1Token],
    ) -> Vec<(usize, Reading)> {
        tokens
            .iter()
            .enumerate()
            .filter_map(|(i, tok)| {
                let id = vocab.id(&tok.surface)?;
                let s = tok.surface.as_str();
                let pos = if matches!(s, "that" | "which" | "who" | "whom" | "whose") {
                    PosSet::single(V2Pos::Rel)
                } else if let Some(&p) = lemma_tags.get(s) {
                    PosSet::single(p)
                } else {
                    reading_set(evidence, id).unwrap_or(PosSet::EMPTY)
                };
                Some((i, Reading::new(id, pos)))
            })
            .collect()
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

        let seam = Self::seam_readings(&self.nsm.vocab, &self.evidence, &self.lemma_tags, &tokens);
        let mut readings = Vec::with_capacity(seam.len() + 1);
        let mut conf_by_id: HashMap<WordId, (f32, u32)> = HashMap::new();
        let tokens_in_vocab = seam.len();

        for (i, t) in seam {
            let conf = if well_aligned {
                flat_confs[i]
            } else {
                sentence.mean_conf
            };
            let entry = conf_by_id.entry(t.id).or_insert((0.0, 0));
            entry.0 += conf;
            entry.1 += 1;
            readings.push(t);
        }
        readings.push(Reading::stop());

        TaggedSentence {
            readings,
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
                    readings,
                    conf_by_id,
                    tokens_total,
                    tokens_in_vocab,
                    well_aligned,
                } = self.tag_sentence(page, &sentence);
                let parse = parse_readings(&readings);
                let alternative_triples = parse.alternative.len();
                let spos = parse.certain;

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
                    alternative_triples,
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

    // ── LOW_CONFIDENCE_THRESHOLD: measured against the real resgrid.pgm
    // resolution ladder (`examples/conf_cliff_probe.rs`), not asserted.
    // Disable-verified: reverting to the old pin (70.0) makes
    // `confident_garbage_grade_confidence_is_flagged_not_trusted_as_context`
    // fail (90.04 >= 70.0, so cell 15's mean would NOT have been flagged —
    // exactly the defect this constant was raised to close).

    #[test]
    fn clean_grade_confidence_is_never_flagged() {
        // Cell 12's floor (every word across the clean 0-12 range sits at
        // or above this) must never be treated as needing correction.
        let cell_12_floor_min_conf: f32 = 99.32;
        assert!(
            cell_12_floor_min_conf >= LOW_CONFIDENCE_THRESHOLD,
            "a perfectly-recognized cell's worst word must not be flagged"
        );
    }

    #[test]
    fn confident_garbage_grade_confidence_is_flagged_not_trusted_as_context() {
        // Cell 15's mean (CER 0.814 — "decodes to confident garbage" per
        // quality_resolution_grid.rs's own doc). Must fall below the bar so
        // `recover()` neither skips correcting it NOR uses it as trusted
        // context for another role in the same triple.
        let cell_15_mean_conf: f32 = 90.04;
        assert!(
            cell_15_mean_conf < LOW_CONFIDENCE_THRESHOLD,
            "confident-garbage-grade confidence must be flagged; the old \
             70.0 pin let 90.04 through as trusted context"
        );
        // The regression this constant closes, stated directly: the OLD
        // value would have compared as "clean enough".
        let old_threshold: f32 = 70.0;
        assert!(
            cell_15_mean_conf >= old_threshold,
            "if this fails, the OLD value already caught the case — the \
             fixture numbers changed and this test needs re-deriving, not \
             the threshold"
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

    /// The committed v1 COCA vocabulary in the sibling lance-graph checkout.
    /// The seam test needs no other asset, so a missing directory is a
    /// broken checkout and fails loudly rather than skipping.
    fn v1_vocab_dir() -> std::path::PathBuf {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../lance-graph/crates/deepnsm/word_frequency");
        assert!(
            dir.join("word_rank_lookup.csv").exists(),
            "v1 vocabulary missing at {} — the lance-graph sibling checkout is required",
            dir.display()
        );
        dir
    }

    /// One fixed sentence through the current v1 → v2 seam: v1 tokenizer and
    /// tags, [`GraphEngine::map_pos`], a v2 routing vocabulary built from the
    /// fixture's own surfaces, then v2's [`parse_to_spo`]. Rendered as
    /// `surface:Pos …` plus the triples as words, so a pin reads as text.
    fn seam_snapshot(reasoner: &SentenceReasoner, text: &str) -> String {
        let tokens = reasoner.vocab().tokenize(text);
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(tokens.iter().map(|t| t.surface.as_str()));
        let seam = GraphEngine::seam_tags(&vocab, &tokens);
        let mut tagged: Vec<Tagged> = seam.iter().map(|(_, t)| *t).collect();
        tagged.push(Tagged::new(0, V2Pos::Stop));
        let word = |id: WordId| vocab.word(id).unwrap_or("?");
        let tags: Vec<String> = seam
            .iter()
            .map(|(_, t)| format!("{}:{:?}", word(t.id), t.pos))
            .collect();
        let triples: Vec<String> = parse_to_spo(&tagged)
            .into_iter()
            .map(|s| {
                format!(
                    "({},{},{})",
                    word(s.subject),
                    word(s.predicate),
                    word(s.object)
                )
            })
            .collect();
        format!("{} | {}", tags.join(" "), triples.join(" "))
    }

    /// W0 seam pin: the CURRENT v1-tag → `map_pos` → v2 FSM behaviour on fixed
    /// input, recorded before the multi-reading decoder lands. It must go red
    /// if the v1 tagger, `map_pos` or the v2 FSM changes what this path emits.
    ///
    /// Recorded facts, not endorsements:
    /// - COCA `d` (determiner) never leaves `map_pos` as `Det`: `some` arrives
    ///   from v1 as `Modal`, `this` and `all` as `Adverb` (an earlier COCA row
    ///   wins), and all three leave as `Other`.
    /// - Relativizers become `Rel` by surface; pronouns become `Noun`.
    /// - v1's context-free tagger reads `slept` as a noun, so the relative
    ///   clause yields the wrong triple `(slept,woke,man)`.
    /// - `record` arrives as one tag (`Verb`) wherever it stands, so "that
    ///   record" closes no triple.
    #[test]
    fn v1_to_v2_seam_is_pinned_on_fixed_input() {
        let reasoner = SentenceReasoner::from_vocab_dir(&v1_vocab_dir()).expect("load v1 vocab");
        let cases: [(&str, &str); 4] = [
            ("This dog saw all the men.", "this:Other dog:Noun saw:Verb all:Other the:Det men:Noun | (dog,saw,men)"),
            ("The man who slept woke the child.", "the:Det man:Noun who:Rel slept:Noun woke:Verb the:Det child:Noun | (slept,woke,man)"),
            ("They record the deeds.", "they:Noun record:Verb the:Det deeds:Noun | (they,record,deeds)"),
            ("Some people found that record.", "some:Other people:Noun found:Verb that:Rel record:Verb | "),
        ];
        let got: Vec<String> = cases
            .iter()
            .map(|(text, _)| seam_snapshot(&reasoner, text))
            .collect();
        let want: Vec<&str> = cases.iter().map(|(_, w)| *w).collect();
        assert_eq!(got, want, "the v1 → v2 seam drifted");
    }

    /// A token outside the v2 vocabulary is dropped, and the tokens after it
    /// keep their ORIGINAL v1 positions: `tag_sentence` indexes per-word OCR
    /// confidence by these positions.
    #[test]
    fn seam_tags_skips_oov_and_keeps_original_positions() {
        let reasoner = SentenceReasoner::from_vocab_dir(&v1_vocab_dir()).expect("load v1 vocab");
        let tokens = reasoner.vocab().tokenize("The dog saw the men.");
        assert_eq!(tokens.len(), 5, "fixture tokenization changed");
        let mut vocab = PaletteVocab::new();
        // `saw` (position 2) is left out of the v2 vocabulary.
        vocab.from_frequency_ranked(["the", "dog", "men"]);
        let seam = GraphEngine::seam_tags(&vocab, &tokens);
        let got: Vec<(usize, &str)> = seam
            .iter()
            .map(|(i, t)| (*i, vocab.word(t.id).unwrap_or("?")))
            .collect();
        assert_eq!(got, [(0, "the"), (1, "dog"), (3, "the"), (4, "men")]);
    }

    /// The same guarantee on the production path: [`GraphEngine::seam_readings`]
    /// drops a token outside the v2 vocabulary and keeps the original v1
    /// positions of the tokens after it.
    #[test]
    fn seam_readings_skips_oov_and_keeps_original_positions() {
        let reasoner = SentenceReasoner::from_vocab_dir(&v1_vocab_dir()).expect("load v1 vocab");
        let tokens = reasoner.vocab().tokenize("The dog saw the men.");
        assert_eq!(tokens.len(), 5, "fixture tokenization changed");
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(["the", "dog", "men"]);
        let (evidence, lemma_tags) = load_lexical(&v1_vocab_dir(), &vocab).expect("lexical layer");
        let seam = GraphEngine::seam_readings(&vocab, &evidence, &lemma_tags, &tokens);
        let got: Vec<(usize, &str)> = seam
            .iter()
            .map(|(i, r)| (*i, vocab.word(r.id).unwrap_or("?")))
            .collect();
        assert_eq!(got, [(0, "the"), (1, "dog"), (3, "the"), (4, "men")]);
    }

    /// `PosSet` as `Noun|Verb`, in the FSM's tag order.
    fn set_name(set: PosSet) -> String {
        if set.is_empty() {
            return "unknown".into();
        }
        set.iter()
            .map(|p| format!("{p:?}"))
            .collect::<Vec<_>>()
            .join("|")
    }

    /// The same fixed sentence through the v2 lexical seam
    /// ([`GraphEngine::seam_readings`] → [`parse_readings`]): readings per
    /// token as `surface:entered>survived` where the structure narrowed them,
    /// then the certain triples and, after `~`, the alternatives.
    fn v2_seam_snapshot(reasoner: &SentenceReasoner, text: &str) -> String {
        let tokens = reasoner.vocab().tokenize(text);
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(tokens.iter().map(|t| t.surface.as_str()));
        let (evidence, lemma_tags) = load_lexical(&v1_vocab_dir(), &vocab).expect("lexical layer");
        let seam = GraphEngine::seam_readings(&vocab, &evidence, &lemma_tags, &tokens);
        let mut readings: Vec<Reading> = seam.iter().map(|(_, r)| *r).collect();
        readings.push(Reading::stop());
        let parse = parse_readings(&readings);
        let word = |id: WordId| vocab.word(id).unwrap_or("?");
        let tags: Vec<String> = seam
            .iter()
            .enumerate()
            .map(|(k, (_, r))| {
                let narrowed = parse
                    .ambiguous
                    .iter()
                    .find(|sv| sv.index == k && sv.survived != sv.entered);
                match narrowed {
                    Some(sv) => format!(
                        "{}:{}>{}",
                        word(r.id),
                        set_name(sv.entered),
                        set_name(sv.survived)
                    ),
                    None => format!("{}:{}", word(r.id), set_name(r.pos)),
                }
            })
            .collect();
        let fmt = |v: &[deepnsm_v2::Spo]| -> String {
            v.iter()
                .map(|s| {
                    format!(
                        "({},{},{})",
                        word(s.subject),
                        word(s.predicate),
                        word(s.object)
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        format!(
            "{} | {} ~ {}",
            tags.join(" "),
            fmt(&parse.certain),
            fmt(&parse.alternative)
        )
    }

    /// T8 — the seam after D-LXC-2: v2's own lexical layer and multi-reading
    /// FSM on the four sentences pinned by
    /// [`v1_to_v2_seam_is_pinned_on_fixed_input`], plus two that exercise a
    /// homograph the lemma table does not decide. Measured change against
    /// that pin:
    /// - `this`/`all`/`some` arrive as `Det` (COCA `d`), not `Other`.
    /// - `slept` is a verb, so the relative clause is intransitive and the
    ///   matrix triple is `(man,woke,child)` instead of `(slept,woke,man)`.
    /// - "found that record" now closes `(people,found,record)`.
    /// - Regression, recorded: "They record the deeds" loses its triple. The
    ///   lemma table's first row tags `record` a noun (F9, kept by D-LXC-3)
    ///   and `deeds` is not in the COCA tables, so nothing is left to close
    ///   the clause; v1's own tagger had read `record` as a verb.
    /// - `locks` (not in the lemma table; noun and verb in `word_forms.csv`)
    ///   loses its verb reading after `the`, and keeps both after a subject
    ///   noun, where its triple is an alternative.
    #[test]
    fn v2_lexical_seam_is_pinned_on_fixed_input() {
        let reasoner = SentenceReasoner::from_vocab_dir(&v1_vocab_dir()).expect("load v1 vocab");
        let cases: [(&str, &str); 6] = [
            (
                "This dog saw all the men.",
                "this:Det dog:Noun saw:Verb all:Det the:Det men:Noun | (dog,saw,men) ~ ",
            ),
            (
                "The man who slept woke the child.",
                "the:Det man:Noun who:Rel slept:Verb woke:Verb the:Det child:Noun | (man,woke,child) ~ ",
            ),
            (
                "They record the deeds.",
                "they:Noun record:Noun the:Det deeds:unknown |  ~ ",
            ),
            (
                "Some people found that record.",
                "some:Det people:Noun found:Verb that:Rel record:Noun | (people,found,record) ~ ",
            ),
            (
                "The locks held the door.",
                "the:Det locks:Noun|Verb>Noun held:Verb the:Det door:Noun | (locks,held,door) ~ ",
            ),
            (
                "The guard locks the door.",
                "the:Det guard:Noun locks:Noun|Verb the:Det door:Noun |  ~ (guard,locks,door)",
            ),
        ];
        let got: Vec<String> = cases
            .iter()
            .map(|(text, _)| v2_seam_snapshot(&reasoner, text))
            .collect();
        let want: Vec<&str> = cases.iter().map(|(_, w)| *w).collect();
        assert_eq!(got, want, "the v2 lexical seam drifted");
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
