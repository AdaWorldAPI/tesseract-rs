//! Row builder for the `AUTO` matching tier: plain archive metadata in,
//! [`ArchiveRow`]s and a [`Mapping`] out.
//!
//! ```text
//!   PURE.  NO I/O, NO TANTIVY, NO LANCEDB, NO `DocumentRow`.
//! ```
//!
//! [`crate::auto_match`] mines rows of dense ids. The archive stores something
//! else: sparse `definition_id`s that are never reused, a `retired` flag, an
//! `is_auto` flag, assignments keyed by content hash, and free text. This
//! module is the translation between the two, and nothing more. It is the
//! spec's R4 (row builder), R5 (term vocabulary) and R6 (field keys)
//! (`.claude/plans/archive-metadata-auto-match-v3.md`).
//!
//! # Why it is pure
//!
//! Feature `auto-match` must not drag in the storage engine or the search
//! index, so the module takes plain structs ([`Definition`], [`Assignment`],
//! [`DocContent`]) and never a `DocumentRow`. The glue that loads those structs
//! from the archive lives elsewhere, under the wider feature gate.
//!
//! **Tokenization is injected.** Every entry point takes a
//! `&dyn Fn(&str) -> Vec<String>`. Nothing here knows which analyzer produced
//! the tokens, so the search index's own analyzer can be wired in by the glue
//! (spec R5, firewall F2) without this module importing it. The same closure
//! must be used to build a [`TrainingSet`] and later to call
//! [`Mapping::row_for`]; a second tokenizer would count a different population.
//!
//! # The firewall: an AUTO definition is never a cue (F2)
//!
//! [`Mapping::cue_allowed`] answers `false` for every AUTO definition and
//! [`Mapping::target_eligible`] answers `true` only for AUTO ones. Passed to
//! [`crate::auto_match::MineScope`], that closes the accept, cue, accept loop:
//! a suggestion is never the evidence for another suggestion, so lift is never
//! measured on labels the rules themselves produced.
//!
//! # The retired category
//!
//! A single-valued kind (correspondent, document type) gets ONE extra dense id
//! past the live definitions, "retired", shared by every retired definition of
//! that kind. A document on a retired correspondent is therefore still
//! "assigned": it gets no correspondent suggestion, and it is not mined as
//! unassigned (spec R3, the overclaim R4 note). The category carries no
//! meaning, so it is neither an eligible target nor an allowed cue. Tags have
//! no such category: a retired tag's assignment is dropped from the row.
//!
//! # Policy pins, not measurements
//!
//! [`VocabParams::default`] holds four policy pins (length 3, no digits,
//! `min_df` 5, at most half the reviewed documents). None has been measured on
//! a real archive. Each has an inertness test: switching it off admits a
//! specific token.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::auto_match::{ArchiveDomains, ArchiveRow, Cue, Target, MAX_BINARY_FEATURES};

/// Which single- or multi-valued metadata family a definition belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// Who a document is from. Single-valued.
    Correspondent,
    /// What sort of document it is. Single-valued.
    DocumentType,
    /// A label. A document carries a set of them.
    Tag,
}

/// One row of the definitions table, reduced to what the row builder reads.
///
/// `definition_id` is consumer-local filing metadata (spec F10): it is never a
/// classid and is unique per [`Kind`], not across kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    /// Which family this definition belongs to.
    pub kind: Kind,
    /// Stable id within `kind`. Sparse: retired ids are never reallocated.
    pub definition_id: u32,
    /// Display name. Not read by the builder.
    pub name: String,
    /// Is this definition's matching algorithm AUTO? Only these are targets.
    pub is_auto: bool,
    /// A retired definition is never a target and never a cue.
    pub retired: bool,
}

/// One document carrying one definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    /// The document, by content hash (hex).
    pub content_sha256_hex: String,
    /// Which family `definition_id` belongs to.
    pub kind: Kind,
    /// The assigned definition.
    pub definition_id: u32,
}

/// The observable content of one document, as plain data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocContent {
    /// The document, by content hash (hex).
    pub content_sha256_hex: String,
    /// The document's text. Tokenized by the injected closure.
    pub text: String,
    /// The custom-field keys set on the document. Duplicates are harmless.
    pub field_keys: Vec<String>,
}

/// The four token filters of spec R5.
///
/// **Policy pins, not measurements.** None has been measured on a real
/// archive; each is proven live by a test that switches it off and watches a
/// specific token appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabParams {
    /// A token shorter than this many `char`s is dropped.
    pub min_token_len: usize,
    /// Drop any token containing an ASCII digit.
    pub reject_digits: bool,
    /// A token must occur in at least this many reviewed documents.
    pub min_df: u32,
    /// A token may occur in at most this many parts per million of the
    /// reviewed documents. The comparison is `df * 1_000_000 <= ppm * n`, in
    /// `u64`, so no rounding decides it.
    pub max_df_ppm: u32,
}

impl Default for VocabParams {
    /// Policy pins (3 chars, no digits, `min_df` 5, at most 50%), not values
    /// measured on an archive.
    fn default() -> Self {
        Self {
            min_token_len: 3,
            reject_digits: true,
            min_df: 5,
            max_df_ppm: 500_000,
        }
    }
}

/// Why a training set could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowBuildError {
    /// No content term survived the filters and the feature budget. A model
    /// with no content cues cannot fire on a new upload, so this is refused
    /// rather than mined.
    NoContentCues,
    /// The non-retired tags alone exceed [`MAX_BINARY_FEATURES`].
    TooManyFeatures {
        /// The binary feature count the tags asked for.
        features: u64,
        /// The cap.
        max: u32,
    },
}

impl core::fmt::Display for RowBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoContentCues => write!(
                f,
                "no content term survived the vocabulary filters and the feature budget"
            ),
            Self::TooManyFeatures { features, max } => {
                write!(f, "{features} tags exceed the cap of {max} binary features")
            }
        }
    }
}

impl std::error::Error for RowBuildError {}

/// Everything a mining run needs: the domains, the rows, and the mapping that
/// built them.
#[derive(Debug, Clone)]
pub struct TrainingSet {
    /// Sizes of the dense id domains.
    pub domains: ArchiveDomains,
    /// One row per reviewed document, in the order given.
    pub rows: Vec<ArchiveRow>,
    /// The dense-id maps this run used; needed again at suggestion time.
    pub mapping: Mapping,
}

/// Where a `definition_id` lands in one kind's dense id space.
enum Slot {
    /// A live definition, at this dense id.
    Live(u32),
    /// A retired definition of a kind that has a retired category.
    Retired,
    /// Not a definition of this kind, or retired in a kind with no category.
    Dropped,
}

/// The dense id space of one [`Kind`].
#[derive(Debug, Clone)]
struct KindMap {
    kind: Kind,
    /// Non-retired definitions, sorted by `definition_id`. The index is the
    /// dense id; the value is `(definition_id, is_auto)`.
    live: Vec<(u32, bool)>,
    /// Retired `definition_id`s of this kind.
    retired: BTreeSet<u32>,
}

impl KindMap {
    /// Build the map for `kind` from `defs`, ignoring other kinds. A repeated
    /// `(kind, definition_id)` keeps the last one seen.
    fn from_defs(kind: Kind, defs: &[Definition]) -> Self {
        let mut by_id: BTreeMap<u32, &Definition> = BTreeMap::new();
        for d in defs.iter().filter(|d| d.kind == kind) {
            by_id.insert(d.definition_id, d);
        }
        let mut live = Vec::new();
        let mut retired = BTreeSet::new();
        for (id, d) in by_id {
            if d.retired {
                retired.insert(id);
            } else {
                live.push((id, d.is_auto));
            }
        }
        Self {
            kind,
            live,
            retired,
        }
    }

    /// Does this kind have a shared retired category (single-valued kinds)?
    fn has_retired_category(&self) -> bool {
        self.kind != Kind::Tag
    }

    /// The dense id of the shared retired category: one past the live ones.
    fn retired_dense(&self) -> u32 {
        count_u32(self.live.len())
    }

    /// The size of this kind's dense id domain, retired category included.
    fn domain(&self) -> u32 {
        if self.has_retired_category() {
            self.retired_dense().saturating_add(1)
        } else {
            self.retired_dense()
        }
    }

    /// Classify `definition_id`.
    fn slot(&self, definition_id: u32) -> Slot {
        if let Ok(i) = self.live.binary_search_by_key(&definition_id, |d| d.0) {
            return u32::try_from(i).map_or(Slot::Dropped, Slot::Live);
        }
        if self.has_retired_category() && self.retired.contains(&definition_id) {
            Slot::Retired
        } else {
            Slot::Dropped
        }
    }

    /// The live definition at dense id `dense`, if any: `(definition_id, is_auto)`.
    fn get(&self, dense: u32) -> Option<(u32, bool)> {
        usize::try_from(dense)
            .ok()
            .and_then(|i| self.live.get(i))
            .copied()
    }
}

/// A `u32` count, saturating at `u32::MAX`.
fn count_u32(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// The dense-id maps of one mining run.
///
/// Rebuilt per run: dense ids are only meaningful together with the
/// [`ArchiveDomains`] and the rows they were built with. Keep the
/// [`TrainingSet`]'s `mapping` next to the [`crate::auto_match::AutoMatcher`]
/// it trained.
#[derive(Debug, Clone)]
pub struct Mapping {
    correspondents: KindMap,
    document_types: KindMap,
    tags: KindMap,
    /// Dense term id to token, sorted lexicographically.
    terms: Vec<String>,
    term_index: HashMap<String, u32>,
    /// Dense field-key id to key, sorted lexicographically.
    field_keys: Vec<String>,
    key_index: HashMap<String, u32>,
}

impl Mapping {
    /// The row for one document: its own `assignments`, its text through
    /// `tokenize`, and its field keys, all mapped to this run's dense ids.
    ///
    /// `assignments` may be a larger slice; only entries whose hash equals
    /// `doc.content_sha256_hex` are read. Anything unknown to this run's
    /// vocabulary or definitions is ignored, so a suggestion-time row can be
    /// built for a document that was never part of training.
    ///
    /// Correspondent and document type are single-valued. The store enforces
    /// that (spec R2); if several arrive anyway, the one with the smallest
    /// `definition_id` wins, as a defensive tie-break and nothing more.
    #[must_use]
    pub fn row_for(
        &self,
        assignments: &[Assignment],
        doc: &DocContent,
        tokenize: &dyn Fn(&str) -> Vec<String>,
    ) -> ArchiveRow {
        self.row_from(
            assignments
                .iter()
                .filter(|a| a.content_sha256_hex == doc.content_sha256_hex),
            doc,
            tokenize,
        )
    }

    /// [`Self::row_for`] over an iterator of this document's assignments.
    fn row_from<'a>(
        &self,
        own: impl Iterator<Item = &'a Assignment>,
        doc: &DocContent,
        tokenize: &dyn Fn(&str) -> Vec<String>,
    ) -> ArchiveRow {
        let own: Vec<&Assignment> = own.collect();
        let single = |map: &KindMap| -> Option<u32> {
            own.iter()
                .filter(|a| a.kind == map.kind)
                .filter(|a| !matches!(map.slot(a.definition_id), Slot::Dropped))
                .min_by_key(|a| a.definition_id)
                .and_then(|a| match map.slot(a.definition_id) {
                    Slot::Live(d) => Some(d),
                    Slot::Retired => Some(map.retired_dense()),
                    Slot::Dropped => None,
                })
        };
        let tags: BTreeSet<u32> = own
            .iter()
            .filter(|a| a.kind == Kind::Tag)
            .filter_map(|a| match self.tags.slot(a.definition_id) {
                Slot::Live(d) => Some(d),
                Slot::Retired | Slot::Dropped => None,
            })
            .collect();
        let terms: BTreeSet<u32> = tokenize(&doc.text)
            .iter()
            .filter_map(|t| self.term_index.get(t).copied())
            .collect();
        let field_keys: BTreeSet<u32> = doc
            .field_keys
            .iter()
            .filter_map(|k| self.key_index.get(k).copied())
            .collect();
        ArchiveRow {
            correspondent: single(&self.correspondents),
            document_type: single(&self.document_types),
            tags: tags.into_iter().collect(),
            field_keys: field_keys.into_iter().collect(),
            terms: terms.into_iter().collect(),
        }
    }

    /// May `t` be the consequent of a rule? True iff it maps to a non-retired
    /// definition with `is_auto`. The retired category is never eligible.
    #[must_use]
    pub fn target_eligible(&self, t: Target) -> bool {
        let (map, dense) = match t {
            Target::Correspondent(c) => (&self.correspondents, c),
            Target::DocumentType(c) => (&self.document_types, c),
            Target::Tag(c) => (&self.tags, c),
        };
        map.get(dense).is_some_and(|(_, auto)| auto)
    }

    /// May `c` be the antecedent of a rule?
    ///
    /// Terms and field keys always. A correspondent, type or tag only if it
    /// maps to a NON-auto, non-retired definition: an AUTO definition is never
    /// a cue (F2), and the retired category carries no meaning.
    #[must_use]
    pub fn cue_allowed(&self, c: Cue) -> bool {
        let (map, dense) = match c {
            Cue::Term(_) | Cue::FieldKey(_) => return true,
            Cue::Correspondent(d) => (&self.correspondents, d),
            Cue::DocumentType(d) => (&self.document_types, d),
            Cue::Tag(d) => (&self.tags, d),
        };
        map.get(dense).is_some_and(|(_, auto)| !auto)
    }

    /// The definition a dense target stands for: `(kind, definition_id)`.
    /// `None` for the retired category and for an out-of-range id.
    #[must_use]
    pub fn definition_of(&self, t: Target) -> Option<(Kind, u32)> {
        let (map, dense) = match t {
            Target::Correspondent(c) => (&self.correspondents, c),
            Target::DocumentType(c) => (&self.document_types, c),
            Target::Tag(c) => (&self.tags, c),
        };
        map.get(dense).map(|(id, _)| (map.kind, id))
    }

    /// Dense term id to token. Sorted lexicographically.
    #[must_use]
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// Dense field-key id to key. Sorted lexicographically.
    #[must_use]
    pub fn field_keys(&self) -> &[String] {
        &self.field_keys
    }
}

/// Does a token with document frequency `df` over `n` reviewed documents pass
/// the frequency pins?
fn df_admits(df: u32, n: u64, vocab: &VocabParams) -> bool {
    df >= vocab.min_df && u64::from(df) * 1_000_000 <= u64::from(vocab.max_df_ppm).saturating_mul(n)
}

/// Document frequency: each document counts each distinct item once.
fn count_df(per_doc: impl Iterator<Item = Vec<String>>) -> BTreeMap<String, u32> {
    let mut df: BTreeMap<String, u32> = BTreeMap::new();
    for items in per_doc {
        let distinct: BTreeSet<String> = items.into_iter().collect();
        for item in distinct {
            *df.entry(item).or_insert(0) += 1;
        }
    }
    df
}

/// Keep the `budget` highest-DF items (ties: lexicographic ascending), then
/// sort them lexicographically. The sorted order is the dense id order.
fn top_by_df(df: BTreeMap<String, u32>, budget: u32) -> Vec<String> {
    let mut ranked: Vec<(String, u32)> = df.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(usize::try_from(budget).unwrap_or(usize::MAX));
    let mut kept: Vec<String> = ranked.into_iter().map(|(item, _)| item).collect();
    kept.sort();
    kept
}

/// Index a sorted list: item to its position.
fn index_of(items: &[String]) -> HashMap<String, u32> {
    items
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), count_u32(i)))
        .collect()
}

/// Turn plain archive metadata into mining input.
///
/// Only `reviewed` documents become rows, and the vocabulary is counted over
/// that one population (spec R3a, R5). Assignments for other documents are
/// read by nobody.
///
/// The binary feature budget is [`MAX_BINARY_FEATURES`], spent in this order:
/// tags (every non-retired one), then field keys, then content terms. Keys and
/// terms each keep their highest-DF survivors (ties: lexicographic) within
/// what is left.
///
/// # Errors
///
/// - [`RowBuildError::TooManyFeatures`] when the non-retired tags alone exceed
///   the cap.
/// - [`RowBuildError::NoContentCues`] when no content term survives the filters
///   and the remaining budget.
pub fn build_training(
    defs: &[Definition],
    assignments: &[Assignment],
    reviewed: &[DocContent],
    tokenize: &dyn Fn(&str) -> Vec<String>,
    vocab: VocabParams,
) -> Result<TrainingSet, RowBuildError> {
    let correspondents = KindMap::from_defs(Kind::Correspondent, defs);
    let document_types = KindMap::from_defs(Kind::DocumentType, defs);
    let tags = KindMap::from_defs(Kind::Tag, defs);

    let tag_count = tags.live.len();
    let tag_budget = u32::try_from(tag_count)
        .ok()
        .filter(|&t| t <= MAX_BINARY_FEATURES)
        .ok_or(RowBuildError::TooManyFeatures {
            features: u64::try_from(tag_count).unwrap_or(u64::MAX),
            max: MAX_BINARY_FEATURES,
        })?;
    // Safe: `tag_budget <= MAX_BINARY_FEATURES` was just enforced.
    let remaining = MAX_BINARY_FEATURES - tag_budget;

    let n = u64::try_from(reviewed.len()).unwrap_or(u64::MAX);

    let key_df: BTreeMap<String, u32> = count_df(reviewed.iter().map(|d| d.field_keys.clone()))
        .into_iter()
        .filter(|&(_, df)| df_admits(df, n, &vocab))
        .collect();
    let field_keys = top_by_df(key_df, remaining);
    // Safe: `top_by_df` keeps at most `remaining` keys.
    let remaining_for_terms = remaining - count_u32(field_keys.len());

    let term_df: BTreeMap<String, u32> = count_df(reviewed.iter().map(|d| tokenize(&d.text)))
        .into_iter()
        .filter(|(tok, df)| {
            tok.chars().count() >= vocab.min_token_len
                && !(vocab.reject_digits && tok.chars().any(|c| c.is_ascii_digit()))
                && df_admits(*df, n, &vocab)
        })
        .collect();
    let terms = top_by_df(term_df, remaining_for_terms);
    if terms.is_empty() {
        return Err(RowBuildError::NoContentCues);
    }

    let domains = ArchiveDomains {
        correspondents: correspondents.domain(),
        document_types: document_types.domain(),
        tags: tags.domain(),
        field_keys: count_u32(field_keys.len()),
        terms: count_u32(terms.len()),
    };
    let mapping = Mapping {
        correspondents,
        document_types,
        tags,
        term_index: index_of(&terms),
        terms,
        key_index: index_of(&field_keys),
        field_keys,
    };

    let mut by_hash: HashMap<&str, Vec<&Assignment>> = HashMap::new();
    for a in assignments {
        by_hash
            .entry(a.content_sha256_hex.as_str())
            .or_default()
            .push(a);
    }
    let rows = reviewed
        .iter()
        .map(|doc| {
            let own = by_hash
                .get(doc.content_sha256_hex.as_str())
                .map_or(&[][..], Vec::as_slice);
            mapping.row_from(own.iter().copied(), doc, tokenize)
        })
        .collect();

    Ok(TrainingSet {
        domains,
        rows,
        mapping,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_match::{AutoMatchParams, AutoMatcher, MineScope};

    /// Whitespace split, lowercased.
    fn tok(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_lowercase).collect()
    }

    fn def(kind: Kind, id: u32, is_auto: bool, retired: bool) -> Definition {
        Definition {
            kind,
            definition_id: id,
            name: format!("{kind:?}-{id}"),
            is_auto,
            retired,
        }
    }

    fn doc(hash: &str, text: &str, keys: &[&str]) -> DocContent {
        DocContent {
            content_sha256_hex: hash.to_owned(),
            text: text.to_owned(),
            field_keys: keys.iter().map(|k| (*k).to_owned()).collect(),
        }
    }

    fn asg(hash: &str, kind: Kind, id: u32) -> Assignment {
        Assignment {
            content_sha256_hex: hash.to_owned(),
            kind,
            definition_id: id,
        }
    }

    /// Every filter off: any token of any length, any DF, any share.
    fn loose() -> VocabParams {
        VocabParams {
            min_token_len: 1,
            reject_digits: false,
            min_df: 1,
            max_df_ppm: 1_000_000,
        }
    }

    fn build(
        defs: &[Definition],
        assignments: &[Assignment],
        reviewed: &[DocContent],
        vocab: VocabParams,
    ) -> Result<TrainingSet, RowBuildError> {
        build_training(defs, assignments, reviewed, &tok, vocab)
    }

    /// `n` live non-auto tag definitions with ids `0..n`.
    fn tag_defs(n: u32) -> Vec<Definition> {
        (0..n).map(|i| def(Kind::Tag, i, false, false)).collect()
    }

    /// Correspondents 5 (auto, retired), 6 (non-auto), 7 (auto); tags 10
    /// (auto), 11 (non-auto), 12 (auto, retired); three reviewed documents and
    /// one unreviewed document `d4` that has assignments.
    fn base() -> (Vec<Definition>, Vec<Assignment>, Vec<DocContent>) {
        let defs = vec![
            def(Kind::Correspondent, 5, true, true),
            def(Kind::Correspondent, 6, false, false),
            def(Kind::Correspondent, 7, true, false),
            def(Kind::Tag, 10, true, false),
            def(Kind::Tag, 11, false, false),
            def(Kind::Tag, 12, true, true),
        ];
        let assignments = vec![
            asg("d1", Kind::Correspondent, 5),
            asg("d1", Kind::Tag, 10),
            asg("d1", Kind::Tag, 12),
            asg("d2", Kind::Correspondent, 7),
            asg("d2", Kind::Tag, 11),
            asg("d3", Kind::Correspondent, 6),
            asg("d4", Kind::Correspondent, 7),
            asg("d4", Kind::Tag, 10),
        ];
        let reviewed = vec![
            doc("d1", "alpha beta", &[]),
            doc("d2", "alpha gamma", &[]),
            doc("d3", "beta gamma", &[]),
        ];
        (defs, assignments, reviewed)
    }

    fn base_set() -> TrainingSet {
        let (defs, assignments, reviewed) = base();
        build(&defs, &assignments, &reviewed, loose()).expect("base fixture builds")
    }

    /// Eligible targets and allowed cues, over every dense id the fixture has
    /// plus out-of-range ones.
    ///
    /// Disable: make `cue_allowed` admit AUTO definitions, or
    /// `target_eligible` admit non-auto ones or the retired category, and the
    /// matching assertion goes red.
    #[test]
    fn eligibility_and_cue_matrix() {
        let m = base_set().mapping;
        // Correspondents: dense 0 = def 6 (non-auto), 1 = def 7 (auto),
        // 2 = the retired category (def 5 lives there).
        assert!(!m.target_eligible(Target::Correspondent(0)));
        assert!(m.cue_allowed(Cue::Correspondent(0)));
        assert!(m.target_eligible(Target::Correspondent(1)));
        assert!(!m.cue_allowed(Cue::Correspondent(1)));
        assert!(!m.target_eligible(Target::Correspondent(2)));
        assert!(!m.cue_allowed(Cue::Correspondent(2)));
        assert!(!m.target_eligible(Target::Correspondent(3)));
        assert!(!m.cue_allowed(Cue::Correspondent(3)));
        // Tags: dense 0 = def 10 (auto), 1 = def 11 (non-auto). Def 12 is
        // retired and has no dense id.
        assert!(m.target_eligible(Target::Tag(0)));
        assert!(!m.cue_allowed(Cue::Tag(0)));
        assert!(!m.target_eligible(Target::Tag(1)));
        assert!(m.cue_allowed(Cue::Tag(1)));
        assert!(!m.target_eligible(Target::Tag(2)));
        assert!(!m.cue_allowed(Cue::Tag(2)));
        // No document type is defined: dense 0 is the retired category.
        assert!(!m.target_eligible(Target::DocumentType(0)));
        assert!(!m.cue_allowed(Cue::DocumentType(0)));
        // Content cues are always allowed.
        assert!(m.cue_allowed(Cue::Term(0)));
        assert!(m.cue_allowed(Cue::FieldKey(0)));
    }

    /// G8. A retired correspondent's assignment lands in the retired category
    /// (dense id `n`), which stands for no definition.
    ///
    /// Disable: map a retired assignment to `None` and the row assertion goes
    /// red; return a definition for the retired category and the
    /// `definition_of` assertion goes red.
    #[test]
    fn a_retired_correspondent_maps_to_the_retired_category() {
        let ts = base_set();
        // Two live correspondents, so the retired category is dense id 2.
        assert_eq!(ts.domains.correspondents, 3);
        assert_eq!(ts.rows[0].correspondent, Some(2));
        assert_eq!(ts.mapping.definition_of(Target::Correspondent(2)), None);
        // The live ones still resolve, so the `None` above is not a blanket.
        assert_eq!(
            ts.mapping.definition_of(Target::Correspondent(1)),
            Some((Kind::Correspondent, 7))
        );
        assert_eq!(ts.rows[1].correspondent, Some(1));
        assert_eq!(ts.rows[2].correspondent, Some(0));
    }

    /// A retired tag's assignment is absent from the row, and the retired tag
    /// has no dense id.
    ///
    /// Disable: give retired tags a dense id, or keep their assignments, and
    /// `tags` on the first row (or `domains.tags`) changes.
    #[test]
    fn a_retired_tag_is_dropped_from_the_row() {
        let ts = base_set();
        // d1 carries tag 10 (dense 0) and the retired tag 12.
        assert_eq!(ts.rows[0].tags, vec![0]);
        // d2 carries tag 11 (dense 1).
        assert_eq!(ts.rows[1].tags, vec![1]);
        assert_eq!(ts.domains.tags, 2);
    }

    /// G9 support. Only `reviewed` documents become rows, though assignments
    /// exist for a fourth hash.
    ///
    /// Disable: build a row per assignment hash and `rows.len()` becomes 4.
    #[test]
    fn only_reviewed_documents_become_rows() {
        let (defs, assignments, reviewed) = base();
        assert!(assignments.iter().any(|a| a.content_sha256_hex == "d4"));
        assert_eq!(reviewed.len(), 3);
        let ts = build(&defs, &assignments, &reviewed, loose()).unwrap();
        assert_eq!(ts.rows.len(), 3);
    }

    /// A single-valued kind with several assignments takes the smallest
    /// `definition_id`; an unknown definition is ignored.
    ///
    /// Disable: take the last or the largest and the first assertion fails;
    /// map an unknown id to a slot and the second fails.
    #[test]
    fn single_valued_tie_break_and_unknown_definitions() {
        let (defs, mut assignments, reviewed) = base();
        assignments.push(asg("d1", Kind::Correspondent, 7));
        assignments.push(asg("d1", Kind::Correspondent, 6));
        assignments.push(asg("d1", Kind::Tag, 99));
        assignments.push(asg("d3", Kind::Correspondent, 99));
        let ts = build(&defs, &assignments, &reviewed, loose()).unwrap();
        // d1 now has correspondents 5 (retired), 6, 7: the smallest id is 5.
        assert_eq!(ts.rows[0].correspondent, Some(2));
        // d1's unknown tag 99 adds nothing.
        assert_eq!(ts.rows[0].tags, vec![0]);
        // d3 has 6 and the unknown 99: 99 is ignored, 6 stays.
        assert_eq!(ts.rows[2].correspondent, Some(0));
    }

    /// `row_for` reads only the given document's own assignments, and ignores
    /// tokens outside the vocabulary.
    ///
    /// Disable: drop the hash filter and the row picks up other documents'
    /// assignments; index unknown tokens and `terms` gains an entry.
    #[test]
    fn row_for_reads_only_its_own_assignments_and_known_tokens() {
        let ts = base_set();
        let (_, assignments, _) = base();
        let novel = doc("new", "alpha zzzz", &["unknown-key"]);
        let row = ts.mapping.row_for(&assignments, &novel, &tok);
        assert_eq!(row.correspondent, None);
        assert_eq!(row.tags.len(), 0);
        assert_eq!(row.field_keys.len(), 0);
        assert_eq!(row.terms.len(), 1);
        assert_eq!(ts.mapping.terms()[row.terms[0] as usize], "alpha");
        // d2's own row from the full slice equals the training row.
        let d2 = doc("d2", "alpha gamma", &[]);
        assert_eq!(ts.mapping.row_for(&assignments, &d2, &tok), ts.rows[1]);
    }

    /// The training set feeds the miner: its ids are in range, and the
    /// mapping's eligibility and cue functions are accepted as a scope.
    ///
    /// Disable: emit an id past a domain and `mine_eligible` returns
    /// `IdOutOfRange`.
    #[test]
    fn a_training_set_is_accepted_by_the_miner() {
        let ts = base_set();
        let target = |t: Target| ts.mapping.target_eligible(t);
        let cue = |c: Cue| ts.mapping.cue_allowed(c);
        let mined = AutoMatcher::mine_eligible(
            &ts.domains,
            &ts.rows,
            AutoMatchParams::default(),
            MineScope {
                target: &target,
                cue: &cue,
            },
        );
        assert!(mined.is_ok(), "{mined:?}");
    }

    /// The reviewed fixture of ten documents for the DF tests, with the
    /// expected DF of each token written out by hand:
    /// rent 6, invoice 5, bill 5, paid 4, zeta 1.
    fn df_docs() -> Vec<DocContent> {
        [
            "rent invoice invoice paid",
            "rent invoice zeta",
            "rent bill",
            "bill paid",
            "invoice bill",
            "rent",
            "paid paid rent",
            "bill",
            "invoice rent",
            "bill invoice paid",
        ]
        .iter()
        .enumerate()
        .map(|(i, t)| doc(&format!("h{i}"), t, &[]))
        .collect()
    }

    /// Brute-force DF, written separately from the builder: how many
    /// documents contain `token` at least once.
    fn brute_df(docs: &[DocContent], token: &str) -> u32 {
        let mut count = 0;
        for d in docs {
            if d.text.split_whitespace().any(|w| w.to_lowercase() == token) {
                count += 1;
            }
        }
        count
    }

    /// G3. The kept vocabulary is exactly the tokens whose brute-force DF
    /// clears `min_df` and the share cap, and each kept term's DF as read back
    /// from the rows equals the brute-force count.
    ///
    /// Disable: count occurrences instead of documents (rent would be 7 and
    /// `paid` 5), or count over a different tokenizer, and a DF comparison
    /// goes red.
    #[test]
    fn document_frequency_equals_a_brute_force_count() {
        let docs = df_docs();
        let n = u64::try_from(docs.len()).unwrap();
        let all = ["rent", "invoice", "bill", "paid", "zeta"];
        // Anti-vacuity: the fixture really separates the tokens.
        assert_eq!(brute_df(&docs, "rent"), 6);
        assert_eq!(brute_df(&docs, "zeta"), 1);

        for (min_df, ppm) in [(3u32, 1_000_000u32), (3, 550_000), (5, 1_000_000)] {
            let vocab = VocabParams {
                min_df,
                max_df_ppm: ppm,
                ..loose()
            };
            let ts = build(&[], &[], &docs, vocab).unwrap();
            let expected: Vec<&str> = {
                let mut v: Vec<&str> = all
                    .iter()
                    .copied()
                    .filter(|t| {
                        let df = brute_df(&docs, t);
                        df >= min_df && u64::from(df) * 1_000_000 <= u64::from(ppm) * n
                    })
                    .collect();
                v.sort_unstable();
                v
            };
            assert_ne!(expected.len(), 0);
            let kept: Vec<&str> = ts.mapping.terms().iter().map(String::as_str).collect();
            assert_eq!(kept, expected, "min_df {min_df}, ppm {ppm}");
            for (id, term) in ts.mapping.terms().iter().enumerate() {
                let from_rows = ts
                    .rows
                    .iter()
                    .filter(|r| r.terms.contains(&u32::try_from(id).unwrap()))
                    .count();
                assert_eq!(
                    u32::try_from(from_rows).unwrap(),
                    brute_df(&docs, term),
                    "{term}"
                );
            }
        }
    }

    fn terms_of(docs: &[DocContent], vocab: VocabParams) -> Vec<String> {
        build(&[], &[], docs, vocab)
            .expect("fixture builds")
            .mapping
            .terms()
            .to_vec()
    }

    /// G3b, `min_token_len`. A 2-char token is dropped at 3 and admitted at 2.
    ///
    /// Disable: ignore `min_token_len` and "ab" appears in the first list.
    #[test]
    fn min_token_len_is_not_decoration() {
        let docs: Vec<DocContent> = (0..4)
            .map(|i| doc(&format!("l{i}"), "alpha ab", &[]))
            .collect();
        let strict = VocabParams {
            min_token_len: 3,
            ..loose()
        };
        let relaxed = VocabParams {
            min_token_len: 2,
            ..loose()
        };
        assert_eq!(terms_of(&docs, strict), vec!["alpha"]);
        assert_eq!(terms_of(&docs, relaxed), vec!["ab", "alpha"]);
    }

    /// G3b, `reject_digits`. "a1b2" is dropped when digits are rejected and
    /// admitted when they are not.
    ///
    /// Disable: ignore `reject_digits` and "a1b2" appears in the first list.
    #[test]
    fn reject_digits_is_not_decoration() {
        let docs: Vec<DocContent> = (0..4)
            .map(|i| doc(&format!("g{i}"), "alpha a1b2", &[]))
            .collect();
        let strict = VocabParams {
            reject_digits: true,
            ..loose()
        };
        let relaxed = VocabParams {
            reject_digits: false,
            ..loose()
        };
        assert_eq!(terms_of(&docs, strict), vec!["alpha"]);
        assert_eq!(terms_of(&docs, relaxed), vec!["a1b2", "alpha"]);
    }

    /// G3b, `min_df`. A token in exactly 2 of 4 documents is dropped at 3 and
    /// admitted at 2.
    ///
    /// Disable: ignore `min_df` and "rare" appears in the first list.
    #[test]
    fn min_df_is_not_decoration() {
        let docs = vec![
            doc("m0", "alpha rare", &[]),
            doc("m1", "alpha rare", &[]),
            doc("m2", "alpha", &[]),
            doc("m3", "alpha", &[]),
        ];
        let strict = VocabParams {
            min_df: 3,
            ..loose()
        };
        let relaxed = VocabParams {
            min_df: 2,
            ..loose()
        };
        assert_eq!(terms_of(&docs, strict), vec!["alpha"]);
        assert_eq!(terms_of(&docs, relaxed), vec!["alpha", "rare"]);
    }

    /// G3b, `max_df_ppm`. A token in every document is dropped at `500_000` and
    /// admitted at `1_000_000`. "half" is in exactly half and survives both, so
    /// the strict run is not empty.
    ///
    /// Disable: ignore `max_df_ppm` and "common" appears in the first list.
    #[test]
    fn max_df_ppm_is_not_decoration() {
        let docs = vec![
            doc("x0", "common half", &[]),
            doc("x1", "common half", &[]),
            doc("x2", "common", &[]),
            doc("x3", "common", &[]),
        ];
        let strict = VocabParams {
            max_df_ppm: 500_000,
            ..loose()
        };
        let relaxed = VocabParams {
            max_df_ppm: 1_000_000,
            ..loose()
        };
        assert_eq!(terms_of(&docs, strict), vec!["half"]);
        assert_eq!(terms_of(&docs, relaxed), vec!["common", "half"]);
    }

    /// Field keys pass the DF filters but not the length or digit filters
    /// (R6). "k1" has a digit and 2 chars and is kept; a key in one document
    /// is dropped by `min_df`; a key in every document is dropped by the share
    /// cap.
    ///
    /// Disable: apply the token filters to keys and "k1" disappears; skip the
    /// DF filters on keys and "rare" (or "k1" in the capped run) appears.
    #[test]
    fn field_keys_use_the_df_filters_only() {
        let docs = vec![
            doc("f0", "alpha half", &["k1", "rare"]),
            doc("f1", "alpha half", &["k1"]),
            doc("f2", "beta", &["k1"]),
            doc("f3", "beta", &["k1"]),
        ];
        let vocab = VocabParams {
            min_token_len: 3,
            reject_digits: true,
            min_df: 2,
            max_df_ppm: 1_000_000,
        };
        let ts = build(&[], &[], &docs, vocab).unwrap();
        assert_eq!(ts.mapping.field_keys(), ["k1"]);
        // Row side: k1 maps to dense 0 on every document, rare to nothing.
        assert!(ts.rows.iter().all(|r| r.field_keys == vec![0]));

        let capped = VocabParams {
            max_df_ppm: 500_000,
            ..vocab
        };
        let ts = build(&[], &[], &docs, capped).unwrap();
        assert_eq!(ts.mapping.field_keys().len(), 0);
    }

    /// G7a. 256 tags leave no room, so no content term fits.
    ///
    /// Disable: build the matcher anyway (skip the empty-terms check) and the
    /// first assertion goes red. The 255-tag twin proves the terms are there
    /// to be lost: one slot of room admits exactly one term.
    #[test]
    fn a_full_tag_budget_leaves_no_content_cues() {
        let docs = vec![
            doc("b0", "alpha beta", &[]),
            doc("b1", "alpha gamma", &[]),
            doc("b2", "beta gamma", &[]),
        ];
        let full = build(&tag_defs(256), &[], &docs, loose());
        assert_eq!(full.unwrap_err(), RowBuildError::NoContentCues);

        let one_room = build(&tag_defs(255), &[], &docs, loose()).unwrap();
        // alpha, beta, gamma all have DF 2: lexicographic tie-break keeps alpha.
        assert_eq!(one_room.mapping.terms(), ["alpha"]);
    }

    /// G7a, second case. 255 tags plus one surviving key use the whole budget
    /// and again leave no room for terms.
    ///
    /// Disable: let terms exceed the remaining budget and the first assertion
    /// goes red; the 254-tag twin shows the key does not by itself cause it.
    #[test]
    fn a_key_can_eat_the_last_slot() {
        let docs = vec![
            doc("c0", "alpha", &["k"]),
            doc("c1", "alpha", &["k"]),
            doc("c2", "alpha", &["k"]),
        ];
        let tight = build(&tag_defs(255), &[], &docs, loose());
        assert_eq!(tight.unwrap_err(), RowBuildError::NoContentCues);

        let roomy = build(&tag_defs(254), &[], &docs, loose()).unwrap();
        assert_eq!(roomy.mapping.field_keys(), ["k"]);
        assert_eq!(roomy.mapping.terms(), ["alpha"]);
    }

    /// G7b. 257 tags exceed the cap. A retired tag does not count toward it.
    ///
    /// Disable: count retired tags, or drop the cap check, and one of the two
    /// assertions goes red.
    #[test]
    fn too_many_tags_is_refused_and_retired_tags_do_not_count() {
        let docs = vec![doc("t0", "alpha", &[]), doc("t1", "alpha", &[])];
        let err = build(&tag_defs(257), &[], &docs, loose()).unwrap_err();
        assert_eq!(
            err,
            RowBuildError::TooManyFeatures {
                features: 257,
                max: MAX_BINARY_FEATURES,
            }
        );

        // 258 definitions, 3 of them retired: 255 live tags fit.
        let mut defs = tag_defs(258);
        for d in defs.iter_mut().take(3) {
            d.retired = true;
        }
        let ts = build(&defs, &[], &docs, loose()).unwrap();
        assert_eq!(ts.domains.tags, 255);
    }

    /// Budget tie-break. With room for two terms, higher DF wins, and equal DF
    /// is trimmed lexicographically. "aaa" has the lowest DF and sorts first,
    /// so it can only be absent because DF outranks the alphabet.
    ///
    /// Disable: rank by the alphabet alone and "aaa" appears; rank ties in
    /// descending order and "ddd" replaces "bbb".
    #[test]
    fn budget_trimming_is_by_df_then_lexicographic() {
        let docs = vec![
            doc("z0", "aaa bbb ccc ddd", &[]),
            doc("z1", "bbb ccc ddd", &[]),
        ];
        // 254 tags leave exactly two slots.
        let ts = build(&tag_defs(254), &[], &docs, loose()).unwrap();
        assert_eq!(ts.mapping.terms(), ["bbb", "ccc"]);
    }

    /// Dense ids are deterministic: two builds agree, and the input order of
    /// the reviewed documents does not change the vocabulary.
    ///
    /// Disable: number terms in first-seen order and the reversed build
    /// disagrees; iterate a `HashMap` for the kept list and the two builds
    /// can disagree.
    #[test]
    fn dense_ids_are_stable() {
        let docs = df_docs();
        let vocab = VocabParams {
            min_df: 3,
            ..loose()
        };
        let a = build(&[], &[], &docs, vocab).unwrap();
        let b = build(&[], &[], &docs, vocab).unwrap();
        assert_eq!(a.mapping.terms(), b.mapping.terms());
        assert_eq!(a.rows, b.rows);

        let mut reversed = docs.clone();
        reversed.reverse();
        let c = build(&[], &[], &reversed, vocab).unwrap();
        assert_eq!(a.mapping.terms(), c.mapping.terms());
        // Anti-vacuity: there is an order to preserve, and it is sorted.
        assert!(a.mapping.terms().len() > 1);
        assert!(a.mapping.terms().windows(2).all(|w| w[0] < w[1]));
    }
}
