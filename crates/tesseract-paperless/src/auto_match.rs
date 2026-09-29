//! The `AUTO` matching tier: association rules mined over the archive.
//!
//! ```text
//!   A RULE IS A SUGGESTION WITH A TRUTH VALUE.  NOTHING IN THIS FILE ASSIGNS.
//! ```
//!
//! paperless-ngx's `matching_algorithm = AUTO` is a learned classifier; S-8's
//! `matching.rs` left it never-matching. This module is that tier, built as
//! integer association rules over one row per archived document, mined by
//! `lance-graph-arm-discovery` (an Aerial+ transcoding: a codebook-distance
//! probe proposes a consequent, integer counts confirm it on the data).
//! "Documents from correspondent 3 carry tag 7 in 19 of 20 cases" becomes a
//! [`Suggestion`] `Tag(7)` with a NARS truth, the antecedent facts that fired
//! it ([`Cue`]), and its evidence count.
//!
//! # Suggestions, never assignments
//!
//! [`AutoMatcher::suggest`] takes `&self` and a borrowed row and returns
//! values. There is no path from here to a write: applying a suggestion is the
//! caller's decision, exactly as a human accepting one in the paperless-ngx
//! UI. A rule that has a truth value can be revised or rejected; a rule that
//! silently assigned could only be undone.
//!
//! # Why the oracle is built from the data
//!
//! `extract_rules` proposes, for each feature outside the antecedent, ONLY the
//! single category with the smallest codebook distance from the antecedent
//! (first minimum wins), then confirms it on the data. A uniform oracle
//! therefore always proposes category 0 -- and for a tag encoded
//! absent = 0 / present = 1 that is "absent", so it could never propose a tag.
//! The oracle here is [`CooccurrenceDistance`]: `PPM - P(b | a) * PPM`, built
//! from the same rows, so the proposed category is the empirically most
//! likely one and the probe is not blind.
//!
//! # Antecedents are what is observable at ingest
//!
//! For a document that has just arrived, its correspondent, type and tags are
//! exactly the unknowns, so rules keyed only on them would rarely fire. The
//! antecedents that matter are what ingest can see: content **terms**
//! (caller-supplied term ids -- e.g. the archive's top-N document-frequency
//! words; this module does no tokenizing) and custom-field keys, plus any
//! metadata already assigned by hand or by an S-8 rule. Suggestions are never
//! chained: a [`Suggestion`] is not fed back as an antecedent for another.
//!
//! # Lift, not just confidence
//!
//! A common consequent clears a confidence floor for any antecedent at all
//! (tag on 80% of documents: every correspondent "implies" it at 0.8). So a
//! rule must also have lift `P(target | cue) / P(target)` of at least
//! [`AutoMatchParams::min_lift_ppm`]. A consequent with base rate above
//! `PPM / min_lift_ppm` can therefore never pass: with the default 1.5 a tag
//! carried by more than two thirds of the archive is never suggested, which is
//! correct -- it is uninformative.
//!
//! # Encoding
//!
//! One dataset row per [`ArchiveRow`]. Feature 0 is the correspondent
//! (cardinality `correspondents + 1`; 0 = none, `c + 1` = correspondent `c`),
//! feature 1 the document type (same scheme), then one binary feature per tag
//! (0 absent, 1 present), then one binary feature per custom-field key, then one per content term.
//! Only POSITIVE facts are ever cues, and only positive correspondents,
//! document types and tags are ever targets (never a field key or a term): "tag 3 is absent" may be the
//! most probable category, but it is not something to suggest.
//!
//! # Policy pins, not measurements
//!
//! [`AutoMatchParams::default`] (5 rows of evidence, 0.7 confidence, lift 1.5, `k = 5`)
//! are policy pins chosen to be conservative, **not** values measured on a real
//! archive. Single-fact rules only (`max_antecedent = 1`): pairs of antecedent
//! facts grow combinatorially in the number of tags and fields, and a single
//! fact is also the only rule whose `because` a person can read at a glance.

use std::collections::{BTreeMap, HashMap};

pub use lance_graph_arm_discovery::NarsTruth;
use lance_graph_arm_discovery::{
    arm_to_nars, extract_rules, CandidateRule, CodebookDistance, Dataset, ExtractParams,
    FeatureSpec, Item, PPM,
};

/// Sizes of the archive's metadata domains. Ids are 0-based and dense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchiveDomains {
    /// Number of correspondents; valid ids are `0..correspondents`.
    pub correspondents: u32,
    /// Number of document types; valid ids are `0..document_types`.
    pub document_types: u32,
    /// Number of tags; valid ids are `0..tags`.
    pub tags: u32,
    /// Number of custom-field keys; valid ids are `0..field_keys`.
    pub field_keys: u32,
    /// Number of content terms; valid ids are `0..terms`.
    pub terms: u32,
}

/// One archived document's metadata. Ids index into [`ArchiveDomains`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchiveRow {
    /// The document's correspondent, if any.
    pub correspondent: Option<u32>,
    /// The document's document type, if any.
    pub document_type: Option<u32>,
    /// The tags on the document. Duplicates are harmless (presence).
    pub tags: Vec<u32>,
    /// The custom-field keys set on the document. Duplicates are harmless.
    pub field_keys: Vec<u32>,
    /// Content-term ids present in the document's text (caller-supplied).
    /// Duplicates are harmless.
    pub terms: Vec<u32>,
}

/// What a suggestion assigns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Target {
    /// Assign this correspondent.
    Correspondent(u32),
    /// Assign this document type.
    DocumentType(u32),
    /// Add this tag.
    Tag(u32),
}

/// An antecedent fact a rule fired on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cue {
    /// The document has this correspondent.
    Correspondent(u32),
    /// The document has this document type.
    DocumentType(u32),
    /// The document carries this tag.
    Tag(u32),
    /// The document has this custom field set.
    FieldKey(u32),
    /// The document's text contains this content term.
    Term(u32),
}

/// One proposed assignment, with the evidence for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// What would be assigned.
    pub target: Target,
    /// NARS truth of the rule that produced it (the best rule for this target).
    pub truth: NarsTruth,
    /// Documents in the archive where the rule's cue and target co-occur.
    pub evidence: u32,
    /// Every cue on the document that has a rule for this target, sorted.
    pub because: Vec<Cue>,
}

/// The mining thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoMatchParams {
    /// Minimum number of archived documents on which cue and target co-occur.
    pub min_evidence: u32,
    /// Minimum `P(target | cue)`, in parts per million.
    pub min_confidence_ppm: u32,
    /// Minimum lift `P(target | cue) / P(target)`, in parts per million.
    pub min_lift_ppm: u32,
    /// NARS personality constant; larger demands more evidence for the same
    /// confidence. Must be positive.
    pub k: u32,
}

impl Default for AutoMatchParams {
    /// Policy pins (5 rows, 0.7, lift 1.5, `k = 5`), not values measured on an
    /// archive.
    fn default() -> Self {
        Self {
            min_evidence: 5,
            min_confidence_ppm: 700_000,
            min_lift_ppm: 1_500_000,
            k: 5,
        }
    }
}

/// Why mining refused an archive or its parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoMatchError {
    /// A row names an id outside its domain.
    IdOutOfRange {
        /// Which domain: `"correspondent"`, `"document_type"`, `"tag"`,
        /// `"field_key"` or `"term"`.
        what: &'static str,
        /// The offending id.
        id: u32,
        /// The domain size it had to be below.
        domain: u32,
    },
    /// `k` was zero, which would make any single co-occurrence dogmatic.
    ZeroK,
    /// `tags + field_keys + terms` exceeds [`MAX_BINARY_FEATURES`].
    TooManyFeatures {
        /// The binary feature count the domains asked for.
        features: u64,
        /// The cap.
        max: u32,
    },
}

/// The most binary features (tags + field keys + terms) one mining run takes.
///
/// A policy pin, not a measurement. The miner holds the archive as a dense
/// table (4 bytes per feature per document) plus one bitset per item, and
/// probes every feature for every antecedent, so its cost grows with the
/// square of the width. At the cap and 100,000 documents that is about 100 MB
/// and a width² of about 66,000 probes. A larger vocabulary is refused with
/// [`AutoMatchError::TooManyFeatures`] rather than allowed to stall; mine a
/// smaller term set, or one tag family at a time.
pub const MAX_BINARY_FEATURES: u32 = 256;

impl core::fmt::Display for AutoMatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IdOutOfRange { what, id, domain } => {
                write!(f, "{what} id {id} is outside its domain of {domain}")
            }
            Self::ZeroK => write!(f, "NARS personality constant k must be positive"),
            Self::TooManyFeatures { features, max } => write!(
                f,
                "{features} binary features (tags + field keys + terms) exceed the cap of {max}"
            ),
        }
    }
}

impl std::error::Error for AutoMatchError {}

/// The `CodebookDistance` oracle built from the archive itself:
/// `distance(a, b) = PPM - PPM * count(a and b) / count(a)`.
///
/// Items of one feature are never each other's consequent, so they are
/// `u32::MAX` apart unless identical. An item that never occurs is
/// `u32::MAX` from everything.
///
/// **Counts are sparse.** A row states one category for each of features 0-1
/// (correspondent, document type; category 0 is "none") and, for every binary
/// feature (tags, field keys, terms), whether it is present. Only the *stated*
/// items are counted: the two multi-category values and the present binaries.
/// A row with `p` stated items costs `p²` pair updates, not `width²`. Counts
/// that involve an absent binary are derived from `n` and the stated counts
/// (inclusion-exclusion), so every distance is still exact.
struct CooccurrenceDistance {
    spec: FeatureSpec,
    /// Rows counted.
    n: u32,
    /// Per slot, how many rows state that item. Only meaningful for stated
    /// items; an absent binary's count is derived in [`Self::item_count`].
    singles: Vec<u32>,
    /// Per unordered pair of stated slots `(lo, hi)`, how many rows state both.
    pairs: HashMap<(u32, u32), u32>,
    /// Per slot, is this item an ineligible target? Such an item is never a
    /// consequent (every distance TO it is `u32::MAX`), but stays an
    /// antecedent: distances FROM it are unchanged.
    blocked: Vec<bool>,
}

/// Binary features (tags, field keys, terms) start after correspondent and
/// document type.
const FIRST_BINARY_FEATURE: u32 = 2;

/// Is `item` the "absent" category of a binary feature?
fn is_absent_binary(item: Item) -> bool {
    item.feature >= FIRST_BINARY_FEATURE && item.category == 0
}

impl CooccurrenceDistance {
    /// Count stated items and stated-item pairs over `data`.
    fn build(data: &Dataset) -> Self {
        let spec = data.spec.clone();
        let mut singles = vec![0u32; spec.dim()];
        let mut pairs: HashMap<(u32, u32), u32> = HashMap::new();
        let mut stated: Vec<u32> = Vec::new();
        for row in &data.rows {
            stated.clear();
            for (f, &c) in row.iter().enumerate() {
                let f = u32::try_from(f)
                    .expect("feature index fits u32: the spec is built from u32 domains");
                let item = Item::new(f, c);
                if !is_absent_binary(item) {
                    let slot = u32::try_from(spec.slot(item))
                        .expect("slot fits u32: the spec is built from u32 domains");
                    stated.push(slot);
                }
            }
            for (i, &a) in stated.iter().enumerate() {
                singles[a as usize] += 1;
                for &b in &stated[i + 1..] {
                    *pairs.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
        }
        let blocked = vec![false; spec.dim()];
        Self {
            spec,
            n: u32::try_from(data.len()).expect("row count fits u32"),
            singles,
            pairs,
            blocked,
        }
    }

    /// Block every item whose target `eligible` rejects.
    ///
    /// Eligibility has to act here, inside the probe, and not on the mined
    /// rules or the suggestions: the probe proposes only the nearest category
    /// of each feature, so an ineligible correspondent that is nearer than an
    /// eligible one would pre-empt it, and no later filter could bring the
    /// eligible rule back.
    fn block_ineligible(&mut self, layout: Layout, eligible: &dyn Fn(Target) -> bool) {
        for f in 0..self.spec.num_features() {
            let f = u32::try_from(f).expect("feature index fits u32");
            for c in 0..self.spec.cardinality(f as usize) {
                let item = Item::new(f, c);
                if layout.target(item).is_some_and(|t| !eligible(t)) {
                    let slot = self.spec.slot(item);
                    self.blocked[slot] = true;
                }
            }
        }
    }

    /// The stated item this item is about: itself, or for an absent binary
    /// its present twin.
    fn stated_twin(item: Item) -> Item {
        if is_absent_binary(item) {
            Item::new(item.feature, 1)
        } else {
            item
        }
    }

    /// How many rows contain `item`.
    fn item_count(&self, item: Item) -> u32 {
        let stated = self.singles[self.spec.slot(Self::stated_twin(item))];
        if is_absent_binary(item) {
            self.n - stated
        } else {
            stated
        }
    }

    /// How many rows contain both `a` and `b` (different features).
    fn pair_count(&self, a: Item, b: Item) -> u32 {
        let (ta, tb) = (Self::stated_twin(a), Self::stated_twin(b));
        let (sa, sb) = (self.spec.slot(ta), self.spec.slot(tb));
        let key = (
            u32::try_from(sa.min(sb)).expect("slot fits u32"),
            u32::try_from(sa.max(sb)).expect("slot fits u32"),
        );
        let both = self.pairs.get(&key).copied().unwrap_or(0);
        match (is_absent_binary(a), is_absent_binary(b)) {
            (false, false) => both,
            (true, false) => self.singles[sb] - both,
            (false, true) => self.singles[sa] - both,
            // Add before subtracting (`n - a - b` alone can go below zero), in
            // u64 so `n + both` cannot overflow. The result is a row count.
            (true, true) => {
                let rows = u64::from(self.n) + u64::from(both)
                    - u64::from(self.singles[sa])
                    - u64::from(self.singles[sb]);
                u32::try_from(rows).expect("a pair count is at most n")
            }
        }
    }
}

impl CodebookDistance for CooccurrenceDistance {
    fn distance(&self, a: Item, b: Item) -> u32 {
        if self.blocked[self.spec.slot(b)] {
            return u32::MAX;
        }
        if a.feature == b.feature {
            return if a.category == b.category {
                0
            } else {
                u32::MAX
            };
        }
        let count_a = u64::from(self.item_count(a));
        if count_a == 0 {
            return u32::MAX;
        }
        let both = u64::from(self.pair_count(a, b));
        // At most PPM (1_000_000), so it always fits.
        u32::try_from(PPM - both * PPM / count_a).unwrap_or(u32::MAX)
    }
}

/// Where each feature group starts, for decoding items back into facts.
#[derive(Debug, Clone, Copy)]
struct Layout {
    tags: u32,
    field_keys: u32,
}

impl Layout {
    /// The feature index of tag `t`.
    fn tag_feature(t: u32) -> u32 {
        2 + t
    }

    /// The positive fact an item states, or `None` for "absent"/"none".
    fn cue(self, item: Item) -> Option<Cue> {
        match item.feature {
            0 => item.category.checked_sub(1).map(Cue::Correspondent),
            1 => item.category.checked_sub(1).map(Cue::DocumentType),
            f if f < 2 + self.tags => (item.category == 1).then(|| Cue::Tag(f - 2)),
            f if f < 2 + self.tags + self.field_keys => {
                (item.category == 1).then(|| Cue::FieldKey(f - 2 - self.tags))
            }
            f => (item.category == 1).then(|| Cue::Term(f - 2 - self.tags - self.field_keys)),
        }
    }

    /// The positive assignable target an item states, or `None`.
    fn target(self, item: Item) -> Option<Target> {
        match self.cue(item)? {
            Cue::Correspondent(c) => Some(Target::Correspondent(c)),
            Cue::DocumentType(t) => Some(Target::DocumentType(t)),
            Cue::Tag(t) => Some(Target::Tag(t)),
            Cue::FieldKey(_) | Cue::Term(_) => None,
        }
    }
}

/// Check every id in `rows` against `domains`.
fn validate(domains: &ArchiveDomains, rows: &[ArchiveRow]) -> Result<(), AutoMatchError> {
    let check = |what, id: u32, domain| {
        if id < domain {
            Ok(())
        } else {
            Err(AutoMatchError::IdOutOfRange { what, id, domain })
        }
    };
    for r in rows {
        if let Some(c) = r.correspondent {
            check("correspondent", c, domains.correspondents)?;
        }
        if let Some(t) = r.document_type {
            check("document_type", t, domains.document_types)?;
        }
        for &t in &r.tags {
            check("tag", t, domains.tags)?;
        }
        for &k in &r.field_keys {
            check("field_key", k, domains.field_keys)?;
        }
        for &t in &r.terms {
            check("term", t, domains.terms)?;
        }
    }
    Ok(())
}

/// Encode validated `rows` as a dataset (layout in the module doc).
fn encode(domains: &ArchiveDomains, rows: &[ArchiveRow]) -> Dataset {
    let mut cards = vec![domains.correspondents + 1, domains.document_types + 1];
    let binary = u64::from(domains.tags) + u64::from(domains.field_keys) + u64::from(domains.terms);
    cards.extend(std::iter::repeat_n(
        2,
        usize::try_from(binary).expect("binary feature count fits usize"),
    ));
    let spec = FeatureSpec::new(cards);
    let width = spec.num_features();
    let data_rows = rows
        .iter()
        .map(|r| {
            let mut v = vec![0u32; width];
            v[0] = r.correspondent.map_or(0, |c| c + 1);
            v[1] = r.document_type.map_or(0, |t| t + 1);
            for &t in &r.tags {
                v[Layout::tag_feature(t) as usize] = 1;
            }
            for &k in &r.field_keys {
                v[(2 + domains.tags + k) as usize] = 1;
            }
            for &t in &r.terms {
                v[(2 + domains.tags + domains.field_keys + t) as usize] = 1;
            }
            v
        })
        .collect();
    Dataset::new(spec, data_rows)
}

/// Which targets and cues a mining run may use; see
/// [`AutoMatcher::mine_eligible`].
#[derive(Clone, Copy)]
pub struct MineScope<'a> {
    /// May this target be a rule's consequent?
    pub target: &'a dyn Fn(Target) -> bool,
    /// May this fact be a rule's antecedent?
    pub cue: &'a dyn Fn(Cue) -> bool,
}

impl MineScope<'static> {
    /// Every target and every cue: what [`AutoMatcher::mine`] uses.
    #[must_use]
    pub fn everything() -> Self {
        Self {
            target: &|_| true,
            cue: &|_| true,
        }
    }
}

impl core::fmt::Debug for MineScope<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("MineScope { .. }")
    }
}

/// One mined rule, decoded.
#[derive(Debug, Clone)]
struct MinedRule {
    cue: Cue,
    target: Target,
    rule: CandidateRule,
}

/// The rules mined from an archive, ready to suggest for new documents.
#[derive(Debug, Clone)]
pub struct AutoMatcher {
    rules: Vec<MinedRule>,
    k: u32,
}

impl AutoMatcher {
    /// Mine `rows` for single-fact association rules.
    ///
    /// Empty `rows` yield a matcher with no rules, not an error. Every id is
    /// validated against `domains` before anything is encoded, so a bad id is
    /// an [`AutoMatchError::IdOutOfRange`] rather than a panic inside the
    /// miner.
    ///
    /// # Errors
    ///
    /// - [`AutoMatchError::IdOutOfRange`] for an id outside its domain.
    /// - [`AutoMatchError::ZeroK`] when `params.k == 0`.
    /// - [`AutoMatchError::TooManyFeatures`] when `tags + field_keys + terms`
    ///   exceeds [`MAX_BINARY_FEATURES`].
    pub fn mine(
        domains: &ArchiveDomains,
        rows: &[ArchiveRow],
        params: AutoMatchParams,
    ) -> Result<Self, AutoMatchError> {
        Self::mine_eligible(domains, rows, params, MineScope::everything())
    }

    /// [`Self::mine`], restricted by `scope`.
    ///
    /// - `scope.target` decides which targets may be a rule's consequent. It
    ///   acts inside the oracle: filtering the suggestions afterwards would
    ///   lose every eligible target that an ineligible one of the same kind
    ///   out-ranked. Ineligible targets still count as assignments.
    /// - `scope.cue` decides which facts may be a rule's antecedent. Dropping
    ///   a rule by its antecedent never affects another antecedent's rules, so
    ///   this one is a plain filter on the mined rules.
    ///
    /// The archive passes "AUTO definitions only" for the target and "never an
    /// AUTO definition" for the cue, so a suggestion is never chained back as
    /// the evidence for another suggestion.
    ///
    /// # Errors
    ///
    /// As [`Self::mine`].
    pub fn mine_eligible(
        domains: &ArchiveDomains,
        rows: &[ArchiveRow],
        params: AutoMatchParams,
        scope: MineScope<'_>,
    ) -> Result<Self, AutoMatchError> {
        let eligible = scope.target;
        if params.k == 0 {
            return Err(AutoMatchError::ZeroK);
        }
        let features =
            u64::from(domains.tags) + u64::from(domains.field_keys) + u64::from(domains.terms);
        if features > u64::from(MAX_BINARY_FEATURES) {
            return Err(AutoMatchError::TooManyFeatures {
                features,
                max: MAX_BINARY_FEATURES,
            });
        }
        validate(domains, rows)?;
        if rows.is_empty() {
            return Ok(Self {
                rules: Vec::new(),
                k: params.k,
            });
        }
        let data = encode(domains, rows);
        let layout = Layout {
            tags: domains.tags,
            field_keys: domains.field_keys,
        };
        let mut oracle = CooccurrenceDistance::build(&data);
        oracle.block_ineligible(layout, eligible);
        let n = rows.len() as u64;
        // Floor, not ceil: `passes` compares the FLOORED support of a rule, so
        // a ceiling could reject a rule sitting exactly on `min_evidence`.
        // The explicit `cooccur >= min_evidence` below removes any admitted
        // excess.
        let min_support_ppm =
            u32::try_from(u64::from(params.min_evidence) * PPM / n).unwrap_or(u32::MAX);
        let extract = ExtractParams {
            theta: u32::MAX,
            // Single-fact rules: pairs grow combinatorially in tags and fields.
            max_antecedent: 1,
            min_support_ppm,
            min_confidence_ppm: params.min_confidence_ppm,
        };
        let window = rows.len() as u128;
        let min_lift = u128::from(params.min_lift_ppm);
        let rules = extract_rules(&oracle, &data, &extract)
            .into_iter()
            .filter(|r| r.cooccur >= params.min_evidence)
            // lift = (cooccur / antecedent_count) / (count(y) / window) >= min_lift
            // cross-multiplied in u128 (u64 overflows for a large min_lift_ppm).
            .filter(|r| {
                let Some(&y) = r.consequent.first() else {
                    return false;
                };
                let count_y = u128::from(oracle.item_count(y));
                u128::from(r.cooccur) * window * u128::from(PPM)
                    >= min_lift * u128::from(r.antecedent_count) * count_y
            })
            .filter_map(|r| {
                let cue = layout.cue(*r.antecedent.first()?)?;
                if !(scope.cue)(cue) {
                    return None;
                }
                let target = layout.target(*r.consequent.first()?)?;
                Some(MinedRule {
                    cue,
                    target,
                    rule: r,
                })
            })
            .collect();
        Ok(Self { rules, k: params.k })
    }

    /// How many rules survived mining.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Suggest assignments for `doc`, sorted by [`Target`].
    ///
    /// A target `doc` already carries is never suggested, and a document that
    /// already has a correspondent (or document type) gets no correspondent
    /// (or document type) suggestion at all. Rules for one target are merged:
    /// the highest-expectation truth wins (ties: more evidence), and
    /// `because` lists every cue that had a rule for it. Correspondent and
    /// document type are single-valued, so at most one suggestion of each kind
    /// is returned -- the best by expectation.
    #[must_use]
    pub fn suggest(&self, doc: &ArchiveRow) -> Vec<Suggestion> {
        let mut cues: Vec<Cue> = Vec::new();
        cues.extend(doc.correspondent.map(Cue::Correspondent));
        cues.extend(doc.document_type.map(Cue::DocumentType));
        cues.extend(doc.tags.iter().map(|&t| Cue::Tag(t)));
        cues.extend(doc.field_keys.iter().map(|&k| Cue::FieldKey(k)));
        cues.extend(doc.terms.iter().map(|&t| Cue::Term(t)));

        let assigned = |t: Target| match t {
            Target::Correspondent(_) => doc.correspondent.is_some(),
            Target::DocumentType(_) => doc.document_type.is_some(),
            Target::Tag(id) => doc.tags.contains(&id),
        };

        let mut merged: BTreeMap<Target, Suggestion> = BTreeMap::new();
        for m in &self.rules {
            if !cues.contains(&m.cue) || assigned(m.target) {
                continue;
            }
            let truth = arm_to_nars(&m.rule, self.k);
            let evidence = m.rule.cooccur;
            let entry = merged.entry(m.target).or_insert_with(|| Suggestion {
                target: m.target,
                truth,
                evidence,
                because: Vec::new(),
            });
            if better(truth, evidence, entry.truth, entry.evidence) {
                entry.truth = truth;
                entry.evidence = evidence;
            }
            entry.because.push(m.cue);
        }

        let mut out: Vec<Suggestion> = merged.into_values().collect();
        for s in &mut out {
            s.because.sort();
            s.because.dedup();
        }
        keep_best_of_kind(&mut out, |t| matches!(t, Target::Correspondent(_)));
        keep_best_of_kind(&mut out, |t| matches!(t, Target::DocumentType(_)));
        out
    }
}

/// Does `(truth, evidence)` beat `(cur_truth, cur_evidence)`?
fn better(truth: NarsTruth, evidence: u32, cur_truth: NarsTruth, cur_evidence: u32) -> bool {
    match truth.expectation().total_cmp(&cur_truth.expectation()) {
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Equal => evidence > cur_evidence,
        core::cmp::Ordering::Less => false,
    }
}

/// Among suggestions whose target satisfies `kind`, keep only the best one.
fn keep_best_of_kind(out: &mut Vec<Suggestion>, kind: impl Fn(Target) -> bool) {
    let mut best: Option<usize> = None;
    for (i, s) in out.iter().enumerate() {
        if !kind(s.target) {
            continue;
        }
        // `out` is sorted by target, so on a full tie the earlier target stays.
        if best.is_none_or(|b| better(s.truth, s.evidence, out[b].truth, out[b].evidence)) {
            best = Some(i);
        }
    }
    let mut idx = 0;
    out.retain(|s| {
        let keep = !kind(s.target) || Some(idx) == best;
        idx += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOMAINS: ArchiveDomains = ArchiveDomains {
        correspondents: 5,
        document_types: 3,
        tags: 5,
        field_keys: 2,
        terms: 10,
    };

    fn row(corr: Option<u32>, ty: Option<u32>, tags: &[u32]) -> ArchiveRow {
        ArchiveRow {
            correspondent: corr,
            document_type: ty,
            tags: tags.to_vec(),
            field_keys: Vec::new(),
            terms: Vec::new(),
        }
    }

    fn tag_targets(s: &[Suggestion]) -> Vec<u32> {
        s.iter()
            .filter_map(|x| match x.target {
                Target::Tag(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    /// `with_tag` docs of correspondent 0 carrying tag 2, `without_tag` docs of
    /// correspondent 0 without it, and `filler` docs of correspondent 1 or 2
    /// that never carry tag 2 (tag 0 on every third, deterministically).
    fn corr0_archive(with_tag: usize, without_tag: usize, filler: usize) -> Vec<ArchiveRow> {
        let mut rows = Vec::new();
        for _ in 0..with_tag {
            rows.push(row(Some(0), None, &[2]));
        }
        for _ in 0..without_tag {
            rows.push(row(Some(0), None, &[]));
        }
        for i in 0..filler {
            let tags: &[u32] = if i % 3 == 0 { &[0] } else { &[] };
            rows.push(row(Some(1 + u32::try_from(i % 2).unwrap()), None, tags));
        }
        rows
    }

    fn new_doc(corr: u32) -> ArchiveRow {
        row(Some(corr), None, &[])
    }

    #[test]
    fn a_correspondent_that_always_carries_a_tag_is_suggested() {
        let m = AutoMatcher::mine(
            &DOMAINS,
            &corr0_archive(20, 0, 40),
            AutoMatchParams::default(),
        )
        .unwrap();
        assert!(m.rule_count() > 0);
        let s = m.suggest(&new_doc(0));
        let hit = s
            .iter()
            .find(|x| x.target == Target::Tag(2))
            .expect("tag 2 must be suggested");
        assert!(hit.because.contains(&Cue::Correspondent(0)));
        assert!(hit.truth.frequency > 0.95, "{:?}", hit.truth);
        assert_eq!(hit.evidence, 20);
    }

    #[test]
    fn independent_tags_yield_no_suggestion() {
        let rows: Vec<ArchiveRow> = (0..200u32)
            .map(|i| row(Some(i % 5), None, if i % 2 == 0 { &[1] } else { &[] }))
            .collect();
        // Anti-vacuity: the archive really has tags and several correspondents.
        assert!(rows.iter().filter(|r| !r.tags.is_empty()).count() > 50);
        assert!(rows.iter().filter(|r| r.correspondent == Some(3)).count() > 10);
        let m = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap();
        for c in 0..5 {
            assert!(
                tag_targets(&m.suggest(&new_doc(c))).is_empty(),
                "correspondent {c} must not imply a tag"
            );
        }
    }

    #[test]
    fn an_independent_cue_does_not_fire_even_for_a_common_tag() {
        // Tag 2 on 80% of documents, independent of the correspondent
        // (i % 3 vs i % 5 are coprime; 3 stays inside the fixture's 5 correspondents).
        let rows: Vec<ArchiveRow> = (0..200u32)
            .map(|i| row(Some(i % 3), None, if i % 5 != 0 { &[2] } else { &[] }))
            .collect();
        // Anti-vacuity: the CONFIDENCE floor alone would admit every one of
        // these rules, so it is the lift floor that keeps them silent.
        for c in 0..3u32 {
            let of_c: Vec<&ArchiveRow> =
                rows.iter().filter(|r| r.correspondent == Some(c)).collect();
            let tagged = of_c.iter().filter(|r| r.tags.contains(&2)).count();
            assert!(
                tagged * 10 >= of_c.len() * 7,
                "corr {c}: {tagged}/{}",
                of_c.len()
            );
        }
        let m = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap();
        for c in 0..3 {
            assert!(!tag_targets(&m.suggest(&new_doc(c))).contains(&2));
        }
        // Can-fire twin: with the lift floor off, the same archive does suggest.
        let no_lift = AutoMatchParams {
            min_lift_ppm: 0,
            ..AutoMatchParams::default()
        };
        let m = AutoMatcher::mine(&DOMAINS, &rows, no_lift).unwrap();
        assert!(tag_targets(&m.suggest(&new_doc(0))).contains(&2));
    }

    #[test]
    fn the_lift_floor_is_not_decoration() {
        // 100 docs, tag 1 on 40 (P = 0.4). Correspondent 0 on 20, 16 tagged
        // (confidence 0.8, lift 2.0).
        let mut rows = Vec::new();
        for i in 0..20 {
            rows.push(row(Some(0), None, if i < 16 { &[1] } else { &[] }));
        }
        for i in 0..80u32 {
            rows.push(row(Some(1 + i % 2), None, if i < 24 { &[1] } else { &[] }));
        }
        let n = rows.len() as u64;
        let ant = rows.iter().filter(|r| r.correspondent == Some(0)).count() as u64;
        let both = rows
            .iter()
            .filter(|r| r.correspondent == Some(0) && r.tags.contains(&1))
            .count() as u64;
        let y = rows.iter().filter(|r| r.tags.contains(&1)).count() as u64;
        let lift_ppm = both * n * PPM / (ant * y);
        assert!(
            (1_500_000..=3_000_000).contains(&lift_ppm),
            "lift {lift_ppm}"
        );
        let with = |min_lift_ppm| {
            let p = AutoMatchParams {
                min_lift_ppm,
                ..AutoMatchParams::default()
            };
            AutoMatcher::mine(&DOMAINS, &rows, p)
                .unwrap()
                .suggest(&new_doc(0))
        };
        assert!(tag_targets(&with(1_500_000)).contains(&1));
        assert!(!tag_targets(&with(3_000_000)).contains(&1));
    }

    #[test]
    fn a_content_term_that_predicts_a_tag_fires_on_an_unassigned_document() {
        let mut rows = Vec::new();
        for _ in 0..25 {
            rows.push(ArchiveRow {
                terms: vec![7],
                tags: vec![2],
                ..ArchiveRow::default()
            });
        }
        for i in 0..60u32 {
            rows.push(ArchiveRow {
                correspondent: Some(i % 3),
                terms: if i % 4 == 0 { vec![3] } else { Vec::new() },
                tags: if i % 3 == 0 { vec![0] } else { Vec::new() },
                ..ArchiveRow::default()
            });
        }
        let m = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap();
        // A new document with ONLY a term: no correspondent, type, tags, fields.
        let doc = ArchiveRow {
            terms: vec![7],
            ..ArchiveRow::default()
        };
        let s = m.suggest(&doc);
        let hit = s
            .iter()
            .find(|x| x.target == Target::Tag(2))
            .expect("term 7 must predict tag 2");
        assert!(hit.because.contains(&Cue::Term(7)));
        assert!(hit.truth.frequency > 0.95);
        // Can-stay-silent: a document with an unrelated term gets no tag 2.
        let other = ArchiveRow {
            terms: vec![3],
            ..ArchiveRow::default()
        };
        assert!(!tag_targets(&m.suggest(&other)).contains(&2));
    }

    /// The explicit `cooccur >= min_evidence` filter, where it binds. The ppm
    /// support floor passed to `extract_rules` rounds down, and below about a
    /// million documents that rounding never admits a rule under
    /// `min_evidence`, so `the_evidence_floor_is_not_decoration` cannot see the
    /// filter. At 3,000,000 rows `floor(4·PPM/n) == floor(5·PPM/n) == 1`, so a
    /// 4-document rule clears the ppm floor for `min_evidence: 5` and only the
    /// explicit filter rejects it.
    #[test]
    fn the_explicit_evidence_filter_binds_where_the_ppm_floor_rounds() {
        const N: u64 = 3_000_000;
        let tiny = ArchiveDomains {
            correspondents: 1,
            document_types: 0,
            tags: 1,
            field_keys: 0,
            terms: 0,
        };
        let mut rows: Vec<ArchiveRow> = (0..4).map(|_| row(Some(0), None, &[0])).collect();
        rows.resize(usize::try_from(N).unwrap(), ArchiveRow::default());
        // Anti-vacuity, checked at compile time: the rounding really does make
        // the two floors equal, and the rule clears the ppm floor at all.
        const {
            assert!(4 * PPM / N == 5 * PPM / N);
            assert!(4 * PPM / N > 0);
        }
        let rules = |min_evidence| {
            let p = AutoMatchParams {
                min_evidence,
                ..AutoMatchParams::default()
            };
            AutoMatcher::mine(&tiny, &rows, p).unwrap().rule_count()
        };
        assert!(rules(4) > 0, "can fire: 4 documents meet min_evidence 4");
        assert_eq!(rules(5), 0, "4 documents must not meet min_evidence 5");
    }

    /// The sparse oracle must give exactly the counts a dense `width²` count
    /// would. The fixture mixes present and absent binaries, correspondent
    /// "none" and several types, so all four inclusion-exclusion branches of
    /// `pair_count` (absent on either side, both, neither) are exercised.
    #[test]
    fn sparse_counts_equal_dense_counts_for_every_item_pair() {
        let rows: Vec<ArchiveRow> = (0..97u32)
            .map(|i| ArchiveRow {
                correspondent: (i % 4 != 0).then_some(i % 5),
                document_type: (i % 3 != 0).then_some(i % 3),
                tags: (0..5).filter(|t| (i + t) % (t + 2) == 0).collect(),
                field_keys: (0..2).filter(|k| (i * (k + 1)) % 7 < 3).collect(),
                terms: (0..10).filter(|t| (i * 3 + t) % (t + 3) == 0).collect(),
            })
            .collect();
        let data = encode(&DOMAINS, &rows);
        let oracle = CooccurrenceDistance::build(&data);

        // The dense reference: count every pair of items in every row.
        let spec = &data.spec;
        let dim = spec.dim();
        let mut dense = vec![0u32; dim * dim];
        for row in &data.rows {
            let slots: Vec<usize> = row
                .iter()
                .enumerate()
                .map(|(f, &c)| spec.slot(Item::new(u32::try_from(f).unwrap(), c)))
                .collect();
            for &a in &slots {
                for &b in &slots {
                    dense[a * dim + b] += 1;
                }
            }
        }

        let items: Vec<Item> = (0..spec.num_features())
            .flat_map(|f| {
                let f32_ = u32::try_from(f).unwrap();
                (0..spec.cardinality(f)).map(move |c| Item::new(f32_, c))
            })
            .collect();
        let mut absent_pairs = 0;
        for &a in &items {
            let sa = spec.slot(a);
            assert_eq!(oracle.item_count(a), dense[sa * dim + sa], "count {a:?}");
            for &b in &items {
                if a.feature == b.feature {
                    continue;
                }
                let sb = spec.slot(b);
                assert_eq!(
                    oracle.pair_count(a, b),
                    dense[sa * dim + sb],
                    "pair {a:?} {b:?}"
                );
                if is_absent_binary(a) && is_absent_binary(b) && dense[sa * dim + sb] > 0 {
                    absent_pairs += 1;
                }
            }
        }
        // Anti-vacuity: the both-absent branch was reached with a non-zero count.
        assert!(absent_pairs > 0);
    }

    #[test]
    fn a_vocabulary_past_the_cap_is_refused_not_mined() {
        let at_cap = ArchiveDomains {
            correspondents: 1,
            document_types: 0,
            tags: 1,
            field_keys: 0,
            terms: MAX_BINARY_FEATURES - 1,
        };
        let rows = vec![row(Some(0), None, &[0])];
        assert!(AutoMatcher::mine(&at_cap, &rows, AutoMatchParams::default()).is_ok());
        let past = ArchiveDomains {
            terms: MAX_BINARY_FEATURES,
            ..at_cap
        };
        assert_eq!(
            AutoMatcher::mine(&past, &rows, AutoMatchParams::default()).unwrap_err(),
            AutoMatchError::TooManyFeatures {
                features: u64::from(MAX_BINARY_FEATURES) + 1,
                max: MAX_BINARY_FEATURES,
            }
        );
    }

    #[test]
    fn the_evidence_floor_is_not_decoration() {
        let rows = corr0_archive(6, 0, 54);
        let with = |min_evidence| {
            let p = AutoMatchParams {
                min_evidence,
                ..AutoMatchParams::default()
            };
            AutoMatcher::mine(&DOMAINS, &rows, p)
                .unwrap()
                .suggest(&new_doc(0))
        };
        assert!(tag_targets(&with(5)).contains(&2));
        assert!(!tag_targets(&with(7)).contains(&2));
    }

    #[test]
    fn the_confidence_floor_is_not_decoration() {
        // 8 of the 10 correspondent-0 documents carry tag 1 (0.8).
        let mut rows = Vec::new();
        for i in 0..10 {
            rows.push(row(Some(0), None, if i < 8 { &[1] } else { &[] }));
        }
        for _ in 0..30 {
            rows.push(row(Some(1), None, &[]));
        }
        let with = |min_confidence_ppm| {
            let p = AutoMatchParams {
                min_confidence_ppm,
                ..AutoMatchParams::default()
            };
            AutoMatcher::mine(&DOMAINS, &rows, p)
                .unwrap()
                .suggest(&new_doc(0))
        };
        assert!(tag_targets(&with(700_000)).contains(&1));
        assert!(!tag_targets(&with(900_000)).contains(&1));
    }

    #[test]
    fn an_assigned_target_is_never_suggested_again() {
        let m = AutoMatcher::mine(
            &DOMAINS,
            &corr0_archive(20, 0, 40),
            AutoMatchParams::default(),
        )
        .unwrap();
        // Baseline: the same setup does suggest it.
        assert!(tag_targets(&m.suggest(&new_doc(0))).contains(&2));
        let carrying = row(Some(0), None, &[2]);
        assert!(!tag_targets(&m.suggest(&carrying)).contains(&2));
    }

    #[test]
    fn absence_is_never_suggested() {
        // Correspondent 0 always carries tag 1 and never tag 3; correspondent 1
        // always carries tag 3. Given corr 0, "tag 3 absent" is certain.
        let mut rows = Vec::new();
        for _ in 0..20 {
            rows.push(row(Some(0), None, &[1]));
        }
        for _ in 0..20 {
            rows.push(row(Some(1), None, &[3]));
        }
        let data = encode(&DOMAINS, &rows);
        let oracle = CooccurrenceDistance::build(&data);
        let corr0 = Item::new(0, 1);
        let absent = oracle.distance(corr0, Item::new(Layout::tag_feature(3), 0));
        let present = oracle.distance(corr0, Item::new(Layout::tag_feature(3), 1));
        assert!(
            absent < present,
            "encoding must make absence the likelier category"
        );

        let m = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap();
        let s = m.suggest(&new_doc(0));
        assert!(
            tag_targets(&s).contains(&1),
            "positive fact still suggested"
        );
        assert!(!tag_targets(&s).contains(&3));
        for x in &s {
            match x.target {
                Target::Correspondent(c) => assert!(c < DOMAINS.correspondents),
                Target::DocumentType(t) => assert!(t < DOMAINS.document_types),
                Target::Tag(t) => assert!(t < DOMAINS.tags),
            }
        }
    }

    #[test]
    fn an_out_of_range_id_is_an_error_not_a_panic() {
        let rows = vec![row(Some(0), None, &[DOMAINS.tags])];
        let err = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap_err();
        assert_eq!(
            err,
            AutoMatchError::IdOutOfRange {
                what: "tag",
                id: DOMAINS.tags,
                domain: DOMAINS.tags
            }
        );
        let zero_k = AutoMatchParams {
            k: 0,
            ..AutoMatchParams::default()
        };
        assert_eq!(
            AutoMatcher::mine(&DOMAINS, &[], zero_k).unwrap_err(),
            AutoMatchError::ZeroK
        );
        // Empty archive is not an error.
        let empty = AutoMatcher::mine(&DOMAINS, &[], AutoMatchParams::default()).unwrap();
        assert_eq!(empty.rule_count(), 0);
    }

    #[test]
    fn single_valued_targets_yield_at_most_one_suggestion() {
        let mut rows = Vec::new();
        // tag 0 -> type 0 with confidence 1.0 (20 of 20).
        for _ in 0..20 {
            rows.push(row(None, Some(0), &[0]));
        }
        // tag 1 -> type 1 with confidence 0.8 (20 of 25).
        for _ in 0..20 {
            rows.push(row(None, Some(1), &[1]));
        }
        for _ in 0..5 {
            rows.push(row(None, None, &[1]));
        }
        for _ in 0..15 {
            rows.push(row(None, None, &[]));
        }
        let m = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap();
        // Both single-valued rules were mined, so the cap is what removes one.
        let mined = |t| m.rules.iter().any(|r| r.target == Target::DocumentType(t));
        assert!(mined(0) && mined(1));
        let s = m.suggest(&row(None, None, &[0, 1]));
        let types: Vec<&Suggestion> = s
            .iter()
            .filter(|x| matches!(x.target, Target::DocumentType(_)))
            .collect();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].target, Target::DocumentType(0));
    }

    /// Term 0 on 20 documents: 12 carry correspondent 0 (NOT eligible, e.g.
    /// an S-8 definition), 8 carry correspondent 1 (eligible, AUTO). 40
    /// filler documents have neither the term nor a correspondent, so both
    /// targets clear lift easily.
    fn preemption_archive() -> Vec<ArchiveRow> {
        let mut rows = Vec::new();
        for i in 0..20 {
            let corr = u32::from(i >= 12);
            rows.push(ArchiveRow {
                correspondent: Some(corr),
                terms: vec![0],
                ..ArchiveRow::default()
            });
        }
        for _ in 0..40 {
            rows.push(ArchiveRow::default());
        }
        rows
    }

    fn preemption_params() -> AutoMatchParams {
        AutoMatchParams {
            min_confidence_ppm: 300_000,
            ..AutoMatchParams::default()
        }
    }

    fn only_corr1(t: Target) -> bool {
        !matches!(t, Target::Correspondent(0))
    }

    fn new_doc_with_term0() -> ArchiveRow {
        ArchiveRow {
            terms: vec![0],
            ..ArchiveRow::default()
        }
    }

    #[test]
    fn an_ineligible_target_is_never_suggested() {
        // G2a: correspondent 0 has the strongest rule, but it is not eligible.
        let m = AutoMatcher::mine_eligible(
            &DOMAINS,
            &preemption_archive(),
            preemption_params(),
            MineScope {
                target: &only_corr1,
                cue: &|_| true,
            },
        )
        .unwrap();
        let s = m.suggest(&new_doc_with_term0());
        assert!(
            s.iter().all(|x| x.target != Target::Correspondent(0)),
            "{s:?}"
        );
        // Anti-vacuity: with everything eligible, correspondent 0 IS suggested,
        // so the fixture really has a rule for it.
        let all = AutoMatcher::mine(&DOMAINS, &preemption_archive(), preemption_params()).unwrap();
        assert!(all
            .suggest(&new_doc_with_term0())
            .iter()
            .any(|x| x.target == Target::Correspondent(0)));
    }

    #[test]
    fn an_eligible_target_outranked_by_an_ineligible_one_still_comes_back() {
        // G2b: correspondent 0 (P = 0.6) is nearer than correspondent 1
        // (P = 0.4), so the probe proposes only correspondent 0. Filtering the
        // mined rules or the suggestions afterwards would leave nothing.
        let m = AutoMatcher::mine_eligible(
            &DOMAINS,
            &preemption_archive(),
            preemption_params(),
            MineScope {
                target: &only_corr1,
                cue: &|_| true,
            },
        )
        .unwrap();
        let s = m.suggest(&new_doc_with_term0());
        assert!(
            s.iter().any(|x| x.target == Target::Correspondent(1)),
            "the eligible correspondent was pre-empted: {s:?}"
        );
        // The post-filter shape this replaces really does lose it.
        let all = AutoMatcher::mine(&DOMAINS, &preemption_archive(), preemption_params()).unwrap();
        let post_filtered: Vec<_> = all
            .suggest(&new_doc_with_term0())
            .into_iter()
            .filter(|x| only_corr1(x.target))
            .collect();
        assert!(
            post_filtered
                .iter()
                .all(|x| !matches!(x.target, Target::Correspondent(_))),
            "{post_filtered:?}"
        );
    }

    #[test]
    fn an_ineligible_target_still_works_as_a_cue() {
        // Correspondent 0 is ineligible as a target but its documents carry
        // tag 2: the cue must still fire.
        let rows = corr0_archive(10, 0, 30);
        let m = AutoMatcher::mine_eligible(
            &DOMAINS,
            &rows,
            AutoMatchParams::default(),
            MineScope {
                target: &|t| !matches!(t, Target::Correspondent(_)),
                cue: &|_| true,
            },
        )
        .unwrap();
        let s = m.suggest(&row(Some(0), None, &[]));
        assert_eq!(tag_targets(&s), vec![2], "{s:?}");
    }

    #[test]
    fn an_excluded_cue_never_carries_a_rule() {
        // G13: correspondent 0 predicts tag 2 perfectly; with correspondents
        // excluded as cues that rule must not exist, and a document carrying
        // only correspondent 0 gets no tag suggestion.
        let rows = corr0_archive(10, 0, 30);
        let everything = AutoMatcher::mine(&DOMAINS, &rows, AutoMatchParams::default()).unwrap();
        // Anti-vacuity: without the exclusion the rule exists.
        assert_eq!(
            tag_targets(&everything.suggest(&row(Some(0), None, &[]))),
            vec![2]
        );
        let m = AutoMatcher::mine_eligible(
            &DOMAINS,
            &rows,
            AutoMatchParams::default(),
            MineScope {
                target: &|_| true,
                cue: &|c| !matches!(c, Cue::Correspondent(_)),
            },
        )
        .unwrap();
        assert!(tag_targets(&m.suggest(&row(Some(0), None, &[]))).is_empty());
    }
}
