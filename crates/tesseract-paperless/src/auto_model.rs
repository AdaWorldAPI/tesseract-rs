//! The AUTO model and the S-8 ingest step: the glue between the archive's
//! stored metadata and the miner.
//!
//! Spec: `.claude/plans/archive-metadata-auto-match-v3.md` R3, R7, R8. This
//! module is the only one that needs all four of `store`, `search`,
//! `matching` and `auto-match`: it reads definitions and assignments from
//! [`crate::archive_meta`], turns stored `match_algorithm` bytes into
//! [`MatchRule`]s, tokenizes with the index's own analyzer
//! ([`SearchIndex::tokenize_text`]), and mines through
//! [`crate::auto_rows`] and [`AutoMatcher::mine_eligible`]. Keeping the glue
//! here is what lets each of those modules stay behind its own single
//! feature.
//!
//! # One model, one vocabulary
//!
//! An [`AutoModel`] owns its [`Mapping`] (dense ids, vocabulary, eligibility)
//! and the [`AutoMatcher`] mined against it, and [`AutoModel::suggest_for`]
//! maps a document's text to term ids with that same mapping. A caller never
//! handles a term id, so it cannot mix one run's ids with another run's
//! rules. A re-mine produces a new `AutoModel`; the web app swaps the whole
//! value in one step.
//!
//! # S-8 at ingest
//!
//! [`s8_matches`] is paperless-ngx's `set_correspondent` / `set_document_type`
//! / `set_tags` for this archive. Candidates are the non-retired, non-AUTO
//! definitions in **name order** (paperless-ngx's model ordering). For a
//! single-valued kind the first match is taken, and only when the document
//! has no assignment of that kind yet. For tags every match not already on
//! the document is taken. An AUTO definition compiles to never-match, so S-8
//! never writes one (spec F1).

use crate::archive_meta::{AssignmentRow, DefinitionRow, MetaKind};
use crate::auto_match::{
    AutoMatchError, AutoMatchParams, AutoMatcher, MineScope, Suggestion, Target,
};
use crate::auto_rows::{
    build_training, Assignment, Definition, DocContent, Kind, Mapping, RowBuildError, VocabParams,
};
use crate::matching::{MatchAlgorithm, MatchRule};
use crate::search::SearchIndex;
use crate::store::{DocumentRow, LanceStore, StoreError};

/// Why a model could not be built.
#[derive(Debug)]
pub enum AutoModelError {
    /// Reading the archive failed.
    Store(StoreError),
    /// A stored document's IR did not decode.
    DocIr {
        /// The document.
        hash: String,
        /// Why.
        reason: String,
    },
    /// The archive could not be turned into mining input (for example no
    /// content term survived the vocabulary filters).
    Rows(RowBuildError),
    /// The miner refused the input.
    Mine(AutoMatchError),
}

impl core::fmt::Display for AutoModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "reading the archive: {e}"),
            Self::DocIr { hash, reason } => write!(f, "document {hash}: {reason}"),
            Self::Rows(e) => write!(f, "{e}"),
            Self::Mine(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AutoModelError {}

impl From<StoreError> for AutoModelError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

/// Is `def` a target of the AUTO tier? Its stored `match_algorithm` is
/// paperless-ngx's `MATCH_AUTO`.
#[must_use]
pub fn is_auto(def: &DefinitionRow) -> bool {
    MatchAlgorithm::from_paperless(def.match_algorithm) == Some(MatchAlgorithm::Auto)
}

/// The rule `def` carries, or `None` for an algorithm number paperless-ngx
/// does not define (such a definition never matches).
#[must_use]
pub fn match_rule(def: &DefinitionRow) -> Option<MatchRule> {
    Some(MatchRule {
        algorithm: MatchAlgorithm::from_paperless(def.match_algorithm)?,
        pattern: def.match_pattern.clone(),
        case_insensitive: def.case_insensitive,
    })
}

fn kind_of(k: MetaKind) -> Kind {
    match k {
        MetaKind::Correspondent => Kind::Correspondent,
        MetaKind::DocumentType => Kind::DocumentType,
        MetaKind::Tag => Kind::Tag,
    }
}

fn meta_kind_of(k: Kind) -> MetaKind {
    match k {
        Kind::Correspondent => MetaKind::Correspondent,
        Kind::DocumentType => MetaKind::DocumentType,
        Kind::Tag => MetaKind::Tag,
    }
}

fn definitions(defs: &[DefinitionRow]) -> Vec<Definition> {
    defs.iter()
        .map(|d| Definition {
            kind: kind_of(d.kind),
            definition_id: d.definition_id,
            name: d.name.clone(),
            is_auto: is_auto(d),
            retired: d.retired,
        })
        .collect()
}

fn assignments(rows: &[AssignmentRow]) -> Vec<Assignment> {
    rows.iter()
        .map(|a| Assignment {
            content_sha256_hex: a.content_sha256_hex.clone(),
            kind: kind_of(a.kind),
            definition_id: a.definition_id,
        })
        .collect()
}

/// A stored document as mining input: its plain text and field keys.
///
/// # Errors
/// [`AutoModelError::DocIr`] if the stored IR does not decode.
pub fn doc_content(row: &DocumentRow) -> Result<DocContent, AutoModelError> {
    let ir = row.doc_ir().map_err(|e| AutoModelError::DocIr {
        hash: row.content_sha256_hex.clone(),
        reason: e.to_string(),
    })?;
    Ok(DocContent {
        content_sha256_hex: row.content_sha256_hex.clone(),
        text: crate::render::plain_text(&ir),
        field_keys: ir.fields.iter().map(|f| f.key.clone()).collect(),
    })
}

/// Everything a mining run reads from the archive, loaded up front so the
/// CPU-bound build can run off the async runtime.
#[derive(Debug, Clone)]
pub struct MineInputs {
    /// Every definition, retired ones included.
    pub definitions: Vec<DefinitionRow>,
    /// Every assignment.
    pub assignments: Vec<AssignmentRow>,
    /// The reviewed documents, the only training rows (spec R3a).
    pub reviewed: Vec<DocContent>,
}

impl MineInputs {
    /// Load the inputs from `store`. A reviewed hash whose document is gone
    /// (an orphan the next reconcile removes) is skipped.
    ///
    /// # Errors
    /// [`AutoModelError::Store`] or [`AutoModelError::DocIr`].
    pub async fn load(store: &LanceStore) -> Result<Self, AutoModelError> {
        let meta = store.meta();
        let definitions = meta.definitions().await?;
        let assignments = meta.all_assignments().await?;
        let mut reviewed = Vec::new();
        for hash in meta.reviewed_hashes().await? {
            if let Some(row) = store.get(&hash).await? {
                reviewed.push(doc_content(&row)?);
            }
        }
        Ok(Self {
            definitions,
            assignments,
            reviewed,
        })
    }
}

/// One AUTO suggestion, resolved back to the definition it would assign.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoSuggestion {
    /// The definition's kind.
    pub kind: MetaKind,
    /// The definition.
    pub definition_id: u32,
    /// Its name at mining time.
    pub name: String,
    /// The miner's suggestion: truth, evidence, and the cues that fired.
    pub suggestion: Suggestion,
}

/// A mined AUTO model. See the module doc.
#[derive(Debug)]
pub struct AutoModel {
    mapping: Mapping,
    matcher: AutoMatcher,
    definitions: Vec<DefinitionRow>,
}

impl AutoModel {
    /// Mine a model from loaded inputs. Targets are the non-retired AUTO
    /// definitions; an AUTO definition is never a cue (spec R3, F2).
    ///
    /// # Errors
    /// [`AutoModelError::Rows`] (for example no content term survived) or
    /// [`AutoModelError::Mine`].
    pub fn build(
        inputs: &MineInputs,
        tokenize: &dyn Fn(&str) -> Vec<String>,
        params: AutoMatchParams,
        vocab: VocabParams,
    ) -> Result<Self, AutoModelError> {
        let training = build_training(
            &definitions(&inputs.definitions),
            &assignments(&inputs.assignments),
            &inputs.reviewed,
            tokenize,
            vocab,
        )
        .map_err(AutoModelError::Rows)?;
        let mapping = training.mapping;
        let matcher = AutoMatcher::mine_eligible(
            &training.domains,
            &training.rows,
            params,
            MineScope {
                target: &|t| mapping.target_eligible(t),
                cue: &|c| mapping.cue_allowed(c),
            },
        )
        .map_err(AutoModelError::Mine)?;
        Ok(Self {
            mapping,
            matcher,
            definitions: inputs.definitions.clone(),
        })
    }

    /// Load from `store` and mine, tokenizing with `search`'s analyzer.
    ///
    /// The build itself is CPU-bound; a server calls [`MineInputs::load`]
    /// and runs [`Self::build`] on a blocking thread instead.
    ///
    /// # Errors
    /// As [`MineInputs::load`] and [`Self::build`].
    pub async fn mine(
        store: &LanceStore,
        search: &SearchIndex,
        params: AutoMatchParams,
        vocab: VocabParams,
    ) -> Result<Self, AutoModelError> {
        let inputs = MineInputs::load(store).await?;
        Self::build(&inputs, &index_tokenizer(search), params, vocab)
    }

    /// How many rules the model holds.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.matcher.rule_count()
    }

    /// Suggestions for `doc`, given all of ITS assignments (any source).
    ///
    /// Text is mapped to term ids with this model's own vocabulary. A kind
    /// the document already has gets no suggestion, and neither does a tag
    /// it already carries.
    #[must_use]
    pub fn suggest_for(
        &self,
        doc_assignments: &[AssignmentRow],
        doc: &DocContent,
        tokenize: &dyn Fn(&str) -> Vec<String>,
    ) -> Vec<AutoSuggestion> {
        let row = self
            .mapping
            .row_for(&assignments(doc_assignments), doc, tokenize);
        self.matcher
            .suggest(&row)
            .into_iter()
            .filter_map(|s| {
                let (kind, definition_id) = self.mapping.definition_of(s.target)?;
                let kind = meta_kind_of(kind);
                let name = self
                    .definitions
                    .iter()
                    .find(|d| d.kind == kind && d.definition_id == definition_id)
                    .map(|d| d.name.clone())?;
                Some(AutoSuggestion {
                    kind,
                    definition_id,
                    name,
                    suggestion: s,
                })
            })
            .collect()
    }

    /// The target a dense id stands for; for tests and diagnostics.
    #[must_use]
    pub fn definition_of(&self, t: Target) -> Option<(MetaKind, u32)> {
        self.mapping
            .definition_of(t)
            .map(|(k, id)| (meta_kind_of(k), id))
    }
}

/// `search`'s analyzer as a tokenizer closure.
///
/// [`SearchIndex::tokenize_text`] only fails if the `text` field has no
/// registered analyzer, which this crate's schema rules out; such a failure
/// yields no tokens rather than a panic.
pub fn index_tokenizer(search: &SearchIndex) -> impl Fn(&str) -> Vec<String> + '_ {
    move |t: &str| search.tokenize_text(t).unwrap_or_default()
}

/// S-8 at ingest: the assignments paperless-ngx's matching rules would make
/// on a document with text `text` and current assignments `existing`.
///
/// `defs` must be in [`crate::archive_meta::MetaStore::definitions`] order
/// (kind, then name), which is what makes "first match" mean first by name.
#[must_use]
pub fn s8_matches(
    defs: &[DefinitionRow],
    existing: &[AssignmentRow],
    text: &str,
) -> Vec<(MetaKind, u32)> {
    let has_kind = |k: MetaKind| existing.iter().any(|a| a.kind == k);
    let has = |k: MetaKind, id: u32| {
        existing
            .iter()
            .any(|a| a.kind == k && a.definition_id == id)
    };
    let mut out = Vec::new();
    for kind in [
        MetaKind::Correspondent,
        MetaKind::DocumentType,
        MetaKind::Tag,
    ] {
        if kind.is_single_valued() && has_kind(kind) {
            continue;
        }
        let mut matches = defs
            .iter()
            // `!is_auto` is belt and braces: `MatchRule::compile` already
            // maps AUTO to never-match, and G11c goes red only when BOTH
            // guards are removed (measured). Kept so the F1 rule does not
            // hang on a detail of the matching module.
            .filter(|d| d.kind == kind && !d.retired && !is_auto(d))
            .filter(|d| match_rule(d).is_some_and(|r| r.matches(text)));
        if kind.is_single_valued() {
            let all: Vec<&DefinitionRow> = matches.collect();
            if all.len() > 1 {
                eprintln!(
                    "tesseract-paperless: {} {} matched; taking the first by name",
                    all.len(),
                    kind.as_str()
                );
            }
            if let Some(first) = all.first() {
                out.push((kind, first.definition_id));
            }
        } else {
            out.extend(
                matches
                    .by_ref()
                    .filter(|d| !has(kind, d.definition_id))
                    .map(|d| (kind, d.definition_id)),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive_meta::Source;
    use crate::kv::ContentSha256;
    use ogar_doc_ir::{BBoxRail, DocIr, DocPage, Geometry, Provenance, Rail, Region, RegionKind};

    fn ir(text: &str) -> DocIr {
        DocIr {
            version: ogar_doc_ir::DOC_IR_VERSION.to_string(),
            source: Provenance::Ocr,
            geometry: Geometry::DomOrder,
            content_sha256: [0u8; 32],
            mime: "image/png".to_string(),
            pages: vec![DocPage {
                number: 0,
                width: 100,
                height: 100,
                regions: vec![Region {
                    kind: RegionKind::Text,
                    bbox: BBoxRail {
                        tl: Rail { x: 0, y: 0 },
                        br: Rail { x: 10, y: 10 },
                    },
                    reading_order: 0,
                    text: Some(text.to_string()),
                    cells: Vec::new(),
                    children: Vec::new(),
                }],
            }],
            fields: Vec::new(),
        }
    }

    struct Fixture {
        _dirs: (tempfile::TempDir, tempfile::TempDir),
        store: LanceStore,
        search: SearchIndex,
        tag: u32,
        next: u32,
    }

    impl Fixture {
        /// An archive with one tag created through the DEFAULT create path.
        async fn new() -> Self {
            let db = tempfile::tempdir().expect("tempdir");
            let ix = tempfile::tempdir().expect("tempdir");
            let store = LanceStore::connect(&db.path().to_string_lossy())
                .await
                .expect("connect");
            let search = SearchIndex::open_or_create(ix.path()).expect("index");
            let tag = store
                .meta()
                .create_definition(MetaKind::Tag, "Energie", 1)
                .await
                .expect("create")
                .definition_id;
            Self {
                _dirs: (db, ix),
                store,
                search,
                tag,
                next: 0,
            }
        }

        /// Store a document; returns its hash.
        async fn doc(&mut self, text: &str, tagged: bool, reviewed: bool) -> String {
            self.next += 1;
            let hash = ContentSha256::of(format!("{text} #{}", self.next).as_bytes());
            let hex = format!("{hash:?}");
            self.store
                .put(&hash, None, 90, false, &ir(text), 1, None)
                .await
                .expect("put");
            if tagged {
                self.store
                    .meta()
                    .assign(&hex, MetaKind::Tag, self.tag, Source::Manual, 1)
                    .await
                    .expect("assign");
            }
            if reviewed {
                self.store
                    .meta()
                    .mark_reviewed(&hex, 1)
                    .await
                    .expect("review");
            }
            hex
        }

        /// 10 reviewed "stadtwerke" documents carrying the tag, 20 reviewed
        /// "miete" documents without it.
        async fn base(&mut self) {
            for _ in 0..10 {
                self.doc("stadtwerke abrechnung strom", true, true).await;
            }
            for _ in 0..20 {
                self.doc("miete wohnung vermieter", false, true).await;
            }
        }

        async fn mine(&self) -> AutoModel {
            AutoModel::mine(
                &self.store,
                &self.search,
                AutoMatchParams::default(),
                VocabParams::default(),
            )
            .await
            .expect("mine")
        }

        fn new_doc(text: &str) -> DocContent {
            DocContent {
                content_sha256_hex: "new".to_string(),
                text: text.to_string(),
                field_keys: Vec::new(),
            }
        }

        fn energie(&self, model: &AutoModel, text: &str) -> Option<AutoSuggestion> {
            model
                .suggest_for(&[], &Self::new_doc(text), &index_tokenizer(&self.search))
                .into_iter()
                .find(|s| s.kind == MetaKind::Tag && s.definition_id == self.tag)
        }
    }

    /// G14: a tag made through the default create path is an AUTO target and,
    /// given evidence, is suggested. Red if create defaulted to ANY.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_default_created_tag_is_suggested() {
        let mut f = Fixture::new().await;
        f.base().await;
        let model = f.mine().await;
        let s = f.energie(&model, "Stadtwerke Rechnung").expect("suggested");
        assert_eq!(s.name, "Energie");
        assert_eq!(s.suggestion.evidence, 10);
        // Can-stay-silent: text without the cue gets nothing.
        assert!(f.energie(&model, "miete wohnung").is_none());
    }

    /// G9: unreviewed documents are not training rows. 20 unreviewed
    /// "stadtwerke" documents without the tag would pull the rule's
    /// confidence to 10/30, below the 0.7 floor.
    #[tokio::test(flavor = "multi_thread")]
    async fn unreviewed_documents_do_not_train() {
        let mut f = Fixture::new().await;
        f.base().await;
        for _ in 0..20 {
            f.doc("stadtwerke abrechnung strom", false, false).await;
        }
        let model = f.mine().await;
        assert!(f.energie(&model, "stadtwerke").is_some());
    }

    /// G10: accepting a suggestion on unreviewed documents changes no rule
    /// statistic; marking them reviewed then raises the evidence by exactly
    /// their number. Red if accept also marked the document reviewed.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_accept_counts_only_once_the_document_is_reviewed() {
        let mut f = Fixture::new().await;
        f.base().await;
        let mut pending = Vec::new();
        for _ in 0..5 {
            pending.push(f.doc("stadtwerke abrechnung strom", false, false).await);
        }
        for h in &pending {
            f.store
                .meta()
                .assign(h, MetaKind::Tag, f.tag, Source::AutoAccepted, 2)
                .await
                .expect("accept");
        }
        let before = f.energie(&f.mine().await, "stadtwerke").expect("suggested");
        assert_eq!(before.suggestion.evidence, 10);
        for h in &pending {
            f.store.meta().mark_reviewed(h, 3).await.expect("review");
        }
        let after = f.energie(&f.mine().await, "stadtwerke").expect("suggested");
        assert_eq!(after.suggestion.evidence, 15);
    }

    /// G10b: the negative channel can fire. Reviewing documents that carry
    /// the cue but not the tag lowers the rule's frequency.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_reviewed_negative_lowers_the_rule() {
        let mut f = Fixture::new().await;
        f.base().await;
        let before = f.energie(&f.mine().await, "stadtwerke").expect("suggested");
        for _ in 0..3 {
            f.doc("stadtwerke abrechnung strom", false, true).await;
        }
        let after = f
            .energie(&f.mine().await, "stadtwerke")
            .expect("still suggested");
        assert!(
            after.suggestion.truth.frequency < before.suggestion.truth.frequency,
            "{:?} -> {:?}",
            before.suggestion.truth,
            after.suggestion.truth
        );
    }

    /// G15: `suggest_for` maps text with the model's OWN vocabulary. Two
    /// models mined from archives with disjoint vocabularies each answer only
    /// to their own cue.
    #[tokio::test(flavor = "multi_thread")]
    async fn each_model_answers_from_its_own_vocabulary() {
        let mut a = Fixture::new().await;
        a.base().await;
        let mut b = Fixture::new().await;
        for _ in 0..10 {
            b.doc("gaswerk zaehler verbrauch", true, true).await;
        }
        for _ in 0..20 {
            b.doc("versicherung police beitrag", false, true).await;
        }
        let (ma, mb) = (a.mine().await, b.mine().await);
        assert!(a.energie(&ma, "stadtwerke").is_some());
        assert!(a.energie(&ma, "gaswerk").is_none());
        assert!(b.energie(&mb, "gaswerk").is_some());
        assert!(b.energie(&mb, "stadtwerke").is_none());
    }

    fn def(kind: MetaKind, id: u32, name: &str, alg: u8, pattern: &str) -> DefinitionRow {
        DefinitionRow {
            kind,
            definition_id: id,
            name: name.to_string(),
            match_algorithm: alg,
            match_pattern: pattern.to_string(),
            case_insensitive: true,
            retired: false,
            created_at_unix_ms: 0,
        }
    }

    fn assigned(kind: MetaKind, id: u32) -> AssignmentRow {
        AssignmentRow {
            content_sha256_hex: "h".to_string(),
            kind,
            definition_id: id,
            source: Source::Manual,
            assigned_at_unix_ms: 0,
        }
    }

    const ANY: u8 = 1;

    /// Definitions in `MetaStore::definitions` order: kind, then name.
    fn sorted(mut v: Vec<DefinitionRow>) -> Vec<DefinitionRow> {
        v.sort_by(|a, b| (a.kind.as_str(), &a.name).cmp(&(b.kind.as_str(), &b.name)));
        v
    }

    /// G11a: S-8 never overwrites an existing single-valued assignment.
    #[test]
    fn s8_never_overwrites_an_assignment() {
        let defs = sorted(vec![def(
            MetaKind::Correspondent,
            0,
            "Stadtwerke",
            ANY,
            "stadtwerke",
        )]);
        let fresh = s8_matches(&defs, &[], "Stadtwerke Rechnung");
        assert_eq!(fresh, vec![(MetaKind::Correspondent, 0)]);
        let kept = s8_matches(
            &defs,
            &[assigned(MetaKind::Correspondent, 7)],
            "Stadtwerke Rechnung",
        );
        assert!(kept.is_empty(), "{kept:?}");
    }

    /// G11b: several matches for a single-valued kind take the first by NAME,
    /// not by id. Red if candidates were ordered by id.
    #[test]
    fn s8_takes_the_first_match_by_name() {
        let defs = sorted(vec![
            def(MetaKind::Correspondent, 0, "Zeta Werke", ANY, "rechnung"),
            def(MetaKind::Correspondent, 1, "Alpha Werke", ANY, "rechnung"),
        ]);
        assert_eq!(
            s8_matches(&defs, &[], "Rechnung"),
            vec![(MetaKind::Correspondent, 1)]
        );
    }

    /// G11c: S-8 never writes an AUTO definition, even one whose stored
    /// pattern would match; nor a retired one. Tags take every other match
    /// not already present.
    #[test]
    fn s8_skips_auto_retired_and_present_tags() {
        let mut retired = def(MetaKind::Tag, 2, "Alt", ANY, "rechnung");
        retired.retired = true;
        let defs = sorted(vec![
            def(
                MetaKind::Tag,
                0,
                "Auto",
                crate::archive_meta::MATCH_AUTO,
                "rechnung",
            ),
            def(MetaKind::Tag, 1, "Rechnung", ANY, "rechnung"),
            def(MetaKind::Tag, 3, "Strom", ANY, "strom"),
            def(MetaKind::Tag, 4, "Schon da", ANY, "rechnung"),
            retired,
        ]);
        let got = s8_matches(&defs, &[assigned(MetaKind::Tag, 4)], "Rechnung Strom");
        assert_eq!(got, vec![(MetaKind::Tag, 1), (MetaKind::Tag, 3)]);
    }
}
