//! The archive's organizing layer on lance-graph's report substrate.
//!
//! ```text
//!   A SEARCH HIT IS A MASK, NOT AN ID LIST.  A COUNT IS A FOLD, NOT A QUERY.
//! ```
//!
//! paperless-ngx organizes a document along a small, fixed set of axes —
//! correspondent, document type, tags, storage path (`OGAR-DOC-INGESTION-SPINE`
//! S-7) — and every screen that summarizes the archive is a count over them:
//! the per-tag `document_count`, saved-view totals, the dashboard. There the
//! count is an ORM `GROUP BY` behind a permission subquery, and a full-text
//! search enters it as an id list (`search_ids` → `pk__in=ids`).
//!
//! Here the same things are lanes and masks a report folds:
//!
//! | paperless-ngx | here |
//! |---|---|
//! | correspondent / type / storage path FK | an ordinal coordinate lane (`0` = none) |
//! | created date, grouped by month | an ordinal coordinate lane (month index) |
//! | tags (M2M) | ONE resident mask per tag — a set, never an exploded row |
//! | owner / viewers / viewer groups | an owner lane + one mask per viewer / group |
//! | soft delete (`deleted_at`) | one mask |
//! | `build_permission_filter` | [`visible_to`] — a lazy [`Selection`], never a bitmap |
//! | Tantivy `search_ids` | [`SearchMaskCollector`] — writes bits, emits no ids |
//!
//! Nothing in this module evaluates a predicate or counts a row; it only says
//! WHERE the axes live. The fold is `lance-graph-report`'s, lowered into the
//! one mask-RISC evaluator.
//!
//! Tags are a SET coordinate ([`tags_axis`], lance-graph's
//! `CoordSpec::MaskSet`): one plan puts a document in every tag it carries, so
//! "count per tag" and "tag by correspondent" are ordinary pivots. A tag cell
//! does not sum to the selected population — a document with two tags counts
//! in both, one with none counts in neither, as in paperless-ngx.
//!
//! **Known gap, stated where it bites:** the substrate still runs one
//! population pass per tag. Folding every tag in one pass needs a keyed
//! multi-membership aggregation that mask-RISC does not have yet.

use std::sync::Arc;

use lance_graph_report::{
    AbiBatch, BatchError, CmpOp, Column, CoordSpec, FieldId, MaskId, Scalar, Selection, SourceId,
};
use tantivy::collector::{Collector, SegmentCollector};
use tantivy::columnar::Column as FastColumn;
use tantivy::{DocId, Score, SegmentOrdinal, SegmentReader};

/// Correspondent axis — ordinal `0` = no correspondent, `c + 1` = correspondent `c`.
pub const CORRESPONDENT: FieldId = FieldId(1);
/// Document-type axis — same encoding as [`CORRESPONDENT`].
pub const DOCUMENT_TYPE: FieldId = FieldId(2);
/// Storage-path axis — same encoding as [`CORRESPONDENT`].
pub const STORAGE_PATH: FieldId = FieldId(3);
/// Created month, as a month index the caller chose an origin for.
pub const CREATED_MONTH: FieldId = FieldId(4);
/// Owner — ordinal `0` = no owner (public in paperless-ngx), `u + 1` = user `u`.
pub const OWNER: FieldId = FieldId(5);

/// Soft-deleted documents (the trash).
pub const DELETED: MaskId = MaskId(1);
/// The current full-text match, attached per query with [`with_search`].
pub const SEARCH: MaskId = MaskId(2);

const SPAN: u32 = 1 << 20;
const TAG_BASE: u32 = SPAN;
const VIEWER_BASE: u32 = 2 * SPAN;
const GROUP_BASE: u32 = 3 * SPAN;

/// The resident mask holding every document carrying `tag`.
///
/// # Panics
/// If `tag` is outside the mask id span (2^20).
#[must_use]
pub fn tag_mask(tag: u32) -> MaskId {
    assert!(tag < SPAN, "tag id {tag} exceeds the mask id span");
    MaskId(TAG_BASE + tag)
}

/// The tag axis: a set coordinate over the tag masks [`build_batch`] attaches,
/// member `t` being tag `t`. A plan over an archive with no tags is refused
/// (`ReportError::EmptyMaskSet`) rather than read as an empty axis.
#[must_use]
pub fn tags_axis(domains: &AxisDomains) -> CoordSpec {
    CoordSpec::MaskSet {
        base: tag_mask(0),
        count: domains.tags,
    }
}

/// The resident mask of documents explicitly shared with `user`.
///
/// # Panics
/// If `user` is outside the mask id span (2^20).
#[must_use]
pub fn viewer_mask(user: u32) -> MaskId {
    assert!(user < SPAN, "user id {user} exceeds the mask id span");
    MaskId(VIEWER_BASE + user)
}

/// The resident mask of documents shared with `group`.
///
/// # Panics
/// If `group` is outside the mask id span (2^20).
#[must_use]
pub fn group_mask(group: u32) -> MaskId {
    assert!(group < SPAN, "group id {group} exceeds the mask id span");
    MaskId(GROUP_BASE + group)
}

/// Domain sizes of every axis — ids are dense `0..n` per axis.
#[derive(Debug, Clone, Copy)]
pub struct AxisDomains {
    /// Number of correspondents.
    pub correspondents: u32,
    /// Number of document types.
    pub document_types: u32,
    /// Number of storage paths.
    pub storage_paths: u32,
    /// Number of month buckets.
    pub months: u32,
    /// Number of users.
    pub users: u32,
    /// Number of groups.
    pub groups: u32,
    /// Number of tags.
    pub tags: u32,
}

/// One archived document's organizing metadata — what paperless-ngx keeps on
/// its `Document` row and M2M tables, with ids already dense per axis.
#[derive(Debug, Clone, Default)]
pub struct ArchiveDoc {
    /// Correspondent, if any.
    pub correspondent: Option<u32>,
    /// Document type, if any.
    pub document_type: Option<u32>,
    /// Storage path, if any.
    pub storage_path: Option<u32>,
    /// Created month index.
    pub created_month: u32,
    /// Owner, if any (none = visible to everyone).
    pub owner: Option<u32>,
    /// Tags.
    pub tags: Vec<u32>,
    /// Users the document is explicitly shared with.
    pub viewers: Vec<u32>,
    /// Groups the document is shared with.
    pub viewer_groups: Vec<u32>,
    /// Soft-deleted.
    pub deleted: bool,
}

/// Why the archive's axes could not be laid out.
#[derive(Debug)]
pub enum AxesError {
    /// An id is outside its declared domain.
    OutOfDomain {
        /// Which axis.
        axis: &'static str,
        /// The offending id.
        id: u32,
        /// The declared domain size.
        domain: u32,
    },
    /// The report batch refused a lane or mask.
    Batch(BatchError),
}

impl std::fmt::Display for AxesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfDomain { axis, id, domain } => {
                write!(f, "{axis} id {id} outside domain 0..{domain}")
            }
            Self::Batch(e) => write!(f, "report batch: {e:?}"),
        }
    }
}

impl std::error::Error for AxesError {}

impl From<BatchError> for AxesError {
    fn from(e: BatchError) -> Self {
        Self::Batch(e)
    }
}

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

fn set(words: &mut [u64], row: usize) {
    words[row >> 6] |= 1 << (row & 63);
}

fn nullable(axis: &'static str, v: Option<u32>, domain: u32) -> Result<u32, AxesError> {
    match v {
        None => Ok(0),
        Some(id) if id < domain => Ok(id + 1),
        Some(id) => Err(AxesError::OutOfDomain { axis, id, domain }),
    }
}

fn check(axis: &'static str, id: u32, domain: u32) -> Result<(), AxesError> {
    if id < domain {
        Ok(())
    } else {
        Err(AxesError::OutOfDomain { axis, id, domain })
    }
}

/// Lay the archive's axes out as one report batch: row `i` is `docs[i]`.
///
/// Every tag, viewer and group mask is attached even when empty, so a plan
/// naming any id in the declared domain resolves.
///
/// # Errors
/// [`AxesError::OutOfDomain`] when a document names an id outside
/// `domains`; [`AxesError::Batch`] if the batch refuses a lane.
pub fn build_batch(
    source: SourceId,
    generation: u32,
    domains: AxisDomains,
    docs: &[ArchiveDoc],
) -> Result<AbiBatch, AxesError> {
    let n = docs.len();
    let w = words_for(n);
    let mut corr = Vec::with_capacity(n);
    let mut kind = Vec::with_capacity(n);
    let mut path = Vec::with_capacity(n);
    let mut month = Vec::with_capacity(n);
    let mut owner = Vec::with_capacity(n);
    let mut tags = vec![vec![0u64; w]; domains.tags as usize];
    let mut viewers = vec![vec![0u64; w]; domains.users as usize];
    let mut groups = vec![vec![0u64; w]; domains.groups as usize];
    let mut deleted = vec![0u64; w];
    for (row, d) in docs.iter().enumerate() {
        corr.push(nullable(
            "correspondent",
            d.correspondent,
            domains.correspondents,
        )?);
        kind.push(nullable(
            "document_type",
            d.document_type,
            domains.document_types,
        )?);
        path.push(nullable(
            "storage_path",
            d.storage_path,
            domains.storage_paths,
        )?);
        check("created_month", d.created_month, domains.months)?;
        month.push(d.created_month);
        owner.push(nullable("owner", d.owner, domains.users)?);
        for &t in &d.tags {
            check("tag", t, domains.tags)?;
            set(&mut tags[t as usize], row);
        }
        for &u in &d.viewers {
            check("viewer", u, domains.users)?;
            set(&mut viewers[u as usize], row);
        }
        for &g in &d.viewer_groups {
            check("viewer_group", g, domains.groups)?;
            set(&mut groups[g as usize], row);
        }
        if d.deleted {
            set(&mut deleted, row);
        }
    }
    let mut b = AbiBatch::new(source, generation, n)
        .with_column(Column::coordinate(
            CORRESPONDENT,
            corr.into(),
            domains.correspondents + 1,
        ))?
        .with_column(Column::coordinate(
            DOCUMENT_TYPE,
            kind.into(),
            domains.document_types + 1,
        ))?
        .with_column(Column::coordinate(
            STORAGE_PATH,
            path.into(),
            domains.storage_paths + 1,
        ))?
        .with_column(Column::coordinate(
            CREATED_MONTH,
            month.into(),
            domains.months,
        ))?
        .with_column(Column::coordinate(OWNER, owner.into(), domains.users + 1))?
        .with_mask(DELETED, deleted.into())?;
    for (t, words) in (0..domains.tags).zip(tags) {
        b = b.with_mask(tag_mask(t), words.into())?;
    }
    for (u, words) in (0..domains.users).zip(viewers) {
        b = b.with_mask(viewer_mask(u), words.into())?;
    }
    for (g, words) in (0..domains.groups).zip(groups) {
        b = b.with_mask(group_mask(g), words.into())?;
    }
    Ok(b)
}

/// The batch with a search result attached as [`SEARCH`]. Lanes and resident
/// masks are shared by `Arc`; only the new plane is added.
///
/// # Errors
/// [`BatchError`] if `words` does not cover the batch or a search mask is
/// already attached.
pub fn with_search(batch: &AbiBatch, words: Arc<[u64]>) -> Result<AbiBatch, BatchError> {
    batch.clone().with_mask(SEARCH, words)
}

/// paperless-ngx's visibility rule for a non-superuser
/// (`documents/search/_backend.py::build_permission_filter`): unowned, owned
/// by `user`, shared with `user`, or shared with one of `groups` — and not in
/// the trash. Lazy: nothing is evaluated until a plan folds it.
#[must_use]
pub fn visible_to(user: u32, groups: &[u32]) -> Selection {
    let mut s = Selection::cmp(OWNER, CmpOp::Eq, Scalar::Ordinal(0))
        .or(Selection::cmp(OWNER, CmpOp::Eq, Scalar::Ordinal(user + 1)))
        .or(Selection::Mask(viewer_mask(user)));
    for &g in groups {
        s = s.or(Selection::Mask(group_mask(g)));
    }
    s.and_not(Selection::Mask(DELETED))
}

/// A superuser sees every document not in the trash
/// (`documents/permissions.py::get_document_count_filter_for_user`).
#[must_use]
pub fn superuser() -> Selection {
    Selection::All.and_not(Selection::Mask(DELETED))
}

/// A Tantivy [`Collector`] that turns a match into a row mask directly.
///
/// Each matching document's archive row is read from a `u64` FAST field and
/// its bit set; nothing ranks, nothing is stored, and no id list is formed —
/// the mask is what crosses into the report, where paperless-ngx hands
/// `search_ids` back to the ORM as `pk__in=ids`.
#[derive(Debug, Clone)]
pub struct SearchMaskCollector {
    row_field: String,
    n_rows: usize,
}

impl SearchMaskCollector {
    /// Collect into a mask over `n_rows` archive rows, reading each match's
    /// row from the `u64` FAST field `row_field`.
    #[must_use]
    pub fn new(row_field: impl Into<String>, n_rows: usize) -> Self {
        Self {
            row_field: row_field.into(),
            n_rows,
        }
    }
}

/// A search result as a row mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchMask {
    /// One bit per archive row.
    pub words: Vec<u64>,
    /// Matches whose row was missing or outside the archive. A non-zero
    /// value means the index and the batch disagree — a caller must refuse
    /// the mask rather than fold it.
    pub stray: u64,
}

impl SearchMask {
    /// Number of matched rows.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }
}

/// Per-segment half of [`SearchMaskCollector`].
pub struct SegmentMask {
    rows: FastColumn<u64>,
    mask: SearchMask,
    n_rows: usize,
}

impl Collector for SearchMaskCollector {
    type Fruit = SearchMask;
    type Child = SegmentMask;

    fn for_segment(
        &self,
        _segment_local_id: SegmentOrdinal,
        segment: &SegmentReader,
    ) -> tantivy::Result<SegmentMask> {
        Ok(SegmentMask {
            rows: segment.fast_fields().u64(&self.row_field)?,
            mask: SearchMask {
                words: vec![0; words_for(self.n_rows)],
                stray: 0,
            },
            n_rows: self.n_rows,
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, fruits: Vec<SearchMask>) -> tantivy::Result<SearchMask> {
        let mut out = SearchMask {
            words: vec![0; words_for(self.n_rows)],
            stray: 0,
        };
        for f in fruits {
            for (o, w) in out.words.iter_mut().zip(&f.words) {
                *o |= w;
            }
            out.stray += f.stray;
        }
        Ok(out)
    }
}

impl SegmentCollector for SegmentMask {
    type Fruit = SearchMask;

    fn collect(&mut self, doc: DocId, _score: Score) {
        // A row that does not fit `usize` is out of range by definition, so the
        // conversion failing and the bound failing are the same stray.
        match self
            .rows
            .first(doc)
            .and_then(|r| usize::try_from(r).ok())
            .filter(|&r| r < self.n_rows)
        {
            Some(r) => set(&mut self.mask.words, r),
            None => self.mask.stray += 1,
        }
    }

    fn harvest(self) -> SearchMask {
        self.mask
    }
}
