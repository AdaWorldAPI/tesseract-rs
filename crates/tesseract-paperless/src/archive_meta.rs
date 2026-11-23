//! Filing metadata for the archive: definitions (correspondent, document
//! type, tag), their assignments to documents, and the "reviewed" flag.
//!
//! # The three tables
//!
//! | table | one row per | columns |
//! |---|---|---|
//! | [`TAXONOMY_TABLE`] | `(kind, definition_id)` | kind, definition_id, name, match_algorithm, match_pattern, case_insensitive, retired, created_at_unix_ms |
//! | [`ASSIGNMENTS_TABLE`] | `(document, kind, definition_id)` | content_sha256_hex, kind, definition_id, source, assigned_at_unix_ms |
//! | [`REVIEWS_TABLE`] | reviewed document | content_sha256_hex, reviewed_at_unix_ms |
//! | [`CUSTOMERS_TABLE`] | document with a customer | content_sha256_hex, customer_code, assigned_at_unix_ms |
//!
//! # The customer is an access fact, not a filing definition
//!
//! A document's customer decides who may see it (the web crate's customer
//! policy), so it is not a [`MetaKind`]: a definition can be renamed, retired
//! and suggested by AUTO, and none of that may move a document between
//! customers. The code is the report-rs customer code (`K01`), validated by
//! [`valid_customer_code`], at most one per document. Only an explicit action
//! writes it: nothing here infers a customer from a document's content.
//!
//! # Why they are NOT columns of `documents`
//!
//! `crate::store::LanceStore::put` is a whole-row `merge_insert`: a second
//! `put` of the same hash rewrites every column of the document row. Metadata
//! stored there would be overwritten by any direct re-`put`, a double upload
//! or a future writer. Keeping it in its own tables makes that a non-event
//! (spec R2, gate G1). The exposure is at the store level, not on the web
//! path, which dedups before `put`.
//!
//! # Ids are filing ids, never classids
//!
//! `definition_id` is a consumer-local filing id, `max + 1` per kind and
//! never reused, even after a retire. It is NOT an OGAR classid and is never
//! passed to `NodeGuid`, `FacetCascade` or `mint_for` (spec F10).
//!
//! # Rules are plain columns
//!
//! `match_algorithm` (paperless numbering), `match_pattern` and
//! `case_insensitive` are stored as plain columns. This module knows exactly
//! one algorithm number, [`MATCH_AUTO`]. The conversion to `MatchRule` lives
//! in glue under `all(store, matching)`, because `store` must not imply
//! `matching` (spec R1). New definitions default to AUTO with
//! `case_insensitive = true`, as the paperless-ngx UI does; the model's own
//! default (ANY with an empty pattern) would leave a new definition dead on
//! both tiers.
//!
//! # Delete order contract
//!
//! Callers delete the document row FIRST, then call
//! [`MetaStore::forget_document`]. A crash between the two leaves orphan
//! metadata rows (swept by [`MetaStore::reconcile`]), never a document whose
//! metadata is gone (spec R2a).
//!
//! # Nothing here marks a document reviewed as a side effect
//!
//! [`MetaStore::assign`] and [`MetaStore::unassign`] never touch the reviews
//! table. Only [`MetaStore::mark_reviewed`] writes it, only
//! [`MetaStore::mark_unreviewed`] (or a forget/reconcile) removes from it
//! (spec R3a, gate G5).
//!
//! # Concurrency
//!
//! Every write goes through one async mutex inside [`MetaStore`], so the
//! read-then-write operations are serialized by the store itself, not by an
//! assumption about its callers. A web server serves requests concurrently:
//! without the lock, two `create_definition` calls for one kind can read the
//! same `max`, mint the same `definition_id`, and the second `merge_insert`
//! silently overwrites the first definition (and a single-valued `assign`
//! racing another can leave two rows). Reads take no lock. The lock is
//! per process: two processes on one archive are still not serialized.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
    UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};

use crate::store::StoreError;

/// Definitions table name.
pub const TAXONOMY_TABLE: &str = "taxonomy";
/// Assignments table name.
pub const ASSIGNMENTS_TABLE: &str = "assignments";
/// Reviews table name.
pub const REVIEWS_TABLE: &str = "reviews";
/// Customers table name: one row per document that has a customer.
pub const CUSTOMERS_TABLE: &str = "customers";

/// Longest customer code accepted.
pub const MAX_CUSTOMER_CODE: usize = 32;

/// `true` for a customer code: 1 to [`MAX_CUSTOMER_CODE`] ASCII letters,
/// digits, `-` or `_`. Anything else is refused rather than normalised, so a
/// code is the same string in the policy, the table and report-rs.
#[must_use]
pub fn valid_customer_code(code: &str) -> bool {
    !code.is_empty()
        && code.len() <= MAX_CUSTOMER_CODE
        && code
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// paperless-ngx's `MATCH_AUTO` number. The ONLY algorithm number store code
/// knows; converting a stored number to a `MatchRule` is glue under
/// `all(store, matching)`.
pub const MATCH_AUTO: u8 = 6;

/// Column names, named once so a rename cannot desync a schema from a builder.
mod col {
    pub const KIND: &str = "kind";
    pub const DEFINITION_ID: &str = "definition_id";
    pub const NAME: &str = "name";
    pub const MATCH_ALGORITHM: &str = "match_algorithm";
    pub const MATCH_PATTERN: &str = "match_pattern";
    pub const CASE_INSENSITIVE: &str = "case_insensitive";
    pub const RETIRED: &str = "retired";
    pub const CREATED_AT: &str = "created_at_unix_ms";
    pub const HASH: &str = "content_sha256_hex";
    pub const SOURCE: &str = "source";
    pub const ASSIGNED_AT: &str = "assigned_at_unix_ms";
    pub const REVIEWED_AT: &str = "reviewed_at_unix_ms";
    pub const CUSTOMER: &str = "customer_code";
}

/// What a definition files documents under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetaKind {
    /// Who the document is from. Single-valued.
    Correspondent,
    /// What sort of document it is. Single-valued.
    DocumentType,
    /// A label; a document has a set of them.
    Tag,
}

impl MetaKind {
    /// The stored spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correspondent => "correspondent",
            Self::DocumentType => "document_type",
            Self::Tag => "tag",
        }
    }

    /// Inverse of [`Self::as_str`]; `None` for anything else.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "correspondent" => Some(Self::Correspondent),
            "document_type" => Some(Self::DocumentType),
            "tag" => Some(Self::Tag),
            _ => None,
        }
    }

    /// A document has at most one correspondent and one document type; tags
    /// are a set.
    #[must_use]
    pub fn is_single_valued(self) -> bool {
        !matches!(self, Self::Tag)
    }
}

/// Provenance of an assignment. Shown in the UI; has no effect on mining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// A person set it.
    Manual,
    /// A non-AUTO matching rule set it at ingest.
    Rule,
    /// A person accepted an AUTO suggestion.
    AutoAccepted,
}

impl Source {
    /// The stored spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Rule => "rule",
            Self::AutoAccepted => "auto_accepted",
        }
    }

    /// Inverse of [`Self::as_str`]; `None` for anything else.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "manual" => Some(Self::Manual),
            "rule" => Some(Self::Rule),
            "auto_accepted" => Some(Self::AutoAccepted),
            _ => None,
        }
    }
}

/// One definition, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinitionRow {
    /// Which kind of filing entity this is.
    pub kind: MetaKind,
    /// Consumer-local filing id, `max + 1` per kind, never reused.
    pub definition_id: u32,
    /// Display name, unique per kind (retired ones included).
    pub name: String,
    /// paperless-ngx numbering. See [`MATCH_AUTO`].
    pub match_algorithm: u8,
    /// The rule's pattern, possibly empty.
    pub match_pattern: String,
    /// Whether matching ignores case.
    pub case_insensitive: bool,
    /// Retired definitions keep their row (and id) but take no new assignments.
    pub retired: bool,
    /// Creation time, milliseconds since the Unix epoch.
    pub created_at_unix_ms: i64,
}

/// One assignment, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentRow {
    /// Hex `content_sha256` of the document.
    pub content_sha256_hex: String,
    /// Kind of the assigned definition.
    pub kind: MetaKind,
    /// Id of the assigned definition.
    pub definition_id: u32,
    /// How the assignment came about.
    pub source: Source,
    /// Assignment time, milliseconds since the Unix epoch.
    pub assigned_at_unix_ms: i64,
}

/// One review flag, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewRow {
    /// Hex `content_sha256` of the document.
    pub content_sha256_hex: String,
    /// When it was marked reviewed, milliseconds since the Unix epoch.
    pub reviewed_at_unix_ms: i64,
}

/// Every metadata row at one moment: what the S3 copy keeps and a restore
/// writes back ([`MetaStore::export`], [`MetaStore::import`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetaSnapshot {
    /// Every definition, retired ones included, with its id.
    pub definitions: Vec<DefinitionRow>,
    /// Every assignment.
    pub assignments: Vec<AssignmentRow>,
    /// Every review flag.
    pub reviews: Vec<ReviewRow>,
    /// Every document's customer, as `(hash, customer code)`, sorted by hash.
    /// A snapshot taken before customers existed has none.
    pub customers: Vec<(String, String)>,
}

/// What [`MetaStore::reconcile`] removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaReconcileReport {
    /// Assignment rows removed (rows, not distinct documents).
    pub assignments_removed: usize,
    /// Review rows removed.
    pub reviews_removed: usize,
    /// Customer rows removed.
    pub customers_removed: usize,
}

/// Why a metadata operation failed.
#[derive(Debug)]
pub enum MetaError {
    /// The underlying store failed.
    Store(StoreError),
    /// A definition of this kind (retired or not) already has this name.
    NameTaken {
        /// The kind that already holds the name.
        kind: MetaKind,
        /// The contested (trimmed) name.
        name: String,
    },
    /// No such definition, or it is retired and cannot take assignments.
    UnknownDefinition {
        /// The kind looked up.
        kind: MetaKind,
        /// The id looked up.
        definition_id: u32,
    },
    /// A customer code [`valid_customer_code`] refuses.
    InvalidCustomer,
}

impl core::fmt::Display for MetaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "{e}"),
            Self::NameTaken { kind, name } => {
                write!(f, "a {} named {name:?} already exists", kind.as_str())
            }
            Self::UnknownDefinition {
                kind,
                definition_id,
            } => write!(
                f,
                "no usable {} definition with id {definition_id}",
                kind.as_str()
            ),
            Self::InvalidCustomer => write!(
                f,
                "a customer code is 1 to {MAX_CUSTOMER_CODE} letters, digits, '-' or '_'"
            ),
        }
    }
}

impl std::error::Error for MetaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StoreError> for MetaError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

impl From<lancedb::Error> for MetaError {
    fn from(e: lancedb::Error) -> Self {
        Self::Store(StoreError::Db(e))
    }
}

fn taxonomy_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(col::KIND, DataType::Utf8, false),
        Field::new(col::DEFINITION_ID, DataType::UInt32, false),
        Field::new(col::NAME, DataType::Utf8, false),
        Field::new(col::MATCH_ALGORITHM, DataType::UInt8, false),
        Field::new(col::MATCH_PATTERN, DataType::Utf8, false),
        Field::new(col::CASE_INSENSITIVE, DataType::Boolean, false),
        Field::new(col::RETIRED, DataType::Boolean, false),
        Field::new(col::CREATED_AT, DataType::Int64, false),
    ]))
}

fn assignments_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(col::HASH, DataType::Utf8, false),
        Field::new(col::KIND, DataType::Utf8, false),
        Field::new(col::DEFINITION_ID, DataType::UInt32, false),
        Field::new(col::SOURCE, DataType::Utf8, false),
        Field::new(col::ASSIGNED_AT, DataType::Int64, false),
    ]))
}

fn customers_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(col::HASH, DataType::Utf8, false),
        Field::new(col::CUSTOMER, DataType::Utf8, false),
        Field::new(col::ASSIGNED_AT, DataType::Int64, false),
    ]))
}

fn reviews_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(col::HASH, DataType::Utf8, false),
        Field::new(col::REVIEWED_AT, DataType::Int64, false),
    ]))
}

/// Open the table if it exists, else create it empty. Open first, exactly as
/// `LanceStore::connect` does: `create_empty_table` in `exist_ok` mode refuses
/// an existing table whose schema differs.
async fn open_or_create(
    db: &Connection,
    name: &str,
    schema: SchemaRef,
) -> Result<Table, StoreError> {
    match db.open_table(name).execute().await {
        Ok(t) => Ok(t),
        Err(lancedb::Error::TableNotFound { .. }) => Ok(db
            .create_empty_table(name, schema)
            .mode(lancedb::database::CreateTableMode::exist_ok(|req| req))
            .execute()
            .await?),
        Err(e) => Err(e.into()),
    }
}

/// Escape a string for a single-quoted SQL literal, exactly as `store.rs`.
fn sql_quote(s: &str) -> String {
    s.replace('\'', "''")
}

fn typed<'a, T: 'static>(batch: &'a RecordBatch, name: &'static str) -> Result<&'a T, StoreError> {
    batch
        .column_by_name(name)
        .ok_or(StoreError::Malformed(name))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or(StoreError::Malformed(name))
}

async fn collect(table: &Table, only_if: Option<String>) -> Result<Vec<RecordBatch>, StoreError> {
    let stream = match only_if {
        Some(p) => table.query().only_if(p).execute().await?,
        None => table.query().execute().await?,
    };
    Ok(stream.try_collect().await?)
}

/// `merge_insert` one batch on `keys` (update on match, insert otherwise) —
/// the same call shape as `LanceStore::put`.
async fn upsert(
    table: &Table,
    keys: &[&str],
    batch: RecordBatch,
    schema: SchemaRef,
) -> Result<(), StoreError> {
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut merge = table.merge_insert(keys);
    merge
        .when_matched_update_all(None)
        .when_not_matched_insert_all();
    merge.execute(Box::new(reader)).await?;
    Ok(())
}

fn definition_batch(d: &DefinitionRow) -> Result<RecordBatch, StoreError> {
    RecordBatch::try_new(
        taxonomy_schema(),
        vec![
            Arc::new(StringArray::from(vec![d.kind.as_str()])),
            Arc::new(UInt32Array::from(vec![d.definition_id])),
            Arc::new(StringArray::from(vec![d.name.as_str()])),
            Arc::new(UInt8Array::from(vec![d.match_algorithm])),
            Arc::new(StringArray::from(vec![d.match_pattern.as_str()])),
            Arc::new(BooleanArray::from(vec![d.case_insensitive])),
            Arc::new(BooleanArray::from(vec![d.retired])),
            Arc::new(Int64Array::from(vec![d.created_at_unix_ms])),
        ],
    )
    .map_err(|_| StoreError::Malformed("taxonomy batch assembly"))
}

fn definitions_from_batches(batches: &[RecordBatch]) -> Result<Vec<DefinitionRow>, StoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let kind = typed::<StringArray>(batch, col::KIND)?;
        let id = typed::<UInt32Array>(batch, col::DEFINITION_ID)?;
        let name = typed::<StringArray>(batch, col::NAME)?;
        let alg = typed::<UInt8Array>(batch, col::MATCH_ALGORITHM)?;
        let pat = typed::<StringArray>(batch, col::MATCH_PATTERN)?;
        let ci = typed::<BooleanArray>(batch, col::CASE_INSENSITIVE)?;
        let retired = typed::<BooleanArray>(batch, col::RETIRED)?;
        let created = typed::<Int64Array>(batch, col::CREATED_AT)?;
        for i in 0..batch.num_rows() {
            out.push(DefinitionRow {
                kind: MetaKind::parse(kind.value(i)).ok_or(StoreError::Malformed(col::KIND))?,
                definition_id: id.value(i),
                name: name.value(i).to_string(),
                match_algorithm: alg.value(i),
                match_pattern: pat.value(i).to_string(),
                case_insensitive: ci.value(i),
                retired: retired.value(i),
                created_at_unix_ms: created.value(i),
            });
        }
    }
    Ok(out)
}

fn assignments_from_batches(batches: &[RecordBatch]) -> Result<Vec<AssignmentRow>, StoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let hash = typed::<StringArray>(batch, col::HASH)?;
        let kind = typed::<StringArray>(batch, col::KIND)?;
        let id = typed::<UInt32Array>(batch, col::DEFINITION_ID)?;
        let source = typed::<StringArray>(batch, col::SOURCE)?;
        let at = typed::<Int64Array>(batch, col::ASSIGNED_AT)?;
        for i in 0..batch.num_rows() {
            out.push(AssignmentRow {
                content_sha256_hex: hash.value(i).to_string(),
                kind: MetaKind::parse(kind.value(i)).ok_or(StoreError::Malformed(col::KIND))?,
                definition_id: id.value(i),
                source: Source::parse(source.value(i)).ok_or(StoreError::Malformed(col::SOURCE))?,
                assigned_at_unix_ms: at.value(i),
            });
        }
    }
    Ok(out)
}

fn customers_from_batches(batches: &[RecordBatch]) -> Result<Vec<(String, String)>, StoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let hash = typed::<StringArray>(batch, col::HASH)?;
        let code = typed::<StringArray>(batch, col::CUSTOMER)?;
        out.extend(
            (0..batch.num_rows()).map(|i| (hash.value(i).to_string(), code.value(i).to_string())),
        );
    }
    Ok(out)
}

fn reviews_from_batches(batches: &[RecordBatch]) -> Result<Vec<ReviewRow>, StoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let hash = typed::<StringArray>(batch, col::HASH)?;
        let at = typed::<Int64Array>(batch, col::REVIEWED_AT)?;
        out.extend((0..batch.num_rows()).map(|i| ReviewRow {
            content_sha256_hex: hash.value(i).to_string(),
            reviewed_at_unix_ms: at.value(i),
        }));
    }
    Ok(out)
}

fn hashes_from_batches(batches: &[RecordBatch]) -> Result<Vec<String>, StoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let hash = typed::<StringArray>(batch, col::HASH)?;
        out.extend((0..batch.num_rows()).map(|i| hash.value(i).to_string()));
    }
    Ok(out)
}

/// Handles to the three metadata tables.
pub struct MetaStore {
    taxonomy: Table,
    assignments: Table,
    reviews: Table,
    customers: Table,
    /// Serializes every write (see the module doc's Concurrency section).
    writes: tokio::sync::Mutex<()>,
}

impl MetaStore {
    /// Open (or create) each of the three tables separately, so a database
    /// that predates this module gains them and one that has them reuses them.
    ///
    /// # Errors
    /// [`StoreError::Db`] if opening or creating any table fails.
    pub async fn open(db: &Connection) -> Result<Self, StoreError> {
        let taxonomy = open_or_create(db, TAXONOMY_TABLE, taxonomy_schema()).await?;
        let assignments = open_or_create(db, ASSIGNMENTS_TABLE, assignments_schema()).await?;
        let reviews = open_or_create(db, REVIEWS_TABLE, reviews_schema()).await?;
        let customers = open_or_create(db, CUSTOMERS_TABLE, customers_schema()).await?;
        Ok(Self {
            taxonomy,
            assignments,
            reviews,
            customers,
            writes: tokio::sync::Mutex::new(()),
        })
    }

    async fn upsert_definition(&self, d: &DefinitionRow) -> Result<(), StoreError> {
        upsert(
            &self.taxonomy,
            &[col::KIND, col::DEFINITION_ID],
            definition_batch(d)?,
            taxonomy_schema(),
        )
        .await
    }

    async fn find_definition(
        &self,
        kind: MetaKind,
        definition_id: u32,
    ) -> Result<DefinitionRow, MetaError> {
        self.definitions()
            .await?
            .into_iter()
            .find(|d| d.kind == kind && d.definition_id == definition_id)
            .ok_or(MetaError::UnknownDefinition {
                kind,
                definition_id,
            })
    }

    /// Create a definition. The id is `max(existing ids of this kind,
    /// retired included) + 1`, or 0 if there are none, so an id is never
    /// reused. Defaults: [`MATCH_AUTO`], empty pattern, case-insensitive,
    /// not retired (paperless UI parity, spec R1).
    ///
    /// The name is trimmed and compared case-sensitively against every
    /// definition of the same kind, retired ones included.
    ///
    /// # Errors
    /// [`MetaError::NameTaken`] on a duplicate name; [`MetaError::Store`] on
    /// a store failure.
    pub async fn create_definition(
        &self,
        kind: MetaKind,
        name: &str,
        now_ms: i64,
    ) -> Result<DefinitionRow, MetaError> {
        let _write = self.writes.lock().await;
        let name = name.trim();
        let existing = self.definitions().await?;
        let of_kind = || existing.iter().filter(|d| d.kind == kind);
        if of_kind().any(|d| d.name == name) {
            return Err(MetaError::NameTaken {
                kind,
                name: name.to_string(),
            });
        }
        let definition_id = match of_kind().map(|d| d.definition_id).max() {
            None => 0,
            Some(m) => m
                .checked_add(1)
                .ok_or(StoreError::Malformed("definition_id overflow"))?,
        };
        let row = DefinitionRow {
            kind,
            definition_id,
            name: name.to_string(),
            match_algorithm: MATCH_AUTO,
            match_pattern: String::new(),
            case_insensitive: true,
            retired: false,
            created_at_unix_ms: now_ms,
        };
        self.upsert_definition(&row).await?;
        Ok(row)
    }

    /// Rename a definition (trimmed, case-sensitive uniqueness per kind).
    /// Renaming to its own current name is a no-op success.
    ///
    /// # Errors
    /// [`MetaError::UnknownDefinition`] if absent; [`MetaError::NameTaken`]
    /// if another definition of the kind has the name.
    pub async fn rename_definition(
        &self,
        kind: MetaKind,
        definition_id: u32,
        name: &str,
    ) -> Result<(), MetaError> {
        let _write = self.writes.lock().await;
        let name = name.trim();
        let all = self.definitions().await?;
        let mut row = all
            .iter()
            .find(|d| d.kind == kind && d.definition_id == definition_id)
            .cloned()
            .ok_or(MetaError::UnknownDefinition {
                kind,
                definition_id,
            })?;
        if all
            .iter()
            .any(|d| d.kind == kind && d.definition_id != definition_id && d.name == name)
        {
            return Err(MetaError::NameTaken {
                kind,
                name: name.to_string(),
            });
        }
        row.name = name.to_string();
        self.upsert_definition(&row).await?;
        Ok(())
    }

    /// Replace a definition's rule columns.
    ///
    /// # Errors
    /// [`MetaError::UnknownDefinition`] if absent.
    pub async fn set_rule(
        &self,
        kind: MetaKind,
        definition_id: u32,
        match_algorithm: u8,
        match_pattern: &str,
        case_insensitive: bool,
    ) -> Result<(), MetaError> {
        let _write = self.writes.lock().await;
        let mut row = self.find_definition(kind, definition_id).await?;
        row.match_algorithm = match_algorithm;
        row.match_pattern = match_pattern.to_string();
        row.case_insensitive = case_insensitive;
        self.upsert_definition(&row).await?;
        Ok(())
    }

    /// Mark a definition retired. The row (and id) is kept.
    ///
    /// # Errors
    /// [`MetaError::UnknownDefinition`] if absent.
    pub async fn retire_definition(
        &self,
        kind: MetaKind,
        definition_id: u32,
    ) -> Result<(), MetaError> {
        let _write = self.writes.lock().await;
        let mut row = self.find_definition(kind, definition_id).await?;
        row.retired = true;
        self.upsert_definition(&row).await?;
        Ok(())
    }

    /// Every definition of every kind, retired included, sorted by
    /// `(kind, name)`. Name order is load-bearing: S-8 takes the first match
    /// by name (spec F7). Names sort by byte order (case-sensitive).
    ///
    /// # Errors
    /// [`StoreError::Db`] on a read failure; [`StoreError::Malformed`] on an
    /// undecodable row.
    pub async fn definitions(&self) -> Result<Vec<DefinitionRow>, StoreError> {
        let batches = collect(&self.taxonomy, None).await?;
        let mut rows = definitions_from_batches(&batches)?;
        rows.sort_by(|a, b| {
            a.kind
                .as_str()
                .cmp(b.kind.as_str())
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.definition_id.cmp(&b.definition_id))
        });
        Ok(rows)
    }

    /// Assign a definition to a document.
    ///
    /// For a single-valued kind every existing `(hash, kind)` row is deleted
    /// FIRST, then the new one inserted, so a document never holds two
    /// correspondents (G12). For a tag the write is a `merge_insert` on
    /// `(hash, kind, definition_id)`, so re-assigning is idempotent.
    ///
    /// **This never touches the reviews table.** Assigning does not mark a
    /// document reviewed (spec R3a, G5).
    ///
    /// # Errors
    /// [`MetaError::UnknownDefinition`] if the definition is absent or retired.
    pub async fn assign(
        &self,
        hash: &str,
        kind: MetaKind,
        definition_id: u32,
        source: Source,
        now_ms: i64,
    ) -> Result<(), MetaError> {
        let _write = self.writes.lock().await;
        let def = self.find_definition(kind, definition_id).await?;
        if def.retired {
            return Err(MetaError::UnknownDefinition {
                kind,
                definition_id,
            });
        }
        if kind.is_single_valued() {
            let predicate = format!(
                "{} = '{}' AND {} = '{}'",
                col::HASH,
                sql_quote(hash),
                col::KIND,
                kind.as_str()
            );
            self.assignments.delete(&predicate).await?;
        }
        let batch = RecordBatch::try_new(
            assignments_schema(),
            vec![
                Arc::new(StringArray::from(vec![hash])),
                Arc::new(StringArray::from(vec![kind.as_str()])),
                Arc::new(UInt32Array::from(vec![definition_id])),
                Arc::new(StringArray::from(vec![source.as_str()])),
                Arc::new(Int64Array::from(vec![now_ms])),
            ],
        )
        .map_err(|_| StoreError::Malformed("assignments batch assembly"))?;
        upsert(
            &self.assignments,
            &[col::HASH, col::KIND, col::DEFINITION_ID],
            batch,
            assignments_schema(),
        )
        .await?;
        Ok(())
    }

    /// Remove one assignment. Absent is not an error. Never touches reviews.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a delete failure.
    pub async fn unassign(
        &self,
        hash: &str,
        kind: MetaKind,
        definition_id: u32,
    ) -> Result<(), StoreError> {
        let _write = self.writes.lock().await;
        let predicate = format!(
            "{} = '{}' AND {} = '{}' AND {} = {}",
            col::HASH,
            sql_quote(hash),
            col::KIND,
            kind.as_str(),
            col::DEFINITION_ID,
            definition_id
        );
        self.assignments.delete(&predicate).await?;
        Ok(())
    }

    /// Every assignment of one document.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn assignments_for(&self, hash: &str) -> Result<Vec<AssignmentRow>, StoreError> {
        let predicate = format!("{} = '{}'", col::HASH, sql_quote(hash));
        let batches = collect(&self.assignments, Some(predicate)).await?;
        assignments_from_batches(&batches)
    }

    /// Every assignment of every document.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn all_assignments(&self) -> Result<Vec<AssignmentRow>, StoreError> {
        let batches = collect(&self.assignments, None).await?;
        assignments_from_batches(&batches)
    }

    /// Mark a document reviewed. Idempotent (`merge_insert` on the hash; a
    /// repeat refreshes the timestamp). Only an explicit action calls this.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a write failure.
    pub async fn mark_reviewed(&self, hash: &str, now_ms: i64) -> Result<(), StoreError> {
        let _write = self.writes.lock().await;
        let batch = RecordBatch::try_new(
            reviews_schema(),
            vec![
                Arc::new(StringArray::from(vec![hash])),
                Arc::new(Int64Array::from(vec![now_ms])),
            ],
        )
        .map_err(|_| StoreError::Malformed("reviews batch assembly"))?;
        upsert(&self.reviews, &[col::HASH], batch, reviews_schema()).await
    }

    /// Clear a document's reviewed flag. Absent is not an error.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a delete failure.
    pub async fn mark_unreviewed(&self, hash: &str) -> Result<(), StoreError> {
        let _write = self.writes.lock().await;
        let predicate = format!("{} = '{}'", col::HASH, sql_quote(hash));
        self.reviews.delete(&predicate).await?;
        Ok(())
    }

    /// Whether a document is marked reviewed.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn is_reviewed(&self, hash: &str) -> Result<bool, StoreError> {
        let predicate = format!("{} = '{}'", col::HASH, sql_quote(hash));
        let stream = self
            .reviews
            .query()
            .only_if(predicate)
            .limit(1)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        Ok(batches.iter().any(|b| b.num_rows() > 0))
    }

    /// Hex hashes of every reviewed document.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn reviewed_hashes(&self) -> Result<Vec<String>, StoreError> {
        let stream = self
            .reviews
            .query()
            .select(lancedb::query::Select::columns(&[col::HASH]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        hashes_from_batches(&batches)
    }

    /// Set (`Some`) or clear (`None`) a document's customer. Setting replaces
    /// any previous customer (`merge_insert` on the hash).
    ///
    /// # Errors
    /// [`MetaError::InvalidCustomer`] for a code [`valid_customer_code`]
    /// refuses (nothing is written); [`MetaError::Store`] on a store failure.
    pub async fn set_customer(
        &self,
        hash: &str,
        code: Option<&str>,
        now_ms: i64,
    ) -> Result<(), MetaError> {
        if let Some(c) = code {
            if !valid_customer_code(c) {
                return Err(MetaError::InvalidCustomer);
            }
        }
        let _write = self.writes.lock().await;
        let Some(code) = code else {
            let predicate = format!("{} = '{}'", col::HASH, sql_quote(hash));
            self.customers.delete(&predicate).await?;
            return Ok(());
        };
        let batch = RecordBatch::try_new(
            customers_schema(),
            vec![
                Arc::new(StringArray::from(vec![hash])),
                Arc::new(StringArray::from(vec![code])),
                Arc::new(Int64Array::from(vec![now_ms])),
            ],
        )
        .map_err(|_| StoreError::Malformed("customers batch assembly"))?;
        upsert(&self.customers, &[col::HASH], batch, customers_schema()).await?;
        Ok(())
    }

    /// A document's customer, if it has one.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn customer_of(&self, hash: &str) -> Result<Option<String>, StoreError> {
        let predicate = format!("{} = '{}'", col::HASH, sql_quote(hash));
        let batches = collect(&self.customers, Some(predicate)).await?;
        Ok(customers_from_batches(&batches)?
            .into_iter()
            .next()
            .map(|(_, code)| code))
    }

    /// Every document's customer, by hash. Documents without one are absent.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn customers(&self) -> Result<std::collections::HashMap<String, String>, StoreError> {
        let batches = collect(&self.customers, None).await?;
        Ok(customers_from_batches(&batches)?.into_iter().collect())
    }

    /// Delete a document's assignments AND its review row. Called by
    /// `LanceStore::delete` AFTER the document row is gone (spec R2a), so the
    /// worst case is an orphan, never a document with its metadata missing.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a delete failure.
    pub async fn forget_document(&self, hash: &str) -> Result<(), StoreError> {
        let _write = self.writes.lock().await;
        let predicate = format!("{} = '{}'", col::HASH, sql_quote(hash));
        self.assignments.delete(&predicate).await?;
        self.reviews.delete(&predicate).await?;
        self.customers.delete(&predicate).await?;
        Ok(())
    }

    /// Remove assignment and review rows whose hash is not in `live_hashes`
    /// (the archived documents). Runs at startup so a re-upload of a deleted
    /// hash never inherits a stale review flag.
    ///
    /// # Errors
    /// As [`Self::definitions`] and [`Self::forget_document`].
    pub async fn reconcile(
        &self,
        live_hashes: &HashSet<String>,
    ) -> Result<MetaReconcileReport, StoreError> {
        let _write = self.writes.lock().await;
        let assignment_rows = self.all_assignments().await?;
        let dead_assigned: HashSet<&str> = assignment_rows
            .iter()
            .map(|r| r.content_sha256_hex.as_str())
            .filter(|h| !live_hashes.contains(*h))
            .collect();
        let assignments_removed = assignment_rows
            .iter()
            .filter(|r| dead_assigned.contains(r.content_sha256_hex.as_str()))
            .count();

        let reviewed = self.reviewed_hashes().await?;
        let dead_reviewed: HashSet<&str> = reviewed
            .iter()
            .map(String::as_str)
            .filter(|h| !live_hashes.contains(*h))
            .collect();
        let reviews_removed = reviewed
            .iter()
            .filter(|h| dead_reviewed.contains(h.as_str()))
            .count();

        let customer_rows = self.customers().await?;
        let dead_customers: HashSet<&str> = customer_rows
            .keys()
            .map(String::as_str)
            .filter(|h| !live_hashes.contains(*h))
            .collect();
        let customers_removed = dead_customers.len();

        delete_hashes(&self.assignments, &dead_assigned).await?;
        delete_hashes(&self.reviews, &dead_reviewed).await?;
        delete_hashes(&self.customers, &dead_customers).await?;
        Ok(MetaReconcileReport {
            assignments_removed,
            reviews_removed,
            customers_removed,
        })
    }
}

impl MetaStore {
    /// Every metadata row, read under the write lock so the three tables
    /// describe one moment.
    ///
    /// # Errors
    /// As [`Self::definitions`].
    pub async fn export(&self) -> Result<MetaSnapshot, StoreError> {
        let _write = self.writes.lock().await;
        let definitions = self.definitions().await?;
        let assignments = self.all_assignments().await?;
        let reviews = reviews_from_batches(&collect(&self.reviews, None).await?)?;
        let mut customers = customers_from_batches(&collect(&self.customers, None).await?)?;
        customers.sort();
        Ok(MetaSnapshot {
            definitions,
            assignments,
            reviews,
            customers,
        })
    }

    /// Write a snapshot back, keeping every definition id. Assignments and
    /// reviews whose document is not in `live_hashes` are dropped, as
    /// [`Self::reconcile`] would drop them. Upserts, so importing the same
    /// snapshot twice changes nothing.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a write failure.
    pub async fn import(
        &self,
        snapshot: &MetaSnapshot,
        live_hashes: &HashSet<String>,
    ) -> Result<(), StoreError> {
        let _write = self.writes.lock().await;
        for d in &snapshot.definitions {
            self.upsert_definition(d).await?;
        }
        for a in snapshot
            .assignments
            .iter()
            .filter(|a| live_hashes.contains(&a.content_sha256_hex))
        {
            let batch = RecordBatch::try_new(
                assignments_schema(),
                vec![
                    Arc::new(StringArray::from(vec![a.content_sha256_hex.as_str()])),
                    Arc::new(StringArray::from(vec![a.kind.as_str()])),
                    Arc::new(UInt32Array::from(vec![a.definition_id])),
                    Arc::new(StringArray::from(vec![a.source.as_str()])),
                    Arc::new(Int64Array::from(vec![a.assigned_at_unix_ms])),
                ],
            )
            .map_err(|_| StoreError::Malformed("assignments batch assembly"))?;
            upsert(
                &self.assignments,
                &[col::HASH, col::KIND, col::DEFINITION_ID],
                batch,
                assignments_schema(),
            )
            .await?;
        }
        for r in snapshot
            .reviews
            .iter()
            .filter(|r| live_hashes.contains(&r.content_sha256_hex))
        {
            let batch = RecordBatch::try_new(
                reviews_schema(),
                vec![
                    Arc::new(StringArray::from(vec![r.content_sha256_hex.as_str()])),
                    Arc::new(Int64Array::from(vec![r.reviewed_at_unix_ms])),
                ],
            )
            .map_err(|_| StoreError::Malformed("reviews batch assembly"))?;
            upsert(&self.reviews, &[col::HASH], batch, reviews_schema()).await?;
        }
        // A customer decides who sees a document, so a code that does not
        // validate is never written back; it is skipped, as a row whose
        // document is gone is.
        for (hash, code) in snapshot
            .customers
            .iter()
            .filter(|(h, c)| live_hashes.contains(h) && valid_customer_code(c))
        {
            let batch = RecordBatch::try_new(
                customers_schema(),
                vec![
                    Arc::new(StringArray::from(vec![hash.as_str()])),
                    Arc::new(StringArray::from(vec![code.as_str()])),
                    Arc::new(Int64Array::from(vec![0_i64])),
                ],
            )
            .map_err(|_| StoreError::Malformed("customers batch assembly"))?;
            upsert(&self.customers, &[col::HASH], batch, customers_schema()).await?;
        }
        Ok(())
    }
}

/// Delete every row whose hash is in `dead`, in bounded `IN (...)` chunks.
async fn delete_hashes(table: &Table, dead: &HashSet<&str>) -> Result<(), StoreError> {
    let dead: Vec<&str> = dead.iter().copied().collect();
    for chunk in dead.chunks(200) {
        let list = chunk
            .iter()
            .map(|h| format!("'{}'", sql_quote(h)))
            .collect::<Vec<_>>()
            .join(", ");
        let predicate = format!("{} IN ({list})", col::HASH);
        table.delete(&predicate).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_uri() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let uri = dir.path().to_string_lossy().to_string();
        (dir, uri)
    }

    async fn fresh() -> (tempfile::TempDir, MetaStore) {
        let (dir, uri) = tmp_uri();
        let db = lancedb::connect(&uri).execute().await.expect("connect");
        let store = MetaStore::open(&db).await.expect("open");
        (dir, store)
    }

    fn set(hashes: &[&str]) -> HashSet<String> {
        hashes.iter().map(|h| (*h).to_string()).collect()
    }

    /// Slice C: a snapshot written into an empty store reproduces every
    /// definition (ids, rules, retired flag), every assignment and every review
    /// of a live document, and drops rows of documents that are not live.
    #[tokio::test(flavor = "multi_thread")]
    async fn export_then_import_reproduces_the_metadata() {
        let (_a, src) = fresh().await;
        let tag = src
            .create_definition(MetaKind::Tag, "Strom", 10)
            .await
            .expect("tag");
        let corr = src
            .create_definition(MetaKind::Correspondent, "Stadtwerke", 11)
            .await
            .expect("corr");
        let old = src
            .create_definition(MetaKind::Tag, "Alt", 12)
            .await
            .expect("old");
        src.set_rule(MetaKind::Tag, tag.definition_id, 3, "strom", false)
            .await
            .expect("rule");
        src.retire_definition(MetaKind::Tag, old.definition_id)
            .await
            .expect("retire");
        src.assign("h1", MetaKind::Tag, tag.definition_id, Source::Manual, 20)
            .await
            .expect("assign");
        src.assign(
            "h1",
            MetaKind::Correspondent,
            corr.definition_id,
            Source::Rule,
            21,
        )
        .await
        .expect("assign");
        src.assign("gone", MetaKind::Tag, tag.definition_id, Source::Manual, 22)
            .await
            .expect("assign");
        src.mark_reviewed("h1", 30).await.expect("review");
        src.mark_reviewed("gone", 31).await.expect("review");

        let snap = src.export().await.expect("export");
        assert_eq!(snap.definitions.len(), 3);
        assert_eq!(snap.assignments.len(), 3);
        assert_eq!(snap.reviews.len(), 2);

        let (_b, dst) = fresh().await;
        dst.import(&snap, &set(&["h1"])).await.expect("import");
        // Twice: idempotent.
        dst.import(&snap, &set(&["h1"]))
            .await
            .expect("import again");

        assert_eq!(dst.definitions().await.expect("defs"), snap.definitions);
        let mut assigns = dst.all_assignments().await.expect("assigns");
        assigns.sort_by_key(|a| a.assigned_at_unix_ms);
        let mut want: Vec<_> = snap
            .assignments
            .iter()
            .filter(|a| a.content_sha256_hex == "h1")
            .cloned()
            .collect();
        want.sort_by_key(|a| a.assigned_at_unix_ms);
        assert_eq!(assigns, want);
        let exported = dst.export().await.expect("export");
        assert_eq!(
            exported.reviews,
            vec![ReviewRow {
                content_sha256_hex: "h1".to_string(),
                reviewed_at_unix_ms: 30
            }]
        );
        // The next id continues after the restored ones.
        let next = dst
            .create_definition(MetaKind::Tag, "Neu", 40)
            .await
            .expect("next");
        assert!(next.definition_id > old.definition_id);
    }

    /// G14 support. Disable: default `match_algorithm` to ANY (1) or
    /// `case_insensitive` to false in `create_definition`.
    #[tokio::test(flavor = "multi_thread")]
    async fn create_defaults_to_auto_and_case_insensitive() {
        let (_d, s) = fresh().await;
        let row = s
            .create_definition(MetaKind::Correspondent, "  ACME  ", 5)
            .await
            .expect("create");
        assert_eq!(row.match_algorithm, MATCH_AUTO);
        assert!(row.case_insensitive);
        assert!(!row.retired);
        assert_eq!(row.match_pattern, "");
        assert_eq!(row.name, "ACME", "the name is stored trimmed");
        let stored = s.definitions().await.expect("definitions");
        assert_eq!(stored, vec![row], "what was returned is what was stored");
    }

    /// Ids are never reused. Disable: allocate `count` instead of `max + 1`
    /// (c would get id 1, colliding with retired b).
    #[tokio::test(flavor = "multi_thread")]
    async fn ids_are_never_reused_after_a_retire() {
        let (_d, s) = fresh().await;
        let a = s.create_definition(MetaKind::Tag, "a", 1).await.expect("a");
        let b = s.create_definition(MetaKind::Tag, "b", 2).await.expect("b");
        assert_eq!((a.definition_id, b.definition_id), (0, 1));
        s.retire_definition(MetaKind::Tag, 1).await.expect("retire");
        let c = s.create_definition(MetaKind::Tag, "c", 3).await.expect("c");
        assert_eq!(c.definition_id, 2);
        // Retire keeps the row.
        let all = s.definitions().await.expect("definitions");
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|d| d.definition_id == 1 && d.retired));
        // Ids are per kind.
        let corr = s
            .create_definition(MetaKind::Correspondent, "x", 4)
            .await
            .expect("x");
        assert_eq!(corr.definition_id, 0);
    }

    /// Name uniqueness per kind, retired included. Disable: skip the name
    /// check, or filter retired rows out of it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_duplicate_name_is_refused_within_a_kind_including_retired() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Correspondent, "Acme", 1)
            .await
            .expect("first");
        let dup = s
            .create_definition(MetaKind::Correspondent, " Acme ", 2)
            .await;
        assert!(
            matches!(dup, Err(MetaError::NameTaken { kind: MetaKind::Correspondent, ref name }) if name == "Acme"),
            "got {dup:?}"
        );
        // Case-sensitive: a different case is a different name.
        s.create_definition(MetaKind::Correspondent, "acme", 3)
            .await
            .expect("case differs");
        // Retired names stay taken.
        s.retire_definition(MetaKind::Correspondent, 0)
            .await
            .expect("retire");
        let again = s
            .create_definition(MetaKind::Correspondent, "Acme", 4)
            .await;
        assert!(matches!(again, Err(MetaError::NameTaken { .. })));
        // Same name, different kind: fine.
        s.create_definition(MetaKind::Tag, "Acme", 5)
            .await
            .expect("other kind");
        // Rename collides with another id, not with itself.
        let other = s
            .create_definition(MetaKind::Tag, "Other", 6)
            .await
            .expect("o");
        let clash = s
            .rename_definition(MetaKind::Tag, other.definition_id, "Acme")
            .await;
        assert!(matches!(clash, Err(MetaError::NameTaken { .. })));
        s.rename_definition(MetaKind::Tag, other.definition_id, "Other")
            .await
            .expect("renaming to its own name is fine");
        s.rename_definition(MetaKind::Tag, other.definition_id, "Renamed")
            .await
            .expect("rename");
        assert!(s
            .definitions()
            .await
            .expect("definitions")
            .iter()
            .any(|d| d.kind == MetaKind::Tag && d.name == "Renamed"));
    }

    /// G12. Disable: skip the delete-first step for single-valued kinds (two
    /// correspondent rows remain). Tags: replace the `merge_insert` with a
    /// plain append (tag 0 twice leaves two rows).
    #[tokio::test(flavor = "multi_thread")]
    async fn single_valued_kinds_replace_and_tags_accumulate() {
        let (_d, s) = fresh().await;
        for n in ["c0", "c1"] {
            s.create_definition(MetaKind::Correspondent, n, 1)
                .await
                .expect("corr");
        }
        for n in ["t0", "t1"] {
            s.create_definition(MetaKind::Tag, n, 1).await.expect("tag");
        }
        s.assign("h", MetaKind::Correspondent, 0, Source::Manual, 10)
            .await
            .expect("assign c0");
        s.assign("h", MetaKind::Correspondent, 1, Source::AutoAccepted, 11)
            .await
            .expect("assign c1");
        let corr: Vec<_> = s
            .assignments_for("h")
            .await
            .expect("for")
            .into_iter()
            .filter(|r| r.kind == MetaKind::Correspondent)
            .collect();
        assert_eq!(corr.len(), 1, "exactly one correspondent row");
        assert_eq!(corr[0].definition_id, 1);
        assert_eq!(corr[0].source, Source::AutoAccepted);

        s.assign("h", MetaKind::Tag, 0, Source::Manual, 12)
            .await
            .expect("tag 0");
        s.assign("h", MetaKind::Tag, 0, Source::Rule, 13)
            .await
            .expect("tag 0 again");
        let tags = |rows: Vec<AssignmentRow>| {
            rows.into_iter()
                .filter(|r| r.kind == MetaKind::Tag)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            tags(s.assignments_for("h").await.expect("for")).len(),
            1,
            "re-assigning a tag is idempotent"
        );
        s.assign("h", MetaKind::Tag, 1, Source::Manual, 14)
            .await
            .expect("tag 1");
        assert_eq!(
            tags(s.assignments_for("h").await.expect("for")).len(),
            2,
            "tags accumulate"
        );
        // Another document is untouched by all of this.
        s.assign("other", MetaKind::Correspondent, 0, Source::Manual, 15)
            .await
            .expect("other doc");
        assert_eq!(s.all_assignments().await.expect("all").len(), 4);
    }

    /// Disable: drop the retired check in `assign`, or the absent lookup.
    #[tokio::test(flavor = "multi_thread")]
    async fn assigning_an_absent_or_retired_definition_is_refused() {
        let (_d, s) = fresh().await;
        let absent = s.assign("h", MetaKind::Tag, 9, Source::Manual, 1).await;
        assert!(matches!(
            absent,
            Err(MetaError::UnknownDefinition {
                kind: MetaKind::Tag,
                definition_id: 9
            })
        ));
        s.create_definition(MetaKind::Tag, "t", 1).await.expect("t");
        s.retire_definition(MetaKind::Tag, 0).await.expect("retire");
        let retired = s.assign("h", MetaKind::Tag, 0, Source::Manual, 2).await;
        assert!(matches!(retired, Err(MetaError::UnknownDefinition { .. })));
        assert_eq!(s.assignments_for("h").await.expect("for").len(), 0);
        // Wrong kind for a real id is also unknown.
        let wrong_kind = s
            .assign("h", MetaKind::Correspondent, 0, Source::Manual, 3)
            .await;
        assert!(matches!(
            wrong_kind,
            Err(MetaError::UnknownDefinition { .. })
        ));
    }

    /// G5 support. Disable: make `assign` (or `unassign`) call
    /// `mark_reviewed`. Also checks a review survives an unassign.
    #[tokio::test(flavor = "multi_thread")]
    async fn assign_and_unassign_never_touch_the_review_flag() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Correspondent, "c", 1)
            .await
            .expect("c");
        s.assign("h", MetaKind::Correspondent, 0, Source::Manual, 2)
            .await
            .expect("assign");
        assert!(!s.is_reviewed("h").await.expect("is_reviewed"));
        s.unassign("h", MetaKind::Correspondent, 0)
            .await
            .expect("unassign");
        assert!(!s.is_reviewed("h").await.expect("is_reviewed"));
        assert_eq!(s.assignments_for("h").await.expect("for").len(), 0);

        s.mark_reviewed("h", 3).await.expect("mark");
        s.assign("h", MetaKind::Correspondent, 0, Source::Manual, 4)
            .await
            .expect("assign again");
        s.unassign("h", MetaKind::Correspondent, 0)
            .await
            .expect("unassign again");
        assert!(
            s.is_reviewed("h").await.expect("is_reviewed"),
            "an unassign must not clear the review"
        );
    }

    /// Disable: replace the `merge_insert` in `mark_reviewed` with a plain
    /// append (two rows after a repeat), or make `mark_unreviewed` a no-op.
    #[tokio::test(flavor = "multi_thread")]
    async fn mark_reviewed_is_idempotent_and_unreview_clears() {
        let (_d, s) = fresh().await;
        s.mark_reviewed("h", 1).await.expect("first");
        s.mark_reviewed("h", 2).await.expect("second");
        assert_eq!(s.reviewed_hashes().await.expect("hashes"), vec!["h"]);
        assert!(s.is_reviewed("h").await.expect("is"));
        assert!(!s.is_reviewed("nope").await.expect("is"));
        s.mark_unreviewed("h").await.expect("unreview");
        assert!(!s.is_reviewed("h").await.expect("is"));
        assert_eq!(s.reviewed_hashes().await.expect("hashes").len(), 0);
        s.mark_unreviewed("h").await.expect("absent is fine");
    }

    /// Disable: drop the sort in `definitions` (creation order returns
    /// "zeta" first).
    #[tokio::test(flavor = "multi_thread")]
    async fn definitions_are_sorted_by_kind_then_name() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Tag, "zeta", 1)
            .await
            .expect("z");
        s.create_definition(MetaKind::Tag, "alpha", 2)
            .await
            .expect("a");
        s.create_definition(MetaKind::Correspondent, "mid", 3)
            .await
            .expect("m");
        let got: Vec<(MetaKind, String)> = s
            .definitions()
            .await
            .expect("definitions")
            .into_iter()
            .map(|d| (d.kind, d.name))
            .collect();
        assert_eq!(
            got,
            vec![
                (MetaKind::Correspondent, "mid".to_string()),
                (MetaKind::Tag, "alpha".to_string()),
                (MetaKind::Tag, "zeta".to_string()),
            ]
        );
    }

    /// Rule columns round-trip. Disable: make `set_rule` a no-op.
    #[tokio::test(flavor = "multi_thread")]
    async fn set_rule_persists_the_rule_columns() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Tag, "t", 1).await.expect("t");
        s.set_rule(MetaKind::Tag, 0, 3, "invoice", false)
            .await
            .expect("set_rule");
        let d = &s.definitions().await.expect("definitions")[0];
        assert_eq!(
            (
                d.match_algorithm,
                d.match_pattern.as_str(),
                d.case_insensitive
            ),
            (3, "invoice", false)
        );
        let missing = s.set_rule(MetaKind::Tag, 5, 3, "x", true).await;
        assert!(matches!(missing, Err(MetaError::UnknownDefinition { .. })));
    }

    /// Disable: delete only assignments (or only reviews) in
    /// `forget_document`, or drop the hash predicate (the other document
    /// would lose its rows too).
    #[tokio::test(flavor = "multi_thread")]
    async fn forget_document_removes_that_hash_only() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Tag, "t", 1).await.expect("t");
        for h in ["gone", "kept"] {
            s.assign(h, MetaKind::Tag, 0, Source::Manual, 2)
                .await
                .expect("assign");
            s.mark_reviewed(h, 3).await.expect("mark");
        }
        s.forget_document("gone").await.expect("forget");
        assert_eq!(s.assignments_for("gone").await.expect("for").len(), 0);
        assert!(!s.is_reviewed("gone").await.expect("is"));
        assert_eq!(s.assignments_for("kept").await.expect("for").len(), 1);
        assert!(s.is_reviewed("kept").await.expect("is"));
    }

    /// G6 support. Disable: skip the review sweep (a dead hash keeps its
    /// review), or sweep live hashes too. The first half (everything live)
    /// proves the sweep is discriminating, not always-on.
    #[tokio::test(flavor = "multi_thread")]
    async fn reconcile_removes_orphans_only_and_reports_counts() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Tag, "t0", 1)
            .await
            .expect("t0");
        s.create_definition(MetaKind::Tag, "t1", 1)
            .await
            .expect("t1");
        for (h, ids) in [("live", &[0u32, 1][..]), ("dead", &[0, 1][..])] {
            for id in ids {
                s.assign(h, MetaKind::Tag, *id, Source::Manual, 2)
                    .await
                    .expect("assign");
            }
            s.mark_reviewed(h, 3).await.expect("mark");
        }
        // A review with no assignments, and a hash unknown to the archive.
        s.mark_reviewed("dead-review-only", 4).await.expect("mark");

        let nothing = s
            .reconcile(&set(&["live", "dead", "dead-review-only"]))
            .await
            .expect("reconcile all live");
        assert_eq!(
            nothing,
            MetaReconcileReport {
                assignments_removed: 0,
                reviews_removed: 0,
                customers_removed: 0
            }
        );
        assert_eq!(s.all_assignments().await.expect("all").len(), 4);

        let report = s.reconcile(&set(&["live"])).await.expect("reconcile");
        assert_eq!(
            report,
            MetaReconcileReport {
                assignments_removed: 2,
                reviews_removed: 2,
                customers_removed: 0
            }
        );
        assert!(
            !s.is_reviewed("dead").await.expect("is"),
            "G6: stale review gone"
        );
        assert!(!s.is_reviewed("dead-review-only").await.expect("is"));
        assert!(s.is_reviewed("live").await.expect("is"));
        assert_eq!(s.assignments_for("live").await.expect("for").len(), 2);
        assert_eq!(s.assignments_for("dead").await.expect("for").len(), 0);
    }

    /// A quote in a hash must be data, not SQL. Disable: drop `sql_quote`
    /// from one predicate (the query errors or matches everything).
    #[tokio::test(flavor = "multi_thread")]
    async fn a_quote_in_a_hash_does_not_break_or_widen_the_predicate() {
        let (_d, s) = fresh().await;
        s.create_definition(MetaKind::Tag, "t", 1).await.expect("t");
        s.assign("innocent", MetaKind::Tag, 0, Source::Manual, 2)
            .await
            .expect("assign");
        let evil = "' OR '1'='1";
        assert_eq!(s.assignments_for(evil).await.expect("for").len(), 0);
        s.forget_document(evil).await.expect("forget");
        assert_eq!(s.assignments_for("innocent").await.expect("for").len(), 1);
    }

    /// Each table is opened, not recreated. Disable: `create_empty_table`
    /// without the open-first branch, or `CreateTableMode::Overwrite`
    /// (data would be gone after the reopen).
    #[tokio::test(flavor = "multi_thread")]
    async fn data_persists_across_a_reopen() {
        let (_dir, uri) = tmp_uri();
        {
            let db = lancedb::connect(&uri).execute().await.expect("connect");
            let s = MetaStore::open(&db).await.expect("open");
            s.create_definition(MetaKind::Correspondent, "Acme", 1)
                .await
                .expect("create");
            s.assign("h", MetaKind::Correspondent, 0, Source::Rule, 2)
                .await
                .expect("assign");
            s.mark_reviewed("h", 3).await.expect("mark");
        }
        let db = lancedb::connect(&uri).execute().await.expect("connect");
        let s = MetaStore::open(&db).await.expect("reopen");
        assert_eq!(s.definitions().await.expect("definitions").len(), 1);
        let rows = s.assignments_for("h").await.expect("for");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].source, Source::Rule);
        assert!(s.is_reviewed("h").await.expect("is"));
        // The next id continues after the reopened max.
        let next = s
            .create_definition(MetaKind::Correspondent, "Beta", 4)
            .await
            .expect("create");
        assert_eq!(next.definition_id, 1);
    }

    /// Concurrent creates, as a web server issues them, each get their own
    /// id and all survive. Without the write lock two creates read the same
    /// `max`, mint the same id, and the second `merge_insert` overwrites the
    /// first definition. Disable: drop `self.writes.lock()` from
    /// `create_definition`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_creates_never_overwrite_each_other() {
        const N: u32 = 24;
        let (_d, s) = fresh().await;
        let s = Arc::new(s);
        let tasks: Vec<_> = (0..N)
            .map(|i| {
                let s = s.clone();
                tokio::spawn(async move {
                    s.create_definition(MetaKind::Tag, &format!("tag {i}"), 1)
                        .await
                        .expect("create")
                        .definition_id
                })
            })
            .collect();
        let mut ids = Vec::new();
        for t in tasks {
            ids.push(t.await.expect("join"));
        }
        ids.sort_unstable();
        assert_eq!(ids, (0..N).collect::<Vec<_>>(), "ids were minted twice");
        assert_eq!(
            s.definitions().await.expect("definitions").len(),
            N as usize,
            "a definition was overwritten"
        );
    }

    /// Concurrent single-valued assigns to one document leave exactly one
    /// row. Disable: drop `self.writes.lock()` from `assign`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_single_valued_assigns_leave_one_row() {
        const N: u32 = 16;
        let (_d, s) = fresh().await;
        for i in 0..N {
            s.create_definition(MetaKind::Correspondent, &format!("c {i}"), 1)
                .await
                .expect("create");
        }
        let s = Arc::new(s);
        let tasks: Vec<_> = (0..N)
            .map(|i| {
                let s = s.clone();
                tokio::spawn(async move {
                    s.assign("h", MetaKind::Correspondent, i, Source::Manual, 2)
                        .await
                        .expect("assign");
                })
            })
            .collect();
        for t in tasks {
            t.await.expect("join");
        }
        assert_eq!(s.assignments_for("h").await.expect("for").len(), 1);
    }

    #[test]
    fn kind_and_source_spellings_round_trip() {
        for k in [
            MetaKind::Correspondent,
            MetaKind::DocumentType,
            MetaKind::Tag,
        ] {
            assert_eq!(MetaKind::parse(k.as_str()), Some(k));
        }
        for s in [Source::Manual, Source::Rule, Source::AutoAccepted] {
            assert_eq!(Source::parse(s.as_str()), Some(s));
        }
        assert_eq!(MetaKind::parse("Tag"), None);
        assert_eq!(Source::parse(""), None);
        assert!(MetaKind::Correspondent.is_single_valued());
        assert!(MetaKind::DocumentType.is_single_valued());
        assert!(!MetaKind::Tag.is_single_valued());
    }

    /// Disable: drop the `valid_customer_code` check in `set_customer`.
    #[tokio::test(flavor = "multi_thread")]
    async fn customer_codes_are_validated_before_anything_is_written() {
        let (_d, s) = fresh().await;
        for bad in [
            "",
            "K 01",
            "K01'",
            "../K01",
            &"K".repeat(MAX_CUSTOMER_CODE + 1),
        ] {
            assert!(
                matches!(
                    s.set_customer("h", Some(bad), 1).await,
                    Err(MetaError::InvalidCustomer)
                ),
                "{bad:?} must be refused"
            );
        }
        assert!(s.customers().await.expect("read").is_empty());
        for good in ["K01", "R10", "a-b_c", &"K".repeat(MAX_CUSTOMER_CODE)] {
            assert!(valid_customer_code(good), "{good:?} must be accepted");
        }
    }

    /// Disable: make `set_customer(None)` a no-op, or insert instead of
    /// `merge_insert` on the hash.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_document_has_at_most_one_customer() {
        let (_d, s) = fresh().await;
        s.set_customer("h", Some("K01"), 1).await.expect("set");
        s.set_customer("h", Some("K02"), 2).await.expect("replace");
        s.set_customer("other", Some("K01"), 3).await.expect("set");
        assert_eq!(
            s.customer_of("h").await.expect("of").as_deref(),
            Some("K02")
        );
        let all = s.customers().await.expect("all");
        assert_eq!(all.len(), 2, "replacing a customer must not add a row");
        s.set_customer("h", None, 4).await.expect("clear");
        assert_eq!(s.customer_of("h").await.expect("of"), None);
        assert_eq!(
            s.customer_of("other").await.expect("of").as_deref(),
            Some("K01")
        );
    }

    /// A deleted document must not leave its customer behind: a re-upload of
    /// the same file would otherwise inherit it, and with it who may see it.
    /// Disable: drop the customers delete from `forget_document` / `reconcile`.
    #[tokio::test(flavor = "multi_thread")]
    async fn forget_and_reconcile_remove_customer_rows() {
        let (_d, s) = fresh().await;
        for h in ["live", "forgotten", "orphan"] {
            s.set_customer(h, Some("K01"), 1).await.expect("set");
        }
        s.forget_document("forgotten").await.expect("forget");
        assert_eq!(s.customer_of("forgotten").await.expect("of"), None);
        let report = s.reconcile(&set(&["live"])).await.expect("reconcile");
        assert_eq!(report.customers_removed, 1);
        assert_eq!(s.customer_of("orphan").await.expect("of"), None);
        assert_eq!(
            s.customer_of("live").await.expect("of").as_deref(),
            Some("K01")
        );
    }

    /// A restore brings each document's customer back, because it decides who
    /// may see the document; a code that does not validate is not written back.
    /// Disable: drop the customers loop from `import`, or the code check.
    #[tokio::test(flavor = "multi_thread")]
    async fn customers_survive_export_and_import() {
        let (_a, src) = fresh().await;
        src.set_customer("h1", Some("K01"), 1).await.expect("h1");
        src.set_customer("h2", Some("K02"), 1).await.expect("h2");
        let mut snap = src.export().await.expect("export");
        assert_eq!(
            snap.customers,
            vec![("h1".into(), "K01".into()), ("h2".into(), "K02".into())]
        );
        snap.customers.push(("h3".into(), "bad code".into()));
        let (_b, dst) = fresh().await;
        dst.import(&snap, &set(&["h1", "h3"]))
            .await
            .expect("import");
        assert_eq!(
            dst.customer_of("h1").await.expect("h1").as_deref(),
            Some("K01")
        );
        assert_eq!(dst.customer_of("h2").await.expect("h2"), None, "not live");
        assert_eq!(
            dst.customer_of("h3").await.expect("h3"),
            None,
            "invalid code"
        );
    }
}
