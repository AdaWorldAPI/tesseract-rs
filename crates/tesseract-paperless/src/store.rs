//! A [`crate::kv::DedupIndex`] + document archive, backed by an embedded
//! `lancedb` table.
//!
//! ```text
//!   S-2 STILL DECIDES.  THIS MODULE IS WHAT S-2 ASKS.
//! ```
//!
//! # Why `lancedb` is not the barrier this crate's own docs warn about
//!
//! `tesseract-rs/CLAUDE.md`'s BBB barrier forbids the `lance-graph` ENGINE —
//! the planner, the thinking substrate, the cognitive columns — from ever
//! entering a customer binary. `lancedb` is a different thing: the embedded
//! columnar database lance-graph itself is built ON, consumed here straight
//! from crates.io per lance-graph's own carve-out ruling
//! (`E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1`, its `CLAUDE.md`: *"lance and
//! lancedb are never used from forks... the upstream is authoritative"*).
//! Nothing in `lancedb`'s dependency tree is the forbidden brain.
//!
//! # What this module does NOT do
//!
//! It does not mint a classid (that stays [`crate::kv::mint_document_root`]'s
//! job, pulling from `ogar-vocab`), does not recognize anything, and does not
//! decide the S-2 verdict — it only ANSWERS the question
//! [`crate::kv::preflight`] asks, by implementing [`crate::kv::DedupIndex`]
//! against a real table instead of a test fixture.
//!
//! # A named gap: only the ORIGINAL hash is matched
//!
//! S-2's second half — matching a re-ingested EXPORT of a held document via
//! its derived-artifact hash — needs a `derived_of` join this table does not
//! yet carry, because nothing in this stack produces a derived artifact yet
//! (no export/rendition pipeline exists to test it against). Filed rather
//! than faked: [`LanceStore::look_up`] answers only [`crate::kv::MatchedOn::Original`].

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, FixedSizeBinaryArray, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt16Array, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::query::{ColumnOrdering, ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table};
use ogar_doc_ir::DocIr;

use crate::kv::{mint_document_root, ContentSha256, DedupIndex, DocumentGuid, MatchedOn};

/// The one table this crate writes.
const TABLE: &str = "documents";

/// Column names, named once so a rename cannot silently desync a `select`
/// projection from a row-builder.
mod col {
    pub const CONTENT_SHA256_HEX: &str = "content_sha256_hex";
    pub const DOCUMENT_GUID: &str = "document_guid";
    pub const FILENAME: &str = "filename";
    pub const MIME: &str = "mime";
    pub const SOURCE: &str = "source";
    pub const PAGE_COUNT: &str = "page_count";
    pub const MEAN_CONFIDENCE: &str = "mean_confidence";
    pub const LOW_CONFIDENCE: &str = "low_confidence";
    pub const TEXT: &str = "text";
    pub const PREVIEW: &str = "preview";
    pub const DOC_IR_JSON: &str = "doc_ir_json";
    pub const INGESTED_AT_UNIX_MS: &str = "ingested_at_unix_ms";
}

/// Why a store operation failed.
#[derive(Debug)]
pub enum StoreError {
    /// The underlying `lancedb` call failed.
    Db(lancedb::Error),
    /// The document's own `doc.v1` -> IR conversion had already been
    /// serialized back to JSON and that serialization failed (should not
    /// happen for a `DocIr` this crate itself built via `ogar_from_docv1`,
    /// kept as a typed error rather than a `.unwrap()`).
    Json(serde_json::Error),
    /// A stored row's Arrow columns did not decode into a well-formed
    /// [`DocumentRow`] — a corrupt or hand-edited table, not a code path
    /// this crate's own writes can produce.
    Malformed(&'static str),
}

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Db(e) => write!(f, "lancedb: {e}"),
            Self::Json(e) => write!(f, "doc_ir_json: {e}"),
            Self::Malformed(e) => write!(f, "malformed row: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<lancedb::Error> for StoreError {
    fn from(e: lancedb::Error) -> Self {
        Self::Db(e)
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(col::CONTENT_SHA256_HEX, DataType::Utf8, false),
        Field::new(col::DOCUMENT_GUID, DataType::FixedSizeBinary(16), false),
        Field::new(col::FILENAME, DataType::Utf8, true),
        Field::new(col::MIME, DataType::Utf8, false),
        Field::new(col::SOURCE, DataType::Utf8, false),
        Field::new(col::PAGE_COUNT, DataType::UInt16, false),
        Field::new(col::MEAN_CONFIDENCE, DataType::UInt32, false),
        Field::new(col::LOW_CONFIDENCE, DataType::Boolean, false),
        Field::new(col::TEXT, DataType::Utf8, false),
        Field::new(col::PREVIEW, DataType::Utf8, false),
        Field::new(col::DOC_IR_JSON, DataType::Utf8, false),
        Field::new(col::INGESTED_AT_UNIX_MS, DataType::Int64, false),
    ]))
}

/// One archived document, as read back from the table — the paperless-ngx-
/// style "document list" row.
#[derive(Debug, Clone)]
pub struct DocumentRow {
    /// Hex `content_sha256` — the convergence key, and this table's primary
    /// key (S-2's own name for it).
    pub content_sha256_hex: String,
    /// The minted root address ([`crate::kv::mint_document_root`]), 16 bytes.
    pub document_guid: [u8; 16],
    /// The uploaded filename, if the caller supplied one.
    pub filename: Option<String>,
    /// The original bytes' MIME type.
    pub mime: String,
    /// `"ocr"` or `"dom"` — [`ogar_doc_ir::Provenance`] as a stored string
    /// (kept a plain column rather than round-tripped through the enum, so a
    /// future `Provenance` variant cannot make an old row fail to decode).
    pub source: String,
    /// Page count.
    pub page_count: u16,
    /// Mean word confidence, 0..=100. `0` with [`Self::low_confidence`] unset
    /// means "no words were recognized", mirroring the `-1` sentinel
    /// `tesseract-ocr-web` uses at the f32 layer — stored as `u32` here
    /// because Arrow's `Boolean` column already carries the low-confidence
    /// flag explicitly, so this column never needs a negative sentinel.
    pub mean_confidence: u32,
    /// Whether the recognizer itself was not confident.
    pub low_confidence: bool,
    /// The full plain text ([`crate::render::plain_text`]) — what search
    /// matches against.
    pub text: String,
    /// A short preview ([`crate::render::preview`]) for the document list.
    pub preview: String,
    /// The `DocIr`, serialized ([`ogar_doc_ir::to_json`]) — the document
    /// detail view's source of truth (regions, fields, bboxes).
    pub doc_ir_json: String,
    /// Milliseconds since the Unix epoch, at ingest time.
    pub ingested_at_unix_ms: i64,
}

impl DocumentRow {
    /// Parse [`Self::doc_ir_json`] back into the typed IR — the detail
    /// view's actual read path; kept as a method rather than inlined at each
    /// call site so "the stored JSON is malformed" has exactly one message.
    ///
    /// # Errors
    /// [`ogar_doc_ir::DocIrError`] if the stored JSON does not parse or has
    /// drifted from the closed `doc.v1` vocabulary — structurally
    /// unreachable for a row this crate wrote itself, kept typed rather than
    /// panicking because "this table is append-only forever, never
    /// hand-edited" is an assumption, not a guarantee.
    pub fn doc_ir(&self) -> Result<DocIr, ogar_doc_ir::DocIrError> {
        ogar_doc_ir::from_json(&self.doc_ir_json)
    }
}

/// An open connection to the document archive.
pub struct LanceStore {
    #[allow(dead_code)] // kept for `open_table`/reconnect; not read elsewhere yet
    db: Connection,
    table: Table,
}

impl LanceStore {
    /// Connect to (or create) the archive at `uri` — a local filesystem
    /// path (Railway: a mounted volume) or any URI `lancedb::connect`
    /// accepts. Idempotent: an existing table is opened, not recreated
    /// (`CreateTableMode::exist_ok`), so this is safe to call on every
    /// process start.
    ///
    /// # Errors
    /// [`StoreError::Db`] if the connection or table creation fails.
    pub async fn connect(uri: &str) -> Result<Self, StoreError> {
        let db = connect(uri).execute().await?;
        let table = db
            .create_empty_table(TABLE, schema())
            .mode(lancedb::database::CreateTableMode::exist_ok(|req| req))
            .execute()
            .await?;
        Ok(Self { db, table })
    }

    /// Persist a novel document — S-5's write-order guard lives at the
    /// CALLER (put the blob / run recognition BEFORE this), not here; this
    /// method's only job is the commit. `merge_insert` on the primary key
    /// makes a second call for the SAME hash an update, not a duplicate row
    /// — idempotent under a caller that retries after a partial failure.
    ///
    /// # Errors
    /// [`StoreError::Json`] if `ir` fails to serialize;
    /// [`StoreError::Db`] if the write fails.
    pub async fn put(
        &self,
        hash: &ContentSha256,
        filename: Option<&str>,
        mean_confidence: u32,
        low_confidence: bool,
        ir: &DocIr,
        ingested_at_unix_ms: i64,
    ) -> Result<DocumentGuid, StoreError> {
        let keys = mint_document_root(hash);
        let doc_ir_json = ogar_doc_ir::to_json(ir).map_err(StoreError::Json)?;
        let text = crate::render::plain_text(ir);
        let preview = crate::render::preview(ir, 240);
        let hex = format!("{hash:?}");

        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from(vec![hex])),
                Arc::new(
                    FixedSizeBinaryArray::try_from_iter(std::iter::once(keys.root.0.to_bytes()))
                        .map_err(|_| StoreError::Malformed("document_guid encode"))?,
                ),
                Arc::new(StringArray::from(vec![filename])),
                Arc::new(StringArray::from(vec![ir.mime.clone()])),
                Arc::new(StringArray::from(vec![provenance_str(ir.source)])),
                Arc::new(UInt16Array::from(vec![
                    u16::try_from(ir.pages.len()).unwrap_or(u16::MAX)
                ])),
                Arc::new(UInt32Array::from(vec![mean_confidence])),
                Arc::new(BooleanArray::from(vec![low_confidence])),
                Arc::new(StringArray::from(vec![text])),
                Arc::new(StringArray::from(vec![preview])),
                Arc::new(StringArray::from(vec![doc_ir_json])),
                Arc::new(Int64Array::from(vec![ingested_at_unix_ms])),
            ],
        )
        .map_err(|_| StoreError::Malformed("record batch assembly"))?;

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema());
        let mut merge = self.table.merge_insert(&[col::CONTENT_SHA256_HEX]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(Box::new(reader)).await?;
        Ok(keys.root)
    }

    /// List documents, newest first — the paperless-ngx "document list"
    /// view. `limit` bounds the page; there is no offset/cursor yet (a named
    /// gap: fine for the archive sizes this is built for so far, wrong past
    /// a few thousand rows).
    ///
    /// # Errors
    /// [`StoreError::Db`] on a read failure; [`StoreError::Malformed`] if a
    /// stored row does not decode (see [`DocumentRow`]'s doc comment).
    pub async fn list(&self, limit: usize) -> Result<Vec<DocumentRow>, StoreError> {
        let stream = self
            .table
            .query()
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last(
                col::INGESTED_AT_UNIX_MS.to_string(),
            )]))
            .limit(limit)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        rows_from_batches(&batches)
    }

    /// Full-text search over [`DocumentRow::text`] — a plain SQL
    /// case-sensitive substring match (`LIKE '%term%'`), not an FTS index.
    /// Named honestly as the smaller of two options: `lancedb` ships a real
    /// inverted-index FTS (`Index::FTS`, `full_text_search`), which needs an
    /// index build step this first cut does not yet perform. `LIKE` is
    /// correct-but-slow (full scan) rather than fast-but-absent — the
    /// gap to close, not a permanent design.
    ///
    /// # Errors
    /// Same as [`Self::list`].
    pub async fn search(&self, term: &str, limit: usize) -> Result<Vec<DocumentRow>, StoreError> {
        // Escape the two characters SQL LIKE treats specially, and single
        // quotes (the string delimiter) — a search box is a direct SQL
        // injection surface otherwise ('; DROP ... is exactly this shape).
        let escaped = term
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
            .replace('\'', "''");
        let predicate = format!("{} LIKE '%{escaped}%' ESCAPE '\\'", col::TEXT);
        let stream = self
            .table
            .query()
            .only_if(predicate)
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last(
                col::INGESTED_AT_UNIX_MS.to_string(),
            )]))
            .limit(limit)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        rows_from_batches(&batches)
    }

    /// Fetch one document by its hex `content_sha256` — the detail view.
    ///
    /// # Errors
    /// Same as [`Self::list`].
    pub async fn get(&self, hash_hex: &str) -> Result<Option<DocumentRow>, StoreError> {
        let safe = hash_hex.replace('\'', "''");
        let predicate = format!("{} = '{safe}'", col::CONTENT_SHA256_HEX);
        let stream = self
            .table
            .query()
            .only_if(predicate)
            .limit(1)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        Ok(rows_from_batches(&batches)?.into_iter().next())
    }

    /// Delete a document by its hex `content_sha256`.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a delete failure.
    pub async fn delete(&self, hash_hex: &str) -> Result<(), StoreError> {
        let safe = hash_hex.replace('\'', "''");
        let predicate = format!("{} = '{safe}'", col::CONTENT_SHA256_HEX);
        self.table.delete(&predicate).await?;
        Ok(())
    }

    /// Total document count — for the list page's header.
    ///
    /// # Errors
    /// [`StoreError::Db`] on a read failure.
    pub async fn count(&self) -> Result<usize, StoreError> {
        Ok(self.table.count_rows(None).await?)
    }
}

fn provenance_str(p: ogar_doc_ir::Provenance) -> &'static str {
    match p {
        ogar_doc_ir::Provenance::Ocr => "ocr",
        ogar_doc_ir::Provenance::Dom => "dom",
    }
}

fn rows_from_batches(batches: &[RecordBatch]) -> Result<Vec<DocumentRow>, StoreError> {
    let mut out = Vec::new();
    for batch in batches {
        let hex = downcast_str(batch, col::CONTENT_SHA256_HEX)?;
        let guid = downcast_fixed_binary(batch, col::DOCUMENT_GUID)?;
        let filename = downcast_str_opt(batch, col::FILENAME);
        let mime = downcast_str(batch, col::MIME)?;
        let source = downcast_str(batch, col::SOURCE)?;
        let page_count = downcast_u16(batch, col::PAGE_COUNT)?;
        let mean_confidence = downcast_u32(batch, col::MEAN_CONFIDENCE)?;
        let low_confidence = downcast_bool(batch, col::LOW_CONFIDENCE)?;
        let text = downcast_str(batch, col::TEXT)?;
        let preview = downcast_str(batch, col::PREVIEW)?;
        let doc_ir_json = downcast_str(batch, col::DOC_IR_JSON)?;
        let ingested_at = downcast_i64(batch, col::INGESTED_AT_UNIX_MS)?;

        for i in 0..batch.num_rows() {
            let mut g = [0u8; 16];
            g.copy_from_slice(guid.value(i));
            out.push(DocumentRow {
                content_sha256_hex: hex.value(i).to_string(),
                document_guid: g,
                filename: filename
                    .as_ref()
                    .and_then(|a| (!a.is_null(i)).then(|| a.value(i).to_string())),
                mime: mime.value(i).to_string(),
                source: source.value(i).to_string(),
                page_count: page_count.value(i),
                mean_confidence: mean_confidence.value(i),
                low_confidence: low_confidence.value(i),
                text: text.value(i).to_string(),
                preview: preview.value(i).to_string(),
                doc_ir_json: doc_ir_json.value(i).to_string(),
                ingested_at_unix_ms: ingested_at.value(i),
            });
        }
    }
    Ok(out)
}

fn column<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Arc<dyn Array>, StoreError> {
    batch
        .column_by_name(name)
        .ok_or(StoreError::Malformed(name))
}

fn downcast_str<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a StringArray, StoreError> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(StoreError::Malformed(name))
}

/// Unlike the other `downcast_*` helpers, a missing/wrong-typed optional
/// column is not an error worth failing the whole row over (`filename` is
/// the one nullable column this table has) — `None` covers both "column
/// absent" and "present but not a `StringArray`", which is exactly the
/// caller's own "fall back to unnamed" handling either way.
fn downcast_str_opt<'a>(batch: &'a RecordBatch, name: &'static str) -> Option<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
}

fn downcast_fixed_binary<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a FixedSizeBinaryArray, StoreError> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or(StoreError::Malformed(name))
}

fn downcast_u16<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a UInt16Array, StoreError> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or(StoreError::Malformed(name))
}

fn downcast_u32<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a UInt32Array, StoreError> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or(StoreError::Malformed(name))
}

fn downcast_bool<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a BooleanArray, StoreError> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or(StoreError::Malformed(name))
}

fn downcast_i64<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Int64Array, StoreError> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or(StoreError::Malformed(name))
}

/// [`DedupIndex`] over the live table — what makes S-2's gate real instead
/// of a trait with no implementation.
///
/// Async under the hood, sync at the trait boundary: [`DedupIndex::look_up`]
/// is a plain function (the trait predates any storage backend and is
/// deliberately I/O-shaped-agnostic), so this blocks on the current Tokio
/// runtime via [`tokio::runtime::Handle::block_on`]. Every call site in this
/// crate's web binary already runs inside an async handler on a
/// multi-threaded runtime, so this never blocks the ONLY worker thread — a
/// single-threaded runtime would deadlock here, which is exactly why
/// `AppState` documents "multi-threaded runtime required".
impl DedupIndex for LanceStore {
    fn look_up(&self, hash: &ContentSha256) -> Option<(DocumentGuid, MatchedOn)> {
        let hex = format!("{hash:?}");
        let handle = tokio::runtime::Handle::current();
        let row = tokio::task::block_in_place(|| handle.block_on(self.get(&hex))).ok()??;
        Some((
            DocumentGuid(lance_graph_contract::facet::FacetCascade::from_bytes(
                &row.document_guid,
            )),
            MatchedOn::Original,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_uri() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let uri = dir.path().to_string_lossy().to_string();
        (dir, uri)
    }

    fn sample_ir(mime: &str) -> DocIr {
        DocIr {
            version: ogar_doc_ir::DOC_IR_VERSION.to_string(),
            source: ogar_doc_ir::Provenance::Ocr,
            geometry: ogar_doc_ir::Geometry::DomOrder,
            content_sha256: [0u8; 32],
            mime: mime.to_string(),
            pages: vec![ogar_doc_ir::DocPage {
                number: 0,
                width: 100,
                height: 100,
                regions: vec![ogar_doc_ir::Region {
                    kind: ogar_doc_ir::RegionKind::Text,
                    bbox: ogar_doc_ir::BBoxRail {
                        tl: ogar_doc_ir::Rail { x: 0, y: 0 },
                        br: ogar_doc_ir::Rail { x: 10, y: 10 },
                    },
                    reading_order: 0,
                    text: Some("Rechnung Nr. 1042 — invoice body text here".to_string()),
                    cells: Vec::new(),
                    children: Vec::new(),
                }],
            }],
            fields: Vec::new(),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_put_document_is_read_back_by_list_and_get() {
        let (_dir, uri) = tmp_uri();
        let store = LanceStore::connect(&uri).await.expect("connect");
        let hash = ContentSha256::of(b"invoice one");
        let ir = sample_ir("image/png");
        store
            .put(&hash, Some("invoice.png"), 92, false, &ir, 1_000)
            .await
            .expect("put");

        let listed = store.list(10).await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].filename.as_deref(), Some("invoice.png"));
        assert_eq!(listed[0].mean_confidence, 92);
        assert!(listed[0].text.contains("Rechnung"));

        let got = store
            .get(&format!("{hash:?}"))
            .await
            .expect("get")
            .expect("row present");
        assert_eq!(got.content_sha256_hex, format!("{hash:?}"));
        // Round-trips through the typed IR too, not just the raw columns.
        let round_tripped = got.doc_ir().expect("doc_ir parses");
        assert_eq!(round_tripped.mime, "image/png");
    }

    /// [`DedupIndex`] must actually gate on the STORED hash — a table with
    /// one document must answer `None` for a DIFFERENT document's hash, or
    /// every upload would look like a duplicate.
    #[tokio::test(flavor = "multi_thread")]
    async fn dedup_index_matches_the_stored_hash_and_only_that_hash() {
        let (_dir, uri) = tmp_uri();
        let store = LanceStore::connect(&uri).await.expect("connect");
        let held = ContentSha256::of(b"already archived");
        store
            .put(&held, None, 80, false, &sample_ir("image/png"), 1)
            .await
            .expect("put");

        assert!(store.look_up(&held).is_some());
        let novel = ContentSha256::of(b"never seen before");
        assert!(
            store.look_up(&novel).is_none(),
            "an unseen hash must not be reported as a duplicate"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn putting_the_same_hash_twice_updates_in_place_not_appends() {
        let (_dir, uri) = tmp_uri();
        let store = LanceStore::connect(&uri).await.expect("connect");
        let hash = ContentSha256::of(b"same document, re-ingested");
        store
            .put(&hash, Some("v1.png"), 50, true, &sample_ir("image/png"), 1)
            .await
            .expect("first put");
        store
            .put(&hash, Some("v2.png"), 99, false, &sample_ir("image/png"), 2)
            .await
            .expect("second put");

        assert_eq!(
            store.count().await.expect("count"),
            1,
            "merge_insert must not duplicate the row"
        );
        let row = store
            .get(&format!("{hash:?}"))
            .await
            .expect("get")
            .expect("present");
        assert_eq!(
            row.filename.as_deref(),
            Some("v2.png"),
            "the second put must win"
        );
        assert_eq!(row.mean_confidence, 99);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_matches_substrings_and_is_silent_on_a_miss() {
        let (_dir, uri) = tmp_uri();
        let store = LanceStore::connect(&uri).await.expect("connect");
        store
            .put(
                &ContentSha256::of(b"a"),
                None,
                90,
                false,
                &sample_ir("image/png"),
                1,
            )
            .await
            .expect("put");

        let hits = store.search("Rechnung", 10).await.expect("search");
        assert_eq!(hits.len(), 1, "the term IS present and must be found");

        let miss = store.search("Nichtvorhanden", 10).await.expect("search");
        assert!(
            miss.is_empty(),
            "a term absent from every document must find nothing"
        );
    }

    /// A search term crafted to break out of the `LIKE` string literal must
    /// be treated as DATA, not as SQL — otherwise the search box is an
    /// injection point into `only_if`'s raw predicate string.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_search_term_containing_a_quote_does_not_break_the_predicate() {
        let (_dir, uri) = tmp_uri();
        let store = LanceStore::connect(&uri).await.expect("connect");
        store
            .put(
                &ContentSha256::of(b"a"),
                None,
                90,
                false,
                &sample_ir("image/png"),
                1,
            )
            .await
            .expect("put");
        // If unescaped, this closes the string literal and appends a clause
        // that would match everything — a query error is an acceptable
        // outcome; silently returning every row is not.
        let result = store.search("' OR '1'='1", 10).await;
        if let Ok(hits) = result {
            assert!(
                hits.is_empty(),
                "an injection-shaped term must not match a document that never contained it"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_removes_the_row() {
        let (_dir, uri) = tmp_uri();
        let store = LanceStore::connect(&uri).await.expect("connect");
        let hash = ContentSha256::of(b"to be deleted");
        store
            .put(&hash, None, 90, false, &sample_ir("image/png"), 1)
            .await
            .expect("put");
        assert_eq!(store.count().await.expect("count"), 1);
        store.delete(&format!("{hash:?}")).await.expect("delete");
        assert_eq!(store.count().await.expect("count"), 0);
        assert!(store
            .get(&format!("{hash:?}"))
            .await
            .expect("get")
            .is_none());
    }
}
