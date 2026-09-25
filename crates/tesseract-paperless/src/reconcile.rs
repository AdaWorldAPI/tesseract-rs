//! Bring the search index back in line with the archive.
//!
//! ```text
//!   THE ARCHIVE IS THE AUTHORITY.  THE INDEX IS A LENS OVER IT.
//! ```
//!
//! [`crate::store::LanceStore`] holds the one copy of every document's text
//! (inside its `DocIr`); [`crate::search::SearchIndex`] holds only postings
//! and hashes. Ingest writes the archive first and the index second, with no
//! shared commit, so a crash between the two leaves a document archived but
//! unsearchable, and a delete interrupted the other way round leaves a hit
//! that joins to nothing. Neither is lost content: the index is derived, so
//! the repair re-derives it. [`reconcile`] indexes every archived hash the
//! index lacks and drops every indexed hash the archive lacks. It is also
//! what refills an index [`crate::search::SearchIndex::open_or_create`] had
//! to rebuild after a schema change.

use std::collections::HashSet;

use crate::search::{SearchError, SearchIndex};
use crate::store::{LanceStore, StoreError};

/// What one [`reconcile`] pass changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReconcileReport {
    /// Archived documents that were missing from the index and were indexed.
    pub indexed: usize,
    /// Index entries whose document is no longer archived, now removed.
    pub removed: usize,
    /// Archived documents whose `DocIr` did not parse, so they could not be
    /// indexed. Counted rather than failing the pass.
    pub unreadable: usize,
}

/// Why a [`reconcile`] pass failed.
#[derive(Debug)]
pub enum ReconcileError {
    /// Reading the archive failed.
    Store(StoreError),
    /// Reading or writing the index failed.
    Search(SearchError),
}

impl core::fmt::Display for ReconcileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "archive: {e}"),
            Self::Search(e) => write!(f, "search index: {e}"),
        }
    }
}

impl std::error::Error for ReconcileError {}

impl From<StoreError> for ReconcileError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

impl From<SearchError> for ReconcileError {
    fn from(e: SearchError) -> Self {
        Self::Search(e)
    }
}

/// Make `index` hold exactly the documents `store` holds. Idempotent: a
/// second pass over a consistent pair changes nothing.
///
/// The index calls are synchronous disk I/O; this is meant for process start
/// (or an operator action), not a request path.
///
/// # Errors
/// [`ReconcileError`] if the archive or the index cannot be read or written.
pub async fn reconcile(
    store: &LanceStore,
    index: &SearchIndex,
) -> Result<ReconcileReport, ReconcileError> {
    let archived: HashSet<String> = store.hashes().await?.into_iter().collect();
    let indexed: HashSet<String> = index.indexed_hashes()?.into_iter().collect();
    let mut report = ReconcileReport::default();

    for hash in archived.difference(&indexed) {
        let Some(row) = store.get(hash).await? else {
            continue;
        };
        let Ok(text) = row.text() else {
            report.unreadable += 1;
            continue;
        };
        index.index_document(hash, row.filename.as_deref().unwrap_or_default(), &text)?;
        report.indexed += 1;
    }
    for hash in indexed.difference(&archived) {
        index.delete_document(hash)?;
        report.removed += 1;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::ContentSha256;
    use ogar_doc_ir::DocIr;

    fn ir(text: &str) -> DocIr {
        DocIr {
            version: ogar_doc_ir::DOC_IR_VERSION.to_string(),
            source: ogar_doc_ir::Provenance::Ocr,
            geometry: ogar_doc_ir::Geometry::DomOrder,
            content_sha256: [0u8; 32],
            mime: "image/png".to_string(),
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
                    text: Some(text.to_string()),
                    cells: Vec::new(),
                    children: Vec::new(),
                }],
            }],
            fields: Vec::new(),
        }
    }

    async fn open() -> (tempfile::TempDir, LanceStore, SearchIndex) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = LanceStore::connect(&dir.path().join("archive").to_string_lossy())
            .await
            .expect("store");
        let index = SearchIndex::open_or_create(&dir.path().join("index")).expect("index");
        (dir, store, index)
    }

    /// The crash the design names: the archive write landed, the index write
    /// did not. Reconcile makes the document searchable, with a snippet cut
    /// from the archived text.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_archived_but_unindexed_document_becomes_searchable() {
        let (_dir, store, index) = open().await;
        let hash = ContentSha256::of(b"invoice");
        store
            .put(
                &hash,
                Some("invoice.png"),
                95,
                false,
                &ir("Rechnung fuer Gartenarbeit"),
                1,
                None,
            )
            .await
            .expect("put");
        // ...process dies here, before `index_document`.
        assert!(index
            .search("Gartenarbeit", 10)
            .expect("search")
            .hits
            .is_empty());

        let report = reconcile(&store, &index).await.expect("reconcile");
        assert_eq!(report.indexed, 1);

        let results = index.search("Gartenarbeit", 10).expect("search");
        assert_eq!(results.hits.len(), 1);
        let row = store
            .get(&results.hits[0].hash_hex)
            .await
            .expect("get")
            .expect("the hit joins back to the archive");
        let html = results.snippet_html(&row.text().expect("text"));
        assert!(html.contains("<b>Gartenarbeit</b>"), "got: {html}");
    }

    /// A delete interrupted after the archive row went but before the index
    /// entry did: reconcile removes the dangling hit.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_indexed_but_unarchived_hash_is_removed() {
        let (_dir, store, index) = open().await;
        index
            .index_document("deadbeef", "gone.png", "orphaned wording")
            .expect("index");
        let report = reconcile(&store, &index).await.expect("reconcile");
        assert_eq!(report.removed, 1);
        assert!(index
            .search("orphaned", 10)
            .expect("search")
            .hits
            .is_empty());
    }

    /// A consistent pair is left alone -- the pass does not re-index
    /// everything on every start.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_consistent_pair_changes_nothing() {
        let (_dir, store, index) = open().await;
        let hash = ContentSha256::of(b"kept");
        let doc = ir("already consistent");
        store
            .put(&hash, None, 95, false, &doc, 1, None)
            .await
            .expect("put");
        index
            .index_document(&format!("{hash:?}"), "", "already consistent")
            .expect("index");
        let report = reconcile(&store, &index).await.expect("reconcile");
        assert_eq!(report, ReconcileReport::default());
    }
}
