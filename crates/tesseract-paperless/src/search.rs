//! Full-text search over the archive: a real, PERSISTENT Tantivy index --
//! BM25 ranking + snippet/highlight generation.
//!
//! ```text
//!   THIS IS AN ORDINARY TANTIVY INDEX.  IT IS NOT THE RESIDENT-LANE SEAM.
//! ```
//!
//! # Why this is a SEPARATE thing from `token::seam_tantivy`
//!
//! `token::seam_tantivy::ReceiptTokenizer` is a research probe: a `Tokenizer`
//! that reads pre-tokenized BPE ids out of an in-RAM `TokenLane` loaded once
//! from a fixed corpus, proving a shared-tokenization architecture with
//! DeepNSM-v2. It has no add/delete/persist-across-restart story, which is
//! exactly what an archive that grows one document at a time and survives a
//! Railway redeploy needs. Reusing it here would mean building that story
//! from scratch underneath a tool designed for a different job. This module
//! is the OTHER, much more ordinary use of Tantivy -- the one every Tantivy
//! user writes: a plain on-disk index over plain document text, the direct
//! Rust analogue of the Whoosh index paperless-ngx itself runs.
//!
//! # Design
//!
//! One Tantivy index, on disk, three fields: `hash` (`STRING | STORED`, the
//! primary key -- exact-match only, never tokenized, so `delete_term` and a
//! hash lookup are exact), `filename` (`TEXT | STORED`, searched + shown),
//! `text` (`TEXT | STORED`, searched + the snippet source -- STORED because
//! [`SnippetGenerator`] needs the field's stored value at query time).
//!
//! Indexing is idempotent: [`SearchIndex::index_document`] deletes any
//! existing doc with the same hash before adding the new one, then commits --
//! the same delete-then-insert shape [`crate::store::LanceStore::put`]'s
//! `merge_insert` already uses, so a retried write never leaves two
//! searchable copies of the same document.
//!
//! A single [`IndexWriter`] is held for the process lifetime, guarded by a
//! [`Mutex`] -- only [`IndexWriter::commit`] needs `&mut`; `add_document`/
//! `delete_term` are `&self` and already thread-safe on their own (tantivy's
//! own internal indexing threads). The mutex exists to serialize COMMITS,
//! matching tantivy's own "there must be only one writer at a time" rule --
//! it is not protecting the writer's internal state from concurrent reads.

use std::path::Path;
use std::sync::Mutex;

use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::snippet::SnippetGenerator;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

/// One ranked search result.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Hex `content_sha256` -- joins back to [`crate::store::DocumentRow`].
    pub hash_hex: String,
    /// BM25 relevance score. [`SearchIndex::search`] already returns hits
    /// highest-score-first; this is carried for display, not re-sorting.
    pub score: f32,
    /// An HTML snippet (`<b>`-highlighted matches) generated from the
    /// indexed text -- the piece a plain SQL `LIKE` scan could never give.
    pub snippet_html: String,
}

/// Why a search-index operation failed.
#[derive(Debug)]
pub enum SearchError {
    /// The index itself failed to create/write/commit/read.
    Tantivy(tantivy::TantivyError),
    /// The user's query string did not parse (unbalanced quotes, an unknown
    /// field prefix, ...). A search box is user input; this is expected to
    /// happen and callers should show it, not treat it as a server error.
    Query(tantivy::query::QueryParserError),
    /// The on-disk index directory could not be opened.
    Directory(tantivy::directory::error::OpenDirectoryError),
    /// The directory could not be created.
    Io(std::io::Error),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tantivy(e) => write!(f, "tantivy: {e}"),
            Self::Query(e) => write!(f, "query: {e}"),
            Self::Directory(e) => write!(f, "index directory: {e}"),
            Self::Io(e) => write!(f, "index directory: {e}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<tantivy::TantivyError> for SearchError {
    fn from(e: tantivy::TantivyError) -> Self {
        Self::Tantivy(e)
    }
}

impl From<tantivy::query::QueryParserError> for SearchError {
    fn from(e: tantivy::query::QueryParserError) -> Self {
        Self::Query(e)
    }
}

impl From<tantivy::directory::error::OpenDirectoryError> for SearchError {
    fn from(e: tantivy::directory::error::OpenDirectoryError) -> Self {
        Self::Directory(e)
    }
}

struct Fields {
    hash: Field,
    filename: Field,
    text: Field,
}

fn schema_and_fields() -> (Schema, Fields) {
    let mut b = Schema::builder();
    let hash = b.add_text_field("hash", STRING | STORED);
    let filename = b.add_text_field("filename", TEXT | STORED);
    let text = b.add_text_field("text", TEXT | STORED);
    (
        b.build(),
        Fields {
            hash,
            filename,
            text,
        },
    )
}

/// An open, persistent full-text index.
pub struct SearchIndex {
    index: Index,
    fields: Fields,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
}

impl SearchIndex {
    /// Open the index at `dir`, creating it (and the directory) if absent.
    ///
    /// # Errors
    /// [`SearchError`] if the directory, index, writer, or reader fail to
    /// open.
    pub fn open_or_create(dir: &Path) -> Result<Self, SearchError> {
        std::fs::create_dir_all(dir).map_err(SearchError::Io)?;
        let (schema, fields) = schema_and_fields();
        let mmap = MmapDirectory::open(dir)?;
        let index = Index::open_or_create(mmap, schema)?;
        let writer: IndexWriter = index.writer(50_000_000)?;
        // `Manual`, not `OnCommitWithDelay`: the delayed policy reloads the
        // reader ASYNCHRONOUSLY off a filesystem watcher, so a search that
        // runs immediately after `commit()` can race it and see the STALE
        // (pre-write) snapshot -- measured directly: every test that wrote
        // then searched in the same call chain returned zero hits under
        // `OnCommitWithDelay`, while a fresh `SearchIndex` opened against the
        // same already-committed directory (no race, no delay) found them
        // fine. This archive needs a document searchable the moment ingest
        // returns, so [`Self::index_document`]/[`Self::delete_document`] call
        // [`IndexReader::reload`] synchronously right after `commit()`.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(Self {
            index,
            fields,
            writer: Mutex::new(writer),
            reader,
        })
    }

    /// Index (or re-index) one document. Idempotent: a document already
    /// carrying `hash_hex` is deleted first, so calling this twice for the
    /// same document never leaves two searchable copies.
    ///
    /// # Errors
    /// [`SearchError::Tantivy`] on a write/commit failure.
    pub fn index_document(
        &self,
        hash_hex: &str,
        filename: &str,
        text: &str,
    ) -> Result<(), SearchError> {
        let mut writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        writer.delete_term(Term::from_field_text(self.fields.hash, hash_hex));
        writer.add_document(doc!(
            self.fields.hash => hash_hex,
            self.fields.filename => filename,
            self.fields.text => text,
        ))?;
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// Remove a document from the index by its hash. A no-op (not an error)
    /// if the hash was never indexed -- `delete_term` on an absent term is
    /// itself a no-op, and a bare commit afterward is harmless.
    ///
    /// # Errors
    /// [`SearchError::Tantivy`] on a delete/commit failure.
    pub fn delete_document(&self, hash_hex: &str) -> Result<(), SearchError> {
        let mut writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        writer.delete_term(Term::from_field_text(self.fields.hash, hash_hex));
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// Ranked full-text search over `filename` + `text`, with an HTML
    /// snippet per hit, highest score first.
    ///
    /// # Errors
    /// [`SearchError::Query`] if `query_str` does not parse;
    /// [`SearchError::Tantivy`] on a search failure.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<SearchHit>, SearchError> {
        let searcher = self.reader.searcher();
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.fields.filename, self.fields.text]);
        let query = query_parser.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;
        let snippet_generator = SnippetGenerator::create(&searcher, &*query, self.fields.text)?;

        let mut out = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let hash_hex = doc
                .get_first(self.fields.hash)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let snippet_html = snippet_generator.snippet_from_doc(&doc).to_html();
            out.push(SearchHit {
                hash_hex,
                score,
                snippet_html,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> (tempfile::TempDir, SearchIndex) {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = SearchIndex::open_or_create(dir.path()).expect("open_or_create");
        (dir, idx)
    }

    /// The basic round-trip: index one document, find it by a word that
    /// only appears in its body.
    #[test]
    fn indexed_document_is_findable_by_body_text() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "invoice.pdf", "the quick brown fox jumps")
            .expect("index");
        let hits = idx.search("fox", 10).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hash_hex, "aa11");
    }

    /// A term that appears nowhere returns zero hits -- the can-stay-silent
    /// half, proving `search` does not just return everything indexed.
    #[test]
    fn a_term_that_does_not_appear_finds_nothing() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "invoice.pdf", "the quick brown fox jumps")
            .expect("index");
        let hits = idx.search("giraffe", 10).expect("search");
        assert!(hits.is_empty());
    }

    /// Ranking: a document repeating the query term scores higher than one
    /// mentioning it once -- proves this is BM25 ranking, not an unordered
    /// substring match dressed up as "search".
    #[test]
    fn a_document_with_more_matches_scores_higher() {
        let (_dir, idx) = index();
        idx.index_document("weak", "a.txt", "widgets are useful sometimes")
            .expect("index");
        idx.index_document(
            "strong",
            "b.txt",
            "widgets widgets widgets widgets everywhere widgets",
        )
        .expect("index");
        let hits = idx.search("widgets", 10).expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].hash_hex, "strong",
            "higher term frequency must rank first"
        );
        assert!(hits[0].score > hits[1].score);
    }

    /// Re-indexing the same hash must REPLACE, not duplicate -- the
    /// idempotence `index_document`'s own doc comment claims.
    #[test]
    fn reindexing_the_same_hash_replaces_not_duplicates() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "v1.pdf", "the original wording")
            .expect("index v1");
        idx.index_document("aa11", "v2.pdf", "the revised wording")
            .expect("index v2");
        let hits = idx.search("wording", 10).expect("search");
        assert_eq!(hits.len(), 1, "must not leave two searchable copies");
        assert!(hits[0].snippet_html.contains("revised"));
    }

    /// Delete actually removes the document from search results.
    #[test]
    fn deleted_document_is_no_longer_findable() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "invoice.pdf", "the quick brown fox jumps")
            .expect("index");
        assert_eq!(idx.search("fox", 10).expect("search").len(), 1);
        idx.delete_document("aa11").expect("delete");
        assert!(idx.search("fox", 10).expect("search").is_empty());
    }

    /// Deleting a hash that was never indexed must not error -- a delete
    /// racing an ingest failure, or a double-click on the delete button,
    /// must degrade gracefully rather than surfacing an internal error.
    #[test]
    fn deleting_an_unindexed_hash_is_not_an_error() {
        let (_dir, idx) = index();
        idx.delete_document("never-indexed")
            .expect("delete of absent hash must not error");
    }

    /// The snippet is real HTML highlighting, not the raw text echoed back
    /// -- proves `SnippetGenerator` is actually wired, not stubbed.
    #[test]
    fn snippet_highlights_the_matched_term() {
        let (_dir, idx) = index();
        idx.index_document(
            "aa11",
            "report.pdf",
            "quarterly revenue increased significantly this year",
        )
        .expect("index");
        let hits = idx.search("revenue", 10).expect("search");
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].snippet_html.contains("<b>")
                && hits[0].snippet_html.to_lowercase().contains("revenue"),
            "got: {}",
            hits[0].snippet_html
        );
    }

    /// A malformed query (unbalanced quote) must return a typed
    /// `SearchError::Query`, not a panic and not silently no hits.
    #[test]
    fn a_malformed_query_reports_a_query_error() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "a.txt", "hello world")
            .expect("index");
        let err = idx.search("\"unterminated", 10).unwrap_err();
        assert!(matches!(err, SearchError::Query(_)), "got: {err:?}");
    }

    /// Reopening an existing index directory must not lose what was
    /// indexed -- the whole point of using `MmapDirectory` instead of an
    /// in-RAM index is surviving a process restart.
    #[test]
    fn reopening_the_index_directory_preserves_documents() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let idx = SearchIndex::open_or_create(dir.path()).expect("open_or_create #1");
            idx.index_document("aa11", "a.txt", "persistent survives a restart")
                .expect("index");
        }
        let idx2 = SearchIndex::open_or_create(dir.path()).expect("open_or_create #2");
        let hits = idx2.search("persistent", 10).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hash_hex, "aa11");
    }
}
