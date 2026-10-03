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
//! hash lookup are exact), and `filename` + `text` (`TEXT`, **indexed but not
//! stored**).
//!
//! # The index holds references, not text
//!
//! The one stored copy of a document's text is its `DocIr`, archived by
//! `crate::store`. This index keeps only the postings needed to rank a query
//! and the hash that joins a hit back to that copy. Snippets are therefore
//! cut from text the CALLER supplies ([`SearchResults::snippet_html`]), which
//! is the archived `DocIr`'s derived plain text -- not from a second copy kept
//! here. The consequence worth knowing: the index is disposable. It can be
//! deleted and rebuilt from the archive at any time (`crate::reconcile`),
//! which is also how a schema change is handled
//! ([`SearchIndex::open_or_create`]).
//!
//! # Stemming, per language
//!
//! The body text is indexed three times: `text` with Tantivy's default
//! analyzer (exact surface, and the analyzer the AUTO vocabulary reads via
//! [`SearchIndex::tokenize_text`], so AUTO's terms do not move), plus one
//! stemmed field per language in [`STEM_LANGUAGES`] (`text_en`, `text_de`).
//! The archive records no per-document language -- it OCRs with one model --
//! so the language choice is made at QUERY time instead: a query is analyzed
//! by every field's own stemmer, `Rechnungen` reaches `rechnung` through the
//! German field and `invoices` reaches `invoice` through the English one.
//! Only postings are added; no text is stored. Tantivy does not persist
//! analyzers, so they are registered on every [`SearchIndex::open_or_create`].
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

use tantivy::collector::{DocSetCollector, TopDocs};
use tantivy::directory::MmapDirectory;
use tantivy::query::{AllQuery, QueryParser};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING, TEXT,
};
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer,
};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

/// The languages the body text is stemmed for, each into its own field:
/// `(field name, registered analyzer name, Snowball language)`. English and
/// German match the two OCR models this workspace ships (`eng`, `deu`).
pub const STEM_LANGUAGES: [(&str, &str, Language); 2] = [
    ("text_en", "paperless_stem_en", Language::English),
    ("text_de", "paperless_stem_de", Language::German),
];

/// The stemmed analyzer for one language: Tantivy's default chain
/// (simple tokenizer, drop tokens over 40 bytes, lowercase) plus a Snowball
/// stemmer. The stemmer does not lowercase, so the order matters.
fn stem_analyzer(language: Language) -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(Stemmer::new(language))
        .build()
}

/// One ranked search result: a reference into the archive, never text.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Hex `content_sha256` -- joins back to [`crate::store::DocumentRow`].
    pub hash_hex: String,
    /// BM25 relevance score. [`SearchIndex::search`] already returns hits
    /// highest-score-first; this is carried for display, not re-sorting.
    pub score: f32,
}

/// The ranked hits of one query, plus what is needed to highlight them.
pub struct SearchResults {
    /// Hits, highest score first.
    pub hits: Vec<SearchHit>,
    /// One generator per body field (plain, then each stem field): a query
    /// can match a document only through a stem field, and a generator only
    /// highlights the terms its own field produced.
    snippets: Vec<SnippetGenerator>,
}

impl SearchResults {
    /// An HTML snippet (`<b>`-highlighted matches) cut from `text` -- the
    /// hit's document text as derived from the archive. Tantivy's highlighter
    /// HTML-escapes everything around the `<b>` tags, so the result is safe
    /// to render unescaped.
    #[must_use]
    pub fn snippet_html(&self, text: &str) -> String {
        // The first field whose terms actually occur in `text` wins; if none
        // highlights anything, fall back to the plain field's excerpt.
        self.snippets
            .iter()
            .map(|g| g.snippet(text))
            .find(|s| !s.highlighted().is_empty())
            .or_else(|| self.snippets.first().map(|g| g.snippet(text)))
            .map(|s| s.to_html())
            .unwrap_or_default()
    }
}

impl core::fmt::Debug for SearchResults {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SearchResults")
            .field("hits", &self.hits)
            .finish_non_exhaustive()
    }
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
    /// One per [`STEM_LANGUAGES`] entry, in that order.
    text_stem: [Field; STEM_LANGUAGES.len()],
}

fn schema_and_fields() -> (Schema, Fields) {
    let mut b = Schema::builder();
    let hash = b.add_text_field("hash", STRING | STORED);
    let filename = b.add_text_field("filename", TEXT);
    let text = b.add_text_field("text", TEXT);
    let text_stem = STEM_LANGUAGES.map(|(field, analyzer, _)| {
        let indexing = TextFieldIndexing::default()
            .set_tokenizer(analyzer)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        b.add_text_field(field, TextOptions::default().set_indexing_options(indexing))
    });
    (
        b.build(),
        Fields {
            hash,
            filename,
            text,
            text_stem,
        },
    )
}

/// An open, persistent full-text index.
pub struct SearchIndex {
    index: Index,
    fields: Fields,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    rebuilt: bool,
    #[cfg(test)]
    commits: std::sync::atomic::AtomicUsize,
}

impl SearchIndex {
    /// Open the index at `dir`, creating it (and the directory) if absent.
    ///
    /// An existing index whose schema differs from this build's -- e.g. one
    /// written before the text field stopped being stored -- is discarded and
    /// recreated empty rather than refused: the index holds no content the
    /// archive does not, so `crate::reconcile` repopulates it.
    /// [`Self::was_rebuilt`] reports when that happened. `dir` must be
    /// dedicated to this index, since a rebuild clears it.
    ///
    /// # Errors
    /// [`SearchError`] if the directory, index, writer, or reader fail to
    /// open.
    pub fn open_or_create(dir: &Path) -> Result<Self, SearchError> {
        std::fs::create_dir_all(dir).map_err(SearchError::Io)?;
        let (schema, fields) = schema_and_fields();
        let mut rebuilt = false;
        if Index::exists(&MmapDirectory::open(dir)?).map_err(tantivy::TantivyError::from)? {
            let existing = Index::open(MmapDirectory::open(dir)?)?;
            if existing.schema() != schema {
                drop(existing);
                for entry in std::fs::read_dir(dir).map_err(SearchError::Io)? {
                    let path = entry.map_err(SearchError::Io)?.path();
                    if path.is_dir() {
                        std::fs::remove_dir_all(&path)
                    } else {
                        std::fs::remove_file(&path)
                    }
                    .map_err(SearchError::Io)?;
                }
                rebuilt = true;
            }
        }
        let mmap = MmapDirectory::open(dir)?;
        let index = Index::open_or_create(mmap, schema)?;
        for (_, analyzer, language) in STEM_LANGUAGES {
            index
                .tokenizers()
                .register(analyzer, stem_analyzer(language));
        }
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
            rebuilt,
            #[cfg(test)]
            commits: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Split `text` into tokens exactly as the index splits its `text` field.
    ///
    /// The AUTO matcher's vocabulary is built with this, so its content terms
    /// are the index's own tokens, not a second tokenizer's guess at them
    /// (spec `archive-metadata-auto-match-v3.md` R5). Tokens come back in
    /// order, repeats included.
    ///
    /// # Errors
    /// [`SearchError::Tantivy`] if the field has no registered analyzer.
    pub fn tokenize_text(&self, text: &str) -> Result<Vec<String>, SearchError> {
        let mut analyzer = self.index.tokenizer_for_field(self.fields.text)?;
        let mut stream = analyzer.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        Ok(out)
    }

    /// Whether [`Self::open_or_create`] discarded an index with a stale
    /// schema. When true the index is empty until reconciled.
    #[must_use]
    pub fn was_rebuilt(&self) -> bool {
        self.rebuilt
    }

    /// Every indexed document's hash -- what `crate::reconcile` compares
    /// against the archive.
    ///
    /// # Errors
    /// [`SearchError::Tantivy`] on a read failure.
    pub fn indexed_hashes(&self) -> Result<Vec<String>, SearchError> {
        let searcher = self.reader.searcher();
        let addrs = searcher.search(&AllQuery, &DocSetCollector)?;
        let mut out = Vec::with_capacity(addrs.len());
        for addr in addrs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(h) = doc.get_first(self.fields.hash).and_then(|v| v.as_str()) {
                out.push(h.to_string());
            }
        }
        Ok(out)
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
        self.apply_batch(&[(hash_hex, filename, text)], &[])
    }

    /// Index every `(hash, filename, text)` in `upserts` and remove every
    /// hash in `deletes`, under one writer lock and ONE commit. A commit
    /// writes and fsyncs a segment, so doing this per document would cost
    /// one segment per document -- which is what rebuilding a whole index
    /// through [`Self::index_document`] would do. Upserts are idempotent, as
    /// in [`Self::index_document`].
    ///
    /// # Errors
    /// [`SearchError::Tantivy`] on a write/commit failure.
    pub fn apply_batch(
        &self,
        upserts: &[(&str, &str, &str)],
        deletes: &[&str],
    ) -> Result<(), SearchError> {
        let mut writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for hash_hex in deletes {
            writer.delete_term(Term::from_field_text(self.fields.hash, hash_hex));
        }
        for &(hash_hex, filename, text) in upserts {
            writer.delete_term(Term::from_field_text(self.fields.hash, hash_hex));
            let mut d = doc!(
                self.fields.hash => hash_hex,
                self.fields.filename => filename,
                self.fields.text => text,
            );
            for field in self.fields.text_stem {
                d.add_text(field, text);
            }
            writer.add_document(d)?;
        }
        writer.commit()?;
        drop(writer);
        #[cfg(test)]
        self.commits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.reader.reload()?;
        Ok(())
    }

    /// Commits made through this handle. Segment count cannot stand in for
    /// it: one commit yields a segment per indexing thread.
    #[cfg(all(test, feature = "store"))]
    pub(crate) fn commit_count(&self) -> usize {
        self.commits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Remove a document from the index by its hash. A no-op (not an error)
    /// if the hash was never indexed -- `delete_term` on an absent term is
    /// itself a no-op, and a bare commit afterward is harmless.
    ///
    /// # Errors
    /// [`SearchError::Tantivy`] on a delete/commit failure.
    pub fn delete_document(&self, hash_hex: &str) -> Result<(), SearchError> {
        self.apply_batch(&[], &[hash_hex])
    }

    /// Ranked full-text search over `filename` + `text`, highest score
    /// first. Highlight a hit with [`SearchResults::snippet_html`], passing
    /// that document's text from the archive.
    ///
    /// # Errors
    /// [`SearchError::Query`] if `query_str` does not parse;
    /// [`SearchError::Tantivy`] on a search failure.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<SearchResults, SearchError> {
        let searcher = self.reader.searcher();
        let mut fields = vec![self.fields.filename, self.fields.text];
        fields.extend(self.fields.text_stem);
        let query_parser = QueryParser::for_index(&self.index, fields);
        let query = query_parser.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;
        let mut snippets = vec![SnippetGenerator::create(
            &searcher,
            &*query,
            self.fields.text,
        )?];
        for field in self.fields.text_stem {
            snippets.push(SnippetGenerator::create(&searcher, &*query, field)?);
        }

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let hash_hex = doc
                .get_first(self.fields.hash)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            hits.push(SearchHit { hash_hex, score });
        }
        Ok(SearchResults { hits, snippets })
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

    /// `tokenize_text` is the index's own analyzer, not a whitespace split:
    /// it lower-cases, splits on punctuation, and drops tokens over 40 bytes.
    /// A plain `split_whitespace` fails all three. And every token it returns
    /// is findable, which is the property the AUTO vocabulary relies on.
    #[test]
    fn tokenize_text_is_the_index_analyzer() {
        let (_dir, idx) = index();
        let long = "x".repeat(41);
        let text = format!("Invoice, STADTWERKE; invoice {long}");
        let tokens = idx.tokenize_text(&text).expect("tokenize");
        assert_eq!(tokens, vec!["invoice", "stadtwerke", "invoice"]);
        idx.index_document("aa11", "a.pdf", &text).expect("index");
        for t in &tokens {
            assert_eq!(idx.search(t, 10).expect("search").hits.len(), 1, "{t}");
        }
    }

    /// The basic round-trip: index one document, find it by a word that
    /// only appears in its body.
    #[test]
    fn indexed_document_is_findable_by_body_text() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "invoice.pdf", "the quick brown fox jumps")
            .expect("index");
        let hits = idx.search("fox", 10).expect("search").hits;
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
        let hits = idx.search("giraffe", 10).expect("search").hits;
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
        let hits = idx.search("widgets", 10).expect("search").hits;
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
        let hits = idx.search("wording", 10).expect("search").hits;
        assert_eq!(hits.len(), 1, "must not leave two searchable copies");
        assert!(
            idx.search("original", 10).expect("search").hits.is_empty(),
            "the replaced wording must no longer match"
        );
        assert_eq!(idx.search("revised", 10).expect("search").hits.len(), 1);
    }

    /// Delete actually removes the document from search results.
    #[test]
    fn deleted_document_is_no_longer_findable() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "invoice.pdf", "the quick brown fox jumps")
            .expect("index");
        assert_eq!(idx.search("fox", 10).expect("search").hits.len(), 1);
        idx.delete_document("aa11").expect("delete");
        assert!(idx.search("fox", 10).expect("search").hits.is_empty());
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
        let results = idx.search("revenue", 10).expect("search");
        assert_eq!(results.hits.len(), 1);
        let html = results.snippet_html("quarterly revenue increased significantly this year");
        assert!(
            html.contains("<b>") && html.to_lowercase().contains("revenue"),
            "got: {html}"
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
        let hits = idx2.search("persistent", 10).expect("search").hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hash_hex, "aa11");
    }

    /// The index keeps the hash and nothing else: the text is indexed for
    /// ranking but not stored, so the archive's `DocIr` stays its only copy.
    #[test]
    fn the_index_stores_no_text() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "invoice.pdf", "the quick brown fox jumps")
            .expect("index");
        let searcher = idx.reader.searcher();
        let addrs = searcher.search(&AllQuery, &DocSetCollector).expect("all");
        assert_eq!(addrs.len(), 1);
        let doc: TantivyDocument = searcher.doc(*addrs.iter().next().unwrap()).expect("doc");
        assert_eq!(
            doc.get_first(idx.fields.hash).and_then(|v| v.as_str()),
            Some("aa11")
        );
        assert!(
            doc.get_first(idx.fields.text).is_none(),
            "text must not be stored"
        );
        assert!(
            doc.get_first(idx.fields.filename).is_none(),
            "filename must not be stored"
        );
    }

    /// The snippet is cut from the text the caller passes, not from anything
    /// the index kept -- highlighting a text the index never saw still works.
    #[test]
    fn the_snippet_is_cut_from_the_supplied_text() {
        let (_dir, idx) = index();
        idx.index_document("aa11", "a.txt", "revenue")
            .expect("index");
        let results = idx.search("revenue", 10).expect("search");
        let html = results.snippet_html("an entirely different sentence about revenue growth");
        assert!(html.contains("<b>revenue</b>"), "got: {html}");
        assert!(
            html.contains("growth"),
            "the snippet comes from the supplied text: {html}"
        );
    }

    /// An index written with an older schema (text STORED) is discarded and
    /// recreated empty instead of refusing to open; reconcile refills it.
    #[test]
    fn a_stale_schema_index_is_rebuilt_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut b = Schema::builder();
            let hash = b.add_text_field("hash", STRING | STORED);
            let text = b.add_text_field("text", TEXT | STORED);
            let old = Index::create_in_dir(dir.path(), b.build()).expect("old index");
            let mut w: IndexWriter = old.writer(15_000_000).expect("writer");
            w.add_document(doc!(hash => "old1", text => "stored text"))
                .expect("add");
            w.commit().expect("commit");
        }
        let idx = SearchIndex::open_or_create(dir.path()).expect("a stale index must reopen");
        assert!(idx.was_rebuilt());
        assert!(idx.indexed_hashes().expect("hashes").is_empty());
        idx.index_document("new1", "b.txt", "fresh wording")
            .expect("index");
        assert_eq!(idx.search("fresh", 10).expect("search").hits.len(), 1);
    }

    /// A current-schema index is reopened as is -- the silence half of the
    /// rebuild rule.
    #[test]
    fn a_current_schema_index_is_not_rebuilt() {
        let dir = tempfile::tempdir().expect("tempdir");
        SearchIndex::open_or_create(dir.path())
            .expect("first open")
            .index_document("aa11", "a.txt", "kept")
            .expect("index");
        let idx = SearchIndex::open_or_create(dir.path()).expect("reopen");
        assert!(!idx.was_rebuilt());
        assert_eq!(
            idx.indexed_hashes().expect("hashes"),
            vec!["aa11".to_string()]
        );
    }

    /// German inflection reaches the German stem field: the plural finds a
    /// document that only has the singular. The plain `text` field alone
    /// cannot do this (`rechnungen` != `rechnung`).
    #[test]
    fn a_german_plural_finds_its_singular() {
        let (_dir, idx) = index();
        idx.index_document("de01", "a.pdf", "Die Rechnung vom Mai")
            .expect("index");
        let hits = idx.search("Rechnungen", 10).expect("search").hits;
        assert_eq!(hits.len(), 1, "German plural must match the singular");
        assert_eq!(hits[0].hash_hex, "de01");
    }

    /// The English half of the same rule.
    #[test]
    fn an_english_plural_finds_its_singular() {
        let (_dir, idx) = index();
        idx.index_document("en01", "a.pdf", "one invoice attached")
            .expect("index");
        let hits = idx.search("invoices", 10).expect("search").hits;
        assert_eq!(hits.len(), 1, "English plural must match the singular");
    }

    /// The silence twin: stemming widens a word to its own family, not to
    /// unrelated words. A stem field that matched everything would pass the
    /// two tests above and fail this one.
    #[test]
    fn stemming_does_not_match_unrelated_words() {
        let (_dir, idx) = index();
        idx.index_document("de01", "a.pdf", "Die Rechnung vom Mai")
            .expect("index");
        assert!(idx.search("Vertrag", 10).expect("search").hits.is_empty());
        assert!(idx.search("contracts", 10).expect("search").hits.is_empty());
    }

    /// A stemmed match is highlighted: the snippet marks the surface word the
    /// query reached through its stem, not nothing.
    #[test]
    fn a_stemmed_match_is_highlighted() {
        let (_dir, idx) = index();
        let text = "one invoice attached";
        idx.index_document("en01", "a.pdf", text).expect("index");
        let results = idx.search("invoices", 10).expect("search");
        assert!(
            results.snippet_html(text).contains("<b>invoice</b>"),
            "{}",
            results.snippet_html(text)
        );
    }

    /// Analyzers are not persisted by Tantivy; they are registered on every
    /// open. A reopened index must still stem, or a restart would silently
    /// fall back to unknown-tokenizer errors or exact matching.
    #[test]
    fn a_reopened_index_still_stems() {
        let dir = tempfile::tempdir().expect("tempdir");
        SearchIndex::open_or_create(dir.path())
            .expect("first open")
            .index_document("de01", "a.pdf", "Die Rechnung vom Mai")
            .expect("index");
        let idx = SearchIndex::open_or_create(dir.path()).expect("reopen");
        assert_eq!(idx.search("Rechnungen", 10).expect("search").hits.len(), 1);
    }
}
