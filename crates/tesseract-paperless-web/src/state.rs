//! Shared, read-only-ish application state: the loaded recognizer, the
//! archive connection, and the full-text search index.

use std::path::Path;
use std::sync::Arc;

use tesseract_ogar::reasoning::SentenceReasoner;
use tesseract_ogar::OcrExecutor;
use tesseract_paperless::search::SearchIndex;
use tesseract_paperless::store::LanceStore;
use tokio::sync::Semaphore;

/// The archive's tissue, loaded once at startup and shared.
pub struct AppState {
    /// The pure-Rust recognizer, dict-optional. `eng` only — this crate is
    /// the archive demo, not the full language-selection surface
    /// `tesseract-ocr-web` already owns; add a `lang` field the same way
    /// that crate did if a second model becomes worth carrying here.
    pub executor: OcrExecutor,
    /// The document archive.
    pub store: LanceStore,
    /// The full-text search index (BM25 + snippets) over the archive's
    /// text. A SEPARATE persistent store from `store` (see `search.rs`'s
    /// module doc for why): both are kept in sync by `ingest.rs`/`routes.rs`
    /// calling `search.index_document`/`search.delete_document` alongside
    /// every `store.put`/`store.delete`.
    pub search: SearchIndex,
    /// Sentence assembly + deepnsm SPO/`NarsTruth` extraction
    /// (`tesseract_ogar::{sentences,reasoning}`) — `None` when the deepnsm
    /// `word_frequency/` vocabulary is not present at `DEEPNSM_VOCAB_DIR`
    /// (or its build-time fallback), the SAME graceful-degrade shape
    /// `tesseract-ogar/examples/ocr_demo.rs`'s own step 6 already uses:
    /// absence means `ingest.rs` skips extraction and archives the document
    /// with `spo_json: None`, never a startup failure. This is a
    /// post-processing layer over an already-recognized page, not a 15th
    /// `OcrExecutor` capability — see `reasoning.rs`'s module docs for the
    /// AS-IS BOUNDARY this sits on.
    pub reasoner: Option<SentenceReasoner>,
    /// Bounds concurrent CPU-bound recognitions, same reasoning as
    /// `tesseract-ocr-web::AppState::recognize_permits`.
    pub recognize_permits: Arc<Semaphore>,
}

impl AppState {
    /// Load the model from `model_dir`, connect the archive at
    /// `lancedb_uri`, and open (or create) the search index at
    /// `search_index_dir`.
    ///
    /// # Errors
    /// A human-readable message on any failure — the caller prints it and
    /// exits, matching `tesseract-ocr-web`'s startup contract.
    pub async fn load(
        model_dir: &Path,
        lancedb_uri: &str,
        search_index_dir: &Path,
        deepnsm_vocab_dir: &Path,
    ) -> Result<Self, String> {
        let path = |name: &str| model_dir.join(format!("eng.{name}"));
        let opt = |name: &str| path(name).exists().then(|| path(name));
        let executor = OcrExecutor::from_data_paths(
            &path("lstm"),
            &path("lstm-unicharset"),
            &path("lstm-recoder"),
            opt("lstm-word-dawg").as_deref(),
            opt("lstm-punc-dawg").as_deref(),
            opt("lstm-number-dawg").as_deref(),
        )
        .map_err(|e| format!("load eng model from {}: {e:?}", model_dir.display()))?;

        let store = LanceStore::connect(lancedb_uri)
            .await
            .map_err(|e| format!("connect archive at {lancedb_uri}: {e}"))?;

        // Synchronous, but a one-time startup call (not per-request) —
        // the same tradeoff `OcrExecutor::from_data_paths` above already
        // makes in this same function.
        let search = SearchIndex::open_or_create(search_index_dir)
            .map_err(|e| format!("open search index at {}: {e}", search_index_dir.display()))?;

        // Graceful degrade, not a startup failure — mirrors
        // `tesseract-ogar/examples/ocr_demo.rs`'s own step 6: absence of the
        // deepnsm vocabulary means SPO extraction is skipped per-document,
        // never that the archive refuses to start.
        let reasoner = if deepnsm_vocab_dir.join("word_rank_lookup.csv").exists() {
            match SentenceReasoner::from_vocab_dir(deepnsm_vocab_dir) {
                Ok(r) => Some(r),
                Err(e) => {
                    eprintln!(
                        "tesseract-paperless-web: deepnsm vocabulary at {} failed to load ({e}) \
                         — SPO extraction disabled, archiving still proceeds",
                        deepnsm_vocab_dir.display()
                    );
                    None
                }
            }
        } else {
            eprintln!(
                "tesseract-paperless-web: deepnsm vocabulary not present at {} \
                 — SPO extraction disabled, archiving still proceeds",
                deepnsm_vocab_dir.display()
            );
            None
        };

        let permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);

        Ok(Self {
            executor,
            store,
            search,
            reasoner,
            recognize_permits: Arc::new(Semaphore::new(permits)),
        })
    }
}
