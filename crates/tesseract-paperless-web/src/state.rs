//! Shared, read-only-ish application state: the loaded recognizer, the
//! archive connection, and the full-text search index.

use std::path::Path;
use std::sync::Arc;

use tesseract_ogar::reasoning::SentenceReasoner;
use tesseract_ogar::OcrExecutor;
use tesseract_paperless::auto_match::AutoMatchParams;
use tesseract_paperless::auto_model::{index_tokenizer, AutoModel, MineInputs};
use tesseract_paperless::auto_rows::VocabParams;
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
    /// The AUTO model (spec `archive-metadata-auto-match-v3.md` R7). Swapped
    /// whole on every re-mine; a reader clones the slot and never holds the
    /// lock across an `.await`.
    pub auto: std::sync::RwLock<AutoSlot>,
}

/// Where the AUTO model stands.
#[derive(Clone)]
pub enum AutoSlot {
    /// No mine has finished yet. Mining runs off the boot path, so the app
    /// serves before it has a model (spec R7).
    Pending,
    /// The current model.
    Ready(Arc<AutoModel>),
    /// The last mine was refused; the text is shown on the definitions page
    /// (for example "no content term survived", spec R5).
    Refused(String),
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

        // The index holds references into the archive, never text of its
        // own, so it is brought back in line with the archive on every start:
        // documents archived before a crash reached the index are indexed,
        // entries whose document was deleted are dropped, and an index rebuilt
        // for a schema change is refilled. A failure here degrades search,
        // never the archive, so it is logged rather than fatal.
        if search.was_rebuilt() {
            eprintln!(
                "tesseract-paperless-web: search index at {} had a stale schema and was rebuilt",
                search_index_dir.display()
            );
        }
        match tesseract_paperless::reconcile::reconcile(&store, &search).await {
            Ok(r) if r != tesseract_paperless::reconcile::ReconcileReport::default() => eprintln!(
                "tesseract-paperless-web: search index reconciled: {} indexed, {} removed, \
                 {} unreadable",
                r.indexed, r.removed, r.unreadable
            ),
            Ok(_) => {}
            Err(e) => eprintln!("tesseract-paperless-web: search index reconcile failed: {e}"),
        }

        // Assignment and review rows whose document is gone are orphans: a
        // delete that failed between the document row and its metadata leaves
        // them (spec `archive-metadata-auto-match-v3.md` R2a). Sweeping them
        // here also keeps a re-upload of a deleted hash from inheriting its old
        // review flag. Logged, never fatal, like the index reconcile above.
        let live = store
            .hashes()
            .await
            .map(|h| h.into_iter().collect::<std::collections::HashSet<_>>());
        match live {
            Ok(live) => match store.meta().reconcile(&live).await {
                Ok(r) if r.assignments_removed + r.reviews_removed > 0 => eprintln!(
                    "tesseract-paperless-web: metadata reconciled: {} assignments, {} reviews \
                     removed",
                    r.assignments_removed, r.reviews_removed
                ),
                Ok(_) => {}
                Err(e) => eprintln!("tesseract-paperless-web: metadata reconcile failed: {e}"),
            },
            Err(e) => eprintln!("tesseract-paperless-web: metadata reconcile skipped: {e}"),
        }

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
            auto: std::sync::RwLock::new(AutoSlot::Pending),
        })
    }

    /// The current AUTO slot.
    #[must_use]
    pub fn auto_slot(&self) -> AutoSlot {
        self.auto
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Mine a fresh AUTO model from the archive and swap it in.
    ///
    /// The archive is read on the async runtime; the CPU-bound build runs on
    /// a blocking thread. A refusal replaces the slot with its reason (the
    /// previous model is dropped: it was mined from an archive that no longer
    /// exists). A read failure leaves the slot as it was.
    pub async fn remine(self: Arc<Self>) {
        let inputs = match MineInputs::load(&self.store).await {
            Ok(i) => i,
            Err(e) => {
                eprintln!("tesseract-paperless-web: AUTO mine could not read the archive: {e}");
                return;
            }
        };
        let st = self.clone();
        let built = tokio::task::spawn_blocking(move || {
            AutoModel::build(
                &inputs,
                &index_tokenizer(&st.search),
                AutoMatchParams::default(),
                VocabParams::default(),
            )
        })
        .await;
        let slot = match built {
            Ok(Ok(model)) => {
                eprintln!(
                    "tesseract-paperless-web: AUTO model mined, {} rules",
                    model.rule_count()
                );
                AutoSlot::Ready(Arc::new(model))
            }
            Ok(Err(e)) => {
                eprintln!("tesseract-paperless-web: AUTO mine refused: {e}");
                AutoSlot::Refused(e.to_string())
            }
            Err(e) => {
                eprintln!("tesseract-paperless-web: AUTO mine task failed: {e}");
                return;
            }
        };
        *self
            .auto
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = slot;
    }
}
