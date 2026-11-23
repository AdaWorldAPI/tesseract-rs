//! Shared, read-only-ish application state: the loaded recognizer, the
//! archive connection, and the full-text search index.

use std::path::Path;
use std::sync::Arc;

use tesseract_ogar::reasoning::SentenceReasoner;
use tesseract_ogar::OcrExecutor;
use tesseract_paperless::auto_match::AutoMatchParams;
use tesseract_paperless::auto_model::{index_tokenizer, AutoModel, MineInputs};
use tesseract_paperless::auto_rows::VocabParams;
use tesseract_paperless::blobs::BlobStore;

use crate::replica::Replica;
use tesseract_paperless::kv::ContentSha256;
use tesseract_paperless::search::SearchIndex;
use tesseract_paperless::store::LanceStore;
use tokio::sync::Semaphore;

/// What the last replication pass found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplicaStatus {
    /// When the last pass ended, milliseconds since the Unix epoch.
    pub checked_at_unix_ms: Option<i64>,
    /// Documents still without a copy in S3 after that pass.
    pub backlog: Option<usize>,
    /// Why the last pass failed, if it did.
    pub error: Option<String>,
}

/// How many S3 copies run at once in the background.
pub const REPLICA_PERMITS: usize = 4;

/// How many locks [`AppState::hash_lock`] spreads hashes over.
pub const HASH_LOCK_STRIPES: usize = 64;

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
    /// The uploaded originals, keyed by their SHA-256 (`blobs.rs`). Written
    /// before the document row (S-5), so a row never points at a missing file
    /// unless the row predates this store.
    pub blobs: Arc<BlobStore>,
    /// The S3 copy (`replica.rs`), when `PAPERLESS_S3_*` is configured.
    pub replica: Option<Arc<Replica>>,
    /// What the last replication pass found, for the dashboard.
    pub replica_status: std::sync::Mutex<ReplicaStatus>,
    /// Bounds how many S3 copies run at once in the background.
    replica_permits: Arc<Semaphore>,
    /// Serializes the original and row writes of an upload with the row and
    /// original deletes of a delete, for the same hash (see [`Self::hash_lock`]).
    hash_locks: Box<[Arc<tokio::sync::Mutex<()>>]>,
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
    /// `lancedb_uri`, open (or create) the search index at
    /// `search_index_dir` and the original-file store at `blob_dir`.
    ///
    /// # Errors
    /// A human-readable message on any failure — the caller prints it and
    /// exits, matching `tesseract-ocr-web`'s startup contract.
    pub async fn load(
        model_dir: &Path,
        lancedb_uri: &str,
        search_index_dir: &Path,
        blob_dir: &Path,
        deepnsm_vocab_dir: &Path,
        replica: Option<Replica>,
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

        let blobs = Arc::new(
            BlobStore::open(blob_dir)
                .map_err(|e| format!("open original-file store at {}: {e}", blob_dir.display()))?,
        );

        // An empty volume (or an interrupted restore) is filled from the S3
        // copy before anything else reads the archive, so the reconcile below
        // indexes the restored documents (paperless-rs `docs/RECOVERY.md` §6).
        // A failure stops the start: an empty archive must not begin taking
        // uploads while its history is unreachable.
        let replica = replica.map(Arc::new);
        if let Some(r) = &replica {
            let restore = crate::replica::needs_restore(&store, blob_dir)
                .await
                .map_err(|e| format!("check whether to restore: {e}"))?;
            if restore {
                let report = crate::replica::restore(r, &store, &blobs)
                    .await
                    .map_err(|e| {
                        format!(
                            "restore from the S3 copy failed: {e} (the archive is empty or a \
                             restore was interrupted; it will retry on the next start)"
                        )
                    })?;
                eprintln!(
                    "tesseract-paperless-web: restored from S3: {} documents ({} already local, \
                     {} skipped, {} without original), metadata {}",
                    report.restored,
                    report.already_local,
                    report.skipped,
                    report.without_original,
                    if report.metadata { "imported" } else { "absent" }
                );
            }
        }

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
            blobs,
            replica,
            replica_status: std::sync::Mutex::new(ReplicaStatus::default()),
            replica_permits: Arc::new(Semaphore::new(REPLICA_PERMITS)),
            hash_locks: (0..HASH_LOCK_STRIPES)
                .map(|_| Arc::new(tokio::sync::Mutex::new(())))
                .collect(),
            reasoner,
            recognize_permits: Arc::new(Semaphore::new(permits)),
            auto: std::sync::RwLock::new(AutoSlot::Pending),
        })
    }

    /// The lock an upload holds while it writes a document's original and row,
    /// and a delete holds while it removes them. Without it, a delete can
    /// remove the original a concurrent upload of the same file just wrote,
    /// and the upload then commits a row whose original is gone.
    ///
    /// Striped by the first hash byte: two different documents share a lock
    /// only 1 time in [`HASH_LOCK_STRIPES`], and then only wait for each
    /// other's short write, never for recognition.
    ///
    /// Shared ownership so a task can hold it after the request that started
    /// the task is gone (see `routes::document_delete`).
    #[must_use]
    pub fn hash_lock(&self, key: &ContentSha256) -> Arc<tokio::sync::Mutex<()>> {
        self.hash_locks[usize::from(key.as_bytes()[0]) % self.hash_locks.len()].clone()
    }

    /// Copy one document to S3 after its local commit, in the background: the
    /// request that committed it does not wait for S3. The copy takes the
    /// document's lock, so it stays ordered with a delete of the same
    /// document. A failure is logged; the replicator copies it later
    /// (`docs/RECOVERY.md` §4 and §5). Nothing is lost if the process stops
    /// before the copy runs: the replicator finds every document whose S3
    /// copy is missing or lacks its original.
    pub fn replicate_document(self: &Arc<Self>, hash: &str) {
        if self.replica.is_none() {
            return;
        }
        let Some(key) = ContentSha256::from_hex(hash) else {
            return;
        };
        let state = Arc::clone(self);
        let hash = hash.to_string();
        tokio::spawn(async move {
            let Ok(_permit) = Arc::clone(&state.replica_permits).acquire_owned().await else {
                return;
            };
            let Some(r) = &state.replica else { return };
            if let Err(e) = r
                .upload_document(&state.store, &state.blobs, &hash, state.hash_lock(&key))
                .await
            {
                eprintln!("replica: copying {hash} to S3 failed: {e}; the replicator will retry");
            }
        });
    }

    /// Upload the metadata snapshot after a metadata change, in the
    /// background. Each upload reads the metadata when it runs, under the
    /// snapshot lock, so whichever runs last uploads the newest state. A
    /// failure is logged; the replicator uploads it again.
    pub fn replicate_metadata(self: &Arc<Self>) {
        if self.replica.is_none() {
            return;
        }
        let state = Arc::clone(self);
        tokio::spawn(async move {
            let Ok(_permit) = Arc::clone(&state.replica_permits).acquire_owned().await else {
                return;
            };
            let Some(r) = &state.replica else { return };
            if let Err(e) = r.upload_snapshot(&state.store).await {
                eprintln!("replica: metadata snapshot upload failed: {e}; the replicator will retry");
            }
        });
    }

    /// One replication pass: copy what is missing in S3, upload the snapshot,
    /// and record the backlog for the dashboard.
    pub async fn replicate_now(&self) {
        let Some(r) = &self.replica else { return };
        let outcome = r
            .replicate(&self.store, &self.blobs, &|k| self.hash_lock(k))
            .await;
        let backlog = r.backlog(&self.store, &self.blobs).await;
        let now = crate::ingest::now_unix_ms();
        let mut status = self
            .replica_status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        status.checked_at_unix_ms = Some(now);
        match (outcome, backlog) {
            (Ok(report), Ok(n)) => {
                if report.uploaded + report.failed > 0 {
                    eprintln!(
                        "replica: {} copied to S3, {} failed, {n} still missing",
                        report.uploaded, report.failed
                    );
                }
                status.backlog = Some(n);
                status.error = None;
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("replica: replication pass failed: {e}");
                status.error = Some(e.to_string());
            }
        }
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
