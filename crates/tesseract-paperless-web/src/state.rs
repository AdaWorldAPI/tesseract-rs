//! Shared, read-only-ish application state: the loaded recognizer and the
//! archive connection.

use std::path::Path;
use std::sync::Arc;

use tesseract_ogar::OcrExecutor;
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
    /// Bounds concurrent CPU-bound recognitions, same reasoning as
    /// `tesseract-ocr-web::AppState::recognize_permits`.
    pub recognize_permits: Arc<Semaphore>,
}

impl AppState {
    /// Load the model from `model_dir` and connect the archive at
    /// `lancedb_uri`.
    ///
    /// # Errors
    /// A human-readable message on either failure — the caller prints it and
    /// exits, matching `tesseract-ocr-web`'s startup contract.
    pub async fn load(model_dir: &Path, lancedb_uri: &str) -> Result<Self, String> {
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

        let permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);

        Ok(Self {
            executor,
            store,
            recognize_permits: Arc::new(Semaphore::new(permits)),
        })
    }
}
