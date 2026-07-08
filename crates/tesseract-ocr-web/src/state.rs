//! Shared, read-only application state: the loaded recognizer + dictionary.
//!
//! `LstmRecognizer` and `DictLite` are both plain owned data (no interior
//! mutability), so `&AppState` is `Send + Sync` and can be shared across all
//! request handlers via `Arc` — recognition is `&self`, never mutating.

use std::path::Path;
use std::sync::Arc;

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;
use tokio::sync::Semaphore;

/// The model tissue, loaded once at startup and shared read-only.
pub struct AppState {
    /// The pure-Rust LSTM recognizer (network + charset + recoder).
    pub recognizer: LstmRecognizer,
    /// The dictionary DAWGs for the production dict beam (optional).
    pub dict: Option<DictLite>,
    /// Bounds how many CPU-bound recognitions run at once. Recognition is heavy
    /// synchronous work dispatched via `spawn_blocking`; without a cap a burst
    /// of uploads would flood the blocking pool and the box's memory. Sized to
    /// the machine's parallelism so we saturate cores without oversubscribing.
    pub recognize_permits: Arc<Semaphore>,
}

impl AppState {
    /// Load the six `eng.lstm*` components from `model_dir` and assemble the
    /// recognizer + dictionary. Returns a human-readable `Err` on any failure
    /// (the caller prints it and exits — a server with no model is useless).
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<Vec<u8>, String> {
            std::fs::read(model_dir.join(name))
                .map_err(|e| format!("read {}/{name}: {e}", model_dir.display()))
        };
        let lstm = read("eng.lstm")?;
        let uni = String::from_utf8(read("eng.lstm-unicharset")?)
            .map_err(|e| format!("eng.lstm-unicharset is not UTF-8: {e}"))?;
        let rec = read("eng.lstm-recoder")?;
        let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
            .map_err(|e| format!("assemble recognizer: {e}"))?;

        // The dict is optional: without the DAWGs the recognizer still runs
        // (plain beam), so a missing dawg degrades gracefully rather than
        // failing startup.
        let dict = match (
            read("eng.lstm-word-dawg"),
            read("eng.lstm-punc-dawg"),
            read("eng.lstm-number-dawg"),
        ) {
            (Ok(word), Ok(punc), Ok(number)) => {
                match DictLite::from_components(&word, &punc, &number) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        eprintln!("warning: dict DAWGs present but failed to load ({e:?}); running without dict");
                        None
                    }
                }
            }
            _ => {
                eprintln!("note: dict DAWGs not found in model dir; running without dict beam");
                None
            }
        };

        // One permit per hardware thread: enough to keep every core busy under
        // load without letting a burst of uploads oversubscribe CPU + memory.
        let permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let recognize_permits = Arc::new(Semaphore::new(permits));

        Ok(Self {
            recognizer,
            dict,
            recognize_permits,
        })
    }
}
