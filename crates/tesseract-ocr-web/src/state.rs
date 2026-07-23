//! Shared, read-only application state: the loaded recognizer(s) + dictionary.
//!
//! `LstmRecognizer` and `DictLite` are both plain owned data (no interior
//! mutability), so `&AppState` is `Send + Sync` and can be shared across all
//! request handlers via `Arc` — recognition is `&self`, never mutating.

use std::path::Path;
use std::sync::Arc;

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;
use tokio::sync::Semaphore;

/// One loaded language's model tissue: the recognizer plus its optional dict
/// beam DAWGs.
pub struct LangModel {
    /// The pure-Rust LSTM recognizer (network + charset + recoder).
    pub recognizer: LstmRecognizer,
    /// The dictionary DAWGs for the production dict beam (optional).
    pub dict: Option<DictLite>,
}

/// The model tissue, loaded once at startup and shared read-only.
pub struct AppState {
    /// `eng` — the default model, REQUIRED (startup fails without it, exactly
    /// as before this type existed).
    pub eng: LangModel,
    /// `deu` — OPTIONAL, same graceful-degrade rule as the dict DAWGs: if
    /// `deu.lstm*` is absent from `model_dir`, requests for `lang=deu` just
    /// fall back to `eng` (see [`AppState::model`]) rather than failing
    /// startup over a language nobody may ask for.
    pub deu: Option<LangModel>,
    /// Bounds how many CPU-bound recognitions run at once, across BOTH
    /// languages (one shared throughput cap, not one per language).
    /// Recognition is heavy synchronous work dispatched via `spawn_blocking`;
    /// without a cap a burst of uploads would flood the blocking pool and the
    /// box's memory. Sized to the machine's parallelism so we saturate cores
    /// without oversubscribing.
    pub recognize_permits: Arc<Semaphore>,
}

/// Load one language's `{lang}.lstm*` components from `model_dir`. `Ok(None)`
/// means the REQUIRED trio (`.lstm`/`.lstm-unicharset`/`.lstm-recoder`) is
/// simply absent — the graceful-degrade case for an optional language; any
/// other failure (present-but-unreadable/corrupt) is a hard `Err`, same as a
/// required-language load.
fn try_load_lang(model_dir: &Path, lang: &str) -> Result<Option<LangModel>, String> {
    let path = |name: &str| model_dir.join(format!("{lang}.{name}"));
    if !path("lstm").exists() {
        return Ok(None);
    }
    let read = |name: &str| -> Result<Vec<u8>, String> {
        std::fs::read(path(name)).map_err(|e| format!("read {}: {e}", path(name).display()))
    };
    let lstm = read("lstm")?;
    let uni = String::from_utf8(read("lstm-unicharset")?)
        .map_err(|e| format!("{lang}.lstm-unicharset is not UTF-8: {e}"))?;
    let rec = read("lstm-recoder")?;
    let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
        .map_err(|e| format!("assemble {lang} recognizer: {e}"))?;

    // The dict is optional: without the DAWGs the recognizer still runs
    // (plain beam), so a missing dawg degrades gracefully rather than
    // failing startup.
    let dict = match (
        read("lstm-word-dawg"),
        read("lstm-punc-dawg"),
        read("lstm-number-dawg"),
    ) {
        (Ok(word), Ok(punc), Ok(number)) => {
            match DictLite::from_components(&word, &punc, &number) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("warning: {lang} dict DAWGs present but failed to load ({e:?}); running without dict");
                    None
                }
            }
        }
        _ => {
            eprintln!("note: {lang} dict DAWGs not found in model dir; running without dict beam");
            None
        }
    };

    Ok(Some(LangModel { recognizer, dict }))
}

impl AppState {
    /// Load `eng` (required) and `deu` (optional) from `model_dir`. Returns a
    /// human-readable `Err` on any failure loading the required `eng` model
    /// (the caller prints it and exits — a server with no model is useless);
    /// a missing/corrupt `deu` model is logged and degrades gracefully (every
    /// `lang=deu` request falls back to `eng` — see [`AppState::model`]).
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let eng = try_load_lang(model_dir, "eng")?.ok_or_else(|| {
            format!(
                "eng.lstm not found in {} (eng is required)",
                model_dir.display()
            )
        })?;
        let deu = match try_load_lang(model_dir, "deu") {
            Ok(m) => m,
            Err(e) => {
                eprintln!("warning: deu model present but failed to load ({e}); lang=deu will fall back to eng");
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
            eng,
            deu,
            recognize_permits,
        })
    }

    /// Select the model for a request's `lang` field. `Some("deu")` returns
    /// the German model IF it loaded; every other case — `None`, `Some("eng")`,
    /// an unrecognized value, or `Some("deu")` when `deu` failed to load —
    /// falls back to `eng`. Never errors: an unrecognized/unavailable language
    /// is not a request worth rejecting, the same "forgiving field" rule
    /// [`crate::ocr::OutputFormat::from_field`] already uses. Returns the
    /// canonical code actually selected (`"eng"` or `"deu"`) alongside the
    /// model, so callers can report the truth even when they fell back.
    #[must_use]
    pub fn model(&self, lang: Option<&str>) -> (&'static str, &LangModel) {
        match (lang, &self.deu) {
            (Some("deu"), Some(m)) => ("deu", m),
            _ => ("eng", &self.eng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model")
    }

    /// `AppState::load` picks up `deu.*` from `corpus/model` alongside `eng.*`
    /// — both models loaded, distinguishable by their real, different
    /// `null_char` (`E-OCR-DEU-PARITY-MODEL-AGNOSTIC-1`: eng=110, deu=114 —
    /// the German model self-derives different constants, not a copy of eng's).
    #[test]
    fn load_picks_up_both_eng_and_deu_from_corpus_model() {
        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = AppState::load(&dir).expect("load model");
        assert_eq!(state.eng.recognizer.null_char, 110);
        if dir.join("deu.lstm").exists() {
            let deu = state.deu.as_ref().expect("deu.lstm present but not loaded");
            assert_eq!(deu.recognizer.null_char, 114);
        } else {
            assert!(state.deu.is_none(), "no deu.lstm on disk => deu unloaded");
        }
    }

    /// [`AppState::model`]'s selection rule: `Some("deu")` selects German
    /// (when loaded); `None`, `Some("eng")`, and an unrecognized value all
    /// fall back to English. Never errors on an unrecognized language.
    #[test]
    fn model_selects_deu_only_on_exact_match_else_falls_back_to_eng() {
        let dir = model_dir();
        if !dir.join("eng.lstm").exists() || !dir.join("deu.lstm").exists() {
            eprintln!("skipping: eng/deu corpus model absent");
            return;
        }
        let state = AppState::load(&dir).expect("load model");

        let (code, m) = state.model(None);
        assert_eq!(code, "eng");
        assert_eq!(m.recognizer.null_char, 110);

        let (code, m) = state.model(Some("eng"));
        assert_eq!(code, "eng");
        assert_eq!(m.recognizer.null_char, 110);

        let (code, m) = state.model(Some("deu"));
        assert_eq!(code, "deu");
        assert_eq!(m.recognizer.null_char, 114);

        // Forgiving field: an unrecognized value is not a request worth
        // rejecting, same rule as OutputFormat::from_field.
        let (code, m) = state.model(Some("klingon"));
        assert_eq!(code, "eng");
        assert_eq!(m.recognizer.null_char, 110);
    }
}
