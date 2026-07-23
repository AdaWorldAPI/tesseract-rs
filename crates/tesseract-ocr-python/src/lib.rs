//! `tesseract_rs` -- the Python SDK for `tesseract-rs`'s pure-Rust,
//! byte-parity Tesseract OCR transcode.
//!
//! Implements `docs/SDK-PYTHON-AND-POWER-PLATFORM.md` Section 1: a PyO3 +
//! maturin wheel wrapping [`tesseract_ogar::OcrExecutor`] -- **the only**
//! non-`pyo3` dependency this crate has (BBB-clean; no lance-graph engine,
//! no OGAR brain crates, no new recognition logic). Every method here is a
//! thin adapter: decode -> `OcrExecutor::execute` -> convert the response
//! into a native Python value.
//!
//! ```python
//! import tesseract_rs as ocr
//!
//! engine = ocr.Engine.from_model_dir("corpus/model", lang="deu")  # or "eng"
//! doc = engine.recognize_document(image_bytes)   # -> dict (doc.v1)
//! txt = engine.recognize_text(image_bytes)       # -> str
//! ```
//!
//! **Known gap:** [`Engine::searchable_pdf`] raises `NotImplementedError`.
//! See that method's doc comment and this crate's README -- it is a real
//! `tesseract-ogar` surface gap (missing type re-exports), not missing glue
//! code here.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList};
use pyo3::IntoPyObject;

use tesseract_ogar::{ImageDecodeError, OcrExecError, OcrExecutor, OcrRequest, OcrResponse};

/// Map an [`OcrExecError`] (component I/O, recognizer assembly, or a
/// recognize/render call failure) to a Python `RuntimeError`.
fn exec_err(e: OcrExecError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Map an [`ImageDecodeError`] (unreadable bytes, undecodable container, or
/// a decoded image outside the recognizer's size floor/budget) to a Python
/// `ValueError` -- the input, not the engine, is at fault.
fn decode_err(e: ImageDecodeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Recursively convert a `serde_json::Value` (the parsed `tesseract-rs/
/// doc.v1` tree) into a native Python object: a JSON object becomes a
/// `dict`, a JSON array becomes a `list`, scalars become the matching
/// Python scalar. This is what makes `doc["pages"][0]["regions"]` directly
/// iterable from Python instead of a JSON string the caller has to
/// `json.loads` themselves -- the design note in
/// `docs/SDK-PYTHON-AND-POWER-PLATFORM.md` Section 1 ("`doc.v1` bildet sich
/// natürlich auf ein Python-`dict` ab... Tabellen kommen als
/// `regions[].cells` durch -- direkt iterierbar").
///
/// Follows the exact scalar-conversion idiom already proven in this
/// workspace's `lance-graph-python::graph::json_to_python` (same
/// `IntoPyObject` + `.unbind().into()` pattern), extended here with genuine
/// recursion for arrays/objects instead of stringifying them -- `doc.v1`'s
/// nested regions/lines/cells need to stay real Python containers.
fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value as J;
    match value {
        J::Null => Ok(py.None()),
        J::Bool(b) => Ok(PyBool::new(py, *b).to_owned().into()),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.unbind().into())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.unbind().into())
            } else {
                Err(PyValueError::new_err(format!(
                    "tesseract-rs/doc.v1: unrepresentable JSON number: {n}"
                )))
            }
        }
        J::String(s) => Ok(s.as_str().into_pyobject(py)?.unbind().into()),
        J::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.unbind().into())
        }
        J::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.unbind().into())
        }
    }
}

/// The Python-facing OCR engine: a loaded [`OcrExecutor`] (recognizer
/// network/charset/recoder, plus an optional dictionary beam) ready to
/// recognize images handed in as raw bytes.
#[pyclass(module = "tesseract_rs", name = "Engine")]
struct Engine {
    executor: OcrExecutor,
}

#[pymethods]
impl Engine {
    /// `Engine.from_model_dir(dir, lang="eng")` -- load the recognizer's
    /// network/unicharset/recoder from `dir/{lang}.lstm{,-unicharset,
    /// -recoder}`, mirroring the on-disk layout of `corpus/model/{eng,
    /// deu}.lstm*` in this repo. If `dir/{lang}.lstm-{word,punc,number}-dawg`
    /// are ALL present, the dictionary beam is also loaded (mirrors
    /// [`OcrExecutor::from_data_paths`]'s all-or-nothing dict contract);
    /// otherwise recognition runs without a dictionary.
    #[staticmethod]
    #[pyo3(signature = (dir, lang=None))]
    fn from_model_dir(dir: &str, lang: Option<&str>) -> PyResult<Self> {
        let lang = lang.unwrap_or("eng");
        let base = std::path::Path::new(dir);
        let lstm = base.join(format!("{lang}.lstm"));
        let unicharset = base.join(format!("{lang}.lstm-unicharset"));
        let recoder = base.join(format!("{lang}.lstm-recoder"));
        let word_dawg = base.join(format!("{lang}.lstm-word-dawg"));
        let punc_dawg = base.join(format!("{lang}.lstm-punc-dawg"));
        let number_dawg = base.join(format!("{lang}.lstm-number-dawg"));

        let (word_dawg, punc_dawg, number_dawg) =
            if word_dawg.exists() && punc_dawg.exists() && number_dawg.exists() {
                (Some(word_dawg), Some(punc_dawg), Some(number_dawg))
            } else {
                (None, None, None)
            };

        let executor = OcrExecutor::from_data_paths(
            &lstm,
            &unicharset,
            &recoder,
            word_dawg.as_deref(),
            punc_dawg.as_deref(),
            number_dawg.as_deref(),
        )
        .map_err(exec_err)?;

        Ok(Self { executor })
    }

    /// `engine.recognize_document(image) -> dict` -- decode `image` (any
    /// pure-Rust-supported container: PNG / JPEG / WebP / TIFF / GIF / BMP /
    /// PNM) and run the one-shot document recognizer
    /// (`OcrRequest::RecognizeDocument`): classified regions (text / table /
    /// figure / header / footer), table cell grids, and page quality -- the
    /// full `tesseract-rs/doc.v1` JSON, parsed into a native Python `dict`
    /// (see [`json_to_py`]; never a raw JSON string).
    ///
    /// Runs without the dictionary beam and without a field-harvest profile
    /// (`with_dict=False`, `harvest_profile=None` on the underlying
    /// request) -- the design doc's Python signature takes only `image`, so
    /// neither is exposed as a parameter here.
    fn recognize_document(&self, py: Python<'_>, image: &[u8]) -> PyResult<Py<PyAny>> {
        let (grey, width, height) = tesseract_ogar::decode_image(image).map_err(decode_err)?;
        let response = self
            .executor
            .execute(OcrRequest::RecognizeDocument {
                grey: &grey,
                width,
                height,
                with_dict: false,
                harvest_profile: None,
            })
            .map_err(exec_err)?;
        let OcrResponse::DocumentOut { doc_json, .. } = response else {
            return Err(PyRuntimeError::new_err(
                "OcrExecutor::execute(RecognizeDocument) returned an unexpected OcrResponse variant",
            ));
        };
        let value: serde_json::Value = serde_json::from_str(&doc_json).map_err(|e| {
            PyRuntimeError::new_err(format!("tesseract-rs/doc.v1 JSON failed to parse: {e}"))
        })?;
        json_to_py(py, &value)
    }

    /// `engine.recognize_text(image) -> str` -- decode `image` and return
    /// the page's plain recognized text (`OcrRequest::RecognizePage` ->
    /// `OcrResponse::PageText.text`, the already `'\n'`-joined per-line
    /// text). Runs without the dictionary beam (`with_dict=False`), for the
    /// same reason as [`Engine::recognize_document`].
    fn recognize_text(&self, image: &[u8]) -> PyResult<String> {
        let (grey, width, height) = tesseract_ogar::decode_image(image).map_err(decode_err)?;
        let response = self
            .executor
            .execute(OcrRequest::RecognizePage {
                grey: &grey,
                width,
                height,
                with_dict: false,
            })
            .map_err(exec_err)?;
        let OcrResponse::PageText { text, .. } = response else {
            return Err(PyRuntimeError::new_err(
                "OcrExecutor::execute(RecognizePage) returned an unexpected OcrResponse variant",
            ));
        };
        Ok(text)
    }

    /// `engine.searchable_pdf(image) -> bytes` -- **not yet implemented.**
    ///
    /// Per `docs/SDK-PYTHON-AND-POWER-PLATFORM.md` Section 1, this should
    /// decode `image`, run `recognize_page_words`, and call
    /// `OcrRequest::RenderSearchablePdf { pages: &[PageOcr], dpi }`. It
    /// cannot be built from this crate's declared dependency
    /// (`tesseract-ogar` alone, `pyo3` aside) as things stand today:
    ///
    /// - `RenderSearchablePdf`'s `pages: &[PageOcr]` field needs a
    ///   freshly-built `tesseract_ocr_pdf::PageOcr` (a `GreyImage` plus a
    ///   `Vec<PlacedWord>` -- one `PlacedWord` per recognized word: text +
    ///   a top-down pixel box).
    /// - `tesseract-ogar` (`crates/tesseract-ogar/src/lib.rs`) re-exports
    ///   only `decode_image`/`ImageDecodeError` (behind `image-decode`); it
    ///   does NOT re-export `PageOcr`, `GreyImage`, or `PlacedWord`.
    /// - No `OcrResponse` variant ever hands one of those back either --
    ///   `OcrResponse::LineWordsOut(Vec<LineWords>)` is the closest, and
    ///   `LineWords`/`WordResult` carry bottom-up boxes, not the top-down
    ///   `PlacedWord::box_` shape `PageOcr` needs.
    ///
    /// So a crate depending on `tesseract-ogar` alone has no way to *name*
    /// -- let alone construct -- the type `RenderSearchablePdf` requires
    /// (Rust's extern prelude only exposes a crate's types to a DIRECT
    /// dependent; `tesseract-ogar` never re-exports these). This is a real
    /// gap in `tesseract-ogar`'s public surface, not a missing line of glue
    /// code here. Fixing it needs EITHER (a) `tesseract-ogar` re-exporting
    /// `PageOcr`/`GreyImage`/`PlacedWord` (plus the bottom-up -> top-down
    /// box conversion `tesseract_ocr::renderer::to_image_box` already
    /// does), OR (b) a new convenience method, e.g.
    /// `OcrExecutor::render_searchable_pdf_from_grey(grey, width, height,
    /// with_dict, dpi) -> Result<(Vec<u8>, RenderReport), OcrExecError>`,
    /// that builds `PageOcr` internally from a freshly recognized page and
    /// returns PDF bytes directly. See this crate's README ("Gaps").
    fn searchable_pdf(&self, _image: &[u8]) -> PyResult<Py<PyBytes>> {
        Err(PyNotImplementedError::new_err(
            "Engine.searchable_pdf: blocked on a tesseract-ogar gap, not a missing \
             implementation here -- OcrRequest::RenderSearchablePdf needs a freshly-built \
             PageOcr (tesseract_ocr_pdf::{PageOcr, GreyImage, PlacedWord}), and tesseract-ogar \
             (crates/tesseract-ogar/src/lib.rs) re-exports only decode_image/ImageDecodeError. \
             A crate depending on tesseract-ogar alone cannot name those types, so this \
             BBB-clean SDK cannot construct the request. See this crate's README ('Gaps') for \
             the exact missing re-export or convenience method needed.",
        ))
    }

    fn __repr__(&self) -> &'static str {
        "Engine(<tesseract-rs pure-Rust OCR executor>)"
    }
}

/// `tesseract_rs.decode_image(bytes) -> (grey, width, height)` -- the same
/// pure-Rust encoded-image decode [`Engine`]'s methods use internally
/// (`tesseract_ogar::decode_image`: PNG/JPEG/WebP/TIFF/GIF/BMP/PNM -> 8-bit
/// grey, row-major, bomb-bounded). Exposed standalone for callers who want
/// the raw grey buffer without running OCR.
#[pyfunction]
fn decode_image(py: Python<'_>, bytes: &[u8]) -> PyResult<(Py<PyBytes>, usize, usize)> {
    let (grey, width, height) = tesseract_ogar::decode_image(bytes).map_err(decode_err)?;
    Ok((PyBytes::new(py, &grey).unbind(), width, height))
}

#[pymodule]
fn tesseract_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
