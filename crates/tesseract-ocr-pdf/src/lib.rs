//! # tesseract-ocr-pdf — PDF front-end for the pure-Rust Tesseract transcode
//!
//! Phase 5 (D5.1 + D5.3-skeleton) of `.claude/plans/pdf-to-text-ocr-v1.md`.
//! This crate is **input tooling**, not part of the OCR runtime: it decides,
//! per page, whether a PDF already carries an extractable text layer (the
//! common case — a "digital" / not-scanned PDF) so OCR can be skipped
//! entirely, and otherwise hands a rasterized grey page to the proven
//! pure-Rust recognizer.
//!
//! ## The no-C++-in-OCR-runtime boundary
//!
//! [`lopdf`] is pure Rust, so using it here does not reintroduce a C++
//! dependency into the recognition path — it never touches
//! [`tesseract-core`]/[`tesseract-recognizer`]'s proven, byte-parity leaves.
//! The rule this crate preserves is specifically about the **OCR runtime**
//! (no leptonica, no libtesseract FFI in the pixel → text pipeline); PDF
//! *container* parsing predates and is orthogonal to that pipeline. The
//! eventual raster fallback (D5.2, `pdfium-render`) WOULD add a C++
//! dependency, but that is explicitly scoped as acceptable INPUT tooling per
//! the plan (`pdfium is a C++ dep — OK: it is input tooling, not the OCR
//! runtime`) — it produces the grey pixel buffer that is then handed,
//! unchanged, to the same pure-Rust [`tesseract_ocr::LstmRecognizer`] this
//! crate already drives for text-layer-less pages.
//!
//! ## D5.1 — text-layer fast path
//!
//! [`extract_text_layer`] returns, per page, `Some(text)` when the page has
//! a real text layer and `None` when it is image-only (a scanned page with
//! no text operators, or only whitespace). This is the policy gate: a page
//! that returns `Some` never needs OCR.
//!
//! ## D5.2 — scanned-page image extraction (pragmatic variant)
//!
//! [`extract_page_image`] finds the largest image XObject on a page and
//! decodes it to 8-bit grey — see [`image_extract`] for the full rationale
//! and filter/colour-space coverage matrix. This is the "pragmatic" D5.2:
//! it covers the common case (one full-page scanned image per page) without
//! a full content-stream-interpreting page rasterizer.
//!
//! ## D5.3 — orchestrator
//!
//! [`OcrPipeline`] wraps the proven recognizer + optional dictionary and
//! exposes [`OcrPipeline::ocr_grey_page`] for a pre-rasterized grey page
//! buffer. The `tesseract-ocr-pdf` binary wires [`extract_page_image`]'s
//! output straight into it for image-only pages.

use std::path::Path;

use lopdf::Document;
use tesseract_core::dawg::DawgError;
use tesseract_core::DictLite;
use tesseract_ocr::{LstmRecognizer, RecognizerError};

mod image_extract;
pub use image_extract::{extract_page_image, GreyImage};

/// Failures from the PDF text-layer fast path ([`extract_text_layer`]) and
/// the scanned-page image extraction path ([`extract_page_image`]).
#[derive(Debug)]
pub enum PdfError {
    /// `lopdf` failed to parse the PDF container.
    Load(lopdf::Error),
    /// `lopdf` failed to extract text from a specific page (font/encoding
    /// lookup failure, malformed content stream, ...). Carries the 1-based
    /// page number lopdf reports.
    Extract(lopdf::Error),
    /// `lopdf` failed to read the page's XObject dictionary or the chosen
    /// image stream (malformed dictionary, missing required key, dangling
    /// reference, ...).
    ImageObject(lopdf::Error),
    /// An image XObject's `Width`/`Height` couldn't be represented as
    /// `usize` on this platform (e.g. negative, which is spec-illegal).
    InvalidDimensions,
    /// An image XObject had no `/BitsPerComponent` entry.
    MissingBitsPerComponent,
    /// The image's `/Filter` is recognized but not implemented in this
    /// (D5.2 pragmatic-variant) module. Carries a human-readable filter
    /// name and, where useful, a pointer to the future leaf that would
    /// implement it.
    UnsupportedFilter(String),
    /// The image's colour space (or its combination with
    /// `/BitsPerComponent`) is recognized but not implemented in this
    /// module (e.g. `Indexed`, `ICCBased`, or an unsupported bit depth).
    UnsupportedColorSpace(String),
    /// The image uses a PDF feature this module deliberately does not
    /// implement in v1 (currently: `/SMask` soft-mask compositing).
    UnsupportedFeature(String),
    /// The image's `/Decode` array differs from the PDF default for its
    /// colour space/bit depth (PDF 32000-1:2008 §8.9.5.2, Table 90). Carries
    /// the array that was found.
    UnsupportedDecodeArray(String),
    /// The (decompressed, or raw) image sample buffer is shorter than
    /// `Width * Height * <components>` requires.
    TruncatedImageData {
        /// Expected minimum byte length.
        expected: usize,
        /// Actual byte length found.
        got: usize,
    },
    /// A `DCTDecode` (JPEG) stream failed to decode.
    Jpeg(zune_jpeg::errors::DecodeErrors),
    /// A decoded JPEG's dimensions disagree with the PDF image dictionary's
    /// `/Width`/`/Height`.
    JpegDimensionMismatch {
        /// `(width, height)` from the PDF image dictionary.
        pdf: (usize, usize),
        /// `(width, height)` reported by the JPEG decoder.
        jpeg: (usize, usize),
    },
    /// A decoded JPEG's output colour space is neither `Luma` nor `RGB`
    /// (e.g. `CMYK`/`YCCK`), which this module does not convert to grey.
    UnsupportedJpegColorspace(String),
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(e) => write!(f, "PDF load: {e}"),
            Self::Extract(e) => write!(f, "PDF text extraction: {e}"),
            Self::ImageObject(e) => write!(f, "PDF image XObject: {e}"),
            Self::InvalidDimensions => write!(f, "image XObject has invalid (negative) dimensions"),
            Self::MissingBitsPerComponent => write!(f, "image XObject has no /BitsPerComponent"),
            Self::UnsupportedFilter(name) => write!(f, "unsupported image filter: {name}"),
            Self::UnsupportedColorSpace(cs) => write!(f, "unsupported image colour space: {cs}"),
            Self::UnsupportedFeature(feature) => write!(f, "unsupported PDF feature: {feature}"),
            Self::UnsupportedDecodeArray(arr) => {
                write!(f, "unsupported (non-default) /Decode array: {arr}")
            }
            Self::TruncatedImageData { expected, got } => write!(
                f,
                "image sample data too short: expected at least {expected} bytes, got {got}"
            ),
            Self::Jpeg(e) => write!(f, "JPEG (DCTDecode) decode: {e}"),
            Self::JpegDimensionMismatch { pdf, jpeg } => write!(
                f,
                "JPEG dimensions {}x{} disagree with PDF image dictionary {}x{}",
                jpeg.0, jpeg.1, pdf.0, pdf.1
            ),
            Self::UnsupportedJpegColorspace(cs) => {
                write!(f, "unsupported JPEG output colour space: {cs}")
            }
        }
    }
}

impl std::error::Error for PdfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load(e) | Self::Extract(e) | Self::ImageObject(e) => Some(e),
            Self::Jpeg(e) => Some(e),
            Self::InvalidDimensions
            | Self::MissingBitsPerComponent
            | Self::UnsupportedFilter(_)
            | Self::UnsupportedColorSpace(_)
            | Self::UnsupportedFeature(_)
            | Self::UnsupportedDecodeArray(_)
            | Self::TruncatedImageData { .. }
            | Self::JpegDimensionMismatch { .. }
            | Self::UnsupportedJpegColorspace(_) => None,
        }
    }
}

/// Extract the per-page text layer of a PDF (D5.1).
///
/// For each page (in PDF page order), returns `Some(text)` when the page's
/// content stream contains a non-whitespace text layer, or `None` when the
/// page is image-only (a scanned page: no `Tj`/`TJ` operators, or the
/// decoded text is all whitespace). A `None` page is the signal to fall back
/// to OCR (D5.2/D5.3); a `Some` page never needs OCR.
///
/// # Errors
///
/// [`PdfError::Load`] if the byte stream is not a parseable PDF;
/// [`PdfError::Extract`] if a specific page's content stream cannot be
/// decoded (this is a hard error, not classified as "image-only" — a
/// genuinely image-only page decodes fine and simply yields no text).
pub fn extract_text_layer(pdf_bytes: &[u8]) -> Result<Vec<Option<String>>, PdfError> {
    let doc = Document::load_mem(pdf_bytes).map_err(PdfError::Load)?;
    let pages = doc.get_pages();
    let mut out = Vec::with_capacity(pages.len());
    for &page_number in pages.keys() {
        let text = doc
            .extract_text(&[page_number])
            .map_err(PdfError::Extract)?;
        out.push(if text.trim().is_empty() {
            None
        } else {
            Some(text)
        });
    }
    Ok(out)
}

/// Failures assembling or running [`OcrPipeline`].
#[derive(Debug)]
pub enum PipelineError {
    /// A component file (network/unicharset/recoder/dawg) could not be read.
    Io(std::path::PathBuf, std::io::Error),
    /// The recognizer failed to assemble from its components.
    Recognizer(RecognizerError),
    /// The dictionary failed to assemble from its DAWG components.
    Dawg(DawgError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "reading {}: {e}", path.display()),
            Self::Recognizer(e) => write!(f, "assembling recognizer: {e}"),
            Self::Dawg(e) => write!(f, "assembling dictionary: {e:?}"),
        }
    }
}

impl std::error::Error for PipelineError {}

fn read_component(path: &Path) -> Result<Vec<u8>, PipelineError> {
    std::fs::read(path).map_err(|e| PipelineError::Io(path.to_path_buf(), e))
}

fn read_component_text(path: &Path) -> Result<String, PipelineError> {
    std::fs::read_to_string(path).map_err(|e| PipelineError::Io(path.to_path_buf(), e))
}

/// The OCR arm of the orchestrator (D5.3-skeleton): a fully assembled
/// pure-Rust recognizer + optional dictionary, ready to recognize a
/// pre-rasterized grey page. Raster production itself (PDF page → grey
/// pixels) is D5.2, not part of this pipeline — callers supply the grey
/// buffer (e.g. from a `.pgm`, or, once D5.2 lands, from `pdfium-render`).
pub struct OcrPipeline {
    recognizer: LstmRecognizer,
    dict: Option<DictLite>,
}

impl OcrPipeline {
    /// Load the recognizer network/charset/recoder and, if all three DAWG
    /// paths are given, the word/punctuation/number dictionary, from files
    /// on disk. Mirrors the component loading in
    /// `tesseract-ocr/examples/recognize_words_dump.rs`.
    ///
    /// # Errors
    ///
    /// [`PipelineError::Io`] if any component file cannot be read;
    /// [`PipelineError::Recognizer`] if the network/charset/recoder fail to
    /// assemble; [`PipelineError::Dawg`] if the dictionary DAWGs fail to
    /// assemble.
    pub fn from_data_paths(
        lstm: &Path,
        unicharset: &Path,
        recoder: &Path,
        word_dawg: Option<&Path>,
        punc_dawg: Option<&Path>,
        number_dawg: Option<&Path>,
    ) -> Result<Self, PipelineError> {
        let lstm_bytes = read_component(lstm)?;
        let uni_text = read_component_text(unicharset)?;
        let rec_bytes = read_component(recoder)?;
        let recognizer = LstmRecognizer::from_components(&lstm_bytes, &uni_text, &rec_bytes)
            .map_err(PipelineError::Recognizer)?;

        let dict = match (word_dawg, punc_dawg, number_dawg) {
            (Some(w), Some(p), Some(n)) => {
                let word = read_component(w)?;
                let punc = read_component(p)?;
                let number = read_component(n)?;
                let dict = DictLite::from_components(&word, &punc, &number)
                    .map_err(PipelineError::Dawg)?;
                Some(dict)
            }
            _ => None,
        };

        Ok(Self { recognizer, dict })
    }

    /// Recognize a single pre-rasterized grey page (D3.0 line-segmentation
    /// composition, `seg-approx`; see [`LstmRecognizer::recognize_page`] for
    /// the approximation-vs-transcode scope). `grey` is a row-major 8-bit
    /// grey buffer, `w`×`h` pixels.
    ///
    /// # Errors
    ///
    /// [`RecognizerError`] from the underlying recognizer, if any line band
    /// fails to recognize.
    pub fn ocr_grey_page(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
    ) -> Result<String, RecognizerError> {
        self.recognizer
            .recognize_page(grey, w, h, self.dict.as_ref())
    }
}
