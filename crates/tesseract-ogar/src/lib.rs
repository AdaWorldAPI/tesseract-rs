//! # tesseract-ogar — the in-binary executor for the OGAR OCR action table
//!
//! [`ogar_vocab::ocr_actions`] is the **authoritative** declaration of the
//! eight OCR capabilities `tesseract-rs` exposes (`recognize_line` /
//! `recognize_page` / `extract_text_layer` / `extract_page_image` /
//! `render_text` / `render_tsv` / `render_hocr` / `render_searchable_pdf`).
//! This crate is that table's executor: OGAR declares, this crate runs.
//!
//! ## Typed, not serialized
//!
//! Every consumer of this crate lives in the SAME binary as OGAR and the two
//! OCR foundations (`tesseract-core`, `tesseract-recognizer`) — there is no
//! process boundary here, so [`OcrRequest`]/[`OcrResponse`] are plain Rust
//! enums, not a wire DTO. No `serde`, no JSON, no schema round-trip: the
//! "OpenAPI-shaped" surface (one request type per declared capability, a
//! matching typed response) exists so a caller gets the SAME shape an
//! external API would advertise, but every call is a monomorphized function
//! call in-process — the operator's framing: "wie OpenAPI aussieht, aber in
//! der gleichen Binary ohne serde auskommt."
//!
//! ## The exhaustiveness fuse
//!
//! [`OCR_ACTION_NAMES`](ogar_vocab::ocr_actions::OCR_ACTION_NAMES) is OGAR's
//! `const`-evaluable fingerprint of the declared capability names.
//! [`COVERED_CAPABILITIES`] is this crate's own fingerprint of what
//! [`OcrExecutor::execute`] handles. A `const` assertion below pins their
//! *lengths* equal at compile time (a cheap, allocation-free tripwire); the
//! `every_declared_capability_is_covered_and_vice_versa` test in this
//! crate's test module pins the actual *names* equal, in both directions —
//! so a capability added to OGAR without a matching `OcrRequest` arm here
//! fails the test, and a capability removed from OGAR without pruning this
//! crate's coverage also fails it.
//!
//! ## Drift = build/test failure, not a runtime surprise
//!
//! This crate never re-implements OCR logic — every [`OcrExecutor::execute`]
//! arm is a thin dispatch onto the proven [`tesseract_ocr`]/
//! [`tesseract_ocr_pdf`] public API. The value this crate adds is the
//! *join*: proving, at compile time and test time, that the declared
//! capability table and the actual executable surface never diverge.

use std::path::Path;

use tesseract_core::dawg::DawgError;
use tesseract_core::DictLite;
use tesseract_ocr::{LineWords, LstmRecognizer};
use tesseract_ocr_pdf::{GreyImage, PageOcr, PdfError, RenderReport, SearchablePdfError};

/// Every OCR capability this crate's [`OcrExecutor::execute`] handles, in the
/// same order as [`ogar_vocab::ocr_actions::OCR_ACTION_NAMES`] — this
/// crate's half of the exhaustiveness fuse (see the module docs).
pub const COVERED_CAPABILITIES: &[&str] = &[
    "recognize_line",
    "recognize_page",
    "extract_text_layer",
    "extract_page_image",
    "render_text",
    "render_tsv",
    "render_hocr",
    "render_searchable_pdf",
];

/// This crate's hot-plug declaration — the GENERIC pattern every consumer
/// migrates to (operator, 2026-07-07): one const naming the classids this
/// executor hot-plugs and the capabilities it covers. The authority
/// (`ogar_vocab::capability_registry::resolve_hotplug`, reachable through
/// the `lance_graph_contract::hotplug::CapabilityAuthority` socket) verifies
/// the plug and returns BOTH the vocab rows and the action surface for
/// exactly these classids — classid is the join key on both sides. Drift
/// bangs once, in this binary, no serialization, no per-consumer plug
/// mechanism beyond this const.
pub const HOT_PLUG: lance_graph_contract::hotplug::HotPlug =
    lance_graph_contract::hotplug::HotPlug {
        consumer: "tesseract-ogar",
        classids: ogar_vocab::ocr_actions::OCR_SUBJECT_CLASSIDS,
        covered: COVERED_CAPABILITIES,
    };

// The cheap, allocation-free half of the fuse: OGAR's `OCR_ACTION_NAMES` and
// this crate's `COVERED_CAPABILITIES` must have the same length at compile
// time. This does NOT check the actual names (that needs `ActionDef`, which
// isn't `const`-constructible — see `ogar_vocab::ocr_actions`'s module doc,
// "why a `fn`, not a `const`") — the name-level check is the
// `every_declared_capability_is_covered_and_vice_versa` test below.
const _: () = assert!(
    ogar_vocab::ocr_actions::OCR_ACTION_NAMES.len() == COVERED_CAPABILITIES.len(),
    "tesseract-ogar::COVERED_CAPABILITIES has drifted from ogar_vocab::ocr_actions::OCR_ACTION_NAMES's length"
);

/// One typed request per declared OGAR OCR capability. Plain Rust types,
/// zero serialization — see the module docs.
#[derive(Debug, Clone, Copy)]
pub enum OcrRequest<'a> {
    /// `recognize_line` — a single pre-cropped grey text-line strip.
    /// `grey` is row-major 8-bit, `width`×`height` pixels. `with_dict`
    /// selects the dictionary-beam decode when this executor was assembled
    /// with a dictionary (see [`OcrExecutor::from_data_paths`]); it is
    /// silently equivalent to `false` when no dictionary was loaded.
    RecognizeLine {
        /// Row-major 8-bit grey line strip.
        grey: &'a [u8],
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Use the loaded dictionary beam, if any.
        with_dict: bool,
    },
    /// `recognize_page` — a full grey page, segmented into line bands via
    /// the `seg-approx` projection-profile finder (see
    /// [`tesseract_ocr::LstmRecognizer::recognize_page`] for the
    /// approximation-vs-transcode scope).
    RecognizePage {
        /// Row-major 8-bit grey page.
        grey: &'a [u8],
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Use the loaded dictionary beam, if any.
        with_dict: bool,
    },
    /// `extract_text_layer` — the D5.1 fast path: per-page `Some(text)`/
    /// `None` classification of a digital PDF's existing text layer.
    ExtractTextLayer {
        /// The PDF file's raw bytes.
        pdf_bytes: &'a [u8],
    },
    /// `extract_page_image` — the D5.2 pragmatic scanned-page image
    /// extraction (largest image XObject on the page, decoded to grey).
    ExtractPageImage {
        /// The PDF file's raw bytes.
        pdf_bytes: &'a [u8],
        /// 1-based page number (matches [`tesseract_ocr_pdf::extract_page_image`]).
        page: u32,
    },
    /// `render_text` — plain-text join of already-recognized line/word
    /// output (`ResultIterator::IterateAndAppendUTF8TextlineText` transcode).
    RenderText {
        /// Recognized lines, in reading order.
        lines: &'a [LineWords],
    },
    /// `render_tsv` — Tesseract TSV rendering of already-recognized
    /// line/word output.
    RenderTsv {
        /// Recognized lines, in reading order.
        lines: &'a [LineWords],
        /// Page width in pixels.
        page_w: u32,
        /// Page height in pixels.
        page_h: u32,
    },
    /// `render_hocr` — hOCR rendering of already-recognized line/word
    /// output.
    RenderHocr {
        /// Recognized lines, in reading order.
        lines: &'a [LineWords],
        /// Page width in pixels.
        page_w: u32,
        /// Page height in pixels.
        page_h: u32,
        /// The `<title>`/`ocr_page` image file name to embed.
        image_name: &'a str,
    },
    /// `render_searchable_pdf` — the D4.5 invisible-text-layer searchable
    /// PDF assembly, one or more OCR'd pages.
    RenderSearchablePdf {
        /// One entry per output page.
        pages: &'a [PageOcr],
        /// The embedded image resolution, in DPI.
        dpi: u32,
    },
}

/// One typed response per declared OGAR OCR capability — see
/// [`OcrRequest`] for the matching request shape and
/// [`ogar_vocab::ocr_actions::OcrActionSpec::produces`] for the declared
/// output names each variant below corresponds to.
#[derive(Debug, Clone, PartialEq)]
pub enum OcrResponse {
    /// `recognize_line`'s `text, unichar_ids` outputs.
    Recognized {
        /// Recognized unichar ids, in reading order.
        unichar_ids: Vec<u32>,
        /// Recognized text.
        text: String,
    },
    /// `recognize_page`'s `textlines, text` outputs. `textlines` is derived
    /// from `text` by splitting on `'\n'` and dropping empty entries — a
    /// lossless recovery, since [`tesseract_ocr::LstmRecognizer::recognize_page`]
    /// itself builds `text` by `'\n'`-joining exactly the non-empty per-line
    /// results (see that method's doc comment), and no single recognized
    /// line ever contains an internal `'\n'`.
    PageText {
        /// The page's text, split back into per-line strings.
        textlines: Vec<String>,
        /// The whole page's text (lines joined by `'\n'`).
        text: String,
    },
    /// `extract_text_layer`'s `page_texts` output — one entry per page,
    /// `None` for an image-only page.
    PageTexts(Vec<Option<String>>),
    /// `extract_page_image`'s `grey_image` output — `None` when the page
    /// has no (supported) image XObject.
    GreyImage(Option<GreyImage>),
    /// `render_text`'s `text` output.
    Text(String),
    /// `render_tsv`'s `tsv` output.
    Tsv(String),
    /// `render_hocr`'s `hocr` output.
    Hocr(String),
    /// `render_searchable_pdf`'s `pdf_bytes` output, plus the WinAnsi
    /// substitution [`RenderReport`] the underlying function also returns
    /// (not part of OGAR's declared `produces`, but free diagnostic data
    /// from the same call — carrying it costs nothing and drops nothing).
    PdfBytes {
        /// The assembled PDF's raw bytes.
        bytes: Vec<u8>,
        /// Per-page WinAnsi lossy-substitution counts.
        report: RenderReport,
    },
}

/// A failure loading [`OcrExecutor`] or executing an [`OcrRequest`].
#[derive(Debug)]
pub enum OcrExecError {
    /// A component file (network/unicharset/recoder/dawg) could not be read.
    Io(std::path::PathBuf, std::io::Error),
    /// The recognizer failed to assemble from its components, or a
    /// recognize/render call into [`tesseract_ocr`] failed.
    Recognizer(tesseract_ocr::RecognizerError),
    /// The dictionary failed to assemble from its DAWG components.
    Dawg(DawgError),
    /// A PDF-facing call into [`tesseract_ocr_pdf`] failed.
    Pdf(PdfError),
    /// [`tesseract_ocr_pdf::render_searchable_pdf`] failed.
    SearchablePdf(SearchablePdfError),
}

impl std::fmt::Display for OcrExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "reading {}: {e}", path.display()),
            Self::Recognizer(e) => write!(f, "recognizer: {e}"),
            Self::Dawg(e) => write!(f, "dictionary assembly: {e:?}"),
            Self::Pdf(e) => write!(f, "PDF: {e}"),
            Self::SearchablePdf(e) => write!(f, "searchable PDF render: {e}"),
        }
    }
}

impl std::error::Error for OcrExecError {}

fn read_component(path: &Path) -> Result<Vec<u8>, OcrExecError> {
    std::fs::read(path).map_err(|e| OcrExecError::Io(path.to_path_buf(), e))
}

fn read_component_text(path: &Path) -> Result<String, OcrExecError> {
    std::fs::read_to_string(path).map_err(|e| OcrExecError::Io(path.to_path_buf(), e))
}

/// The in-binary executor: a loaded pure-Rust recognizer (+ optional
/// dictionary), ready to dispatch any [`OcrRequest`] to its matching
/// [`tesseract_ocr`]/[`tesseract_ocr_pdf`] call.
#[derive(Debug)]
pub struct OcrExecutor {
    recognizer: LstmRecognizer,
    dict: Option<DictLite>,
}

impl OcrExecutor {
    /// Load the recognizer network/charset/recoder and, if all three DAWG
    /// paths are given, the word/punctuation/number dictionary, from files
    /// on disk. Mirrors [`tesseract_ocr_pdf::OcrPipeline::from_data_paths`]
    /// (this crate cannot reuse that type directly — its `recognizer`/`dict`
    /// fields are private — so the loading is repeated here against the
    /// same public component-loading API).
    ///
    /// # Errors
    ///
    /// [`OcrExecError::Io`] if any component file cannot be read;
    /// [`OcrExecError::Recognizer`] if the network/charset/recoder fail to
    /// assemble; [`OcrExecError::Dawg`] if the dictionary DAWGs fail to
    /// assemble.
    pub fn from_data_paths(
        lstm: &Path,
        unicharset: &Path,
        recoder: &Path,
        word_dawg: Option<&Path>,
        punc_dawg: Option<&Path>,
        number_dawg: Option<&Path>,
    ) -> Result<Self, OcrExecError> {
        let lstm_bytes = read_component(lstm)?;
        let uni_text = read_component_text(unicharset)?;
        let rec_bytes = read_component(recoder)?;
        let recognizer = LstmRecognizer::from_components(&lstm_bytes, &uni_text, &rec_bytes)
            .map_err(OcrExecError::Recognizer)?;

        let dict = match (word_dawg, punc_dawg, number_dawg) {
            (Some(w), Some(p), Some(n)) => {
                let word = read_component(w)?;
                let punc = read_component(p)?;
                let number = read_component(n)?;
                let dict =
                    DictLite::from_components(&word, &punc, &number).map_err(OcrExecError::Dawg)?;
                Some(dict)
            }
            _ => None,
        };

        Ok(Self { recognizer, dict })
    }

    /// The declared capability name this request implements — the join key
    /// to [`ogar_vocab::ocr_actions::ocr_actions`]'s `def.predicate`.
    #[must_use]
    pub fn capability_of(req: &OcrRequest<'_>) -> &'static str {
        match req {
            OcrRequest::RecognizeLine { .. } => "recognize_line",
            OcrRequest::RecognizePage { .. } => "recognize_page",
            OcrRequest::ExtractTextLayer { .. } => "extract_text_layer",
            OcrRequest::ExtractPageImage { .. } => "extract_page_image",
            OcrRequest::RenderText { .. } => "render_text",
            OcrRequest::RenderTsv { .. } => "render_tsv",
            OcrRequest::RenderHocr { .. } => "render_hocr",
            OcrRequest::RenderSearchablePdf { .. } => "render_searchable_pdf",
        }
    }

    /// Execute one [`OcrRequest`], dispatching to the matching proven
    /// [`tesseract_ocr`]/[`tesseract_ocr_pdf`] call. Pure dispatch — no
    /// logic beyond adapting parameter/return shapes lives here.
    ///
    /// # Errors
    ///
    /// [`OcrExecError`] from the underlying recognizer/PDF/render call.
    pub fn execute(&self, req: OcrRequest<'_>) -> Result<OcrResponse, OcrExecError> {
        match req {
            OcrRequest::RecognizeLine {
                grey,
                width,
                height,
                with_dict,
            } => {
                let dict = if with_dict { self.dict.clone() } else { None };
                let (unichar_ids, text) = self
                    .recognizer
                    .recognize_grey_line(grey, width, height, dict)
                    .map_err(OcrExecError::Recognizer)?;
                let unichar_ids = unichar_ids.into_iter().map(|id| id as u32).collect();
                Ok(OcrResponse::Recognized { unichar_ids, text })
            }
            OcrRequest::RecognizePage {
                grey,
                width,
                height,
                with_dict,
            } => {
                let dict = if with_dict { self.dict.as_ref() } else { None };
                let text = self
                    .recognizer
                    .recognize_page(grey, width, height, dict)
                    .map_err(OcrExecError::Recognizer)?;
                let textlines = text
                    .split('\n')
                    .filter(|line| !line.is_empty())
                    .map(str::to_owned)
                    .collect();
                Ok(OcrResponse::PageText { textlines, text })
            }
            OcrRequest::ExtractTextLayer { pdf_bytes } => {
                let page_texts =
                    tesseract_ocr_pdf::extract_text_layer(pdf_bytes).map_err(OcrExecError::Pdf)?;
                Ok(OcrResponse::PageTexts(page_texts))
            }
            OcrRequest::ExtractPageImage { pdf_bytes, page } => {
                let image = tesseract_ocr_pdf::extract_page_image(pdf_bytes, page)
                    .map_err(OcrExecError::Pdf)?;
                Ok(OcrResponse::GreyImage(image))
            }
            OcrRequest::RenderText { lines } => Ok(OcrResponse::Text(tesseract_ocr::render_text(
                lines,
                &self.recognizer.charset,
            ))),
            OcrRequest::RenderTsv {
                lines,
                page_w,
                page_h,
            } => Ok(OcrResponse::Tsv(tesseract_ocr::render_tsv(
                lines,
                &self.recognizer.charset,
                page_w,
                page_h,
            ))),
            OcrRequest::RenderHocr {
                lines,
                page_w,
                page_h,
                image_name,
            } => Ok(OcrResponse::Hocr(tesseract_ocr::render_hocr(
                lines,
                &self.recognizer.charset,
                page_w,
                page_h,
                image_name,
            ))),
            OcrRequest::RenderSearchablePdf { pages, dpi } => {
                let (bytes, report) = tesseract_ocr_pdf::render_searchable_pdf(pages, dpi)
                    .map_err(OcrExecError::SearchablePdf)?;
                Ok(OcrResponse::PdfBytes { bytes, report })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// The full confirmation loop, closed generically: this crate's
    /// [`HOT_PLUG`] resolves through the authority — every hot-plugged
    /// classid minted and capability-bearing, consumer expected, coverage
    /// both directions — and returns BOTH the vocab rows and the action
    /// surface. Any drift is a NAMED `HotplugDrift` arm failing this test.
    #[test]
    fn hotplug_activation_is_green() {
        let (concepts, capabilities) = ogar_vocab::capability_registry::resolve_hotplug(
            HOT_PLUG.consumer,
            HOT_PLUG.classids,
            HOT_PLUG.covered,
        )
        .expect("hot-plug drifted from the authoritative OGAR tables");
        assert_eq!(concepts.len(), 3, "3 hot-plugged concepts");
        assert!(concepts.contains(&("textline", 0x0805)));
        assert_eq!(capabilities.len(), COVERED_CAPABILITIES.len());
    }

    /// Both directions of the exhaustiveness fuse: every capability OGAR
    /// declares is covered here, AND every capability this crate claims to
    /// cover is actually declared in OGAR. A capability added upstream
    /// without a matching arm fails the first direction; a capability
    /// removed upstream without pruning `COVERED_CAPABILITIES` fails the
    /// second.
    #[test]
    fn every_declared_capability_is_covered_and_vice_versa() {
        let actions = ogar_vocab::ocr_actions::ocr_actions();
        let declared: BTreeSet<&str> = actions.iter().map(|s| s.def.predicate.as_str()).collect();
        let covered: BTreeSet<&str> = COVERED_CAPABILITIES.iter().copied().collect();
        assert_eq!(
            declared, covered,
            "tesseract-ogar coverage has drifted from ogar_vocab::ocr_actions"
        );
    }

    /// `capability_of` on one sample request per variant always returns a
    /// name OGAR actually declared.
    #[test]
    fn capability_of_matches_declared_names_for_each_variant() {
        let declared: BTreeSet<&str> = ogar_vocab::ocr_actions::OCR_ACTION_NAMES
            .iter()
            .copied()
            .collect();
        let samples: Vec<OcrRequest<'_>> = vec![
            OcrRequest::RecognizeLine {
                grey: &[],
                width: 0,
                height: 0,
                with_dict: false,
            },
            OcrRequest::RecognizePage {
                grey: &[],
                width: 0,
                height: 0,
                with_dict: false,
            },
            OcrRequest::ExtractTextLayer { pdf_bytes: &[] },
            OcrRequest::ExtractPageImage {
                pdf_bytes: &[],
                page: 1,
            },
            OcrRequest::RenderText { lines: &[] },
            OcrRequest::RenderTsv {
                lines: &[],
                page_w: 0,
                page_h: 0,
            },
            OcrRequest::RenderHocr {
                lines: &[],
                page_w: 0,
                page_h: 0,
                image_name: "",
            },
            OcrRequest::RenderSearchablePdf { pages: &[], dpi: 0 },
        ];
        assert_eq!(
            samples.len(),
            COVERED_CAPABILITIES.len(),
            "one sample per covered capability"
        );
        for req in &samples {
            let cap = OcrExecutor::capability_of(req);
            assert!(
                declared.contains(cap),
                "capability_of returned undeclared name: {cap}"
            );
        }
    }

    /// Per-capability mapping from an OGAR-declared param name to the
    /// corresponding [`OcrRequest`] field name. Rust has no runtime
    /// enum-variant field-name reflection, so this table IS the assertion
    /// that our field naming matches OGAR's naming (or knowingly diverges,
    /// e.g. `grey_line`/`grey_page` both map to this crate's `grey` field —
    /// one buffer field name shared across the two request shapes rather
    /// than two OGAR-specific names).
    fn ogar_param_to_request_field(cap: &str, ogar_name: &str) -> Option<&'static str> {
        match (cap, ogar_name) {
            ("recognize_line", "grey_line") => Some("grey"),
            ("recognize_line", "width") => Some("width"),
            ("recognize_line", "height") => Some("height"),
            ("recognize_line", "with_dict") => Some("with_dict"),
            ("recognize_page", "grey_page") => Some("grey"),
            ("recognize_page", "width") => Some("width"),
            ("recognize_page", "height") => Some("height"),
            ("recognize_page", "with_dict") => Some("with_dict"),
            ("extract_text_layer", "pdf_bytes") => Some("pdf_bytes"),
            ("extract_page_image", "pdf_bytes") => Some("pdf_bytes"),
            ("extract_page_image", "page") => Some("page"),
            ("render_text", "lines") => Some("lines"),
            ("render_tsv", "lines") => Some("lines"),
            ("render_tsv", "page_w") => Some("page_w"),
            ("render_tsv", "page_h") => Some("page_h"),
            ("render_hocr", "lines") => Some("lines"),
            ("render_hocr", "page_w") => Some("page_w"),
            ("render_hocr", "page_h") => Some("page_h"),
            ("render_hocr", "image_name") => Some("image_name"),
            ("render_searchable_pdf", "pages") => Some("pages"),
            ("render_searchable_pdf", "dpi") => Some("dpi"),
            _ => None,
        }
    }

    /// Every mandatory OGAR param has a documented `OcrRequest` field
    /// counterpart — the mechanical name-level seam check.
    #[test]
    fn every_mandatory_ogar_param_maps_to_a_request_field() {
        for spec in ogar_vocab::ocr_actions::ocr_actions() {
            for p in spec.params.iter().filter(|p| p.mandatory) {
                assert!(
                    ogar_param_to_request_field(&spec.def.predicate, p.name).is_some(),
                    "{}: mandatory param `{}` has no OcrRequest field mapping",
                    spec.def.predicate,
                    p.name
                );
            }
        }
    }

    /// End-to-end smoke test against the real proven `eng` model data, when
    /// present in this environment (`/tmp/eng.lstm*` + `/tmp/line36.pgm`,
    /// produced by the recognizer's own oracle-comparison workflow — see
    /// `tesseract-rs/CLAUDE.md`'s "the proven method"). Early-returns (with
    /// an explanation) when the data isn't present, so this test never fails
    /// CI in an environment that hasn't staged those files — it only proves
    /// the executor reproduces the ALREADY-proven `"qLLiy,,"` regression
    /// when the data IS present.
    #[test]
    fn smoke_recognize_line_matches_proven_regression() {
        let lstm = Path::new("/tmp/eng.lstm");
        let unicharset = Path::new("/tmp/eng.lstm-unicharset");
        let recoder = Path::new("/tmp/eng.lstm-recoder");
        let pgm = Path::new("/tmp/line36.pgm");
        if !(lstm.exists() && unicharset.exists() && recoder.exists() && pgm.exists()) {
            eprintln!(
                "smoke_recognize_line_matches_proven_regression: skipping — \
                 /tmp/eng.lstm* and/or /tmp/line36.pgm not present in this environment"
            );
            return;
        }

        let executor = OcrExecutor::from_data_paths(lstm, unicharset, recoder, None, None, None)
            .expect("recognizer assembles from real /tmp components");
        let bytes = std::fs::read(pgm).expect("read /tmp/line36.pgm");
        let (grey, w, h) = tesseract_ocr::parse_pgm(&bytes).expect("parse /tmp/line36.pgm");

        let response = executor
            .execute(OcrRequest::RecognizeLine {
                grey: &grey,
                width: w,
                height: h,
                with_dict: false,
            })
            .expect("recognize_line executes against real data");

        match response {
            OcrResponse::Recognized { text, .. } => {
                assert_eq!(
                    text, "qLLiy,,",
                    "regression vs the proven eng.lstm baseline"
                );
            }
            other => panic!("unexpected response variant: {other:?}"),
        }
    }
}
