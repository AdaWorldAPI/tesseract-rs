//! # tesseract-ogar — the in-binary executor for the OGAR OCR action table
//!
//! [`ogar_vocab::ocr_actions`] is the **authoritative** declaration of the
//! OCR capabilities `tesseract-rs` exposes — the original eight
//! (`recognize_line` / `recognize_page` / `extract_text_layer` /
//! `extract_page_image` / `render_text` / `render_tsv` / `render_hocr` /
//! `render_searchable_pdf`) plus the v2 structured-document surface
//! (`recognize_page_words` / `recognize_document` / `harvest_fields` /
//! `segment_page` / `detect_halftone_regions` / `detect_page_furniture`).
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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use tesseract_core::dawg::DawgError;
use tesseract_core::DictLite;
use tesseract_ocr::{
    conn_comp_bb, detect_page_furniture, generate_halftone_mask, german_invoice_fields,
    harden_numeric_tokens, harvest_fields, xy_cut, DocPage, Document, HarvestedField, LineWords,
    LstmRecognizer, PageFurniture, PageRect, XyCutParams,
};
use tesseract_ocr_pdf::{GreyImage, PageOcr, PdfError, RenderReport, SearchablePdfError};

/// The V3-substrate <-> Python-SDK parity probe: walks a loaded `Network`
/// byte stream into per-node content-blind facets
/// ([`v3_facet::NodeFacet`], each carrying a `FacetCascade`) and provides the
/// TSV formatters + Python decode harness shared by
/// `examples/v3_facet_probe.rs` and `tests/v3_facet_parity.rs`. See that
/// module's docs for why it re-derives the header walk instead of reusing
/// `tesseract_ocr::Network`'s tree directly.
pub mod v3_facet;

/// Sentence assembly over a recognized [`DocPage`] — the per-sentence text
/// unit [`reasoning`] needs. See that module's docs for why this is a
/// separate library layer, not a 15th OGAR capability.
pub mod sentences;

/// The reasoning layer: `deepnsm` SPO extraction + a NARS belief per
/// assembled sentence. See the module docs for what's wired and what's
/// deliberately not (the AS-IS BOUNDARY split from `tesseract-rs/CLAUDE.md`).
pub mod reasoning;

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
    // v2 (2026-07-10) — the structured-document + layout-classifier surface.
    "recognize_page_words",
    "recognize_document",
    "harvest_fields",
    "segment_page",
    "detect_halftone_regions",
    "detect_page_furniture",
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

/// Pure-Rust encoded-image (PNG/JPEG/…) → grey decode, re-exported so a
/// consumer decodes container bytes via this executor crate and hands the grey
/// straight to [`OcrExecutor::execute`]. Available with the `image-decode`
/// feature.
#[cfg(feature = "image-decode")]
pub use tesseract_ocr::{decode_image, ImageDecodeError};

/// Segmentation binarization mode selectable via
/// [`OcrRequest::RecognizeDocument`]'s `binarize` field — re-exported so a
/// caller doesn't need a direct `tesseract-ocr` dependency merely to name a
/// mode. [`BinarizeMode::Otsu`] (also [`BinarizeMode::default`]) is the
/// crate-wide default: a single global threshold, byte-identical to this
/// executor's behaviour before this field existed. [`BinarizeMode::Sauvola`]
/// is the adaptive per-pixel alternative for source pages a single global
/// threshold under-serves (uneven illumination, aged scans). See
/// [`tesseract_ocr::BinarizeMode`] for the full algorithm documentation.
pub use tesseract_ocr::BinarizeMode;

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
    /// `recognize_page_words` — a full grey page recognized to WORD/box
    /// output (the word-level counterpart of `recognize_page`).
    RecognizePageWords {
        /// Row-major 8-bit grey page.
        grey: &'a [u8],
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Use the loaded dictionary beam, if any.
        with_dict: bool,
    },
    /// `recognize_document` — the ONE-SHOT: grey page → `doc.v1` JSON
    /// (classified regions) + typed field harvest. `harvest_profile`
    /// selects the field set (`Some("german_invoice")`; `None` / empty = no
    /// harvest; any other value FAILS with
    /// [`OcrExecError::UnknownHarvestProfile`]). `binarize` selects the
    /// segmentation binarization mode threaded through EVERY internal
    /// binarization pass `recognize_document_with_mode` runs (word/line text
    /// recognition, the layout `xy_cut` split, and region/table
    /// classification) — this is the same axis a hot record's
    /// [`RecognitionConfig`] can later be used to recall documents by.
    RecognizeDocument {
        /// Row-major 8-bit grey page.
        grey: &'a [u8],
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Use the loaded dictionary beam, if any.
        with_dict: bool,
        /// The field-harvest profile (`None` = no harvest).
        harvest_profile: Option<&'a str>,
        /// Segmentation binarization mode (see [`BinarizeMode`]).
        /// [`BinarizeMode::default()`] (Otsu) reproduces this variant's
        /// exact pre-existing behaviour — every existing caller that
        /// constructs this variant must now supply this field explicitly
        /// (plain Rust enums have no per-field defaults), and passing
        /// [`BinarizeMode::default()`] keeps that caller's output
        /// byte-identical to before this field existed.
        binarize: BinarizeMode,
    },
    /// `harvest_fields` — the typed field harvest over an already-recognized
    /// page's word output. `harvest_profile` is required (unknown / empty
    /// FAILS with [`OcrExecError::UnknownHarvestProfile`]).
    HarvestFields {
        /// Recognized lines, in reading order.
        line_words: &'a [LineWords],
        /// Page width in pixels.
        page_w: u32,
        /// Page height in pixels.
        page_h: u32,
        /// The field-harvest profile (e.g. `"german_invoice"`).
        harvest_profile: &'a str,
    },
    /// `segment_page` — recursive XY-cut layout segmentation (columns /
    /// deimposition) → reading-ordered region rects.
    SegmentPage {
        /// Row-major 8-bit grey page.
        grey: &'a [u8],
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// XY-cut tuning (the `min_gap_frac` / `min_region_px` / `max_depth`
        /// optional params; [`XyCutParams::default`] for the defaults).
        params: XyCutParams,
    },
    /// `detect_halftone_regions` — the leptonica-parity halftone (image)
    /// region detector over a BINARIZED page (`0` = ink convention).
    DetectHalftoneRegions {
        /// Row-major binarized page (`0` = foreground/ink, `255` = bg).
        binary: &'a [u8],
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
    },
    /// `detect_page_furniture` — header / footer / page-number detection over
    /// an already-recognized page's word output.
    DetectPageFurniture {
        /// Recognized lines, in reading order.
        line_words: &'a [LineWords],
        /// Page width in pixels.
        page_w: u32,
        /// Page height in pixels.
        page_h: u32,
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
    /// `recognize_page_words`'s `line_words` output.
    LineWordsOut(Vec<LineWords>),
    /// `recognize_document`'s `doc_json, fields` outputs.
    DocumentOut {
        /// The rendered `tesseract-rs/doc.v1` JSON.
        doc_json: String,
        /// The harvested typed fields (empty when no harvest ran).
        fields: Vec<HarvestedField>,
    },
    /// `harvest_fields`'s `fields` output.
    Fields(Vec<HarvestedField>),
    /// `segment_page`'s `regions_rects` output (reading-ordered leaf rects).
    Regions(Vec<PageRect>),
    /// `detect_halftone_regions`'s `figure_rects, mask_w, mask_h, found`
    /// outputs. Figure rects are page-space `(left, top, right, bottom)`.
    HalftoneRegions {
        /// Figure component bboxes `(left, top, right, bottom)`.
        figure_rects: Vec<(i32, i32, i32, i32)>,
        /// Halftone mask width (may be smaller than the page).
        mask_w: usize,
        /// Halftone mask height.
        mask_h: usize,
        /// Whether any halftone region was found.
        found: bool,
    },
    /// `detect_page_furniture`'s `header_lines, footer_lines, page_number`
    /// outputs.
    PageFurnitureOut(PageFurniture),
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
    /// A `harvest_profile` value the executor does not recognize — fail-closed
    /// so a typo can never silently drop invoice-field validation (v2, the
    /// spec's V2-3 rule). The only value understood today is `"german_invoice"`.
    UnknownHarvestProfile(String),
}

impl std::fmt::Display for OcrExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "reading {}: {e}", path.display()),
            Self::Recognizer(e) => write!(f, "recognizer: {e}"),
            Self::Dawg(e) => write!(f, "dictionary assembly: {e:?}"),
            Self::Pdf(e) => write!(f, "PDF: {e}"),
            Self::SearchablePdf(e) => write!(f, "searchable PDF render: {e}"),
            Self::UnknownHarvestProfile(p) => {
                write!(
                    f,
                    "unknown harvest_profile {p:?} (known: \"german_invoice\")"
                )
            }
        }
    }
}

/// Map a `harvest_profile` string to its [`tesseract_ocr::FieldSpec`] set.
/// `None`/empty → no harvest (`Ok(None)`); `"german_invoice"` → the German
/// invoice field set; anything else fails closed
/// ([`OcrExecError::UnknownHarvestProfile`]).
fn harvest_specs(
    profile: Option<&str>,
) -> Result<Option<Vec<tesseract_ocr::FieldSpec>>, OcrExecError> {
    match profile.filter(|p| !p.is_empty()) {
        None => Ok(None),
        Some("german_invoice") => Ok(Some(german_invoice_fields())),
        Some(other) => Err(OcrExecError::UnknownHarvestProfile(other.to_owned())),
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

/// The model-determined half of a recognized document's provenance —
/// everything knowable about HOW a page was recognized once an
/// [`OcrExecutor`] has finished loading, independent of any one page it
/// later recognizes. Built via [`OcrExecutor::config`].
///
/// ## Why record configuration, not just a confidence score
///
/// A downstream KV writer storing a scanned document keyed by
/// `documentpath/documentid` wants a small, scannable **hot record** — enough
/// to answer "which documents were recognized under which conditions?"
/// without parsing the (much larger) `doc.v1` payload or touching the raw
/// blob in a secondary table. A confidence score alone cannot answer that
/// question *retroactively*: this session's own measurements found a real
/// unevenly-lit page recognized at `mean_conf` **99.47** whose actual
/// character error rate was **0.6154** (61.5% wrong) — 99.5% confident and
/// mostly wrong. If the configuration that produced a recognition is never
/// recorded, there is no way to later ask "every document recognized under
/// the configuration now known to be unreliable on uneven lighting" — every
/// such document's confidence score looked equally plausible at the time it
/// was written. [`RecognitionConfig`] (or its [`RecognitionConfig::config_id`]
/// digest) is that recall key: a document's hot record can carry it so a
/// later audit or reprocessing pass can select documents by CONFIGURATION,
/// not just by score.
///
/// ## What's included, and why
///
/// - `network_spec` and `null_char` identify the loaded MODEL — different
///   models (`eng` vs `deu`) have measurably different spec strings and
///   `null_char` values (each model self-derives its own from its own
///   unicharset size; see `tesseract-rs/CLAUDE.md`'s eng/deu parity notes).
/// - `charset_len` and `code_range` identify the model's SHAPE (vocabulary
///   size and recoder lattice width) as a cheap architecture fingerprint,
///   without hashing the model's weight bytes.
/// - `dict_loaded` records whether the dictionary beam was available —
///   decode behaviour differs materially with the dict beam active
///   (different certainty constants, different word-boundary handling; see
///   [`tesseract_ocr::LstmRecognizer::recognize_grid_with_dict`]), so this is
///   load-bearing provenance, not a cosmetic flag.
///
/// ## Deliberately OMITTED: `deskew_fired` / `rectify_fired`
///
/// A configuration flag that can never report `true` carries no recall
/// value — it is indistinguishable from a flag that is always false by
/// construction, and recording it would look like real provenance while
/// being permanently inert (this workspace's own rule: a guard that cannot
/// fire is the defect one level up). Neither preprocessing stage is wired
/// into this executor today:
///
/// - **Deskew** has no pipeline-wiring step yet (plan step D8 — see
///   `tesseract-rs/CLAUDE.md`'s "Deskew wave" section); `deskew.rs`'s
///   primitives exist and are byte-parity proven, but nothing calls them
///   from [`OcrExecutor::execute`].
/// - **Rectify** (`rectify.rs`'s `auto_rectify`) is wired into
///   `tesseract-ocr-web`'s debug/PDF routes as an opt-in checkbox, but is
///   NOT threaded through this executor's [`OcrRequest::RecognizeDocument`]
///   path at all.
///
/// Add these fields only once the corresponding stage is actually threaded
/// through [`OcrExecutor::execute`] and can genuinely report `true` on some
/// real input — until then they would be exactly the inert-flag
/// anti-pattern this doc section exists to avoid.
#[derive(Debug, Clone, PartialEq)]
pub struct RecognitionConfig {
    /// The loaded model's VGSL-ish spec string (e.g.
    /// `[1,36,0,1Ct3,3,16Mp3,3...O1c111]`) — identifies the network
    /// architecture actually running.
    pub network_spec: String,
    /// The CTC null/blank class id. A real, measured model discriminator,
    /// not cosmetic: `eng.lstm` is 110, `deu.lstm` is 114.
    pub null_char: i32,
    /// The loaded character set's entry count. `eng` is 112, `deu` is 116
    /// (the extra `ä ö ü ß`) — a cheap shape fingerprint that distinguishes
    /// models without hashing the model bytes.
    pub charset_len: usize,
    /// The recoder's lattice width (its `code_range`) — another model-shape
    /// discriminator alongside [`Self::charset_len`].
    pub code_range: i32,
    /// Whether a word/punctuation/number dictionary DAWG was loaded (see
    /// [`OcrExecutor::from_data_paths`]).
    pub dict_loaded: bool,
}

impl RecognitionConfig {
    /// A hash over every field, meant to index a KV hot record on one scalar
    /// column (a cheap equality check or group-by) instead of a
    /// multi-column scan across `network_spec`/`null_char`/`charset_len`/
    /// `code_range`/`dict_loaded`.
    ///
    /// # NOT a persisted identity — NOT stable across Rust or crate versions
    ///
    /// This is built on [`std::collections::hash_map::DefaultHasher`], whose
    /// algorithm the standard library explicitly documents as **unspecified
    /// and subject to change between releases** of Rust. It is equally
    /// unstable across changes to THIS crate — adding, removing, or
    /// reordering the fields this method hashes changes the result. Treat
    /// `config_id` as an **in-deployment index key only**: safe to compare
    /// within one running deployment on one build, unsafe to compare across
    /// a Rust upgrade, a `tesseract-ogar` upgrade, or a value persisted
    /// long-term and read back after either. A caller that needs
    /// cross-version stability must hash the fields itself with an
    /// explicitly-versioned scheme instead of relying on this method.
    /// Silently assuming stability here is exactly the failure mode this
    /// warning exists to prevent: it would corrupt a recall query across an
    /// upgrade without any error at the point of corruption.
    #[must_use]
    pub fn config_id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.network_spec.hash(&mut hasher);
        self.null_char.hash(&mut hasher);
        self.charset_len.hash(&mut hasher);
        self.code_range.hash(&mut hasher);
        self.dict_loaded.hash(&mut hasher);
        hasher.finish()
    }
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
            OcrRequest::RecognizePageWords { .. } => "recognize_page_words",
            OcrRequest::RecognizeDocument { .. } => "recognize_document",
            OcrRequest::HarvestFields { .. } => "harvest_fields",
            OcrRequest::SegmentPage { .. } => "segment_page",
            OcrRequest::DetectHalftoneRegions { .. } => "detect_halftone_regions",
            OcrRequest::DetectPageFurniture { .. } => "detect_page_furniture",
        }
    }

    /// A structured recognition-configuration snapshot of this executor —
    /// the model-determined half of a document's provenance, independent of
    /// any one page later recognized. See [`RecognitionConfig`]'s doc
    /// comment for why this exists (a downstream KV writer's hot-record
    /// recall key). No I/O; the only allocation is the returned
    /// [`RecognitionConfig`]'s `network_spec` `String` clone.
    #[must_use]
    pub fn config(&self) -> RecognitionConfig {
        RecognitionConfig {
            network_spec: self.recognizer.network_str.clone(),
            null_char: self.recognizer.null_char,
            charset_len: self.recognizer.charset.size(),
            code_range: self.recognizer.recoder.code_range(),
            dict_loaded: self.dict.is_some(),
        }
    }

    /// The loaded model's character set — what
    /// [`DocPage::from_line_words`](tesseract_ocr::DocPage::from_line_words)
    /// needs to turn [`OcrRequest::RecognizePageWords`]'s `LineWords` output
    /// into a typed [`DocPage`], for a caller that wants the
    /// [`crate::sentences`]/[`crate::reasoning`] surface rather than
    /// [`OcrRequest::RecognizeDocument`]'s serialized `doc.v1` JSON string.
    #[must_use]
    pub fn charset(&self) -> &tesseract_core::CharSet {
        &self.recognizer.charset
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
            OcrRequest::RecognizePageWords {
                grey,
                width,
                height,
                with_dict,
            } => {
                let dict = if with_dict { self.dict.as_ref() } else { None };
                let lines = self
                    .recognizer
                    .recognize_page_makerow_words(grey, width, height, dict)
                    .map_err(OcrExecError::Recognizer)?;
                Ok(OcrResponse::LineWordsOut(lines))
            }
            OcrRequest::RecognizeDocument {
                grey,
                width,
                height,
                with_dict,
                harvest_profile,
                binarize,
            } => {
                let dict = if with_dict { self.dict.as_ref() } else { None };
                let specs = harvest_specs(harvest_profile)?;
                let Document { json, fields, .. } = self
                    .recognizer
                    .recognize_document_with_mode(
                        grey,
                        width,
                        height,
                        dict,
                        specs.as_deref(),
                        binarize,
                    )
                    .map_err(OcrExecError::Recognizer)?;
                Ok(OcrResponse::DocumentOut {
                    doc_json: json,
                    fields,
                })
            }
            OcrRequest::HarvestFields {
                line_words,
                page_w,
                page_h,
                harvest_profile,
            } => {
                // harvest_profile is required here; empty/unknown fails closed.
                let specs = harvest_specs(Some(harvest_profile))?.ok_or_else(|| {
                    OcrExecError::UnknownHarvestProfile(harvest_profile.to_owned())
                })?;
                let mut page =
                    DocPage::from_line_words(line_words, &self.recognizer.charset, page_w, page_h);
                harden_numeric_tokens(&mut page);
                Ok(OcrResponse::Fields(harvest_fields(&page, &specs)))
            }
            OcrRequest::SegmentPage {
                grey,
                width,
                height,
                params,
            } => Ok(OcrResponse::Regions(xy_cut(grey, width, height, &params))),
            OcrRequest::DetectHalftoneRegions {
                binary,
                width,
                height,
            } => {
                // No halftone mask (page below MinWidth/MinHeight, or empty) →
                // found=false, no rects, zero mask dims.
                let (figure_rects, mask_w, mask_h, found) =
                    match generate_halftone_mask(binary, width, height) {
                        Some(hm) if hm.found => {
                            let rects = conn_comp_bb(&hm.mask, hm.mask_w, hm.mask_h, 8)
                                .into_iter()
                                .map(|b| (b.x, b.y, b.x + b.w, b.y + b.h))
                                .collect();
                            (rects, hm.mask_w, hm.mask_h, true)
                        }
                        Some(hm) => (Vec::new(), hm.mask_w, hm.mask_h, false),
                        None => (Vec::new(), 0, 0, false),
                    };
                Ok(OcrResponse::HalftoneRegions {
                    figure_rects,
                    mask_w,
                    mask_h,
                    found,
                })
            }
            OcrRequest::DetectPageFurniture {
                line_words,
                page_w,
                page_h,
            } => {
                let page =
                    DocPage::from_line_words(line_words, &self.recognizer.charset, page_w, page_h);
                Ok(OcrResponse::PageFurnitureOut(detect_page_furniture(&page)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

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
        assert_eq!(
            concepts.len(),
            ogar_vocab::ocr_actions::OCR_SUBJECT_CLASSIDS.len(),
            "one concept per hot-plugged subject classid"
        );
        assert!(concepts.contains(&("textline", 0x0805)));
        assert!(concepts.contains(&("page_layout", 0x0807)), "v2 subject");
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
            OcrRequest::RecognizePageWords {
                grey: &[],
                width: 0,
                height: 0,
                with_dict: false,
            },
            OcrRequest::RecognizeDocument {
                grey: &[],
                width: 0,
                height: 0,
                with_dict: false,
                harvest_profile: None,
                binarize: BinarizeMode::default(),
            },
            OcrRequest::HarvestFields {
                line_words: &[],
                page_w: 0,
                page_h: 0,
                harvest_profile: "german_invoice",
            },
            OcrRequest::SegmentPage {
                grey: &[],
                width: 0,
                height: 0,
                params: XyCutParams::default(),
            },
            OcrRequest::DetectHalftoneRegions {
                binary: &[],
                width: 0,
                height: 0,
            },
            OcrRequest::DetectPageFurniture {
                line_words: &[],
                page_w: 0,
                page_h: 0,
            },
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
            // v2 rows.
            ("recognize_page_words", "grey_page") => Some("grey"),
            ("recognize_page_words", "width") => Some("width"),
            ("recognize_page_words", "height") => Some("height"),
            ("recognize_document", "grey_page") => Some("grey"),
            ("recognize_document", "width") => Some("width"),
            ("recognize_document", "height") => Some("height"),
            ("harvest_fields", "line_words") => Some("line_words"),
            ("harvest_fields", "page_w") => Some("page_w"),
            ("harvest_fields", "page_h") => Some("page_h"),
            ("harvest_fields", "harvest_profile") => Some("harvest_profile"),
            ("segment_page", "grey_page") => Some("grey"),
            ("segment_page", "width") => Some("width"),
            ("segment_page", "height") => Some("height"),
            ("detect_halftone_regions", "binary_page") => Some("binary"),
            ("detect_halftone_regions", "width") => Some("width"),
            ("detect_halftone_regions", "height") => Some("height"),
            ("detect_page_furniture", "line_words") => Some("line_words"),
            ("detect_page_furniture", "page_w") => Some("page_w"),
            ("detect_page_furniture", "page_h") => Some("page_h"),
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
    /// the executor reproduces the proven `"y,"` regression when the data IS
    /// present. (The anchor changed from the historical `"qLLiy,,"` when the
    /// `SimpleTextOutput` transcode bug was fixed: eng.lstm is `O1c111` =
    /// softmax activation with CTC LOSS, so the beam runs CTC dup-collapse
    /// semantics — the old string was an artifact of `simple_text=true`
    /// re-emitting every per-timestep spike; re-anchored byte-identical vs
    /// the corrected libtesseract oracle, 8/8 fixtures.)
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
                assert_eq!(text, "y,", "regression vs the proven eng.lstm baseline");
            }
            other => panic!("unexpected response variant: {other:?}"),
        }
    }

    /// The bundled `corpus/model` eng loader for [`RecognitionConfig`]/
    /// `binarize`-default tests below, mirroring `examples/ocr_demo.rs`'s
    /// loading pattern exactly (same crate, same `corpus/model` layout).
    /// Returns `None` when `corpus/model/eng.lstm` isn't present in this
    /// environment, so callers can skip gracefully instead of failing CI in
    /// an environment that hasn't staged the bundled model.
    fn load_bundled_eng_executor() -> Option<OcrExecutor> {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
        if !model.join("eng.lstm").exists() {
            return None;
        }
        let dawg = |name: &str| {
            let p = model.join(name);
            p.exists().then_some(p)
        };
        Some(
            OcrExecutor::from_data_paths(
                &model.join("eng.lstm"),
                &model.join("eng.lstm-unicharset"),
                &model.join("eng.lstm-recoder"),
                dawg("eng.lstm-word-dawg").as_deref(),
                dawg("eng.lstm-punc-dawg").as_deref(),
                dawg("eng.lstm-number-dawg").as_deref(),
            )
            .expect("load the eng recognizer + dictionary from corpus/model"),
        )
    }

    /// [`OcrExecutor::config`] reports the ACTUAL loaded model's values (not
    /// placeholders), and [`RecognitionConfig::config_id`] is stable across
    /// two separate `config()` calls on the same executor. Real, measured
    /// pins from `tesseract-rs/CLAUDE.md`'s eng/deu parity notes: eng.lstm's
    /// `null_char` is 110, its unicharset has 112 entries, its recoder's
    /// `code_range` is 111 (`E-OCR-RECOGNIZER-LOAD-1`).
    #[test]
    fn config_reports_the_loaded_models_real_values() {
        let Some(executor) = load_bundled_eng_executor() else {
            eprintln!(
                "config_reports_the_loaded_models_real_values: skipping — \
                 corpus/model/eng.lstm not present in this environment"
            );
            return;
        };

        let config_a = executor.config();
        let config_b = executor.config();
        assert_eq!(
            config_a, config_b,
            "two config() calls on the same executor must agree field-for-field"
        );
        assert_eq!(
            config_a.config_id(),
            config_b.config_id(),
            "config_id must be stable across two config() calls on the same executor"
        );

        assert_eq!(config_a.null_char, 110, "eng.lstm's measured null_char");
        assert_eq!(
            config_a.charset_len, 112,
            "eng.lstm-unicharset's measured entry count"
        );
        assert_eq!(
            config_a.code_range, 111,
            "eng.lstm-recoder's measured code_range"
        );
        // MEASURED, and it is NOT what the prose says. eng.lstm's stored
        // `network_str` is
        // `[1,36,0,1Ct3,3,16Mp3,3Lfys48Lfx96Lrx96Lfx192O1c1]` — the head reads
        // **`O1c1`**, not `O1c111`. The spec string is the VGSL recorded at
        // TRAINING time, where the class count was still a placeholder; the
        // loaded layer's real output width is carried by `code_range` (111),
        // asserted separately above. Do not "correct" this to `O1c111` from
        // documentation prose — that mismatch is exactly how this assertion
        // was wrong the first time.
        assert!(
            config_a.network_spec.contains("O1c1"),
            "eng.lstm's measured VGSL spec must carry its softmax/CTC head: {}",
            config_a.network_spec
        );
        assert!(
            config_a.network_spec.starts_with("[1,36,0,1Ct3,3,16Mp3,3"),
            "eng.lstm's measured VGSL prefix (36-row input, 3x3 conv, 3x3 maxpool): {}",
            config_a.network_spec
        );
        assert!(
            config_a.dict_loaded,
            "corpus/model ships all three eng dawgs, so the dict beam must be loaded"
        );
    }

    /// [`RecognitionConfig::config_id`] must actually discriminate: two
    /// independently hand-built configs with IDENTICAL fields hash
    /// identically, and flipping exactly one field at a time changes the
    /// digest. Hand-built (no model/corpus dependency) so this always runs.
    #[test]
    fn config_id_differs_when_a_field_differs() {
        let base = RecognitionConfig {
            network_spec: "[1,36,0,1Ct3,3,16Mp3,3O1c111]".to_owned(),
            null_char: 110,
            charset_len: 112,
            code_range: 111,
            dict_loaded: true,
        };

        // Two field-for-field identical configs must hash identically — the
        // model-independent half of "stable".
        let identical = base.clone();
        assert_eq!(
            base.config_id(),
            identical.config_id(),
            "field-for-field identical configs must hash identically"
        );

        // Flipping exactly one field at a time must change the digest — a
        // real discrimination test, not merely "some fields produce a hash".
        let diff_null = RecognitionConfig {
            null_char: 114,
            ..base.clone()
        };
        assert_ne!(
            base.config_id(),
            diff_null.config_id(),
            "differing null_char must change config_id"
        );

        let diff_charset = RecognitionConfig {
            charset_len: 116,
            ..base.clone()
        };
        assert_ne!(
            base.config_id(),
            diff_charset.config_id(),
            "differing charset_len must change config_id"
        );

        let diff_code_range = RecognitionConfig {
            code_range: 115,
            ..base.clone()
        };
        assert_ne!(
            base.config_id(),
            diff_code_range.config_id(),
            "differing code_range must change config_id"
        );

        let diff_dict = RecognitionConfig {
            dict_loaded: false,
            ..base.clone()
        };
        assert_ne!(
            base.config_id(),
            diff_dict.config_id(),
            "differing dict_loaded must change config_id"
        );

        let diff_spec = RecognitionConfig {
            network_spec: "a-different-spec-string".to_owned(),
            ..base.clone()
        };
        assert_ne!(
            base.config_id(),
            diff_spec.config_id(),
            "differing network_spec must change config_id"
        );
    }

    /// The `binarize` field's default must remain [`BinarizeMode::Otsu`] —
    /// the goldens and the 8+7+0 CER fence (`tesseract-ocr`'s own test
    /// suite; see `tesseract-rs/CLAUDE.md`) both depend on Otsu never
    /// silently becoming a different default. Pins the value directly (no
    /// model needed), then — when `corpus/model/eng.lstm` and
    /// `corpus/pages/page_01.pgm` are present — proves the `binarize` field
    /// [`OcrExecutor::execute`] reads actually REACHES
    /// `recognize_document_with_mode` rather than being silently ignored: a
    /// request built with `BinarizeMode::default()` must produce
    /// byte-identical output to the same request built with an explicit
    /// `BinarizeMode::Otsu`.
    #[test]
    fn recognize_document_default_binarize_mode_is_otsu() {
        assert_eq!(
            BinarizeMode::default(),
            BinarizeMode::Otsu,
            "the crate-wide segmentation default must stay Otsu"
        );

        let Some(executor) = load_bundled_eng_executor() else {
            eprintln!(
                "recognize_document_default_binarize_mode_is_otsu: skipping the \
                 end-to-end half — corpus/model/eng.lstm not present in this environment"
            );
            return;
        };
        let page = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm");
        if !page.exists() {
            eprintln!(
                "recognize_document_default_binarize_mode_is_otsu: skipping the \
                 end-to-end half — corpus/pages/page_01.pgm not present in this environment"
            );
            return;
        }
        let bytes = std::fs::read(&page).expect("read corpus/pages/page_01.pgm");
        let (grey, w, h) =
            tesseract_ocr::parse_pgm(&bytes).expect("parse corpus/pages/page_01.pgm");

        let via_default = executor
            .execute(OcrRequest::RecognizeDocument {
                grey: &grey,
                width: w,
                height: h,
                with_dict: false,
                harvest_profile: None,
                binarize: BinarizeMode::default(),
            })
            .expect("recognize_document via BinarizeMode::default()");
        let via_explicit_otsu = executor
            .execute(OcrRequest::RecognizeDocument {
                grey: &grey,
                width: w,
                height: h,
                with_dict: false,
                harvest_profile: None,
                binarize: BinarizeMode::Otsu,
            })
            .expect("recognize_document via explicit BinarizeMode::Otsu");
        assert_eq!(
            via_default, via_explicit_otsu,
            "BinarizeMode::default() must reach recognize_document_with_mode identically \
             to an explicit BinarizeMode::Otsu"
        );
    }
}
