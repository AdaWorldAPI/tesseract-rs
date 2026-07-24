//! HTTP surface: the upload/URL form and the OCR handler.

use std::sync::Arc;

use askama::Template;
use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;

use tesseract_ocr_pdf::layout::{
    doc_v1_layout, grey_png_data_uri, render_pdf, render_preview_html, searchable_layout,
};
use tesseract_ocr_pdf::{render_searchable_pdf, GreyImage, PageOcr, PlacedWord};

use crate::fetch::fetch_image_url;
use crate::ocr::{
    ocr_image_bytes, ocr_image_bytes_debug, ocr_image_bytes_json, OcrDebugOutcome, OcrJsonOutcome,
    OcrOutcome, OutputFormat,
};
use crate::state::AppState;

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    error: Option<String>,
}

#[derive(Template)]
#[template(path = "result.html")]
struct ResultTemplate {
    width: usize,
    height: usize,
    /// "Characters" (text mode) or "Words" (JSON mode — a character count is
    /// meaningless for a JSON document).
    primary_label: &'static str,
    primary_count: usize,
    line_count: usize,
    elapsed_ms: String,
    /// Mean word confidence rendered for display (`"96"`, or `"—"` when no
    /// words were recognized).
    confidence: String,
    /// `true` when the recognizer was not confident — the result page shows a
    /// warning banner (likely handwriting / low-resolution / non-printed text).
    low_confidence: bool,
    /// The text to show in the result `<pre>` block: recognized text, or the
    /// rendered JSON document. Askama HTML-escapes this on render.
    text: String,
    download_datauri: String,
    /// The download link's filename: `ocr.txt` or `result.json`.
    download_filename: &'static str,
}

/// Render a mean-confidence value (`-1` sentinel = no words) for display.
fn confidence_str(mean_conf: f32) -> String {
    if mean_conf < 0.0 {
        "\u{2014}".to_string() // em dash
    } else {
        format!("{}", mean_conf.round() as i32)
    }
}

/// Build the application router. Uploads are capped at 12 MB — this needs BOTH
/// limits: axum's per-extractor `DefaultBodyLimit` defaults to 2 MB (and would
/// reject larger multipart uploads before the handler runs), and tower-http's
/// `RequestBodyLimitLayer` caps the raw body; the smaller of the two wins, so
/// both are raised together. The URL-fetch arm has its own 10 MB cap in
/// [`fetch_image_url`].
///
/// Merges in [`crate::api::router`] — the machine-facing `/api/v1/*` +
/// `/openapi.json` surface (Power Platform custom connector) — BEFORE the
/// upload-size layers, so those same limits (and not a separate/looser cap)
/// bound the API routes too.
pub fn router(state: Arc<AppState>) -> Router {
    const MAX_UPLOAD: usize = 12 * 1024 * 1024;
    Router::new()
        .route("/", get(index))
        .route("/ocr", post(ocr))
        // Searchable-PDF export. `?mode=structured` selects reconstruction "B";
        // default / `?mode=searchable` selects the scan+text facsimile "A".
        .route("/pdf", post(pdf))
        // Verbose debug preview: A and B side by side + region overlays + stats
        // + an honest algorithms-used trace.
        .route("/debug", get(debug_get).post(debug_post))
        .merge(crate::api::router())
        .layer(DefaultBodyLimit::max(MAX_UPLOAD))
        .layer(RequestBodyLimitLayer::new(MAX_UPLOAD))
        .with_state(state)
}

fn render<T: Template>(t: &T) -> Html<String> {
    match t.render() {
        Ok(s) => Html(s),
        Err(e) => {
            // The templates only Display `usize`/`String`, so this is effectively
            // unreachable; keep the fallback a static string (never interpolate
            // `e` into raw HTML) and log the detail.
            eprintln!("template render error: {e}");
            Html("<h1>internal template error</h1>".to_string())
        }
    }
}

async fn index() -> Html<String> {
    render(&IndexTemplate { error: None })
}

fn err_page(msg: impl Into<String>) -> Html<String> {
    render(&IndexTemplate {
        error: Some(msg.into()),
    })
}

async fn ocr(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Html<String> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut url: Option<String> = None;
    let mut format = OutputFormat::Text;
    let mut lang: Option<String> = None;

    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or_default().to_string();
                match name.as_str() {
                    "file" => match field.bytes().await {
                        Ok(b) if !b.is_empty() => file_bytes = Some(b.to_vec()),
                        Ok(_) => {}
                        Err(e) => return err_page(format!("upload read error: {e}")),
                    },
                    "url" => {
                        if let Ok(t) = field.text().await {
                            if !t.trim().is_empty() {
                                url = Some(t.trim().to_string());
                            }
                        }
                    }
                    "format" => {
                        // Never a hard error: an unrecognized/malformed format
                        // field just falls back to text (OutputFormat::from_field).
                        if let Ok(t) = field.text().await {
                            format = OutputFormat::from_field(Some(t.trim()));
                        }
                    }
                    "lang" => {
                        // Never a hard error: an unrecognized/absent lang field
                        // just falls back to eng (AppState::model).
                        if let Ok(t) = field.text().await {
                            if !t.trim().is_empty() {
                                lang = Some(t.trim().to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(None) => break,
            Err(e) => return err_page(format!("malformed upload: {e}")),
        }
    }

    // File wins over URL when both are present.
    let bytes = if let Some(b) = file_bytes {
        b
    } else if let Some(u) = url {
        match fetch_image_url(&u).await {
            Ok(b) => b,
            Err(e) => return err_page(e),
        }
    } else {
        return err_page("please choose an image file or paste an image URL");
    };

    // Recognition is heavy synchronous CPU work. Bound how many run at once
    // (a permit), then run it OFF the async worker threads via `spawn_blocking`
    // so a slow/large OCR can never stall the executor (healthcheck + other
    // requests keep flowing). The permit is moved into the blocking task and
    // released when it finishes.
    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return err_page("server is shutting down"),
    };
    let st = state.clone();
    match format {
        OutputFormat::Text => {
            let outcome = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                ocr_image_bytes(&st, &bytes, lang.as_deref())
            })
            .await;
            match outcome {
                Ok(Ok(out)) => render(&result_of_text(out)),
                Ok(Err(e)) => err_page(e),
                Err(e) => {
                    eprintln!("ocr: recognition task failed: {e}");
                    err_page("recognition failed unexpectedly")
                }
            }
        }
        OutputFormat::Json => {
            let outcome = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                ocr_image_bytes_json(&st, &bytes, lang.as_deref())
            })
            .await;
            match outcome {
                Ok(Ok(out)) => render(&result_of_json(out)),
                Ok(Err(e)) => err_page(e),
                Err(e) => {
                    eprintln!("ocr: recognition task failed: {e}");
                    err_page("recognition failed unexpectedly")
                }
            }
        }
    }
}

// ===========================================================================
// PDF export (A / B) + verbose debug preview (A vs B side by side)
// ===========================================================================

/// The `?mode=`/`?lang=` selectors for [`pdf`]. `mode="structured"` →
/// reconstruction "B" ([`doc_v1_layout`] + [`render_pdf`]); anything else —
/// `"searchable"`, an unknown value, or an absent field — → the searchable
/// facsimile "A". `lang="deu"` → the German model; anything else → English
/// (the pre-existing default) — see [`crate::state::AppState::model`].
///
/// `pub(crate)`: also used by [`crate::api`] so `POST /api/v1/pdf` accepts
/// the exact same `?mode=`/`?lang=` contract as the HTML `/pdf` route (which
/// reads `lang` from the multipart body instead, alongside `file`/`url`, and
/// so ignores this struct's `lang` field).
#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct PdfQuery {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    pub(crate) lang: Option<String>,
}

impl PdfQuery {
    pub(crate) fn is_structured(&self) -> bool {
        matches!(self.mode.as_deref(), Some("structured"))
    }
}

/// The `?lang=` selector for [`crate::api`] endpoints that have no `?mode=`
/// concept of their own (`/api/v1/recognize`, `/api/v1/pdf/structured`).
/// `"deu"` → the German model; anything else → English — see
/// [`crate::state::AppState::model`].
#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct LangQuery {
    #[serde(default)]
    pub(crate) lang: Option<String>,
}

/// A minimal `tesseract-rs/doc.v1` deserialize — only the fields the debug
/// preview + the searchable-word layer read. The canonical emitter is
/// `tesseract_ocr::structured::render_json_with_regions`; this consumer-side
/// subset mirrors just its `pages[].{width,height,quality,regions,fields}`
/// shape (a demo consumer parsing the documented seam — see
/// `docs/CONSUMER-GUIDE.md` "The doc.v1 seed shape").
mod docv1 {
    use serde::Deserialize;

    #[derive(Debug, Default, Deserialize)]
    pub struct Doc {
        #[serde(default)]
        pub pages: Vec<Page>,
    }

    // Only the fields the debug preview reads. serde ignores the rest of the
    // `doc.v1` page (`width` / `height` / `quality` / ...) — those stats come
    // from the recognition pass (`OcrDebugOutcome`) directly, so re-reading them
    // from the JSON would be redundant.
    #[derive(Debug, Default, Deserialize)]
    pub struct Page {
        #[serde(default)]
        pub regions: Vec<Region>,
        #[serde(default)]
        pub fields: Vec<Field>,
    }

    #[derive(Debug, Default, Deserialize)]
    pub struct Region {
        #[serde(rename = "type", default)]
        pub kind: String,
        #[serde(default)]
        pub bbox: [i64; 4],
        #[serde(default)]
        pub lines: Vec<Line>,
        #[serde(default)]
        pub rows: usize,
        #[serde(default)]
        pub cols: usize,
    }

    #[derive(Debug, Default, Deserialize)]
    pub struct Line {
        #[serde(default)]
        pub words: Vec<Word>,
    }

    #[derive(Debug, Default, Deserialize)]
    pub struct Word {
        #[serde(default)]
        pub text: String,
        #[serde(default)]
        pub bbox: [i64; 4],
    }

    #[derive(Debug, Default, Deserialize)]
    pub struct Field {
        #[serde(default)]
        pub key: String,
        #[serde(default)]
        pub value: String,
        #[serde(default)]
        pub bbox: [i64; 4],
    }
}

/// Clamp a signed `[l,t,r,b]` JSON bbox to a `u32` top-down tuple.
fn to_u32_bbox(b: [i64; 4]) -> (u32, u32, u32, u32) {
    let c = |v: i64| -> u32 { v.clamp(0, i64::from(u32::MAX)) as u32 };
    (c(b[0]), c(b[1]), c(b[2]), c(b[3]))
}

fn bbox_str(bbox: (u32, u32, u32, u32)) -> String {
    let (l, t, r, b) = bbox;
    format!("[{l}, {t}, {r}, {b}]")
}

/// Every recognized word (text + top-down image-pixel bbox) across every region
/// — the searchable-PDF "A" word layer, taken from the SAME `doc.v1` that drives
/// "B" so the two projections share one coordinate space.
fn placed_words_from_doc(doc: &docv1::Doc) -> Vec<PlacedWord> {
    let mut words = Vec::new();
    if let Some(page) = doc.pages.first() {
        for region in &page.regions {
            for line in &region.lines {
                for w in &line.words {
                    words.push(PlacedWord {
                        text: w.text.clone(),
                        box_: to_u32_bbox(w.bbox),
                    });
                }
            }
        }
    }
    words
}

/// Overlay/legend colour for a `doc.v1` region type (the types are additive —
/// an unknown kind renders grey rather than being dropped).
fn region_color(kind: &str) -> &'static str {
    match kind {
        "text" | "paragraph" => "#2563eb", // blue
        "header" => "#7c3aed",             // violet
        "footer" => "#0891b2",             // cyan
        "table" => "#16a34a",              // green
        "figure" => "#dc2626",             // red
        _ => "#6b7280",                    // grey
    }
}

/// CSS `left/top/width/height` in PER-CENT of the page box for a top-down bbox,
/// or `None` for a degenerate/zero-area box. Percentages make the overlay
/// scale-independent — it lines up at whatever width the page container renders.
fn pct_css(bbox: (u32, u32, u32, u32), w: usize, h: usize) -> Option<String> {
    let (l, t, r, b) = bbox;
    if r <= l || b <= t || w == 0 || h == 0 {
        return None;
    }
    let (wf, hf) = (w as f64, h as f64);
    Some(format!(
        "left:{:.3}%;top:{:.3}%;width:{:.3}%;height:{:.3}%",
        f64::from(l) / wf * 100.0,
        f64::from(t) / hf * 100.0,
        f64::from(r - l) / wf * 100.0,
        f64::from(b - t) / hf * 100.0,
    ))
}

/// One colour-coded region rectangle in the overlay (position as CSS %).
struct RegionOverlay {
    kind: String,
    color: &'static str,
    css: String,
}

/// One legend chip (a region type present on the page + its colour).
struct LegendItem {
    kind: String,
    color: &'static str,
}

/// A region-type tally (`text: 12`, `table: 1`, ...).
struct KindCount {
    kind: String,
    count: usize,
}

/// A located table/figure region, for the stats panel.
struct RegionPos {
    kind: String,
    bbox: String,
}

/// A harvested typed field row (key/value + where it was found).
struct FieldRow {
    key: String,
    value: String,
    bbox: String,
}

/// One line of the honest algorithms-used trace.
struct AlgoStep {
    name: String,
    detail: String,
}

/// Everything the `debug.html` template renders for one uploaded page: the
/// stats, the region overlay, the algorithms trace, and the two rendered
/// previews (A + B) that go into side-by-side `<iframe srcdoc>` panels.
struct DebugView {
    width: usize,
    height: usize,
    model: &'static str,
    lang: &'static str,
    network_spec: String,
    null_char: i32,
    dict_on: bool,
    /// `true` when `?rectify` was requested AND actually changed the page
    /// (`auto_rectify` is a no-op on an already-straight page — see
    /// `tesseract_ocr::rectify`'s module docs).
    rectified: bool,
    mean_conf: String,
    low_confidence: bool,
    word_count: usize,
    line_count: usize,
    elapsed_ms: String,
    region_counts: Vec<KindCount>,
    overlay_bg: String,
    overlays: Vec<RegionOverlay>,
    legend: Vec<LegendItem>,
    placed: Vec<RegionPos>,
    fields: Vec<FieldRow>,
    algorithms: Vec<AlgoStep>,
    /// The rendered "A" preview HTML (scan + invisible/searchable word layer),
    /// embedded verbatim into an `<iframe srcdoc>` (Askama escapes it for the
    /// attribute; the browser un-escapes and renders the document).
    preview_a: String,
    /// The rendered "B" preview HTML (the `doc.v1` structural reconstruction).
    preview_b: String,
}

#[derive(Template)]
#[template(path = "debug.html")]
struct DebugTemplate {
    error: Option<String>,
    result: Option<DebugView>,
}

/// The honest "what actually ran" trace — reflects whether the dict beam was
/// active. NO language/script confidence is claimed (OSD is not transcoded);
/// the model + mean confidence are the honest proxy (surfaced in the stats).
fn algorithms_trace(dbg: &OcrDebugOutcome) -> Vec<AlgoStep> {
    let decode = if dbg.dict_on {
        "CTC beam decode with the production dict beam (dict_ratio 2.25, cert_offset \u{2212}0.085, worst_dict_cert \u{2212}25) over the word / punc / number DAWGs."
    } else {
        "CTC beam decode, plain \u{2014} no dictionary loaded (dict_ratio 1.0, cert_offset 0.0)."
    };
    vec![
        AlgoStep {
            name: "1. Image decode".into(),
            detail: "pure-Rust `image` crate (PNG / JPEG / WebP / TIFF / GIF / BMP / PNM) \u{2192} 8-bit grey. Zero C libraries.".into(),
        },
        AlgoStep {
            name: "2. Line segmentation".into(),
            detail: "make-row row crops (seg-approx): the page is split into text-line bands, recognized top-to-bottom.".into(),
        },
        AlgoStep {
            name: "3. Binarization".into(),
            detail: "global Otsu (fixed-128 fallback) for the region / table classifier. Sauvola adaptive binarization is transcoded and available, but is NOT the segmentation default.".into(),
        },
        AlgoStep {
            name: "4. LSTM forward".into(),
            detail: "int8 network forward \u{2014} byte-parity transcode of libtesseract (no leptonica / OpenCV at runtime).".into(),
        },
        AlgoStep {
            name: "5. Decode".into(),
            detail: decode.into(),
        },
        AlgoStep {
            name: "6. Region classification".into(),
            detail: "page furniture (header / footer), XY-cut layout blocks (reading order), `pixGetRegionsBinary` halftone figures, `pixDecideIfTable` table detection \u{2014} all byte-parity leptonica leaves.".into(),
        },
        AlgoStep {
            name: "7. Table structure".into(),
            detail: "whitespace-column cell-grid reconstruction over the recognized words (a doc.v1 output surface, not a TableFinder transcode).".into(),
        },
        AlgoStep {
            name: "8. Field harvest".into(),
            detail: "German-invoice typed fields (IBAN mod-97, amount \u{2192} cents, GUID shape).".into(),
        },
    ]
}

/// The `(model file, human label)` shown in the debug stats for a canonical
/// language code (`"eng"` / `"deu"`, [`OcrDebugOutcome::lang`] — always one of
/// those two, `AppState::model` never returns anything else).
fn model_label(lang: &str) -> (&'static str, &'static str) {
    match lang {
        "deu" => ("deu.lstm", "German (deu)"),
        _ => ("eng.lstm", "English (eng)"),
    }
}

/// Build the full debug view from one recognition pass. A (searchable
/// facsimile) and B (doc.v1 reconstruction) are BOTH derived from the same
/// `OcrDebugOutcome` (one `doc.v1` + one grey raster), so the two previews share
/// one coordinate space and line up side by side. Heavy synchronous work — run
/// via `spawn_blocking`.
fn build_debug_view(
    state: &AppState,
    bytes: &[u8],
    lang: Option<&str>,
    rectify: bool,
) -> Result<DebugView, String> {
    let dbg = ocr_image_bytes_debug(state, bytes, lang, rectify)?;
    let doc: docv1::Doc =
        serde_json::from_str(&dbg.doc_json).map_err(|e| format!("parsing doc.v1: {e}"))?;
    let (w, h) = (dbg.width, dbg.height);

    // The honest algorithms trace — computed before the grey raster is moved out
    // of `dbg` (it only needs `dbg.dict_on`; the remaining stats read below are
    // all `Copy`, so a later partial move of `dbg.grey` leaves them accessible).
    let algorithms = algorithms_trace(&dbg);

    // ONE grey raster feeds both the "B" figure crops and the "A" background.
    let grey = GreyImage {
        data: dbg.grey,
        w,
        h,
    };

    // B — the doc.v1 structural reconstruction (borrows the raster for figures).
    let mut layout_b = doc_v1_layout(&dbg.doc_json, std::slice::from_ref(&grey))
        .map_err(|e| format!("reconstructing layout: {e}"))?;
    layout_b.dpi = 72;
    let preview_b = render_preview_html(&layout_b);

    // The overlay panel's browser-safe scan background (grey \u{2192} PNG data URI).
    let overlay_bg = grey_png_data_uri(&grey).unwrap_or_default();

    // Region overlays + counts + legend + table/figure positions, from doc.v1.
    let mut overlays: Vec<RegionOverlay> = Vec::new();
    let mut region_counts: Vec<KindCount> = Vec::new();
    let mut legend: Vec<LegendItem> = Vec::new();
    let mut placed: Vec<RegionPos> = Vec::new();
    let mut fields: Vec<FieldRow> = Vec::new();
    if let Some(page) = doc.pages.first() {
        for region in &page.regions {
            let color = region_color(&region.kind);
            let bb = to_u32_bbox(region.bbox);
            if let Some(css) = pct_css(bb, w, h) {
                overlays.push(RegionOverlay {
                    kind: region.kind.clone(),
                    color,
                    css,
                });
            }
            match region_counts.iter_mut().find(|c| c.kind == region.kind) {
                Some(c) => c.count += 1,
                None => region_counts.push(KindCount {
                    kind: region.kind.clone(),
                    count: 1,
                }),
            }
            if !legend.iter().any(|l| l.kind == region.kind) {
                legend.push(LegendItem {
                    kind: region.kind.clone(),
                    color,
                });
            }
            if region.kind == "table" || region.kind == "figure" {
                let label = if region.kind == "table" && (region.rows > 0 || region.cols > 0) {
                    format!("table {}\u{00d7}{}", region.rows, region.cols)
                } else {
                    region.kind.clone()
                };
                placed.push(RegionPos {
                    kind: label,
                    bbox: bbox_str(bb),
                });
            }
        }
        for f in &page.fields {
            fields.push(FieldRow {
                key: f.key.clone(),
                value: f.value.clone(),
                bbox: bbox_str(to_u32_bbox(f.bbox)),
            });
        }
    }

    // A — the searchable facsimile (moves the raster in as the background).
    let words = placed_words_from_doc(&doc);
    let page_ocr = PageOcr { grey, words };
    let mut layout_a = searchable_layout(vec![page_ocr]);
    layout_a.dpi = 72;
    let preview_a = render_preview_html(&layout_a);

    let (model, lang) = model_label(dbg.lang);
    Ok(DebugView {
        width: w,
        height: h,
        model,
        lang,
        network_spec: dbg.network_spec,
        null_char: dbg.null_char,
        dict_on: dbg.dict_on,
        rectified: dbg.rectified,
        mean_conf: confidence_str(dbg.mean_conf),
        low_confidence: dbg.low_confidence,
        word_count: dbg.word_count,
        line_count: dbg.line_count,
        elapsed_ms: format!("{:.1}", dbg.elapsed_ms),
        region_counts,
        overlay_bg,
        overlays,
        legend,
        placed,
        fields,
        algorithms,
        preview_a,
        preview_b,
    })
}

/// Build one exported PDF: `structured` → reconstruction "B"
/// ([`doc_v1_layout`] + [`render_pdf`]); otherwise the searchable facsimile "A"
/// ([`render_searchable_pdf`]). Both are laid out at 72 dpi (1 px = 1 pt) so the
/// two exports share the searchable/reconstruction geometry. Returns the PDF
/// bytes + a download filename. Heavy synchronous work — run via
/// `spawn_blocking`.
///
/// `pub(crate)`: also used by [`crate::api`] (the `SearchablePdf`/
/// `StructuredPdf` connector actions) — one PDF-building path for both the
/// HTML demo and the machine API.
pub(crate) fn build_pdf(
    state: &AppState,
    bytes: &[u8],
    structured: bool,
    lang: Option<&str>,
    rectify: bool,
) -> Result<(Vec<u8>, &'static str), String> {
    let dbg = ocr_image_bytes_debug(state, bytes, lang, rectify)?;
    let grey = GreyImage {
        data: dbg.grey,
        w: dbg.width,
        h: dbg.height,
    };
    if structured {
        let mut layout = doc_v1_layout(&dbg.doc_json, std::slice::from_ref(&grey))
            .map_err(|e| format!("reconstructing layout: {e}"))?;
        layout.dpi = 72;
        let (pdf, _report) = render_pdf(&layout).map_err(|e| format!("rendering PDF: {e}"))?;
        Ok((pdf, "structured.pdf"))
    } else {
        let doc: docv1::Doc =
            serde_json::from_str(&dbg.doc_json).map_err(|e| format!("parsing doc.v1: {e}"))?;
        let words = placed_words_from_doc(&doc);
        let page = PageOcr { grey, words };
        let (pdf, _report) =
            render_searchable_pdf(&[page], 72).map_err(|e| format!("rendering PDF: {e}"))?;
        Ok((pdf, "searchable.pdf"))
    }
}

/// The result of [`read_image_upload`]: the raw image bytes plus the raw
/// `lang` form field, unvalidated — [`AppState::model`] is where an
/// unrecognized/absent value falls back to `eng` — and whether the `rectify`
/// checkbox was ticked.
struct UploadedImage {
    bytes: Vec<u8>,
    lang: Option<String>,
    rectify: bool,
}

/// Read the `file`/`url`/`lang`/`rectify` fields from a multipart upload
/// (File wins over URL) and return the raw image bytes + requested language +
/// rectify flag, or a user-safe error string. Shared by [`pdf`] and
/// [`debug_post`]; the URL arm goes through the same SSRF guard as `/ocr`.
async fn read_image_upload(mut multipart: Multipart) -> Result<UploadedImage, String> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut url: Option<String> = None;
    let mut lang: Option<String> = None;
    // An HTML checkbox only sends its field AT ALL when checked (any value,
    // conventionally "on") — its mere presence is the signal, not its text.
    let mut rectify = false;
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or_default().to_string();
                match name.as_str() {
                    "file" => match field.bytes().await {
                        Ok(b) if !b.is_empty() => file_bytes = Some(b.to_vec()),
                        Ok(_) => {}
                        Err(e) => return Err(format!("upload read error: {e}")),
                    },
                    "url" => {
                        if let Ok(t) = field.text().await {
                            if !t.trim().is_empty() {
                                url = Some(t.trim().to_string());
                            }
                        }
                    }
                    "lang" => {
                        if let Ok(t) = field.text().await {
                            if !t.trim().is_empty() {
                                lang = Some(t.trim().to_string());
                            }
                        }
                    }
                    "rectify" => {
                        // Discard the value (conventionally "on") — the
                        // field's mere presence is the checkbox signal, same
                        // consume-the-body discipline as every other field.
                        let _ = field.text().await;
                        rectify = true;
                    }
                    _ => {}
                }
            }
            Ok(None) => break,
            Err(e) => return Err(format!("malformed upload: {e}")),
        }
    }

    let bytes = if let Some(b) = file_bytes {
        b
    } else if let Some(u) = url {
        fetch_image_url(&u).await?
    } else {
        return Err("please choose an image file or paste an image URL".to_string());
    };
    Ok(UploadedImage {
        bytes,
        lang,
        rectify,
    })
}

/// `application/pdf` attachment response with a download filename. Built by
/// hand (rather than a header tuple) because `Vec<u8>::into_response()` sets
/// `content-type: application/octet-stream`, which must be REPLACED — not
/// appended — with `application/pdf`.
///
/// `pub(crate)`: also used by [`crate::api`] so the JSON/binary API's PDF
/// actions return byte-identical responses to the HTML `/pdf` route.
pub(crate) fn pdf_response(bytes: Vec<u8>, filename: &str) -> Response {
    use axum::http::{header, HeaderValue};
    let mut resp = bytes.into_response();
    let headers = resp.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/pdf"),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{filename}\"")) {
        headers.insert(header::CONTENT_DISPOSITION, v);
    }
    resp
}

/// `POST /pdf` — multipart image → a searchable ("A", default) or structured
/// ("B", `?mode=structured`) PDF download. Same permit + `spawn_blocking`
/// discipline as `/ocr`.
async fn pdf(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PdfQuery>,
    multipart: Multipart,
) -> Response {
    let uploaded = match read_image_upload(multipart).await {
        Ok(u) => u,
        Err(e) => return err_page(e).into_response(),
    };
    let structured = q.is_structured();

    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return err_page("server is shutting down").into_response(),
    };
    let st = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        build_pdf(
            &st,
            &uploaded.bytes,
            structured,
            uploaded.lang.as_deref(),
            uploaded.rectify,
        )
    })
    .await;
    match outcome {
        Ok(Ok((pdf_bytes, filename))) => pdf_response(pdf_bytes, filename),
        Ok(Err(e)) => err_page(e).into_response(),
        Err(e) => {
            eprintln!("pdf: recognition task failed: {e}");
            err_page("recognition failed unexpectedly").into_response()
        }
    }
}

/// `GET /debug` — the upload form for the verbose A-vs-B preview.
async fn debug_get() -> Html<String> {
    render(&DebugTemplate {
        error: None,
        result: None,
    })
}

/// `POST /debug` — multipart image → the verbose preview (A + B side by side,
/// region overlays, stats, algorithms trace). Same permit + `spawn_blocking`
/// discipline as `/ocr`.
async fn debug_post(State(state): State<Arc<AppState>>, multipart: Multipart) -> Html<String> {
    let uploaded = match read_image_upload(multipart).await {
        Ok(u) => u,
        Err(e) => {
            return render(&DebugTemplate {
                error: Some(e),
                result: None,
            })
        }
    };

    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return render(&DebugTemplate {
                error: Some("server is shutting down".to_string()),
                result: None,
            })
        }
    };
    let st = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        build_debug_view(
            &st,
            &uploaded.bytes,
            uploaded.lang.as_deref(),
            uploaded.rectify,
        )
    })
    .await;
    match outcome {
        Ok(Ok(view)) => render(&DebugTemplate {
            error: None,
            result: Some(view),
        }),
        Ok(Err(e)) => render(&DebugTemplate {
            error: Some(e),
            result: None,
        }),
        Err(e) => {
            eprintln!("debug: recognition task failed: {e}");
            render(&DebugTemplate {
                error: Some("recognition failed unexpectedly".to_string()),
                result: None,
            })
        }
    }
}

fn result_of_text(out: OcrOutcome) -> ResultTemplate {
    let datauri = format!(
        "data:text/plain;charset=utf-8;base64,{}",
        base64_encode(out.text.as_bytes())
    );
    ResultTemplate {
        width: out.width,
        height: out.height,
        primary_label: "Characters",
        primary_count: out.char_count,
        line_count: out.line_count,
        elapsed_ms: format!("{:.1}", out.elapsed_ms),
        confidence: confidence_str(out.mean_conf),
        low_confidence: out.low_confidence,
        text: out.text,
        download_datauri: datauri,
        download_filename: "ocr.txt",
    }
}

fn result_of_json(out: OcrJsonOutcome) -> ResultTemplate {
    let datauri = format!(
        "data:application/json;charset=utf-8;base64,{}",
        base64_encode(out.json.as_bytes())
    );
    ResultTemplate {
        width: out.width,
        height: out.height,
        primary_label: "Words",
        primary_count: out.word_count,
        line_count: out.line_count,
        elapsed_ms: format!("{:.1}", out.elapsed_ms),
        confidence: confidence_str(out.mean_conf),
        low_confidence: out.low_confidence,
        text: out.json,
        download_datauri: datauri,
        download_filename: "result.json",
    }
}

/// Standard base64 (RFC 4648) — a tiny inline encoder so the download link
/// needs no extra dependency.
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::IpAddr;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model")
    }

    #[test]
    fn base64_roundtrips_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn ssrf_guard_blocks_private_loopback_metadata() {
        use crate::fetch::ip_is_blocked;
        for ip in [
            // RFC1918 + loopback + link-local + unspecified.
            "127.0.0.1",
            "10.0.0.1",
            "172.16.5.4",
            "192.168.1.1",
            "169.254.169.254", // AWS/GCP/Azure metadata
            "0.0.0.0",
            "0.1.2.3", // 0.0.0.0/8 "this network"
            // Non-RFC1918 special-use that still targets internal infra.
            "100.64.0.1",      // CGNAT 100.64.0.0/10
            "100.100.100.200", // Alibaba Cloud metadata (inside CGNAT)
            "198.18.0.5",      // benchmarking 198.18.0.0/15
            "192.0.2.10",      // TEST-NET-1
            "198.51.100.10",   // TEST-NET-2
            "203.0.113.10",    // TEST-NET-3
            "224.0.0.1",       // multicast
            "240.0.0.1",       // reserved 240/4
            "255.255.255.255", // broadcast
            // IPv6 forms, incl. IPv4 embeddings.
            "::1",
            "fc00::1",       // ULA
            "fe80::1",       // link-local
            "ff02::1",       // multicast
            "::7f00:1",      // IPv4-compatible ::127.0.0.1
            "2002:7f00:1::", // 6to4 wrapping 127.0.0.1
        ] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(ip_is_blocked(ip), "{ip} must be blocked");
        }
        // Public addresses must be allowed.
        for ip in [
            "1.1.1.1",
            "8.8.8.8",
            "93.184.216.34",
            "2606:4700:4700::1111",
        ] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(!ip_is_blocked(ip), "{ip} must be allowed");
        }
    }

    #[tokio::test]
    async fn fetch_rejects_non_http_scheme() {
        let e = fetch_image_url("file:///etc/passwd").await.unwrap_err();
        assert!(e.contains("scheme"), "got: {e}");
    }

    #[test]
    fn ocr_a_corpus_page_produces_text() {
        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = AppState::load(&dir).expect("load model");
        let page = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm");
        let bytes = std::fs::read(&page).expect("read page_01.pgm");
        let out = ocr_image_bytes(&state, &bytes, None).expect("ocr");
        assert!(out.width > 0 && out.height > 0);
        assert!(
            out.line_count >= 2,
            "expected multiple lines, got {}",
            out.line_count
        );
        assert!(
            out.text.contains("clock"),
            "expected 'clock' from page_01, got: {:?}",
            out.text
        );
    }

    #[test]
    fn ocr_a_corpus_page_produces_json() {
        use crate::ocr::ocr_image_bytes_json;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = AppState::load(&dir).expect("load model");
        let page = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm");
        let bytes = std::fs::read(&page).expect("read page_01.pgm");
        let out = ocr_image_bytes_json(&state, &bytes, None).expect("ocr json");
        assert!(out.width > 0 && out.height > 0);
        assert!(
            out.json.starts_with("{\"schema\":\"tesseract-rs/doc.v1\""),
            "got: {:?}",
            &out.json[..out.json.len().min(80)]
        );
        assert!(
            out.json.contains("\"words\""),
            "expected a words array, got: {:?}",
            out.json
        );
    }

    #[tokio::test]
    async fn get_index_returns_200() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Build a minimal `multipart/form-data` body with a single `file` field.
    /// Returns `(content_type_header, body_bytes)`.
    fn multipart_file(bytes: &[u8]) -> (String, Vec<u8>) {
        const B: &str = "TESSBOUNDARY9f8e7d6c";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{B}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"page.pgm\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{B}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={B}"), body)
    }

    fn page_01_bytes() -> Vec<u8> {
        let page = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm");
        std::fs::read(&page).expect("read page_01.pgm")
    }

    /// Same as [`multipart_file`] plus a `lang` field — for exercising the
    /// language selector through the HTTP layer.
    fn multipart_file_with_lang(bytes: &[u8], lang: &str) -> (String, Vec<u8>) {
        const B: &str = "TESSBOUNDARYLANG1234";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{B}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"page.pgm\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(
            format!("\r\n--{B}\r\nContent-Disposition: form-data; name=\"lang\"\r\n\r\n{lang}")
                .as_bytes(),
        );
        body.extend_from_slice(format!("\r\n--{B}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={B}"), body)
    }

    /// Same as [`multipart_file`] plus a checked `rectify` checkbox field.
    fn multipart_file_with_rectify(bytes: &[u8]) -> (String, Vec<u8>) {
        const B: &str = "TESSBOUNDARYRECTIFY99";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{B}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"page.pgm\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(
            format!("\r\n--{B}\r\nContent-Disposition: form-data; name=\"rectify\"\r\n\r\non")
                .as_bytes(),
        );
        body.extend_from_slice(format!("\r\n--{B}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={B}"), body)
    }

    #[tokio::test]
    async fn get_debug_returns_200() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/debug")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn post_pdf_returns_pdf_bytes() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file(&page_01_bytes());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pdf")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .map(|v| v.to_str().unwrap()),
            Some("application/pdf")
        );
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(out.starts_with(b"%PDF-"), "expected a PDF magic header");
    }

    #[tokio::test]
    async fn post_pdf_structured_returns_pdf_bytes() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file(&page_01_bytes());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pdf?mode=structured")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .map(|v| v.to_str().unwrap()),
            Some("application/pdf")
        );
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(out.starts_with(b"%PDF-"), "expected a PDF magic header");
    }

    #[tokio::test]
    async fn post_debug_renders_a_and_b_panels() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file(&page_01_bytes());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let html = String::from_utf8_lossy(&out);
        // Both panels present, each embedding a rendered preview via srcdoc.
        assert!(html.contains("id=\"panel-a\""), "A panel present");
        assert!(html.contains("id=\"panel-b\""), "B panel present");
        assert_eq!(
            html.matches("iframe class=\"preview\"").count(),
            2,
            "two preview iframes (A + B)"
        );
        assert!(html.contains("srcdoc="), "previews embedded as srcdoc");
        // The region overlay + algorithms trace rendered.
        assert!(
            html.contains("class=\"regionmap\""),
            "region overlay present"
        );
        assert!(html.contains("Algorithms used"), "algorithms trace present");
    }

    /// A plain `/debug` request (no `lang` field) still reports the English
    /// model — the pre-existing default is unchanged by adding language
    /// selection. `page_01` is an English test page; `null_char` and
    /// `network_str` differ per model (`E-OCR-DEU-PARITY-MODEL-AGNOSTIC-1`),
    /// so asserting `110` (eng's null_char) here is a real behavioural check,
    /// not just a string match on the label.
    #[tokio::test]
    async fn post_debug_default_lang_reports_english_model() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file(&page_01_bytes());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let html = String::from_utf8_lossy(&out);
        assert!(html.contains("eng.lstm"), "reports the eng model file");
        assert!(html.contains("English (eng)"), "reports the eng label");
        assert!(html.contains(">110<"), "reports eng's null_char (110)");
    }

    /// `lang=deu` selects the German model end-to-end through the HTTP layer
    /// — the actual ask this test guards: requesting German must not
    /// silently keep running `eng.lstm`. Gracefully skips if `deu.lstm` isn't
    /// in the model dir (the same optional-language degrade `AppState::load`
    /// itself uses), so this test is honest about what it actually proves
    /// when the fixture is absent, rather than passing vacuously.
    #[tokio::test]
    async fn post_debug_with_lang_deu_reports_german_model() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        if !dir.join("deu.lstm").exists() {
            eprintln!("skipping: deu model absent from corpus/model");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file_with_lang(&page_01_bytes(), "deu");
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let html = String::from_utf8_lossy(&out);
        assert!(html.contains("deu.lstm"), "reports the deu model file");
        assert!(html.contains("German (deu)"), "reports the deu label");
        assert!(html.contains(">114<"), "reports deu's null_char (114)");
    }

    /// An unrecognized `lang` value (neither `"eng"` nor `"deu"`) falls back
    /// to English rather than erroring — the same "forgiving field" rule
    /// [`crate::ocr::OutputFormat::from_field`] already uses.
    #[tokio::test]
    async fn post_debug_with_unknown_lang_falls_back_to_english() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file_with_lang(&page_01_bytes(), "klingon");
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let html = String::from_utf8_lossy(&out);
        assert!(html.contains("eng.lstm"), "unknown lang falls back to eng");
    }

    /// `rectify` wires through the whole HTTP stack without erroring, and
    /// `tesseract_ocr::rectify::auto_rectify`'s documented no-op guarantee
    /// holds end-to-end: `page_01.pgm` is a clean digital render (no
    /// rotation/keystone), so the stats panel must honestly report "no
    /// change" rather than claiming a correction that didn't happen. The
    /// correction algorithm itself (does it actually fix a distorted page)
    /// is unit-tested exhaustively in `tesseract_ocr::rectify`'s own test
    /// module — this test's job is only the wiring.
    #[tokio::test]
    async fn post_debug_with_rectify_is_a_no_op_on_a_clean_page() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = router(state);
        let (ct, body) = multipart_file_with_rectify(&page_01_bytes());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let html = String::from_utf8_lossy(&out);
        assert!(
            html.contains("no change"),
            "a clean page must report no change from auto-rectify"
        );
        // Still recognizes correctly — rectify=true must not corrupt an
        // already-good page.
        assert!(html.contains("clock"), "recognition still works: {html}");
    }
}
