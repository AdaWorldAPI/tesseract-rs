//! The paperless-ngx-shaped HTTP surface: upload, list, search, view, delete,
//! plus the filing metadata (definitions, assignments, review, AUTO
//! suggestions).

// The metadata handlers return `Result<Redirect, Response>`: the error side is
// a complete error page, built at most once per request. Boxing it would only
// add an allocation to a path that ends the request.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use askama::Template;
use axum::extract::{DefaultBodyLimit, Form, Multipart, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;

use ogar_doc_ir::{DocIr, RegionKind};
use tesseract_paperless::archive_meta::{
    AssignmentRow, DefinitionRow, MetaError, MetaKind, Source,
};
use tesseract_paperless::auto_match::Cue;
use tesseract_paperless::auto_model::{doc_content, index_tokenizer};
use tesseract_paperless::kv::ContentSha256;
use tesseract_paperless::matching::MatchAlgorithm;
use tesseract_paperless::store::DocumentRow;

use crate::fetch::fetch_image_url;
use crate::ingest::{ingest, now_unix_ms, IngestOutcome};
use crate::state::{AppState, AutoSlot};

/// Gives every page template access to the deployment's branding
/// ([`crate::branding`]) as `self.brand()`, so no handler has to pass it in.
macro_rules! branded {
    ($($t:ty),+ $(,)?) => {$(
        impl $t {
            fn brand(&self) -> &'static crate::branding::Branding {
                crate::branding::get()
            }
            /// Whether sign-in is on, so the page offers "Sign out".
            fn sign_in(&self) -> bool {
                crate::auth::enabled()
            }
        }
    )+};
}
branded!(
    IndexTemplate,
    DocumentsTemplate,
    DocumentDetailTemplate,
    DefinitionsTemplate,
    DashboardTemplate
);

// Which main-menu entry a page belongs to (`base.html` marks it current).
impl IndexTemplate {
    fn nav(&self) -> &'static str {
        "upload"
    }
}
impl DocumentsTemplate {
    fn nav(&self) -> &'static str {
        self.nav
    }
}
impl DocumentDetailTemplate {
    fn nav(&self) -> &'static str {
        if self.from_inbox {
            "inbox"
        } else {
            "documents"
        }
    }
}
impl DefinitionsTemplate {
    fn nav(&self) -> &'static str {
        self.nav
    }
}
impl DashboardTemplate {
    fn nav(&self) -> &'static str {
        "dashboard"
    }
}

/// Uploads capped at 20 MB — scans run larger than the line-image demos this
/// stack's other web crate targets; both the per-extractor and the raw-body
/// limit must move together, same reasoning as `tesseract-ocr-web::routes`.
const MAX_UPLOAD: usize = 20 * 1024 * 1024;

/// Liveness for the platform healthcheck. Answers without a session (the sign-in
/// gate exempts it) and reveals nothing about the archive.
async fn healthz() -> &'static str {
    "ok"
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/dashboard", get(dashboard))
        .route("/inbox", get(inbox))
        .route("/healthz", get(healthz))
        .route("/upload", post(upload))
        .route("/documents", get(documents))
        .route("/documents/:hash", get(document_detail))
        .route("/documents/:hash/original", get(document_original))
        .route("/documents/:hash/delete", post(document_delete))
        .route("/documents/:hash/assign", post(document_assign))
        .route("/documents/:hash/unassign", post(document_unassign))
        .route("/documents/:hash/accept", post(document_accept))
        .route("/documents/:hash/review", post(document_review))
        .route("/definitions", get(definitions_page))
        .route("/definitions/create", post(definition_create))
        .route("/definitions/rename", post(definition_rename))
        .route("/definitions/rule", post(definition_rule))
        .route("/definitions/retire", post(definition_retire))
        .route("/auto/mine", post(auto_mine))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD))
        .layer(RequestBodyLimitLayer::new(MAX_UPLOAD))
        .with_state(state)
}

fn render<T: Template>(t: &T) -> Html<String> {
    match t.render() {
        Ok(s) => Html(s),
        Err(e) => {
            eprintln!("template render error: {e}");
            Html("<h1>internal template error</h1>".to_string())
        }
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    error: Option<String>,
}

async fn index() -> Html<String> {
    render(&IndexTemplate { error: None })
}

/// The `file`/`url` upload form's parsed fields — mirrors
/// `tesseract-ocr-web`'s `UploadedImage`, minus the recognition-affecting
/// checkboxes that crate carries (this crate always runs with the
/// dictionary beam on; deskew/rectify are display-quality knobs that don't
/// yet have a place in the archive's stored `DocIr`).
struct UploadedFile {
    bytes: Vec<u8>,
    filename: Option<String>,
    mime: String,
}

async fn read_upload(mut multipart: Multipart) -> Result<UploadedFile, String> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut url: Option<String> = None;

    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or_default().to_string();
                match name.as_str() {
                    "file" => {
                        file_name = field.file_name().map(str::to_string);
                        match field.bytes().await {
                            Ok(b) if !b.is_empty() => file_bytes = Some(b.to_vec()),
                            Ok(_) => {}
                            Err(e) => return Err(format!("upload read error: {e}")),
                        }
                    }
                    "url" => {
                        if let Ok(t) = field.text().await {
                            if !t.trim().is_empty() {
                                url = Some(t.trim().to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(None) => break,
            Err(e) => return Err(format!("malformed upload: {e}")),
        }
    }

    let (bytes, filename, mime) = if let Some(b) = file_bytes {
        (b, file_name, "application/octet-stream".to_string())
    } else if let Some(u) = url {
        let bytes = fetch_image_url(&u).await?;
        (bytes, None, "application/octet-stream".to_string())
    } else {
        return Err("please choose a file or paste an image URL".to_string());
    };
    Ok(UploadedFile {
        bytes,
        filename,
        mime,
    })
}

async fn upload(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    let uploaded = match read_upload(multipart).await {
        Ok(u) => u,
        Err(e) => return render(&IndexTemplate { error: Some(e) }).into_response(),
    };

    match ingest(&state, uploaded.bytes, uploaded.filename, &uploaded.mime).await {
        Ok(outcome) => {
            log_ingest_outcome(&outcome);
            Redirect::to(&format!("/documents/{}", outcome.hash_hex())).into_response()
        }
        Err(e) => render(&IndexTemplate {
            error: Some(format!("ingestion failed: {e}")),
        })
        .into_response(),
    }
}

/// One informational line per ingest, to stdout (Railway logs) — the same
/// "loss must be loud" convention `tesseract-rs`'s own doc-drop findings keep
/// re-learning applied to a quieter case: a duplicate silently skipping
/// recognition, or a low-confidence page landing in the archive unflagged,
/// should both be visible in the log even though neither is an error.
fn log_ingest_outcome(outcome: &IngestOutcome) {
    match outcome {
        IngestOutcome::Stored {
            hash_hex,
            document_guid,
            page_count,
            mean_confidence,
            low_confidence,
            triple_count,
            spo_extraction_ran,
        } => {
            println!(
                "ingest: stored {hash_hex} (guid {}) -- {page_count} page(s), confidence {mean_confidence}{}, \
                 {}",
                hex16(document_guid),
                if *low_confidence { ", LOW CONFIDENCE" } else { "" },
                if *spo_extraction_ran {
                    format!("{triple_count} SPO triple(s) extracted")
                } else {
                    "SPO extraction skipped (no deepnsm vocabulary loaded)".to_string()
                }
            );
        }
        IngestOutcome::Duplicate {
            hash_hex,
            document_guid,
            matched,
            original_added,
        } => {
            println!(
                "ingest: {hash_hex} (guid {}) already held (matched: {matched:?}) -- recognition skipped{}",
                hex16(document_guid),
                if *original_added {
                    "; original kept for a document stored before originals were"
                } else {
                    ""
                }
            );
        }
    }
}

fn hex16(bytes: &[u8; 16]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A row on the document list — a display projection of [`DocumentRow`], not
/// the row itself (Askama renders against plain fields, and the confidence
/// display rule — "—" for zero-word pages — belongs at the render boundary,
/// not inside the stored type).
struct DocumentListItem {
    hash_hex: String,
    filename: String,
    preview: String,
    /// `Some` only in search-result mode: the tantivy-generated `<b>`-
    /// highlighted HTML snippet (see `search.rs`), pre-escaped by
    /// `Snippet::to_html` (which HTML-entity-encodes the surrounding
    /// document text and inserts only the `<b>`/`</b>` wrapper itself --
    /// verified against the fork's own source, not assumed -- so it is safe
    /// to render with `|safe` in the template). `None` for a plain listing,
    /// where the template falls back to the auto-escaped `preview` field.
    snippet_html: Option<String>,
    page_count: u16,
    confidence: String,
    low_confidence: bool,
}

/// Characters of document text shown as a list preview.
const PREVIEW_CHARS: usize = 240;

impl From<DocumentRow> for DocumentListItem {
    fn from(r: DocumentRow) -> Self {
        // Text is derived from the archived `DocIr`, the one stored copy. A
        // row whose IR does not parse lists with an empty preview rather than
        // failing the whole page.
        let text = r.text().unwrap_or_default();
        Self::from_row_with_text(r, &text)
    }
}

impl DocumentListItem {
    /// Build a search-result row: archive metadata from `row`, but the
    /// preview slot filled by the hit's highlighted snippet instead of the
    /// plain first-N-chars preview -- the paperless-ngx-shaped result. The
    /// snippet is cut from the row's archived text, since the index keeps
    /// no copy of it.
    fn from_search_hit(
        row: DocumentRow,
        results: &tesseract_paperless::search::SearchResults,
    ) -> Self {
        let text = row.text().unwrap_or_default();
        Self {
            snippet_html: Some(results.snippet_html(&text)),
            ..Self::from_row_with_text(row, &text)
        }
    }

    /// Build a row from text already derived from its IR, so the IR is
    /// parsed once per row rather than once per field.
    fn from_row_with_text(r: DocumentRow, text: &str) -> Self {
        Self {
            hash_hex: r.content_sha256_hex,
            filename: r.filename.unwrap_or_else(|| "(untitled)".to_string()),
            preview: tesseract_paperless::render::preview_of_text(text, PREVIEW_CHARS),
            snippet_html: None,
            page_count: r.page_count,
            confidence: confidence_str(r.mean_confidence, text),
            low_confidence: r.low_confidence,
        }
    }
}

fn confidence_str(mean_confidence: u32, text: &str) -> String {
    if text.trim().is_empty() {
        "\u{2014}".to_string() // em dash — no words recognized
    } else {
        mean_confidence.to_string()
    }
}

#[derive(Template)]
#[template(path = "documents.html")]
struct DocumentsTemplate {
    /// `"documents"` or `"inbox"`: the same list template serves both.
    nav: &'static str,
    /// Page links for the inbox; `None` on the document list.
    pager: Option<Pager>,
    query: String,
    count: usize,
    documents: Vec<DocumentListItem>,
    error: Option<String>,
}

#[derive(serde::Deserialize)]
struct DocumentsQuery {
    q: Option<String>,
}

const LIST_LIMIT: usize = 200;

async fn documents(
    State(state): State<Arc<AppState>>,
    Query(q): Query<DocumentsQuery>,
) -> Html<String> {
    let query = q.q.unwrap_or_default();

    if query.trim().is_empty() {
        return match state.store.list(LIST_LIMIT).await {
            Ok(rows) => render(&DocumentsTemplate {
                nav: "documents",
                pager: None,
                query,
                count: rows.len(),
                documents: rows.into_iter().map(DocumentListItem::from).collect(),
                error: None,
            }),
            Err(e) => render(&DocumentsTemplate {
                nav: "documents",
                pager: None,
                query,
                count: 0,
                documents: Vec::new(),
                error: Some(format!("archive read failed: {e}")),
            }),
        };
    }

    // Ranked full-text search (BM25 + snippet generation) is CPU/disk work,
    // dispatched off the async runtime the same way OCR and the search-index
    // write already are.
    let st = state.clone();
    let q_for_search = query.clone();
    let results = match tokio::task::spawn_blocking(move || {
        st.search.search(&q_for_search, LIST_LIMIT)
    })
    .await
    {
        Ok(Ok(results)) => results,
        Ok(Err(e)) => {
            return render(&DocumentsTemplate {
                nav: "documents",
                pager: None,
                query,
                count: 0,
                documents: Vec::new(),
                error: Some(format!("search failed: {e}")),
            })
        }
        Err(e) => {
            return render(&DocumentsTemplate {
                nav: "documents",
                pager: None,
                query,
                count: 0,
                documents: Vec::new(),
                error: Some(format!("search task failed: {e}")),
            })
        }
    };

    // Each hit is joined back to its archive row for display metadata the
    // search index does not carry (page_count/confidence/low_confidence).
    // N+1 lookups over a LIST_LIMIT-bounded result set -- a named, honest
    // cost rather than a hidden one; see `ingest.rs`'s doc comment on the
    // store/index consistency gap this join can also surface (a hit with no
    // matching row) via `Ok(None)` below.
    let mut documents = Vec::with_capacity(results.hits.len());
    for hit in &results.hits {
        match state.store.get(&hit.hash_hex).await {
            Ok(Some(row)) => documents.push(DocumentListItem::from_search_hit(row, &results)),
            Ok(None) => eprintln!(
                "search hit {} has no matching archive row (index/archive drift)",
                hit.hash_hex
            ),
            Err(e) => eprintln!("archive read failed for search hit {}: {e}", hit.hash_hex),
        }
    }

    render(&DocumentsTemplate {
        nav: "documents",
        pager: None,
        count: documents.len(),
        query,
        documents,
        error: None,
    })
}

/// One region, flattened for display — the detail page shows a document as a
/// linear reading-order list rather than reconstructing the tree, so a
/// nested [`ogar_doc_ir::Region::children`] walk collapses to a depth-first
/// append here rather than a recursive template (Askama templates cannot
/// recurse into a Rust-side tree without a second `Template` type per level).
struct RegionView {
    kind: &'static str,
    text: String,
}

fn kind_label(k: RegionKind) -> &'static str {
    match k {
        RegionKind::Header => "header",
        RegionKind::Footer => "footer",
        RegionKind::Main => "main",
        RegionKind::Nav => "nav",
        RegionKind::Table => "table",
        RegionKind::Figure => "figure",
        RegionKind::Text => "text",
    }
}

/// One extracted SPO triple, flattened for the detail template's table --
/// the sentence it came from is carried alongside each row rather than
/// grouped, matching [`RegionView`]'s own flat-list-over-tree shape (Askama
/// cannot recurse a Rust-side tree, and one flat table is simpler than a
/// nested per-sentence template here too).
struct TripleView {
    sentence: String,
    subject: String,
    predicate: String,
    /// Empty string for an intransitive triple's `None` object -- Askama
    /// renders `Option<String>` awkwardly in a table cell, and "no object"
    /// is exactly what an empty cell already communicates.
    object: String,
}

/// Parse [`tesseract_paperless::store::DocumentRow::spo_json`] (the shape
/// `tesseract-paperless-web::ingest::spo_beliefs_to_json` writes) into the
/// flat rows the template renders. Malformed/absent JSON degrades to an
/// empty list -- this is a display-only enrichment of an already-successful
/// archive read, never a reason to fail the whole detail page.
fn parse_spo_triples(spo_json: &str) -> Vec<TripleView> {
    let Ok(sentences) = serde_json::from_str::<serde_json::Value>(spo_json) else {
        return Vec::new();
    };
    let Some(sentences) = sentences.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for s in sentences {
        let text = s["text"].as_str().unwrap_or_default().to_string();
        let Some(triples) = s["triples"].as_array() else {
            continue;
        };
        for t in triples {
            let subject = t["subject"].as_str().unwrap_or_default().to_string();
            let predicate = t["predicate"].as_str().unwrap_or_default().to_string();
            if subject.is_empty() && predicate.is_empty() {
                continue;
            }
            out.push(TripleView {
                sentence: text.clone(),
                subject,
                predicate,
                object: t["object"].as_str().unwrap_or_default().to_string(),
            });
        }
    }
    out
}

fn flatten_regions(ir: &DocIr, out: &mut Vec<RegionView>) {
    fn walk(r: &ogar_doc_ir::Region, out: &mut Vec<RegionView>) {
        let mut text = r.text.clone().unwrap_or_default();
        if r.kind == RegionKind::Table && !r.cells.is_empty() {
            let mut rows: Vec<(u8, u8, &str)> = r
                .cells
                .iter()
                .map(|c| (c.row, c.col, c.text.as_str()))
                .collect();
            rows.sort_by_key(|(row, col, _)| (*row, *col));
            let cells: Vec<String> = rows.into_iter().map(|(_, _, t)| t.to_string()).collect();
            text = cells.join(" | ");
        }
        if !text.trim().is_empty() {
            out.push(RegionView {
                kind: kind_label(r.kind),
                text,
            });
        }
        for c in &r.children {
            walk(c, out);
        }
    }
    for page in &ir.pages {
        for r in &page.regions {
            walk(r, out);
        }
    }
}

#[derive(Template)]
#[template(path = "document.html")]
struct DocumentDetailTemplate {
    hash_hex: String,
    filename: String,
    mime: String,
    source: String,
    page_count: u16,
    confidence: String,
    low_confidence: bool,
    ingested_at: String,
    text: String,
    regions: Vec<RegionView>,
    fields: Vec<(String, String)>,
    triples: Vec<TripleView>,
    /// Current assignments and assignable definitions, one group per kind.
    groups: Vec<KindGroup>,
    /// Whether the document is marked reviewed (spec R3a).
    reviewed: bool,
    /// AUTO suggestions for this document (read-only: rendering them stores
    /// nothing, spec G4).
    suggestions: Vec<SuggestionView>,
    /// Why there are no suggestions, when there are none.
    suggestions_note: Option<String>,
    /// Opened from the inbox: the back link and "Mark reviewed, next" lead there.
    from_inbox: bool,
    /// Whether the uploaded original is kept (false for documents stored
    /// before the original-file store).
    has_original: bool,
    error: Option<String>,
}

/// One current assignment, for display.
struct AssignedView {
    definition_id: u32,
    name: String,
    /// `manual` / `rule` / `auto_accepted`.
    source: &'static str,
}

/// One definition offered in an assign `<select>`.
struct OptionView {
    definition_id: u32,
    name: String,
}

/// The metadata card's per-kind block.
struct KindGroup {
    /// The form value ([`MetaKind::as_str`]).
    kind: &'static str,
    label: &'static str,
    assigned: Vec<AssignedView>,
    options: Vec<OptionView>,
}

/// One AUTO suggestion, for display.
struct SuggestionView {
    kind: &'static str,
    label: &'static str,
    definition_id: u32,
    name: String,
    /// NARS expectation as a whole percentage.
    percent: u32,
    evidence: u32,
    /// The cues that fired, already joined for display.
    cues: String,
}

const KIND_LABELS: [(MetaKind, &str); 3] = [
    (MetaKind::Correspondent, "Correspondent"),
    (MetaKind::DocumentType, "Document type"),
    (MetaKind::Tag, "Tags"),
];

fn kind_groups(defs: &[DefinitionRow], assigns: &[AssignmentRow]) -> Vec<KindGroup> {
    KIND_LABELS
        .iter()
        .map(|&(kind, label)| KindGroup {
            kind: kind.as_str(),
            label,
            assigned: assigns
                .iter()
                .filter(|a| a.kind == kind)
                .map(|a| {
                    let name = match defs
                        .iter()
                        .find(|d| d.kind == kind && d.definition_id == a.definition_id)
                    {
                        Some(d) if d.retired => format!("{} (retired)", d.name),
                        Some(d) => d.name.clone(),
                        None => format!("(unknown #{})", a.definition_id),
                    };
                    AssignedView {
                        definition_id: a.definition_id,
                        name,
                        source: a.source.as_str(),
                    }
                })
                .collect(),
            options: defs
                .iter()
                .filter(|d| d.kind == kind && !d.retired)
                .map(|d| OptionView {
                    definition_id: d.definition_id,
                    name: d.name.clone(),
                })
                .collect(),
        })
        .collect()
}

/// A cue as text. The ids inside a [`Cue`] are the model's dense ids, not
/// definition ids, and there is deliberately no reverse vocabulary lookup, so
/// terms and field keys show their id and the three assignment cues show
/// only their kind.
fn cue_label(c: Cue) -> String {
    match c {
        Cue::Term(id) => format!("term {id}"),
        Cue::FieldKey(id) => format!("field key {id}"),
        Cue::Correspondent(_) => "correspondent".to_string(),
        Cue::DocumentType(_) => "document type".to_string(),
        Cue::Tag(_) => "tag".to_string(),
    }
}

/// The suggestions block. Reads the model slot; writes nothing.
fn suggestions_for(
    state: &AppState,
    row: &DocumentRow,
    assigns: &[AssignmentRow],
) -> (Vec<SuggestionView>, Option<String>) {
    match state.auto_slot() {
        AutoSlot::Pending => (
            Vec::new(),
            Some("No suggestions yet (the model is still being mined).".to_string()),
        ),
        AutoSlot::Refused(reason) => (
            Vec::new(),
            Some(format!(
                "No suggestions: the last mine was refused ({reason})."
            )),
        ),
        AutoSlot::Ready(model) => {
            let content = match doc_content(row) {
                Ok(c) => c,
                Err(e) => return (Vec::new(), Some(format!("Suggestions unavailable: {e}"))),
            };
            let tokenize = index_tokenizer(&state.search);
            let views: Vec<SuggestionView> = model
                .suggest_for(assigns, &content, &tokenize)
                .into_iter()
                .map(|s| SuggestionView {
                    kind: s.kind.as_str(),
                    label: KIND_LABELS
                        .iter()
                        .find(|(k, _)| *k == s.kind)
                        .map_or("", |(_, l)| *l),
                    definition_id: s.definition_id,
                    name: s.name,
                    percent: (s.suggestion.truth.expectation() * 100.0)
                        .round()
                        .clamp(0.0, 100.0) as u32,
                    evidence: s.suggestion.evidence,
                    cues: s
                        .suggestion
                        .because
                        .iter()
                        .map(|c| cue_label(*c))
                        .collect::<Vec<_>>()
                        .join(", "),
                })
                .collect();
            if views.is_empty() {
                (
                    views,
                    Some("The model has no suggestion for this document.".to_string()),
                )
            } else {
                (views, None)
            }
        }
    }
}

fn not_found_detail(hash_hex: &str, error: impl Into<String>) -> DocumentDetailTemplate {
    DocumentDetailTemplate {
        hash_hex: hash_hex.to_string(),
        filename: String::new(),
        mime: String::new(),
        source: String::new(),
        page_count: 0,
        confidence: String::new(),
        low_confidence: false,
        ingested_at: String::new(),
        text: String::new(),
        regions: Vec::new(),
        fields: Vec::new(),
        triples: Vec::new(),
        groups: Vec::new(),
        reviewed: false,
        suggestions: Vec::new(),
        suggestions_note: None,
        from_inbox: false,
        has_original: false,
        error: Some(error.into()),
    }
}

#[derive(serde::Deserialize)]
struct DetailQuery {
    /// `inbox` when opened from the inbox; anything else is ignored.
    from: Option<String>,
}

async fn document_detail(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
    Query(dq): Query<DetailQuery>,
) -> Html<String> {
    let from_inbox = dq.from.as_deref() == Some("inbox");
    let row = match state.store.get(&hash).await {
        Ok(Some(r)) => r,
        Ok(None) => return render(&not_found_detail(&hash, "no document with this hash")),
        Err(e) => {
            return render(&not_found_detail(
                &hash,
                format!("archive read failed: {e}"),
            ))
        }
    };
    let ir = match row.doc_ir() {
        Ok(ir) => ir,
        Err(e) => {
            return render(&not_found_detail(
                &hash,
                format!("stored doc.v1 is corrupt: {e:?}"),
            ))
        }
    };

    let text = tesseract_paperless::render::plain_text(&ir);
    let mut regions = Vec::new();
    flatten_regions(&ir, &mut regions);
    let fields = ir
        .fields
        .iter()
        .map(|f| (f.key.clone(), f.value.clone()))
        .collect();
    let triples = row
        .spo_json
        .as_deref()
        .map(parse_spo_triples)
        .unwrap_or_default();

    // Read-only from here to the render (spec G4): this handler never
    // assigns, reviews or re-mines. Metadata is read, suggestions are
    // computed from the current model slot, and nothing is stored.
    let meta = state.store.meta();
    let (defs, assigns, reviewed) = match (
        meta.definitions().await,
        meta.assignments_for(&hash).await,
        meta.is_reviewed(&hash).await,
    ) {
        (Ok(d), Ok(a), Ok(r)) => (d, a, r),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
            return render(&not_found_detail(
                &hash,
                format!("archive metadata read failed: {e}"),
            ))
        }
    };
    let (suggestions, suggestions_note) = suggestions_for(&state, &row, &assigns);
    let has_original =
        ContentSha256::from_hex(&row.content_sha256_hex).is_some_and(|k| state.blobs.contains(&k));

    render(&DocumentDetailTemplate {
        groups: kind_groups(&defs, &assigns),
        reviewed,
        suggestions,
        suggestions_note,
        hash_hex: row.content_sha256_hex,
        filename: row.filename.unwrap_or_else(|| "(untitled)".to_string()),
        mime: row.mime,
        source: row.source,
        page_count: row.page_count,
        confidence: confidence_str(row.mean_confidence, &text),
        low_confidence: row.low_confidence,
        ingested_at: format_unix_ms(row.ingested_at_unix_ms),
        text,
        regions,
        fields,
        triples,
        from_inbox,
        has_original,
        error: None,
    })
}

/// Serve a document's original upload, byte for byte.
///
/// Only for a document the archive holds: a blob without a row (left by a
/// crash between the two writes) is not served. The bytes are re-hashed on
/// read and refused if they no longer match. Always an attachment with
/// `nosniff`, so an upload is never rendered as a page of this origin.
async fn document_original(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
) -> Response {
    let Some(key) = ContentSha256::from_hex(&hash) else {
        return fail(StatusCode::NOT_FOUND, "no document with this hash");
    };
    let row = match state.store.get(&hash).await {
        Ok(Some(r)) => r,
        Ok(None) => return fail(StatusCode::NOT_FOUND, "no document with this hash"),
        Err(e) => {
            return fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("archive read failed: {e}"),
            )
        }
    };
    let st = state.clone();
    let read = match tokio::task::spawn_blocking(move || st.blobs.get(&key)).await {
        Ok(r) => r,
        Err(e) => {
            return fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("original read task failed: {e}"),
            )
        }
    };
    let bytes = match read {
        Ok(Some(b)) => b,
        Ok(None) => {
            return fail(
                StatusCode::NOT_FOUND,
                "the original of this document is not available: it was stored before \
                 originals were kept. Upload the same file again to attach it.",
            )
        }
        Err(e) => {
            eprintln!("original {hash}: {e}");
            return fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                "the stored original failed its integrity check and is not served",
            );
        }
    };
    let name = download_name(row.filename.as_deref(), &hash);
    (
        [
            (header::CONTENT_TYPE, download_mime(&row.mime).to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!(
                    "attachment; filename=\"{}\"; filename*=UTF-8''{}",
                    ascii_fallback(&name),
                    percent_encode_filename(&name)
                ),
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
            (header::CACHE_CONTROL, "private, no-store".to_string()),
        ],
        bytes,
    )
        .into_response()
}

/// The stored MIME type when it is a plain raster image type, else
/// `application/octet-stream`. The stored value comes from the upload request,
/// so it is not trusted to name anything a browser would execute.
fn download_mime(mime: &str) -> &str {
    let plain = mime.len() <= 64
        && mime.strip_prefix("image/").is_some_and(|sub| {
            !sub.is_empty()
                && !sub.contains("svg")
                && sub
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"+-.".contains(&b))
        });
    if plain {
        mime
    } else {
        "application/octet-stream"
    }
}

/// The stored filename without any directory part, or the hash when there is
/// none.
fn download_name(filename: Option<&str>, hash: &str) -> String {
    filename
        .and_then(|f| f.rsplit(['/', '\\']).next())
        .map(str::trim)
        .filter(|f| !f.is_empty() && *f != "." && *f != "..")
        .map_or_else(|| hash.to_string(), str::to_string)
}

/// The filename with every character outside printable ASCII, and the two
/// that would end the quoted string, replaced by `_`.
fn ascii_fallback(name: &str) -> String {
    name.chars()
        .map(|c| {
            if (c.is_ascii_graphic() && c != '"' && c != '\\') || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// RFC 5987 encoding for `filename*`.
fn percent_encode_filename(name: &str) -> String {
    let mut out = String::new();
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || b"!#$&+-.^_`|~".contains(&b) {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

async fn document_delete(State(state): State<Arc<AppState>>, Path(hash): Path<String>) -> Response {
    // The search-index delete only runs when the archive delete actually
    // succeeded. Doing both unconditionally (codex P1 on PR #88) meant a
    // transient LanceDB failure left the document ARCHIVED but UNSEARCHABLE
    // -- worse than doing nothing, and silently so, since the redirect below
    // looks identical to a real success either way. Gating on success keeps
    // a failed delete's outcome identical to the pre-delete state (both
    // stores untouched) rather than a half-deleted one.
    //
    // The hash lock is held across the row and original deletes so a
    // concurrent upload of the same file cannot write its original in between
    // and then lose it to this delete. The sequence runs in its own task that
    // owns the lock: if the request is cancelled (the client goes away), the
    // task still finishes and releases the lock only after the original is
    // gone. Holding the guard in the handler would drop it on cancellation
    // while a started unlink kept running.
    let task = tokio::spawn(delete_document_and_original(state, hash));
    if let Err(e) = task.await {
        eprintln!("delete task failed: {e}");
    }
    Redirect::to("/documents").into_response()
}

/// The delete sequence behind [`document_delete`]: row, search entry, then
/// original, under the document's hash lock.
async fn delete_document_and_original(state: Arc<AppState>, hash: String) {
    let key = ContentSha256::from_hex(&hash);
    let _held = match &key {
        Some(k) => Some(state.hash_lock(k).lock_owned().await),
        None => None,
    };
    match state.store.delete(&hash).await {
        Ok(()) => {
            // Same off-runtime dispatch as every other search-index write.
            let st = state.clone();
            let hash_for_index = hash.clone();
            match tokio::task::spawn_blocking(move || st.search.delete_document(&hash_for_index))
                .await
            {
                Ok(Err(e)) => eprintln!("delete {hash} from search index failed: {e}"),
                Err(e) => eprintln!("delete {hash} from search index task failed: {e}"),
                Ok(Ok(())) => {}
            }
            // The original goes after its row: a failure here leaves an
            // unreferenced blob, never a row without its original.
            if let Some(key) = key {
                let st = state.clone();
                match tokio::task::spawn_blocking(move || st.blobs.delete(&key)).await {
                    Ok(Err(e)) => eprintln!("delete original of {hash} failed: {e}"),
                    Err(e) => eprintln!("delete original of {hash} task failed: {e}"),
                    Ok(Ok(())) => {}
                }
            }
            // Then the S3 copy, record before original (`docs/RECOVERY.md` §4),
            // in the background: the delete answers without waiting for S3.
            // A failure leaves the S3 objects; a later restore would bring the
            // document back, so it is logged with the hash.
            state.replicate_delete(&hash);
        }
        Err(e) => eprintln!("delete {hash} from archive failed: {e} (search index left untouched)"),
    }
}

// ---- metadata: errors ------------------------------------------------------

/// A short error page with a real status code. Reuses the index template's
/// error banner, the same surface `upload` shows its failures on.
fn fail(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        render(&IndexTemplate {
            error: Some(msg.into()),
        }),
    )
        .into_response()
}

fn meta_fail(e: &MetaError) -> Response {
    let status = match e {
        MetaError::NameTaken { .. } => StatusCode::CONFLICT,
        MetaError::UnknownDefinition { .. } => StatusCode::NOT_FOUND,
        MetaError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    fail(status, e.to_string())
}

fn store_fail(e: &tesseract_paperless::store::StoreError) -> Response {
    fail(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("archive write failed: {e}"),
    )
}

/// An unknown kind string is a 400.
fn parse_kind(s: &str) -> Result<MetaKind, Response> {
    MetaKind::parse(s).ok_or_else(|| {
        fail(
            StatusCode::BAD_REQUEST,
            format!("unknown kind {s:?} (expected correspondent, document_type or tag)"),
        )
    })
}

/// A name must have content once trimmed; `MetaStore` accepts an empty one,
/// so the check is here.
fn require_name(name: &str) -> Result<&str, Response> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        Err(fail(StatusCode::BAD_REQUEST, "a name must not be empty"))
    } else {
        Ok(trimmed)
    }
}

/// The document must exist before metadata is written for it, or a typo in
/// a URL would leave orphan rows.
async fn require_document(state: &AppState, hash: &str) -> Result<(), Response> {
    match state.store.get(hash).await {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(fail(StatusCode::NOT_FOUND, "no document with this hash")),
        Err(e) => Err(fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("archive read failed: {e}"),
        )),
    }
}

/// Back to the document page; when the form came from the inbox
/// (`from=inbox`), keep that origin so the inbox actions stay on the page.
fn back_to_document(hash: &str, from: Option<&str>) -> Redirect {
    if from == Some("inbox") {
        Redirect::to(&format!("/documents/{hash}?from=inbox"))
    } else {
        Redirect::to(&format!("/documents/{hash}"))
    }
}

// ---- metadata: document actions -------------------------------------------

#[derive(serde::Deserialize)]
struct AssignForm {
    kind: String,
    definition_id: u32,
    /// `inbox` when the page was opened from the inbox (see `back_to_document`).
    #[serde(default)]
    from: Option<String>,
}

#[derive(serde::Deserialize)]
struct ReviewForm {
    reviewed: String,
    /// `inbox`: after marking reviewed, open the next document still to
    /// review (or the empty inbox). Absent: back to this document.
    #[serde(default)]
    next: Option<String>,
    /// `inbox` when the page was opened from the inbox (see `back_to_document`).
    #[serde(default)]
    from: Option<String>,
}

/// Manual assign: `Source::Manual`. Single-valued kinds replace.
async fn document_assign(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
    Form(f): Form<AssignForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    require_document(&state, &hash).await?;
    state
        .store
        .meta()
        .assign(&hash, kind, f.definition_id, Source::Manual, now_unix_ms())
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(back_to_document(&hash, f.from.as_deref()))
}

async fn document_unassign(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
    Form(f): Form<AssignForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    require_document(&state, &hash).await?;
    state
        .store
        .meta()
        .unassign(&hash, kind, f.definition_id)
        .await
        .map_err(|e| store_fail(&e))?;
    state.replicate_metadata();
    Ok(back_to_document(&hash, f.from.as_deref()))
}

/// Accept an AUTO suggestion: exactly one `AutoAccepted` assignment and
/// NOTHING else. It never marks the document reviewed (spec R3a, G5).
async fn document_accept(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
    Form(f): Form<AssignForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    require_document(&state, &hash).await?;
    state
        .store
        .meta()
        .assign(
            &hash,
            kind,
            f.definition_id,
            Source::AutoAccepted,
            now_unix_ms(),
        )
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(back_to_document(&hash, f.from.as_deref()))
}

/// The reviewed toggle: `reviewed=true` marks, `reviewed=false` clears.
async fn document_review(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
    Form(f): Form<ReviewForm>,
) -> Result<Redirect, Response> {
    require_document(&state, &hash).await?;
    // Validate `next` before anything is written: a refused request changes nothing.
    let to_inbox = match f.next.as_deref() {
        None | Some("") => false,
        Some("inbox") => true,
        Some(other) => {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!("next must be inbox, not {other:?}"),
            ))
        }
    };
    let meta = state.store.meta();
    match f.reviewed.as_str() {
        "true" => meta
            .mark_reviewed(&hash, now_unix_ms())
            .await
            .map_err(|e| store_fail(&e))?,
        "false" => meta
            .mark_unreviewed(&hash)
            .await
            .map_err(|e| store_fail(&e))?,
        other => {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!("reviewed must be true or false, not {other:?}"),
            ))
        }
    }
    // The review flag is written: copy it before either redirect, so a failed
    // inbox lookup below cannot skip the copy.
    state.replicate_metadata();
    if !to_inbox {
        return Ok(back_to_document(&hash, f.from.as_deref()));
    }
    let next = unreviewed_rows(&state)
        .await
        .map_err(|e| store_fail(&e))?
        .into_iter()
        .find(|r| r.content_sha256_hex != hash);
    Ok(match next {
        Some(r) => Redirect::to(&format!("/documents/{}?from=inbox", r.content_sha256_hex)),
        None => Redirect::to("/inbox"),
    })
}

// ---- inbox and dashboard ---------------------------------------------------

/// Every archived document not marked reviewed, newest first. Reads the whole
/// table (`count` rows) and the reviewed set; the archive is small enough for
/// that today, and the result is exact, unlike a filter over the newest
/// `LIST_LIMIT` rows, which would hide older unreviewed documents.
async fn unreviewed_rows(
    state: &AppState,
) -> Result<Vec<DocumentRow>, tesseract_paperless::store::StoreError> {
    let total = state.store.count().await?;
    let reviewed: std::collections::HashSet<String> = state
        .store
        .meta()
        .reviewed_hashes()
        .await?
        .into_iter()
        .collect();
    Ok(state
        .store
        .list(total.max(1))
        .await?
        .into_iter()
        .filter(|r| !reviewed.contains(&r.content_sha256_hex))
        .collect())
}

/// The inbox: documents to review, as paperless-ngx's inbox. Shows at most
/// `LIST_LIMIT`, and counts all of them.
/// Inbox page links: 1-based page, previous and next page when they exist.
struct Pager {
    page: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

impl Pager {
    /// The page shown for `total` documents when `requested` was asked for:
    /// clamped to `1..=pages`, with neighbours only where they exist.
    fn for_total(total: usize, requested: Option<usize>) -> Self {
        let pages = total.div_ceil(LIST_LIMIT).max(1);
        let page = requested.unwrap_or(1).clamp(1, pages);
        Pager {
            page,
            prev: (page > 1).then(|| page - 1),
            next: (page < pages).then(|| page + 1),
        }
    }
}

#[derive(serde::Deserialize)]
struct InboxQuery {
    /// 1-based page of `LIST_LIMIT` documents; absent, 0 or past the end is
    /// clamped to a valid page.
    page: Option<usize>,
}

/// The inbox: documents to review, as paperless-ngx's inbox, `LIST_LIMIT`
/// per page, every page reachable.
async fn inbox(State(state): State<Arc<AppState>>, Query(q): Query<InboxQuery>) -> Html<String> {
    match unreviewed_rows(&state).await {
        Ok(rows) => {
            let pager = Pager::for_total(rows.len(), q.page);
            let page = pager.page;
            render(&DocumentsTemplate {
                nav: "inbox",
                pager: Some(pager),
                query: String::new(),
                count: rows.len(),
                documents: rows
                    .into_iter()
                    .skip((page - 1) * LIST_LIMIT)
                    .take(LIST_LIMIT)
                    .map(DocumentListItem::from)
                    .collect(),
                error: None,
            })
        }
        Err(e) => render(&DocumentsTemplate {
            nav: "inbox",
            pager: None,
            query: String::new(),
            count: 0,
            documents: Vec::new(),
            error: Some(format!("archive read failed: {e}")),
        }),
    }
}

/// One kind's tile on the dashboard.
struct KindStat {
    kind: &'static str,
    label: &'static str,
    /// Definitions of this kind that are not retired.
    definitions: usize,
}

/// One inbox entry on the dashboard.
struct RecentItem {
    hash_hex: String,
    filename: String,
    ingested_at: String,
}

#[derive(Template)]
#[template(path = "dashboard.html")]
struct DashboardTemplate {
    total: usize,
    unreviewed: usize,
    kinds: Vec<KindStat>,
    /// The newest documents to review, at most `DASHBOARD_RECENT`.
    recent: Vec<RecentItem>,
    /// The S3 copy's state, `None` when no copy is configured.
    replica: Option<String>,
    error: Option<String>,
}

/// One line about the S3 copy for the dashboard, or `None` without a copy.
fn replica_summary(state: &AppState) -> Option<String> {
    state.replica.as_ref()?;
    let status = state
        .replica_status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    Some(match (status.checked_at_unix_ms, status.error, status.backlog) {
        (None, _, _) => "S3 copy: first check pending".to_string(),
        (Some(at), Some(e), _) => {
            format!("S3 copy: last check failed ({}): {e}", format_unix_ms(at))
        }
        (Some(at), None, Some(0)) => {
            format!(
                "S3 copy: no documents waiting to be copied (estimate, checked {})",
                format_unix_ms(at)
            )
        }
        (Some(at), None, Some(n)) => format!(
            "S3 copy: about {n} document{} not yet copied (estimate, checked {})",
            if n == 1 { "" } else { "s" },
            format_unix_ms(at)
        ),
        (Some(at), None, None) => format!("S3 copy: checked {}", format_unix_ms(at)),
    })
}

const DASHBOARD_RECENT: usize = 5;

async fn dashboard(State(state): State<Arc<AppState>>) -> Html<String> {
    let read = async {
        let unreviewed = unreviewed_rows(&state).await?;
        let total = state.store.count().await?;
        let defs = state.store.meta().definitions().await?;
        Ok::<_, tesseract_paperless::store::StoreError>((unreviewed, total, defs))
    };
    match read.await {
        Ok((unreviewed, total, defs)) => render(&DashboardTemplate {
            total,
            unreviewed: unreviewed.len(),
            kinds: DEF_SECTIONS
                .iter()
                .map(|&(kind, label)| KindStat {
                    kind: kind.as_str(),
                    label,
                    definitions: defs.iter().filter(|d| d.kind == kind && !d.retired).count(),
                })
                .collect(),
            recent: unreviewed
                .into_iter()
                .take(DASHBOARD_RECENT)
                .map(|r| RecentItem {
                    hash_hex: r.content_sha256_hex,
                    filename: r.filename.unwrap_or_else(|| "(untitled)".to_string()),
                    ingested_at: format_unix_ms(r.ingested_at_unix_ms),
                })
                .collect(),
            replica: replica_summary(&state),
            error: None,
        }),
        Err(e) => render(&DashboardTemplate {
            total: 0,
            unreviewed: 0,
            kinds: Vec::new(),
            recent: Vec::new(),
            replica: replica_summary(&state),
            error: Some(format!("archive read failed: {e}")),
        }),
    }
}

// ---- metadata: definitions page -------------------------------------------

/// The seven paperless-ngx algorithm names, by number.
fn algorithm_name(v: u8) -> &'static str {
    match MatchAlgorithm::from_paperless(v) {
        Some(MatchAlgorithm::None) => "None",
        Some(MatchAlgorithm::Any) => "Any",
        Some(MatchAlgorithm::All) => "All",
        Some(MatchAlgorithm::Literal) => "Literal",
        Some(MatchAlgorithm::Regex) => "Regex",
        Some(MatchAlgorithm::Fuzzy) => "Fuzzy",
        Some(MatchAlgorithm::Auto) => "Auto",
        None => "Unknown",
    }
}

/// One option of a rule form's algorithm `<select>`.
struct AlgOption {
    value: u8,
    name: &'static str,
    selected: bool,
}

/// One definition row on the definitions page.
struct DefView {
    definition_id: u32,
    name: String,
    algorithm_name: &'static str,
    pattern: String,
    case_insensitive: bool,
    retired: bool,
    alg_options: Vec<AlgOption>,
    /// Documents this definition is assigned to.
    documents: usize,
}

struct DefSection {
    kind: &'static str,
    label: &'static str,
    rows: Vec<DefView>,
}

#[derive(Template)]
#[template(path = "definitions.html")]
struct DefinitionsTemplate {
    sections: Vec<DefSection>,
    auto_status: String,
    /// `"definitions"` for all kinds, else the kind's name (menu entry).
    nav: &'static str,
    /// The kind shown, as the forms send it back (`""` for all kinds).
    view: &'static str,
    heading: &'static str,
}

/// The three kinds, in menu order, with their headings.
const DEF_SECTIONS: [(MetaKind, &str); 3] = [
    (MetaKind::Correspondent, "Correspondents"),
    (MetaKind::DocumentType, "Document types"),
    (MetaKind::Tag, "Tags"),
];

fn def_sections(
    defs: &[DefinitionRow],
    assigns: &[AssignmentRow],
    only: Option<MetaKind>,
) -> Vec<DefSection> {
    DEF_SECTIONS
        .iter()
        .filter(|(kind, _)| only.is_none_or(|o| o == *kind))
        .map(|&(kind, label)| DefSection {
            kind: kind.as_str(),
            label,
            rows: defs
                .iter()
                .filter(|d| d.kind == kind)
                .map(|d| DefView {
                    definition_id: d.definition_id,
                    name: d.name.clone(),
                    algorithm_name: algorithm_name(d.match_algorithm),
                    pattern: d.match_pattern.clone(),
                    case_insensitive: d.case_insensitive,
                    retired: d.retired,
                    alg_options: (0u8..=6)
                        .map(|v| AlgOption {
                            value: v,
                            name: algorithm_name(v),
                            selected: v == d.match_algorithm,
                        })
                        .collect(),
                    documents: assigns
                        .iter()
                        .filter(|a| a.kind == kind && a.definition_id == d.definition_id)
                        .count(),
                })
                .collect(),
        })
        .collect()
}

fn auto_status(state: &AppState) -> String {
    match state.auto_slot() {
        AutoSlot::Pending => "Pending: the first mine has not finished.".to_string(),
        AutoSlot::Ready(model) => format!("Ready: {} rules.", model.rule_count()),
        AutoSlot::Refused(reason) => format!("Refused: {reason}"),
    }
}

#[derive(serde::Deserialize)]
struct DefinitionsQuery {
    /// One kind (`correspondent`, `document_type`, `tag`); absent or empty:
    /// all three. Anything else is a 400.
    kind: Option<String>,
}

async fn definitions_page(
    State(state): State<Arc<AppState>>,
    Query(q): Query<DefinitionsQuery>,
) -> Response {
    let only = match q.kind.as_deref() {
        None | Some("") => None,
        Some(k) => match parse_kind(k) {
            Ok(kind) => Some(kind),
            Err(r) => return r,
        },
    };
    let meta = state.store.meta();
    match (meta.definitions().await, meta.all_assignments().await) {
        (Ok(defs), Ok(assigns)) => {
            let (nav, view, heading) = match only {
                None => ("definitions", "", "Definitions"),
                Some(kind) => {
                    let label = DEF_SECTIONS
                        .iter()
                        .find(|(k, _)| *k == kind)
                        .map_or("Definitions", |(_, l)| *l);
                    (kind.as_str(), kind.as_str(), label)
                }
            };
            render(&DefinitionsTemplate {
                sections: def_sections(&defs, &assigns, only),
                auto_status: auto_status(&state),
                nav,
                view,
                heading,
            })
            .into_response()
        }
        (Err(e), _) | (_, Err(e)) => fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("archive read failed: {e}"),
        ),
    }
}

/// Where a definition form returns to: the kind's own page when it came
/// from one (`view` names a known kind), else the page with all kinds. Only
/// a parsed kind reaches the URL, so `view` cannot inject a location.
fn definitions_location(view: Option<&str>) -> String {
    match view.and_then(MetaKind::parse) {
        Some(kind) => format!("/definitions?kind={}", kind.as_str()),
        None => "/definitions".to_string(),
    }
}

#[derive(serde::Deserialize)]
struct CreateForm {
    kind: String,
    name: String,
    /// The page the form came from (see `definitions_location`).
    #[serde(default)]
    view: Option<String>,
}

#[derive(serde::Deserialize)]
struct RenameForm {
    kind: String,
    definition_id: u32,
    name: String,
    /// The page the form came from (see `definitions_location`).
    #[serde(default)]
    view: Option<String>,
}

#[derive(serde::Deserialize)]
struct RuleForm {
    kind: String,
    definition_id: u32,
    /// Wider than `u8` so 7..=255 and beyond reach the 400 check below
    /// instead of a form-parse rejection.
    match_algorithm: u32,
    #[serde(default)]
    match_pattern: String,
    /// An unchecked checkbox is simply absent from the body.
    #[serde(default)]
    case_insensitive: Option<String>,
    /// The page the form came from (see `definitions_location`).
    #[serde(default)]
    view: Option<String>,
}

#[derive(serde::Deserialize)]
struct RetireForm {
    kind: String,
    definition_id: u32,
    /// The page the form came from (see `definitions_location`).
    #[serde(default)]
    view: Option<String>,
}

/// Create a definition. `MetaStore::create_definition` defaults it to AUTO
/// (spec R1, G14); the empty-name check is this handler's.
async fn definition_create(
    State(state): State<Arc<AppState>>,
    Form(f): Form<CreateForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    let name = require_name(&f.name)?;
    state
        .store
        .meta()
        .create_definition(kind, name, now_unix_ms())
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(Redirect::to(&definitions_location(f.view.as_deref())))
}

async fn definition_rename(
    State(state): State<Arc<AppState>>,
    Form(f): Form<RenameForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    let name = require_name(&f.name)?;
    state
        .store
        .meta()
        .rename_definition(kind, f.definition_id, name)
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(Redirect::to(&definitions_location(f.view.as_deref())))
}

async fn definition_rule(
    State(state): State<Arc<AppState>>,
    Form(f): Form<RuleForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    let algorithm = u8::try_from(f.match_algorithm)
        .ok()
        .filter(|v| MatchAlgorithm::from_paperless(*v).is_some())
        .ok_or_else(|| {
            fail(
                StatusCode::BAD_REQUEST,
                format!(
                    "unknown match algorithm {} (expected 0 to 6)",
                    f.match_algorithm
                ),
            )
        })?;
    state
        .store
        .meta()
        .set_rule(
            kind,
            f.definition_id,
            algorithm,
            &f.match_pattern,
            f.case_insensitive.is_some(),
        )
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(Redirect::to(&definitions_location(f.view.as_deref())))
}

async fn definition_retire(
    State(state): State<Arc<AppState>>,
    Form(f): Form<RetireForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    state
        .store
        .meta()
        .retire_definition(kind, f.definition_id)
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(Redirect::to(&definitions_location(f.view.as_deref())))
}

/// Re-mine the AUTO model in the background and go back to the definitions
/// page, which shows the slot's state.
async fn auto_mine(State(state): State<Arc<AppState>>) -> Redirect {
    tokio::spawn(state.clone().remine());
    Redirect::to("/definitions")
}

/// Milliseconds since the Unix epoch -> a plain `YYYY-MM-DD HH:MM:SS UTC`
/// string, hand-rolled rather than pulling `chrono`/`time` for one call site.
fn format_unix_ms(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (h, m, s) = (
        secs_of_day / 3600,
        (secs_of_day / 60) % 60,
        secs_of_day % 60,
    );

    // Civil-from-days (Howard Hinnant's algorithm) — proleptic Gregorian,
    // valid for the entire range this archive will ever store a timestamp
    // in, and avoids a chrono/time dependency for one formatting call site.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m_num = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m_num <= 2 { y + 1 } else { y };

    format!("{y:04}-{m_num:02}-{d:02} {h:02}:{m:02}:{s:02} UTC")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no `BRAND_*` variable set (the test process sets none) the upload
    /// page is the unbranded archive: its name, tagline and accent, no logo
    /// and no "powered by" credit. The branding values come from one source,
    /// so the custom case is covered by `branding`'s own tests.
    #[test]
    fn the_upload_page_renders_the_default_branding() {
        let html = IndexTemplate { error: None }.render().expect("render");
        assert!(html.contains("<title>tesseract-paperless — document archive</title>"));
        assert!(html.contains("<h1>tesseract-paperless</h1>"));
        assert!(html.contains("deduplicated by content hash"));
        assert!(html.contains("background: #2563eb"));
        assert!(!html.contains("{{"), "no unrendered template syntax");
        assert!(!html.contains("powered by tesseract-rs"));
        assert!(!html.contains("class=\"logo\""));
        // The hover state must not re-colour the button: on_accent() picks the
        // text colour for the unfiltered accent, so a filter can push it
        // below 4.5:1 (white on #757575 under brightness(1.08) is ~4.04:1).
        assert!(html.contains("button:hover { outline:"));
        assert!(!html.contains("button:hover { filter"));
    }

    /// A real 16-byte guid renders as exactly 32 lowercase hex chars, in
    /// byte order — the log line's only way of showing which document a
    /// stored/duplicate outcome refers to besides its hash.
    #[test]
    fn hex16_renders_32_lowercase_chars_in_byte_order() {
        let mut g = [0u8; 16];
        g[0] = 0x08;
        g[1] = 0x0b;
        g[15] = 0xff;
        assert_eq!(hex16(&g), "080b00000000000000000000000000ff");
    }

    /// A known instant (2024-01-15 12:30:45 UTC = 1705321845000 ms) — the
    /// Howard Hinnant civil-from-days algorithm checked against a value
    /// computed independently, not just "doesn't panic".
    #[test]
    fn format_unix_ms_matches_a_known_instant() {
        assert_eq!(format_unix_ms(1_705_321_845_000), "2024-01-15 12:30:45 UTC");
    }

    /// The Unix epoch itself — the zero-point boundary.
    #[test]
    fn format_unix_ms_handles_the_epoch() {
        assert_eq!(format_unix_ms(0), "1970-01-01 00:00:00 UTC");
    }

    /// A page with real words never shows the em-dash placeholder, even at
    /// `mean_confidence: 0` (a genuinely low but real score) — proves the
    /// branch is keyed on "were there words", not on the confidence value
    /// itself.
    #[test]
    fn confidence_str_shows_zero_when_words_exist() {
        assert_eq!(confidence_str(0, "hello"), "0");
    }

    /// An empty-text page shows the placeholder regardless of the stored
    /// confidence number (which is meaningless with no words to average).
    #[test]
    fn confidence_str_shows_placeholder_when_text_is_empty() {
        assert_eq!(confidence_str(97, "   "), "\u{2014}");
    }

    /// The exact shape `ingest::spo_beliefs_to_json` writes — parsed back
    /// into flat rows, one per triple, each carrying its own sentence text.
    #[test]
    fn parse_spo_triples_flattens_sentences_into_rows() {
        let json = r#"[
            {"text":"The dog sees the cat.","coverage":1.0,
             "truth":{"frequency":0.9,"confidence":0.75},
             "triples":[{"subject":"dog","predicate":"see","object":"cat"}]},
            {"text":"The dog barks.","coverage":0.8,
             "truth":{"frequency":0.8,"confidence":0.5},
             "triples":[{"subject":"dog","predicate":"bark","object":null}]}
        ]"#;
        let rows = parse_spo_triples(json);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].sentence, "The dog sees the cat.");
        assert_eq!(rows[0].subject, "dog");
        assert_eq!(rows[0].predicate, "see");
        assert_eq!(rows[0].object, "cat");
        // A `null` object must render as an empty cell, not the literal
        // string "null" or a dropped row.
        assert_eq!(rows[1].object, "");
        assert_eq!(rows[1].sentence, "The dog barks.");
    }

    /// A sentence with zero triples contributes NO rows — proven by feeding
    /// a mixed array where only one of two sentences has triples, so a
    /// version that emitted a placeholder row per sentence would fail this
    /// alongside the previous test's row count.
    #[test]
    fn parse_spo_triples_skips_sentences_with_no_triples() {
        let json = r#"[
            {"text":"Zxqvblorptfizz wobbledoop.","coverage":0.0,
             "truth":{"frequency":0.0,"confidence":0.0},"triples":[]},
            {"text":"The dog sees the cat.","coverage":1.0,
             "truth":{"frequency":0.9,"confidence":0.75},
             "triples":[{"subject":"dog","predicate":"see","object":"cat"}]}
        ]"#;
        let rows = parse_spo_triples(json);
        assert_eq!(
            rows.len(),
            1,
            "the zero-triple sentence must contribute no row"
        );
        assert_eq!(rows[0].sentence, "The dog sees the cat.");
    }

    /// Malformed or absent `spo_json` (the "no reasoner ran" state) must
    /// degrade to an empty list, never panic the detail page.
    #[test]
    fn parse_spo_triples_degrades_on_malformed_json() {
        assert!(parse_spo_triples("not json").is_empty());
        assert!(parse_spo_triples("{}").is_empty());
        assert!(parse_spo_triples("[]").is_empty());
    }

    // ---- router harness (spec order step 8) -------------------------------
    //
    // `tower::oneshot` over `router()` with an `AppState` on tempdirs and
    // store rows seeded directly. Nothing here uploads or runs OCR: the model
    // is loaded (that is fine) but never asked to recognize.

    use std::path::Path as FsPath;

    use crate::ingest::IngestError;
    use axum::body::Body;
    use axum::http::{header, Request};
    use tesseract_paperless::archive_meta::MATCH_AUTO;
    use tesseract_paperless::kv::ContentSha256;
    use tower::ServiceExt as _;

    const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../corpus/model");

    struct Harness {
        state: Arc<AppState>,
        app: Router,
        _dirs: (tempfile::TempDir, tempfile::TempDir, tempfile::TempDir),
    }

    async fn harness() -> Harness {
        let archive = tempfile::tempdir().expect("archive tempdir");
        let index = tempfile::tempdir().expect("index tempdir");
        let blobs = tempfile::tempdir().expect("blob tempdir");
        let state = AppState::load(
            FsPath::new(MODEL_DIR),
            &archive.path().to_string_lossy(),
            index.path(),
            blobs.path(),
            FsPath::new("/nonexistent/deepnsm/word_frequency"),
            None,
        )
        .await
        .expect("AppState::load");
        let state = Arc::new(state);
        let app = router(state.clone());
        Harness {
            state,
            app,
            _dirs: (archive, index, blobs),
        }
    }

    /// A one-page IR with a single text region (the shape of
    /// `store.rs`'s `sample_ir`).
    fn sample_ir() -> DocIr {
        DocIr {
            version: ogar_doc_ir::DOC_IR_VERSION.to_string(),
            source: ogar_doc_ir::Provenance::Ocr,
            geometry: ogar_doc_ir::Geometry::DomOrder,
            content_sha256: [0u8; 32],
            mime: "image/png".to_string(),
            pages: vec![ogar_doc_ir::DocPage {
                number: 0,
                width: 100,
                height: 100,
                regions: vec![ogar_doc_ir::Region {
                    kind: RegionKind::Text,
                    bbox: ogar_doc_ir::BBoxRail {
                        tl: ogar_doc_ir::Rail { x: 0, y: 0 },
                        br: ogar_doc_ir::Rail { x: 10, y: 10 },
                    },
                    reading_order: 0,
                    text: Some("Stadtwerke Rechnung Strom".to_string()),
                    cells: Vec::new(),
                    children: Vec::new(),
                }],
            }],
            fields: Vec::new(),
        }
    }

    /// Seed one document straight into the store; returns its hex hash.
    async fn seed(h: &Harness, name: &[u8]) -> String {
        let hash = ContentSha256::of(name);
        h.state
            .store
            .put(&hash, Some("doc.png"), 90, false, &sample_ir(), 1, None)
            .await
            .expect("put");
        format!("{hash:?}")
    }

    fn post_form(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from(body.to_string()))
            .expect("request")
    }

    fn get_req(uri: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .expect("request")
    }

    /// Status, headers and raw body bytes.
    async fn send_raw(
        h: &Harness,
        req: Request<Body>,
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let resp = h.app.clone().oneshot(req).await.expect("oneshot");
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, headers, bytes.to_vec())
    }

    /// A real page the recognizer reads, so the new-document path runs OCR.
    fn page_bytes() -> Vec<u8> {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/pages/page_01.pgm"
        ))
        .expect("corpus page")
    }

    /// Every blob file under the store, excluding nothing: a leftover
    /// temporary file would be counted too.
    fn blob_files(h: &Harness) -> usize {
        let mut n = 0;
        for d in std::fs::read_dir(h.state.blobs.root()).expect("blob root") {
            let d = d.expect("entry").path();
            if d.is_dir() {
                n += std::fs::read_dir(&d).expect("fan-out dir").count();
            }
        }
        n
    }

    fn blob_path(h: &Harness, hash: &str) -> std::path::PathBuf {
        h.state.blobs.root().join(&hash[..2]).join(hash)
    }

    /// Slice B: upload then download returns the exact bytes, as an attachment
    /// that a browser cannot sniff into a page. A second upload of the same
    /// bytes runs no recognition and stores no second blob.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_upload_keeps_its_original_and_a_duplicate_adds_nothing() {
        let h = harness().await;
        let bytes = page_bytes();
        let outcome = ingest(
            &h.state,
            bytes.clone(),
            Some("scan 01.pgm".to_string()),
            "image/x-portable-graymap",
        )
        .await
        .expect("ingest");
        let IngestOutcome::Stored { hash_hex: hash, .. } = outcome else {
            panic!("first upload must be stored, got {outcome:?}");
        };
        assert_eq!(blob_files(&h), 1);

        let (status, headers, body) =
            send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body == bytes, "downloaded bytes differ from the upload");
        assert_eq!(headers[header::CONTENT_TYPE], "image/x-portable-graymap");
        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        assert_eq!(
            headers[header::CONTENT_DISPOSITION],
            "attachment; filename=\"scan 01.pgm\"; filename*=UTF-8''scan%2001.pgm"
        );
        let (_, _, page) = send(&h, get_req(&format!("/documents/{hash}"))).await;
        assert!(page.contains(&format!("href=\"/documents/{hash}/original\"")));

        let row_before = h.state.store.get(&hash).await.expect("get").expect("row");
        let again = ingest(
            &h.state,
            bytes,
            Some("other name.pgm".to_string()),
            "image/png",
        )
        .await
        .expect("ingest again");
        match again {
            IngestOutcome::Duplicate {
                hash_hex,
                original_added,
                ..
            } => {
                assert_eq!(hash_hex, hash);
                assert!(!original_added, "an intact original was rewritten");
            }
            other => panic!("second upload must be a duplicate, got {other:?}"),
        }
        assert_eq!(blob_files(&h), 1, "a duplicate stored a second blob");
        let row_after = h.state.store.get(&hash).await.expect("get").expect("row");
        assert_eq!(
            row_after.ingested_at_unix_ms,
            row_before.ingested_at_unix_ms
        );
        assert_eq!(row_after.filename, row_before.filename);
    }

    /// Slice B: a document stored before the blob store has a row and no
    /// original. Its detail page says so and the download is 404. Uploading
    /// the same bytes keeps them as its original and changes neither the row
    /// nor its metadata.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_duplicate_of_a_document_without_original_attaches_it() {
        let h = harness().await;
        let legacy = b"legacy upload bytes";
        let hash = seed(&h, legacy).await;
        let def = h
            .state
            .store
            .meta()
            .create_definition(MetaKind::Tag, "Alt", 1)
            .await
            .expect("create");
        let (status, _, _) = send(
            &h,
            post_form(
                &format!("/documents/{hash}/assign"),
                &format!("kind=tag&definition_id={}", def.definition_id),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);

        let (_, _, page) = send(&h, get_req(&format!("/documents/{hash}"))).await;
        assert!(page.contains("The original is not available"));
        assert!(!page.contains(&format!("/documents/{hash}/original")));
        let (status, _, _) = send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let row_before = h.state.store.get(&hash).await.expect("get").expect("row");
        let assigns_before = h
            .state
            .store
            .meta()
            .assignments_for(&hash)
            .await
            .expect("for");
        // Not an image: proves the bytes never reached the decoder or OCR.
        let outcome = ingest(&h.state, legacy.to_vec(), None, "image/png")
            .await
            .expect("ingest");
        assert!(
            matches!(
                outcome,
                IngestOutcome::Duplicate {
                    original_added: true,
                    ..
                }
            ),
            "{outcome:?}"
        );
        let row_after = h.state.store.get(&hash).await.expect("get").expect("row");
        assert_eq!(
            row_after.ingested_at_unix_ms,
            row_before.ingested_at_unix_ms
        );
        assert_eq!(row_after.doc_ir_json, row_before.doc_ir_json);
        assert_eq!(row_after.filename, row_before.filename);
        assert_eq!(
            h.state
                .store
                .meta()
                .assignments_for(&hash)
                .await
                .expect("for"),
            assigns_before
        );

        let (status, _, body) = send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, legacy);
        let (_, _, page) = send(&h, get_req(&format!("/documents/{hash}"))).await;
        assert!(page.contains("Download original"));
    }

    /// Slice B, S-5: when the original cannot be written, no row is written
    /// either. A blob left without a row (a crash after the blob, before the
    /// row) is not served, and the next upload of the same bytes completes
    /// the document using that blob.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_document_row_is_never_written_without_its_original() {
        let h = harness().await;
        let bytes = page_bytes();
        let key = ContentSha256::of(&bytes);
        let hash = format!("{key:?}");

        // A file where the fan-out directory belongs makes the blob write fail.
        let block = h.state.blobs.root().join(&hash[..2]);
        std::fs::write(&block, b"not a directory").expect("block");
        match ingest(&h.state, bytes.clone(), None, "image/x-portable-graymap").await {
            Err(IngestError::Blob(_)) => {}
            other => panic!("expected a blob failure, got {other:?}"),
        }
        assert!(
            h.state.store.get(&hash).await.expect("get").is_none(),
            "a row was written although its original was not"
        );
        std::fs::remove_file(&block).expect("unblock");

        // The crash case: the blob exists, the row does not. Not served.
        h.state.blobs.put(&key, &bytes).expect("orphan blob");
        let (status, _, _) = send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "an orphan blob was served");

        let outcome = ingest(&h.state, bytes.clone(), None, "image/x-portable-graymap")
            .await
            .expect("ingest");
        assert!(
            matches!(outcome, IngestOutcome::Stored { .. }),
            "{outcome:?}"
        );
        assert_eq!(blob_files(&h), 1);
        let (status, _, body) = send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body == bytes);
    }

    /// Slice B: an original survives a restart. A new `AppState` over the same
    /// directories (what a redeploy with the volume does) serves the same
    /// bytes, and their SHA-256 is the document's key.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_original_survives_a_restart() {
        let archive = tempfile::tempdir().expect("archive tempdir");
        let index = tempfile::tempdir().expect("index tempdir");
        let blobs = tempfile::tempdir().expect("blob tempdir");
        let load = || async {
            Arc::new(
                AppState::load(
                    FsPath::new(MODEL_DIR),
                    &archive.path().to_string_lossy(),
                    index.path(),
                    blobs.path(),
                    FsPath::new("/nonexistent/deepnsm/word_frequency"),
                    None,
                )
                .await
                .expect("AppState::load"),
            )
        };
        let bytes = page_bytes();
        let first = load().await;
        let hash = ingest(
            &first,
            bytes.clone(),
            Some("p.pgm".to_string()),
            "image/png",
        )
        .await
        .expect("ingest")
        .hash_hex()
        .to_string();
        drop(first);

        let state = load().await;
        let h = Harness {
            app: router(state.clone()),
            state,
            _dirs: (
                tempfile::tempdir().expect("unused"),
                tempfile::tempdir().expect("unused"),
                tempfile::tempdir().expect("unused"),
            ),
        };
        let (status, _, body) = send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(format!("{:?}", ContentSha256::of(&body)), hash);
        assert!(body == bytes);
    }

    /// Slice B: a delete and an upload of the same document take the same
    /// lock, so the delete cannot remove an original between the upload's
    /// original and row writes. Holding the lock here stalls both.
    #[tokio::test(flavor = "multi_thread")]
    async fn delete_and_upload_of_one_document_wait_for_each_other() {
        let h = harness().await;
        let legacy = b"locked document";
        let hash = seed(&h, legacy).await;
        let key = ContentSha256::from_hex(&hash).expect("hex");
        let lock = h.state.hash_lock(&key);
        let held = lock.lock().await;

        let st = h.state.clone();
        let upload =
            tokio::spawn(async move { ingest(&st, legacy.to_vec(), None, "image/png").await });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            !h.state.blobs.contains(&key),
            "the upload wrote its original without the lock"
        );
        assert!(!upload.is_finished());

        let app = h.app.clone();
        let del = tokio::spawn(async move {
            app.oneshot(post_form(&format!("/documents/{hash}/delete"), ""))
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(!del.is_finished(), "the delete ran without the lock");
        let hash = format!("{key:?}");
        assert!(h.state.store.get(&hash).await.expect("get").is_some());

        drop(held);
        let outcome = upload.await.expect("join").expect("ingest");
        assert!(matches!(
            outcome,
            IngestOutcome::Duplicate {
                original_added: true,
                ..
            }
        ));
        del.await.expect("join").expect("delete");
        // The lock is first-come: the upload attached the original, then the
        // delete removed row and original together.
        assert!(h.state.store.get(&hash).await.expect("get").is_none());
        assert!(!h.state.blobs.contains(&key));
    }

    /// Slice B: a delete that started runs to the end even when its request is
    /// cancelled, so the hash lock is never released halfway through it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_cancelled_delete_still_finishes_under_the_lock() {
        let h = harness().await;
        let original = b"delete survives cancellation";
        let hash = seed(&h, original).await;
        let key = ContentSha256::from_hex(&hash).expect("hex");
        h.state.blobs.put(&key, original).expect("put");

        // Hold the lock so the delete is parked inside its sequence, then
        // cancel the request.
        let lock = h.state.hash_lock(&key);
        let held = lock.lock().await;
        let app = h.app.clone();
        let uri = format!("/documents/{hash}/delete");
        let req = tokio::spawn(async move { app.oneshot(post_form(&uri, "")).await });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        req.abort();
        assert!(req.await.expect_err("aborted").is_cancelled());
        assert!(h.state.store.get(&hash).await.expect("get").is_some());

        drop(held);
        // The delete completes on its own: row and original both gone.
        let mut done = false;
        for _ in 0..50 {
            if h.state.store.get(&hash).await.expect("get").is_none()
                && !h.state.blobs.contains(&key)
            {
                done = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(done, "the cancelled delete did not finish");
        // And it released the lock when it was done.
        assert!(lock.try_lock().is_ok());
    }

    /// Slice B: the duplicate check runs before the lock, so a document can be
    /// deleted after the check said "already held". The upload re-checks under
    /// the lock and stores the document as new instead of pointing at nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_duplicate_whose_document_was_deleted_meanwhile_is_stored_anew() {
        let h = harness().await;
        let bytes = page_bytes();
        let key = ContentSha256::of(&bytes);
        let hash = format!("{key:?}");
        h.state
            .store
            .put(&key, Some("old.pgm"), 90, false, &sample_ir(), 1, None)
            .await
            .expect("seed");

        // The upload passes the gate as a duplicate, then waits on the lock
        // while the document is deleted underneath it.
        let lock = h.state.hash_lock(&key);
        let held = lock.lock().await;
        let st = h.state.clone();
        let b = bytes.clone();
        let upload = tokio::spawn(async move {
            ingest(
                &st,
                b,
                Some("new.pgm".to_string()),
                "image/x-portable-graymap",
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(!upload.is_finished());
        h.state.store.delete(&hash).await.expect("delete");
        drop(held);

        let outcome = upload.await.expect("join").expect("ingest");
        assert!(
            matches!(outcome, IngestOutcome::Stored { .. }),
            "a deleted document was answered as a duplicate: {outcome:?}"
        );
        let row = h.state.store.get(&hash).await.expect("get").expect("row");
        assert_eq!(row.filename.as_deref(), Some("new.pgm"));
        let (status, _, body) = send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body == bytes);
    }

    /// Slice B: altered stored bytes are refused, not served.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_tampered_original_is_refused() {
        let h = harness().await;
        let original = b"original bytes";
        let hash = seed(&h, original).await;
        let key = ContentSha256::from_hex(&hash).expect("hex");
        h.state.blobs.put(&key, original).expect("put");
        std::fs::write(blob_path(&h, &hash), b"altered bytes").expect("tamper");

        let (status, headers, body) =
            send_raw(&h, get_req(&format!("/documents/{hash}/original"))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(headers.get(header::CONTENT_DISPOSITION).is_none());
        assert!(!body.windows(13).any(|w| w == b"altered bytes"));
    }

    /// Slice B: deleting a document removes its original; a malformed or
    /// unknown hash is 404 without touching the filesystem by that name.
    #[tokio::test(flavor = "multi_thread")]
    async fn delete_removes_the_original_and_bad_hashes_are_404() {
        let h = harness().await;
        let original = b"to be deleted";
        let hash = seed(&h, original).await;
        let key = ContentSha256::from_hex(&hash).expect("hex");
        h.state.blobs.put(&key, original).expect("put");

        let (status, _, _) = send(&h, post_form(&format!("/documents/{hash}/delete"), "")).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(
            !h.state.blobs.contains(&key),
            "the original outlived its document"
        );

        for bad in [
            "nothex".to_string(),
            "..%2F..%2Fetc%2Fpasswd".to_string(),
            hash.to_uppercase(),
            format!("{:?}", ContentSha256::of(b"never stored")),
        ] {
            let (status, _, _) = send_raw(&h, get_req(&format!("/documents/{bad}/original"))).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{bad}");
        }
    }

    #[test]
    fn only_plain_image_types_are_served_as_such() {
        assert_eq!(download_mime("image/png"), "image/png");
        assert_eq!(
            download_mime("image/x-portable-graymap"),
            "image/x-portable-graymap"
        );
        for risky in [
            "image/svg+xml",
            "text/html",
            "image/",
            "image/png; charset=utf-8",
            "image/PNG",
            "application/pdf",
            "",
        ] {
            assert_eq!(download_mime(risky), "application/octet-stream", "{risky}");
        }
    }

    #[test]
    fn download_names_drop_directories_and_quote_safely() {
        let h = "ab".repeat(32);
        assert_eq!(download_name(Some("../../etc/passwd"), &h), "passwd");
        assert_eq!(download_name(Some("C:\\scans\\a.png"), &h), "a.png");
        assert_eq!(download_name(Some(".."), &h), h);
        assert_eq!(download_name(None, &h), h);
        assert_eq!(
            ascii_fallback("Rechnung \"März\".png"),
            "Rechnung _M_rz_.png"
        );
        assert_eq!(percent_encode_filename("März 1.png"), "M%C3%A4rz%201.png");
    }

    /// Status, `Location` header (if any) and body text.
    async fn send(h: &Harness, req: Request<Body>) -> (StatusCode, Option<String>, String) {
        let resp = h.app.clone().oneshot(req).await.expect("oneshot");
        let status = resp.status();
        let location = resp
            .headers()
            .get(header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        (
            status,
            location,
            String::from_utf8_lossy(&bytes).to_string(),
        )
    }

    /// G4. Disable: make `document_detail` (or `suggestions_for`) write --
    /// auto-accept a suggestion, call `mark_reviewed`, or `remine`. The
    /// page-rendered assertions keep the test from passing on an error page
    /// that never reached the metadata code.
    #[tokio::test(flavor = "multi_thread")]
    async fn get_document_detail_writes_nothing() {
        let h = harness().await;
        let hash = seed(&h, b"g4").await;
        h.state
            .store
            .meta()
            .create_definition(MetaKind::Tag, "Energie", 1)
            .await
            .expect("create");

        let (status, _, body) = send(&h, get_req(&format!("/documents/{hash}"))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Not reviewed"), "the metadata card rendered");
        assert!(body.contains("Energie"), "the definition is offered");

        let meta = h.state.store.meta();
        assert!(
            meta.assignments_for(&hash).await.expect("for").is_empty(),
            "G4: a GET assigned something"
        );
        assert!(
            !meta.is_reviewed(&hash).await.expect("is"),
            "G4: a GET marked the document reviewed"
        );
    }

    /// G5. Disable: have `document_accept` also call `mark_reviewed`, or
    /// write `Source::Manual`.
    #[tokio::test(flavor = "multi_thread")]
    async fn accept_writes_one_auto_accepted_row_and_no_review() {
        let h = harness().await;
        let hash = seed(&h, b"g5").await;
        let tag = h
            .state
            .store
            .meta()
            .create_definition(MetaKind::Tag, "Energie", 1)
            .await
            .expect("create")
            .definition_id;

        let (status, location, _) = send(
            &h,
            post_form(
                &format!("/documents/{hash}/accept"),
                &format!("kind=tag&definition_id={tag}"),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(
            location.as_deref(),
            Some(format!("/documents/{hash}").as_str())
        );

        let meta = h.state.store.meta();
        let rows = meta.assignments_for(&hash).await.expect("for");
        assert_eq!(rows.len(), 1, "exactly one assignment: {rows:?}");
        assert_eq!(rows[0].kind, MetaKind::Tag);
        assert_eq!(rows[0].definition_id, tag);
        assert_eq!(rows[0].source, Source::AutoAccepted);
        assert!(
            !meta.is_reviewed(&hash).await.expect("is"),
            "G5: accept must not mark the document reviewed"
        );
    }

    /// Disable: make the `"true"` arm of `document_review` a no-op, or swap
    /// the two arms.
    #[tokio::test(flavor = "multi_thread")]
    async fn review_toggle_marks_and_clears() {
        let h = harness().await;
        let hash = seed(&h, b"review").await;
        let uri = format!("/documents/{hash}/review");

        let (status, _, _) = send(&h, post_form(&uri, "reviewed=true")).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(h.state.store.meta().is_reviewed(&hash).await.expect("is"));

        let (status, _, _) = send(&h, post_form(&uri, "reviewed=false")).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(!h.state.store.meta().is_reviewed(&hash).await.expect("is"));

        let (status, _, _) = send(&h, post_form(&uri, "reviewed=maybe")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// G12 via HTTP. Disable: have `document_assign` force `MetaKind::Tag`
    /// (the correspondent ids then resolve to nothing and both POSTs fail),
    /// or drop the delete-first step in `MetaStore::assign`.
    #[tokio::test(flavor = "multi_thread")]
    async fn assigning_a_correspondent_twice_leaves_one_row() {
        let h = harness().await;
        let hash = seed(&h, b"g12").await;
        for name in ["Alpha", "Beta"] {
            h.state
                .store
                .meta()
                .create_definition(MetaKind::Correspondent, name, 1)
                .await
                .expect("create");
        }
        let uri = format!("/documents/{hash}/assign");
        for id in [0, 1] {
            let (status, _, _) = send(
                &h,
                post_form(&uri, &format!("kind=correspondent&definition_id={id}")),
            )
            .await;
            assert_eq!(status, StatusCode::SEE_OTHER);
        }
        let rows = h
            .state
            .store
            .meta()
            .assignments_for(&hash)
            .await
            .expect("for");
        let corr: Vec<_> = rows
            .iter()
            .filter(|r| r.kind == MetaKind::Correspondent)
            .collect();
        assert_eq!(corr.len(), 1, "{rows:?}");
        assert_eq!(
            corr[0].definition_id, 1,
            "the later POST replaced the earlier"
        );
        assert_eq!(corr[0].source, Source::Manual);
    }

    /// G14 via HTTP. Disable: default the created definition to ANY (in
    /// `MetaStore::create_definition` or by a `set_rule` in the handler).
    #[tokio::test(flavor = "multi_thread")]
    async fn create_via_http_yields_an_auto_definition() {
        let h = harness().await;
        let (status, location, _) = send(
            &h,
            post_form("/definitions/create", "kind=tag&name=Energie"),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location.as_deref(), Some("/definitions"));

        let defs = h
            .state
            .store
            .meta()
            .definitions()
            .await
            .expect("definitions");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "Energie");
        assert_eq!(defs[0].kind, MetaKind::Tag);
        assert_eq!(defs[0].match_algorithm, MATCH_AUTO);
    }

    /// Disable: drop `require_name` from `definition_create` -- `MetaStore`
    /// stores the empty name and the POST redirects.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_empty_name_is_refused_and_nothing_is_stored() {
        let h = harness().await;
        let (status, _, body) = send(
            &h,
            post_form("/definitions/create", "kind=tag&name=%20%20%20"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("must not be empty"), "{body}");
        assert!(h
            .state
            .store
            .meta()
            .definitions()
            .await
            .expect("definitions")
            .is_empty());
    }

    /// Disable: drop the `from_paperless` filter in `definition_rule`
    /// (algorithm 7 would be stored), or `parse_kind`'s `ok_or_else`.
    #[tokio::test(flavor = "multi_thread")]
    async fn unknown_kind_and_algorithm_are_400() {
        let h = harness().await;
        let (status, _, _) = send(&h, post_form("/definitions/create", "kind=folder&name=x")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        h.state
            .store
            .meta()
            .create_definition(MetaKind::Tag, "Energie", 1)
            .await
            .expect("create");
        let (status, _, _) = send(
            &h,
            post_form(
                "/definitions/rule",
                "kind=tag&definition_id=0&match_algorithm=7&match_pattern=x",
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let d = &h
            .state
            .store
            .meta()
            .definitions()
            .await
            .expect("definitions")[0];
        assert_eq!(
            d.match_algorithm, MATCH_AUTO,
            "the refused rule stored nothing"
        );

        // The can-succeed half: a valid rule is stored, and an unchecked
        // checkbox (absent field) means case-sensitive.
        let (status, _, _) = send(
            &h,
            post_form(
                "/definitions/rule",
                "kind=tag&definition_id=0&match_algorithm=1&match_pattern=strom",
            ),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let d = &h
            .state
            .store
            .meta()
            .definitions()
            .await
            .expect("definitions")[0];
        assert_eq!(
            (
                d.match_algorithm,
                d.match_pattern.as_str(),
                d.case_insensitive
            ),
            (1, "strom", false)
        );
    }

    /// Disable: register no `/definitions` route, or render the sections
    /// from an empty list. The before/after pair keeps the name from being
    /// present for another reason.
    #[tokio::test(flavor = "multi_thread")]
    async fn definitions_page_lists_a_created_definition() {
        let h = harness().await;
        let (status, _, before) = send(&h, get_req("/definitions")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!before.contains("Energie"));
        assert!(before.contains("Pending"), "the AUTO status is shown");

        send(
            &h,
            post_form("/definitions/create", "kind=tag&name=Energie"),
        )
        .await;
        let (status, _, after) = send(&h, get_req("/definitions")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(after.contains("Energie"));
        assert!(
            after.contains("class=\"badge\">Auto<"),
            "a new definition shows its algorithm badge"
        );
    }

    /// The shell: every page carries the same menu, and exactly the page's own
    /// entry is marked current.
    #[tokio::test(flavor = "multi_thread")]
    async fn every_page_has_the_menu_with_its_own_entry_current() {
        let h = harness().await;
        let hash = seed(&h, b"shell").await;
        for (uri, current) in [
            ("/".to_string(), "/\""),
            ("/dashboard".to_string(), "/dashboard\""),
            ("/inbox".to_string(), "/inbox\""),
            ("/documents".to_string(), "/documents\""),
            (format!("/documents/{hash}"), "/documents\""),
            (format!("/documents/{hash}?from=inbox"), "/inbox\""),
            (
                "/definitions?kind=tag".to_string(),
                "/definitions?kind=tag\"",
            ),
            (
                "/definitions?kind=correspondent".to_string(),
                "/definitions?kind=correspondent\"",
            ),
        ] {
            let (status, _, body) = send(&h, get_req(&uri)).await;
            assert_eq!(status, StatusCode::OK, "{uri}");
            assert!(body.contains("<nav class=\"main\""), "{uri}: no menu");
            assert_eq!(
                body.matches(" aria-current=\"page\">").count(),
                1,
                "{uri}: exactly one current entry"
            );
            assert!(
                body.contains(&format!("href=\"{current} aria-current=\"page\"")),
                "{uri}: wrong current entry"
            );
            assert!(!body.contains("{{"), "{uri}: unrendered template syntax");
        }
        // The page listing all kinds marks no menu entry current.
        let (_, _, body) = send(&h, get_req("/definitions")).await;
        assert_eq!(body.matches(" aria-current=\"page\">").count(), 0);
    }

    /// The inbox lists exactly the documents not marked reviewed; marking one
    /// reviewed takes it off. Disable: drop the `reviewed` filter in
    /// `unreviewed_rows`.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_inbox_lists_exactly_the_unreviewed_documents() {
        let h = harness().await;
        let a = seed(&h, b"inbox-a").await;
        let b = seed(&h, b"inbox-b").await;
        let (_, _, body) = send(&h, get_req("/inbox")).await;
        assert!(body.contains(&a) && body.contains(&b), "both to review");
        assert!(body.contains("2 documents"), "{body}");

        h.state
            .store
            .meta()
            .mark_reviewed(&a, 1)
            .await
            .expect("mark");
        let (_, _, body) = send(&h, get_req("/inbox")).await;
        assert!(!body.contains(&a), "a reviewed document leaves the inbox");
        assert!(body.contains(&b));
        assert!(body.contains("1 document "), "{body}");
        // The document list still shows both.
        let (_, _, body) = send(&h, get_req("/documents")).await;
        assert!(body.contains(&a) && body.contains(&b));
    }

    /// "Mark reviewed, next" marks the document and opens the next one still to
    /// review, then the empty inbox. Without `next` the old behaviour holds;
    /// an unknown `next` is a 400 and marks nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn review_and_next_walks_the_inbox() {
        let h = harness().await;
        let a = seed(&h, b"next-a").await;
        let b = seed(&h, b"next-b").await;

        let (status, loc, _) = send(
            &h,
            post_form(
                &format!("/documents/{a}/review"),
                "reviewed=true&next=inbox",
            ),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(
            loc.as_deref(),
            Some(format!("/documents/{b}?from=inbox").as_str())
        );
        assert!(h.state.store.meta().is_reviewed(&a).await.expect("is"));

        let (_, loc, _) = send(
            &h,
            post_form(
                &format!("/documents/{b}/review"),
                "reviewed=true&next=inbox",
            ),
        )
        .await;
        assert_eq!(loc.as_deref(), Some("/inbox"));

        let (_, loc, _) = send(
            &h,
            post_form(&format!("/documents/{a}/review"), "reviewed=false"),
        )
        .await;
        assert_eq!(loc.as_deref(), Some(format!("/documents/{a}").as_str()));

        let (status, _, _) = send(
            &h,
            post_form(
                &format!("/documents/{a}/review"),
                "reviewed=true&next=elsewhere",
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            !h.state.store.meta().is_reviewed(&a).await.expect("is"),
            "a refused request marks nothing"
        );

        // The button exists only when the page was opened from the inbox.
        let (_, _, from_inbox) = send(&h, get_req(&format!("/documents/{a}?from=inbox"))).await;
        let (_, _, plain) = send(&h, get_req(&format!("/documents/{a}"))).await;
        assert!(from_inbox.contains("Mark reviewed, next"));
        assert!(!plain.contains("Mark reviewed, next"));
    }

    /// The dashboard counts documents, documents to review and live definitions
    /// per kind.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_dashboard_counts_what_the_archive_holds() {
        let h = harness().await;
        let a = seed(&h, b"dash-a").await;
        seed(&h, b"dash-b").await;
        h.state
            .store
            .meta()
            .mark_reviewed(&a, 1)
            .await
            .expect("mark");
        let meta = h.state.store.meta();
        meta.create_definition(MetaKind::Tag, "Energie", 1)
            .await
            .expect("tag");
        let retired = meta
            .create_definition(MetaKind::Tag, "Alt", 1)
            .await
            .expect("tag")
            .definition_id;
        meta.retire_definition(MetaKind::Tag, retired)
            .await
            .expect("retire");

        let (status, _, body) = send(&h, get_req("/dashboard")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.contains("<span class=\"n\">2</span><span class=\"l\">documents</span>"),
            "{body}"
        );
        assert!(
            body.contains("<span class=\"n\">1</span><span class=\"l\">to review</span>"),
            "{body}"
        );
        assert!(
            body.contains("<span class=\"n\">1</span><span class=\"l\">tags</span>"),
            "a retired tag is not counted: {body}"
        );
        assert!(
            !body.contains(&format!("/documents/{a}?from=inbox")),
            "a reviewed document is not in the inbox card"
        );
    }

    /// One kind per page: `?kind=` shows only that section, with document
    /// counts; an unknown kind is a 400; a form returns to the page it came
    /// from, and only a known kind reaches the redirect.
    #[tokio::test(flavor = "multi_thread")]
    async fn definitions_split_by_kind_and_forms_return_to_their_page() {
        let h = harness().await;
        let hash = seed(&h, b"kinds").await;
        let meta = h.state.store.meta();
        let tag = meta
            .create_definition(MetaKind::Tag, "Energie", 1)
            .await
            .expect("tag")
            .definition_id;
        meta.create_definition(MetaKind::Correspondent, "Stadtwerke", 1)
            .await
            .expect("corr");
        meta.assign(&hash, MetaKind::Tag, tag, Source::Manual, 1)
            .await
            .expect("assign");

        let (_, _, tags) = send(&h, get_req("/definitions?kind=tag")).await;
        assert!(
            tags.contains("Energie") && !tags.contains("Stadtwerke"),
            "{tags}"
        );
        assert!(tags.contains("1 document<"), "{tags}");
        let (_, _, all) = send(&h, get_req("/definitions")).await;
        assert!(all.contains("Energie") && all.contains("Stadtwerke"));
        let (status, _, _) = send(&h, get_req("/definitions?kind=storage_path")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (_, loc, _) = send(
            &h,
            post_form("/definitions/create", "kind=tag&name=Wasser&view=tag"),
        )
        .await;
        assert_eq!(loc.as_deref(), Some("/definitions?kind=tag"));
        let (_, loc, _) = send(&h, post_form("/definitions/create", "kind=tag&name=Gas")).await;
        assert_eq!(loc.as_deref(), Some("/definitions"));
        let (_, loc, _) = send(
            &h,
            post_form(
                "/definitions/create",
                "kind=tag&name=Luft&view=https%3A%2F%2Fevil.example",
            ),
        )
        .await;
        assert_eq!(
            loc.as_deref(),
            Some("/definitions"),
            "an unknown view never reaches the URL"
        );
    }

    /// Every inbox document is reachable: pages of `LIST_LIMIT`, clamped,
    /// with neighbours only where they exist.
    #[test]
    fn the_inbox_pager_reaches_every_page() {
        let p = Pager::for_total(0, None);
        assert_eq!((p.page, p.prev, p.next), (1, None, None));
        let p = Pager::for_total(LIST_LIMIT, Some(2));
        assert_eq!((p.page, p.prev, p.next), (1, None, None), "one full page");
        let p = Pager::for_total(LIST_LIMIT + 1, None);
        assert_eq!((p.page, p.prev, p.next), (1, None, Some(2)));
        let p = Pager::for_total(LIST_LIMIT + 1, Some(2));
        assert_eq!((p.page, p.prev, p.next), (2, Some(1), None));
        let p = Pager::for_total(3 * LIST_LIMIT, Some(0));
        assert_eq!(p.page, 1, "page 0 clamps to 1");
        let p = Pager::for_total(3 * LIST_LIMIT, Some(99));
        assert_eq!(
            (p.page, p.prev, p.next),
            (3, Some(2), None),
            "past the end clamps"
        );
    }

    /// A form on a page opened from the inbox returns to it with the inbox
    /// origin; the same form from the document list returns without it.
    #[tokio::test(flavor = "multi_thread")]
    async fn document_forms_keep_the_inbox_origin() {
        let h = harness().await;
        let hash = seed(&h, b"origin").await;
        let tag = h
            .state
            .store
            .meta()
            .create_definition(MetaKind::Tag, "Energie", 1)
            .await
            .expect("tag")
            .definition_id;
        let inbox_loc = format!("/documents/{hash}?from=inbox");
        let plain_loc = format!("/documents/{hash}");
        for (action, body) in [
            ("assign", format!("kind=tag&definition_id={tag}")),
            ("unassign", format!("kind=tag&definition_id={tag}")),
            ("accept", format!("kind=tag&definition_id={tag}")),
            ("review", "reviewed=false".to_string()),
        ] {
            let uri = format!("/documents/{hash}/{action}");
            let (_, loc, _) = send(&h, post_form(&uri, &format!("{body}&from=inbox"))).await;
            assert_eq!(loc.as_deref(), Some(inbox_loc.as_str()), "{action}");
            let (_, loc, _) = send(&h, post_form(&uri, &body)).await;
            assert_eq!(loc.as_deref(), Some(plain_loc.as_str()), "{action}");
            let (_, loc, _) = send(
                &h,
                post_form(&uri, &format!("{body}&from=https%3A%2F%2Fevil.example")),
            )
            .await;
            assert_eq!(
                loc.as_deref(),
                Some(plain_loc.as_str()),
                "{action}: unknown origin ignored"
            );
        }
        // The page opened from the inbox puts the origin into every form.
        let (_, _, page) = send(&h, get_req(&inbox_loc)).await;
        assert!(page.contains("name=\"from\" value=\"inbox\""));
        let (_, _, plain) = send(&h, get_req(&plain_loc)).await;
        assert!(!plain.contains("name=\"from\" value=\"inbox\""));
    }
}
