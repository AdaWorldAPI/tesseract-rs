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
use axum::http::StatusCode;
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
use tesseract_paperless::matching::MatchAlgorithm;
use tesseract_paperless::store::DocumentRow;

use crate::fetch::fetch_image_url;
use crate::ingest::{ingest, now_unix_ms, IngestOutcome};
use crate::state::{AppState, AutoSlot};

/// Uploads capped at 20 MB — scans run larger than the line-image demos this
/// stack's other web crate targets; both the per-extractor and the raw-body
/// limit must move together, same reasoning as `tesseract-ocr-web::routes`.
const MAX_UPLOAD: usize = 20 * 1024 * 1024;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/upload", post(upload))
        .route("/documents", get(documents))
        .route("/documents/:hash", get(document_detail))
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
        } => {
            println!(
                "ingest: {hash_hex} (guid {}) already held (matched: {matched:?}) -- recognition skipped",
                hex16(document_guid)
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
                query,
                count: rows.len(),
                documents: rows.into_iter().map(DocumentListItem::from).collect(),
                error: None,
            }),
            Err(e) => render(&DocumentsTemplate {
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
                query,
                count: 0,
                documents: Vec::new(),
                error: Some(format!("search failed: {e}")),
            })
        }
        Err(e) => {
            return render(&DocumentsTemplate {
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
        error: Some(error.into()),
    }
}

async fn document_detail(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
) -> Html<String> {
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
        error: None,
    })
}

async fn document_delete(State(state): State<Arc<AppState>>, Path(hash): Path<String>) -> Response {
    // The search-index delete only runs when the archive delete actually
    // succeeded. Doing both unconditionally (codex P1 on PR #88) meant a
    // transient LanceDB failure left the document ARCHIVED but UNSEARCHABLE
    // -- worse than doing nothing, and silently so, since the redirect below
    // looks identical to a real success either way. Gating on success keeps
    // a failed delete's outcome identical to the pre-delete state (both
    // stores untouched) rather than a half-deleted one.
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
        }
        Err(e) => eprintln!("delete {hash} from archive failed: {e} (search index left untouched)"),
    }
    Redirect::to("/documents").into_response()
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

fn back_to_document(hash: &str) -> Redirect {
    Redirect::to(&format!("/documents/{hash}"))
}

// ---- metadata: document actions -------------------------------------------

#[derive(serde::Deserialize)]
struct AssignForm {
    kind: String,
    definition_id: u32,
}

#[derive(serde::Deserialize)]
struct ReviewForm {
    reviewed: String,
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
    Ok(back_to_document(&hash))
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
    Ok(back_to_document(&hash))
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
    Ok(back_to_document(&hash))
}

/// The reviewed toggle: `reviewed=true` marks, `reviewed=false` clears.
async fn document_review(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
    Form(f): Form<ReviewForm>,
) -> Result<Redirect, Response> {
    require_document(&state, &hash).await?;
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
    Ok(back_to_document(&hash))
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
}

fn def_sections(defs: &[DefinitionRow]) -> Vec<DefSection> {
    const SECTIONS: [(MetaKind, &str); 3] = [
        (MetaKind::Correspondent, "Correspondents"),
        (MetaKind::DocumentType, "Document types"),
        (MetaKind::Tag, "Tags"),
    ];
    SECTIONS
        .iter()
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

async fn definitions_page(State(state): State<Arc<AppState>>) -> Response {
    match state.store.meta().definitions().await {
        Ok(defs) => render(&DefinitionsTemplate {
            sections: def_sections(&defs),
            auto_status: auto_status(&state),
        })
        .into_response(),
        Err(e) => fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("archive read failed: {e}"),
        ),
    }
}

#[derive(serde::Deserialize)]
struct CreateForm {
    kind: String,
    name: String,
}

#[derive(serde::Deserialize)]
struct RenameForm {
    kind: String,
    definition_id: u32,
    name: String,
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
}

#[derive(serde::Deserialize)]
struct RetireForm {
    kind: String,
    definition_id: u32,
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
    Ok(Redirect::to("/definitions"))
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
    Ok(Redirect::to("/definitions"))
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
    Ok(Redirect::to("/definitions"))
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
    Ok(Redirect::to("/definitions"))
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

    use axum::body::Body;
    use axum::http::{header, Request};
    use tesseract_paperless::archive_meta::MATCH_AUTO;
    use tesseract_paperless::kv::ContentSha256;
    use tower::ServiceExt as _;

    const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../corpus/model");

    struct Harness {
        state: Arc<AppState>,
        app: Router,
        _dirs: (tempfile::TempDir, tempfile::TempDir),
    }

    async fn harness() -> Harness {
        let archive = tempfile::tempdir().expect("archive tempdir");
        let index = tempfile::tempdir().expect("index tempdir");
        let state = AppState::load(
            FsPath::new(MODEL_DIR),
            &archive.path().to_string_lossy(),
            index.path(),
            FsPath::new("/nonexistent/deepnsm/word_frequency"),
        )
        .await
        .expect("AppState::load");
        let state = Arc::new(state);
        let app = router(state.clone());
        Harness {
            state,
            app,
            _dirs: (archive, index),
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
}
