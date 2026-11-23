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
use axum::{Extension, Router};
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

use crate::auth::Caller;
use crate::customers::Scope;
use crate::fetch::fetch_image_url;
use crate::ingest::{ingest_for_customer, now_unix_ms, IngestError, IngestOutcome};
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
        .route("/documents/:hash/customer", post(document_customer))
        .route(
            "/api/v1/customers/:code/documents",
            get(crate::api::customer_documents),
        )
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

// ---- customer scope --------------------------------------------------------

/// The scope of this request's caller. Without a customer policy every caller
/// sees everything (the archive's earlier behaviour). With one, a caller the
/// gate attached no grants to sees nothing: the start rule makes that
/// unreachable (a policy needs sign-in), and failing closed keeps it so.
fn scope_of(state: &AppState, caller: Option<&Extension<Caller>>) -> Scope {
    match (&state.policy, caller) {
        (None, _) => Scope::All,
        (Some(p), Some(Extension(c))) => p.scope_for(&c.grants),
        (Some(_), None) => Scope::Customers(Default::default()),
    }
}

/// Keep the rows `scope` may see. One read of the customers table, not one
/// per row.
async fn filter_visible(
    state: &AppState,
    scope: &Scope,
    rows: Vec<DocumentRow>,
) -> Result<Vec<DocumentRow>, tesseract_paperless::store::StoreError> {
    if *scope == Scope::All {
        return Ok(rows);
    }
    let customers = state.store.meta().customers().await?;
    Ok(rows
        .into_iter()
        .filter(|r| scope.covers(customers.get(&r.content_sha256_hex).map(String::as_str)))
        .collect())
}

/// The document must exist AND be visible to the caller. Both failures are
/// the same 404, so a caller cannot tell another customer's document from a
/// hash that is not in the archive.
async fn require_visible(state: &AppState, scope: &Scope, hash: &str) -> Result<(), Response> {
    let not_found = || fail(StatusCode::NOT_FOUND, "no document with this hash");
    let read_failed = |e: tesseract_paperless::store::StoreError| {
        fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("archive read failed: {e}"),
        )
    };
    match state.store.get(hash).await {
        Ok(Some(_)) => {}
        Ok(None) => return Err(not_found()),
        Err(e) => return Err(read_failed(e)),
    }
    if *scope == Scope::All {
        return Ok(());
    }
    // A failed customer read answers 404 too: a 500 here, for a hash whose
    // row exists, would tell another customer's document from an unknown one.
    let Ok(customer) = state.store.meta().customer_of(hash).await else {
        return Err(not_found());
    };
    if scope.covers(customer.as_deref()) {
        Ok(())
    } else {
        Err(not_found())
    }
}

/// [`require_visible`], checked while holding the document's hash lock, which
/// is returned to be held across the write that follows. Refiling, deletes,
/// assignments, reviews and an upload's duplicate path all take this lock, so a
/// move to another customer cannot land between the check and the write: a
/// write that waited for the lock sees the document's customer as it is then.
async fn lock_visible(
    state: &AppState,
    scope: &Scope,
    hash: &str,
) -> Result<tokio::sync::OwnedMutexGuard<()>, Response> {
    let Some(key) = ContentSha256::from_hex(hash) else {
        return Err(fail(StatusCode::NOT_FOUND, "no document with this hash"));
    };
    #[cfg(test)]
    crate::state::lock_probe::reached(hash);
    let held = state.hash_lock(&key).lock_owned().await;
    require_visible(state, scope, hash).await?;
    Ok(held)
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    error: Option<String>,
    /// The customers the caller may file under; `None`: any code, or none.
    customer_choices: Option<Vec<String>>,
}

async fn index(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
) -> Html<String> {
    render(&IndexTemplate {
        error: None,
        customer_choices: scope_of(&state, caller.as_ref()).choices(),
    })
}

/// The `file`/`url` upload form's parsed fields — mirrors
/// `tesseract-ocr-web`'s `UploadedImage`, minus the recognition-affecting
/// checkboxes that crate carries (this crate always runs with the
/// dictionary beam on; deskew/rectify are display-quality knobs that don't
/// yet have a place in the archive's stored `DocIr`).
struct UploadedFile {
    source: UploadSource,
    /// The customer to file the document under; `None` when left empty.
    customer: Option<String>,
}

/// Where the upload's bytes come from. A URL is fetched only after the
/// customer check, so a refused upload causes no outbound request.
enum UploadSource {
    File {
        bytes: Vec<u8>,
        filename: Option<String>,
    },
    Url(String),
}

impl UploadSource {
    /// The bytes, the filename and the MIME guess.
    async fn into_bytes(self) -> Result<(Vec<u8>, Option<String>, String), String> {
        let octets = "application/octet-stream".to_string();
        match self {
            Self::File { bytes, filename } => Ok((bytes, filename, octets)),
            Self::Url(u) => Ok((fetch_image_url(&u).await?, None, octets)),
        }
    }
}

async fn read_upload(mut multipart: Multipart) -> Result<UploadedFile, String> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut url: Option<String> = None;
    let mut customer: Option<String> = None;

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
                    "customer" => {
                        if let Ok(t) = field.text().await {
                            if !t.trim().is_empty() {
                                customer = Some(t.trim().to_string());
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

    let source = if let Some(bytes) = file_bytes {
        UploadSource::File {
            bytes,
            filename: file_name,
        }
    } else if let Some(u) = url {
        UploadSource::Url(u)
    } else {
        return Err("please choose a file or paste an image URL".to_string());
    };
    Ok(UploadedFile { source, customer })
}

/// Whether `scope` may file a document under `customer` (`None`: no customer).
/// A scoped caller must name one of its own customers; only [`Scope::All`]
/// may leave a document unfiled.
fn may_file_under(scope: &Scope, customer: Option<&str>) -> Result<(), String> {
    if let Some(c) = customer {
        if !tesseract_paperless::archive_meta::valid_customer_code(c) {
            return Err(format!(
                "{c:?} is not a customer code (1 to {} letters, digits, '-' or '_')",
                tesseract_paperless::archive_meta::MAX_CUSTOMER_CODE
            ));
        }
    }
    match (scope, customer) {
        (Scope::All, _) => Ok(()),
        (Scope::Customers(_), None) => Err("choose the customer this document belongs to".into()),
        (s, Some(c)) if s.covers(Some(c)) => Ok(()),
        (_, Some(c)) => Err(format!("you cannot file documents under customer {c}")),
    }
}

async fn upload(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    multipart: Multipart,
) -> Response {
    let scope = scope_of(&state, caller.as_ref());
    let page_error = |e: String| {
        render(&IndexTemplate {
            error: Some(e),
            customer_choices: scope.choices(),
        })
        .into_response()
    };
    let uploaded = match read_upload(multipart).await {
        Ok(u) => u,
        Err(e) => return page_error(e),
    };
    // Refused before a URL is fetched or recognition is spent.
    if let Err(e) = may_file_under(&scope, uploaded.customer.as_deref()) {
        return page_error(e);
    }
    let (bytes, filename, mime) = match uploaded.source.into_bytes().await {
        Ok(b) => b,
        Err(e) => return page_error(e),
    };
    // Whether the caller may see a document these bytes already belong to is
    // decided inside ingest, under that document's hash lock and before
    // anything is changed, so a concurrent move to another customer cannot
    // slip in between. A re-upload never changes a document's customer.
    match ingest_for_customer(
        &state,
        bytes,
        filename,
        &mime,
        uploaded.customer.as_deref(),
        &scope,
    )
    .await
    {
        Ok(outcome) => {
            log_ingest_outcome(&outcome);
            Redirect::to(&format!("/documents/{}", outcome.hash_hex())).into_response()
        }
        Err(IngestError::Hidden) => page_error(
            "this file is already in the archive, filed under a customer you cannot see".into(),
        ),
        Err(e) => page_error(format!("ingestion failed: {e}")),
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
    caller: Option<Extension<Caller>>,
    Query(q): Query<DocumentsQuery>,
) -> Html<String> {
    let query = q.q.unwrap_or_default();
    let scope = scope_of(&state, caller.as_ref());

    if query.trim().is_empty() {
        // A scoped caller reads the whole table and keeps its own rows, so the
        // newest LIST_LIMIT of THEIR documents show, not their share of the
        // newest LIST_LIMIT overall (the same exactness `unreviewed_rows` keeps).
        let listed = async {
            let limit = if scope == Scope::All {
                LIST_LIMIT
            } else {
                state.store.count().await?.max(1)
            };
            let rows = state.store.list(limit).await?;
            let mut rows = filter_visible(&state, &scope, rows).await?;
            rows.truncate(LIST_LIMIT);
            Ok::<_, tesseract_paperless::store::StoreError>(rows)
        };
        return match listed.await {
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
    // A scoped caller's results are filtered after the search, so it searches
    // every entry in the index: cutting to LIST_LIMIT first would let other
    // customers' documents push its own matches out, and a limit taken from
    // the archive's row count would let stale index entries (no archive row)
    // use up slots. The cut to LIST_LIMIT comes after the filter.
    let scoped = scope != Scope::All;
    let st = state.clone();
    let q_for_search = query.clone();
    let results = match tokio::task::spawn_blocking(move || {
        if scoped {
            st.search.search_every_match(&q_for_search)
        } else {
            st.search.search(&q_for_search, LIST_LIMIT)
        }
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
    // The index holds every customer's documents; a hit the caller may not see
    // is dropped here, before its snippet is cut.
    let customers = if scope == Scope::All {
        Default::default()
    } else {
        match state.store.meta().customers().await {
            Ok(c) => c,
            Err(e) => {
                return render(&DocumentsTemplate {
                    nav: "documents",
                    pager: None,
                    query,
                    count: 0,
                    documents: Vec::new(),
                    error: Some(format!("archive read failed: {e}")),
                })
            }
        }
    };
    let mut documents = Vec::with_capacity(results.hits.len().min(LIST_LIMIT));
    for hit in &results.hits {
        if !scope.covers(customers.get(&hit.hash_hex).map(String::as_str)) {
            continue;
        }
        if documents.len() == LIST_LIMIT {
            break;
        }
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
    /// The customer the document is filed under.
    customer: Option<String>,
    /// The customers the caller may file it under; `None`: any code, or none.
    customer_choices: Option<Vec<String>>,
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
        customer: None,
        customer_choices: None,
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
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
    Query(dq): Query<DetailQuery>,
) -> Response {
    let scope = scope_of(&state, caller.as_ref());
    if let Err(refused) = require_visible(&state, &scope, &hash).await {
        return refused;
    }
    document_detail_page(&state, &scope, hash, dq)
        .await
        .into_response()
}

async fn document_detail_page(
    state: &Arc<AppState>,
    scope: &Scope,
    hash: String,
    dq: DetailQuery,
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
    let (defs, assigns, reviewed, customer) = match (
        meta.definitions().await,
        meta.assignments_for(&hash).await,
        meta.is_reviewed(&hash).await,
        meta.customer_of(&hash).await,
    ) {
        (Ok(d), Ok(a), Ok(r), Ok(c)) => (d, a, r, c),
        (Err(e), _, _, _) | (_, Err(e), _, _) | (_, _, Err(e), _) | (_, _, _, Err(e)) => {
            return render(&not_found_detail(
                &hash,
                format!("archive metadata read failed: {e}"),
            ))
        }
    };
    let (suggestions, suggestions_note) = suggestions_for(state, &row, &assigns);
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
        customer,
        customer_choices: scope.choices(),
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
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
) -> Response {
    let Some(key) = ContentSha256::from_hex(&hash) else {
        return fail(StatusCode::NOT_FOUND, "no document with this hash");
    };
    if let Err(refused) = require_visible(&state, &scope_of(&state, caller.as_ref()), &hash).await {
        return refused;
    }
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

async fn document_delete(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
) -> Response {
    let scope = scope_of(&state, caller.as_ref());
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
    // while a started unlink kept running. The caller's access is checked in
    // that task too, under the lock, so a move to another customer while the
    // delete waits for it is honoured.
    let task = tokio::spawn(delete_document_and_original(state, hash, scope));
    match task.await {
        Ok(Err(refused)) => return refused,
        Ok(Ok(())) => {}
        Err(e) => eprintln!("delete task failed: {e}"),
    }
    Redirect::to("/documents").into_response()
}

/// The delete sequence behind [`document_delete`]: row, search entry, then
/// original, under the document's hash lock, after checking under that lock
/// that `scope` covers the document.
async fn delete_document_and_original(
    state: Arc<AppState>,
    hash: String,
    scope: Scope,
) -> Result<(), Response> {
    let _held = lock_visible(&state, &scope, &hash).await?;
    let key = ContentSha256::from_hex(&hash);
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
    Ok(())
}

// ---- metadata: errors ------------------------------------------------------

/// A short error page with a real status code. Reuses the index template's
/// error banner, the same surface `upload` shows its failures on.
fn fail(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        render(&IndexTemplate {
            error: Some(msg.into()),
            // The upload form under an error page re-checks on submit.
            customer_choices: None,
        }),
    )
        .into_response()
}

fn meta_fail(e: &MetaError) -> Response {
    let status = match e {
        MetaError::NameTaken { .. } => StatusCode::CONFLICT,
        MetaError::UnknownDefinition { .. } => StatusCode::NOT_FOUND,
        MetaError::InvalidCustomer => StatusCode::BAD_REQUEST,
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
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
    Form(f): Form<AssignForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    let _held = lock_visible(&state, &scope_of(&state, caller.as_ref()), &hash).await?;
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
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
    Form(f): Form<AssignForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    let _held = lock_visible(&state, &scope_of(&state, caller.as_ref()), &hash).await?;
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
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
    Form(f): Form<AssignForm>,
) -> Result<Redirect, Response> {
    let kind = parse_kind(&f.kind)?;
    let _held = lock_visible(&state, &scope_of(&state, caller.as_ref()), &hash).await?;
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
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
    Form(f): Form<ReviewForm>,
) -> Result<Redirect, Response> {
    let scope = scope_of(&state, caller.as_ref());
    let held = lock_visible(&state, &scope, &hash).await?;
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
    drop(held);
    // The review flag is written: copy it before either redirect, so a failed
    // inbox lookup below cannot skip the copy.
    state.replicate_metadata();
    if !to_inbox {
        return Ok(back_to_document(&hash, f.from.as_deref()));
    }
    let next = unreviewed_rows(&state, &scope)
        .await
        .map_err(|e| store_fail(&e))?
        .into_iter()
        .find(|r| r.content_sha256_hex != hash);
    Ok(match next {
        Some(r) => Redirect::to(&format!("/documents/{}?from=inbox", r.content_sha256_hex)),
        None => Redirect::to("/inbox"),
    })
}

#[derive(serde::Deserialize)]
struct CustomerForm {
    /// The customer code; empty clears it (only a caller who sees every
    /// customer may leave a document unfiled).
    #[serde(default)]
    customer: String,
    #[serde(default)]
    from: Option<String>,
}

/// File a document under a customer, or clear it. The caller must see the
/// document now and be allowed to file under the new customer, so a scoped
/// caller can move a document only between its own customers.
async fn document_customer(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Path(hash): Path<String>,
    Form(f): Form<CustomerForm>,
) -> Result<Redirect, Response> {
    let scope = scope_of(&state, caller.as_ref());
    let customer = Some(f.customer.trim()).filter(|c| !c.is_empty());
    let _held = lock_visible(&state, &scope, &hash).await?;
    may_file_under(&scope, customer).map_err(|e| fail(StatusCode::FORBIDDEN, e))?;
    state
        .store
        .meta()
        .set_customer(&hash, customer, now_unix_ms())
        .await
        .map_err(|e| meta_fail(&e))?;
    state.replicate_metadata();
    Ok(back_to_document(&hash, f.from.as_deref()))
}

// ---- inbox and dashboard ---------------------------------------------------

/// Every archived document not marked reviewed, newest first. Reads the whole
/// table (`count` rows) and the reviewed set; the archive is small enough for
/// that today, and the result is exact, unlike a filter over the newest
/// `LIST_LIMIT` rows, which would hide older unreviewed documents.
async fn unreviewed_rows(
    state: &AppState,
    scope: &Scope,
) -> Result<Vec<DocumentRow>, tesseract_paperless::store::StoreError> {
    let total = state.store.count().await?;
    let reviewed: std::collections::HashSet<String> = state
        .store
        .meta()
        .reviewed_hashes()
        .await?
        .into_iter()
        .collect();
    let rows = state
        .store
        .list(total.max(1))
        .await?
        .into_iter()
        .filter(|r| !reviewed.contains(&r.content_sha256_hex))
        .collect();
    filter_visible(state, scope, rows).await
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
async fn inbox(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Query(q): Query<InboxQuery>,
) -> Html<String> {
    match unreviewed_rows(&state, &scope_of(&state, caller.as_ref())).await {
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
    Some(
        match (status.checked_at_unix_ms, status.error, status.backlog) {
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
        },
    )
}

const DASHBOARD_RECENT: usize = 5;

async fn dashboard(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
) -> Html<String> {
    let scope = scope_of(&state, caller.as_ref());
    // The replica backlog counts every customer's documents, so only a caller
    // who sees every customer is shown it.
    let replica = if scope == Scope::All {
        replica_summary(&state)
    } else {
        None
    };
    let read = async {
        let unreviewed = unreviewed_rows(&state, &scope).await?;
        // The total a scoped caller sees counts only its own documents.
        let total = if scope == Scope::All {
            state.store.count().await?
        } else {
            let all = state.store.hashes().await?;
            let customers = state.store.meta().customers().await?;
            all.iter()
                .filter(|h| scope.covers(customers.get(*h).map(String::as_str)))
                .count()
        };
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
            replica: replica.clone(),
            error: None,
        }),
        Err(e) => render(&DashboardTemplate {
            total: 0,
            unreviewed: 0,
            kinds: Vec::new(),
            recent: Vec::new(),
            replica,
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
        let html = IndexTemplate {
            error: None,
            customer_choices: None,
        }
        .render()
        .expect("render");
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

    use crate::ingest::{ingest, IngestError};
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
        harness_with(None).await
    }

    async fn harness_with(policy: Option<&str>) -> Harness {
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
        let mut state = state;
        state.policy =
            policy.map(|p| crate::customers::CustomerPolicy::from_json(p).expect("test policy"));
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

    /// A customer row a failed attempt left without its document is not
    /// inherited by a later upload of the same bytes without a customer.
    /// Disable: write the customer in ingest only when one is given.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unfiled_upload_does_not_inherit_an_orphan_customer() {
        let h = harness().await;
        let bytes = page_bytes();
        let hash = format!("{:?}", ContentSha256::of(&bytes));
        let meta = h.state.store.meta();
        meta.set_customer(&hash, Some("K02"), 1)
            .await
            .expect("orphan");
        assert_eq!(
            meta.customer_of(&hash).await.expect("c").as_deref(),
            Some("K02"),
            "the orphan must exist, or this test measures nothing"
        );
        let outcome = ingest(&h.state, bytes, None, "image/x-portable-graymap")
            .await
            .expect("ingest");
        assert!(
            matches!(outcome, IngestOutcome::Stored { .. }),
            "{outcome:?}"
        );
        assert_eq!(
            meta.customer_of(&hash).await.expect("c"),
            None,
            "the new document inherited a customer it was not uploaded for"
        );
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

    // ---- customer separation ------------------------------------------------

    const POLICY: &str = r#"{"version": 1, "grants": [
        {"org": "o1", "role_key": "paperless.k01", "customers": ["K01"]},
        {"org": "o1", "role_key": "paperless.admin", "customers": "*"}
    ]}"#;

    /// `req` as sent by a caller the gate admitted with one role.
    fn as_role(mut req: Request<Body>, role: &str) -> Request<Body> {
        req.extensions_mut().insert(Caller {
            subject: "oidc:t".to_string(),
            grants: vec![("o1".to_string(), role.to_string())],
        });
        req
    }

    /// Three documents: one under K01, one under K02, one unfiled.
    async fn three_customers(h: &Harness) -> (String, String, String) {
        let k01 = seed(h, b"k01 document").await;
        let k02 = seed(h, b"k02 document").await;
        let none = seed(h, b"unfiled document").await;
        let meta = h.state.store.meta();
        meta.set_customer(&k01, Some("K01"), 1).await.expect("k01");
        meta.set_customer(&k02, Some("K02"), 1).await.expect("k02");
        (k01, k02, none)
    }

    /// The list, inbox and dashboard show a scoped caller its own documents
    /// only; a caller with `*` sees all three.
    /// Disable: return `rows` unfiltered from `filter_visible`.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_scoped_caller_lists_only_its_customers_documents() {
        let h = harness_with(Some(POLICY)).await;
        let (k01, k02, none) = three_customers(&h).await;
        for uri in ["/documents", "/inbox"] {
            let (_, _, body) = send(&h, as_role(get_req(uri), "paperless.k01")).await;
            assert!(body.contains(&k01), "{uri}: own document listed");
            assert!(
                !body.contains(&k02),
                "{uri}: another customer's document leaked"
            );
            assert!(!body.contains(&none), "{uri}: an unfiled document leaked");
            let (_, _, body) = send(&h, as_role(get_req(uri), "paperless.admin")).await;
            for hash in [&k01, &k02, &none] {
                assert!(
                    body.contains(hash.as_str()),
                    "{uri}: admin sees every document"
                );
            }
        }
        let (_, _, body) = send(&h, as_role(get_req("/dashboard"), "paperless.k01")).await;
        assert!(!body.contains(&k02) && !body.contains(&none));
    }

    /// Every per-document route answers a document outside the caller's scope
    /// exactly as a hash the archive does not hold: 404, and nothing changes.
    /// Disable: make `require_visible` skip the customer check.
    #[tokio::test(flavor = "multi_thread")]
    async fn another_customers_document_is_not_found_on_every_route() {
        let h = harness_with(Some(POLICY)).await;
        let (k01, k02, _) = three_customers(&h).await;
        h.state
            .store
            .meta()
            .create_definition(MetaKind::Tag, "t", 1)
            .await
            .expect("tag");
        let requests = |hash: &str| {
            vec![
                get_req(&format!("/documents/{hash}")),
                get_req(&format!("/documents/{hash}/original")),
                post_form(
                    &format!("/documents/{hash}/assign"),
                    "kind=tag&definition_id=0",
                ),
                post_form(&format!("/documents/{hash}/review"), "reviewed=true"),
                post_form(&format!("/documents/{hash}/customer"), "customer=K01"),
                post_form(&format!("/documents/{hash}/delete"), ""),
            ]
        };
        for req in requests(&k02) {
            let what = req.uri().to_string();
            let (status, _, _) = send(&h, as_role(req, "paperless.k01")).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{what}");
        }
        let meta = h.state.store.meta();
        assert!(
            h.state.store.get(&k02).await.expect("get").is_some(),
            "not deleted"
        );
        assert!(
            meta.assignments_for(&k02).await.expect("a").is_empty(),
            "not assigned"
        );
        assert!(!meta.is_reviewed(&k02).await.expect("r"), "not reviewed");
        assert_eq!(
            meta.customer_of(&k02).await.expect("c").as_deref(),
            Some("K02"),
            "not moved"
        );
        // The same routes work on the caller's own document.
        let (status, _, _) = send(
            &h,
            as_role(get_req(&format!("/documents/{k01}")), "paperless.k01"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    /// A caller no grant matches, and a request the gate attached no caller
    /// to, see nothing once a policy is set: the check fails closed.
    /// Disable: return `Scope::All` for `(Some(_), None)` in `scope_of`.
    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_matching_grant_nothing_is_visible() {
        let h = harness_with(Some(POLICY)).await;
        let (k01, _, none) = three_customers(&h).await;
        for req in [
            as_role(get_req("/documents"), "paperless.other"),
            get_req("/documents"),
        ] {
            let (_, _, body) = send(&h, req).await;
            assert!(!body.contains(&k01) && !body.contains(&none));
        }
        let (status, _, _) = send(&h, get_req(&format!("/documents/{none}"))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    /// Without a policy the archive keeps its earlier behaviour.
    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_policy_every_caller_sees_every_document() {
        let h = harness().await;
        let (k01, k02, none) = three_customers(&h).await;
        let (_, _, body) = send(&h, get_req("/documents")).await;
        for hash in [&k01, &k02, &none] {
            assert!(body.contains(hash.as_str()));
        }
    }

    /// A scoped caller may move its document only between its own customers
    /// and may not unfile it; a caller with `*` may do both.
    /// Disable: skip `may_file_under` in `document_customer`.
    #[tokio::test(flavor = "multi_thread")]
    async fn filing_is_limited_to_the_callers_own_customers() {
        let h = harness_with(Some(POLICY)).await;
        let (k01, _, _) = three_customers(&h).await;
        let uri = format!("/documents/{k01}/customer");
        for body in ["customer=K02", "customer="] {
            let (status, _, _) = send(&h, as_role(post_form(&uri, body), "paperless.k01")).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        }
        let meta = h.state.store.meta();
        assert_eq!(
            meta.customer_of(&k01).await.expect("c").as_deref(),
            Some("K01")
        );
        let (status, _, _) = send(
            &h,
            as_role(post_form(&uri, "customer=K%2002"), "paperless.admin"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "an invalid code is refused for anyone"
        );
        let (status, _, _) = send(
            &h,
            as_role(post_form(&uri, "customer=K02"), "paperless.admin"),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(
            meta.customer_of(&k01).await.expect("c").as_deref(),
            Some("K02")
        );
        let (status, _, _) =
            send(&h, as_role(post_form(&uri, "customer="), "paperless.admin")).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(meta.customer_of(&k01).await.expect("c"), None);
    }

    fn upload_req(bytes: &[u8], customer: Option<&str>) -> Request<Body> {
        let b = "XBOUNDARYX";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.png\"\r\n\
                 Content-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
        if let Some(c) = customer {
            body.extend_from_slice(
                format!(
                    "--{b}\r\nContent-Disposition: form-data; name=\"customer\"\r\n\r\n{c}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(format!("--{b}--\r\n").as_bytes());
        Request::builder()
            .method("POST")
            .uri("/upload")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={b}"),
            )
            .body(Body::from(body))
            .expect("request")
    }

    /// Re-uploading a file another customer already holds neither moves that
    /// document to the uploader nor links to it; a scoped upload without a
    /// customer is refused before any recognition is spent.
    /// Disable: drop the `Duplicate` visibility check in `upload`, or move
    /// `set_customer` ahead of the `Stored` match.
    #[tokio::test(flavor = "multi_thread")]
    async fn uploads_respect_the_callers_customers() {
        let h = harness_with(Some(POLICY)).await;
        let (_, k02, _) = three_customers(&h).await;
        let (status, location, body) = send(
            &h,
            as_role(upload_req(b"k02 document", Some("K01")), "paperless.k01"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "an error page, not a redirect: {location:?}"
        );
        assert!(body.contains("filed under a customer you cannot see"));
        assert!(!body.contains(&k02), "the page must not name the document");
        assert_eq!(
            h.state
                .store
                .meta()
                .customer_of(&k02)
                .await
                .expect("c")
                .as_deref(),
            Some("K02"),
            "a duplicate upload must not move the document"
        );

        let before = h.state.store.count().await.expect("count");
        let (_, _, body) = send(&h, as_role(upload_req(b"new bytes", None), "paperless.k01")).await;
        assert!(body.contains("choose the customer"));
        let (_, _, body) = send(
            &h,
            as_role(upload_req(b"new bytes", Some("K02")), "paperless.k01"),
        )
        .await;
        assert!(body.contains("cannot file documents under customer K02"));
        assert_eq!(
            h.state.store.count().await.expect("count"),
            before,
            "nothing archived"
        );
    }

    /// Sends `req` while this test holds the document's hash lock, so the
    /// request has to wait for it; moves the document from K01 to K02 while it
    /// waits, as a concurrent refile by an unrestricted caller would; then
    /// releases the lock. The move happens only once the request signals that it
    /// is about to take the lock (`lock_probe`), so a write that checked access
    /// before taking the lock always sees the old customer and fails the test.
    async fn send_while_moved(h: &Harness, hash: &str, req: Request<Body>) -> (StatusCode, String) {
        let key = ContentSha256::from_hex(hash).expect("hash");
        let held = h.state.hash_lock(&key).lock_owned().await;
        let reached = crate::state::lock_probe::expect(hash);
        let app = h.app.clone();
        let task = tokio::spawn(async move { app.oneshot(req).await.expect("oneshot") });
        // Move the document only once the request is about to take the lock,
        // never earlier: a request that has not reached the lock yet could
        // read the moved customer even with a check made before the lock.
        tokio::time::timeout(std::time::Duration::from_secs(10), reached)
            .await
            .expect("the request never reached the document's lock")
            .expect("probe dropped");
        assert!(
            !task.is_finished(),
            "the write did not wait for the document's lock"
        );
        h.state
            .store
            .meta()
            .set_customer(hash, Some("K02"), 2)
            .await
            .expect("move");
        drop(held);
        let resp = task.await.expect("task");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    /// A K01 caller's write that waits for the document's lock while the
    /// document moves to K02 is refused with 404 and changes nothing: access is
    /// checked under the lock, not before it.
    /// Disable: check access before taking the lock in `lock_visible`.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_waiting_for_the_lock_honours_a_concurrent_move() {
        let h = harness_with(Some(POLICY)).await;
        let meta = h.state.store.meta();
        let tag = meta
            .create_definition(MetaKind::Tag, "Strom", 1)
            .await
            .expect("tag")
            .definition_id;
        let cases: [(&str, &str); 6] = [
            ("assign", "kind=tag&definition_id=TAG"),
            ("unassign", "kind=tag&definition_id=TAG"),
            ("accept", "kind=tag&definition_id=TAG"),
            ("review", "reviewed=true"),
            ("customer", "customer=K01"),
            ("delete", ""),
        ];
        for (i, (route, form)) in cases.iter().enumerate() {
            let hash = seed(&h, format!("moved document {i}").as_bytes()).await;
            meta.set_customer(&hash, Some("K01"), 1).await.expect("k01");
            if *route == "unassign" {
                meta.assign(&hash, MetaKind::Tag, tag, Source::Manual, 1)
                    .await
                    .expect("assign");
            }
            let body = form.replace("TAG", &tag.to_string());
            let req = as_role(
                post_form(&format!("/documents/{hash}/{route}"), &body),
                "paperless.k01",
            );
            let (status, _) = send_while_moved(&h, &hash, req).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{route}: refused");

            assert!(
                h.state.store.get(&hash).await.expect("get").is_some(),
                "{route}: the document must still exist"
            );
            assert_eq!(
                meta.customer_of(&hash).await.expect("c").as_deref(),
                Some("K02"),
                "{route}: the move must stand"
            );
            assert!(
                !meta.is_reviewed(&hash).await.expect("reviewed"),
                "{route}: nothing marked reviewed"
            );
            let tags = meta
                .assignments_for(&hash)
                .await
                .expect("for")
                .into_iter()
                .filter(|a| a.kind == MetaKind::Tag)
                .count();
            let expected = usize::from(*route == "unassign");
            assert_eq!(tags, expected, "{route}: assignments unchanged");
        }
    }

    /// The same interleaving without the move: the write waits for the lock,
    /// then succeeds, so the guard does not refuse a caller who may write.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_waiting_for_the_lock_succeeds_when_nothing_moves() {
        let h = harness_with(Some(POLICY)).await;
        let hash = seed(&h, b"stays in k01").await;
        let meta = h.state.store.meta();
        meta.set_customer(&hash, Some("K01"), 1).await.expect("k01");
        let key = ContentSha256::from_hex(&hash).expect("hash");
        let held = h.state.hash_lock(&key).lock_owned().await;
        let reached = crate::state::lock_probe::expect(&hash);
        let app = h.app.clone();
        let req = as_role(
            post_form(&format!("/documents/{hash}/review"), "reviewed=true"),
            "paperless.k01",
        );
        let task = tokio::spawn(async move { app.oneshot(req).await.expect("oneshot") });
        tokio::time::timeout(std::time::Duration::from_secs(10), reached)
            .await
            .expect("the request never reached the document's lock")
            .expect("probe dropped");
        assert!(!task.is_finished(), "the write did not wait for the lock");
        drop(held);
        assert_eq!(task.await.expect("task").status(), StatusCode::SEE_OTHER);
        assert!(meta.is_reviewed(&hash).await.expect("reviewed"));
    }

    /// An upload of bytes the archive holds, for a document stored before
    /// originals were kept: allowed when the caller sees the document (the
    /// original is attached), refused without touching it when the document
    /// moves to another customer while the upload waits for its lock.
    /// Disable: drop the `may_see` check in ingest's duplicate path.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_duplicate_upload_is_checked_under_the_lock_before_the_original_is_attached() {
        let h = harness_with(Some(POLICY)).await;
        let meta = h.state.store.meta();

        let allowed = b"legacy k01 scan";
        let hash = seed(&h, allowed).await;
        meta.set_customer(&hash, Some("K01"), 1).await.expect("k01");
        let (status, location, _) = send(
            &h,
            as_role(upload_req(allowed, Some("K01")), "paperless.k01"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::SEE_OTHER,
            "a visible duplicate is accepted"
        );
        assert_eq!(
            location.as_deref(),
            Some(format!("/documents/{hash}").as_str())
        );
        assert_eq!(blob_files(&h), 1, "its original is attached");

        let refused = b"legacy scan that moves";
        let hash = seed(&h, refused).await;
        meta.set_customer(&hash, Some("K01"), 1).await.expect("k01");
        let (_, body) = send_while_moved(
            &h,
            &hash,
            as_role(upload_req(refused, Some("K01")), "paperless.k01"),
        )
        .await;
        assert!(
            body.contains("filed under a customer you cannot see"),
            "refused: {body}"
        );
        assert_eq!(
            blob_files(&h),
            1,
            "no original attached to the moved document"
        );
        assert_eq!(
            meta.customer_of(&hash).await.expect("c").as_deref(),
            Some("K02")
        );
    }

    /// The per-customer JSON lists exactly that customer's documents, to a
    /// caller whose own grants cover it, and only with a policy configured.
    /// Disable: drop the `scope.covers` check in `api::customer_documents`.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_customer_documents_api_is_scoped_by_the_callers_grants() {
        let h = harness_with(Some(POLICY)).await;
        let (k01, k02, none) = three_customers(&h).await;
        let (status, _, body) = send(
            &h,
            as_role(get_req("/api/v1/customers/K01/documents"), "paperless.k01"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["customer"], "K01");
        assert_eq!(v["total"], 1);
        assert_eq!(v["documents"][0]["hash"], k01.as_str());
        assert_eq!(v["documents"][0]["path"], format!("/documents/{k01}"));
        assert!(!body.contains(&k02) && !body.contains(&none));
        assert!(!body.contains("Stadtwerke"), "no document text in the API");

        let (status, _, _) = send(
            &h,
            as_role(get_req("/api/v1/customers/K02/documents"), "paperless.k01"),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "outside the caller's grants");
        let (status, _, _) = send(&h, get_req("/api/v1/customers/K01/documents")).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "no caller, no scope");
        let (status, _, body) = send(
            &h,
            as_role(
                get_req("/api/v1/customers/K02/documents"),
                "paperless.admin",
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(&k02));

        let open = harness().await;
        let (status, _, _) = send(
            &open,
            as_role(
                get_req("/api/v1/customers/K01/documents"),
                "paperless.admin",
            ),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "no policy, no per-customer API"
        );
    }

    /// A scoped caller's search is filtered before it is cut to LIST_LIMIT:
    /// more than LIST_LIMIT higher-ranking matches of another customer must not
    /// hide the caller's own match.
    /// Disable: search with `LIST_LIMIT` for every scope.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_scoped_search_is_not_cut_before_the_customer_filter() {
        let h = harness_with(Some(POLICY)).await;
        let meta = h.state.store.meta();
        for i in 0..=LIST_LIMIT {
            let hash = seed(&h, format!("other {i}").as_bytes()).await;
            meta.set_customer(&hash, Some("K02"), 1).await.expect("k02");
            h.state
                .search
                .index_document(&hash, "doc.png", "Stadtwerke Stadtwerke Stadtwerke")
                .expect("index");
        }
        let own = seed(&h, b"own document").await;
        meta.set_customer(&own, Some("K01"), 1).await.expect("k01");
        h.state
            .search
            .index_document(
                &own,
                "doc.png",
                "Stadtwerke with many more words that lower its rank considerably here",
            )
            .expect("index");
        let (_, _, body) = send(
            &h,
            as_role(get_req("/documents?q=Stadtwerke"), "paperless.k01"),
        )
        .await;
        assert!(body.contains(&own), "the caller's own match must be found");
    }

    /// A stale index entry (no archive row) that outranks the caller's match
    /// does not push it out: the scoped search limit is the index size, not
    /// the archive's row count.
    /// Disable: take the scoped limit from `state.store.count()`.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_stale_index_entry_does_not_hide_a_scoped_match() {
        let h = harness_with(Some(POLICY)).await;
        let stale = format!("{:?}", ContentSha256::of(b"deleted, still indexed"));
        h.state
            .search
            .index_document(&stale, "doc.png", "Stadtwerke Stadtwerke Stadtwerke")
            .expect("index stale");
        let own = seed(&h, b"own document").await;
        h.state
            .store
            .meta()
            .set_customer(&own, Some("K01"), 1)
            .await
            .expect("k01");
        h.state
            .search
            .index_document(
                &own,
                "doc.png",
                "Stadtwerke with many more words that lower its rank considerably here",
            )
            .expect("index");
        assert_eq!(h.state.store.count().await.expect("count"), 1);
        let (_, _, body) = send(
            &h,
            as_role(get_req("/documents?q=Stadtwerke"), "paperless.k01"),
        )
        .await;
        assert!(body.contains(&own), "the caller's own match must be found");
    }

    /// The S3 backlog counts every customer's documents, so the dashboard
    /// shows it only to a caller who sees every customer.
    /// Disable: render `replica_summary` for every caller.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_dashboard_shows_the_s3_backlog_only_to_unrestricted_callers() {
        let archive = tempfile::tempdir().expect("archive tempdir");
        let index = tempfile::tempdir().expect("index tempdir");
        let blobs = tempfile::tempdir().expect("blob tempdir");
        let bucket = Arc::new(crate::s3::memory::MemoryStore::default());
        let mut state = AppState::load(
            FsPath::new(MODEL_DIR),
            &archive.path().to_string_lossy(),
            index.path(),
            blobs.path(),
            FsPath::new("/nonexistent/deepnsm/word_frequency"),
            Some(crate::replica::Replica::new(bucket, "pl")),
        )
        .await
        .expect("AppState::load");
        state.policy = Some(crate::customers::CustomerPolicy::from_json(POLICY).expect("policy"));
        let state = Arc::new(state);
        let h = Harness {
            app: router(state.clone()),
            state,
            _dirs: (archive, index, blobs),
        };
        let (_, _, admin) = send(&h, as_role(get_req("/dashboard"), "paperless.admin")).await;
        assert!(
            admin.contains("S3 copy"),
            "an unrestricted caller sees the backlog"
        );
        let (_, _, scoped) = send(&h, as_role(get_req("/dashboard"), "paperless.k01")).await;
        assert!(!scoped.contains("S3 copy"), "a scoped caller does not");
    }
}
