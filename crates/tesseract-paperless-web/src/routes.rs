//! The paperless-ngx-shaped HTTP surface: upload, list, search, view, delete.

use std::sync::Arc;

use askama::Template;
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;

use ogar_doc_ir::{DocIr, RegionKind};
use tesseract_paperless::store::DocumentRow;

use crate::fetch::fetch_image_url;
use crate::ingest::{ingest, IngestOutcome};
use crate::state::AppState;

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
        } => {
            println!(
                "ingest: stored {hash_hex} (guid {}) -- {page_count} page(s), confidence {mean_confidence}{}",
                hex16(document_guid),
                if *low_confidence { ", LOW CONFIDENCE" } else { "" }
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
    page_count: u16,
    confidence: String,
    low_confidence: bool,
}

impl From<DocumentRow> for DocumentListItem {
    fn from(r: DocumentRow) -> Self {
        Self {
            hash_hex: r.content_sha256_hex,
            filename: r.filename.unwrap_or_else(|| "(untitled)".to_string()),
            preview: r.preview,
            page_count: r.page_count,
            confidence: confidence_str(r.mean_confidence, &r.text),
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
    let result = if query.trim().is_empty() {
        state.store.list(LIST_LIMIT).await
    } else {
        state.store.search(&query, LIST_LIMIT).await
    };
    match result {
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
    }
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
    error: Option<String>,
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

    let mut regions = Vec::new();
    flatten_regions(&ir, &mut regions);
    let fields = ir
        .fields
        .iter()
        .map(|f| (f.key.clone(), f.value.clone()))
        .collect();

    render(&DocumentDetailTemplate {
        hash_hex: row.content_sha256_hex,
        filename: row.filename.unwrap_or_else(|| "(untitled)".to_string()),
        mime: row.mime,
        source: row.source,
        page_count: row.page_count,
        confidence: confidence_str(row.mean_confidence, &row.text),
        low_confidence: row.low_confidence,
        ingested_at: format_unix_ms(row.ingested_at_unix_ms),
        text: row.text,
        regions,
        fields,
        error: None,
    })
}

async fn document_delete(State(state): State<Arc<AppState>>, Path(hash): Path<String>) -> Response {
    if let Err(e) = state.store.delete(&hash).await {
        eprintln!("delete {hash} failed: {e}");
    }
    Redirect::to("/documents").into_response()
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
}
