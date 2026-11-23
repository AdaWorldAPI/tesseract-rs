//! The machine surface: JSON for a service that calls the archive on a user's
//! behalf (report-rs, with that user's token as `Authorization: Bearer`).
//!
//! It answers only with a customer policy configured: without one every
//! admitted user sees every document, and a per-customer listing would
//! pretend to a separation that does not exist. The caller's scope comes from
//! its own grants, exactly as for the HTML pages; the calling service adds no
//! authority.
//!
//! Responses carry metadata and a link, never document text: the text stays
//! behind the archive's own pages.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use serde::Serialize;
use tesseract_paperless::archive_meta::valid_customer_code;

use crate::auth::Caller;
use crate::customers::Scope;
use crate::state::AppState;

/// Most documents one response lists.
pub const MAX_LIMIT: usize = 500;
const DEFAULT_LIMIT: usize = 50;

#[derive(Serialize)]
struct DocumentJson {
    hash: String,
    filename: Option<String>,
    mime: String,
    page_count: u16,
    ingested_at_unix_ms: i64,
    reviewed: bool,
    /// Path of the document's page in the archive, relative to its origin.
    path: String,
}

#[derive(Serialize)]
struct CustomerDocumentsJson {
    customer: String,
    /// Documents filed under the customer, all of them, not only `documents`.
    total: usize,
    /// Newest first, at most `limit`.
    documents: Vec<DocumentJson>,
}

#[derive(Serialize)]
struct ErrorJson<'a> {
    error: &'a str,
}

fn error(status: StatusCode, msg: &str) -> Response {
    json(status, &ErrorJson { error: msg })
}

fn json<T: Serialize>(status: StatusCode, body: &T) -> Response {
    match serde_json::to_vec(body) {
        Ok(bytes) => (
            status,
            [
                (header::CONTENT_TYPE, "application/json"),
                (header::CACHE_CONTROL, "private, no-store"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "serialization failed").into_response(),
    }
}

#[derive(serde::Deserialize)]
pub(crate) struct ListQuery {
    limit: Option<usize>,
}

/// `GET /api/v1/customers/:code/documents?limit=N`.
pub(crate) async fn customer_documents(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Path(code): Path<String>,
    Query(q): Query<ListQuery>,
) -> Response {
    let Some(policy) = &state.policy else {
        return error(
            StatusCode::FORBIDDEN,
            "the archive has no customer policy, so it does not separate documents by customer",
        );
    };
    if !valid_customer_code(&code) {
        return error(StatusCode::BAD_REQUEST, "not a customer code");
    }
    let scope = match caller {
        Some(Extension(c)) => policy.scope_for(&c.grants),
        None => Scope::Customers(Default::default()),
    };
    if !scope.covers(Some(&code)) {
        return error(StatusCode::FORBIDDEN, "this customer is outside your access");
    }
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

    let read = async {
        let meta = state.store.meta();
        let customers = meta.customers().await?;
        let mut hashes: Vec<&String> = customers
            .iter()
            .filter(|(_, c)| **c == code)
            .map(|(h, _)| h)
            .collect();
        hashes.sort();
        let reviewed: std::collections::HashSet<String> =
            meta.reviewed_hashes().await?.into_iter().collect();
        let mut rows = Vec::with_capacity(hashes.len());
        for h in &hashes {
            // A customer row without its document is an orphan the next start
            // sweeps; it is not listed.
            if let Some(row) = state.store.get(h).await? {
                rows.push(row);
            }
        }
        rows.sort_by(|a, b| b.ingested_at_unix_ms.cmp(&a.ingested_at_unix_ms));
        let total = rows.len();
        let documents = rows
            .into_iter()
            .take(limit)
            .map(|r| DocumentJson {
                reviewed: reviewed.contains(&r.content_sha256_hex),
                path: format!("/documents/{}", r.content_sha256_hex),
                hash: r.content_sha256_hex,
                filename: r.filename,
                mime: r.mime,
                page_count: r.page_count,
                ingested_at_unix_ms: r.ingested_at_unix_ms,
            })
            .collect();
        Ok::<_, tesseract_paperless::store::StoreError>((total, documents))
    };
    match read.await {
        Ok((total, documents)) => json(
            StatusCode::OK,
            &CustomerDocumentsJson {
                customer: code,
                total,
                documents,
            },
        ),
        Err(e) => {
            eprintln!("api: archive read failed: {e}");
            error(StatusCode::INTERNAL_SERVER_ERROR, "archive read failed")
        }
    }
}
