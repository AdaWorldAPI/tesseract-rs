//! Machine-facing JSON/binary API surface — the Power Platform custom
//! connector target.
//!
//! Three real routes plus a spec endpoint:
//! - `POST /api/v1/recognize`      — binary or `{content_base64, lang}` JSON in,
//!   `tesseract-rs/doc.v1` JSON out (`RecognizeDocument` in the connector).
//! - `POST /api/v1/pdf`            — same input, searchable PDF out (default;
//!   `?mode=structured` switches to the structured reconstruction)
//!   (`SearchablePdf` in the connector).
//! - `POST /api/v1/pdf/structured` — same input, ALWAYS the structured
//!   reconstruction — a query-param-free alias so Power Automate's action
//!   picker offers it as its own action (`StructuredPdf` in the connector;
//!   OpenAPI 2.0 cannot express two `operationId`s on one path+method, so
//!   this is a real second route, not just documentation).
//! - `GET /openapi.json`           — the Swagger 2.0 document Power Platform
//!   imports, served verbatim from `integrations/power-platform/apiDefinition.swagger.json`.
//!
//! This module adds NO new recognition logic: every handler is a thin
//! wrapper over [`crate::ocr::ocr_image_bytes_json`] / [`crate::routes::build_pdf`]
//! / [`crate::routes::pdf_response`] — the exact functions the HTML routes in
//! [`crate::routes`] already use — so the human form and the machine API can
//! never drift on WHAT gets recognized, only on how the request/response
//! bytes are shaped.
//!
//! See `docs/SDK-PYTHON-AND-POWER-PLATFORM.md` §2 for the design this
//! implements, and `integrations/power-platform/README.md` for the connector
//! import walkthrough + an MS-Graph flow example.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::ocr::ocr_image_bytes_json;
use crate::routes::{build_pdf, pdf_response, PdfQuery};
use crate::state::AppState;

/// The compiled Swagger 2.0 document served at `GET /openapi.json` — the
/// checked-in file is the single source of truth; this module only serves it
/// verbatim (byte-for-byte what a `paconn`/Maker-Portal import reads from
/// disk, or fetches live from the running server).
const OPENAPI_JSON: &str =
    include_str!("../../../integrations/power-platform/apiDefinition.swagger.json");

/// Build the `/api/v1/*` + `/openapi.json` routes. Merged into the main
/// router in [`crate::routes::router`], which also supplies the shared
/// upload-size layers (`DefaultBodyLimit` + `RequestBodyLimitLayer`) — those
/// apply to every route below too.
///
/// `/openapi.json` is deliberately OUTSIDE the API-key gate: it is the
/// connector's own discovery document (what Power Platform's importer or a
/// `paconn` invocation fetches), so it must be readable without a key.
pub fn router() -> Router<Arc<AppState>> {
    let protected = Router::new()
        .route("/api/v1/recognize", post(recognize))
        .route("/api/v1/pdf", post(pdf_searchable_or_query))
        .route("/api/v1/pdf/structured", post(pdf_structured))
        .layer(middleware::from_fn(require_api_key));

    Router::new()
        .merge(protected)
        .route("/openapi.json", get(openapi_json))
}

// ===========================================================================
// Auth — optional `x-api-key` gate, OFF by default (matches today's open demo)
// ===========================================================================

/// Pure authorization check, deliberately separated from env-var reading so
/// it is unit-testable without mutating global process state: `std::env`
/// mutation from inside a multithreaded test binary is a documented
/// flakiness hazard (other tests in this same binary run concurrently and
/// would observe the mutated var). Tests below exercise this function
/// directly instead of setting `TESSERACT_API_KEY` and hitting the router.
fn is_authorized(configured_key: Option<&str>, provided_key: Option<&str>) -> bool {
    match configured_key {
        None | Some("") => true, // unset/empty => auth disabled, open like today
        Some(expected) => provided_key == Some(expected),
    }
}

/// Gate `/api/v1/*` on `x-api-key` when `TESSERACT_API_KEY` is set in the
/// environment; a request with a missing/mismatched header gets `401` before
/// any recognition work starts. When the env var is unset (the default) the
/// gate is a no-op — this is the honest, documented default; see
/// `integrations/power-platform/README.md` §3.
async fn require_api_key(headers: HeaderMap, req: Request, next: Next) -> Response {
    let configured = std::env::var("TESSERACT_API_KEY").ok();
    let provided = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    if is_authorized(configured.as_deref(), provided) {
        next.run(req).await
    } else {
        api_error(
            StatusCode::UNAUTHORIZED,
            "missing or invalid x-api-key header",
        )
    }
}

// ===========================================================================
// Request-body dispatch: raw binary OR `{content_base64, lang}` JSON
// ===========================================================================

/// The JSON alternate form's body shape (see `docs/SDK-PYTHON-AND-POWER-PLATFORM.md`
/// §2). `lang` is accepted so a caller that sends it never gets a parse
/// error, but it is currently INFORMATIONAL ONLY — see [`decode_request_body`].
#[derive(Debug, Deserialize)]
struct RecognizeJsonBody {
    content_base64: String,
    #[serde(default)]
    lang: Option<String>,
}

/// `true` when the request declares a JSON content-type (ignoring any
/// `; charset=...` parameter) — the signal that selects the
/// `{content_base64, lang}` branch over the raw-binary branch.
fn is_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json")
        })
        .unwrap_or(false)
}

/// Decode `body` into raw image bytes for the recognizer, dispatched by
/// content-type:
/// - `application/json` — parsed as [`RecognizeJsonBody`]; `content_base64`
///   is base64-decoded (standard or URL-safe alphabet, padded or not).
/// - anything else (notably `application/octet-stream`, the shape Microsoft
///   Graph's "Get file content" produces) — the body IS the image bytes,
///   verbatim.
///
/// `lang`, when present in the JSON form, is logged, not enforced — this
/// server always recognizes with whichever single model [`AppState`] loaded
/// at startup (`MODEL_DIR`; see `crate::state::AppState::load`). Wiring a
/// per-request language switch would need a multi-model server; that is out
/// of scope for this connector pass (see `integrations/power-platform/README.md`).
fn decode_request_body(headers: &HeaderMap, body: &[u8]) -> Result<Vec<u8>, String> {
    if is_json_content_type(headers) {
        let parsed: RecognizeJsonBody = serde_json::from_slice(body).map_err(|e| {
            format!(
                "invalid JSON body (expected {{\"content_base64\": \"...\", \"lang\": \"eng\"}}): {e}"
            )
        })?;
        if let Some(lang) = parsed.lang.as_deref() {
            eprintln!(
                "api: request declared lang={lang:?} (informational only — this \
                 deployment always uses its single model loaded at startup via MODEL_DIR)"
            );
        }
        decode_base64(parsed.content_base64.trim())
    } else if body.is_empty() {
        Err(
            "empty request body — send raw image bytes (application/octet-stream) or \
             {\"content_base64\": \"...\"}"
                .to_string(),
        )
    } else {
        Ok(body.to_vec())
    }
}

/// Base64-decode `s`, trying the standard alphabet (padded, then unpadded)
/// and the URL-safe alphabet (padded, then unpadded) in turn — tolerant of
/// whichever variant a caller's JSON serializer happens to produce.
fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
    STANDARD
        .decode(s)
        .or_else(|_| STANDARD_NO_PAD.decode(s))
        .or_else(|_| URL_SAFE.decode(s))
        .or_else(|_| URL_SAFE_NO_PAD.decode(s))
        .map_err(|_| "content_base64 is not valid base64".to_string())
}

// ===========================================================================
// Errors — always `{"error": "..."}` JSON, never the HTML error page
// ===========================================================================

/// A `{"error": "..."}` JSON body — every non-2xx response from this module
/// uses this shape (unlike [`crate::routes`], which renders an HTML error
/// page — this is the machine-facing surface, so errors stay JSON).
#[derive(Serialize)]
struct ApiErrorBody {
    error: String,
}

fn api_error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(ApiErrorBody { error: msg.into() })).into_response()
}

/// The `doc.v1` JSON response — the rendered document string, verbatim, with
/// `Content-Type: application/json`. NOT wrapped in `Json(...)`: `doc_json`
/// is already a complete JSON document, and `Json(String)` would re-encode
/// it as an escaped JSON *string literal* instead of leaving it as the body.
fn doc_json_response(doc_json: String) -> Response {
    let mut resp = doc_json.into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp
}

// ===========================================================================
// Handlers
// ===========================================================================

/// `POST /api/v1/recognize` — binary or `{content_base64, lang}` JSON in,
/// `tesseract-rs/doc.v1` JSON out. Same permit + `spawn_blocking` discipline
/// as the HTML `/ocr` route (recognition is heavy synchronous CPU work; the
/// permit bounds concurrent recognitions, `spawn_blocking` keeps the async
/// executor free while it runs).
async fn recognize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let bytes = match decode_request_body(&headers, &body) {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e),
    };

    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return api_error(StatusCode::SERVICE_UNAVAILABLE, "server is shutting down"),
    };
    let st = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        ocr_image_bytes_json(&st, &bytes)
    })
    .await;
    match outcome {
        Ok(Ok(out)) => doc_json_response(out.json),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e),
        Err(e) => {
            eprintln!("api/recognize: recognition task failed: {e}");
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "recognition failed unexpectedly",
            )
        }
    }
}

/// `POST /api/v1/pdf` — `SearchablePdf` in the connector: defaults to the
/// searchable facsimile ("A"); `?mode=structured` switches to the same
/// structured reconstruction ("B") [`pdf_structured`] always returns. Kept
/// for exact parity with `docs/SDK-PYTHON-AND-POWER-PLATFORM.md` §2's
/// `POST /api/v1/pdf?mode=searchable|structured` shape.
async fn pdf_searchable_or_query(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PdfQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    pdf_impl(state, &headers, &body, q.is_structured()).await
}

/// `POST /api/v1/pdf/structured` — `StructuredPdf` in the connector: a
/// dedicated, query-param-free alias for the structured reconstruction, so
/// Power Automate's action picker offers "Structured PDF" as its own action.
async fn pdf_structured(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    pdf_impl(state, &headers, &body, true).await
}

/// Shared body for [`pdf_searchable_or_query`] / [`pdf_structured`] — decode,
/// acquire a recognition permit, render off the async runtime, respond.
async fn pdf_impl(
    state: Arc<AppState>,
    headers: &HeaderMap,
    body: &[u8],
    structured: bool,
) -> Response {
    let bytes = match decode_request_body(headers, body) {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e),
    };

    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return api_error(StatusCode::SERVICE_UNAVAILABLE, "server is shutting down"),
    };
    let st = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        build_pdf(&st, &bytes, structured)
    })
    .await;
    match outcome {
        Ok(Ok((pdf_bytes, filename))) => pdf_response(pdf_bytes, filename),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e),
        Err(e) => {
            eprintln!("api/pdf: recognition task failed: {e}");
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "recognition failed unexpectedly",
            )
        }
    }
}

/// `GET /openapi.json` — the Swagger 2.0 document, served verbatim from the
/// checked-in file. Never gated by [`require_api_key`] — see [`router`]'s doc
/// comment.
async fn openapi_json() -> Response {
    let mut resp = OPENAPI_JSON.into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model")
    }

    fn page_01_bytes() -> Vec<u8> {
        let page = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/pages/page_01.pgm");
        std::fs::read(&page).expect("read page_01.pgm")
    }

    #[test]
    fn openapi_json_is_valid_and_declares_the_three_operations() {
        let parsed: serde_json::Value = serde_json::from_str(OPENAPI_JSON)
            .expect("apiDefinition.swagger.json must be valid JSON");
        assert_eq!(parsed["swagger"], "2.0");
        assert_eq!(
            parsed["paths"]["/api/v1/recognize"]["post"]["operationId"],
            "RecognizeDocument"
        );
        assert_eq!(
            parsed["paths"]["/api/v1/pdf"]["post"]["operationId"],
            "SearchablePdf"
        );
        assert_eq!(
            parsed["paths"]["/api/v1/pdf/structured"]["post"]["operationId"],
            "StructuredPdf"
        );
        assert_eq!(
            parsed["securityDefinitions"]["api_key"]["name"],
            "x-api-key"
        );
    }

    #[test]
    fn is_json_content_type_matches_application_json_only() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        assert!(is_json_content_type(&headers));

        let mut headers_charset = HeaderMap::new();
        headers_charset.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        assert!(is_json_content_type(&headers_charset));

        let mut headers_binary = HeaderMap::new();
        headers_binary.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/octet-stream"),
        );
        assert!(!is_json_content_type(&headers_binary));

        assert!(!is_json_content_type(&HeaderMap::new()));
    }

    #[test]
    fn decode_base64_accepts_standard_and_url_safe_alphabets() {
        assert_eq!(decode_base64("Zm9v").unwrap(), b"foo");
        assert_eq!(decode_base64("Zm9vYmFy").unwrap(), b"foobar");

        // Exercise a byte sequence that actually differs between the
        // standard ('+' '/') and URL-safe ('-' '_') alphabets.
        let bytes: Vec<u8> = (0..64).collect();
        let url_safe = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes);
        assert_eq!(decode_base64(&url_safe).unwrap(), bytes);

        assert!(decode_base64("not base64!!!").is_err());
    }

    #[test]
    fn is_authorized_open_when_unconfigured() {
        assert!(is_authorized(None, None));
        assert!(is_authorized(None, Some("anything")));
        assert!(is_authorized(Some(""), Some("anything"))); // empty configured => open
        assert!(is_authorized(Some(""), None));
    }

    #[test]
    fn is_authorized_gates_when_configured() {
        assert!(is_authorized(Some("secret"), Some("secret")));
        assert!(!is_authorized(Some("secret"), Some("wrong")));
        assert!(!is_authorized(Some("secret"), None));
    }

    #[tokio::test]
    async fn get_openapi_json_returns_200_with_json_content_type() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .map(|v| v.to_str().unwrap()),
            Some("application/json")
        );
    }

    #[tokio::test]
    async fn post_recognize_binary_returns_doc_v1_json() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/api/v1/recognize")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(page_01_bytes()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .map(|v| v.to_str().unwrap()),
            Some("application/json")
        );
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(text.starts_with("{\"schema\":\"tesseract-rs/doc.v1\""));
    }

    #[tokio::test]
    async fn post_recognize_base64_json_returns_doc_v1_json() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let b64 = base64::engine::general_purpose::STANDARD.encode(page_01_bytes());
        let body = serde_json::json!({ "content_base64": b64, "lang": "eng" }).to_string();
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/api/v1/recognize")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(text.starts_with("{\"schema\":\"tesseract-rs/doc.v1\""));
    }

    #[tokio::test]
    async fn post_recognize_rejects_empty_binary_body() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/api/v1/recognize")
                    .header("content-type", "application/octet-stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let out = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&out).contains("\"error\""));
    }

    #[tokio::test]
    async fn post_pdf_default_returns_searchable_pdf() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/api/v1/pdf")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(page_01_bytes()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .map(|v| v.to_str().unwrap()),
            Some("application/pdf")
        );
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(out.starts_with(b"%PDF-"));
    }

    #[tokio::test]
    async fn post_pdf_structured_alias_returns_pdf() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/api/v1/pdf/structured")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(page_01_bytes()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .map(|v| v.to_str().unwrap()),
            Some("application/pdf")
        );
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(out.starts_with(b"%PDF-"));
    }
}
