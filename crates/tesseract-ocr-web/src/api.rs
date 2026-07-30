//! Machine-facing JSON/binary API surface — the Power Platform custom
//! connector target.
//!
//! Four real routes plus a spec endpoint:
//! - `POST /api/v1/recognize`      — binary or `{content_base64, lang, rectify}`
//!   JSON in, `tesseract-rs/doc.v1` JSON out (`RecognizeDocument` in the
//!   connector). The `doc.v1` payload carries two Power-Automate-ergonomic
//!   additive fields on top of the canonical schema —
//!   [`tesseract_ocr::structured`]'s module docs cover the shape;
//!   `page.plain_text` (the whole page as one string) and
//!   `page.fields_map` (`fields`, reshaped `key -> value`) — see that
//!   module's `render_doc` for exactly what each contains.
//! - `POST /api/v1/pdf`            — same input, searchable PDF out (default;
//!   `?mode=structured` switches to the structured reconstruction)
//!   (`SearchablePdf` in the connector).
//! - `POST /api/v1/pdf/structured` — same input, ALWAYS the structured
//!   reconstruction — a query-param-free alias so Power Automate's action
//!   picker offers it as its own action (`StructuredPdf` in the connector;
//!   OpenAPI 2.0 cannot express two `operationId`s on one path+method, so
//!   this is a real second route, not just documentation).
//! - `GET /api/v1/health`          — liveness + loaded-model probe, no auth,
//!   no request body, no recognition work (`HealthCheck` in the connector —
//!   the "Test operation" target).
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
use crate::routes::{build_pdf, pdf_response, LangQuery, PdfQuery};
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
/// `/openapi.json` and `/api/v1/health` are deliberately OUTSIDE the API-key
/// gate: the former is the connector's own discovery document (what Power
/// Platform's importer or a `paconn` invocation fetches), the latter is the
/// liveness/capability probe Power Platform's "Test operation" step (and any
/// external monitor) needs to work with no connection configured yet — gating
/// either one defeats its purpose.
pub fn router() -> Router<Arc<AppState>> {
    let protected = Router::new()
        .route("/api/v1/recognize", post(recognize))
        .route("/api/v1/pdf", post(pdf_searchable_or_query))
        .route("/api/v1/pdf/structured", post(pdf_structured))
        .layer(middleware::from_fn(require_api_key));

    Router::new()
        .merge(protected)
        .route("/api/v1/health", get(health))
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
// Request-body dispatch: raw binary OR `{content_base64, lang, rectify}` JSON
// ===========================================================================

/// The JSON alternate form's body shape (see `docs/SDK-PYTHON-AND-POWER-PLATFORM.md`
/// §2). `lang` selects the model via [`crate::state::AppState::model`]
/// (`"deu"` → German, anything else → English); `rectify` runs
/// [`tesseract_ocr::rectify::auto_rectify`] before recognition when `true`
/// (default `false` when absent — combined with the `?rectify=` query
/// parameter via OR, not an override like `lang`, since a plain flag only
/// needs either source to enable it) — see [`decode_request_body`].
#[derive(Debug, Deserialize)]
struct RecognizeJsonBody {
    content_base64: String,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    rectify: bool,
}

/// `true` when the request declares a JSON content-type (ignoring any
/// `; charset=...` parameter) — the signal that selects the
/// `{content_base64, lang, rectify}` branch over the raw-binary branch.
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

/// Decode `body` into raw image bytes + the requested language (if any) + the
/// requested rectify flag for the recognizer, dispatched by content-type:
/// - `application/json` — parsed as [`RecognizeJsonBody`]; `content_base64`
///   is base64-decoded (standard or URL-safe alphabet, padded or not), and
///   `lang`/`rectify` (if present) are returned alongside.
/// - anything else (notably `application/octet-stream`, the shape Microsoft
///   Graph's "Get file content" produces) — the body IS the image bytes,
///   verbatim, with no `lang`/`rectify` of its own (a binary POST has no body
///   field for either — callers pass `?lang=`/`?rectify=` instead; see each
///   handler).
///
/// `lang` selects the model via [`crate::state::AppState::model`] (`"deu"` →
/// German, anything else → English, the pre-existing default) — never a hard
/// error for an unrecognized value, same "forgiving field" rule the rest of
/// this crate's request parsing uses. `rectify` runs
/// [`tesseract_ocr::rectify::auto_rectify`] before recognition — unlike
/// `lang`, there is no "which source wins" precedence to pick between the
/// body and the query parameter: each caller combines them with OR (`true`
/// from either source is enough), since a plain flag has no competing values
/// to choose between the way `"eng"` vs `"deu"` does.
fn decode_request_body(
    headers: &HeaderMap,
    body: &[u8],
) -> Result<(Vec<u8>, Option<String>, bool), String> {
    if is_json_content_type(headers) {
        let parsed: RecognizeJsonBody = serde_json::from_slice(body).map_err(|e| {
            format!(
                "invalid JSON body (expected {{\"content_base64\": \"...\", \"lang\": \"eng\", \
                 \"rectify\": false}}): {e}"
            )
        })?;
        let bytes = decode_base64(parsed.content_base64.trim())?;
        Ok((bytes, parsed.lang, parsed.rectify))
    } else if body.is_empty() {
        Err(
            "empty request body — send raw image bytes (application/octet-stream) or \
             {\"content_base64\": \"...\"}"
                .to_string(),
        )
    } else {
        Ok((body.to_vec(), None, false))
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

/// `POST /api/v1/recognize` — binary or `{content_base64, lang, rectify}`
/// JSON in, `tesseract-rs/doc.v1` JSON out. Same permit + `spawn_blocking`
/// discipline as the HTML `/ocr` route (recognition is heavy synchronous CPU
/// work; the permit bounds concurrent recognitions, `spawn_blocking` keeps
/// the async executor free while it runs).
///
/// `lang` comes from the JSON body's `lang` field when present; a binary POST
/// (no body field for it) instead reads `?lang=` (`Query<LangQuery>`) — the
/// JSON body wins if a caller somehow sends both. `rectify` is the OR of the
/// JSON body's own `rectify` field and `?rectify=true` (either source enabling
/// it is enough — see [`decode_request_body`]'s doc comment for why this
/// differs from `lang`'s override precedence).
async fn recognize(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LangQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let (bytes, body_lang, body_rectify) = match decode_request_body(&headers, &body) {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e),
    };
    // Computed before `q.lang` is moved out below — `q.wants_rectify()`
    // borrows `q`, which a partial move would otherwise make unusable.
    let rectify = body_rectify || q.wants_rectify();
    let lang = body_lang.or(q.lang);

    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return api_error(StatusCode::SERVICE_UNAVAILABLE, "server is shutting down"),
    };
    let st = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        ocr_image_bytes_json(&st, &bytes, lang.as_deref(), rectify)
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
    // Computed before `q.lang` is moved out below — both borrow `q`, and a
    // partial move would otherwise make it unusable for the second call.
    let structured = q.is_structured();
    let rectify = q.wants_rectify();
    pdf_impl(state, &headers, &body, structured, q.lang, rectify).await
}

/// `POST /api/v1/pdf/structured` — `StructuredPdf` in the connector: a
/// dedicated, query-param-free-for-`mode` alias for the structured
/// reconstruction, so Power Automate's action picker offers "Structured PDF"
/// as its own action. `?lang=` is still accepted (there is no `mode` to
/// select here, but the language switch is orthogonal).
async fn pdf_structured(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LangQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Computed before `q.lang` is moved out below — see the identical note in
    // `pdf_searchable_or_query`.
    let rectify = q.wants_rectify();
    pdf_impl(state, &headers, &body, true, q.lang, rectify).await
}

/// Shared body for [`pdf_searchable_or_query`] / [`pdf_structured`] — decode,
/// acquire a recognition permit, render off the async runtime, respond.
/// `query_lang` is the `?lang=` value (if any); the JSON body's own `lang`
/// field (when the request is `application/json`) wins over it — same
/// precedence as [`recognize`]. `query_rectify` is `?rectify=true`'s value;
/// it combines with the JSON body's own `rectify` field via OR rather than an
/// override — again, same rule as [`recognize`] (see
/// [`decode_request_body`]'s doc comment for why `rectify` doesn't need
/// `lang`'s "which source wins" precedence).
async fn pdf_impl(
    state: Arc<AppState>,
    headers: &HeaderMap,
    body: &[u8],
    structured: bool,
    query_lang: Option<String>,
    query_rectify: bool,
) -> Response {
    let (bytes, body_lang, body_rectify) = match decode_request_body(headers, body) {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e),
    };
    let lang = body_lang.or(query_lang);
    let rectify = body_rectify || query_rectify;

    let permit = match state.recognize_permits.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return api_error(StatusCode::SERVICE_UNAVAILABLE, "server is shutting down"),
    };
    let st = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        build_pdf(&st, &bytes, structured, lang.as_deref(), rectify)
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

/// The `GET /api/v1/health` response — which recognition models this
/// instance actually loaded at startup (`eng` is always present; `deu` only
/// when `MODEL_DIR` had its `deu.lstm*` components — see
/// [`AppState`](crate::state::AppState)'s own doc comment). Reading this,
/// not just an HTTP `200`, is what lets a caller ask "can I request
/// `lang=deu` here" before finding out the hard way via a silent English
/// fallback.
#[derive(Serialize)]
struct HealthBody {
    status: &'static str,
    models: Vec<&'static str>,
}

/// `GET /api/v1/health` — `HealthCheck` in the connector: a liveness +
/// capability probe with NO recognition work and NO request body, so it is
/// cheap enough to be Power Platform's "Test operation" target (the button a
/// connection author clicks before ever uploading a real document) and safe
/// to poll from an external monitor. Never gated — see [`router`]'s doc
/// comment.
async fn health(State(state): State<Arc<AppState>>) -> Response {
    let mut models = vec!["eng"];
    if state.deu.is_some() {
        models.push("deu");
    }
    Json(HealthBody {
        status: "ok",
        models,
    })
    .into_response()
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
    fn openapi_json_is_valid_and_declares_the_four_operations() {
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
            parsed["paths"]["/api/v1/health"]["get"]["operationId"],
            "HealthCheck"
        );
        // The health check must not force an api_key prompt on a caller who
        // is just testing the connection — the server itself never gates
        // this route (see `router`'s doc comment), so the spec's own
        // security override must agree: an explicit empty array, not merely
        // absent (which would inherit the global `security` requirement).
        assert_eq!(
            parsed["paths"]["/api/v1/health"]["get"]["security"],
            serde_json::json!([]),
            "HealthCheck must override the global security requirement to an empty array"
        );
        assert_eq!(
            parsed["securityDefinitions"]["api_key"]["name"],
            "x-api-key"
        );
        assert_eq!(
            parsed["definitions"]["RecognizeJsonBody"]["properties"]["rectify"]["type"],
            "boolean"
        );

        // Power Automate ergonomics: every POST operation carries an
        // x-ms-summary (a friendly action-picker label), and the two new
        // doc.v1 response fields are documented in the DocPage schema.
        for (path, method, opid) in [
            ("/api/v1/recognize", "post", "RecognizeDocument"),
            ("/api/v1/pdf", "post", "SearchablePdf"),
            ("/api/v1/pdf/structured", "post", "StructuredPdf"),
            ("/api/v1/health", "get", "HealthCheck"),
        ] {
            let op = &parsed["paths"][path][method];
            assert_eq!(op["operationId"], opid);
            assert!(
                op["x-ms-summary"].is_string(),
                "{opid} must carry an x-ms-summary for the Power Automate action picker"
            );
        }
        assert_eq!(
            parsed["definitions"]["DocPage"]["properties"]["plain_text"]["type"],
            "string"
        );
        assert_eq!(
            parsed["definitions"]["DocPage"]["properties"]["fields_map"]["type"],
            "object"
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

    #[tokio::test]
    async fn get_health_returns_200_without_auth_and_lists_eng() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request as HttpRequest;
        use tower::ServiceExt;

        let dir = model_dir();
        if !dir.join("eng.lstm").exists() {
            eprintln!("skipping: corpus model absent");
            return;
        }
        // SAFETY (test-only): confirms the health route ignores
        // TESSERACT_API_KEY even when a real deployment has it configured —
        // exercised WITHOUT an x-api-key header below.
        std::env::set_var("TESSERACT_API_KEY", "some-secret-configured-elsewhere");
        let state = Arc::new(AppState::load(&dir).expect("load model"));
        let app = crate::routes::router(state);
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        std::env::remove_var("TESSERACT_API_KEY");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "health must answer even when the API key gate is configured"
        );
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["status"], "ok");
        let models: Vec<&str> = v["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap())
            .collect();
        assert!(models.contains(&"eng"), "eng is always loaded: {models:?}");
    }

    #[tokio::test]
    async fn post_recognize_response_carries_plain_text_and_fields_map() {
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
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        let page = &v["pages"][0];

        // Anti-vacuity: plain_text must be REAL recognized text, not merely
        // present — page_01.pgm's known first line, and it must join every
        // line (not just the first) via '\n'.
        let plain_text = page["plain_text"].as_str().expect("plain_text is a string");
        assert!(
            plain_text.contains("The old clock ticked all night."),
            "plain_text must contain real recognized text: {plain_text:?}"
        );
        assert!(
            plain_text.contains('\n'),
            "plain_text must join multiple lines: {plain_text:?}"
        );
        let line_count_in_regions: usize = page["regions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["lines"].as_array().map_or(0, Vec::len))
            .sum();
        assert_eq!(
            plain_text.lines().count(),
            line_count_in_regions,
            "plain_text must cover every recognized line, not a subset"
        );

        // fields_map must be present as an object (page_01.pgm has no
        // harvested fields — no harvest_profile was requested — so it is
        // legitimately empty; the KEY existing and being object-typed is
        // what a Power Automate expression like fields_map['iban'] depends
        // on, regardless of whether any entries are populated).
        assert!(
            page["fields_map"].is_object(),
            "fields_map must always be an object, even when empty: {:?}",
            page["fields_map"]
        );
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

    /// `?rectify=true` on a binary body wires through to
    /// [`crate::ocr::ocr_image_bytes_json`] without erroring —
    /// `page_01.pgm` is a clean digital render, so
    /// `tesseract_ocr::rectify::auto_rectify`'s no-op guarantee means this
    /// must still return a normal `doc.v1` document, same as the plain
    /// [`post_recognize_binary_returns_doc_v1_json`] case.
    #[tokio::test]
    async fn post_recognize_with_rectify_query_is_a_no_op_on_a_clean_page() {
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
                    .uri("/api/v1/recognize?rectify=true")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(page_01_bytes()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(text.starts_with("{\"schema\":\"tesseract-rs/doc.v1\""));
    }

    /// Same as the query-parameter case above, but `rectify` arrives as the
    /// JSON body's own field instead — exercises [`RecognizeJsonBody::rectify`]
    /// end-to-end through [`decode_request_body`].
    #[tokio::test]
    async fn post_recognize_json_body_rectify_field_is_accepted() {
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
        let body = serde_json::json!({ "content_base64": b64, "lang": "eng", "rectify": true })
            .to_string();
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
}
