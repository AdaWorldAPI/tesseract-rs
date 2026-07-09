//! HTTP surface: the upload/URL form and the OCR handler.

use std::sync::Arc;

use askama::Template;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::response::Html;
use axum::routing::{get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;

use crate::fetch::fetch_image_url;
use crate::ocr::{ocr_image_bytes, ocr_image_bytes_json, OcrJsonOutcome, OcrOutcome, OutputFormat};
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
    /// The text to show in the result `<pre>` block: recognized text, or the
    /// rendered JSON document. Askama HTML-escapes this on render.
    text: String,
    download_datauri: String,
    /// The download link's filename: `ocr.txt` or `result.json`.
    download_filename: &'static str,
}

/// Build the application router. Uploads are capped at 12 MB — this needs BOTH
/// limits: axum's per-extractor `DefaultBodyLimit` defaults to 2 MB (and would
/// reject larger multipart uploads before the handler runs), and tower-http's
/// `RequestBodyLimitLayer` caps the raw body; the smaller of the two wins, so
/// both are raised together. The URL-fetch arm has its own 10 MB cap in
/// [`fetch_image_url`].
pub fn router(state: Arc<AppState>) -> Router {
    const MAX_UPLOAD: usize = 12 * 1024 * 1024;
    Router::new()
        .route("/", get(index))
        .route("/ocr", post(ocr))
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
                ocr_image_bytes(&st, &bytes)
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
                ocr_image_bytes_json(&st, &bytes)
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
        let out = ocr_image_bytes(&state, &bytes).expect("ocr");
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
        let out = ocr_image_bytes_json(&state, &bytes).expect("ocr json");
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
}
