//! Response headers that keep the archive's pages from being framed, sniffed
//! or leaking their URLs, on every response including the sign-in pages.
//!
//! Framing: by default only the archive itself may frame its pages
//! (`frame-ancestors 'self'`, plus `X-Frame-Options: SAMEORIGIN` for browsers
//! that ignore CSP). `PAPERLESS_FRAME_ANCESTORS` lists further origins that may
//! embed it, comma separated, each exactly `https://host[:port]`: report-rs's
//! origin, so the archive can show inside its shell. With origins listed the
//! `X-Frame-Options` header is left out, since it cannot name another origin
//! and would override the list in browsers that honour both.
//!
//! The session cookie is `SameSite=Lax`, which a browser sends to a framed page
//! only when the framing page is on the same site (the same registrable
//! domain). So report-rs and the archive must share a parent domain for the
//! embedded view to be signed in.

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;

/// The headers, computed once at start.
#[derive(Debug, Clone)]
pub struct SecurityHeaders {
    csp: HeaderValue,
    frame_options: Option<HeaderValue>,
}

impl SecurityHeaders {
    /// Read `PAPERLESS_FRAME_ANCESTORS` through `get`.
    ///
    /// # Errors
    /// An entry that is not exactly an https origin. A bad entry refuses the
    /// start rather than being dropped: dropping it would silently break the
    /// embedded view, accepting a malformed one could widen it.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let mut origins: Vec<String> = Vec::new();
        if let Some(list) = get("PAPERLESS_FRAME_ANCESTORS") {
            for item in list.split(',').map(str::trim).filter(|i| !i.is_empty()) {
                let origin = https_origin(item).ok_or_else(|| {
                    format!("PAPERLESS_FRAME_ANCESTORS: {item:?} is not https://host[:port]")
                })?;
                if !origins.contains(&origin) {
                    origins.push(origin);
                }
            }
        }
        let mut csp = String::from("frame-ancestors 'self'");
        for o in &origins {
            csp.push(' ');
            csp.push_str(o);
        }
        Ok(Self {
            csp: HeaderValue::from_str(&csp).map_err(|_| "frame-ancestors value".to_string())?,
            frame_options: origins
                .is_empty()
                .then(|| HeaderValue::from_static("SAMEORIGIN")),
        })
    }

    /// The process environment version of [`Self::from_vars`].
    ///
    /// # Errors
    /// As [`Self::from_vars`].
    pub fn from_env() -> Result<Self, String> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// One log line.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Content-Security-Policy: {}",
            self.csp.to_str().unwrap_or("?")
        )
    }
}

/// `https://host[:port]` exactly: no path beyond `/`, no query, fragment or
/// credentials. Returned in the browser's own serialization.
fn https_origin(s: &str) -> Option<String> {
    let u = reqwest::Url::parse(s).ok()?;
    let ok = u.scheme() == "https"
        && u.host_str().is_some_and(|h| !h.is_empty())
        && u.username().is_empty()
        && u.password().is_none()
        && u.path() == "/"
        && u.query().is_none()
        && u.fragment().is_none()
        && !s.trim_end_matches('/').contains('*');
    ok.then(|| u.origin().ascii_serialization())
}

/// Middleware adding the headers to every response. A header a handler set
/// itself (the original download's `nosniff`) is kept.
pub async fn add(State(h): State<SecurityHeaders>, req: Request, next: Next) -> Response {
    let mut resp = next.run(req).await;
    let headers = resp.headers_mut();
    headers.insert(header::CONTENT_SECURITY_POLICY, h.csp.clone());
    if let Some(v) = &h.frame_options {
        headers.insert(header::X_FRAME_OPTIONS, v.clone());
    }
    headers
        .entry(header::X_CONTENT_TYPE_OPTIONS)
        .or_insert(HeaderValue::from_static("nosniff"));
    headers
        .entry(header::REFERRER_POLICY)
        .or_insert(HeaderValue::from_static("same-origin"));
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(v: Option<&str>) -> Result<SecurityHeaders, String> {
        let v = v.map(str::to_string);
        SecurityHeaders::from_vars(move |_| v.clone())
    }

    /// Disable: emit `frame-ancestors *`, or keep `X-Frame-Options` when
    /// origins are listed.
    #[test]
    fn framing_is_self_only_unless_origins_are_listed() {
        let none = with(None).expect("default");
        assert_eq!(none.csp, "frame-ancestors 'self'");
        assert_eq!(
            none.frame_options.as_ref().map(|v| v.as_bytes()),
            Some(&b"SAMEORIGIN"[..])
        );

        let listed = with(Some(
            "https://report.example.de, https://report.example.de/, https://x.example.de:8443",
        ))
        .expect("listed");
        assert_eq!(
            listed.csp,
            "frame-ancestors 'self' https://report.example.de https://x.example.de:8443"
        );
        assert!(listed.frame_options.is_none());
    }

    #[test]
    fn a_malformed_origin_refuses_the_start() {
        for bad in [
            "http://report.example.de",
            "https://report.example.de/view",
            "https://report.example.de?x=1",
            "https://*.example.de",
            "report.example.de",
            "https://u:p@report.example.de",
            "*",
        ] {
            assert!(with(Some(bad)).is_err(), "{bad} must be refused");
        }
    }
}
