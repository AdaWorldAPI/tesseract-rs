//! The "web link" arm: fetch an image from a user-supplied URL — SSRF-guarded.
//!
//! This is a public endpoint, so a naive `GET` is an SSRF hole: a user could
//! point it at `http://169.254.169.254/` (cloud metadata) or an internal
//! service. The guard: (1) http/https only; (2) every resolved IP must be
//! public — loopback / private / link-local / ULA / unspecified are rejected;
//! (3) redirects are disabled (a redirect could bounce past the guard to an
//! internal host); (4) a 10 MB + 10 s cap bounds DoS.

use std::net::IpAddr;
use std::time::Duration;

use futures_util::StreamExt;

/// Max bytes we will download from a remote URL.
const MAX_BYTES: usize = 10 * 1024 * 1024;
/// Per-request timeout.
const TIMEOUT: Duration = Duration::from_secs(10);

/// True if `ip` must NOT be fetched from — the SSRF blocklist. Kept as a pure
/// function so it is unit-testable on literal IPs without any network.
pub(crate) fn ip_is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()      // 127.0.0.0/8
                || v4.is_private() // 10/8, 172.16/12, 192.168/16
                || v4.is_link_local() // 169.254.0.0/16 (incl. 169.254.169.254 metadata)
                || v4.is_unspecified() // 0.0.0.0
                || v4.is_broadcast()
        }
        IpAddr::V6(v6) => {
            // v4-mapped (::ffff:a.b.c.d) — unwrap and apply the v4 rules.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return ip_is_blocked(IpAddr::V4(v4));
            }
            let seg0 = v6.segments()[0];
            v6.is_loopback()               // ::1
                || v6.is_unspecified()     // ::
                || (seg0 & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                || (seg0 & 0xffc0) == 0xfe80 // fe80::/10 link-local
        }
    }
}

/// Validate the URL scheme and return `(host, port)` for resolution, or a
/// user-safe error. Only `http`/`https` are allowed.
fn parse_target(url: &str) -> Result<(String, u16), String> {
    let parsed = reqwest::Url::parse(url).map_err(|_| "invalid URL".to_string())?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "unsupported URL scheme '{other}' (http/https only)"
            ))
        }
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| "URL has no host".to_string())?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| "URL has no port".to_string())?;
    Ok((host, port))
}

/// Fetch an image from `url`, SSRF-guarded and size/time-capped. Returns the
/// raw bytes for the decoder, or a user-safe error message.
pub async fn fetch_image_url(url: &str) -> Result<Vec<u8>, String> {
    let (host, port) = parse_target(url)?;

    // Resolve and reject if ANY resolved address is non-public. `lookup_host`
    // needs "host:port".
    let mut any = false;
    for addr in tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|e| format!("could not resolve host: {e}"))?
    {
        any = true;
        if ip_is_blocked(addr.ip()) {
            return Err("refusing to fetch a private / loopback / link-local address".to_string());
        }
    }
    if !any {
        return Err("host did not resolve to any address".to_string());
    }

    // Redirects OFF: a 3xx could bounce past the guard to an internal host.
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("remote returned HTTP {}", resp.status().as_u16()));
    }
    // Early reject if the server honestly declares an oversized body.
    if let Some(len) = resp.content_length() {
        if len as usize > MAX_BYTES {
            return Err(format!("image too large ({len} bytes; max {MAX_BYTES})"));
        }
    }

    // Stream with a hard cap so a lying/omitted content-length can't OOM us.
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("download error: {e}"))?;
        if body.len() + chunk.len() > MAX_BYTES {
            return Err(format!("image exceeds the {MAX_BYTES}-byte cap"));
        }
        body.extend_from_slice(&chunk);
    }
    if body.is_empty() {
        return Err("remote returned an empty body".to_string());
    }
    Ok(body)
}
