//! The "web link" arm: fetch an image from a user-supplied URL — SSRF-guarded.
//!
//! This is a public endpoint, so a naive `GET` is an SSRF hole: a user could
//! point it at `http://169.254.169.254/` (cloud metadata) or an internal
//! service. The guard:
//! 1. **http/https only**;
//! 2. resolve the host **once**, reject if any address is non-global (loopback /
//!    private / CGNAT / link-local / ULA / multicast / reserved / test ranges),
//!    then **pin the request to those vetted addresses** so reqwest cannot
//!    re-resolve to a different (rebinding) IP at connect time;
//! 3. **no proxy** — an env proxy would resolve + connect on our behalf,
//!    defeating the IP vetting;
//! 4. **redirects disabled** (a 3xx could bounce past the guard);
//! 5. a **10 MB + 10 s** cap bounds DoS; client-facing errors are generic (the
//!    detail is logged) so the endpoint is not a host/port existence oracle.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use futures_util::StreamExt;

/// Max bytes we will download from a remote URL.
const MAX_BYTES: usize = 10 * 1024 * 1024;
/// Per-request timeout (also bounds the DNS resolution step).
const TIMEOUT: Duration = Duration::from_secs(10);

/// True if `ip` must NOT be fetched from — the SSRF blocklist. Kept as a pure
/// function so it is unit-testable on literal IPs without any network. Rejects
/// every non-globally-routable range, not just RFC1918, and unwraps IPv6 forms
/// that embed an IPv4 address (mapped / compatible / 6to4 / Teredo).
pub(crate) fn ip_is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_is_blocked(v4),
        IpAddr::V6(v6) => {
            // ::ffff:a.b.c.d — IPv4-mapped: apply the v4 rules to the embedded addr.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return v4_is_blocked(v4);
            }
            let s = v6.segments();
            // ::a.b.c.d — deprecated IPv4-compatible (high 96 bits zero).
            if s[..6].iter().all(|&seg| seg == 0) && !v6.is_unspecified() && !v6.is_loopback() {
                return v4_is_blocked(embedded_v4(s[6], s[7]));
            }
            // 2002:AABB:CCDD::/16 — 6to4: embedded v4 is segments 1..3.
            if s[0] == 0x2002 {
                return v4_is_blocked(embedded_v4(s[1], s[2]));
            }
            // 2001:0000::/32 — Teredo: client v4 is the bit-complement of the low 32 bits.
            if s[0] == 0x2001 && s[1] == 0x0000 {
                return v4_is_blocked(embedded_v4(!s[6], !s[7]));
            }
            v6.is_loopback()                 // ::1
                || v6.is_unspecified()       // ::
                || (s[0] & 0xfe00) == 0xfc00 // fc00::/7  unique-local
                || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || (s[0] & 0xff00) == 0xff00 // ff00::/8  multicast
        }
    }
}

/// Reassemble an IPv4 address from the two low IPv6 hextets.
fn embedded_v4(hi: u16, lo: u16) -> Ipv4Addr {
    Ipv4Addr::new(
        (hi >> 8) as u8,
        (hi & 0xff) as u8,
        (lo >> 8) as u8,
        (lo & 0xff) as u8,
    )
}

/// The IPv4 half of [`ip_is_blocked`] — hand-rolled ranges (the `Ipv4Addr`
/// `is_*` helpers for these are still unstable on the pinned toolchain).
fn v4_is_blocked(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_loopback()          // 127.0.0.0/8
        || v4.is_private()    // 10/8, 172.16/12, 192.168/16
        || v4.is_link_local() // 169.254.0.0/16 (incl. 169.254.169.254 metadata)
        || v4.is_unspecified()// 0.0.0.0
        || v4.is_broadcast()  // 255.255.255.255
        || v4.is_multicast()  // 224.0.0.0/4
        || o[0] == 0                                  // 0.0.0.0/8  "this network"
        || (o[0] == 100 && (o[1] & 0xc0) == 64)       // 100.64.0.0/10 CGNAT (Alibaba md 100.100.100.200)
        || o[0] >= 240                                // 240.0.0.0/4 reserved (+ 255.x)
        || (o[0] == 192 && o[1] == 0 && o[2] == 2)    // 192.0.2.0/24   TEST-NET-1
        || (o[0] == 198 && o[1] == 51 && o[2] == 100) // 198.51.100.0/24 TEST-NET-2
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)  // 203.0.113.0/24  TEST-NET-3
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18.0.0/15   benchmarking
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
/// raw bytes for the decoder, or a user-safe error message. Transport-level
/// detail is logged to stderr, never returned to the caller (avoids a
/// host/port existence oracle).
pub async fn fetch_image_url(url: &str) -> Result<Vec<u8>, String> {
    let (host, port) = parse_target(url)?;

    // Resolve ONCE (with a timeout — this call is not covered by the client
    // timeout) and vet every address.
    let resolved: Vec<SocketAddr> =
        match tokio::time::timeout(TIMEOUT, tokio::net::lookup_host((host.as_str(), port))).await {
            Ok(Ok(addrs)) => addrs.collect(),
            Ok(Err(e)) => {
                eprintln!("fetch: DNS resolution failed: {e}");
                return Err("could not resolve the URL's host".to_string());
            }
            Err(_) => return Err("DNS resolution timed out".to_string()),
        };
    if resolved.is_empty() {
        return Err("the URL's host did not resolve to any address".to_string());
    }
    for addr in &resolved {
        if ip_is_blocked(addr.ip()) {
            return Err("refusing to fetch a private / loopback / link-local address".to_string());
        }
    }

    // Pin reqwest to the vetted addresses so its own connect-time resolution
    // cannot rebind to a different IP; disable env proxies; redirects off.
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .resolve_to_addrs(&host, &resolved)
        .build()
        .map_err(|e| {
            eprintln!("fetch: client build failed: {e}");
            "internal error preparing the request".to_string()
        })?;

    let resp = client.get(url).send().await.map_err(|e| {
        eprintln!("fetch: request failed: {e}");
        "could not fetch the image URL".to_string()
    })?;
    if !resp.status().is_success() {
        return Err(format!("remote returned HTTP {}", resp.status().as_u16()));
    }
    // Early reject if the server honestly declares an oversized body.
    if let Some(len) = resp.content_length() {
        if len > MAX_BYTES as u64 {
            return Err(format!("image too large ({len} bytes; max {MAX_BYTES})"));
        }
    }

    // Stream with a hard cap so a lying/omitted content-length can't OOM us.
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            eprintln!("fetch: download error: {e}");
            "error downloading the image".to_string()
        })?;
        if body.len() + chunk.len() > MAX_BYTES {
            return Err(format!("image exceeds the {MAX_BYTES}-byte cap"));
        }
        body.extend_from_slice(&chunk);
    }
    if body.is_empty() {
        return Err("the remote returned an empty body".to_string());
    }
    Ok(body)
}
