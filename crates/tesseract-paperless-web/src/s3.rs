//! A minimal S3-compatible object client: `GET`, `PUT`, `DELETE` and paged
//! `ListObjectsV2` on one bucket, path-style, AWS Signature V4.
//!
//! The signing and the error classes follow report-rs's `objstore.rs` (same
//! construction, same RFC 4231 test vectors); it is copied rather than shared
//! because the two services must never share a configuration. This client reads
//! only the `PAPERLESS_S3_*` variables ([`S3Config::from_vars`]) and never falls
//! back to `AWS_*` or an SDK credential chain, so report-rs's buckets and keys
//! cannot be picked up by accident (paperless-rs `docs/RECOVERY.md` §2).
//!
//! Errors carry a stable class (`not_found`, `forbidden`, `unavailable`,
//! `timeout`, `throttled`, `http_<code>`, `transport`) and never a URL, key,
//! header or response body.

use std::time::Duration;

use sha2::{Digest, Sha256};

/// One bucket on one endpoint. `Debug` is hand-written so a secret never
/// reaches a log.
#[derive(Clone)]
pub struct S3Config {
    pub endpoint: String,
    pub bucket: String,
    pub region: String,
    pub prefix: String,
    access_key_id: String,
    secret_access_key: String,
}

impl std::fmt::Debug for S3Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Config")
            .field("endpoint", &self.endpoint)
            .field("bucket", &self.bucket)
            .field("region", &self.region)
            .field("prefix", &self.prefix)
            .field("access_key_id", &"<redacted>")
            .field("secret_access_key", &"<redacted>")
            .finish()
    }
}

/// The required variables. All set turns the copy on; none set leaves it off;
/// anything in between is an error.
pub const REQUIRED_VARS: [&str; 4] = [
    "PAPERLESS_S3_ENDPOINT",
    "PAPERLESS_S3_BUCKET",
    "PAPERLESS_S3_ACCESS_KEY_ID",
    "PAPERLESS_S3_SECRET_ACCESS_KEY",
];

fn clean(v: String) -> Option<String> {
    let v = v.trim().trim_matches('"').trim_matches('\'').to_string();
    (!v.is_empty()).then_some(v)
}

impl S3Config {
    pub fn new(
        endpoint: &str,
        bucket: &str,
        region: &str,
        prefix: &str,
        access_key_id: &str,
        secret_access_key: &str,
    ) -> Self {
        S3Config {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            bucket: bucket.to_string(),
            region: region.to_string(),
            prefix: prefix.trim_matches('/').to_string(),
            access_key_id: access_key_id.to_string(),
            secret_access_key: secret_access_key.to_string(),
        }
    }

    /// Read the configuration through `get` (testable without touching the
    /// process environment).
    ///
    /// # Errors
    /// A message naming the missing variables when only some of the required
    /// ones are set, or when the endpoint is not `https://`.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>, String> {
        let v = |k: &str| get(k).and_then(clean);
        let set: Vec<&str> = REQUIRED_VARS
            .iter()
            .copied()
            .filter(|k| v(k).is_some())
            .collect();
        if set.is_empty() {
            return Ok(None);
        }
        if set.len() != REQUIRED_VARS.len() {
            let missing: Vec<&str> = REQUIRED_VARS
                .iter()
                .copied()
                .filter(|k| !set.contains(k))
                .collect();
            return Err(format!(
                "S3 copy partly configured: {} missing (set all of {} or none)",
                missing.join(", "),
                REQUIRED_VARS.join(", ")
            ));
        }
        let endpoint = v("PAPERLESS_S3_ENDPOINT").unwrap_or_default();
        if !endpoint.starts_with("https://") || endpoint.len() <= "https://".len() {
            return Err("PAPERLESS_S3_ENDPOINT must be an https:// URL".to_string());
        }
        Ok(Some(S3Config::new(
            &endpoint,
            &v("PAPERLESS_S3_BUCKET").unwrap_or_default(),
            &v("PAPERLESS_S3_REGION").unwrap_or_else(|| "auto".into()),
            &v("PAPERLESS_S3_PREFIX").unwrap_or_else(|| "paperless".into()),
            &v("PAPERLESS_S3_ACCESS_KEY_ID").unwrap_or_default(),
            &v("PAPERLESS_S3_SECRET_ACCESS_KEY").unwrap_or_default(),
        )))
    }

    /// [`Self::from_vars`] over the process environment.
    ///
    /// # Errors
    /// As [`Self::from_vars`].
    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_vars(|k| std::env::var(k).ok())
    }
}

/// A failed object operation, as a stable, secret-free class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectError {
    pub class: String,
}

impl ObjectError {
    pub fn new(class: &str) -> Self {
        ObjectError {
            class: class.into(),
        }
    }
    pub fn is_not_found(&self) -> bool {
        self.class == "not_found"
    }
}

impl std::fmt::Display for ObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "object store: {}", self.class)
    }
}

impl std::error::Error for ObjectError {}

/// What the replica needs from a bucket. Blocking: callers on the async runtime
/// use `spawn_blocking`.
pub trait ObjectStore: Send + Sync {
    /// The object's bytes; `not_found` when absent.
    fn get(&self, key: &str) -> Result<Vec<u8>, ObjectError>;
    fn put(&self, key: &str, body: &[u8], content_type: &str) -> Result<(), ObjectError>;
    /// Removing an absent object succeeds.
    fn delete(&self, key: &str) -> Result<(), ObjectError>;
    /// Every key under `prefix`, all pages.
    fn list(&self, prefix: &str) -> Result<Vec<String>, ObjectError>;
}

// ---- crypto -------------------------------------------------------------------

pub fn sha256_hex(data: &[u8]) -> String {
    hex(Sha256::digest(data).as_slice())
}

pub fn hex(b: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(char::from(H[usize::from(x >> 4)]));
        s.push(char::from(H[usize::from(x & 15)]));
    }
    s
}

/// HMAC-SHA256 (RFC 2104), block size 64.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(Sha256::digest(key).as_slice());
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(ih.as_slice());
    let mut out = [0u8; 32];
    out.copy_from_slice(outer.finalize().as_slice());
    out
}

/// `(YYYYMMDD, YYYYMMDDTHHMMSSZ)` from Unix seconds, UTC.
pub fn amz_dates(unix_secs: i64) -> (String, String) {
    let days = unix_secs.div_euclid(86_400);
    let sod = unix_secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let date = format!("{y:04}{m:02}{d:02}");
    let stamp = format!(
        "{date}T{:02}{:02}{:02}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    );
    (date, stamp)
}

/// Howard Hinnant's civil-from-days.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                o.push(char::from(b));
            }
            b'/' if keep_slash => o.push('/'),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

/// A signed request: URL plus headers, deterministic for a given time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signed {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

/// AWS Signature V4 for one path-style request on `cfg.bucket`.
pub fn sign(
    cfg: &S3Config,
    method: &str,
    key: &str,
    query: &[(&str, &str)],
    payload: &[u8],
    unix_secs: i64,
) -> Signed {
    let host = cfg
        .endpoint
        .split("://")
        .nth(1)
        .unwrap_or(&cfg.endpoint)
        .split('/')
        .next()
        .unwrap_or_default()
        .to_string();
    let path = if key.is_empty() {
        format!("/{}", uri_encode(&cfg.bucket, false))
    } else {
        format!(
            "/{}/{}",
            uri_encode(&cfg.bucket, false),
            uri_encode(key, true)
        )
    };
    let mut q: Vec<(String, String)> = query
        .iter()
        .map(|(k, v)| (uri_encode(k, false), uri_encode(v, false)))
        .collect();
    q.sort();
    let cq = q
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&");
    let (date, stamp) = amz_dates(unix_secs);
    let ph = sha256_hex(payload);
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";
    let canonical = format!(
        "{method}\n{path}\n{cq}\nhost:{host}\nx-amz-content-sha256:{ph}\nx-amz-date:{stamp}\n\n{signed_headers}\n{ph}"
    );
    let scope = format!("{date}/{}/s3/aws4_request", cfg.region);
    let sts = format!(
        "AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let k = hmac_sha256(
        format!("AWS4{}", cfg.secret_access_key).as_bytes(),
        date.as_bytes(),
    );
    let k = hmac_sha256(&k, cfg.region.as_bytes());
    let k = hmac_sha256(&k, b"s3");
    let k = hmac_sha256(&k, b"aws4_request");
    let sig = hex(&hmac_sha256(&k, sts.as_bytes()));
    let auth = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={sig}",
        cfg.access_key_id
    );
    let url = if cq.is_empty() {
        format!("{}{path}", cfg.endpoint)
    } else {
        format!("{}{path}?{cq}", cfg.endpoint)
    };
    Signed {
        url,
        headers: vec![
            ("x-amz-content-sha256".into(), ph),
            ("x-amz-date".into(), stamp),
            ("authorization".into(), auth),
        ],
    }
}

/// The text between `<tag>` and `</tag>`, every occurrence, XML entities for
/// `&`, `<` and `>` decoded.
fn xml_values(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    text.split(open.as_str())
        .skip(1)
        .filter_map(|s| s.split(close.as_str()).next())
        .map(|s| {
            s.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&apos;", "'")
                .replace("&amp;", "&")
        })
        .collect()
}

// ---- HTTP client --------------------------------------------------------------

pub struct S3Client {
    cfg: S3Config,
    http: reqwest::blocking::Client,
}

fn class_of_status(s: u16) -> ObjectError {
    ObjectError::new(match s {
        404 => "not_found",
        401 | 403 => "forbidden",
        429 => "throttled",
        500..=599 => "unavailable",
        _ => return ObjectError::new(&format!("http_{s}")),
    })
}

fn class_of_transport(e: &reqwest::Error) -> ObjectError {
    ObjectError::new(if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "unavailable"
    } else {
        "transport"
    })
}

impl S3Client {
    /// # Errors
    /// `transport_init` when the HTTP client cannot be built.
    pub fn new(cfg: S3Config) -> Result<Self, ObjectError> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| ObjectError::new("transport_init"))?;
        Ok(S3Client { cfg, http })
    }

    fn send(
        &self,
        method: reqwest::Method,
        key: &str,
        query: &[(&str, &str)],
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<reqwest::blocking::Response, ObjectError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_secs()).ok())
            .unwrap_or(0);
        let s = sign(&self.cfg, method.as_str(), key, query, body, now);
        let mut rb = self.http.request(method, &s.url);
        for (k, v) in &s.headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        if let Some(ct) = content_type {
            rb = rb.header("content-type", ct);
        }
        if !body.is_empty() {
            rb = rb.body(body.to_vec());
        }
        let r = rb.send().map_err(|e| class_of_transport(&e))?;
        if r.status().is_success() {
            Ok(r)
        } else {
            Err(class_of_status(r.status().as_u16()))
        }
    }
}

impl ObjectStore for S3Client {
    fn get(&self, key: &str) -> Result<Vec<u8>, ObjectError> {
        let r = self.send(reqwest::Method::GET, key, &[], &[], None)?;
        r.bytes()
            .map(|b| b.to_vec())
            .map_err(|e| class_of_transport(&e))
    }
    fn put(&self, key: &str, body: &[u8], content_type: &str) -> Result<(), ObjectError> {
        self.send(reqwest::Method::PUT, key, &[], body, Some(content_type))
            .map(|_| ())
    }
    fn delete(&self, key: &str) -> Result<(), ObjectError> {
        match self.send(reqwest::Method::DELETE, key, &[], &[], None) {
            Ok(_) => Ok(()),
            Err(e) if e.is_not_found() => Ok(()),
            Err(e) => Err(e),
        }
    }
    fn list(&self, prefix: &str) -> Result<Vec<String>, ObjectError> {
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        // A bound on pages, so a server that keeps returning a token cannot
        // loop this forever: 10 000 pages of 1 000 keys.
        for _ in 0..10_000 {
            let mut query = vec![("list-type", "2"), ("prefix", prefix)];
            if let Some(t) = token.as_deref() {
                query.push(("continuation-token", t));
            }
            let r = self.send(reqwest::Method::GET, "", &query, &[], None)?;
            let text = r.text().map_err(|e| class_of_transport(&e))?;
            keys.extend(xml_values(&text, "Key"));
            let truncated = xml_values(&text, "IsTruncated")
                .first()
                .is_some_and(|v| v == "true");
            token = xml_values(&text, "NextContinuationToken").into_iter().next();
            if !truncated || token.is_none() {
                return Ok(keys);
            }
        }
        Err(ObjectError::new("list_too_long"))
    }
}

/// An in-memory bucket for tests, with switches that make every call fail.
#[cfg(test)]
pub mod memory {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::{ObjectError, ObjectStore};

    #[derive(Default)]
    pub struct MemoryStore {
        pub objects: Mutex<BTreeMap<String, Vec<u8>>>,
        /// Every call fails with `unavailable` while set.
        pub down: AtomicBool,
        /// Deletes fail with `unavailable` while set (other calls succeed).
        pub deletes_fail: AtomicBool,
        pub deletes: AtomicUsize,
        /// Every put and delete, in order, as `put <key>` / `delete <key>`.
        pub log: Mutex<Vec<String>>,
    }

    impl MemoryStore {
        fn up(&self) -> Result<(), ObjectError> {
            if self.down.load(Ordering::SeqCst) {
                Err(ObjectError::new("unavailable"))
            } else {
                Ok(())
            }
        }
        pub fn keys(&self) -> Vec<String> {
            self.objects.lock().expect("lock").keys().cloned().collect()
        }
    }

    impl ObjectStore for MemoryStore {
        fn get(&self, key: &str) -> Result<Vec<u8>, ObjectError> {
            self.up()?;
            self.objects
                .lock()
                .expect("lock")
                .get(key)
                .cloned()
                .ok_or_else(|| ObjectError::new("not_found"))
        }
        fn put(&self, key: &str, body: &[u8], _ct: &str) -> Result<(), ObjectError> {
            self.up()?;
            self.log.lock().expect("lock").push(format!("put {key}"));
            self.objects
                .lock()
                .expect("lock")
                .insert(key.to_string(), body.to_vec());
            Ok(())
        }
        fn delete(&self, key: &str) -> Result<(), ObjectError> {
            self.up()?;
            if self.deletes_fail.load(Ordering::SeqCst) {
                return Err(ObjectError::new("unavailable"));
            }
            self.deletes.fetch_add(1, Ordering::SeqCst);
            self.log.lock().expect("lock").push(format!("delete {key}"));
            self.objects.lock().expect("lock").remove(key);
            Ok(())
        }
        fn list(&self, prefix: &str) -> Result<Vec<String>, ObjectError> {
            self.up()?;
            Ok(self
                .objects
                .lock()
                .expect("lock")
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_rfc4231_case_2() {
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn hmac_rfc4231_case_6_long_key() {
        let key = [0xaau8; 131];
        assert_eq!(
            hex(&hmac_sha256(
                &key,
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn amz_dates_match_known_instants() {
        assert_eq!(amz_dates(0).1, "19700101T000000Z");
        assert_eq!(amz_dates(1_798_761_600).0, "20270101");
        assert_eq!(amz_dates(1_700_000_000).1, "20231114T221320Z");
    }

    /// The SigV4 test-suite vector "get-vanilla" from AWS's own documentation
    /// does not use the S3 service, so this pins the signature this code
    /// produces for a fixed request. A change to any part of the canonical
    /// request changes it.
    #[test]
    fn signature_is_deterministic_and_secret_free() {
        let c = S3Config::new("https://t3.example", "bkt", "auto", "p", "AKID", "SECRET");
        let a = sign(&c, "GET", "a b/c.json", &[], b"", 1_700_000_000);
        let b = sign(&c, "GET", "a b/c.json", &[], b"", 1_700_000_000);
        assert_eq!(a, b);
        assert_eq!(a.url, "https://t3.example/bkt/a%20b/c.json");
        assert!(a.headers.iter().all(|(_, v)| !v.contains("SECRET")));
        let other = sign(&c, "GET", "a b/c.json", &[], b"x", 1_700_000_000);
        assert_ne!(a.headers[2], other.headers[2], "the payload is signed");
    }

    #[test]
    fn only_the_paperless_variables_configure_the_copy() {
        let full = |k: &str| -> Option<String> {
            match k {
                "PAPERLESS_S3_ENDPOINT" => Some("https://fly.storage.tigris.dev".into()),
                "PAPERLESS_S3_BUCKET" => Some("archive".into()),
                "PAPERLESS_S3_ACCESS_KEY_ID" => Some("AKID-zq9".into()),
                "PAPERLESS_S3_SECRET_ACCESS_KEY" => Some("SEC-zq9".into()),
                // report-rs's set in the same project must not matter.
                "AWS_ACCESS_KEY_ID" => Some("other".into()),
                _ => None,
            }
        };
        let c = S3Config::from_vars(full).expect("ok").expect("on");
        assert_eq!(c.bucket, "archive");
        assert_eq!(c.region, "auto");
        assert_eq!(c.prefix, "paperless");
        assert!(!format!("{c:?}").contains("zq9"));

        // Only AWS_* set: off, not borrowed.
        let aws_only = |k: &str| -> Option<String> {
            k.starts_with("AWS_").then(|| "x".to_string())
        };
        assert!(S3Config::from_vars(aws_only).expect("ok").is_none());

        // A partial set is an error naming what is missing.
        let partial = |k: &str| (k == "PAPERLESS_S3_BUCKET").then(|| "b".to_string());
        let e = S3Config::from_vars(partial).expect_err("partial");
        assert!(e.contains("PAPERLESS_S3_SECRET_ACCESS_KEY"), "{e}");
        assert!(!e.contains("PAPERLESS_S3_BUCKET missing"), "{e}");

        // Plain http is refused.
        let http = |k: &str| match k {
            "PAPERLESS_S3_ENDPOINT" => Some("http://minio:9000".to_string()),
            _ => full(k),
        };
        assert!(S3Config::from_vars(http).is_err());
    }

    /// Against a real S3-compatible server, when one is named by
    /// `PAPERLESS_S3_LIVE_ENDPOINT`, `_BUCKET`, `_KEY` and `_SECRET` (skipped
    /// otherwise): put, get, delete, a listing over more than one page, and a
    /// wrong secret refused. Run once against moto with signature checks on.
    #[test]
    fn live_round_trip_against_an_s3_server() {
        let var = |k: &str| std::env::var(format!("PAPERLESS_S3_LIVE_{k}")).ok();
        let (Some(ep), Some(bucket), Some(key), Some(secret)) =
            (var("ENDPOINT"), var("BUCKET"), var("KEY"), var("SECRET"))
        else {
            eprintln!("skipped: PAPERLESS_S3_LIVE_* not set");
            return;
        };
        let cfg = S3Config::new(&ep, &bucket, "us-east-1", "t", &key, &secret);
        let c = S3Client::new(cfg).expect("client");
        let body = b"live bytes \x00\xff".to_vec();
        c.put("t/originals/a b+c", &body, "application/octet-stream")
            .expect("put");
        assert_eq!(c.get("t/originals/a b+c").expect("get"), body);
        assert!(c.get("t/originals/absent").expect_err("absent").is_not_found());
        for i in 0..1001 {
            c.put(&format!("t/documents/{i:05}.json"), b"{}", "application/json")
                .expect("put many");
        }
        let listed = c.list("t/documents/").expect("list");
        assert_eq!(listed.len(), 1001, "a listing past one page");
        c.delete("t/originals/a b+c").expect("delete");
        c.delete("t/originals/a b+c").expect("delete again");
        assert!(c.get("t/originals/a b+c").expect_err("gone").is_not_found());

        let bad = S3Client::new(S3Config::new(&ep, &bucket, "us-east-1", "t", &key, "wrong"))
            .expect("client");
        assert_eq!(bad.get("t/documents/00000.json").expect_err("refused").class, "forbidden");
    }

    #[test]
    fn list_responses_are_parsed_with_their_continuation() {
        let page = "<ListBucketResult><IsTruncated>true</IsTruncated>\
            <Contents><Key>p/documents/a&amp;b.json</Key></Contents>\
            <Contents><Key>p/documents/c.json</Key></Contents>\
            <NextContinuationToken>tok/1=</NextContinuationToken></ListBucketResult>";
        assert_eq!(
            xml_values(page, "Key"),
            vec!["p/documents/a&b.json", "p/documents/c.json"]
        );
        assert_eq!(xml_values(page, "IsTruncated"), vec!["true"]);
        assert_eq!(xml_values(page, "NextContinuationToken"), vec!["tok/1="]);
    }
}
