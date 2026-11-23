//! Sign-in for the archive: OIDC authorization code + PKCE against an existing
//! identity provider (ZITADEL in production), on top of `ogar-oidc` and
//! `ogar-auth`.
//!
//! Off unless `PAPERLESS_OIDC_ISSUER` is set; then every page except `/auth/*`
//! and `/healthz` needs a session:
//!
//! 1. `GET /auth/login` makes a `state`, a `nonce` and a PKCE verifier, keeps them
//!    server-side, sets a short-lived cookie holding `state`, and redirects to the
//!    provider's authorization endpoint (`code_challenge_method=S256`).
//! 2. `GET /auth/callback` requires the cookie's `state` to equal the query's,
//!    consumes the pending login (single use), exchanges the code with the verifier,
//!    and verifies the ID token with [`OidcVerifier`] (signature, issuer, audience,
//!    expiry, nonce). The token must then carry one of the allowed project roles
//!    (`PAPERLESS_OIDC_ROLES`, optionally only under the organizations in
//!    `PAPERLESS_OIDC_ORGS`). Only then `ogar_auth::federation::establish` produces
//!    the verified login and a session is created, in an `HttpOnly`, `Secure`,
//!    `SameSite=Lax` `__Host-` cookie.
//!
//! A service that calls the archive on a user's behalf (report-rs, through its own
//! Zitadel application) sends that user's token as `Authorization: Bearer`. It is
//! accepted when it is addressed to one of `PAPERLESS_API_CLIENTS`, with the same
//! checks minus the nonce and the same role rule, and creates no session. A request
//! that sends an `Authorization` header is never redirected: a bad token is a 401.
//!
//! The provider assigns the roles; this archive only checks that one of the listed
//! ones is present. A role revoked at the provider takes effect at the next
//! sign-in, or when the ID token carrying it expires (a session never outlives its
//! token).
//!
//! Discovery and JWKS are fetched lazily and cached; an unknown `kid` refetches the
//! JWKS once (key rotation).
//!
//! ## Sessions survive a restart
//!
//! A `VerifiedLogin` has no constructor besides `establish`, so the cookie carries
//! what `establish` needs: the ID token and its nonce, with a random session id and
//! the session expiry, sealed with XChaCha20-Poly1305 (`ogar-encryption`) under
//! `PAPERLESS_SESSION_KEY`. A process that sees a cookie it does not know yet opens
//! it and verifies the token again. Without `PAPERLESS_SESSION_KEY` the key is random
//! per process and sessions end at a restart.
//!
//! The design follows report-rs's `src/login.rs` (RPT-LOGIN), without bootstrap
//! bindings and with a role allow-list instead of a policy file.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::Router;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ogar_auth::federation::{establish, AssertionVerifier, TrustedIssuers, VerifiedLogin};
use ogar_auth::identity::{ExternalIdentity, Issuer};
use ogar_auth::user::{AuthBinding, ProviderId, User, UserId, UserStore};
use ogar_encryption::aead::{open_with_key, seal_with_key, NONCE_LEN};
use ogar_oidc::{Client, Discovery, IdToken, KeySet, OidcError, OidcVerifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Session cookie (the `__Host-` prefix pins it to this host, `Secure` and `Path=/`).
pub const SESSION_COOKIE: &str = "__Host-paperless_sid";
/// Login-in-progress cookie holding `state`.
pub const LOGIN_COOKIE: &str = "__Host-paperless_login";
/// How long a started login may take.
pub const LOGIN_TTL: Duration = Duration::from_secs(600);
/// Absolute session lifetime.
pub const SESSION_TTL: Duration = Duration::from_secs(8 * 3600);
const MAX_PENDING: usize = 4096;
const MAX_SESSIONS: usize = 10_000;
/// Browsers keep cookies up to about 4096 bytes including name and attributes.
const MAX_SESSION_COOKIE: usize = 3800;
const SEALED_V1: &str = "v1.";
/// Provider label recorded on the login; never an authorization input.
const PROVIDER: ProviderId = ProviderId("oidc");
/// The claim ZITADEL puts the project roles in (`{role_key: {org_id: org_domain}}`).
pub const DEFAULT_ROLES_CLAIM: &str = "urn:zitadel:iam:org:project:roles";
/// Scopes requested besides `openid`: the project roles.
pub const DEFAULT_EXTRA_SCOPES: &str = "urn:zitadel:iam:org:projects:roles";
/// Where a sign-in lands when the request named no page.
const DEFAULT_RETURN_TO: &str = "/";

/// Whether sign-in is on in this process. Templates read it to show "Sign out".
static ENABLED: OnceLock<bool> = OnceLock::new();

/// `true` once [`protect`] wrapped the app with a sign-in.
pub fn enabled() -> bool {
    ENABLED.get().copied().unwrap_or(false)
}

/// Everything read from the environment.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    issuer: Issuer,
    client_id: String,
    /// Absolute https URL of `/auth/callback`, as registered at the provider.
    redirect_url: String,
    /// `scheme://host[:port]` of `redirect_url`: the archive's own origin. A
    /// cookie-authenticated request that changes state must come from it.
    origin: String,
    /// Only for a confidential client. Never logged.
    client_secret: Option<String>,
    /// `PAPERLESS_SESSION_KEY`, 32 bytes. `None`: random per process. Never logged.
    session_key: Option<SessionKey>,
    roles_claim: String,
    extra_scopes: String,
    /// Role keys that admit a user. Never empty.
    allowed_roles: Vec<String>,
    /// When non-empty, an allowed role counts only under one of these organizations.
    allowed_orgs: Vec<String>,
    /// `PAPERLESS_API_CLIENTS`: client ids whose tokens are accepted as
    /// `Authorization: Bearer` (a service calling on a user's behalf). Empty: none.
    api_clients: Vec<String>,
}

#[derive(Clone)]
struct SessionKey([u8; 32]);

impl std::fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionKey(redacted)")
    }
}

/// Every variable [`AuthConfig::from_vars`] reads.
const ENV_VARS: [&str; 10] = [
    "PAPERLESS_OIDC_ISSUER",
    "PAPERLESS_OIDC_CLIENT_ID",
    "PAPERLESS_OIDC_REDIRECT_URL",
    "PAPERLESS_OIDC_ROLES",
    "PAPERLESS_OIDC_ORGS",
    "PAPERLESS_OIDC_CLIENT_SECRET",
    "PAPERLESS_OIDC_ROLES_CLAIM",
    "PAPERLESS_OIDC_EXTRA_SCOPES",
    "PAPERLESS_SESSION_KEY",
    "PAPERLESS_API_CLIENTS",
];

/// The origin of a callback URL, when it is exactly `https://<host>[:port]/auth/callback`
/// with no credentials, query or fragment. Parsed, so `/wrong?x=/auth/callback` fails.
fn callback_origin(url: &str) -> Option<String> {
    let u = reqwest::Url::parse(url).ok()?;
    let ok = u.scheme() == "https"
        && u.host_str().is_some_and(|h| !h.is_empty())
        && u.username().is_empty()
        && u.password().is_none()
        && u.path() == "/auth/callback"
        && u.query().is_none()
        && u.fragment().is_none();
    ok.then(|| u.origin().ascii_serialization())
}

/// A comma or whitespace separated list of plain keys. A `/` or `*` could forge or
/// widen a match, so it refuses the boot instead of being skipped.
fn parse_list(var: &str, s: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for item in s
        .split([',', ' ', '\n', '\t'])
        .map(str::trim)
        .filter(|x| !x.is_empty())
    {
        if item.contains('/') || item.contains('*') {
            return Err(format!("{var}: an entry contains '/' or '*'"));
        }
        if !out.iter().any(|o| o == item) {
            out.push(item.to_string());
        }
    }
    Ok(out)
}

/// The client ids of `PAPERLESS_API_CLIENTS` (comma separated, as zammad-ticket's
/// `TICKET_API_CLIENTS`): none when unset. An empty item or one with whitespace is an
/// error, never skipped.
fn parse_api_clients(value: Option<String>) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let mut ids: Vec<String> = Vec::new();
    for item in value.split(',') {
        let id = item.trim();
        if id.is_empty() || id.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err("PAPERLESS_API_CLIENTS: an item is empty or contains whitespace".into());
        }
        if !ids.iter().any(|known| known == id) {
            ids.push(id.to_owned());
        }
    }
    Ok(ids)
}

/// 32 bytes as base64 (URL-safe or standard, with or without padding) or 64 hex
/// digits. Anything else is an error, never a shorter key.
fn parse_session_key(k: &str) -> Result<[u8; 32], String> {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE};
    let bytes = if k.len() == 64 && k.bytes().all(|b| b.is_ascii_hexdigit()) {
        (0..32)
            .map(|i| u8::from_str_radix(&k[2 * i..2 * i + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
            .ok()
    } else {
        [&URL_SAFE_NO_PAD, &URL_SAFE, &STANDARD, &STANDARD_NO_PAD]
            .iter()
            .find_map(|e| e.decode(k).ok())
    };
    bytes
        .and_then(|b| <[u8; 32]>::try_from(b).ok())
        .ok_or_else(|| "PAPERLESS_SESSION_KEY must be 32 bytes (base64 or 64 hex digits)".into())
}

impl AuthConfig {
    /// `Ok(None)` when `PAPERLESS_OIDC_ISSUER` is unset or blank. Otherwise the
    /// required variables must be present and valid; a bad value is an error, never
    /// a silent fallback.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>, String> {
        let present = |k: &str| {
            get(k)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let Some(issuer) = present("PAPERLESS_OIDC_ISSUER") else {
            return Ok(None);
        };
        let issuer =
            Issuer::new(&issuer).map_err(|_| "PAPERLESS_OIDC_ISSUER is not an https issuer")?;
        let client_id =
            present("PAPERLESS_OIDC_CLIENT_ID").ok_or("PAPERLESS_OIDC_CLIENT_ID is required")?;
        let redirect_url = present("PAPERLESS_OIDC_REDIRECT_URL")
            .ok_or("PAPERLESS_OIDC_REDIRECT_URL is required")?;
        let origin = callback_origin(&redirect_url).ok_or(
            "PAPERLESS_OIDC_REDIRECT_URL must be https://<host>/auth/callback (no query or fragment)",
        )?;
        let allowed_roles = parse_list(
            "PAPERLESS_OIDC_ROLES",
            &present("PAPERLESS_OIDC_ROLES").unwrap_or_default(),
        )?;
        if allowed_roles.is_empty() {
            return Err(
                "PAPERLESS_OIDC_ROLES is required (the project roles that admit a user)".into(),
            );
        }
        let allowed_orgs = parse_list(
            "PAPERLESS_OIDC_ORGS",
            &present("PAPERLESS_OIDC_ORGS").unwrap_or_default(),
        )?;
        // A set value that lists no organization would silently lift the restriction.
        if allowed_orgs.is_empty() && present("PAPERLESS_OIDC_ORGS").is_some() {
            return Err("PAPERLESS_OIDC_ORGS must list at least one organization id".into());
        }
        let api_clients = parse_api_clients(present("PAPERLESS_API_CLIENTS"))?;
        let session_key = match present("PAPERLESS_SESSION_KEY") {
            None => None,
            Some(k) => Some(SessionKey(parse_session_key(&k)?)),
        };
        Ok(Some(Self {
            issuer,
            client_id,
            redirect_url,
            origin,
            client_secret: present("PAPERLESS_OIDC_CLIENT_SECRET"),
            session_key,
            roles_claim: present("PAPERLESS_OIDC_ROLES_CLAIM")
                .unwrap_or_else(|| DEFAULT_ROLES_CLAIM.to_string()),
            extra_scopes: present("PAPERLESS_OIDC_EXTRA_SCOPES")
                .unwrap_or_else(|| DEFAULT_EXTRA_SCOPES.to_string()),
            allowed_roles,
            allowed_orgs,
            api_clients,
        }))
    }

    /// Reads the process environment. A variable that is set but not valid Unicode
    /// is an error: dropping it would turn sign-in off or lift a restriction.
    pub fn from_env() -> Result<Option<Self>, String> {
        for name in ENV_VARS {
            if std::env::var_os(name).is_some_and(|v| v.to_str().is_none()) {
                return Err(format!("{name} is not valid Unicode"));
            }
        }
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// One log line naming what is configured, never a secret or an identity.
    pub fn summary(&self) -> String {
        format!(
            "sign-in via {} (client {}), roles [{}]{}{}{}",
            self.issuer.as_str(),
            self.client_id,
            self.allowed_roles.join(", "),
            if self.allowed_orgs.is_empty() {
                String::new()
            } else {
                format!(", {} organization(s)", self.allowed_orgs.len())
            },
            if self.api_clients.is_empty() {
                String::new()
            } else {
                format!(
                    ", bearer tokens of {} API client(s)",
                    self.api_clients.len()
                )
            },
            if self.session_key.is_some() {
                ""
            } else {
                "; PAPERLESS_SESSION_KEY unset: sessions end at a restart"
            }
        )
    }
}

/// The two HTTP calls the flow makes, so tests can run it without a network.
pub trait OidcHttp: Send + Sync {
    fn get<'a>(
        &'a self,
        url: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>>;
    fn post_form<'a>(
        &'a self,
        url: &'a str,
        form: &'a [(&'a str, &'a str)],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>>;
}

/// reqwest over rustls, 10 s timeout, https only, no redirects followed.
pub struct ReqwestHttp(reqwest::Client);

impl ReqwestHttp {
    pub fn new() -> Result<Self, String> {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .build()
            .map(Self)
            .map_err(|_| "could not build the OIDC HTTP client".to_string())
    }
}

async fn body(r: reqwest::Result<reqwest::Response>) -> Result<Vec<u8>, String> {
    let r = r.map_err(|_| "request failed".to_string())?;
    if !r.status().is_success() {
        return Err(format!("status {}", r.status().as_u16()));
    }
    r.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|_| "body read failed".to_string())
}

impl OidcHttp for ReqwestHttp {
    fn get<'a>(
        &'a self,
        url: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>> {
        Box::pin(async move { body(self.0.get(url).send().await).await })
    }
    fn post_form<'a>(
        &'a self,
        url: &'a str,
        form: &'a [(&'a str, &'a str)],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>> {
        Box::pin(async move { body(self.0.post(url).form(form).send().await).await })
    }
}

#[derive(Debug, Clone)]
struct Endpoints {
    authorization: String,
    token: String,
    jwks: String,
}

struct Pending {
    verifier: String,
    nonce: String,
    return_to: String,
    created: SystemTime,
}

/// What the sealed cookie carries: enough to verify the login again, nothing else.
#[derive(Serialize, Deserialize)]
struct Sealed {
    sid: String,
    tok: String,
    nonce: String,
    exp: u64,
}

struct Session {
    /// Kept so the session is provably one `establish` produced.
    login: VerifiedLogin,
    expires: SystemTime,
}

/// Why a callback was refused. Only the class is ever shown or logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackError {
    /// No or mismatching login cookie, unknown or expired `state`.
    State,
    /// The provider returned an error instead of a code.
    Provider,
    /// Discovery, JWKS or the token endpoint failed.
    Upstream,
    /// The ID token did not verify.
    Token,
    /// The token carries no project role at all (fix at the provider: grant a role,
    /// and enable "User roles inside ID Token" on the application).
    Unassigned,
    /// The token carries roles, but none of `PAPERLESS_OIDC_ROLES` (under
    /// `PAPERLESS_OIDC_ORGS` when set).
    Denied,
}

impl CallbackError {
    pub fn class(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::Provider => "provider",
            Self::Upstream => "upstream",
            Self::Token => "token",
            Self::Unassigned => "unassigned",
            Self::Denied => "denied",
        }
    }
}

/// The sign-in state of one process.
pub struct Login {
    cfg: AuthConfig,
    http: Box<dyn OidcHttp>,
    trusted: TrustedIssuers,
    endpoints: RwLock<Option<Endpoints>>,
    verifier: RwLock<OidcVerifier>,
    /// The issuer's keys for bearer tokens, refreshed with the sign-in keys.
    bearer_keys: RwLock<KeySet>,
    pending: Mutex<HashMap<String, Pending>>,
    sessions: Mutex<HashMap<String, Session>>,
    revoked: Mutex<HashMap<String, u64>>,
    key: [u8; 32],
    aad: Vec<u8>,
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    // An OS without entropy cannot run a sign-in safely; refusing is the only answer.
    getrandom::getrandom(&mut b).expect("OS entropy available");
    b
}

fn random_token() -> String {
    URL_SAFE_NO_PAD.encode(random_bytes::<32>())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn claims_of(token: &str) -> Option<serde_json::Value> {
    let payload = token.split('.').nth(1)?;
    serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).ok()?).ok()
}

/// `exp` of an ID token the verifier has already accepted.
fn token_exp(token: &str) -> Option<u64> {
    claims_of(token)?.get("exp")?.as_u64()
}

/// The `(organization, role_key)` pairs in claim `claim` of a verified ID token
/// (`{role_key: {org_id: org_domain}}`). Anything else is no role. Sorted, deduped.
fn token_roles(token: &str, claim: &str) -> Vec<(String, String)> {
    let Some(claims) = claims_of(token) else {
        return Vec::new();
    };
    let Some(roles) = claims.get(claim).and_then(|r| r.as_object()) else {
        return Vec::new();
    };
    let mut out: Vec<(String, String)> = roles
        .iter()
        .filter(|(key, _)| !key.trim().is_empty())
        .filter_map(|(key, orgs)| orgs.as_object().map(|o| (key, o)))
        .flat_map(|(key, orgs)| {
            orgs.keys()
                .filter(|org| !org.trim().is_empty())
                .map(move |org| (org.clone(), key.clone()))
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The allowed role keys a token's roles grant: exact role-key match, and the
/// organization must be listed when `orgs` is non-empty.
fn admitted_roles(roles: &[(String, String)], allowed: &[String], orgs: &[String]) -> Vec<String> {
    let mut out: Vec<String> = roles
        .iter()
        .filter(|(org, role)| {
            allowed.iter().any(|a| a == role) && (orgs.is_empty() || orgs.iter().any(|o| o == org))
        })
        .map(|(_, role)| role.clone())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// `true` for a same-origin relative path: one leading `/`, no `//`, no backslash,
/// no control characters. Anything else would make `/auth/login` an open redirect.
pub fn safe_return_to(p: &str) -> bool {
    p.starts_with('/')
        && !p.starts_with("//")
        && !p.contains('\\')
        && !p.chars().any(char::is_control)
}

fn https(url: &str) -> Result<String, CallbackError> {
    if url.starts_with("https://") {
        Ok(url.to_string())
    } else {
        Err(CallbackError::Upstream)
    }
}

/// The one-user store a login resolves against: `user` for exactly `identity`.
struct OneUser {
    identity: ExternalIdentity,
    user: User,
}

impl UserStore for OneUser {
    fn resolve(&self, _binding: &AuthBinding) -> Option<&User> {
        None
    }
    fn user(&self, id: UserId) -> Option<&User> {
        (id == self.user.id).then_some(&self.user)
    }
    fn resolve_external(&self, id: &ExternalIdentity) -> Option<&User> {
        (id == &self.identity).then_some(&self.user)
    }
}

impl Login {
    pub fn new(cfg: AuthConfig, http: Box<dyn OidcHttp>) -> Result<Self, String> {
        let mut trusted = TrustedIssuers::new();
        trusted
            .insert(cfg.issuer.clone(), PROVIDER)
            .map_err(|_| "issuer registered twice".to_string())?;
        let key = match &cfg.session_key {
            Some(k) => k.0,
            None => random_bytes::<32>(),
        };
        let aad = format!(
            "paperless session v1|{SESSION_COOKIE}|{}|{}",
            cfg.issuer.as_str(),
            cfg.client_id
        )
        .into_bytes();
        let mut verifier = OidcVerifier::new();
        verifier.add_client(Client::new(
            cfg.issuer.clone(),
            cfg.client_id.clone(),
            KeySet::default(),
        ));
        Ok(Self {
            cfg,
            http,
            trusted,
            endpoints: RwLock::new(None),
            verifier: RwLock::new(verifier),
            bearer_keys: RwLock::new(KeySet::default()),
            pending: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
            revoked: Mutex::new(HashMap::new()),
            key,
            aad,
        })
    }

    fn seal(&self, s: &Sealed) -> Option<String> {
        let nonce = random_bytes::<NONCE_LEN>();
        let plain = serde_json::to_vec(s).ok()?;
        let ct = seal_with_key(&self.key, &nonce, &self.aad, &plain).ok()?;
        let mut raw = nonce.to_vec();
        raw.extend_from_slice(&ct);
        Some(format!("{SEALED_V1}{}", URL_SAFE_NO_PAD.encode(raw)))
    }

    /// Open a session cookie: authentic, unexpired and not signed out here.
    fn open(&self, cookie: &str) -> Option<Sealed> {
        let raw = URL_SAFE_NO_PAD
            .decode(cookie.strip_prefix(SEALED_V1)?)
            .ok()?;
        if raw.len() <= NONCE_LEN {
            return None;
        }
        let (nonce, ct) = raw.split_at(NONCE_LEN);
        let nonce: &[u8; NONCE_LEN] = nonce.try_into().ok()?;
        let plain = open_with_key(&self.key, nonce, &self.aad, ct).ok()?;
        let s: Sealed = serde_json::from_slice(&plain).ok()?;
        if s.exp <= now_unix() || self.revoked.lock().ok()?.contains_key(&s.sid) {
            return None;
        }
        Some(s)
    }

    fn remember(&self, sid: String, session: Session) {
        let Ok(mut s) = self.sessions.lock() else {
            return;
        };
        let now = SystemTime::now();
        s.retain(|_, v| v.expires > now);
        if s.len() >= MAX_SESSIONS {
            if let Some(oldest) = s
                .iter()
                .min_by_key(|(_, v)| v.expires)
                .map(|(k, _)| k.clone())
            {
                s.remove(&oldest);
            }
        }
        s.insert(sid, session);
    }

    fn verify_once(
        &self,
        input: &IdToken,
    ) -> Result<ogar_auth::federation::VerifiedClaims, OidcError> {
        let v = self.verifier.read().map_err(|_| OidcError::Malformed)?;
        AssertionVerifier::verify(&*v, &self.trusted, input)
    }

    /// Verify an ID token (refetching the JWKS once on an unknown key), check the
    /// roles, and establish the login.
    async fn verify_token(
        &self,
        token: &str,
        nonce: &str,
        jwks: &str,
    ) -> Result<VerifiedLogin, CallbackError> {
        let input = IdToken {
            token: token.to_string(),
            now_unix: now_unix(),
            nonce: Some(nonce.to_string()),
        };
        let claims = match self.verify_once(&input) {
            Err(OidcError::UnknownKey) => {
                self.refresh_keys(jwks).await?;
                self.verify_once(&input)
            }
            other => other,
        }
        .map_err(|_| CallbackError::Token)?;
        let v = self.verifier.read().map_err(|_| CallbackError::Token)?;
        self.admit(&v, &input, &claims)
    }

    /// The role rule and `establish`, for a token `verifier` has accepted.
    fn admit(
        &self,
        verifier: &OidcVerifier,
        input: &IdToken,
        claims: &ogar_auth::federation::VerifiedClaims,
    ) -> Result<VerifiedLogin, CallbackError> {
        let token = input.token.as_str();
        // From here the token is authentic, so its roles claim may be read.
        let roles = token_roles(token, &self.cfg.roles_claim);
        if roles.is_empty() {
            return Err(CallbackError::Unassigned);
        }
        let admitted = admitted_roles(&roles, &self.cfg.allowed_roles, &self.cfg.allowed_orgs);
        if admitted.is_empty() {
            return Err(CallbackError::Denied);
        }
        let store = OneUser {
            identity: claims.identity.clone(),
            user: User {
                id: UserId(0),
                subject: format!("oidc:{}", claims.identity.subject().as_str()),
                tenant: 0,
                roles: admitted,
                memberships: Vec::new(),
                bindings: Vec::new(),
                key_refs: Vec::new(),
            },
        };
        establish(verifier, &self.trusted, input, &store).map_err(|_| CallbackError::Token)
    }

    /// `true` when bearer tokens are accepted at all.
    pub fn accepts_bearer(&self) -> bool {
        !self.cfg.api_clients.is_empty()
    }

    /// A token a service presents as `Authorization: Bearer` on a user's behalf: the
    /// same checks as an ID token (signature, issuer, audience, expiry) for one of the
    /// `PAPERLESS_API_CLIENTS`, without a nonce and without a session, then the same
    /// role rule. Returns the subject. The JWKS is refetched once on an unknown key.
    pub async fn verify_bearer(&self, token: &str) -> Result<String, CallbackError> {
        if self.cfg.api_clients.is_empty() {
            return Err(CallbackError::Token);
        }
        let input = IdToken {
            token: token.to_string(),
            now_unix: now_unix(),
            nonce: None,
        };
        let login = match self.verify_as_api_client(&input) {
            Err(CallbackError::Upstream) => {
                let ep = self.endpoints().await?;
                self.refresh_keys(&ep.jwks).await?;
                self.verify_as_api_client(&input)
            }
            other => other,
        }?;
        Ok(login.authenticated_user().user.subject.clone())
    }

    /// Try the token against one verifier per accepted API client, all over the shared
    /// keys. `Upstream` means "no key matched": refetch and try once more.
    fn verify_as_api_client(&self, input: &IdToken) -> Result<VerifiedLogin, CallbackError> {
        let keys = self
            .bearer_keys
            .read()
            .map_err(|_| CallbackError::Token)?
            .clone();
        let mut unknown_key = false;
        for client_id in &self.cfg.api_clients {
            let mut verifier = OidcVerifier::new();
            verifier.add_client(Client::new(
                self.cfg.issuer.clone(),
                client_id.clone(),
                keys.clone(),
            ));
            match AssertionVerifier::verify(&verifier, &self.trusted, input) {
                Ok(claims) => return self.admit(&verifier, input, &claims),
                Err(OidcError::UnknownKey) => unknown_key = true,
                Err(_) => {}
            }
        }
        Err(if unknown_key {
            CallbackError::Upstream
        } else {
            CallbackError::Token
        })
    }

    async fn endpoints(&self) -> Result<Endpoints, CallbackError> {
        if let Some(e) = self.endpoints.read().ok().and_then(|g| g.clone()) {
            return Ok(e);
        }
        let url = format!(
            "{}/.well-known/openid-configuration",
            self.cfg.issuer.as_str().trim_end_matches('/')
        );
        let doc = self
            .http
            .get(&url)
            .await
            .map_err(|_| CallbackError::Upstream)?;
        let d =
            Discovery::from_json(&self.cfg.issuer, &doc).map_err(|_| CallbackError::Upstream)?;
        let v: serde_json::Value =
            serde_json::from_slice(&doc).map_err(|_| CallbackError::Upstream)?;
        let field = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_str())
                .ok_or(CallbackError::Upstream)
                .and_then(https)
        };
        let e = Endpoints {
            authorization: field("authorization_endpoint")?,
            token: field("token_endpoint")?,
            jwks: d.jwks_uri,
        };
        if let Ok(mut g) = self.endpoints.write() {
            *g = Some(e.clone());
        }
        Ok(e)
    }

    async fn refresh_keys(&self, jwks_uri: &str) -> Result<(), CallbackError> {
        let doc = self
            .http
            .get(jwks_uri)
            .await
            .map_err(|_| CallbackError::Upstream)?;
        let keys = KeySet::from_jwks_json(&doc).map_err(|_| CallbackError::Upstream)?;
        if let Ok(mut b) = self.bearer_keys.write() {
            *b = keys.clone();
        }
        let mut v = self.verifier.write().map_err(|_| CallbackError::Upstream)?;
        if let Some(c) = v.client_mut(&self.cfg.issuer) {
            c.set_keys(keys);
        }
        Ok(())
    }

    /// Start a login: the authorization URL and the `state` for the login cookie.
    pub async fn start(&self, return_to: Option<&str>) -> Result<(String, String), CallbackError> {
        let ep = self.endpoints().await?;
        let state = random_token();
        let nonce = random_token();
        let verifier = random_token();
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let return_to = return_to
            .filter(|p| safe_return_to(p))
            .unwrap_or(DEFAULT_RETURN_TO)
            .to_string();
        {
            let mut p = self.pending.lock().map_err(|_| CallbackError::State)?;
            p.retain(|_, v| v.created.elapsed().map(|e| e < LOGIN_TTL).unwrap_or(false));
            if p.len() >= MAX_PENDING {
                if let Some(oldest) = p
                    .iter()
                    .min_by_key(|(_, v)| v.created)
                    .map(|(k, _)| k.clone())
                {
                    p.remove(&oldest);
                }
            }
            p.insert(
                state.clone(),
                Pending {
                    verifier,
                    nonce: nonce.clone(),
                    return_to,
                    created: SystemTime::now(),
                },
            );
        }
        let scope = format!("openid {}", self.cfg.extra_scopes);
        let q = [
            ("response_type", "code"),
            ("client_id", self.cfg.client_id.as_str()),
            ("redirect_uri", self.cfg.redirect_url.as_str()),
            ("scope", scope.trim()),
            ("state", state.as_str()),
            ("nonce", nonce.as_str()),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
        ];
        let query: String = q
            .iter()
            .map(|(k, v)| format!("{k}={}", urlencode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let sep = if ep.authorization.contains('?') {
            '&'
        } else {
            '?'
        };
        Ok((format!("{}{sep}{query}", ep.authorization), state))
    }

    /// Finish a login: the session cookie value and where to send the browser.
    pub async fn finish(
        &self,
        cookie_state: Option<&str>,
        query_state: Option<&str>,
        code: Option<&str>,
        provider_error: bool,
    ) -> Result<(String, String), CallbackError> {
        let (Some(cs), Some(qs)) = (cookie_state, query_state) else {
            return Err(CallbackError::State);
        };
        if cs != qs {
            return Err(CallbackError::State);
        }
        let pending = self
            .pending
            .lock()
            .map_err(|_| CallbackError::State)?
            .remove(qs)
            .ok_or(CallbackError::State)?;
        if !pending
            .created
            .elapsed()
            .map(|e| e < LOGIN_TTL)
            .unwrap_or(false)
        {
            return Err(CallbackError::State);
        }
        if provider_error {
            return Err(CallbackError::Provider);
        }
        let code = code
            .filter(|c| !c.is_empty())
            .ok_or(CallbackError::Provider)?;
        let ep = self.endpoints().await?;
        let mut form: Vec<(&str, &str)> = vec![
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", self.cfg.redirect_url.as_str()),
            ("client_id", self.cfg.client_id.as_str()),
            ("code_verifier", pending.verifier.as_str()),
        ];
        if let Some(s) = &self.cfg.client_secret {
            form.push(("client_secret", s.as_str()));
        }
        let resp = self
            .http
            .post_form(&ep.token, &form)
            .await
            .map_err(|_| CallbackError::Upstream)?;
        let v: serde_json::Value =
            serde_json::from_slice(&resp).map_err(|_| CallbackError::Upstream)?;
        let token = v
            .get("id_token")
            .and_then(|t| t.as_str())
            .ok_or(CallbackError::Upstream)?;
        let login = self.verify_token(token, &pending.nonce, &ep.jwks).await?;
        let sid = random_token();
        let exp =
            (now_unix() + SESSION_TTL.as_secs()).min(token_exp(token).ok_or(CallbackError::Token)?);
        let cookie = self
            .seal(&Sealed {
                sid: sid.clone(),
                tok: token.to_string(),
                nonce: pending.nonce.clone(),
                exp,
            })
            .ok_or(CallbackError::Upstream)?;
        if cookie.len() > MAX_SESSION_COOKIE {
            eprintln!("auth: the ID token is too large for a session cookie");
            return Err(CallbackError::Upstream);
        }
        self.remember(
            sid,
            Session {
                login,
                expires: UNIX_EPOCH + Duration::from_secs(exp),
            },
        );
        Ok((cookie, pending.return_to))
    }

    /// Make a cookie this process has not seen usable: open it and verify the token
    /// it carries again. A no-op for a known or unusable cookie.
    pub async fn restore(&self, cookie: &str) {
        let Some(s) = self.open(cookie) else {
            return;
        };
        if self
            .sessions
            .lock()
            .map(|m| m.contains_key(&s.sid))
            .unwrap_or(true)
        {
            return;
        }
        let Ok(ep) = self.endpoints().await else {
            return;
        };
        let Ok(login) = self.verify_token(&s.tok, &s.nonce, &ep.jwks).await else {
            return;
        };
        let exp = s.exp.min(token_exp(&s.tok).unwrap_or(0));
        self.remember(
            s.sid,
            Session {
                login,
                expires: UNIX_EPOCH + Duration::from_secs(exp),
            },
        );
    }

    /// The subject (`oidc:<sub>`) of a live session cookie.
    /// See [`same_origin_request`]; the origin is the configured callback's.
    pub fn same_origin(&self, method: &Method, headers: &HeaderMap) -> bool {
        same_origin_request(&self.cfg.origin, method, headers)
    }

    pub fn subject(&self, cookie: &str) -> Option<String> {
        let sealed = self.open(cookie)?;
        let s = self.sessions.lock().ok()?;
        s.get(&sealed.sid)
            .filter(|v| v.expires > SystemTime::now())
            .map(|v| v.login.authenticated_user().user.subject.clone())
    }

    /// Sign a session out in this process.
    pub fn logout(&self, cookie: &str) {
        if let Some(s) = self.open(cookie) {
            if let Ok(mut m) = self.sessions.lock() {
                m.remove(&s.sid);
            }
            if let Ok(mut r) = self.revoked.lock() {
                let now = now_unix();
                r.retain(|_, exp| *exp > now);
                r.insert(s.sid, s.exp);
            }
        }
    }
}

/// Percent-encode a query value (RFC 3986 unreserved characters stay).
pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// The value of cookie `name` in a `Cookie` header.
pub fn cookie<'a>(header: Option<&'a str>, name: &str) -> Option<&'a str> {
    header?
        .split(';')
        .map(str::trim)
        .find_map(|kv| kv.strip_prefix(name).and_then(|r| r.strip_prefix('=')))
        .filter(|v| !v.is_empty())
}

/// `Set-Cookie` for a `__Host-` cookie; `max_age` 0 clears it.
pub fn set_cookie(name: &str, value: &str, max_age: Duration) -> String {
    format!(
        "{name}={value}; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age={}",
        max_age.as_secs()
    )
}

fn cookie_header(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::COOKIE).and_then(|v| v.to_str().ok())
}

// ---------------------------------------------------------------------------
// HTTP surface
// ---------------------------------------------------------------------------

/// Paths that never need a session.
fn is_public(path: &str) -> bool {
    path.starts_with("/auth/") || path == "/healthz"
}

/// The token of exactly one `Authorization: Bearer <token>` header.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    let (Some(value), None) = (values.next(), values.next()) else {
        return None;
    };
    let (scheme, token) = value.to_str().ok()?.split_once(' ')?;
    let plain = !token.is_empty() && !token.chars().any(|c| c.is_whitespace() || c.is_control());
    (scheme.eq_ignore_ascii_case("bearer") && plain).then_some(token)
}

/// Whether a cookie-authenticated request may run. GET and HEAD do not change state
/// and always may. Anything else must come from the archive's own origin: a browser
/// says so with `Sec-Fetch-Site: same-origin` (or `none` for a typed URL) or with an
/// `Origin` equal to the configured origin. `SameSite=Lax` alone would let a sibling
/// subdomain (same site, other origin) post with the session cookie. A request with
/// neither header is not from a current browser and is let through: every browser
/// this serves sends at least one of them on a POST.
fn same_origin_request(origin: &str, method: &Method, headers: &HeaderMap) -> bool {
    if method == Method::GET || method == Method::HEAD {
        return true;
    }
    if let Some(site) = headers.get("sec-fetch-site") {
        return matches!(site.to_str(), Ok("same-origin" | "none"));
    }
    match headers.get(header::ORIGIN) {
        Some(o) => o.to_str().is_ok_and(|o| o == origin),
        None => true,
    }
}

/// Middleware: a live session, an accepted bearer token, or a redirect to sign-in
/// (GET/HEAD without credentials) / 401.
async fn require_session(State(login): State<Arc<Login>>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    if is_public(&path) {
        return next.run(req).await;
    }
    let sid = cookie(cookie_header(req.headers()), SESSION_COOKIE).map(str::to_string);
    if let Some(sid) = sid.as_deref() {
        if login.subject(sid).is_none() {
            login.restore(sid).await;
        }
        if login.subject(sid).is_some() {
            if !login.same_origin(req.method(), req.headers()) {
                return (StatusCode::FORBIDDEN, "cross-origin request refused").into_response();
            }
            return next.run(req).await;
        }
    }
    // A caller that sent credentials is a service, not a browser: never redirect it.
    if req.headers().contains_key(header::AUTHORIZATION) {
        if login.accepts_bearer() {
            if let Some(token) = bearer_token(req.headers()) {
                if login.verify_bearer(token).await.is_ok() {
                    return next.run(req).await;
                }
            }
        }
        return (StatusCode::UNAUTHORIZED, "invalid bearer token").into_response();
    }
    if req.method() == Method::GET || req.method() == Method::HEAD {
        let target = req
            .uri()
            .path_and_query()
            .map(|p| p.as_str().to_string())
            .unwrap_or_else(|| DEFAULT_RETURN_TO.to_string());
        Redirect::to(&format!("/auth/login?return_to={}", urlencode(&target))).into_response()
    } else {
        (StatusCode::UNAUTHORIZED, "sign in required").into_response()
    }
}

#[derive(Deserialize)]
struct LoginQuery {
    return_to: Option<String>,
}

async fn auth_login(State(login): State<Arc<Login>>, Query(q): Query<LoginQuery>) -> Response {
    match login.start(q.return_to.as_deref()).await {
        Ok((url, state)) => {
            let mut resp = Redirect::to(&url).into_response();
            if let Ok(v) = HeaderValue::from_str(&set_cookie(LOGIN_COOKIE, &state, LOGIN_TTL)) {
                resp.headers_mut().append(header::SET_COOKIE, v);
            }
            resp
        }
        Err(e) => refusal(e),
    }
}

#[derive(Deserialize)]
struct CallbackQuery {
    state: Option<String>,
    code: Option<String>,
    error: Option<String>,
}

async fn auth_callback(
    State(login): State<Arc<Login>>,
    headers: HeaderMap,
    Query(q): Query<CallbackQuery>,
) -> Response {
    let cookie_state = cookie(cookie_header(&headers), LOGIN_COOKIE);
    let result = login
        .finish(
            cookie_state,
            q.state.as_deref(),
            q.code.as_deref(),
            q.error.is_some(),
        )
        .await;
    match result {
        Ok((session, return_to)) => {
            let mut resp = Redirect::to(&return_to).into_response();
            for c in [
                set_cookie(SESSION_COOKIE, &session, SESSION_TTL),
                set_cookie(LOGIN_COOKIE, "", Duration::ZERO),
            ] {
                if let Ok(v) = HeaderValue::from_str(&c) {
                    resp.headers_mut().append(header::SET_COOKIE, v);
                }
            }
            resp
        }
        Err(e) => {
            println!("auth: sign-in refused ({})", e.class());
            refusal(e)
        }
    }
}

async fn auth_logout(State(login): State<Arc<Login>>, headers: HeaderMap) -> Response {
    if let Some(sid) = cookie(cookie_header(&headers), SESSION_COOKIE) {
        login.logout(sid);
    }
    let mut resp = Redirect::to("/auth/signed-out").into_response();
    if let Ok(v) = HeaderValue::from_str(&set_cookie(SESSION_COOKIE, "", Duration::ZERO)) {
        resp.headers_mut().append(header::SET_COOKIE, v);
    }
    resp
}

async fn auth_signed_out() -> Response {
    page(
        StatusCode::OK,
        "Signed out",
        "You are signed out of the archive.",
    )
}

/// A refusal page naming the class only; never an identity, a role or a token.
fn refusal(e: CallbackError) -> Response {
    let (status, text) = match e {
        CallbackError::State => (
            StatusCode::BAD_REQUEST,
            "The sign-in expired or was started elsewhere. Please sign in again.",
        ),
        CallbackError::Provider => (
            StatusCode::BAD_GATEWAY,
            "The identity provider did not complete the sign-in.",
        ),
        CallbackError::Upstream => (
            StatusCode::BAD_GATEWAY,
            "The identity provider could not be reached. Please try again.",
        ),
        CallbackError::Token => (StatusCode::FORBIDDEN, "The sign-in could not be verified."),
        CallbackError::Unassigned | CallbackError::Denied => (
            StatusCode::FORBIDDEN,
            "You are signed in, but access to the archive has not been set up for your account.",
        ),
    };
    page(status, "Sign-in", text)
}

fn page(status: StatusCode, title: &str, text: &str) -> Response {
    let html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>{title}</title></head><body style=\"font-family:system-ui,sans-serif;\
         max-width:36rem;margin:3rem auto;padding:0 1rem\"><h1>{title}</h1><p>{text}</p>\
         <p><a href=\"/auth/login\">Sign in</a></p></body></html>"
    );
    (
        status,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
        .into_response()
}

/// The `/auth/*` routes.
fn auth_router(login: Arc<Login>) -> Router {
    Router::new()
        .route("/auth/login", get(auth_login))
        .route("/auth/callback", get(auth_callback))
        .route("/auth/logout", post(auth_logout))
        .route("/auth/signed-out", get(auth_signed_out))
        .with_state(login)
}

/// Wrap `app` with sign-in when `login` is set; otherwise return it unchanged.
pub fn protect(app: Router, login: Option<Arc<Login>>) -> Router {
    let Some(login) = login else {
        return app;
    };
    let _ = ENABLED.set(true);
    Router::new()
        .merge(auth_router(login.clone()))
        .merge(app.layer(axum::middleware::from_fn_with_state(login, require_session)))
}

#[cfg(test)]
mod tests;
