//! Sign-in tests: configuration, role admission, and the whole code + PKCE flow
//! against a fake provider that signs real EdDSA ID tokens.

use super::*;
use axum::body::Body;
use axum::http::Request as HttpRequest;
use ogar_encryption::sign::Keypair;
use std::collections::BTreeMap;
use tower::ServiceExt;

const ISSUER: &str = "https://idp.example";
const CLIENT: &str = "client-1";
const KID: &str = "k1";

fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let m: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |k| m.get(k).cloned()
}

fn base_vars(extra: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    let mut v = vec![
        ("PAPERLESS_OIDC_ISSUER", ISSUER),
        ("PAPERLESS_OIDC_CLIENT_ID", CLIENT),
        (
            "PAPERLESS_OIDC_REDIRECT_URL",
            "https://paperless.example/auth/callback",
        ),
        ("PAPERLESS_OIDC_ROLES", "paperless.user, paperless.admin"),
    ];
    v.extend_from_slice(extra);
    v
}

fn config(extra: &[(&'static str, &'static str)]) -> AuthConfig {
    AuthConfig::from_vars(vars(&base_vars(extra)))
        .expect("valid config")
        .expect("configured")
}

// ---------------------------------------------------------------- config

#[test]
fn sign_in_is_off_without_an_issuer() {
    assert!(AuthConfig::from_vars(vars(&[])).unwrap().is_none());
    assert!(
        AuthConfig::from_vars(vars(&[("PAPERLESS_OIDC_ISSUER", "  ")]))
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_set_issuer_requires_the_rest() {
    let without = |drop: &str| {
        let v: Vec<_> = base_vars(&[])
            .into_iter()
            .filter(|(k, _)| *k != drop)
            .collect();
        AuthConfig::from_vars(vars(&v)).unwrap_err()
    };
    assert!(without("PAPERLESS_OIDC_CLIENT_ID").contains("CLIENT_ID"));
    assert!(without("PAPERLESS_OIDC_REDIRECT_URL").contains("REDIRECT_URL"));
    assert!(without("PAPERLESS_OIDC_ROLES").contains("ROLES"));
    // And with all of them it is accepted.
    assert!(AuthConfig::from_vars(vars(&base_vars(&[])))
        .unwrap()
        .is_some());
}

#[test]
fn bad_values_refuse_the_boot() {
    let err = |extra: &[(&'static str, &'static str)]| {
        AuthConfig::from_vars(vars(&base_vars(extra))).unwrap_err()
    };
    assert!(err(&[("PAPERLESS_OIDC_ISSUER", "http://idp.example")]).contains("ISSUER"));
    assert!(err(&[(
        "PAPERLESS_OIDC_REDIRECT_URL",
        "http://p.example/auth/callback"
    )])
    .contains("REDIRECT_URL"));
    assert!(
        err(&[("PAPERLESS_OIDC_REDIRECT_URL", "https://p.example/callback")])
            .contains("REDIRECT_URL")
    );
    // The path is parsed, not matched as a suffix.
    for bad in [
        "https://p.example/wrong?next=/auth/callback",
        "https://p.example/auth/callback#x",
        "https://p.example/auth/callback?x=1",
        "https://p.example/x/auth/callback",
        "https://u:pw@p.example/auth/callback",
    ] {
        assert!(
            AuthConfig::from_vars(vars(&base_vars(&[("PAPERLESS_OIDC_REDIRECT_URL", bad)])))
                .unwrap_err()
                .contains("REDIRECT_URL"),
            "{bad}"
        );
    }
    assert_eq!(config(&[]).origin, "https://paperless.example");
    assert!(err(&[("PAPERLESS_OIDC_ROLES", "paperless.*")]).contains("ROLES"));
    assert!(err(&[("PAPERLESS_OIDC_ORGS", "1/2")]).contains("ORGS"));
    // Separators only would silently lift the organization restriction.
    assert!(err(&[("PAPERLESS_OIDC_ORGS", ",")]).contains("ORGS"));
    assert!(config(&[("PAPERLESS_OIDC_ORGS", "100")]).allowed_orgs == ["100"]);
    assert!(err(&[("PAPERLESS_SESSION_KEY", "short")]).contains("SESSION_KEY"));
}

#[test]
fn the_session_key_takes_hex_or_base64_of_32_bytes() {
    let hex = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    assert_eq!(parse_session_key(hex).unwrap()[1], 0x11);
    let b64 = URL_SAFE_NO_PAD.encode([7u8; 32]);
    assert_eq!(parse_session_key(&b64).unwrap(), [7u8; 32]);
    assert!(parse_session_key(&URL_SAFE_NO_PAD.encode([7u8; 31])).is_err());
}

#[test]
fn the_summary_names_no_secret() {
    let c = config(&[
        ("PAPERLESS_OIDC_CLIENT_SECRET", "s3cr3t-value"),
        (
            "PAPERLESS_SESSION_KEY",
            "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
        ),
    ]);
    let s = c.summary();
    assert!(s.contains(ISSUER) && s.contains("paperless.user"));
    assert!(!s.contains("s3cr3t") && !s.contains("00112233"));
    assert!(
        !format!("{c:?}").contains("00112233"),
        "Debug redacts the key"
    );
}

// ---------------------------------------------------------------- roles

fn unsigned(claims: serde_json::Value) -> String {
    format!("h.{}.s", URL_SAFE_NO_PAD.encode(claims.to_string()))
}

#[test]
fn roles_are_read_per_organization_and_matched_exactly() {
    let t = unsigned(serde_json::json!({
        DEFAULT_ROLES_CLAIM: {
            "paperless.user": {"100": "a.example", "200": "b.example"},
            "Paperless.Admin": {"100": "a.example"},
            "report-admin": {"100": "a.example"}
        }
    }));
    let roles = token_roles(&t, DEFAULT_ROLES_CLAIM);
    assert_eq!(roles.len(), 4);
    let allowed = vec!["paperless.user".to_string(), "paperless.admin".to_string()];
    // Case-sensitive: `Paperless.Admin` is not `paperless.admin`; other roles never count.
    assert_eq!(
        admitted_roles(&roles, &allowed, &[]),
        vec!["paperless.user"]
    );
    // With an organization filter, a role keyed under another organization does not count.
    assert_eq!(
        admitted_roles(&roles, &allowed, &["200".to_string()]),
        vec!["paperless.user"]
    );
    assert!(admitted_roles(&roles, &allowed, &["300".to_string()]).is_empty());
    // A token without the claim has no roles.
    assert!(token_roles(&unsigned(serde_json::json!({})), DEFAULT_ROLES_CLAIM).is_empty());
}

#[test]
fn return_to_accepts_only_same_origin_paths() {
    assert!(safe_return_to("/documents?q=x"));
    for bad in [
        "//evil.example",
        "https://evil.example",
        "/\\evil",
        "documents",
        "",
        "/a\nb",
    ] {
        assert!(!safe_return_to(bad), "{bad:?}");
    }
}

#[test]
fn cookie_reads_the_named_value_only() {
    let h = Some("a=1; __Host-paperless_sid=abc; __Host-paperless_sidx=zzz");
    assert_eq!(cookie(h, SESSION_COOKIE), Some("abc"));
    assert_eq!(
        cookie(Some("__Host-paperless_sidx=zzz"), SESSION_COOKIE),
        None
    );
}

// ---------------------------------------------------------------- the flow

/// A fake provider: discovery, a JWKS with one Ed25519 key, and a token endpoint
/// that returns whatever ID token the test placed there.
struct Provider {
    key: Keypair,
    id_token: Mutex<Option<String>>,
    token_forms: Mutex<Vec<Vec<(String, String)>>>,
}

impl Provider {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            key: Keypair::from_seed(&[42u8; 32]),
            id_token: Mutex::new(None),
            token_forms: Mutex::new(Vec::new()),
        })
    }

    fn jwks(&self) -> String {
        serde_json::json!({"keys": [{
            "kty": "OKP", "crv": "Ed25519", "use": "sig", "alg": "EdDSA", "kid": KID,
            "x": URL_SAFE_NO_PAD.encode(self.key.public_key())
        }]})
        .to_string()
    }

    /// A signed ID token for `sub` with `roles` (the ZITADEL claim shape).
    fn sign(&self, sub: &str, nonce: &str, roles: serde_json::Value) -> String {
        self.sign_for(CLIENT, Some(nonce), sub, roles)
    }

    /// A signed token addressed to `aud`, with or without a nonce.
    fn sign_for(
        &self,
        aud: &str,
        nonce: Option<&str>,
        sub: &str,
        roles: serde_json::Value,
    ) -> String {
        let now = now_unix();
        let header = serde_json::json!({"alg": "EdDSA", "kid": KID, "typ": "JWT"});
        let mut claims = serde_json::json!({
            "iss": ISSUER, "sub": sub, "aud": aud,
            "exp": now + 3600, "iat": now,
            DEFAULT_ROLES_CLAIM: roles
        });
        if let Some(n) = nonce {
            claims["nonce"] = serde_json::json!(n);
        }
        let signing_input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(header.to_string()),
            URL_SAFE_NO_PAD.encode(claims.to_string())
        );
        let sig = self.key.sign(signing_input.as_bytes());
        format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(sig))
    }
}

struct FakeHttp(Arc<Provider>);

impl OidcHttp for FakeHttp {
    fn get<'a>(
        &'a self,
        url: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>> {
        let out = match url {
            "https://idp.example/.well-known/openid-configuration" => Ok(serde_json::json!({
                "issuer": ISSUER,
                "authorization_endpoint": "https://idp.example/oauth/v2/authorize",
                "token_endpoint": "https://idp.example/oauth/v2/token",
                "jwks_uri": "https://idp.example/oauth/v2/keys"
            })
            .to_string()
            .into_bytes()),
            "https://idp.example/oauth/v2/keys" => Ok(self.0.jwks().into_bytes()),
            _ => Err("not found".to_string()),
        };
        Box::pin(async move { out })
    }

    fn post_form<'a>(
        &'a self,
        url: &'a str,
        form: &'a [(&'a str, &'a str)],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>> {
        self.0.token_forms.lock().unwrap().push(
            form.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        let out = if url == "https://idp.example/oauth/v2/token" {
            self.0
                .id_token
                .lock()
                .unwrap()
                .clone()
                .map(|t| serde_json::json!({"id_token": t}).to_string().into_bytes())
                .ok_or_else(|| "no token".to_string())
        } else {
            Err("not found".to_string())
        };
        Box::pin(async move { out })
    }
}

const KEY_HEX: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

fn login_with(p: &Arc<Provider>, extra: &[(&'static str, &'static str)]) -> Login {
    Login::new(config(extra), Box::new(FakeHttp(p.clone()))).unwrap()
}

fn query_param(url: &str, name: &str) -> String {
    url.split(['?', '&'])
        .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
        .unwrap_or_else(|| panic!("{name} in {url}"))
        .to_string()
}

/// Run one sign-in for `sub` with `roles`; returns the finish result.
async fn sign_in(
    login: &Login,
    p: &Arc<Provider>,
    sub: &str,
    roles: serde_json::Value,
) -> Result<(String, String), CallbackError> {
    let (url, state) = login.start(Some("/documents")).await.unwrap();
    assert_eq!(query_param(&url, "state"), state);
    let nonce = query_param(&url, "nonce");
    *p.id_token.lock().unwrap() = Some(p.sign(sub, &nonce, roles));
    login
        .finish(Some(&state), Some(&state), Some("code-1"), false)
        .await
}

fn user_role() -> serde_json::Value {
    serde_json::json!({"paperless.user": {"100": "a.example"}})
}

#[tokio::test]
async fn an_allowed_role_signs_in_with_pkce() {
    let p = Provider::new();
    let login = login_with(&p, &[]);
    let (url, state) = login.start(Some("/documents")).await.unwrap();
    assert!(url.starts_with("https://idp.example/oauth/v2/authorize?"));
    assert_eq!(query_param(&url, "code_challenge_method"), "S256");
    assert!(query_param(&url, "scope").contains("urn%3Azitadel%3Aiam%3Aorg%3Aprojects%3Aroles"));
    let challenge = query_param(&url, "code_challenge");
    let nonce = query_param(&url, "nonce");
    *p.id_token.lock().unwrap() = Some(p.sign("user-1", &nonce, user_role()));

    let (cookie_value, return_to) = login
        .finish(Some(&state), Some(&state), Some("code-1"), false)
        .await
        .unwrap();
    assert_eq!(return_to, "/documents");
    assert_eq!(login.subject(&cookie_value).as_deref(), Some("oidc:user-1"));

    // The verifier sent at the token endpoint is the one the challenge commits to.
    let forms = p.token_forms.lock().unwrap();
    let verifier = forms[0]
        .iter()
        .find(|(k, _)| k == "code_verifier")
        .map(|(_, v)| v.clone())
        .unwrap();
    assert_eq!(
        URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes())),
        challenge
    );
}

#[tokio::test]
async fn roles_decide_admission_both_ways() {
    let p = Provider::new();
    let login = login_with(&p, &[]);
    // An allowed role admits.
    assert!(sign_in(&login, &p, "u1", user_role()).await.is_ok());
    // Roles, but none of the allowed ones: denied.
    let other = serde_json::json!({"report-admin": {"100": "a.example"}});
    assert_eq!(
        sign_in(&login, &p, "u2", other).await.unwrap_err(),
        CallbackError::Denied
    );
    // No roles at all: unassigned.
    assert_eq!(
        sign_in(&login, &p, "u3", serde_json::json!({}))
            .await
            .unwrap_err(),
        CallbackError::Unassigned
    );
    // An organization filter turns away the same role under another organization.
    let scoped = login_with(&p, &[("PAPERLESS_OIDC_ORGS", "999")]);
    assert_eq!(
        sign_in(&scoped, &p, "u1", user_role()).await.unwrap_err(),
        CallbackError::Denied
    );
}

#[tokio::test]
async fn state_is_checked_and_single_use() {
    let p = Provider::new();
    let login = login_with(&p, &[]);
    let (url, state) = login.start(None).await.unwrap();
    *p.id_token.lock().unwrap() = Some(p.sign("u1", &query_param(&url, "nonce"), user_role()));
    // Cookie and query must agree.
    assert_eq!(
        login
            .finish(Some("other"), Some(&state), Some("c"), false)
            .await
            .unwrap_err(),
        CallbackError::State
    );
    assert_eq!(
        login
            .finish(None, Some(&state), Some("c"), false)
            .await
            .unwrap_err(),
        CallbackError::State
    );
    // The mismatch did not consume it; the right pair works once.
    let (_, return_to) = login
        .finish(Some(&state), Some(&state), Some("c"), false)
        .await
        .unwrap();
    assert_eq!(return_to, "/", "no return_to lands on the upload page");
    assert_eq!(
        login
            .finish(Some(&state), Some(&state), Some("c"), false)
            .await
            .unwrap_err(),
        CallbackError::State
    );
}

#[tokio::test]
async fn a_token_for_another_nonce_or_a_forged_signature_is_refused() {
    let p = Provider::new();
    let login = login_with(&p, &[]);
    let (_, state) = login.start(None).await.unwrap();
    *p.id_token.lock().unwrap() = Some(p.sign("u1", "not-the-nonce", user_role()));
    assert_eq!(
        login
            .finish(Some(&state), Some(&state), Some("c"), false)
            .await
            .unwrap_err(),
        CallbackError::Token
    );

    let (url, state) = login.start(None).await.unwrap();
    let good = p.sign("u1", &query_param(&url, "nonce"), user_role());
    // Swap in a role the user does not hold, keeping the old signature.
    let mut parts: Vec<&str> = good.split('.').collect();
    let mut claims: serde_json::Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap();
    claims[DEFAULT_ROLES_CLAIM] = serde_json::json!({"paperless.admin": {"100": "a"}});
    let forged_payload = URL_SAFE_NO_PAD.encode(claims.to_string());
    parts[1] = &forged_payload;
    *p.id_token.lock().unwrap() = Some(parts.join("."));
    assert_eq!(
        login
            .finish(Some(&state), Some(&state), Some("c"), false)
            .await
            .unwrap_err(),
        CallbackError::Token
    );
}

#[tokio::test]
async fn sessions_survive_a_restart_only_with_a_session_key() {
    let p = Provider::new();
    let first = login_with(&p, &[("PAPERLESS_SESSION_KEY", KEY_HEX)]);
    let (cookie_value, _) = sign_in(&first, &p, "u1", user_role()).await.unwrap();

    // A new process with the same key restores the session from the cookie.
    let second = login_with(&p, &[("PAPERLESS_SESSION_KEY", KEY_HEX)]);
    assert!(
        second.subject(&cookie_value).is_none(),
        "unknown before restore"
    );
    second.restore(&cookie_value).await;
    assert_eq!(second.subject(&cookie_value).as_deref(), Some("oidc:u1"));

    // Without the key (random per process), a keyless process's cookie opens in
    // no other keyless process. Issued keyless, so a fixed fallback key would fail.
    let keyless_a = login_with(&p, &[]);
    let (keyless_cookie, _) = sign_in(&keyless_a, &p, "u2", user_role()).await.unwrap();
    assert_eq!(
        keyless_a.subject(&keyless_cookie).as_deref(),
        Some("oidc:u2")
    );
    let keyless_b = login_with(&p, &[]);
    keyless_b.restore(&keyless_cookie).await;
    assert!(keyless_b.subject(&keyless_cookie).is_none());
    // Nor does the keyed cookie open in a keyless process.
    keyless_b.restore(&cookie_value).await;
    assert!(keyless_b.subject(&cookie_value).is_none());

    // A tampered cookie never opens.
    let mut tampered = cookie_value.clone();
    tampered.push('A');
    second.restore(&tampered).await;
    assert!(second.subject(&tampered).is_none());
}

#[tokio::test]
async fn logout_ends_the_session_in_this_process() {
    let p = Provider::new();
    let login = login_with(&p, &[("PAPERLESS_SESSION_KEY", KEY_HEX)]);
    let (cookie_value, _) = sign_in(&login, &p, "u1", user_role()).await.unwrap();
    assert!(login.subject(&cookie_value).is_some());
    login.logout(&cookie_value);
    assert!(login.subject(&cookie_value).is_none());
    login.restore(&cookie_value).await;
    assert!(
        login.subject(&cookie_value).is_none(),
        "a revoked cookie is not restored"
    );
}

// ---------------------------------------------------------------- the gate

async fn send(app: &Router, method: &str, uri: &str, cookie_value: Option<&str>) -> Response {
    let mut req = HttpRequest::builder().method(method).uri(uri);
    if let Some(c) = cookie_value {
        req = req.header(header::COOKIE, format!("{SESSION_COOKIE}={c}"));
    }
    app.clone()
        .oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

fn secret_app() -> Router {
    Router::new()
        .route("/", get(|| async { "secret" }))
        .route("/documents", get(|| async { "secret" }))
        .route(
            "/documents/:hash/original",
            get(|| async { "original bytes" }),
        )
        .route("/upload", post(|| async { "uploaded" }))
        .route("/healthz", get(|| async { "ok" }))
}

#[tokio::test]
async fn the_gate_admits_only_a_live_session() {
    let p = Provider::new();
    let login = Arc::new(login_with(&p, &[]));
    let app = protect(secret_app(), Some(login.clone()));

    // No session: GET redirects to sign-in with the page to return to, POST is 401.
    let r = send(&app, "GET", "/documents?q=x", None).await;
    assert_eq!(r.status(), StatusCode::SEE_OTHER);
    assert_eq!(
        r.headers()[header::LOCATION],
        "/auth/login?return_to=%2Fdocuments%3Fq%3Dx"
    );
    assert_eq!(
        send(&app, "POST", "/upload", None).await.status(),
        StatusCode::UNAUTHORIZED
    );
    // An original is a page like any other: no session, no bytes.
    let r = send(
        &app,
        "GET",
        &format!("/documents/{}/original", "a".repeat(64)),
        None,
    )
    .await;
    assert_eq!(r.status(), StatusCode::SEE_OTHER);
    let body = axum::body::to_bytes(r.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!body.windows(14).any(|w| w == b"original bytes"));
    // A garbage cookie is no session.
    assert_eq!(
        send(&app, "GET", "/", Some("v1.junk")).await.status(),
        StatusCode::SEE_OTHER
    );
    // Health answers without a session.
    assert_eq!(
        send(&app, "GET", "/healthz", None).await.status(),
        StatusCode::OK
    );

    // With a session the pages answer.
    let (cookie_value, _) = sign_in(&login, &p, "u1", user_role()).await.unwrap();
    assert_eq!(
        send(&app, "GET", "/", Some(&cookie_value)).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, "POST", "/upload", Some(&cookie_value))
            .await
            .status(),
        StatusCode::OK
    );

    // Signing out through the route clears the cookie and closes the pages again.
    let r = send(&app, "POST", "/auth/logout", Some(&cookie_value)).await;
    assert_eq!(r.status(), StatusCode::SEE_OTHER);
    assert!(r.headers()[header::SET_COOKIE]
        .to_str()
        .unwrap()
        .contains("Max-Age=0"));
    assert_eq!(
        send(&app, "GET", "/", Some(&cookie_value)).await.status(),
        StatusCode::SEE_OTHER
    );
}

#[tokio::test]
async fn the_login_route_sets_a_host_cookie_and_redirects_to_the_provider() {
    let p = Provider::new();
    let app = protect(secret_app(), Some(Arc::new(login_with(&p, &[]))));
    let r = send(&app, "GET", "/auth/login?return_to=%2Fdocuments", None).await;
    assert_eq!(r.status(), StatusCode::SEE_OTHER);
    assert!(r.headers()[header::LOCATION]
        .to_str()
        .unwrap()
        .starts_with("https://idp.example/oauth/v2/authorize?"));
    let set = r.headers()[header::SET_COOKIE].to_str().unwrap();
    assert!(set.starts_with(LOGIN_COOKIE));
    for attr in ["Secure", "HttpOnly", "SameSite=Lax", "Path=/"] {
        assert!(set.contains(attr), "{attr} in {set}");
    }
}

#[tokio::test]
async fn without_a_login_the_app_is_unchanged() {
    let app = protect(secret_app(), None);
    assert_eq!(send(&app, "GET", "/", None).await.status(), StatusCode::OK);
    assert_eq!(
        send(&app, "GET", "/auth/login", None).await.status(),
        StatusCode::NOT_FOUND
    );
}

// ---------------------------------------------------------------- bearer (delegation)

#[test]
fn api_clients_are_a_comma_list_of_plain_ids() {
    let c = config(&[(
        "PAPERLESS_API_CLIENTS",
        "report-client, other-client,report-client",
    )]);
    assert_eq!(c.api_clients, vec!["report-client", "other-client"]);
    assert!(c.summary().contains("2 API client(s)"));
    for bad in ["a,,b", "a b", ","] {
        let v = base_vars(&[]);
        let mut v: Vec<(&str, &str)> = v;
        v.push(("PAPERLESS_API_CLIENTS", bad));
        assert!(
            AuthConfig::from_vars(vars(&v))
                .unwrap_err()
                .contains("API_CLIENTS"),
            "{bad:?}"
        );
    }
    assert!(config(&[]).api_clients.is_empty());
}

async fn send_bearer(app: &Router, method: &str, uri: &str, token: &str) -> Response {
    let req = HttpRequest::builder()
        .method(method)
        .uri(uri)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    app.clone().oneshot(req).await.unwrap()
}

#[tokio::test]
async fn bearer_tokens_of_api_clients_pass_the_gate_and_others_do_not() {
    let p = Provider::new();
    let login = Arc::new(login_with(
        &p,
        &[("PAPERLESS_API_CLIENTS", "report-client")],
    ));
    let app = protect(secret_app(), Some(login.clone()));

    // A token report-rs holds for the user (addressed to its client), with an allowed role.
    let delegated = p.sign_for("report-client", None, "u1", user_role());
    assert_eq!(
        send_bearer(&app, "GET", "/documents", &delegated)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        send_bearer(&app, "POST", "/upload", &delegated)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        login.verify_bearer(&delegated).await.as_deref(),
        Ok("oidc:u1")
    );

    // Addressed to a client that is not listed (here: paperless's own): refused, no redirect.
    let own = p.sign_for(CLIENT, None, "u1", user_role());
    assert_eq!(
        send_bearer(&app, "GET", "/documents", &own).await.status(),
        StatusCode::UNAUTHORIZED
    );
    // The right client, but no allowed role: refused.
    let no_role = p.sign_for(
        "report-client",
        None,
        "u2",
        serde_json::json!({"report-admin": {"100": "a"}}),
    );
    assert_eq!(
        send_bearer(&app, "GET", "/documents", &no_role)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    // Not a bearer token at all.
    assert_eq!(
        send_bearer(&app, "GET", "/documents", "not a token")
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    // Without PAPERLESS_API_CLIENTS no bearer token is accepted, even a valid one.
    let closed = protect(secret_app(), Some(Arc::new(login_with(&p, &[]))));
    assert_eq!(
        send_bearer(&closed, "GET", "/documents", &delegated)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn a_session_cookie_from_another_origin_cannot_change_state() {
    let p = Provider::new();
    let login = Arc::new(login_with(&p, &[]));
    let app = protect(secret_app(), Some(login.clone()));
    let (cookie_value, _) = sign_in(&login, &p, "u1", user_role()).await.unwrap();
    let post = |extra: &'static [(&'static str, &'static str)]| {
        let app = app.clone();
        let cookie_value = cookie_value.clone();
        async move {
            let mut b = HttpRequest::builder()
                .method("POST")
                .uri("/upload")
                .header(header::COOKIE, format!("{SESSION_COOKIE}={cookie_value}"));
            for (k, v) in extra {
                b = b.header(*k, *v);
            }
            app.oneshot(b.body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status()
        }
    };
    // A sibling subdomain is same-site, so the cookie travels; the origin decides.
    assert_eq!(
        post(&[("sec-fetch-site", "same-site")]).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        post(&[("origin", "https://evil.paperless.example")]).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        post(&[("sec-fetch-site", "cross-site")]).await,
        StatusCode::FORBIDDEN
    );
    // The archive's own pages pass.
    assert_eq!(
        post(&[("sec-fetch-site", "same-origin")]).await,
        StatusCode::OK
    );
    assert_eq!(
        post(&[("origin", "https://paperless.example")]).await,
        StatusCode::OK
    );
    // Sign-out sits outside the gate, so it checks the origin itself: a sibling
    // origin cannot sign the user out.
    let r = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/auth/logout")
                .header(header::COOKIE, format!("{SESSION_COOKIE}={cookie_value}"))
                .header("origin", "https://evil.paperless.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::FORBIDDEN);
    assert_eq!(login.subject(&cookie_value).as_deref(), Some("oidc:u1"));
    // GET is never refused for its origin.
    let r = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .uri("/")
                .header(header::COOKIE, format!("{SESSION_COOKIE}={cookie_value}"))
                .header("sec-fetch-site", "cross-site")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
}

#[test]
fn require_auth_refuses_an_open_start_and_changes_nothing_when_off() {
    let flag = |v: &'static str| sign_in_required(vars(&[("PAPERLESS_REQUIRE_AUTH", v)]));
    // Off: unset, blank, 0, false. On: 1, true. Anything else is an error.
    assert!(!sign_in_required(vars(&[])).unwrap());
    for off in ["", " ", "0", "false"] {
        assert!(!flag(off).unwrap(), "{off:?}");
    }
    for on in ["1", "true", " 1 "] {
        assert!(flag(on).unwrap(), "{on:?}");
    }
    for bad in ["yes", "TRUE", "2", "on"] {
        assert!(
            flag(bad).unwrap_err().contains("PAPERLESS_REQUIRE_AUTH"),
            "{bad:?}"
        );
    }
    // Required without sign-in refuses; every other combination starts.
    assert!(check_open_mode(true, false).is_err());
    assert!(check_open_mode(true, true).is_ok());
    assert!(check_open_mode(false, false).is_ok());
    assert!(check_open_mode(false, true).is_ok());
}
