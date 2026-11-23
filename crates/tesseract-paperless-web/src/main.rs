//! `tesseract-paperless-web` — a paperless-ngx-shaped document archive: upload
//! a scan or photo, run the pure-Rust OCR, dedup + store it via `lancedb`,
//! search and browse the archive.
//!
//! Deploy target: Railway. Railway injects `PORT` and expects the process to
//! bind `0.0.0.0:$PORT` — read at runtime only, never hardcoded.

mod api;
mod auth;
mod branding;
mod customers;
mod decode;
mod fetch;
mod headers;
mod ingest;
mod replica;
mod routes;
mod s3;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use crate::state::AppState;

#[tokio::main]
async fn main() {
    // Sign-in configuration is read and checked before anything else runs, so
    // a refused start prints only its fatal line: PAPERLESS_REQUIRE_AUTH with no
    // sign-in stops here, before the model loads or the port is bound.
    let required = match auth::sign_in_required_from_env() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("fatal: {e}");
            std::process::exit(1);
        }
    };
    let auth_cfg = match auth::AuthConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fatal: sign-in configuration: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = auth::check_open_mode(required, auth_cfg.is_some()) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
    // The brand is read strictly, after the sign-in preflight (whose fatal line
    // comes first) and before the model loads or the router exists, so no page
    // renders with another brand. A brand document that is set but not valid
    // Unicode is refused, not read as unset.
    for key in ["BRAND_TOML", "BRAND_FILE"] {
        if let Err(std::env::VarError::NotUnicode(_)) = std::env::var(key) {
            eprintln!("fatal: branding: {key} is not valid Unicode");
            std::process::exit(1);
        }
    }
    match branding::Branding::from_sources(
        |k| std::env::var(k).ok(),
        |p| std::fs::read_to_string(std::path::Path::new(p)),
        |w| eprintln!("tesseract-paperless-web: {w}"),
    ) {
        Ok(b) => branding::install(b),
        Err(e) => {
            eprintln!("fatal: branding: {e}");
            std::process::exit(1);
        }
    }
    let policy = match customers::CustomerPolicy::from_env() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fatal: {e}");
            std::process::exit(1);
        }
    };
    // The S3 copy is read just as strictly: a partial configuration stops the
    // start instead of silently running without a copy.
    let s3_cfg = match s3::S3Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fatal: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = customers::check_policy_needs_sign_in(policy.is_some(), auth_cfg.is_some()) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
    let security_headers = match headers::SecurityHeaders::from_env() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("fatal: {e}");
            std::process::exit(1);
        }
    };
    match &policy {
        Some(p) => println!("tesseract-paperless-web: {}", p.summary()),
        None => println!(
            "tesseract-paperless-web: no customer policy (PAPERLESS_CUSTOMER_POLICY unset): \
             every admitted user sees every document"
        ),
    }

    let model_dir =
        PathBuf::from(std::env::var("MODEL_DIR").unwrap_or_else(|_| "corpus/model".to_string()));
    // Local filesystem path (Railway: a mounted volume) or any URI
    // `lancedb::connect` accepts. Falls back to a relative directory so
    // `cargo run` from the repo root works with zero configuration.
    let archive_uri =
        std::env::var("ARCHIVE_URI").unwrap_or_else(|_| "./archive.lance".to_string());
    // Full-text search index (`tesseract_paperless::search::SearchIndex`) --
    // a SEPARATE on-disk directory from the lancedb archive above, per
    // `search.rs`'s module doc. Falls back the same way `ARCHIVE_URI` does.
    let search_index_dir = PathBuf::from(
        std::env::var("SEARCH_INDEX_DIR").unwrap_or_else(|_| "./search_index".to_string()),
    );
    // The uploaded originals, one file per SHA-256. Railway keeps it on the
    // same volume as the archive (`/app/data/blobs`); falls back like the two
    // directories above.
    let blob_dir =
        PathBuf::from(std::env::var("BLOB_DIR").unwrap_or_else(|_| "./blobs".to_string()));
    // deepnsm's `word_frequency/` CSVs — the SPO/`NarsTruth` reasoning
    // layer's vocabulary. Same env-var-then-sibling-path convention as
    // `tesseract-ogar/examples/ocr_demo.rs`'s own step 6: the sibling path
    // only resolves at BUILD time (inside a Docker builder stage's /src
    // layout), so a runtime image sets `DEEPNSM_VOCAB_DIR` to wherever it
    // copied the CSVs instead.
    let deepnsm_vocab_dir =
        PathBuf::from(std::env::var("DEEPNSM_VOCAB_DIR").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../lance-graph/crates/deepnsm/word_frequency"
            )
            .to_string()
        }));

    println!(
        "tesseract-paperless-web: loading model from {}",
        model_dir.display()
    );
    println!("tesseract-paperless-web: archive at {archive_uri}");
    println!(
        "tesseract-paperless-web: search index at {}",
        search_index_dir.display()
    );
    println!(
        "tesseract-paperless-web: originals at {}",
        blob_dir.display()
    );
    println!(
        "tesseract-paperless-web: deepnsm vocabulary at {}",
        deepnsm_vocab_dir.display()
    );

    let replica = match s3_cfg {
        Some(cfg) => {
            println!(
                "tesseract-paperless-web: S3 copy: on ({} bucket {}, prefix {})",
                cfg.endpoint, cfg.bucket, cfg.prefix
            );
            let prefix = cfg.prefix.clone();
            // reqwest's blocking client must not be built on an async worker
            // thread: building it there panics.
            match tokio::task::spawn_blocking(move || s3::S3Client::new(cfg))
                .await
                .map_err(|e| s3::ObjectError::new(&format!("task: {e}")))
                .and_then(|r| r)
            {
                Ok(client) => Some(replica::Replica::new(Arc::new(client), &prefix)),
                Err(e) => {
                    eprintln!("fatal: S3 client: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => {
            println!("tesseract-paperless-web: S3 copy: off (no PAPERLESS_S3_* configured)");
            None
        }
    };

    let state = match AppState::load(
        &model_dir,
        &archive_uri,
        &search_index_dir,
        &blob_dir,
        &deepnsm_vocab_dir,
        replica,
    )
    .await
    {
        Ok(mut s) => {
            s.policy = policy;
            Arc::new(s)
        }
        Err(e) => {
            eprintln!("fatal: could not start: {e}");
            if e.contains("S3") {
                std::process::exit(1);
            }
            eprintln!(
                "hint: set MODEL_DIR to a directory containing eng.lstm, \
                 eng.lstm-unicharset, eng.lstm-recoder (+ optional *-dawg files), \
                 ARCHIVE_URI to a writable lancedb location, \
                 SEARCH_INDEX_DIR and BLOB_DIR to writable directories"
            );
            std::process::exit(1);
        }
    };

    // Mine the AUTO model off the boot path (spec R7): the app serves at once
    // and reports "no suggestions yet" until the first mine lands.
    tokio::spawn(state.clone().remine());

    // Replication (`docs/RECOVERY.md` §5): once at start, then every 15
    // minutes. It only ever adds to S3.
    if state.replica.is_some() {
        let st = state.clone();
        tokio::spawn(async move {
            let mut every = tokio::time::interval(std::time::Duration::from_secs(15 * 60));
            loop {
                every.tick().await;
                st.replicate_now().await;
            }
        });
    }

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{port}");

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fatal: could not bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    println!("tesseract-paperless-web: listening on http://{addr}");

    // Sign-in (OIDC) when PAPERLESS_OIDC_ISSUER is set. A set but invalid
    // configuration stops the boot: an archive that meant to require sign-in
    // must never come up open.
    let login = match auth_cfg {
        Some(cfg) => {
            println!("tesseract-paperless-web: {}", cfg.summary());
            let http = match auth::ReqwestHttp::new() {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("fatal: {e}");
                    std::process::exit(1);
                }
            };
            match auth::Login::new(cfg, Box::new(http)) {
                Ok(l) => Some(Arc::new(l)),
                Err(e) => {
                    eprintln!("fatal: sign-in: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => {
            eprintln!(
                "warning: PAPERLESS_OIDC_ISSUER unset: NO sign-in, anyone who reaches \
                 this service can read and delete documents. Keep it off public networks."
            );
            None
        }
    };
    println!("tesseract-paperless-web: {}", security_headers.summary());
    let app = auth::protect(routes::router(state), login).layer(
        axum::middleware::from_fn_with_state(security_headers, headers::add),
    );
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("fatal: server error: {e}");
        std::process::exit(1);
    }
}
