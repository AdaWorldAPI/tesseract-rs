//! `tesseract-paperless-web` — a paperless-ngx-shaped document archive: upload
//! a scan or photo, run the pure-Rust OCR, dedup + store it via `lancedb`,
//! search and browse the archive.
//!
//! Deploy target: Railway. Railway injects `PORT` and expects the process to
//! bind `0.0.0.0:$PORT` — read at runtime only, never hardcoded.

mod decode;
mod fetch;
mod ingest;
mod routes;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use crate::state::AppState;

#[tokio::main]
async fn main() {
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
        "tesseract-paperless-web: deepnsm vocabulary at {}",
        deepnsm_vocab_dir.display()
    );

    let state = match AppState::load(
        &model_dir,
        &archive_uri,
        &search_index_dir,
        &deepnsm_vocab_dir,
    )
    .await
    {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("fatal: could not start: {e}");
            eprintln!(
                "hint: set MODEL_DIR to a directory containing eng.lstm, \
                 eng.lstm-unicharset, eng.lstm-recoder (+ optional *-dawg files), \
                 ARCHIVE_URI to a writable lancedb location, \
                 and SEARCH_INDEX_DIR to a writable directory"
            );
            std::process::exit(1);
        }
    };

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

    let app = routes::router(state);
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("fatal: server error: {e}");
        std::process::exit(1);
    }
}
