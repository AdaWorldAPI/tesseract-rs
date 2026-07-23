//! `tesseract-ocr-web` — a single-binary web demo for the pure-Rust OCR.
//!
//! Upload an image (or paste an image URL), the server runs the pure-Rust
//! recognizer (no libtesseract / leptonica / OpenCV at runtime) and returns the
//! text plus stats, with a one-click download of the result.
//!
//! Deploy target: Railway. Railway injects the `PORT` env var and expects the
//! process to bind `0.0.0.0:$PORT` — so we read `PORT` at runtime and only fall
//! back to 8080 for local `cargo run`. Do NOT hardcode the port.

mod api;
mod fetch;
mod ocr;
mod routes;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use crate::state::AppState;

#[tokio::main]
async fn main() {
    // Model directory: `MODEL_DIR` env override, else the bundled `corpus/model`
    // (relative to the CWD — the Docker image copies the model next to the binary
    // and sets MODEL_DIR to its absolute path).
    let model_dir =
        PathBuf::from(std::env::var("MODEL_DIR").unwrap_or_else(|_| "corpus/model".to_string()));

    // Informational startup lines go to STDOUT so log platforms (Railway) don't
    // tag them `severity: error` — stderr is reserved for actual `fatal:` failures.
    println!(
        "tesseract-ocr-web: loading model from {}",
        model_dir.display()
    );
    let state = match AppState::load(&model_dir) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("fatal: could not load OCR model: {e}");
            eprintln!(
                "hint: set MODEL_DIR to a directory containing eng.lstm, \
                 eng.lstm-unicharset, eng.lstm-recoder (+ optional *-dawg files)"
            );
            std::process::exit(1);
        }
    };
    println!(
        "tesseract-ocr-web: eng model loaded (dict beam: {}); deu model {}",
        if state.eng.dict.is_some() {
            "on"
        } else {
            "off"
        },
        match &state.deu {
            Some(m) => {
                if m.dict.is_some() {
                    "loaded (dict beam: on)"
                } else {
                    "loaded (dict beam: off)"
                }
            }
            None => "not found — lang=deu will fall back to eng",
        }
    );

    // Railway injects $PORT and replaces the variable itself; 8080 is ONLY the
    // local-dev fallback. Bind 0.0.0.0 so the container is reachable.
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{port}");

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fatal: could not bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    println!("tesseract-ocr-web: listening on http://{addr}");

    let app = routes::router(state);
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("fatal: server error: {e}");
        std::process::exit(1);
    }
}
