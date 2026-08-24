//! `tesseract-paperless` — document intake for the pure-Rust OCR stack.
//!
//! ```text
//!   HASH BEFORE YOU SPEND.  MANY RETINAS, ONE SHAPE.
//!   TOKENIZE ONCE.  PROJECT MANY TIMES.
//! ```
//!
//! # What this crate is
//!
//! The stage between "some bytes arrived" and "a document exists": compute the
//! convergence hash, ask whether we have seen these bytes before, and — only
//! if not — let a producer turn them into one [`ogar_doc_ir::DocIr`].
//!
//! | module | feature | what it owns |
//! |---|---|---|
//! | [`kv`] | — | the S-2 dedup gate and the document subtree's keys |
//! | [`intake`] | — (`ocr` adds the in-process recognizer) | the gate in front of every producer |
//! | [`token`] | `token` | ONE versioned BPE tokenization per span, borrowed by several consumers |
//!
//! # What this crate deliberately is NOT
//!
//! **It holds no store.** [`kv::DedupIndex`] is a trait and nothing here
//! implements it. That is not an omission to be filled in later by this crate;
//! it is the boundary — `OGAR-DOC-W4-BUILD-SPEC` puts the KV blob on the
//! consumer, and recognition in this workspace stays storage-less. What ships
//! here is the *gate*: a hash, a lookup contract, and an ordering rule.
//!
//! **It recognizes nothing.** Under `ocr` it calls
//! `tesseract_ogar::OcrExecutor`, the one sanctioned entry point, and never
//! `LstmRecognizer` / `structured` / the renderers directly.
//!
//! **It mints no identity.** Documents are keyed by `DocIr::content_sha256`
//! and spans by `(page, reading_order)` — all read from the document layer's
//! own IR rather than invented here.
//!
//! # Status
//!
//! [`kv`] and [`intake`] are small and tested. [`token`] is a **probe** and
//! says so in its own docs: it measured the seam and named the gaps between it
//! and a production carrier. Nothing under `token` is a shipping carrier.

#![forbid(unsafe_code)]

pub mod intake;
pub mod kv;
pub mod render;

#[cfg(feature = "search")]
pub mod search;

#[cfg(feature = "store")]
pub mod store;

#[cfg(feature = "token")]
pub mod token;
