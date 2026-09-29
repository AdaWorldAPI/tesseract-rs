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
//! | `store` / `archive_meta` | `store` | the archive, and its definitions, assignments and review flags |
//! | `matching` | `matching` | S-8: paperless-ngx's matching rules |
//! | `auto_match` / `auto_rows` | `auto-match` | the AUTO tier's miner and its input rows |
//! | `auto_model` | `store` + `search` + `matching` + `auto-match` | the mined AUTO model and S-8 at ingest |
//!
//! # What this crate deliberately is NOT
//!
//! **Without the `store` feature it holds no store.** In the default build
//! [`kv::DedupIndex`] is a trait and nothing implements it, so recognition in
//! this workspace stays storage-less (`OGAR-DOC-W4-BUILD-SPEC` puts the KV blob
//! on the consumer). What ships by default is the *gate*: a hash, a lookup
//! contract, and an ordering rule. The consumer's archive is the opt-in
//! `store` feature (`store::LanceStore` implements the trait, and
//! `archive_meta` holds its filing metadata); nothing in the default build or
//! in any recognition crate depends on it.
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

/// S-8: the matching rule a tag, correspondent or document type carries,
/// transcribed from paperless-ngx's `matching.py`.
#[cfg(feature = "matching")]
pub mod matching;

/// paperless-ngx's AUTO matching tier as association rules mined over the
/// archive: suggestions with a NARS truth, never applied on their own.
#[cfg(feature = "auto-match")]
pub mod auto_match;

/// Archive metadata to mining input: dense ids, eligibility, vocabulary.
#[cfg(feature = "auto-match")]
pub mod auto_rows;

/// The AUTO model and S-8 at ingest: archive metadata in, suggestions out.
#[cfg(all(
    feature = "store",
    feature = "search",
    feature = "matching",
    feature = "auto-match"
))]
pub mod auto_model;

#[cfg(feature = "report")]
pub mod axes;

#[cfg(feature = "store")]
pub mod store;

/// Definitions, assignments and the review flag the AUTO tier learns from.
#[cfg(feature = "store")]
pub mod archive_meta;

/// Keeps the search index in line with the archive it is a lens over.
#[cfg(all(feature = "store", feature = "search"))]
pub mod reconcile;

#[cfg(feature = "token")]
pub mod token;

/// A document's real path into lance-graph, and back: v1's tagger feeding
/// v2's real FSM + trained CAM-PQ 96 space, OCR confidence as the muscle-
/// memory anchor, grammar/semantic-consistency-gated correction, byte-exact
/// provenance throughout. Needs both `ocr` (the recognized `DocPage`
/// surface) and `token` (deepnsm-v2's FSM/vocab/space) — see the module's
/// own docs for the full design and its honestly-stated limits.
#[cfg(all(feature = "ocr", feature = "token"))]
pub mod consistency;
