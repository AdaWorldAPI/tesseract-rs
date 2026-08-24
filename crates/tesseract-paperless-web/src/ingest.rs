//! The ingestion pipeline: raw bytes -> S-2 gate -> OCR -> `DocIr` -> archive.
//!
//! ```text
//!   S-2 DECIDES BEFORE RECOGNITION SPENDS ANYTHING.
//! ```
//!
//! Mirrors `tesseract_paperless::intake::ingest_doc_ir`'s shape (gate first,
//! producer only runs on `Verdict::Novel`) with the producer instantiated as
//! THIS binary's own OCR call. `ingest_doc_ir` takes a synchronous closure;
//! this crate's producer is inherently async (recognition is dispatched via
//! `spawn_blocking`, the store write is async I/O), so the pipeline is
//! written out here rather than forced through that generic entry point. The
//! invariant it protects — the gate decides before the producer runs, and a
//! duplicate's bytes never reach the producer at all — holds identically:
//! see [`ingest`]'s early return on [`Verdict::Duplicate`].

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ogar_doc_ir::DocIr;
use serde_json::Value;
use tesseract_ogar::{BinarizeMode, OcrExecError, OcrRequest, OcrResponse};
use tesseract_paperless::kv::{mint_document_root, preflight, MatchedOn, Verdict};
use tesseract_paperless::store::StoreError;

use crate::decode::decode_grey;
use crate::state::AppState;

/// What ingesting one upload produced — the S-2 gate's two outcomes, each
/// carrying what a caller needs to redirect to the document's detail page.
#[derive(Debug, Clone)]
pub enum IngestOutcome {
    /// Bytes not seen before: recognized, converted, and stored.
    Stored {
        /// Hex `content_sha256` — the document's primary key.
        hash_hex: String,
        /// The minted root address.
        document_guid: [u8; 16],
        /// Pages in the recognized document.
        page_count: usize,
        /// Mean word confidence, 0..=100.
        mean_confidence: u32,
        /// Whether the recognizer itself was not confident.
        low_confidence: bool,
    },
    /// Bytes already held — recognition was skipped entirely. The OCR pass
    /// this branch never runs is exactly the spend S-2 exists to save.
    Duplicate {
        /// Hex `content_sha256` of the incoming (and the matched) bytes.
        hash_hex: String,
        /// The already-stored document's root address.
        document_guid: [u8; 16],
        /// Which stored hash matched.
        matched: MatchedOn,
    },
}

impl IngestOutcome {
    /// The hex hash either branch carries — the routing key for a redirect
    /// to `/documents/{hash_hex}`.
    #[must_use]
    pub fn hash_hex(&self) -> &str {
        match self {
            Self::Stored { hash_hex, .. } | Self::Duplicate { hash_hex, .. } => hash_hex,
        }
    }
}

/// Why ingestion failed.
#[derive(Debug)]
pub enum IngestError {
    /// The uploaded bytes did not decode to an image.
    Decode(String),
    /// The `spawn_blocking` recognition task itself panicked or the server
    /// is shutting down (permit acquisition failed).
    Task(String),
    /// The recognizer failed.
    Recognize(OcrExecError),
    /// `doc.v1` -> `DocIr` conversion failed.
    Convert(ogar_from_docv1::FromDocV1Error),
    /// The archive write failed.
    Store(StoreError),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "decode: {e}"),
            Self::Task(e) => write!(f, "recognition task: {e}"),
            Self::Recognize(e) => write!(f, "recognition: {e:?}"),
            Self::Convert(e) => write!(f, "doc.v1 conversion: {e:?}"),
            Self::Store(e) => write!(f, "archive write: {e}"),
        }
    }
}

/// Run the S-2-gated ingest pipeline over one upload's raw bytes.
///
/// `mime` is the caller's best-effort guess at the source media type
/// (informational, carried into [`DocIr::mime`] — the decoder itself sniffs
/// the real image format independently and does not trust this value).
pub async fn ingest(
    state: &Arc<AppState>,
    bytes: Vec<u8>,
    filename: Option<String>,
    mime: &str,
) -> Result<IngestOutcome, IngestError> {
    let (hash, verdict) = preflight(&bytes, &state.store);
    let hex = format!("{hash:?}");

    if let Verdict::Duplicate { matched } = verdict {
        // Deterministic from the hash alone — no store round-trip needed to
        // learn the address a Stored ingest of these exact bytes already
        // minted (`store::LanceStore::put` computes the identical value).
        let guid = mint_document_root(&hash).root.0.to_bytes();
        return Ok(IngestOutcome::Duplicate {
            hash_hex: hex,
            document_guid: guid,
            matched,
        });
    }

    // S-2 cleared: pay for recognition now, off the async runtime, bounded by
    // the shared permit pool so a burst of uploads can't oversubscribe CPU.
    let permit = state
        .recognize_permits
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| IngestError::Task("server is shutting down".to_string()))?;
    let st = state.clone();
    let doc_json = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let (grey, w, h) = decode_grey(&bytes).map_err(IngestError::Decode)?;
        match st.executor.execute(OcrRequest::RecognizeDocument {
            grey: &grey,
            width: w,
            height: h,
            with_dict: true,
            harvest_profile: None,
            binarize: BinarizeMode::default(),
        }) {
            Ok(OcrResponse::DocumentOut { doc_json, .. }) => Ok(doc_json),
            Ok(_) => unreachable!("RecognizeDocument always returns DocumentOut"),
            Err(e) => Err(IngestError::Recognize(e)),
        }
    })
    .await
    .map_err(|e| IngestError::Task(e.to_string()))??;

    let (mean_confidence, low_confidence) = quality_from_doc_json(&doc_json);
    let ir: DocIr = ogar_from_docv1::from_doc_v1(&doc_json, *hash.as_bytes(), mime)
        .map_err(IngestError::Convert)?;
    let page_count = ir.pages.len();

    let ingested_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0);

    let guid = state
        .store
        .put(
            &hash,
            filename.as_deref(),
            mean_confidence,
            low_confidence,
            &ir,
            ingested_at_unix_ms,
        )
        .await
        .map_err(IngestError::Store)?;

    Ok(IngestOutcome::Stored {
        hash_hex: hex,
        document_guid: guid.0.to_bytes(),
        page_count,
        mean_confidence,
        low_confidence,
    })
}

/// Pull `pages[0].quality.{mean_conf,low_confidence}` out of the RAW
/// `doc.v1` JSON.
///
/// `ogar_from_docv1::from_doc_v1`'s conversion into [`DocIr`] drops the
/// quality object entirely — the IR carries no confidence field at all — so
/// this crate's own archive columns for it can only come from the source
/// JSON, read directly rather than through the typed IR. `recognize_document`
/// always emits exactly one page (`tesseract_ocr::structured::render_doc`
/// hardcodes `"page":1` and never loops), so `pages[0]` is safe; a
/// missing/malformed quality object degrades to `(0, false)` rather than
/// failing the whole ingest over a display-only signal.
fn quality_from_doc_json(doc_json: &str) -> (u32, bool) {
    let v: Value = match serde_json::from_str(doc_json) {
        Ok(v) => v,
        Err(_) => return (0, false),
    };
    let quality = &v["pages"][0]["quality"];
    let mean = quality["mean_conf"]
        .as_f64()
        .unwrap_or(0.0)
        .round()
        .clamp(0.0, 100.0) as u32;
    let low = quality["low_confidence"].as_bool().unwrap_or(false);
    (mean, low)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The happy path: both fields present and well-typed.
    #[test]
    fn quality_reads_mean_conf_and_low_confidence() {
        let json = r#"{"pages":[{"quality":{"mean_conf":87.3,"low_confidence":false}}]}"#;
        assert_eq!(quality_from_doc_json(json), (87, false));
    }

    /// `mean_conf` is `null` on a page with no recognized words — the real
    /// shape `structured::render_doc` emits, not a hypothetical one.
    #[test]
    fn quality_defaults_mean_conf_when_null() {
        let json = r#"{"pages":[{"quality":{"mean_conf":null,"low_confidence":false}}]}"#;
        assert_eq!(quality_from_doc_json(json), (0, false));
    }

    /// `low_confidence: true` must survive the extraction, not just the
    /// numeric field — a one-sided test on `mean_conf` alone could pass with
    /// this field hardcoded to `false`.
    #[test]
    fn quality_reads_low_confidence_true() {
        let json = r#"{"pages":[{"quality":{"mean_conf":12.0,"low_confidence":true}}]}"#;
        assert_eq!(quality_from_doc_json(json), (12, true));
    }

    /// Malformed/absent JSON must degrade, never panic — this signal is
    /// display-only and must not be able to fail the whole ingest.
    #[test]
    fn quality_defaults_on_malformed_json() {
        assert_eq!(quality_from_doc_json("not json"), (0, false));
        assert_eq!(quality_from_doc_json("{}"), (0, false));
    }
}
