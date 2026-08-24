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
use tesseract_ogar::reasoning::SentenceBelief;
use tesseract_ogar::sentences::assemble_sentences;
use tesseract_ogar::{BinarizeMode, OcrExecError, OcrRequest, OcrResponse};
use tesseract_paperless::kv::{mint_document_root, preflight, MatchedOn, Verdict};
use tesseract_paperless::search::SearchError;
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
        /// How many SPO triples the reasoning layer extracted across every
        /// sentence on the page. `0` either because extraction never ran
        /// (`spo_extraction_ran == false`, no deepnsm vocabulary loaded) or
        /// because it ran and genuinely found nothing — `spo_extraction_ran`
        /// is what tells the two apart.
        triple_count: usize,
        /// Whether the reasoning layer ran at all — a loaded `AppState`
        /// reasoner, independent of whether any triples were actually
        /// found (see `triple_count`).
        spo_extraction_ran: bool,
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
    /// The document was archived, but the search index write failed --
    /// stored yet unsearchable. Reported rather than swallowed (see
    /// [`ingest`]'s doc comment on why this is a real, if narrow, gap).
    Search(SearchError),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "decode: {e}"),
            Self::Task(e) => write!(f, "recognition task: {e}"),
            Self::Recognize(e) => write!(f, "recognition: {e:?}"),
            Self::Convert(e) => write!(f, "doc.v1 conversion: {e:?}"),
            Self::Store(e) => write!(f, "archive write: {e}"),
            Self::Search(e) => write!(f, "search index write: {e}"),
        }
    }
}

/// Run the S-2-gated ingest pipeline over one upload's raw bytes.
///
/// `mime` is the caller's best-effort guess at the source media type
/// (informational, carried into [`DocIr::mime`] — the decoder itself sniffs
/// the real image format independently and does not trust this value).
///
/// # A named gap: the archive write and the search-index write are not one
/// transaction
///
/// `store.put` and `search.index_document` are two separate stores with no
/// shared commit. If the process dies between them, the document is archived
/// (findable by direct link, by `LanceStore::list`) but not yet in the search
/// index -- an inconsistency, not data loss. `Err(IngestError::Search(_))`
/// surfaces this rather than swallowing it; recovering from it (a periodic
/// reconciliation pass diffing the archive against the index) is future
/// work, filed rather than pretended away.
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
    let (doc_json, spo_json, triple_count) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let (grey, w, h) = decode_grey(&bytes).map_err(IngestError::Decode)?;
        let doc_json = match st.executor.execute(OcrRequest::RecognizeDocument {
            grey: &grey,
            width: w,
            height: h,
            with_dict: true,
            harvest_profile: None,
            binarize: BinarizeMode::default(),
        }) {
            Ok(OcrResponse::DocumentOut { doc_json, .. }) => doc_json,
            Ok(_) => unreachable!("RecognizeDocument always returns DocumentOut"),
            Err(e) => return Err(IngestError::Recognize(e)),
        };

        // Reasoning layer — sentence assembly + deepnsm SPO/`NarsTruth`
        // extraction, run over the SAME already-decoded `grey` buffer as a
        // second recognition pass (`RecognizePageWords`, the typed
        // `LineWords` surface `reasoning.rs`/`sentences.rs` need — distinct
        // from `RecognizeDocument`'s serialized `doc.v1` string above).
        // Mirrors `tesseract-ogar/examples/ocr_demo.rs`'s own step 6.
        // `None` reasoner (no deepnsm vocabulary loaded) means this whole
        // block is skipped, not a failed ingest.
        let (spo_json, triple_count) = match &st.reasoner {
            Some(reasoner) => {
                let words = match st.executor.execute(OcrRequest::RecognizePageWords {
                    grey: &grey,
                    width: w,
                    height: h,
                    with_dict: true,
                }) {
                    Ok(OcrResponse::LineWordsOut(lines)) => lines,
                    Ok(_) => unreachable!("RecognizePageWords always returns LineWordsOut"),
                    Err(e) => return Err(IngestError::Recognize(e)),
                };
                let page = tesseract_ogar::DocPage::from_line_words(
                    &words,
                    st.executor.charset(),
                    w as u32,
                    h as u32,
                );
                let sentences = assemble_sentences(&page);
                let beliefs = reasoner.analyze(sentences);
                let triple_count = beliefs.iter().map(|b| b.triples.len()).sum();
                (Some(spo_beliefs_to_json(&beliefs)), triple_count)
            }
            None => (None, 0),
        };

        Ok((doc_json, spo_json, triple_count))
    })
    .await
    .map_err(|e| IngestError::Task(e.to_string()))??;

    let (mean_confidence, low_confidence) = quality_from_doc_json(&doc_json);
    let ir: DocIr = ogar_from_docv1::from_doc_v1(&doc_json, *hash.as_bytes(), mime)
        .map_err(IngestError::Convert)?;
    let page_count = ir.pages.len();
    let spo_extraction_ran = state.reasoner.is_some();

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
            spo_json.as_deref(),
        )
        .await
        .map_err(IngestError::Store)?;

    // Same text `LanceStore::put` itself just computed and stored -- recomputed
    // here rather than threaded through `put`'s return value, since it is a
    // cheap linear walk over `ir` and keeps `LanceStore`'s API from having to
    // know a search index exists at all.
    let text_for_search = tesseract_paperless::render::plain_text(&ir);
    let display_name = filename.clone().unwrap_or_default();
    let st = state.clone();
    let hex_for_index = hex.clone();
    tokio::task::spawn_blocking(move || {
        st.search
            .index_document(&hex_for_index, &display_name, &text_for_search)
    })
    .await
    .map_err(|e| IngestError::Task(e.to_string()))?
    .map_err(IngestError::Search)?;

    Ok(IngestOutcome::Stored {
        hash_hex: hex,
        document_guid: guid.0.to_bytes(),
        page_count,
        mean_confidence,
        low_confidence,
        triple_count,
        spo_extraction_ran,
    })
}

/// Serialize the reasoning layer's per-sentence output to the JSON shape
/// stored in [`tesseract_paperless::store::DocumentRow::spo_json`]: an array
/// of `{text, coverage, truth: {frequency, confidence}, triples: [{subject,
/// predicate, object}]}` objects, one per assembled sentence (including
/// sentences with zero triples — a low/zero-coverage sentence is still
/// reported, matching `SentenceReasoner::analyze`'s own
/// never-drop-a-sentence guarantee). Neither `SentenceBelief` nor its
/// fields derive `serde::Serialize` (they live in `tesseract-ogar`, which
/// has no reason to carry a serde dependency for this crate's own storage
/// shape), so this hand-builds the `Value` rather than adding derives
/// upstream.
fn spo_beliefs_to_json(beliefs: &[SentenceBelief]) -> String {
    let sentences: Vec<Value> = beliefs
        .iter()
        .map(|b| {
            let triples: Vec<Value> = b
                .triples
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "object": t.object,
                    })
                })
                .collect();
            serde_json::json!({
                "text": b.sentence.text,
                "coverage": b.coverage,
                "truth": {
                    "frequency": b.truth.frequency,
                    "confidence": b.truth.confidence,
                },
                "triples": triples,
            })
        })
        .collect();
    // `Vec<Value>` -> JSON text never fails (every value here is already a
    // valid `Value`, no NaN/Infinity floats can appear from this module's
    // own `sentence_nars_truth` clamp), so a malformed-serialization branch
    // would be dead code -- collapse it to the empty array rather than
    // carry an unreachable `Result`.
    serde_json::to_string(&sentences).unwrap_or_else(|_| "[]".to_string())
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

    use tesseract_ogar::reasoning::{NarsTruth, ResolvedTriple};
    use tesseract_ogar::sentences::AssembledSentence;

    fn sample_belief(text: &str, triples: Vec<ResolvedTriple>) -> SentenceBelief {
        SentenceBelief {
            sentence: AssembledSentence {
                text: text.to_string(),
                bbox: (0, 0, 100, 10),
                line_indices: vec![0],
                mean_conf: 91.5,
            },
            triples,
            coverage: 0.8,
            truth: NarsTruth::new(0.9, 0.75),
        }
    }

    /// The round-trippable shape a consumer of `spo_json` needs: sentence
    /// text, coverage, the truth pair, and every triple field — verified by
    /// parsing the output back rather than substring-matching the string
    /// (a substring check could pass on a shape a real JSON consumer can't
    /// actually read).
    #[test]
    fn spo_beliefs_to_json_round_trips_a_real_triple() {
        let belief = sample_belief(
            "The dog sees the cat.",
            vec![ResolvedTriple {
                subject: "dog".to_string(),
                predicate: "see".to_string(),
                object: Some("cat".to_string()),
            }],
        );
        let json = spo_beliefs_to_json(std::slice::from_ref(&belief));
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed.as_array().expect("array").len(), 1);
        let s0 = &parsed[0];
        assert_eq!(s0["text"], "The dog sees the cat.");
        assert!((s0["coverage"].as_f64().expect("coverage") - 0.8).abs() < 1e-6);
        assert!((s0["truth"]["frequency"].as_f64().expect("frequency") - 0.9).abs() < 1e-6);
        assert!((s0["truth"]["confidence"].as_f64().expect("confidence") - 0.75).abs() < 1e-6);
        let triples = s0["triples"].as_array().expect("triples array");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0]["subject"], "dog");
        assert_eq!(triples[0]["predicate"], "see");
        assert_eq!(triples[0]["object"], "cat");
    }

    /// An intransitive triple's `object: None` must survive as JSON `null`,
    /// not be silently dropped or coerced to an empty string — the
    /// distinction a reader needs to tell "no object" from "empty object".
    #[test]
    fn spo_beliefs_to_json_preserves_a_null_object() {
        let belief = sample_belief(
            "The dog barks.",
            vec![ResolvedTriple {
                subject: "dog".to_string(),
                predicate: "bark".to_string(),
                object: None,
            }],
        );
        let json = spo_beliefs_to_json(std::slice::from_ref(&belief));
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed[0]["triples"][0]["object"].is_null());
    }

    /// A sentence with zero triples (the reasoning layer's own
    /// never-drop-a-sentence guarantee) must still appear in the array —
    /// an implementation that filtered empty-triple sentences out would
    /// silently lose the sentence's own text/coverage/truth from the
    /// stored record.
    #[test]
    fn spo_beliefs_to_json_never_drops_a_zero_triple_sentence() {
        let belief = sample_belief("Zxqvblorptfizz wobbledoop.", Vec::new());
        let json = spo_beliefs_to_json(std::slice::from_ref(&belief));
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed.as_array().expect("array").len(), 1);
        assert!(parsed[0]["triples"]
            .as_array()
            .expect("triples array")
            .is_empty());
    }

    /// An empty belief slice (extraction ran but produced no sentences —
    /// distinct from `spo_extraction_ran == false`) must serialize to the
    /// empty JSON array, never an error string or a malformed value.
    #[test]
    fn spo_beliefs_to_json_of_no_beliefs_is_an_empty_array() {
        assert_eq!(spo_beliefs_to_json(&[]), "[]");
    }
}
