//! Intake: raw bytes → the S-2 gate → one `DocIr`.
//!
//! ```text
//!   HASH BEFORE YOU SPEND.  MANY RETINAS, ONE SHAPE.
//! ```
//!
//! # This module runs the gate; it is not a retina
//!
//! Every producer here already exists elsewhere and already emits
//! [`ogar_doc_ir::DocIr`]. Intake's job is to sit in FRONT of them:
//!
//! | retina | producer | entry point |
//! |---|---|---|
//! | pixel, JSON leg | any `doc.v1` emitter → `ogar-from-docv1` | [`ingest_doc_v1`] |
//! | pixel, full leg | the recognizer in-process | [`ingest_image`] (feature `ocr`) |
//! | anything else | the caller's own producer | [`ingest_doc_ir`] |
//!
//! [`ingest_doc_ir`] is what makes the set open rather than a fixed pair. A
//! DOM crawler, a PDF text layer, an EPUB reader — any producer that can build
//! a `DocIr` is admitted, and this module needs no dependency on it. That is
//! what a source-agnostic IR is FOR, and hard-coding one crawler here would
//! have quietly spent that property: an earlier cut called `spider_doc_ir`
//! directly, which both bound this module to one crawler and inherited its
//! build.
//!
//! # The gate runs FIRST, and that is the whole point of S-2
//!
//! `OGAR-DOC-INGESTION-SPINE` S-2: dedup must precede recognition SPEND. Every
//! entry point hashes and asks the index *before* touching a producer —
//! [`ingest_image`] especially, where the producer is the expensive one. A
//! duplicate costs a hash and a lookup. The gate matches BOTH the original and
//! derived-artifact hashes (S-2's second half), which [`crate::kv::preflight`]
//! already implements.
//!
//! Because that ordering is a property of each ENTRY POINT rather than of the
//! gate, each leg proves it separately, and each proof feeds an input its own
//! producer would reject — reaching `Duplicate` is then only explicable by the
//! gate having answered first.
//!
//! # What `content_sha256` means, stated because the IR's own docs correct
//! their plan on it
//!
//! It is the hash of the ORIGINAL bytes — a **per-acquisition dedup key**, not
//! a cross-retina identity. The same invoice as a scan and as HTML has
//! different bytes and therefore different hashes; cross-retina convergence is
//! a facts question (`ogar_doc_ir::converges_on_facts`). Per-acquisition is
//! exactly what a dedup gate wants, so the two agree here rather than merely
//! coexisting — and [`ingest_doc_ir`] ENFORCES the agreement instead of hoping
//! for it: a producer whose IR is keyed by different bytes than the gate hashed
//! is refused, loudly.

use crate::kv::{ContentSha256, DedupIndex, MatchedOn, Verdict};
use ogar_doc_ir::DocIr;

/// What one intake attempt produced.
#[derive(Debug, Clone)]
pub enum Ingested {
    /// The gate matched. No retina ran.
    Duplicate {
        /// The hash that matched.
        hash: ContentSha256,
        /// Which stored hash it matched — original bytes, or a derived
        /// artifact (S-2's second half).
        matched: MatchedOn,
    },
    /// Novel bytes; a retina ran and produced one shape.
    Novel {
        /// The per-acquisition dedup key.
        hash: ContentSha256,
        /// The perceptual IR. [`DocIr::source`] says which retina.
        ir: Box<DocIr>,
    },
}

impl Ingested {
    /// The IR, if a retina ran.
    #[must_use]
    pub fn ir(&self) -> Option<&DocIr> {
        match self {
            Self::Novel { ir, .. } => Some(ir),
            Self::Duplicate { .. } => None,
        }
    }

    /// The hash either way — a duplicate still has an identity.
    #[must_use]
    pub const fn hash(&self) -> &ContentSha256 {
        match self {
            Self::Novel { hash, .. } | Self::Duplicate { hash, .. } => hash,
        }
    }
}

/// Why an intake attempt failed.
#[derive(Debug)]
pub enum IntakeError {
    /// A caller-supplied [`DocIr`] is keyed by different bytes than the gate
    /// hashed. Never a benign mismatch: it means the producer and the dedup
    /// index disagree about what document this IS, so one of them would go on
    /// addressing the wrong subtree.
    IdentityMismatch {
        /// What [`crate::kv::preflight`] hashed — the original bytes.
        gate: [u8; 32],
        /// What the producer stamped into `DocIr::content_sha256`.
        producer: [u8; 32],
    },
    /// The pixel retina's adapter refused the JSON — malformed, wrong schema,
    /// or a region kind outside the closed vocabulary. Fail-loud is the point:
    /// a producer that drifts is caught at the seam.
    DocV1(ogar_from_docv1::FromDocV1Error),
    /// The recognizer refused the page (`ocr` feature only).
    #[cfg(feature = "ocr")]
    Ocr(tesseract_ogar::OcrExecError),
    /// The recognizer answered, but not with a document (`ocr` feature only).
    /// Structurally unreachable for a `RecognizeDocument` request; kept
    /// because an enum match that cannot fail is a claim, not a guarantee.
    #[cfg(feature = "ocr")]
    NotADocument,
}

impl core::fmt::Display for IntakeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IdentityMismatch { gate, producer } => write!(
                f,
                "identity mismatch: the gate hashed {:02x}{:02x}.., the producer stamped {:02x}{:02x}..",
                gate[0], gate[1], producer[0], producer[1]
            ),
            Self::DocV1(e) => write!(f, "doc.v1 refused at the seam: {e}"),
            #[cfg(feature = "ocr")]
            Self::Ocr(e) => write!(f, "recognition failed: {e:?}"),
            #[cfg(feature = "ocr")]
            Self::NotADocument => {
                write!(f, "the recognizer returned a non-document response")
            }
        }
    }
}

impl std::error::Error for IntakeError {}

/// Any retina: bytes the caller has ALREADY turned into a `DocIr`, gated.
///
/// This is the open end of the set. The caller owns the producer — a DOM
/// crawler, a PDF text layer, an EPUB reader — and this module owns neither
/// the dependency nor the build. `build` is a closure rather than a value so
/// the gate can still short-circuit: on a duplicate it is **never called**,
/// which is what keeps S-2's "before the spend" true for a producer this
/// module has never heard of.
///
/// # Errors
/// [`IntakeError::IdentityMismatch`] if the produced IR is keyed by different
/// bytes than the gate hashed. That is checked rather than assumed: the two
/// hashes agreeing is the entire basis for a `doc.v1` subtree and a dedup
/// index addressing the same document, and a producer that hashes a
/// normalised or re-encoded form of its input would break it silently.
pub fn ingest_doc_ir<I, F>(
    source_bytes: &[u8],
    index: &I,
    build: F,
) -> Result<Ingested, IntakeError>
where
    I: DedupIndex,
    F: FnOnce(&[u8]) -> DocIr,
{
    let (hash, verdict) = crate::kv::preflight(source_bytes, index);
    if let Verdict::Duplicate { matched } = verdict {
        return Ok(Ingested::Duplicate { hash, matched });
    }
    let ir = build(source_bytes);
    if ir.content_sha256 != hash.0 {
        return Err(IntakeError::IdentityMismatch {
            gate: hash.0,
            producer: ir.content_sha256,
        });
    }
    Ok(Ingested::Novel {
        hash,
        ir: Box::new(ir),
    })
}

/// Pixel retina, JSON leg: `doc.v1` from any producer → one `DocIr`, gated.
///
/// `source_bytes` are the ORIGINAL image bytes — what the gate hashes and what
/// the IR's identity is taken over. The JSON is a rendition of them, so
/// hashing the JSON instead would make two renderings of one scan look like
/// two documents.
///
/// # Errors
/// [`IntakeError::DocV1`] if the JSON is malformed, carries the wrong schema
/// marker, or names a region kind outside the closed vocabulary.
pub fn ingest_doc_v1<I: DedupIndex>(
    source_bytes: &[u8],
    doc_v1_json: &str,
    mime: &str,
    index: &I,
) -> Result<Ingested, IntakeError> {
    let (hash, verdict) = crate::kv::preflight(source_bytes, index);
    if let Verdict::Duplicate { matched } = verdict {
        return Ok(Ingested::Duplicate { hash, matched });
    }
    let ir = ogar_from_docv1::from_doc_v1(doc_v1_json, hash.0, mime).map_err(IntakeError::DocV1)?;
    Ok(Ingested::Novel {
        hash,
        ir: Box::new(ir),
    })
}

/// Pixel retina, full leg: a grey page → recognition → `doc.v1` → one `DocIr`,
/// gated. Behind the off-by-default `ocr` feature.
///
/// The gate runs before the executor is touched, so a duplicate page costs a
/// hash and a lookup rather than a recognition pass. That ordering is the
/// reason S-2 exists and it is asserted, not assumed
/// (see `duplicate_never_reaches_the_retina`).
///
/// # Errors
/// [`IntakeError::Ocr`] if recognition fails, [`IntakeError::NotADocument`] if
/// the executor answers with another response shape, [`IntakeError::DocV1`] if
/// its own `doc.v1` does not pass the seam.
#[cfg(feature = "ocr")]
pub fn ingest_image<I: DedupIndex>(
    grey: &[u8],
    width: usize,
    height: usize,
    mime: &str,
    executor: &tesseract_ogar::OcrExecutor,
    index: &I,
) -> Result<Ingested, IntakeError> {
    use tesseract_ogar::{BinarizeMode, OcrRequest, OcrResponse};

    let (hash, verdict) = crate::kv::preflight(grey, index);
    if let Verdict::Duplicate { matched } = verdict {
        return Ok(Ingested::Duplicate { hash, matched });
    }
    let resp = executor
        .execute(OcrRequest::RecognizeDocument {
            grey,
            width,
            height,
            with_dict: false,
            harvest_profile: None,
            binarize: BinarizeMode::default(),
        })
        .map_err(IntakeError::Ocr)?;
    let OcrResponse::DocumentOut { doc_json, .. } = resp else {
        return Err(IntakeError::NotADocument);
    };
    let ir = ogar_from_docv1::from_doc_v1(&doc_json, hash.0, mime).map_err(IntakeError::DocV1)?;
    Ok(Ingested::Novel {
        hash,
        ir: Box::new(ir),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::DocumentGuid;
    use std::cell::Cell;

    /// An index that knows nothing.
    struct Empty;
    impl DedupIndex for Empty {
        fn look_up(&self, _: &ContentSha256) -> Option<(DocumentGuid, MatchedOn)> {
            None
        }
    }

    /// An index that has seen everything, and counts how often it was asked.
    struct SeenAll {
        asked: Cell<usize>,
        matched: MatchedOn,
    }
    impl DedupIndex for SeenAll {
        fn look_up(&self, _: &ContentSha256) -> Option<(DocumentGuid, MatchedOn)> {
            self.asked.set(self.asked.get() + 1);
            Some((
                DocumentGuid(lance_graph_contract::facet::FacetCascade::default()),
                self.matched,
            ))
        }
    }

    /// A stand-in for "some producer that is not in this crate". It builds a
    /// minimal but VALID `DocIr` and — crucially — keys it the same way every
    /// real producer does, by hashing the bytes it was handed.
    fn third_party_producer(bytes: &[u8]) -> DocIr {
        DocIr {
            version: ogar_doc_ir::DOC_IR_VERSION.to_string(),
            source: ogar_doc_ir::Provenance::Dom,
            geometry: ogar_doc_ir::Geometry::DomOrder,
            content_sha256: ContentSha256::of(bytes).0,
            mime: "text/html".to_string(),
            pages: Vec::new(),
            fields: Vec::new(),
        }
    }

    #[test]
    fn a_third_party_producer_lands_in_the_same_shape() {
        let out = ingest_doc_ir(b"<html>...</html>", &Empty, third_party_producer)
            .expect("identities agree");
        let ir = out.ir().expect("novel");
        assert_eq!(ir.source, ogar_doc_ir::Provenance::Dom);
        assert_eq!(ir.version, ogar_doc_ir::DOC_IR_VERSION);
        // It passes the IR's OWN load gate, which is the contract every
        // producer is held to regardless of which retina it models.
        let json = ogar_doc_ir::to_json(ir).expect("serialize");
        assert!(ogar_doc_ir::from_json(&json).is_ok());
    }

    #[test]
    fn a_producer_keyed_by_other_bytes_is_refused() {
        // The guard that replaced an assertion. A producer that hashes a
        // NORMALISED form of its input (here: trimmed) is the realistic way
        // this goes wrong — the document is right, the identity is not, and
        // without this check the gate and the subtree would silently address
        // two different documents.
        let normalising = |b: &[u8]| {
            let mut ir = third_party_producer(b);
            let trimmed: &[u8] = b"<html>...</html>";
            ir.content_sha256 = ContentSha256::of(trimmed).0;
            ir
        };
        let bytes = b"  <html>...</html>  ";
        assert!(matches!(
            ingest_doc_ir(bytes, &Empty, normalising),
            Err(IntakeError::IdentityMismatch { .. })
        ));
        // ...and the SAME producer on bytes that need no normalising is
        // accepted, so the guard is discriminating rather than always-on.
        assert!(matches!(
            ingest_doc_ir(b"<html>...</html>", &Empty, normalising),
            Ok(Ingested::Novel { .. })
        ));
    }

    #[test]
    fn a_duplicate_never_reaches_a_third_party_producer() {
        // The producer PANICS if called. Reaching `Duplicate` therefore
        // proves the gate short-circuited before the closure ran — which is
        // the whole of S-2, and the reason `build` is a closure rather than
        // an already-built `DocIr`.
        let index = SeenAll {
            asked: Cell::new(0),
            matched: MatchedOn::Original,
        };
        let out = ingest_doc_ir(b"seen-before", &index, |_| {
            panic!("the producer must not run for a duplicate")
        })
        .expect("the gate answers first");
        assert!(matches!(out, Ingested::Duplicate { .. }));
        assert_eq!(index.asked.get(), 1, "the gate is asked exactly once");
        // And on NOVEL bytes the very same closure does run — otherwise the
        // assertion above would hold for a producer that is never called at
        // all, which proves nothing about ordering.
        let ran = Cell::new(false);
        let _ = ingest_doc_ir(b"never-seen", &Empty, |b| {
            ran.set(true);
            third_party_producer(b)
        });
        assert!(ran.get(), "a novel document must reach its producer");
    }

    #[test]
    fn derived_artifact_match_is_reported_as_such() {
        let index = SeenAll {
            asked: Cell::new(0),
            matched: MatchedOn::Derived,
        };
        let out = ingest_doc_ir(
            b"an export of a held document",
            &index,
            third_party_producer,
        )
        .expect("gate");
        assert!(matches!(
            out,
            Ingested::Duplicate {
                matched: MatchedOn::Derived,
                ..
            }
        ));
    }

    #[test]
    fn doc_v1_seam_fails_loud_on_a_drifted_producer() {
        let good = r#"{"schema":"tesseract-rs/doc.v1","pages":[{"page":0,"width":100,
            "height":100,"regions":[{"type":"text","bbox":[0,0,50,50],"lines":[]}]}]}"#;
        let img = b"pretend-these-are-image-bytes";
        let ok = ingest_doc_v1(img, good, "image/png", &Empty).expect("valid doc.v1");
        let ir = ok.ir().expect("novel");
        assert_eq!(ir.source, ogar_doc_ir::Provenance::Ocr);
        assert_eq!(ir.mime, "image/png");
        // Identity is over the IMAGE bytes, not the JSON: two renderings of one
        // scan must not look like two documents.
        assert_eq!(&ir.content_sha256, &ContentSha256::of(img).0);

        // An off-vocabulary region kind is refused at the seam.
        let drifted = good.replace(r#""type":"text""#, r#""type":"paragraph""#);
        assert!(matches!(
            ingest_doc_v1(img, &drifted, "image/png", &Empty),
            Err(IntakeError::DocV1(_))
        ));
        // ...and so is a wrong schema marker.
        let wrong = good.replace("tesseract-rs/doc.v1", "tesseract-rs/doc.v2");
        assert!(matches!(
            ingest_doc_v1(img, &wrong, "image/png", &Empty),
            Err(IntakeError::DocV1(_))
        ));
    }

    #[test]
    fn a_duplicate_never_reaches_the_doc_v1_seam() {
        // The pixel leg's own S-2 proof. The JSON below is garbage: if the
        // seam ran at all, `from_doc_v1` would refuse it and this would be
        // `Err(DocV1)`. Landing on `Duplicate` proves the gate answered
        // FIRST — which is the whole of S-2, and is a per-entry-point
        // property, not something the DOM leg's test can establish for this
        // one.
        let junk = "}{ not json";
        let img = b"already-seen-image-bytes";
        let index = SeenAll {
            asked: Cell::new(0),
            matched: MatchedOn::Original,
        };
        let out = ingest_doc_v1(img, junk, "image/png", &index).expect("the gate answers first");
        assert!(matches!(out, Ingested::Duplicate { .. }));
        assert_eq!(index.asked.get(), 1, "the gate is asked exactly once");
        // ...and the same junk as a NOVEL input must fail, or the assertion
        // above would pass for the wrong reason.
        assert!(matches!(
            ingest_doc_v1(img, junk, "image/png", &Empty),
            Err(IntakeError::DocV1(_))
        ));
    }

    #[test]
    fn every_producer_lands_in_one_type() {
        // The point of the whole module: a third-party producer and the
        // built-in doc.v1 seam are the SAME Rust type, distinguishable only
        // by the field that says which retina made it.
        let other = ingest_doc_ir(b"<html>...</html>", &Empty, third_party_producer).expect("ok");
        let pixel = ingest_doc_v1(
            b"img",
            r#"{"schema":"tesseract-rs/doc.v1","pages":[{"page":0,"width":10,"height":10,
               "regions":[{"type":"text","bbox":[0,0,5,5],"lines":[]}]}]}"#,
            "image/png",
            &Empty,
        )
        .expect("valid doc.v1");
        let irs: Vec<&DocIr> = vec![other.ir().expect("novel"), pixel.ir().expect("novel")];
        assert_eq!(irs.len(), 2);
        assert_eq!(irs[0].source, ogar_doc_ir::Provenance::Dom);
        assert_eq!(irs[1].source, ogar_doc_ir::Provenance::Ocr);
        assert!(irs.iter().all(|i| i.version == ogar_doc_ir::DOC_IR_VERSION));
    }
}
