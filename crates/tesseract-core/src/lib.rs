//! # tesseract-core — the pure-Rust Tesseract transcode
//!
//! Operator directive: **transcode Tesseract into Rust, do NOT wrap
//! libtesseract.** The root `tesseract` crate is the legacy FFI wrapper
//! (`tesseract-sys`/`tesseract-plumbing`); this crate is the pure-Rust target
//! that replaces it leaf-by-leaf, each leaf byte-parity-proven against the C++
//! original before it lands.
//!
//! ## Why it consumes `lance-graph-contract` rather than re-implementing
//!
//! Per the Core-First transcode doctrine, the transcoded leaves live in the
//! OGAR Core (`lance-graph-contract`), and consumers like this OCR crate use
//! them — they do not re-transcode. The Core already proved, in-env against a
//! libtesseract oracle:
//!
//! - [`UniCharSet`] — `ccutil/unicharset.cpp`, the id↔unichar bijection,
//!   **112/112 byte-identical** on a real `eng` unicharset.
//! - [`unichar`] — `ccutil/unichar.cpp`, the UTF-8 codec (`utf8_step` +
//!   `utf8_to_utf32`), **268/268 byte-identical** (256 exhaustive lead-byte
//!   values + 12 decode rows).
//!
//! This module re-exports those as the OCR's character-set substrate and adds
//! the first OCR-facing transcoded step on top of them: [`ids_to_text`].
//!
//! ## First landed layer: the character set
//!
//! Tesseract's recognizer emits a sequence of `UNICHAR_ID`s; turning that into
//! text is a `UNICHARSET::id_to_unichar` walk. [`ids_to_text`] is the pure-Rust
//! transcode of that output step.

pub use lance_graph_contract::unichar;
pub use lance_graph_contract::unicharset::{UniCharSet, UniCharSetError};

/// The OCR character set — Tesseract's `UNICHARSET`, transcoded and proven in
/// the OGAR Core. This alias is the OCR core's pure-Rust char-set surface; the
/// recognizer (a later transcoded leaf) reads it to interpret class ids.
pub type CharSet = UniCharSet;

/// Decode a recognizer's `UNICHAR_ID` sequence into text via the character set —
/// the pure-Rust transcode of Tesseract's id→text output step
/// (`UNICHARSET::id_to_unichar` per id, concatenated).
///
/// An id out of the charset's range is **skipped** (the empty contribution),
/// mirroring libtesseract's `INVALID_UNICHAR_ID` drop — a recognizer never
/// emits text for an id the charset does not know.
#[must_use]
pub fn ids_to_text(charset: &UniCharSet, ids: &[u32]) -> String {
    ids.iter()
        .filter_map(|&id| charset.id_to_unichar(id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny in-memory `.unicharset`: id 0 = `NULL`→space (the byte-parity
    /// edge), id 1 = `a`, id 2 = `b`.
    fn sample() -> UniCharSet {
        UniCharSet::load_from_str("3\nNULL 0 Common 0\na 3 0 a Left a a\nb 3 0 b Left b b\n")
            .expect("valid unicharset")
    }

    #[test]
    fn charset_is_the_proven_adapter() {
        let cs = sample();
        // The NULL→space convention the byte-parity probe locked carries through.
        assert_eq!(cs.id_to_unichar(0), Some(" "));
        assert_eq!(cs.id_to_unichar(1), Some("a"));
        assert_eq!(cs.unichar_to_id(" "), Some(0));
        assert_eq!(cs.id_to_unichar(99), None);
    }

    #[test]
    fn ids_to_text_decodes_a_recognition() {
        let cs = sample();
        // "a b a" — id 1, space (id 0), id 2, space, id 1.
        assert_eq!(ids_to_text(&cs, &[1, 0, 2, 0, 1]), "a b a");
        // Unknown ids are dropped (INVALID_UNICHAR_ID semantics), not panicked.
        assert_eq!(ids_to_text(&cs, &[1, 999, 2]), "ab");
        assert_eq!(ids_to_text(&cs, &[]), "");
    }

    #[test]
    fn unichar_codec_is_reexported() {
        // The proven UTF-8 codec is reachable as the OCR core's text layer.
        assert_eq!(unichar::utf8_step(b'A'), 1);
        assert_eq!(
            unichar::utf8_to_utf32(&[0xE4, 0xB8, 0xAD]),
            Some(vec![0x4E2D])
        );
    }
}
