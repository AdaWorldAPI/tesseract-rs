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
//! - [`UniCharSet::get_isalpha`] + the `get_is{lower,upper,digit,punctuation}`
//!   family — `ccutil/unicharset.cpp` property decode (`properties & MASK`),
//!   **112/112 byte-identical** on a real `eng` unicharset against tesseract's
//!   own `get_is*` accessors.
//! - [`UniCharSet::get_script`] + [`UniCharSet::script_of`] — the interned
//!   per-id script table (`ccutil/unicharset.cpp` `add_script`),
//!   **112/112 byte-identical** on a real `eng` unicharset against tesseract's
//!   own `get_script`.
//! - [`UniCharSet::get_other_case`] — the case-pair id per entry (the
//!   `ccutil/unicharset.cpp` size-clamp), **112/112 byte-identical** on a real
//!   `eng` unicharset against tesseract's own `get_other_case`.
//! - [`UniCharSet::get_direction`] + [`UniCharSet::get_mirror`] — the bidi
//!   direction code + mirror id per entry (the columns past the bbox CSV),
//!   **112/112 byte-identical** on a real `eng` unicharset against tesseract's
//!   own `get_direction` / `get_mirror`.
//! - [`unichar`] — `ccutil/unichar.cpp`, the UTF-8 codec (`utf8_step` +
//!   `utf8_to_utf32`), **268/268 byte-identical** (256 exhaustive lead-byte
//!   values + 12 decode rows).
//! - [`UnicharCompress`] — `ccutil/unicharcompress.cpp`, the LSTM recoder's load
//!   side (`DeSerialize` + `EncodeUnichar` / `DecodeUnichar` / `code_range`),
//!   **byte-identical** on the real `eng.lstm-recoder` (112 encode + 112 decode
//!   rows, plus `code_range`).
//!
//! This module re-exports those as the OCR's character-set substrate and adds
//! the OCR-facing transcoded output steps on top of them: [`ids_to_text`] (the
//! id→text walk) and [`recoded_to_text`] (the recoder-fed codes→ids→text path).
//!
//! ## First landed layer: the character set
//!
//! Tesseract's recognizer emits a sequence of `UNICHAR_ID`s; turning that into
//! text is a `UNICHARSET::id_to_unichar` walk. [`ids_to_text`] is the pure-Rust
//! transcode of that output step.

pub mod dict_walker;
pub mod recodebeam;

pub use dict_walker::{DawgPosition, DictLite};
/// The Core's proven dawg-table surface (`SquishedDawg` load + `edge_char_of`
/// traversal, `DawgType`/`PermuterType`, `NodeRef`/`NO_EDGE`) — the table
/// [`dict_walker::DictLite`] walks.
pub use lance_graph_contract::dawg;
/// The Core's proven network-structure surface (`NetworkType`, `NetworkHeader`,
/// the FacetCascade sink — `E-OCR-NETWORK-SINK-1`): the per-node header parse
/// the OCR assembly tier's tree loader consumes (Core-First: structure
/// vocabulary from the Core, compute payloads in the recognizer).
pub use lance_graph_contract::network;
pub use lance_graph_contract::unichar;
pub use lance_graph_contract::unicharcompress::{RecodedCharId, RecoderError, UnicharCompress};
pub use lance_graph_contract::unicharset::{UniCharSet, UniCharSetError};
pub use recodebeam::RecodeBeamSearch;

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

/// The OCR recoder — Tesseract's `UnicharCompress`, transcoded and proven in the
/// OGAR Core. The LSTM recognizer's lattice speaks recoded codes; the recoder
/// maps each back to a `UNICHAR_ID`. This alias mirrors [`CharSet`].
pub type Recoder = UnicharCompress;

/// Decode a recognizer's recoded-code sequence into text — the real OCR output
/// path: each [`RecodedCharId`] → [`UnicharCompress::decode`] → `UNICHAR_ID` →
/// [`ids_to_text`]. A code the recoder cannot decode yields `INVALID_UNICHAR_ID`
/// and contributes nothing, mirroring libtesseract's drop of an unknown code.
#[must_use]
pub fn recoded_to_text(
    recoder: &UnicharCompress,
    charset: &UniCharSet,
    codes: &[RecodedCharId],
) -> String {
    let ids: Vec<u32> = codes
        .iter()
        .filter_map(|code| u32::try_from(recoder.decode(code)).ok())
        .collect();
    ids_to_text(charset, &ids)
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
    fn charset_exposes_proven_properties() {
        // Through the OCR core's `CharSet` surface, the property accessors the
        // Core proved byte-identical (112/112 vs tesseract's `get_is*`) are
        // reachable. Second column is the hex property mask: 0x3=alpha+lower,
        // 0x5=alpha+upper, 0x8=digit, 0x10=punct.
        let cs: CharSet = UniCharSet::load_from_str(
            "4\na 3 0 a Left a a\nA 5 0 A Left A A\n7 8 0 7 Left 7 7\n. 10 0 . Left . .\n",
        )
        .expect("valid unicharset");
        assert!(cs.get_isalpha(0) && cs.get_islower(0) && !cs.get_isupper(0));
        assert!(cs.get_isalpha(1) && cs.get_isupper(1) && !cs.get_islower(1));
        assert!(cs.get_isdigit(2) && !cs.get_isalpha(2));
        assert!(cs.get_ispunctuation(3) && !cs.get_isdigit(3));
        // INVALID_UNICHAR_ID semantics: out-of-range never panics, returns false.
        assert!(!cs.get_isalpha(99));
        assert!(!cs.get_isngram(0)); // plain-table load never sets ngram
    }

    #[test]
    fn charset_exposes_script() {
        // The interned per-id script table (proven 112/112 vs tesseract's
        // get_script) is reachable through CharSet. null_script is seeded at sid
        // 0; real scripts follow in id order (here: Common=1, Latin=2). Mixed
        // tiers: id 0 has no bbox CSV, ids 1-2 do.
        let cs: CharSet = UniCharSet::load_from_str(
            "3\nNULL 0 Common 0\nA 5 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 A\n. 10 0,255,0,255,0,0,0,0,0,0 Common 2 0 2 .\n",
        )
        .expect("valid unicharset");
        assert_eq!(cs.get_script_table_size(), 3);
        assert_eq!(cs.script_of(0), Some("Common")); // the space char's script
        assert_eq!(cs.script_of(1), Some("Latin"));
        assert_eq!(cs.script_of(2), Some("Common"));
        // null_sid_ / INVALID_UNICHAR_ID: out-of-range resolves the null script.
        assert_eq!(cs.get_script(99), 0);
        assert_eq!(cs.script_of(99), Some("NULL"));
    }

    #[test]
    fn charset_exposes_other_case() {
        // The case-pair table (proven 112/112 vs tesseract's get_other_case) is
        // reachable through CharSet: C<->c, and the INVALID_UNICHAR_ID guard.
        let cs: CharSet = UniCharSet::load_from_str(
            "2\nC 5 0,255,0,255,0,0,0,0,0,0 Latin 1 0 0 C\nc 3 0,255,0,255,0,0,0,0,0,0 Latin 0 0 1 c\n",
        )
        .expect("valid unicharset");
        assert_eq!(cs.get_other_case(0), 1); // C -> c
        assert_eq!(cs.get_other_case(1), 0); // c -> C
        assert_eq!(cs.get_other_case(99), -1); // INVALID_UNICHAR_ID
    }

    #[test]
    fn charset_exposes_direction_and_mirror() {
        // The bidi direction + mirror columns (proven 112/112 vs tesseract's
        // get_direction/get_mirror) are reachable through CharSet: a paren pair
        // mirrors, both U_OTHER_NEUTRAL (10), plus the out-of-range guards.
        let cs: CharSet = UniCharSet::load_from_str(
            "2\n( 10 0,255,0,255,0,0,0,0,0,0 Common 0 10 1 (\n) 10 0,255,0,255,0,0,0,0,0,0 Common 0 10 0 )\n",
        )
        .expect("valid unicharset");
        assert_eq!(cs.get_direction(0), 10); // U_OTHER_NEUTRAL
        assert_eq!(cs.get_mirror(0), 1); // ( -> )
        assert_eq!(cs.get_mirror(1), 0); // ) -> (
        assert_eq!(cs.get_direction(99), 10); // out-of-range -> U_OTHER_NEUTRAL
        assert_eq!(cs.get_mirror(99), -1); // out-of-range -> INVALID_UNICHAR_ID
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

    /// A tiny pass-through recoder (id 0→code[0], 1→[1], 2→[2]) in the exact
    /// little-endian wire form `UnicharCompress::Serialize` writes.
    fn sample_recoder() -> Recoder {
        let mut bytes = 3_u32.to_le_bytes().to_vec();
        for code in [0_i32, 1, 2] {
            bytes.push(1); // self_normalized
            bytes.extend_from_slice(&1_i32.to_le_bytes()); // length
            bytes.extend_from_slice(&code.to_le_bytes());
        }
        UnicharCompress::from_le_bytes(&bytes).expect("valid recoder")
    }

    #[test]
    fn recoder_decodes_a_lattice_to_text() {
        // The recoder (proven byte-identical on eng.lstm-recoder) composes with
        // the charset: recognizer codes → decode → UNICHAR_IDs → id_to_unichar.
        let charset = sample(); // id 0 = space, 1 = a, 2 = b
        let recoder = sample_recoder();
        // A recognition lattice for ids 1, 0, 2 — round-tripped through `encode`
        // to get valid RecodedCharIds the way the recognizer would assemble them.
        let codes: Vec<RecodedCharId> = [1_u32, 0, 2]
            .iter()
            .map(|&id| recoder.encode(id).expect("in range").clone())
            .collect();
        assert_eq!(recoded_to_text(&recoder, &charset, &codes), "a b");
        // An unknown/ill-formed code decodes to INVALID_UNICHAR_ID and is dropped.
        assert_eq!(
            recoded_to_text(&recoder, &charset, &[RecodedCharId::default()]),
            ""
        );
        assert_eq!(recoded_to_text(&recoder, &charset, &[]), "");
    }
}
