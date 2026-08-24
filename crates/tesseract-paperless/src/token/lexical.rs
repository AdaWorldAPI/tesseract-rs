//! The `DeepNSM-v2` projection: token ids -> lexical units -> `WordId`.
//!
//! ```text
//!   BPE TOKEN IDS ARE SEQUENTIAL.  DEEPNSM WORD COORDINATES ARE SEMANTIC.
//!   A PROJECTION IS NOT A SECOND VOCABULARY.
//! ```
//!
//! `DeepNSM-v2`'s lexical unit is defined by its shipped consumers
//! (`examples/bible_wave.rs::normalise`, `examples/genre_shapes.rs`): split the
//! text on WHITESPACE, then within each whitespace token keep only the ASCII
//! alphabetic bytes, lowercase them, and drop the result if shorter than two
//! characters. `PaletteVocab::id()` is an exact-match `HashMap` lookup and
//! normalises nothing — its own doc says "caller lowercases/normalizes".
//!
//! The whole point of this module is that the rule above is applied to the
//! CONTRACT'S PER-ID SURFACE TABLE, never to the source text. The function
//! signature is the proof: [`project`] takes a [`TokenStreamView`] and no
//! source bytes at all, so re-reading the source is not merely avoided, it is
//! unavailable.
//!
//! What this module deliberately does NOT do: assign a `WordId` to a BPE token.
//! That would be a second vocabulary wearing `DeepNSM`'s coordinate system, and
//! the two id spaces mean different things. Many BPE tokens map to one word;
//! one token can also contain several words. Both cardinalities are measured
//! rather than assumed.

use crate::token::lane::TokenStreamView;

/// Minimum surviving length, matching `DeepNSM`'s `normalise`.
pub const MIN_WORD_LEN: usize = 2;

/// One lexical unit recovered from the token stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexicalUnit {
    /// The normalised surface — lowercase ASCII alphabetic only.
    pub surface: String,
    /// Byte offset of the unit in the source's canonical text.
    pub byte_from: u32,
    /// End byte offset (exclusive).
    pub byte_to: u32,
    /// Index of the first token this unit spans.
    pub first_token: u32,
    /// Number of tokens this unit spans (>= 1).
    pub token_span: u32,
}

/// Project a borrowed token view into `DeepNSM` lexical units.
///
/// Reads the contract's per-id surface table and nothing else. The source text
/// is not a parameter.
///
/// # Panics
/// If a token surface is longer than `u32::MAX` bytes, which a `<= 255`-entry
/// BPE table cannot produce.
#[must_use]
pub fn project(view: &TokenStreamView<'_>) -> Vec<LexicalUnit> {
    let contract = view.contract();
    let mut out = Vec::new();
    // The unit under construction: normalised bytes, plus the byte span of the
    // whitespace-delimited token it came from, plus the token run it spans.
    let mut buf: Vec<u8> = Vec::new();
    let mut span_from = 0u32;
    let mut span_to = 0u32;
    let mut first_token = 0u32;
    let mut open = false;

    let flush = |buf: &mut Vec<u8>,
                 open: &mut bool,
                 span_from: u32,
                 span_to: u32,
                 first_token: u32,
                 last_token: u32,
                 out: &mut Vec<LexicalUnit>| {
        if *open {
            if buf.len() >= MIN_WORD_LEN {
                out.push(LexicalUnit {
                    surface: String::from_utf8(buf.clone()).unwrap_or_default(),
                    byte_from: span_from,
                    byte_to: span_to,
                    first_token,
                    token_span: last_token - first_token + 1,
                });
            }
            buf.clear();
            *open = false;
        }
    };

    let mut last_token = 0u32;
    for t in view.tokens() {
        let surface = contract.surface(t.id);
        for (k, &b) in surface.iter().enumerate() {
            let at = t.byte_from + u32::try_from(k).expect("token surface is short");
            if b.is_ascii_whitespace() {
                flush(
                    &mut buf,
                    &mut open,
                    span_from,
                    span_to,
                    first_token,
                    last_token,
                    &mut out,
                );
            } else {
                if !open {
                    open = true;
                    span_from = at;
                    first_token = t.position;
                }
                last_token = t.position;
                span_to = at + 1;
                if b.is_ascii_alphabetic() {
                    buf.push(b.to_ascii_lowercase());
                }
            }
        }
    }
    flush(
        &mut buf,
        &mut open,
        span_from,
        span_to,
        first_token,
        last_token,
        &mut out,
    );
    out
}
