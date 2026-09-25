//! The Tantivy seam: a `Tokenizer` that reads the resident lane instead of text.
//!
//! ```text
//!   AN INDEX MAY ACCELERATE THE ABI.  IT MUST NEVER BECOME THE ABI.
//! ```
//!
//! # Why this cannot silently re-tokenize
//!
//! The value put into the indexed text field is not the document text. It is a
//! RECEIPT HANDLE (`rcpt:<sha256>:<page>:<reading_order>`, see [`handle_for`]). The tokenizer resolves the handle against the
//! lane and walks borrowed ids. Re-tokenizing the source is therefore not
//! merely avoided by discipline — Tantivy is never handed the source at all.
//!
//! `Tokenizer::token_stream<'a>(&'a mut self, text: &'a str)` places no
//! requirement that emitted tokens derive from `text`; the only real bounds are
//! `'static + Clone + Send + Sync`, which is why the store is behind an `Arc`.
//!
//! # What Tantivy actually keeps
//!
//! Measured in this fork: the indexer reads `Token::text` (into the term bytes)
//! and `Token::position` (persisted as the position list), uses
//! `position_length` transiently to advance the position cursor, and NEVER
//! reads `offset_from`/`offset_to` outside its own tests
//! (`src/postings/postings_writer.rs::index_text`). Byte offsets are consumed
//! only by snippet generation, which re-tokenizes the STORED text at query time
//! (`src/snippet/mod.rs:211`). Two consequences, both stated rather than
//! discovered later:
//!
//! - the index structurally cannot become the owner of offsets, which is
//!   exactly the demarcation this architecture wants;
//! - Tantivy's built-in snippet generator does not work over a receipt handle.
//!   Highlighting has to be served from the canonical text through the receipt
//!   — which is where it belongs, and is a named consequence, not a defect.
//!
//! # The alternative that was measured and rejected for the resident path
//!
//! `PreTokenizedString { text: String, tokens: Vec<Token> }` allocates one
//! `String` per token, and `segment_writer.rs` deep-clones the whole boxed
//! value before indexing it (`PreTokenizedStream::from(*tok_str.clone())`), so
//! one field costs about `4 + 2N` allocations for N tokens. That is the
//! materialised token-object population the root memory law forbids. The
//! tokenizer below reuses ONE `Token` buffer, the way Tantivy's own
//! `SimpleTokenizer` does.

use std::sync::Arc;

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

use crate::token::contract::TokenizerContract;
use crate::token::docir::SpanKey;
use crate::token::lane::{TokenLane, TokenStreamReceipt};

/// The prefix that marks an indexed field value as a receipt handle.
pub const HANDLE_PREFIX: &str = "rcpt:";

/// What a Tantivy term IS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermMode {
    /// The term is the token's decoded surface. Ordinary lexical retrieval,
    /// with sub-word semantics: a term can be a word fragment.
    Surface,
    /// The term is the token id itself, rendered as two hex digits. The term
    /// dictionary is then bounded by the vocabulary (<= 255 terms) regardless
    /// of corpus size, and phrase queries run over id sequences.
    TokenId,
}

/// The resident side of the seam: one contract, one lane.
#[derive(Debug)]
pub struct SeamStore {
    /// The codebook every id in the lane is read under.
    pub contract: TokenizerContract,
    /// The resident particles and their receipts.
    pub lane: TokenLane,
}

impl SeamStore {
    /// Whether `query` is encodable under this store's contract. A search
    /// path calls this before searching, so a query containing a byte the
    /// archive never trained on is REPORTED as such rather than returning the
    /// same empty result as a genuine miss.
    ///
    /// # Errors
    /// The offset of the first byte outside the trained alphabet.
    pub fn covers(&self, query: &str) -> Result<(), usize> {
        self.contract.covers(query.as_bytes())
    }
}

/// The CONTENT ADDRESS of a receipt, as a Tantivy field value:
/// `rcpt:<hex content_sha256>:<page>:<reading_order>`.
///
/// It names the span by where it lives in the document layer, so it survives
/// a restart, a re-ingest, and any reordering of the lane. A positional index
/// (the earlier `rcpt:<n>`) did none of those — it silently re-pointed at
/// whatever receipt happened to sit at `n`.
///
/// Returns `None` if the receipt's document is not interned in `lane`.
#[must_use]
pub fn handle_for(lane: &TokenLane, r: &TokenStreamReceipt) -> Option<String> {
    use std::fmt::Write as _;
    let sha = lane.document_of(r)?;
    let mut out = String::with_capacity(HANDLE_PREFIX.len() + 64 + 12);
    out.push_str(HANDLE_PREFIX);
    for b in sha {
        let _ = write!(out, "{b:02x}");
    }
    let _ = write!(out, ":{}:{}", r.key.page, r.key.reading_order);
    Some(out)
}

/// Resolve a [`handle_for`] value back to its receipt in `lane`.
#[must_use]
pub fn resolve_handle<'a>(lane: &'a TokenLane, handle: &str) -> Option<&'a TokenStreamReceipt> {
    let mut parts = handle.strip_prefix(HANDLE_PREFIX)?.split(':');
    let hex = parts.next()?;
    let page = parts.next()?.parse().ok()?;
    let reading_order = parts.next()?.parse().ok()?;
    if parts.next().is_some() || hex.len() != 64 {
        return None;
    }
    let mut sha = [0u8; 32];
    for (i, b) in sha.iter_mut().enumerate() {
        *b = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    let doc = lane.document_index(&sha)?;
    lane.receipt_by_key(&SpanKey {
        doc,
        page,
        reading_order,
    })
}

/// A `Tokenizer` that yields the resident lane's ids for a receipt handle, and
/// falls back to encoding the QUERY when handed anything else.
#[derive(Clone)]
pub struct ReceiptTokenizer {
    store: Arc<SeamStore>,
    mode: TermMode,
    token: Token,
}

impl ReceiptTokenizer {
    /// Build over a shared store.
    #[must_use]
    pub fn new(store: Arc<SeamStore>, mode: TermMode) -> Self {
        Self {
            store,
            mode,
            token: Token::default(),
        }
    }
}

/// Where a stream's ids come from.
enum Ids<'a> {
    /// Borrowed straight out of the resident lane — the indexing path.
    Resident(&'a [u8]),
    /// Owned, because this was a query, whose bytes are not in the lane.
    Query(Vec<u8>),
}

impl Ids<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Ids::Resident(s) => s,
            Ids::Query(v) => v.as_slice(),
        }
    }
}

/// The stream. Holds one mutable `Token` buffer, reused for every token.
pub struct ReceiptTokenStream<'a> {
    ids: Ids<'a>,
    contract: &'a TokenizerContract,
    mode: TermMode,
    token: &'a mut Token,
    next: usize,
    cursor: u32,
}

impl Tokenizer for ReceiptTokenizer {
    type TokenStream<'a> = ReceiptTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        // Disjoint field borrows: the token buffer is borrowed mutably while
        // the store is borrowed immutably.
        let Self { store, mode, token } = self;
        token.reset();
        let contract = &store.contract;
        let ids = resolve_handle(&store.lane, text)
            .and_then(|r| store.lane.view(r, contract))
            .map_or_else(
                || {
                    // Not a handle: this is a QUERY. Encoding it is a pass over
                    // QUERY bytes, counted separately from source passes.
                    let owned = contract
                        .try_encode_query(text.as_bytes())
                        .map(|(t, _)| t)
                        .unwrap_or_default();
                    Ids::Query(owned)
                },
                |v| Ids::Resident(v.ids()),
            );
        ReceiptTokenStream {
            ids,
            contract,
            mode: *mode,
            token,
            next: 0,
            cursor: 0,
        }
    }
}

impl TokenStream for ReceiptTokenStream<'_> {
    fn advance(&mut self) -> bool {
        let ids = self.ids.as_slice();
        let Some(&id) = ids.get(self.next) else {
            return false;
        };
        let len = self.contract.byte_len(id);
        self.token.text.clear();
        match self.mode {
            TermMode::Surface => {
                let s = String::from_utf8_lossy(self.contract.surface(id));
                self.token.text.push_str(&s);
            }
            TermMode::TokenId => {
                use std::fmt::Write as _;
                let _ = write!(self.token.text, "{id:02x}");
            }
        }
        self.token.position = self.next;
        self.token.position_length = 1;
        // Offsets are derived here and handed over even though this fork's
        // indexer ignores them: the stream HAS them, and saying so is cheaper
        // than someone later concluding the seam lost them.
        self.token.offset_from = self.cursor as usize;
        self.token.offset_to = (self.cursor + len) as usize;
        self.cursor += len;
        self.next += 1;
        true
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}
