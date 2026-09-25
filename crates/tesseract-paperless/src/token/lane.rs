//! The resident token lane and its framing.
//!
//! ```text
//!   THE POPULATION DOES NOT MOVE.  THE VIEW DOES.
//!   TOKEN_COUNT IS AUTHORITATIVE.  PAD IS NOT A LENGTH.
//! ```
//!
//! The lane is a flat `Vec<[u8; 12]>` — the V3 content-blind 12-byte payload
//! (`lance_graph_contract::facet::FacetCascade` carries exactly this shape:
//! `facet_classid(4) + 6 x FacetTier{lo,hi} = 16 B`, size-asserted). Twelve
//! `u8` token ids per particle, two per `(8:8)` tier, never widened to `u16`.
//!
//! # Framing: what #1012 left open
//!
//! `PROBE-TOKEN-BPE-GEOMETRY-1` measured that EVERY verse of its corpus needed
//! more than one particle (p50 = 4, max = 8), and refused to pick a framing
//! mechanism. There is no shipped token continuation field anywhere in
//! `lance-graph`; the nearest precedent in shape is
//! `rail_geometry::RailCarving::AxisSlab { reg, cont: Option<usize> }`, which
//! chains ONE register to ONE possibly-discontiguous continuation register and
//! therefore caps at `RAIL_MAX_DEPTH = 24` levels. That cap is too short here
//! by construction — a 12-token cap doubled is still under the measured p50 of
//! 4 particles — so this crate takes the other lawful shape: a **contiguous
//! run** described by `first_particle + particle_count`, with `token_count` as
//! the authority.
//!
//! `PAD` fills only the tail of the LAST particle of a run and is never
//! consulted to find a length: a run whose token count is an exact multiple of
//! 12 contains no PAD at all, and inferring its end from padding would read
//! straight into the next receipt. The probe exercises exactly that case.

use std::collections::HashMap;

use crate::token::contract::{TokenizerContract, PAD};
use crate::token::docir::SpanKey;

/// Ids per particle: the 12-byte payload, one `u8` per byte.
pub const IDS_PER_PARTICLE: usize = 12;

/// The same constant where a receipt's 32-bit fields need it. Spelled out
/// rather than cast, so no `as` conversion appears on a framing path.
pub const IDS_PER_PARTICLE_U32: u32 = 12;

/// The resident particle: the V3 content-blind 12-byte payload.
pub type TokenParticle = [u8; IDS_PER_PARTICLE];

/// What one tokenization produced, and everything needed to read it back.
///
/// This is the RECEIPT. It carries no bytes and no offsets: the ids live in the
/// lane, and offsets are a prefix sum over the contract's per-id length table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenStreamReceipt {
    /// WHERE, in the document layer's own address space — never an id this
    /// crate minted. See [`crate::token::docir`].
    pub key: SpanKey,
    /// Which codebook assigned these ids. Without it they are meaningless.
    pub tokenizer_contract_id: [u8; 32],
    /// Byte offset of the span within its REGION's canonical text. A whole
    /// region is 0; the field exists because a sub-region span is lawful and
    /// would not be.
    pub byte_from: u32,
    /// AUTHORITATIVE token count. Not derivable from padding.
    pub token_count: u32,
    /// Index of the first particle of the run.
    pub first_particle: u32,
    /// Number of particles in the run: `ceil(token_count / 12)`.
    pub particle_count: u32,
}

impl TokenStreamReceipt {
    /// Whether the run's tail is exactly full — the case where PAD-inference
    /// would silently read into the next receipt.
    #[must_use]
    pub const fn tail_is_full(&self) -> bool {
        self.token_count.is_multiple_of(IDS_PER_PARTICLE_U32) && self.token_count != 0
    }
}

/// The resident population: particles, plus the receipts that frame them.
///
/// Nothing here owns text. The canonical source text stays authoritative and
/// lives outside; this lane holds ids and framing only.
#[derive(Clone, Debug, Default)]
pub struct TokenLane {
    particles: Vec<TokenParticle>,
    receipts: Vec<TokenStreamReceipt>,
    /// `content_sha256` per document, interned once. A receipt carries a
    /// `u16` index into this, not the hash itself.
    docs: Vec<[u8; 32]>,
    /// `content_sha256 -> docs index`. Derived; rebuilt on load, never saved.
    doc_index: HashMap<[u8; 32], u16>,
    /// `SpanKey -> receipts index`, latest append wins. Derived like
    /// `doc_index`. This is what makes a receipt addressable by WHERE it is in
    /// the document layer rather than by where it happens to sit in the lane.
    key_index: HashMap<SpanKey, usize>,
}

/// Why persisted lane bytes were refused. A lane that loads is a lane whose
/// every receipt frames a run inside its own particles — nothing is trusted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneDecodeError {
    /// Not a lane serialisation (magic mismatch).
    BadMagic,
    /// Truncated, or trailing bytes.
    BadLength,
    /// A receipt that cannot be framed against this lane.
    BadReceipt(usize),
    /// The same `content_sha256` twice in the document table.
    DuplicateDocument(usize),
}

const LANE_MAGIC: &[u8; 8] = b"PLTOKL01";
/// Serialised receipt: doc, page, reading order (3 x u16), contract id (32),
/// `byte_from`, `token_count`, `first_particle`, `particle_count` (4 x u32).
const RECEIPT_BYTES: usize = 6 + 32 + 16;

impl TokenLane {
    /// Empty lane.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a document's `content_sha256`, returning the index a
    /// [`SpanKey`] addresses it by. Re-interning the same hash returns the
    /// same index — which IS the S-2 dedup property, at lane scope: the same
    /// bytes acquired twice are one document here, not two.
    ///
    /// # Panics
    /// If a lane accumulates more than `u16::MAX` documents.
    pub fn intern_document(&mut self, content_sha256: [u8; 32]) -> u16 {
        if let Some(&i) = self.doc_index.get(&content_sha256) {
            return i;
        }
        let i = u16::try_from(self.docs.len()).expect("lane holds <= u16::MAX documents");
        self.docs.push(content_sha256);
        self.doc_index.insert(content_sha256, i);
        i
    }

    /// The document-table index of `content_sha256`, if it was interned.
    #[must_use]
    pub fn document_index(&self, content_sha256: &[u8; 32]) -> Option<u16> {
        self.doc_index.get(content_sha256).copied()
    }

    /// The receipt addressed by `key`, if one was appended. When the same key
    /// was appended more than once, the latest append is the one returned.
    #[must_use]
    pub fn receipt_by_key(&self, key: &SpanKey) -> Option<&TokenStreamReceipt> {
        self.key_index.get(key).and_then(|&i| self.receipts.get(i))
    }

    /// The `content_sha256` a receipt's key addresses.
    #[must_use]
    pub fn document_of(&self, r: &TokenStreamReceipt) -> Option<&[u8; 32]> {
        self.docs.get(r.key.doc as usize)
    }

    /// Documents interned in this lane.
    #[must_use]
    pub fn document_len(&self) -> usize {
        self.docs.len()
    }

    /// Append one tokenized span. The ids are packed 12 per particle with a PAD
    /// tail; `token_count` is recorded because the tail is not a length.
    ///
    /// # Panics
    /// If the lane or the span exceeds `u32::MAX` — a receipt addresses the
    /// lane with 32-bit fields by design, and a silent wrap there would be a
    /// mis-framed span rather than a large one.
    pub fn append(
        &mut self,
        key: SpanKey,
        byte_from: u32,
        contract: &TokenizerContract,
        tokens: &[u8],
    ) -> TokenStreamReceipt {
        let first_particle = u32::try_from(self.particles.len()).expect("lane fits u32");
        for chunk in tokens.chunks(IDS_PER_PARTICLE) {
            let mut p = [PAD; IDS_PER_PARTICLE];
            p[..chunk.len()].copy_from_slice(chunk);
            self.particles.push(p);
        }
        let token_count = u32::try_from(tokens.len()).expect("span fits u32");
        let receipt = TokenStreamReceipt {
            key,
            tokenizer_contract_id: contract.contract_id(),
            byte_from,
            token_count,
            first_particle,
            particle_count: u32::try_from(self.particles.len()).expect("lane fits u32")
                - first_particle,
        };
        self.key_index.insert(key, self.receipts.len());
        self.receipts.push(receipt);
        receipt
    }

    /// The persisted form: documents, particles, receipts, little-endian.
    /// The derived indexes are not written; [`Self::from_bytes`] rebuilds them.
    ///
    /// # Panics
    /// Never for a lane built through [`Self::append`], whose counts already
    /// fit `u32`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            8 + 12
                + self.docs.len() * 32
                + self.particles.len() * IDS_PER_PARTICLE
                + self.receipts.len() * RECEIPT_BYTES,
        );
        out.extend_from_slice(LANE_MAGIC);
        for n in [self.docs.len(), self.particles.len(), self.receipts.len()] {
            out.extend_from_slice(&u32::try_from(n).expect("lane fits u32").to_le_bytes());
        }
        for d in &self.docs {
            out.extend_from_slice(d);
        }
        out.extend_from_slice(self.particles.as_flattened());
        for r in &self.receipts {
            out.extend_from_slice(&r.key.doc.to_le_bytes());
            out.extend_from_slice(&r.key.page.to_le_bytes());
            out.extend_from_slice(&r.key.reading_order.to_le_bytes());
            out.extend_from_slice(&r.tokenizer_contract_id);
            for v in [
                r.byte_from,
                r.token_count,
                r.first_particle,
                r.particle_count,
            ] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Rebuild a lane from [`Self::to_bytes`], validating every receipt's
    /// framing: its document exists, its run lies inside the particles, and
    /// `particle_count == ceil(token_count / 12)`.
    ///
    /// # Errors
    /// [`LaneDecodeError`] on any structural mismatch.
    ///
    /// # Panics
    /// Never: every slice conversion is on a length checked just above it.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LaneDecodeError> {
        let rd_u32 = |at: usize| -> Option<u32> {
            bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        };
        if bytes.len() < 20 {
            return Err(LaneDecodeError::BadLength);
        }
        if &bytes[..8] != LANE_MAGIC {
            return Err(LaneDecodeError::BadMagic);
        }
        let count = |at: usize| -> Result<usize, LaneDecodeError> {
            rd_u32(at)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(LaneDecodeError::BadLength)
        };
        let (n_docs, n_parts, n_rcpts) = (count(8)?, count(12)?, count(16)?);
        let docs_at = 20usize;
        let parts_at = n_docs
            .checked_mul(32)
            .and_then(|n| n.checked_add(docs_at))
            .ok_or(LaneDecodeError::BadLength)?;
        let rcpts_at = n_parts
            .checked_mul(IDS_PER_PARTICLE)
            .and_then(|n| n.checked_add(parts_at))
            .ok_or(LaneDecodeError::BadLength)?;
        let end = n_rcpts
            .checked_mul(RECEIPT_BYTES)
            .and_then(|n| n.checked_add(rcpts_at))
            .ok_or(LaneDecodeError::BadLength)?;
        if bytes.len() != end || n_docs > usize::from(u16::MAX) + 1 {
            return Err(LaneDecodeError::BadLength);
        }
        let mut lane = Self::new();
        for (i, d) in bytes[docs_at..parts_at]
            .as_chunks::<32>()
            .0
            .iter()
            .enumerate()
        {
            if lane.document_index(d).is_some() {
                return Err(LaneDecodeError::DuplicateDocument(i));
            }
            lane.intern_document(*d);
        }
        lane.particles = bytes[parts_at..rcpts_at]
            .as_chunks::<IDS_PER_PARTICLE>()
            .0
            .to_vec();
        for (i, r) in bytes[rcpts_at..end]
            .as_chunks::<RECEIPT_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let u16_at = |at: usize| u16::from_le_bytes([r[at], r[at + 1]]);
            let u32_at = |at: usize| u32::from_le_bytes(r[at..at + 4].try_into().expect("4 bytes"));
            let receipt = TokenStreamReceipt {
                key: SpanKey {
                    doc: u16_at(0),
                    page: u16_at(2),
                    reading_order: u16_at(4),
                },
                tokenizer_contract_id: r[6..38].try_into().expect("32 bytes"),
                byte_from: u32_at(38),
                token_count: u32_at(42),
                first_particle: u32_at(46),
                particle_count: u32_at(50),
            };
            let framed = usize::from(receipt.key.doc) < lane.docs.len()
                && receipt.particle_count == receipt.token_count.div_ceil(IDS_PER_PARTICLE_U32)
                && (receipt.first_particle as usize)
                    .checked_add(receipt.particle_count as usize)
                    .is_some_and(|e| e <= lane.particles.len());
            if !framed {
                return Err(LaneDecodeError::BadReceipt(i));
            }
            lane.key_index.insert(receipt.key, lane.receipts.len());
            lane.receipts.push(receipt);
        }
        Ok(lane)
    }

    /// Every receipt, in append order.
    #[must_use]
    pub fn receipts(&self) -> &[TokenStreamReceipt] {
        &self.receipts
    }

    /// The raw resident particles. Exposed so a probe can demonstrate what a
    /// PAD-scan would read — the framing falsifier needs to see past the
    /// receipt's own boundary to prove that `token_count` is load-bearing.
    #[must_use]
    pub fn particles(&self) -> &[TokenParticle] {
        &self.particles
    }

    /// Resident particle count.
    #[must_use]
    pub fn particle_len(&self) -> usize {
        self.particles.len()
    }

    /// Bytes owned by the resident lane (particles + receipts).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.particles.len() * IDS_PER_PARTICLE
            + self.receipts.len() * core::mem::size_of::<TokenStreamReceipt>()
            + self.docs.len() * 32
    }

    /// A BORROWED view of one receipt's ids. No copy, no allocation: this is a
    /// slice of the resident population, trimmed by the authoritative
    /// `token_count` rather than by looking for PAD.
    ///
    /// Returns `None` if the contract does not match the receipt — an id is
    /// only interpretable under the codebook that assigned it.
    #[must_use]
    pub fn view<'a>(
        &'a self,
        r: &TokenStreamReceipt,
        contract: &'a TokenizerContract,
    ) -> Option<TokenStreamView<'a>> {
        if r.tokenizer_contract_id != contract.contract_id() {
            return None;
        }
        let start = r.first_particle as usize;
        let end = start + r.particle_count as usize;
        let flat = self.particles.get(start..end)?;
        // SAFETY-free reinterpretation: [[u8;12]] is contiguous, so a flat id
        // slice is a borrow, not a copy. `as_flattened` keeps it in safe Rust.
        let ids = &flat.as_flattened()[..r.token_count as usize];
        Some(TokenStreamView {
            ids,
            contract,
            byte_from: r.byte_from,
        })
    }
}

/// One token as the view yields it. Offsets are DERIVED during the walk from
/// the contract's per-id length table; nothing stored them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenRef {
    /// Position in the span, in tokens.
    pub position: u32,
    /// The id.
    pub id: u8,
    /// Byte offset of the token in the source's canonical text.
    pub byte_from: u32,
    /// End byte offset (exclusive).
    pub byte_to: u32,
}

/// A borrowed window onto one receipt's ids. Holds no owned token data.
#[derive(Clone, Copy, Debug)]
pub struct TokenStreamView<'a> {
    ids: &'a [u8],
    contract: &'a TokenizerContract,
    byte_from: u32,
}

impl<'a> TokenStreamView<'a> {
    /// The borrowed id slice — the input surface a forward predictor consumes.
    #[must_use]
    pub const fn ids(&self) -> &'a [u8] {
        self.ids
    }

    /// The contract these ids are read under.
    #[must_use]
    pub const fn contract(&self) -> &'a TokenizerContract {
        self.contract
    }

    /// Token count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the span is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Walk the tokens, deriving offsets by prefix sum. Allocation-free.
    ///
    /// # Panics
    /// If a span exceeds `u32::MAX` tokens; see [`TokenLane::append`].
    pub fn tokens(&self) -> impl Iterator<Item = TokenRef> + '_ {
        let mut cursor = self.byte_from;
        self.ids.iter().enumerate().map(move |(i, &id)| {
            let from = cursor;
            cursor += self.contract.byte_len(id);
            TokenRef {
                position: u32::try_from(i).expect("span fits u32"),
                id,
                byte_from: from,
                byte_to: cursor,
            }
        })
    }

    /// Reconstruct the span's canonical bytes from the ids alone.
    #[must_use]
    pub fn decode(&self) -> Vec<u8> {
        self.contract.decode(self.ids).0
    }
}
