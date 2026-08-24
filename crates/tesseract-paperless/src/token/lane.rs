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
}

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
        if let Some(i) = self.docs.iter().position(|d| *d == content_sha256) {
            return u16::try_from(i).expect("bounded by the check below");
        }
        self.docs.push(content_sha256);
        u16::try_from(self.docs.len() - 1).expect("lane holds <= u16::MAX documents")
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
        self.receipts.push(receipt);
        receipt
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
