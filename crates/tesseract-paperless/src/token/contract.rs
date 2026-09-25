//! The immutable, versioned tokenizer contract — a codebook plus its identity.
//!
//! ```text
//!   A PERSISTED TOKEN ID IS MEANINGLESS WITHOUT THE CODEBOOK THAT ASSIGNED IT.
//!   CONTENT NEVER TRAVELS IN CLASS ADDRESSES.  THE CONTRACT ID IS A FIELD.
//! ```
//!
//! The BPE trainer/encoder/decoder below is carried unchanged in behaviour from
//! `lance-graph` `PROBE-TOKEN-BPE-GEOMETRY-1` (#1012, epiphany
//! `E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1`): base alphabet = the corpus's own
//! distinct bytes, greedy most-frequent-adjacent-pair merges to a 255 cap,
//! deterministic tie-break, `0xFF` reserved as PAD. What is NEW here is
//! everything #1012 deliberately left out of scope for a production carrier:
//!
//! - a **contract id** (a digest over the canonical serialisation of the table
//!   plus the normalisation rule id), so a stored id is interpretable;
//! - a per-id **decoded length** table, which is what makes byte offsets a
//!   derived quantity instead of a stored column;
//! - a per-id **surface** table, which is what lets every downstream projection
//!   run from token ids alone, never re-reading the source.
//!
//! Counters: [`source_passes`] counts tokenizations of SOURCE bytes;
//! [`query_passes`] counts tokenizations of QUERY bytes. They are separate on
//! purpose — a query is different bytes, and hiding it inside one number would
//! make the zero-retokenization gate a lie.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use sha2::{Digest, Sha256};

/// Reserved id: padding inside a particle. Never emitted by encoding.
pub const PAD: u8 = 0xFF;
/// Ids `0..=254` are assignable; `255` is [`PAD`].
pub const VOCAB_CAP: usize = 255;

static SOURCE_PASSES: AtomicUsize = AtomicUsize::new(0);
static QUERY_PASSES: AtomicUsize = AtomicUsize::new(0);
static SOURCE_REFUSALS: AtomicUsize = AtomicUsize::new(0);
static QUERY_REFUSALS: AtomicUsize = AtomicUsize::new(0);

/// How many SOURCE encodes were refused because a byte fell outside the
/// trained alphabet. A refusal is a saturation signal, never a silent empty
/// stream: a lane that drops it would make search silently miss the document.
#[must_use]
pub fn source_refusals() -> usize {
    SOURCE_REFUSALS.load(Ordering::Relaxed)
}

/// How many QUERY encodes were refused for the same reason. Kept apart from
/// [`source_refusals`] so "the archive saturated" and "a user typed a byte the
/// archive never saw" stay distinguishable.
#[must_use]
pub fn query_refusals() -> usize {
    QUERY_REFUSALS.load(Ordering::Relaxed)
}

/// How many times SOURCE bytes have been tokenized in this process.
#[must_use]
pub fn source_passes() -> usize {
    SOURCE_PASSES.load(Ordering::Relaxed)
}

/// How many times QUERY bytes have been tokenized in this process.
#[must_use]
pub fn query_passes() -> usize {
    QUERY_PASSES.load(Ordering::Relaxed)
}

/// What an id expands to.
#[derive(Clone, Copy, Debug)]
pub enum Expansion {
    /// A base alphabet byte.
    Base(u8),
    /// A merge of two earlier ids.
    Pair(u8, u8),
}

/// The normalisation applied to source bytes BEFORE tokenization. Part of the
/// contract identity: the same bytes under a different rule are a different
/// token stream, so the rule id is hashed into the contract id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormRule {
    /// Bytes are taken as-is. Reconstruction is byte-exact.
    Identity,
    /// ASCII-lowercased. Reconstruction is exact against the LOWERCASED text,
    /// which then becomes the canonical text — the authority order is stated
    /// rather than implied.
    AsciiLowercase,
}

impl NormRule {
    const fn id(self) -> u8 {
        match self {
            Self::Identity => 0,
            Self::AsciiLowercase => 1,
        }
    }

    /// Apply the rule, producing the canonical bytes.
    #[must_use]
    pub fn apply(self, src: &[u8]) -> Vec<u8> {
        match self {
            Self::Identity => src.to_vec(),
            Self::AsciiLowercase => src.to_ascii_lowercase(),
        }
    }
}

/// What training observed about the alphabet — the saturation report.
///
/// The cap is 255 assignable ids, base bytes first. A corpus with more distinct
/// bytes than that cannot be represented at all, and one that reaches the cap
/// leaves no room for merges; both are reported here so the condition is
/// visible at training time instead of surfacing later as an empty search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainReport {
    /// Distinct bytes seen in the normalised corpus.
    pub distinct_bytes: usize,
    /// Bytes that did not fit the base alphabet, in first-seen order. Any
    /// later encode containing one of them is refused.
    pub excluded_bytes: Vec<u8>,
    /// Whether the vocabulary reached [`VOCAB_CAP`].
    pub vocab_full: bool,
}

impl TrainReport {
    /// Whether training hit either limit.
    #[must_use]
    pub fn saturated(&self) -> bool {
        self.vocab_full || !self.excluded_bytes.is_empty()
    }
}

/// Why persisted contract bytes were refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractDecodeError {
    /// Not a contract serialisation (magic mismatch).
    BadMagic,
    /// Truncated, or trailing bytes after the table.
    BadLength,
    /// A field held a value no contract can produce.
    Invalid(&'static str),
}

/// A trained, immutable tokenizer codebook plus the identity that makes its
/// output interpretable.
#[derive(Clone, Debug)]
pub struct TokenizerContract {
    expand: Vec<Expansion>,
    base_of: HashMap<u8, u8>,
    merges: Vec<((u8, u8), u8)>,
    /// Decoded surface bytes per id. `<= 255` short byte strings.
    strings: Vec<Vec<u8>>,
    /// `strings[id].len()`, hoisted — the table that makes offsets derived.
    byte_len: Vec<u32>,
    norm: NormRule,
    contract_id: [u8; 32],
}

impl TokenizerContract {
    /// Train on a byte corpus under `norm`.
    ///
    /// # Panics
    /// Never on a non-empty corpus; an empty corpus yields an empty alphabet
    /// and `encode` will then reject every byte (see [`Self::try_encode`]).
    #[must_use]
    pub fn train(corpus: &[u8], norm: NormRule) -> Self {
        Self::train_reported(corpus, norm).0
    }

    /// [`Self::train`], also returning what training observed about the
    /// alphabet. Use this on any archive-scale corpus: a saturated contract
    /// is lawful, but it must be a logged decision, not a discovery.
    ///
    /// # Panics
    /// Never; the base alphabet is capped at [`VOCAB_CAP`] ids so no byte can
    /// be assigned the reserved [`PAD`] id.
    #[must_use]
    pub fn train_reported(corpus: &[u8], norm: NormRule) -> (Self, TrainReport) {
        let corpus = norm.apply(corpus);
        let corpus = corpus.as_slice();
        let mut base_of: HashMap<u8, u8> = HashMap::new();
        let mut expand: Vec<Expansion> = Vec::new();
        let mut strings: Vec<Vec<u8>> = Vec::new();
        let mut seen = [false; 256];
        let mut report = TrainReport::default();
        for &b in corpus {
            if std::mem::replace(&mut seen[b as usize], true) {
                continue;
            }
            report.distinct_bytes += 1;
            // Ids 0..=254 are assignable; a 256th distinct byte would land on
            // PAD and vanish on decode, so it is excluded and reported instead.
            if expand.len() >= VOCAB_CAP {
                report.excluded_bytes.push(b);
                continue;
            }
            expand.push(Expansion::Base(b));
            strings.push(vec![b]);
            base_of.insert(
                b,
                u8::try_from(expand.len() - 1).expect("bounded by VOCAB_CAP"),
            );
        }
        // Excluded bytes are dropped from the training stream only; they can
        // never be encoded, and every encode containing one is refused.
        let mut stream: Vec<u8> = corpus
            .iter()
            .filter_map(|b| base_of.get(b).copied())
            .collect();
        let mut merges = Vec::new();
        while expand.len() < VOCAB_CAP {
            let mut pf: HashMap<(u8, u8), usize> = HashMap::new();
            for w in stream.windows(2) {
                *pf.entry((w[0], w[1])).or_default() += 1;
            }
            // Deterministic tie-break (count desc, then pair asc): the table is
            // reproducible from the corpus alone, with no external state.
            let Some((&pair, &count)) = pf
                .iter()
                .max_by_key(|(&(a, b), &c)| (c, std::cmp::Reverse((a, b))))
            else {
                break;
            };
            if count < 2 {
                break;
            }
            let id = u8::try_from(expand.len()).expect("bounded by VOCAB_CAP");
            expand.push(Expansion::Pair(pair.0, pair.1));
            let mut s = strings[pair.0 as usize].clone();
            s.extend_from_slice(&strings[pair.1 as usize]);
            strings.push(s);
            merges.push((pair, id));
            let mut out = Vec::with_capacity(stream.len());
            let mut i = 0;
            while i < stream.len() {
                if i + 1 < stream.len() && (stream[i], stream[i + 1]) == pair {
                    out.push(id);
                    i += 2;
                } else {
                    out.push(stream[i]);
                    i += 1;
                }
            }
            stream = out;
        }
        let byte_len = strings
            .iter()
            .map(|s| u32::try_from(s.len()).expect("token surface is short"))
            .collect();
        let mut me = Self {
            expand,
            base_of,
            merges,
            strings,
            byte_len,
            norm,
            contract_id: [0; 32],
        };
        me.contract_id = me.compute_contract_id();
        report.vocab_full = me.expand.len() >= VOCAB_CAP;
        (me, report)
    }

    /// The persisted form: exactly the canonical serialisation the contract id
    /// digests. Persisting the identity's own preimage means a reload can
    /// PROVE it restored the same codebook by recomputing the id.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.canonical_bytes()
    }

    /// Rebuild a contract from [`Self::to_bytes`]. Every derived table
    /// (`base_of`, `merges`, `strings`, `byte_len`) is recomputed from the
    /// expansion list, and the id is recomputed from the bytes, so a restored
    /// contract cannot disagree with the one that was saved.
    ///
    /// # Errors
    /// [`ContractDecodeError`] on a wrong magic, a length mismatch, an unknown
    /// normalisation rule, a pair referring forward, or a duplicate base byte.
    ///
    /// # Panics
    /// Never: every slice conversion is on a length checked just above it.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContractDecodeError> {
        const HEADER: usize = 8 + 1 + 1 + 4;
        if bytes.len() < HEADER {
            return Err(ContractDecodeError::BadLength);
        }
        if &bytes[..8] != b"PLTOKC01" {
            return Err(ContractDecodeError::BadMagic);
        }
        let norm = match bytes[8] {
            0 => NormRule::Identity,
            1 => NormRule::AsciiLowercase,
            _ => return Err(ContractDecodeError::Invalid("normalisation rule")),
        };
        if usize::from(bytes[9]) != VOCAB_CAP {
            return Err(ContractDecodeError::Invalid("vocabulary cap"));
        }
        let n = u32::from_le_bytes(bytes[10..14].try_into().expect("4 bytes"));
        let n = usize::try_from(n).map_err(|_| ContractDecodeError::BadLength)?;
        if n > VOCAB_CAP || bytes.len() != HEADER + n * 3 {
            return Err(ContractDecodeError::BadLength);
        }
        let mut expand = Vec::with_capacity(n);
        let mut base_of = HashMap::new();
        let mut merges = Vec::new();
        let mut strings: Vec<Vec<u8>> = Vec::with_capacity(n);
        for (i, e) in bytes[HEADER..].as_chunks::<3>().0.iter().enumerate() {
            let id = u8::try_from(i).expect("n <= VOCAB_CAP");
            match e[0] {
                0 => {
                    if base_of.insert(e[1], id).is_some() {
                        return Err(ContractDecodeError::Invalid("duplicate base byte"));
                    }
                    expand.push(Expansion::Base(e[1]));
                    strings.push(vec![e[1]]);
                }
                1 => {
                    let (l, r) = (e[1], e[2]);
                    if l >= id || r >= id {
                        return Err(ContractDecodeError::Invalid("pair refers forward"));
                    }
                    let mut s = strings[l as usize].clone();
                    s.extend_from_slice(&strings[r as usize]);
                    strings.push(s);
                    expand.push(Expansion::Pair(l, r));
                    merges.push(((l, r), id));
                }
                _ => return Err(ContractDecodeError::Invalid("expansion tag")),
            }
        }
        let byte_len = strings
            .iter()
            .map(|s| u32::try_from(s.len()).expect("token surface is short"))
            .collect();
        let mut me = Self {
            expand,
            base_of,
            merges,
            strings,
            byte_len,
            norm,
            contract_id: [0; 32],
        };
        me.contract_id = me.compute_contract_id();
        Ok(me)
    }

    /// Whether the vocabulary is at [`VOCAB_CAP`] — no merge can be added.
    #[must_use]
    pub fn vocab_full(&self) -> bool {
        self.expand.len() >= VOCAB_CAP
    }

    /// Whether every byte of `q` (after normalisation) is in the trained
    /// alphabet. `Err(offset)` names the first byte that is not — the signal a
    /// search response carries so "no match" and "unencodable query" are never
    /// the same empty result.
    ///
    /// # Errors
    /// The byte offset (in the normalised query) of the first unknown byte.
    pub fn covers(&self, q: &[u8]) -> Result<(), usize> {
        match self
            .norm
            .apply(q)
            .iter()
            .position(|b| !self.base_of.contains_key(b))
        {
            None => Ok(()),
            Some(i) => Err(i),
        }
    }

    /// The canonical serialisation the contract id digests. Stated explicitly so
    /// the identity is reproducible from the table, not from allocation order.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.expand.len() * 4);
        out.extend_from_slice(b"PLTOKC01");
        out.push(self.norm.id());
        out.push(u8::try_from(VOCAB_CAP).expect("255"));
        out.extend_from_slice(
            &u32::try_from(self.expand.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for e in &self.expand {
            match *e {
                Expansion::Base(b) => {
                    out.push(0);
                    out.push(b);
                    out.push(0);
                }
                Expansion::Pair(l, r) => {
                    out.push(1);
                    out.push(l);
                    out.push(r);
                }
            }
        }
        out
    }

    fn compute_contract_id(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        h.finalize().into()
    }

    /// The contract identity. Two contracts with the same id assign the same
    /// meaning to the same id; two with different ids do not.
    #[must_use]
    pub const fn contract_id(&self) -> [u8; 32] {
        self.contract_id
    }

    /// The contract id, hex, for reports and receipts.
    #[must_use]
    pub fn contract_hex(&self) -> String {
        use std::fmt::Write as _;
        self.contract_id.iter().fold(String::new(), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
    }

    /// The normalisation rule this contract was trained under.
    #[must_use]
    pub const fn norm(&self) -> NormRule {
        self.norm
    }

    /// Number of assigned ids.
    #[must_use]
    pub fn vocab_len(&self) -> usize {
        self.expand.len()
    }

    /// Number of merges.
    #[must_use]
    pub fn merge_len(&self) -> usize {
        self.merges.len()
    }

    /// The decoded surface of one id.
    #[must_use]
    pub fn surface(&self, id: u8) -> &[u8] {
        &self.strings[id as usize]
    }

    /// The decoded byte length of one id. This is the whole reason the receipt
    /// carries no offset column: an offset is a prefix sum over this table.
    #[must_use]
    pub fn byte_len(&self, id: u8) -> u32 {
        self.byte_len[id as usize]
    }

    /// Encode SOURCE bytes. Counted by [`source_passes`].
    ///
    /// Returns `None` if the input contains a byte outside the trained
    /// alphabet — a contract is only valid for the alphabet it was trained on,
    /// and silently dropping an unknown byte would break reconstruction.
    pub fn try_encode(&self, src: &[u8]) -> Option<(Vec<u8>, usize)> {
        SOURCE_PASSES.fetch_add(1, Ordering::Relaxed);
        let out = self.encode_inner(src);
        if out.is_none() {
            SOURCE_REFUSALS.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    /// Encode QUERY bytes. Counted by [`query_passes`], never by
    /// [`source_passes`].
    pub fn try_encode_query(&self, q: &[u8]) -> Option<(Vec<u8>, usize)> {
        QUERY_PASSES.fetch_add(1, Ordering::Relaxed);
        let out = self.encode_inner(q);
        if out.is_none() {
            QUERY_REFUSALS.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn encode_inner(&self, src: &[u8]) -> Option<(Vec<u8>, usize)> {
        let norm = self.norm.apply(src);
        let mut stream: Vec<u8> = Vec::with_capacity(norm.len());
        for b in &norm {
            stream.push(*self.base_of.get(b)?);
        }
        let mut probes = 0usize;
        for &(pair, id) in &self.merges {
            let mut out = Vec::with_capacity(stream.len());
            let mut i = 0;
            while i < stream.len() {
                probes += 1;
                if i + 1 < stream.len() && (stream[i], stream[i + 1]) == pair {
                    out.push(id);
                    i += 2;
                } else {
                    out.push(stream[i]);
                    i += 1;
                }
            }
            stream = out;
        }
        Some((stream, probes))
    }

    /// Decode ids to canonical bytes. `PAD` is skipped, so a decode is only as
    /// correct as the framing that told it where to stop — see
    /// [`crate::token::lane`].
    #[must_use]
    pub fn decode(&self, tokens: &[u8]) -> (Vec<u8>, usize) {
        let mut out = Vec::new();
        let mut steps = 0usize;
        let mut stack: Vec<u8> = Vec::new();
        for &t in tokens {
            if t == PAD {
                continue;
            }
            stack.push(t);
            while let Some(id) = stack.pop() {
                steps += 1;
                match self.expand[id as usize] {
                    Expansion::Base(b) => out.push(b),
                    Expansion::Pair(l, r) => {
                        stack.push(r);
                        stack.push(l);
                    }
                }
            }
        }
        (out, steps)
    }
}
