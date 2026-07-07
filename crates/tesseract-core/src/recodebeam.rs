//! `RecodeBeamSearch::Decode` — recognizer **Leaf 7b**, the CTC beam search that
//! turns the LSTM's per-timestep softmax logits into a code sequence (→ text via
//! [`recoded_to_text`](crate::recoded_to_text)). Transcode of Tesseract's
//! `lstm/recodebeam.{h,cpp}` — the **non-dictionary** path (`dict_ == nullptr`,
//! `permuter` fixed at `TOP_CHOICE_PERM`, the dawg beams never populated).
//!
//! This is **compute**, but SIMD-free and recoder-coupled, so it lives in the
//! OCR core next to the recoder tables it walks (the Core's [`UnicharCompress`]
//! beam maps proven in Leaf 7a: `is_valid_first_code` / `get_final_codes` /
//! `get_next_codes`) and the [`recoded_to_text`](crate::recoded_to_text) step it
//! feeds — not in `tesseract-recognizer` (which is the ndarray SIMD forward).
//!
//! ## Why a beam, not a greedy argmax
//!
//! CTC says the probability of a decoding is the SUM over all timestep paths that
//! fold to it. A greedy per-timestep argmax is NOT byte-parity with Tesseract
//! (`recodebeam.h:37-70`): the beam places `X`, `X+Null`, `X+Y+Null` combinations
//! under constrained continuation rules ([`NodeContinuation`]) to recover a much
//! better minimum certainty. The lattice is `beam_[t].beams_[k]` — a
//! `2·NC_COUNT·kNumLengths = 60`-way split (dawg×continuation×code-length) of
//! [`MinHeap`]s, each keeping the `kBeamWidths` best by score.
//!
//! ## Byte-parity surface
//!
//! [`RecodeBeamSearch::extract_best_path_as_labels`] mirrors the public C++
//! `ExtractBestPathAsLabels`. The oracle (`/tmp/recodebeam_oracle.cpp`) constructs
//! a real libtesseract `RecodeBeamSearch(recoder, null_char, simple_text, nullptr)`,
//! calls the public `Decode(GENERIC_2D_ARRAY<float>, 1.0, 0.0, 0.0, nullptr)` on
//! the SAME synthetic softmax matrix (read from a shared `.bin`), and dumps the
//! same labels + xcoords — no private-member access, so the 5.5.0-header /
//! 5.3.4-lib ABI skew cannot bite. eng.lstm's recoder is pass-through (all
//! length-1 → `next_codes_` empty, every beam at length 0), so this proves the
//! CTC core for a simple (non-CJK) script; the multi-code `next_codes_` trie is
//! Han/Hangul, out of `eng` scope (consistent with every prior leaf).
//!
//! ## The float contract
//!
//! Scores accumulate in **f32** (`score = cert + prev.score`, matching FAST_FLOAT
//! `TFloat = float`); `ProbToCertainty(p) = p > e^-20 ? ln(p) : -20` is a raw
//! `ln` (NOT the Leaf-3 activation LUT). The heap keys are `f64::from(score)`
//! (lossless), so the heap order equals the f32 score order. `dict_ratio = 1.0`
//! and `cert_offset = 0.0` for a plain decode (they cancel identically on both
//! sides).

use lance_graph_contract::dawg::PermuterType;
use lance_graph_contract::unicharcompress::{RecodedCharId, UnicharCompress};

use crate::{DawgPosition, DictLite, UniCharSet};

/// `RecodedCharID::kMaxCodeLen` (`unicharcompress.h:35`).
const K_MAX_CODE_LEN: usize = 9;
/// `kNumLengths = kMaxCodeLen + 1` (`recodebeam.h:245`) — one beam per code length.
const K_NUM_LENGTHS: usize = K_MAX_CODE_LEN + 1;
/// `NC_COUNT` (`recodebeam.h:80`) — the three [`NodeContinuation`] kinds.
const NC_COUNT: usize = 3;
/// `kNumBeams = 2·NC_COUNT·kNumLengths` (`recodebeam.h:248`) — dawg×cont×length.
const K_NUM_BEAMS: usize = 2 * NC_COUNT * K_NUM_LENGTHS;
/// `kBeamWidths` (`recodebeam.cpp:31`) — the per-length beam capacity.
const K_BEAM_WIDTHS: [usize; K_NUM_LENGTHS] = [5, 10, 16, 16, 16, 16, 16, 16, 16, 16];
/// `kMinCertainty` (`recodebeam.h:243` / `networkio.cpp:30`) — the certainty floor.
const K_MIN_CERTAINTY: f32 = -20.0;
/// `INVALID_UNICHAR_ID` (`unichar.h`).
const INVALID_UNICHAR_ID: i32 = -1;

/// Return shape of
/// [`RecodeBeamSearch::extract_path_as_unichar_ids_with_boundaries`]:
/// `(unichar_ids, certs, ratings, xcoords, character_boundaries)` — the same
/// 4-tuple as [`RecodeBeamSearch::extract_best_path_as_unichar_ids`] plus the
/// character-boundary x-coordinates.
type UnicharIdsWithBoundaries = (Vec<i32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>);

/// `NodeContinuation::NC_ANYTHING` (`recodebeam.h:73`): this node used only its
/// own score, so anything may follow.
const NC_ANYTHING: usize = 0;
/// `NC_ONLY_DUP` (`recodebeam.h:74`): combined a score without a stand-alone
/// duplicate before, so must be followed by a stand-alone duplicate.
const NC_ONLY_DUP: usize = 1;
/// `NC_NO_DUP` (`recodebeam.h:77`): combined a score after a stand-alone, so can
/// only be followed by a non-duplicate.
const NC_NO_DUP: usize = 2;

/// `PermuterType::TOP_CHOICE_PERM` (`ratngs.h:238`). The non-dict path never uses
/// any other permuter, so the concrete value only ever participates in the
/// (always equal) [`RecodeBeamSearch::update_heap_if_matched`] identity check and
/// the `!= NO_PERM` space test in the unichar extract. The dict path (D1.3) uses
/// the full [`PermuterType`] range via [`DictLite::def_letter_is_okay`].
const TOP_CHOICE_PERM: PermuterType = PermuterType::TopChoicePerm;

/// `PermuterType::NO_PERM` (`ratngs.h:236`) — in the full engine, the marker a
/// dict-path space carries so the unichar extract "forgets" preceding null
/// certainty (`recodebeam.cpp:609-613`). Non-dict nodes are never `NO_PERM`.
const NO_PERM: PermuterType = PermuterType::NoPerm;

/// `UNICHAR_SPACE` (`unicharset.h:36`) — unichar-id 0 is the space in every
/// unicharset; the unichar extract folds a space's leading-null certainty into
/// the PREVIOUS character (`recodebeam.cpp:594-604`).
const UNICHAR_SPACE: i32 = 0;

/// The top-n classification of a code at the current timestep
/// (`TopNState`, `recodebeam.h:84`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TopN {
    /// Winner or second (`TN_TOP2`).
    Top2,
    /// In the top-n but not first or second (`TN_TOPN`).
    Runner,
    /// Not in the top-n (`TN_ALSO_RAN`).
    AlsoRan,
}

/// `NetworkIO::ProbToCertainty` (`networkio.cpp:580`): `log(prob)` clamped at the
/// certainty floor. `kMinProb = e^kMinCertainty` (`networkio.cpp:32`).
#[must_use]
fn prob_to_certainty(prob: f32) -> f32 {
    let k_min_prob = K_MIN_CERTAINTY.exp();
    if prob > k_min_prob {
        prob.ln()
    } else {
        K_MIN_CERTAINTY
    }
}

/// `UNICHARSET::IsSpaceDelimited` (`unicharset.h:668-676`) — true unless the
/// unichar's script is one of the five CJK/Thai scripts that don't delimit words
/// with spaces. `INVALID_UNICHAR_ID` is vacuously space-delimited. Consumed by
/// [`RecodeBeamSearch::continue_unichar`]/[`RecodeBeamSearch::continue_dawg`]
/// (D1.3); implemented here (not in the Core) as a pure consumer of the already-
/// proven [`UniCharSet::get_script`]/[`UniCharSet::script_from_script_id`]
/// (`E-CPP-PARITY-4`) — no new Core primitive needed.
#[must_use]
fn is_space_delimited(charset: &UniCharSet, unichar_id: i32) -> bool {
    if unichar_id == INVALID_UNICHAR_ID {
        return true;
    }
    let script_id = charset.get_script(unichar_id as u32);
    !matches!(
        charset.script_from_script_id(script_id),
        Some("Han" | "Thai" | "Hangul" | "Hiragana" | "Katakana")
    )
}

/// `Dict::IsSpaceDelimitedLang` (`dict.cpp:913-925`) — the language-level check
/// (computed once, at construction, unlike the per-char [`is_space_delimited`]):
/// true unless the charset registers a Han, Katakana, or Thai script anywhere.
/// `eng.lstm-unicharset` registers none of the three, so this is `true` for eng
/// (a constant from [`RecodeBeamSearch::new_with_dict`]'s perspective, but
/// computed generically here rather than hardcoded).
#[must_use]
fn is_space_delimited_lang(charset: &UniCharSet) -> bool {
    !(0..charset.get_script_table_size()).any(|i| {
        matches!(
            charset.script_from_script_id(i as i32),
            Some("Han" | "Katakana" | "Thai")
        )
    })
}

/// `BeamIndex(is_dawg, cont, length)` (`recodebeam.h:260`).
#[must_use]
fn beam_index(is_dawg: bool, cont: usize, length: usize) -> usize {
    (usize::from(is_dawg) * NC_COUNT + cont) * K_NUM_LENGTHS + length
}
/// `LengthFromBeamsIndex` (`recodebeam.h:250`).
#[must_use]
fn length_from_index(index: usize) -> usize {
    index % K_NUM_LENGTHS
}
/// `ContinuationFromBeamsIndex` (`recodebeam.h:253`).
#[must_use]
fn cont_from_index(index: usize) -> usize {
    (index / K_NUM_LENGTHS) % NC_COUNT
}
/// `IsDawgFromBeamsIndex` (`recodebeam.h:256`).
#[must_use]
fn is_dawg_from_index(index: usize) -> bool {
    index / (K_NUM_LENGTHS * NC_COUNT) > 0
}

/// A lattice node — the transcode of `RecodeNode` (`recodebeam.h:92`). `prev` is
/// an **arena index** (not a raw pointer): all nodes across all timesteps live in
/// one [`RecodeBeamSearch::arena`] `Vec`, so the borrowed-`prev`-pointer lattice
/// becomes safe indices. `dawgs` is the arena-owned analogue of the C++ node's
/// owned `DawgPositionVector*` — a plain value clone on node clone/update, rather
/// than the C++ move-masquerading-as-copy (`RecodeNode::operator=`'s `memcpy` +
/// null-the-source dance): observably equivalent (the dawg positions are
/// immutable once attached), simpler, and safe under the arena's `Vec<RecodeNode>`
/// storage. All three dawg-only fields (`start_of_word`/`end_of_word`/`dawgs`)
/// stay at their non-dict defaults (`false`/`false`/`None`) unless the dict path
/// (D1.3) sets them.
#[derive(Clone, Debug)]
struct RecodeNode {
    /// The re-encoded code = index into the network output.
    code: i32,
    /// The decoded unichar-id (valid only at the final code of a sequence).
    unichar_id: i32,
    /// The permuter (always [`TOP_CHOICE_PERM`] in the non-dict path; the full
    /// range in the dict path, from [`DictLite::def_letter_is_okay`]).
    permuter: PermuterType,
    /// True if this is the initial dawg state (always `false` in the non-dict path;
    /// retained for the [`RecodeBeamSearch::update_heap_if_matched`] identity).
    start_of_dawg: bool,
    /// True if this is the first node in a dictionary word (`recodebeam.h:154`).
    /// Always `false` in the non-dict path. Written (matching C++ exactly, for
    /// fidelity + the `update_heap_if_matched`/`Clone` value shape) but not yet
    /// READ: its consumer is `ExtractBestPathAsWords`'s word-boundary walk
    /// (B3-full, D1.4/D1.5 — not yet landed).
    #[allow(
        dead_code,
        reason = "carried for the not-yet-landed B3-full word-boundary extract (D1.4/D1.5); required for byte-faithful RecodeNode shape now"
    )]
    start_of_word: bool,
    /// True if this is a valid candidate end-of-word position (`recodebeam.h:158`).
    /// Always `false` in the non-dict path.
    end_of_word: bool,
    /// True if `code` is a duplicate of `prev.code` (CTC fold-on-the-fly).
    duplicate: bool,
    /// Certainty (log prob) of just this position — read by
    /// [`RecodeBeamSearch::extract_best_path_as_unichar_ids`] (C2).
    certainty: f32,
    /// Total certainty of the path to this position.
    score: f32,
    /// The previous node in the chain, as an arena index.
    prev: Option<u32>,
    /// The currently active dawg positions at this node (`RecodeNode::dawgs`,
    /// `recodebeam.h:171`) — `Some` only for dict-path dawg-beam nodes.
    dawgs: Option<Box<[DawgPosition]>>,
    /// A hash of all codes in the prefix + this code (duplicate-path removal).
    code_hash: u64,
}

/// One heap element: a `KDPairInc<double, RecodeNode>` (`recodebeam.h:177`)
/// reduced to `(key, arena-index)` — the node values live in the arena.
#[derive(Clone, Copy, Debug)]
struct HeapPair {
    /// `f64::from(node.score)` — the sort key (lossless, so heap order == score order).
    key: f64,
    /// Arena index of the node.
    idx: u32,
}

/// Tesseract's `GenericHeap<KDPairInc<…>>` (`genericheap.h`) — a binary **min**
/// heap (worst/smallest key at the top for fast top-n eviction), with the exact
/// `SiftUp`/`SiftDown`/`Reshuffle` so `get(i)` internal order (which the decode
/// walks) is byte-identical to libtesseract's.
#[derive(Default, Clone, Debug)]
struct MinHeap {
    v: Vec<HeapPair>,
}

impl MinHeap {
    fn clear(&mut self) {
        self.v.clear();
    }
    fn len(&self) -> usize {
        self.v.len()
    }
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    /// `heap_[i]` in internal (array) order — NOT sorted order (`GenericHeap::get`).
    fn get(&self, i: usize) -> HeapPair {
        self.v[i]
    }
    /// `PeekTop` = the smallest-key (worst) element (`genericheap.h:108`).
    fn peek_top(&self) -> HeapPair {
        self.v[0]
    }

    /// `GenericHeap::Push` (`genericheap.h:95`): append then sift the hole up.
    fn push(&mut self, entry: HeapPair) {
        let hole0 = self.v.len();
        self.v.push(entry);
        let hole = self.sift_up(hole0, entry);
        self.v[hole] = entry;
    }

    /// `GenericHeap::Pop` (`genericheap.h:120`): remove the top, sift the last
    /// element's hole down from the root.
    fn pop(&mut self) -> Option<HeapPair> {
        let n = self.v.len();
        if n == 0 {
            return None;
        }
        let top = self.v[0];
        if n - 1 > 0 {
            let hole_pair = self.v[n - 1];
            self.v.truncate(n - 1);
            let hole = self.sift_down(0, hole_pair);
            self.v[hole] = hole_pair;
        } else {
            self.v.truncate(0);
        }
        Some(top)
    }

    /// Set element `i`'s key and reshuffle it (`UpdateHeapIfMatched`'s
    /// `i.key() = …; Reshuffle(&i)`).
    fn update_and_reshuffle(&mut self, i: usize, key: f64) {
        self.v[i].key = key;
        self.reshuffle(i);
    }

    /// `GenericHeap::SiftUp` (`genericheap.h:205`) — key-only strict `<`
    /// (`KDPairInc::operator<`, `kdpair.h:67`).
    fn sift_up(&mut self, mut hole: usize, pair: HeapPair) -> usize {
        while hole > 0 {
            // `ParentNode(index) = (index + 1) / 2 - 1` (`genericheap.h:236`);
            // for `hole >= 1` that equals `(hole - 1) / 2` (integer division).
            let parent = (hole - 1) / 2;
            if pair.key < self.v[parent].key {
                self.v[hole] = self.v[parent];
                hole = parent;
            } else {
                break;
            }
        }
        hole
    }

    /// `GenericHeap::SiftDown` (`genericheap.h:217`) — pick the smaller child
    /// (right only if strictly less), stop when the hole's pair is `<=` the child.
    fn sift_down(&mut self, mut hole: usize, pair: HeapPair) -> usize {
        let n = self.v.len();
        loop {
            let mut child = hole * 2 + 1;
            if child >= n {
                break;
            }
            if child + 1 < n && self.v[child + 1].key < self.v[child].key {
                child += 1;
            }
            if self.v[child].key < pair.key {
                self.v[hole] = self.v[child];
                hole = child;
            } else {
                break;
            }
        }
        hole
    }

    /// `GenericHeap::Reshuffle` (`genericheap.h:193`): sift down then up.
    fn reshuffle(&mut self, index: usize) {
        let hole_pair = self.v[index];
        let index = self.sift_down(index, hole_pair);
        let index = self.sift_up(index, hole_pair);
        self.v[index] = hole_pair;
    }
}

/// The re-encode CTC beam search (`RecodeBeamSearch`, `recodebeam.h:181`) — the
/// non-dictionary path. Borrows the recoder; owns the per-timestep lattice.
pub struct RecodeBeamSearch<'a> {
    /// The recoder (its `get_final_codes`/`get_next_codes`/`decode`/`code_range`
    /// are the beam's continuation oracle).
    recoder: &'a UnicharCompress,
    /// The encoded null/blank class.
    null_char: i32,
    /// True if adjacent equal chars are NOT eliminated (simple text mode).
    is_simple_text: bool,
    /// `dict_ratio` — scales every certainty (`1.0` for a plain decode).
    dict_ratio: f32,
    /// `cert_offset` — added to every certainty (`0.0` for a plain decode).
    cert_offset: f32,
    /// `worst_dict_cert` — the certainty floor below which a dawg continuation is
    /// rejected outright (`0.0`/unused on the non-dict path; production value is
    /// `kWorstDictCertainty / kCertaintyScale = -25.0/7.0`, `linerec.cpp:33-35`).
    worst_dict_cert: f32,
    /// The dictionary (word/punc/number dawgs), if this search is dict-enabled
    /// (`Dict *dict_`, `recodebeam.h:416`). `None` is the byte-parity-proven
    /// non-dict path (`E-OCR-RECODEBEAM-1`); untouched by D1.3.
    dict: Option<DictLite>,
    /// The character set — needed alongside `dict` for `IsSpaceDelimited`
    /// (script lookups). Always `Some` iff `dict` is `Some`.
    charset: Option<UniCharSet>,
    /// `space_delimited_` (`recodebeam.h:419`, `Dict::IsSpaceDelimitedLang`):
    /// `true` unless `charset` registers a Han/Katakana/Thai script. `true` (its
    /// default) when `dict` is `None`.
    space_delimited: bool,
    /// Every node across all timesteps (the lattice backing store); `prev` indexes
    /// here.
    arena: Vec<RecodeNode>,
    /// `beam_[t].beams_[k]` — `K_NUM_BEAMS` heaps per timestep.
    beams: Vec<[MinHeap; K_NUM_BEAMS]>,
    /// `beam_[t].best_initial_dawgs_[c]` (`recodebeam.h:295`) — the single best
    /// initial-dawg candidate per [`NodeContinuation`] at each timestep, pushed to
    /// the dawg beam only once, after the whole step is processed (`recodebeam.h`
    /// "so it doesn't blow up the beam"). Always all-`None` on the non-dict path.
    best_initial_dawgs: Vec<[Option<RecodeNode>; NC_COUNT]>,
    /// Number of valid timesteps in `beams`.
    beam_size: usize,
    /// `top_n_flags_` — the current timestep's per-code top-n classification.
    top_n_flags: Vec<TopN>,
    /// The highest-scoring code this timestep (`top_code_`).
    top_code: i32,
    /// The second highest (`second_code_`).
    second_code: i32,
    /// Reused heap for `ComputeTopN`.
    top_heap: MinHeap,
}

/// One extracted word — the information-content equivalent of Tesseract's
/// `WERD_RES` (`ExtractBestPathAsWords`, `recodebeam.cpp:239-322`). We do NOT
/// port `WERD`/`WERD_RES`/`MATRIX`/`BLOB_CHOICE`/`PAGE_RES` — every number
/// those types would have carried for a single output word is here instead:
/// per-character unichar ids/certainties/ratings (the `BLOB_CHOICE` diagonal,
/// `recodebeam.cpp:303-313`), per-character boxes in the caller's `line_box`
/// pixel space (the fake `C_BLOB`s, `recodebeam.cpp:645-657`), the permuter
/// that produced the word (`FakeWordFromRatings`'s argument,
/// `recodebeam.cpp:314-315`), the merged leading/trailing space certainty,
/// and whether a leading space preceded the word.
#[derive(Clone, Debug, PartialEq)]
pub struct WordResult {
    /// Per-character unichar ids, `unichar_ids[word_start..word_end)`.
    pub unichar_ids: Vec<i32>,
    /// Per-character certainties (log-prob), same range.
    pub certs: Vec<f32>,
    /// Per-character CTC ratings (negative summed log-prob), same range.
    pub ratings: Vec<f32>,
    /// Per-character boxes `(left, bottom, right, top)` in the caller's
    /// `line_box` coordinate space (`InitializeWord`, `recodebeam.cpp:642-657`).
    /// May be shorter than `unichar_ids` — `InitializeWord` only emits a box
    /// for cell `i` when `i + 1 < character_boundaries.len()`
    /// (`recodebeam.cpp:646`), exactly as the C++ does.
    pub char_boxes: Vec<(i32, i32, i32, i32)>,
    /// The permuter of the word's last character
    /// (`best_nodes[xcoords[word_end - 1]]->permuter`, the `FakeWordFromRatings`
    /// argument, `recodebeam.cpp:314-315`).
    pub permuter: PermuterType,
    /// `min(space_cert, prev_space_cert)` — the certainty of the space that
    /// terminates the PREVIOUS word, passed to `InitializeWord` as
    /// `space_certainty` (`recodebeam.cpp:300-301`).
    pub space_certainty: f32,
    /// True if a space immediately precedes this word (`recodebeam.cpp:296-297`).
    pub leading_space: bool,
}

/// `RecodeBeamSearch::calculateCharBoundaries` (`recodebeam.cpp:187-198`): turn
/// the per-character `starts`/`ends` xcoord bookkeeping from
/// [`RecodeBeamSearch::extract_path_as_unichar_ids_with_boundaries`] into
/// character-boundary x-coordinates — one more entry than there are
/// characters, so `boundaries[i]`/`boundaries[i + 1]` bracket character `i`'s
/// extent for box construction.
#[must_use]
fn calculate_char_boundaries(starts: &[i32], ends: &[i32], max_width: i32) -> Vec<i32> {
    let mut char_bounds = Vec::with_capacity(ends.len() + 1);
    char_bounds.push(0);
    for (i, &end) in ends.iter().enumerate() {
        let middle = (starts[i + 1] - end) / 2;
        char_bounds.push(end + middle);
    }
    char_bounds.pop();
    char_bounds.push(max_width);
    char_bounds
}

/// Mirrors `static_cast<int16_t>(float)` used by `InitializeWord`'s box
/// construction (`recodebeam.cpp:647-654`): a C++ narrowing float→int16_t on
/// an out-of-range value is undefined behaviour; Rust's `as i16` on a float
/// saturates instead (defined, no UB). The two agree for every value that can
/// legitimately occur (a beam x-coordinate scaled into image-pixel space,
/// always far inside `i16` range for a real line image) and only diverge on
/// an already-UB input.
#[must_use]
fn clip_to_i16(x: f32) -> i16 {
    x as i16
}

impl<'a> RecodeBeamSearch<'a> {
    /// Construct a beam search over `recoder` with the given `null_char`
    /// (`RecodeBeamSearch::RecodeBeamSearch`, `recodebeam.cpp:58`; `dict = nullptr`).
    #[must_use]
    pub fn new(recoder: &'a UnicharCompress, null_char: i32, is_simple_text: bool) -> Self {
        Self {
            recoder,
            null_char,
            is_simple_text,
            dict_ratio: 1.0,
            cert_offset: 0.0,
            worst_dict_cert: 0.0,
            dict: None,
            charset: None,
            space_delimited: true,
            arena: Vec::new(),
            beams: Vec::new(),
            best_initial_dawgs: Vec::new(),
            beam_size: 0,
            top_n_flags: Vec::new(),
            top_code: -1,
            second_code: -1,
            top_heap: MinHeap::default(),
        }
    }

    /// Construct a dict-enabled beam search (`RecodeBeamSearch::RecodeBeamSearch`
    /// with a non-null `dict`, `recodebeam.cpp:58-71`) — D1.3. `space_delimited`
    /// mirrors `dict_ != nullptr && !dict_->IsSpaceDelimitedLang()`: computed once
    /// here from `charset`'s registered scripts, exactly as the C++ constructor
    /// computes it once from `dict_->IsSpaceDelimitedLang()`.
    #[must_use]
    pub fn new_with_dict(
        recoder: &'a UnicharCompress,
        null_char: i32,
        is_simple_text: bool,
        dict: DictLite,
        charset: UniCharSet,
    ) -> Self {
        let space_delimited = is_space_delimited_lang(&charset);
        Self {
            recoder,
            null_char,
            is_simple_text,
            dict_ratio: 1.0,
            cert_offset: 0.0,
            worst_dict_cert: 0.0,
            dict: Some(dict),
            charset: Some(charset),
            space_delimited,
            arena: Vec::new(),
            beams: Vec::new(),
            best_initial_dawgs: Vec::new(),
            beam_size: 0,
            top_n_flags: Vec::new(),
            top_code: -1,
            second_code: -1,
            top_heap: MinHeap::default(),
        }
    }

    /// Decode a softmax matrix (`outputs[t]` = the class-prob row at timestep `t`)
    /// — the public `Decode(GENERIC_2D_ARRAY<float>, dict_ratio, cert_offset,
    /// worst_dict_cert, charset)` (`recodebeam.cpp:100`) with `charset = nullptr`
    /// (no whitelist) and `worst_dict_cert` unused (non-dict). `dict_ratio = 1.0`
    /// and `cert_offset = 0.0` give a plain decode. Unchanged from the
    /// `E-OCR-RECODEBEAM-1` byte-parity surface — regression-critical.
    pub fn decode(&mut self, outputs: &[&[f32]], dict_ratio: f32, cert_offset: f32) {
        self.decode_impl(outputs, dict_ratio, cert_offset, 0.0);
    }

    /// Decode with the dictionary active (D1.3) — the same walk as [`Self::decode`]
    /// but threading `worst_dict_cert` (the dawg-continuation certainty floor,
    /// production value `kWorstDictCertainty / kCertaintyScale = -25.0/7.0`,
    /// `linerec.cpp:33-35,253-254`) into every dawg-branch check. Requires the
    /// search to have been built via [`Self::new_with_dict`]; the dawg beams stay
    /// empty (a no-op) if it was not, matching the non-dict result exactly.
    pub fn decode_with_dict(
        &mut self,
        outputs: &[&[f32]],
        dict_ratio: f32,
        cert_offset: f32,
        worst_dict_cert: f32,
    ) {
        self.decode_impl(outputs, dict_ratio, cert_offset, worst_dict_cert);
    }

    /// Shared body of [`Self::decode`]/[`Self::decode_with_dict`]
    /// (`RecodeBeamSearch::Decode`, `recodebeam.cpp:100-110`).
    fn decode_impl(
        &mut self,
        outputs: &[&[f32]],
        dict_ratio: f32,
        cert_offset: f32,
        worst_dict_cert: f32,
    ) {
        self.dict_ratio = dict_ratio;
        self.cert_offset = cert_offset;
        self.worst_dict_cert = worst_dict_cert;
        self.beam_size = 0;
        self.arena.clear();
        self.beams.clear();
        self.best_initial_dawgs.clear();
        for (t, row) in outputs.iter().enumerate() {
            self.compute_top_n(row, K_BEAM_WIDTHS[0]);
            self.decode_step(row, t);
        }
    }

    /// `ComputeTopN` (`recodebeam.cpp:672`): flag each code TN_TOP2 / TN_TOPN /
    /// TN_ALSO_RAN by its rank among the `top_n` largest logits; `null_char` is
    /// always TN_TOP2.
    fn compute_top_n(&mut self, outputs: &[f32], top_n: usize) {
        let n = outputs.len();
        self.top_n_flags.clear();
        self.top_n_flags.resize(n, TopN::AlsoRan);
        self.top_code = -1;
        self.second_code = -1;
        self.top_heap.clear();
        for (i, &out) in outputs.iter().enumerate() {
            if self.top_heap.len() < top_n || f64::from(out) > self.top_heap.peek_top().key {
                self.top_heap.push(HeapPair {
                    key: f64::from(out),
                    idx: i as u32,
                });
                if self.top_heap.len() > top_n {
                    self.top_heap.pop();
                }
            }
        }
        while !self.top_heap.is_empty() {
            let entry = self.top_heap.pop().expect("non-empty");
            let data = entry.idx as usize;
            if self.top_heap.len() > 1 {
                self.top_n_flags[data] = TopN::Runner;
            } else {
                self.top_n_flags[data] = TopN::Top2;
                if self.top_heap.is_empty() {
                    self.top_code = data as i32;
                } else {
                    self.second_code = data as i32;
                }
            }
        }
        self.top_n_flags[self.null_char as usize] = TopN::Top2;
    }

    /// `DecodeStep` (`recodebeam.cpp:743`): extend the beam for timestep `t`. `t==0`
    /// seeds from nothing; later steps work through the top-n groups (falling back
    /// to the next group only while the NC_ANYTHING beams are empty) over every
    /// previous-beam node.
    fn decode_step(&mut self, outputs: &[f32], t: usize) {
        if t == self.beams.len() {
            self.beams
                .push(core::array::from_fn(|_| MinHeap::default()));
            self.best_initial_dawgs.push([None, None, None]);
        }
        self.beam_size = t + 1;
        for heap in &mut self.beams[t] {
            heap.clear();
        }
        self.best_initial_dawgs[t] = [None, None, None];
        if t == 0 {
            // The first step can only use singles and initials.
            self.continue_context(
                None,
                beam_index(false, NC_ANYTHING, 0),
                outputs,
                TopN::Top2,
                t,
            );
            if self.dict.is_some() {
                self.continue_context(
                    None,
                    beam_index(true, NC_ANYTHING, 0),
                    outputs,
                    TopN::Top2,
                    t,
                );
            }
            return;
        }
        let mut total_beam = 0usize;
        let groups = [TopN::Top2, TopN::Runner, TopN::AlsoRan];
        let mut g = 0;
        while g < groups.len() && total_beam == 0 {
            let top_n = groups[g];
            for index in 0..K_NUM_BEAMS {
                // Backwards through the heap array (as libtesseract does).
                let sz = self.beams[t - 1][index].len();
                for i in (0..sz).rev() {
                    let prev_idx = self.beams[t - 1][index].get(i).idx;
                    self.continue_context(Some(prev_idx), index, outputs, top_n, t);
                }
            }
            total_beam = 0;
            for index in 0..K_NUM_BEAMS {
                if cont_from_index(index) == NC_ANYTHING {
                    total_beam += self.beams[t][index].len();
                }
            }
            g += 1;
        }
        // Special case for the best initial dawg (`recodebeam.cpp:803-812`): push
        // it on the dawg heap if good enough, but there is only one per
        // continuation, so it doesn't blow up the beam. A no-op when `dict` is
        // `None` (`best_initial_dawgs` stays all-`None`, per `push_initial_dawg_
        // if_better` never being called on the non-dict path).
        for cont in 0..NC_COUNT {
            if let Some(node) = self.best_initial_dawgs[t][cont].take() {
                let index = beam_index(true, cont, 0);
                self.push_node_if_better(t, index, K_BEAM_WIDTHS[0], node);
            }
        }
    }

    /// `ContinueContext` (`recodebeam.cpp:891`): add the legal continuations of
    /// `prev` (a beam node at `index`, or `None` at `t==0`) using codes whose
    /// `top_n_flags` matches `top_n_flag`. Builds the recoder `prefix` from the
    /// last `length` real (non-dup, non-null) codes, then walks `get_final_codes`
    /// (completions) and `get_next_codes` (continuations) plus the dup/null rules.
    #[allow(
        clippy::too_many_lines,
        reason = "faithful transcode of one C++ method"
    )]
    fn continue_context(
        &mut self,
        prev: Option<u32>,
        index: usize,
        outputs: &[f32],
        top_n_flag: TopN,
        t: usize,
    ) {
        let length = length_from_index(index);
        let use_dawgs = is_dawg_from_index(index);
        let prev_cont = cont_from_index(index);

        // Build prefix (codes[0..length]) + full_code (prefix ++ code) keys by
        // walking back through the real codes, skipping dups and nulls.
        let mut prefix_codes = vec![0_i32; length];
        let mut full_codes = vec![0_i32; length + 1];
        {
            let mut previous = prev;
            let mut p = length as i32 - 1;
            while p >= 0 && previous.is_some() {
                while let Some(pi) = previous {
                    let node = &self.arena[pi as usize];
                    if node.duplicate || node.code == self.null_char {
                        previous = node.prev;
                    } else {
                        break;
                    }
                }
                let Some(pi) = previous else { break };
                let code = self.arena[pi as usize].code;
                prefix_codes[p as usize] = code;
                full_codes[p as usize] = code;
                previous = self.arena[pi as usize].prev;
                p -= 1;
            }
        }
        let prefix = RecodedCharId::from_codes(&prefix_codes);

        // Dup / null continuations of prev (skipped in simple-text mode).
        if let Some(prev_idx) = prev {
            if !self.is_simple_text {
                let prev_code = self.arena[prev_idx as usize].code;
                let prev_uni = self.arena[prev_idx as usize].unichar_id;
                if self.top_n_flags[prev_code as usize] == top_n_flag {
                    if prev_cont != NC_NO_DUP {
                        let cert =
                            prob_to_certainty(outputs[prev_code as usize]) + self.cert_offset;
                        self.push_dup_or_no_dawg_if_better(
                            t,
                            length,
                            true,
                            prev_code,
                            prev_uni,
                            cert,
                            use_dawgs,
                            NC_ANYTHING,
                            prev,
                        );
                    }
                    if prev_cont == NC_ANYTHING
                        && top_n_flag == TopN::Top2
                        && prev_code != self.null_char
                    {
                        let cert = prob_to_certainty(
                            outputs[prev_code as usize] + outputs[self.null_char as usize],
                        ) + self.cert_offset;
                        self.push_dup_or_no_dawg_if_better(
                            t, length, true, prev_code, prev_uni, cert, use_dawgs, NC_NO_DUP, prev,
                        );
                    }
                }
                if prev_cont == NC_ONLY_DUP {
                    return;
                }
                if prev_code != self.null_char
                    && length > 0
                    && self.top_n_flags[self.null_char as usize] == top_n_flag
                {
                    let cert =
                        prob_to_certainty(outputs[self.null_char as usize]) + self.cert_offset;
                    self.push_dup_or_no_dawg_if_better(
                        t,
                        length,
                        false,
                        self.null_char,
                        INVALID_UNICHAR_ID,
                        cert,
                        use_dawgs,
                        NC_ANYTHING,
                        prev,
                    );
                }
            }
        }

        // Completions: codes that end a sequence from this prefix.
        if let Some(final_codes) = self.recoder.get_final_codes(&prefix).map(<[i32]>::to_vec) {
            for code in final_codes {
                if self.top_n_flags[code as usize] != top_n_flag {
                    continue;
                }
                if let Some(prev_idx) = prev {
                    if self.arena[prev_idx as usize].code == code && !self.is_simple_text {
                        continue;
                    }
                }
                let cert = prob_to_certainty(outputs[code as usize]) + self.cert_offset;
                if cert < K_MIN_CERTAINTY && code != self.null_char {
                    continue;
                }
                full_codes[length] = code;
                let full = RecodedCharId::from_codes(&full_codes);
                let mut unichar_id = self.recoder.decode(&full);
                if length == 0 && code == self.null_char {
                    unichar_id = INVALID_UNICHAR_ID;
                }
                self.continue_unichar(t, code, unichar_id, cert, use_dawgs, NC_ANYTHING, prev);
                if top_n_flag == TopN::Top2 && code != self.null_char {
                    let cert =
                        prob_to_certainty(self.combined_prob(outputs, code, prev, prev_cont))
                            + self.cert_offset;
                    self.continue_unichar(t, code, unichar_id, cert, use_dawgs, NC_ONLY_DUP, prev);
                }
            }
        }

        // Continuations: codes that extend the prefix without completing it.
        if let Some(next_codes) = self.recoder.get_next_codes(&prefix).map(<[i32]>::to_vec) {
            for code in next_codes {
                if self.top_n_flags[code as usize] != top_n_flag {
                    continue;
                }
                if let Some(prev_idx) = prev {
                    if self.arena[prev_idx as usize].code == code && !self.is_simple_text {
                        continue;
                    }
                }
                let cert = prob_to_certainty(outputs[code as usize]) + self.cert_offset;
                self.push_dup_or_no_dawg_if_better(
                    t,
                    length + 1,
                    false,
                    code,
                    INVALID_UNICHAR_ID,
                    cert,
                    use_dawgs,
                    NC_ANYTHING,
                    prev,
                );
                if top_n_flag == TopN::Top2 && code != self.null_char {
                    let cert =
                        prob_to_certainty(self.combined_prob(outputs, code, prev, prev_cont))
                            + self.cert_offset;
                    self.push_dup_or_no_dawg_if_better(
                        t,
                        length + 1,
                        false,
                        code,
                        INVALID_UNICHAR_ID,
                        cert,
                        use_dawgs,
                        NC_ONLY_DUP,
                        prev,
                    );
                }
            }
        }
    }

    /// The TN_TOP2 combined probability `outputs[code] + outputs[null]`, plus
    /// `outputs[prev.code]` when `prev`/`code` are the top-2 pair on an NC_ANYTHING
    /// prev (`recodebeam.cpp:967-974` / `994-1001`).
    fn combined_prob(
        &self,
        outputs: &[f32],
        code: i32,
        prev: Option<u32>,
        prev_cont: usize,
    ) -> f32 {
        let mut prob = outputs[code as usize] + outputs[self.null_char as usize];
        if let Some(prev_idx) = prev {
            let prev_code = self.arena[prev_idx as usize].code;
            if prev_cont == NC_ANYTHING
                && prev_code != self.null_char
                && ((prev_code == self.top_code && code == self.second_code)
                    || (code == self.top_code && prev_code == self.second_code))
            {
                prob += outputs[prev_code as usize];
            }
        }
        prob
    }

    /// `ContinueUnichar` (`recodebeam.cpp:1012-1052`): non-dawg branch pushes a new
    /// unichar node to the length-0 beam for `cont`; the dawg branch (D1.3) routes
    /// to [`Self::continue_dawg`] when `cert` clears the dict certainty floor. The
    /// tail (only reached non-dawg, `dict` present) additionally seeds a dict-word
    /// start when this unichar is a valid word boundary — a space, or any
    /// non-space-delimited character (CJK-style, unreachable for `eng`).
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful transcode of one C++ method"
    )]
    fn continue_unichar(
        &mut self,
        t: usize,
        code: i32,
        unichar_id: i32,
        cert: f32,
        use_dawgs: bool,
        cont: usize,
        prev: Option<u32>,
    ) {
        if use_dawgs {
            if cert > self.worst_dict_cert {
                self.continue_dawg(t, code, unichar_id, cert, cont, prev);
            }
            return;
        }
        let index = beam_index(false, cont, 0);
        self.push_heap_if_better(
            t,
            index,
            K_BEAM_WIDTHS[0],
            code,
            unichar_id,
            TOP_CHOICE_PERM,
            false,
            false,
            false,
            false,
            cert * self.dict_ratio,
            prev,
            None,
        );
        if self.dict.is_some() {
            let space_delim_ok = {
                let charset = self
                    .charset
                    .as_ref()
                    .expect("charset present whenever dict is present");
                (unichar_id == UNICHAR_SPACE && cert > self.worst_dict_cert)
                    || !is_space_delimited(charset, unichar_id)
            };
            if space_delim_ok {
                // Any top-choice position that can start a new word — a space, or
                // any non-space-delimited character — is also considered by the
                // dawg search. A space is flagged NO_PERM (its certainty is not
                // derived from predecessor nulls the way a real dict word is);
                // anything else is scaled by dict_ratio.
                let mut dawg_cert = cert;
                let mut permuter = TOP_CHOICE_PERM;
                if unichar_id == UNICHAR_SPACE {
                    permuter = NO_PERM;
                } else {
                    dawg_cert *= self.dict_ratio;
                }
                self.push_initial_dawg_if_better(
                    t, code, unichar_id, permuter, false, false, dawg_cert, cont, prev,
                );
            }
        }
    }

    /// `ContinueDawg` (`recodebeam.cpp:1057-1136`) — D1.3, the dict path of
    /// [`Self::continue_unichar`]. Walks the dawg beam for a new unichar: an
    /// `INVALID_UNICHAR_ID` (a partial multi-code sequence) rides straight onto
    /// the dawg heap; a completed `UNICHAR_SPACE` after a valid word-end reopens a
    /// fresh word start; otherwise the surviving dawg positions come from
    /// [`DictLite::default_dawgs`] (line start) or `uni_prev.dawgs` (continuing a
    /// word), advanced one letter via [`DictLite::def_letter_is_okay`].
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful transcode of one C++ method"
    )]
    fn continue_dawg(
        &mut self,
        t: usize,
        code: i32,
        unichar_id: i32,
        cert: f32,
        cont: usize,
        prev: Option<u32>,
    ) {
        let dawg_index = beam_index(true, cont, 0);
        let nodawg_index = beam_index(false, cont, 0);
        if unichar_id == INVALID_UNICHAR_ID {
            self.push_heap_if_better(
                t,
                dawg_index,
                K_BEAM_WIDTHS[0],
                code,
                unichar_id,
                NO_PERM,
                false,
                false,
                false,
                false,
                cert,
                prev,
                None,
            );
            return;
        }
        // Avoid the dictionary probe if the score is already a total loss in
        // both destination heaps (a pure performance short-circuit: either
        // condition alone would already cause the eventual push to reject).
        let prev_score = prev.map_or(0.0, |p| self.arena[p as usize].score);
        let score = cert + prev_score;
        let dawg_len = self.beams[t][dawg_index].len();
        let nodawg_len = self.beams[t][nodawg_index].len();
        if dawg_len >= K_BEAM_WIDTHS[0]
            && score <= self.arena[self.beams[t][dawg_index].peek_top().idx as usize].score
            && nodawg_len >= K_BEAM_WIDTHS[0]
            && score <= self.arena[self.beams[t][nodawg_index].peek_top().idx as usize].score
        {
            return;
        }
        // `prev` may be a partial code, null_char, or duplicate; scan back to the
        // last node with a valid unichar_id.
        let mut uni_prev = prev;
        while let Some(pi) = uni_prev {
            let node = &self.arena[pi as usize];
            if node.unichar_id == INVALID_UNICHAR_ID || node.duplicate {
                uni_prev = node.prev;
            } else {
                break;
            }
        }
        let charset = self
            .charset
            .as_ref()
            .expect("charset present whenever dict is present");
        if unichar_id == UNICHAR_SPACE {
            if let Some(pi) = uni_prev {
                let node = &self.arena[pi as usize];
                if node.end_of_word {
                    // Space is good: push an initial state to the dawg beam, and
                    // a regular space to the top-choice (non-dawg) beam.
                    let permuter = node.permuter;
                    self.push_initial_dawg_if_better(
                        t, code, unichar_id, permuter, false, false, cert, cont, prev,
                    );
                    self.push_heap_if_better(
                        t,
                        nodawg_index,
                        K_BEAM_WIDTHS[0],
                        code,
                        unichar_id,
                        permuter,
                        false,
                        false,
                        false,
                        false,
                        cert,
                        prev,
                        None,
                    );
                }
            }
            return;
        }
        if let Some(pi) = uni_prev {
            let node = &self.arena[pi as usize];
            if node.start_of_dawg
                && node.unichar_id != UNICHAR_SPACE
                && is_space_delimited(charset, node.unichar_id)
                && is_space_delimited(charset, unichar_id)
            {
                return; // Can't break words between space-delimited chars.
            }
        }
        let dict = self
            .dict
            .as_ref()
            .expect("dict present whenever continue_dawg is reached");
        let (active_dawgs, word_start): (Box<[DawgPosition]>, bool) = if let Some(pi) = uni_prev {
            let node = &self.arena[pi as usize];
            let Some(dawgs) = node.dawgs.clone() else {
                return; // Can't continue if not a dict word.
            };
            (dawgs, node.start_of_dawg)
        } else {
            // Starting from the beginning of the line.
            (dict.default_dawgs(false).into_boxed_slice(), true)
        };
        let (updated, permuter, valid_end) =
            dict.def_letter_is_okay(&active_dawgs, charset, unichar_id as u32, false, NO_PERM);
        if permuter != NO_PERM {
            self.push_heap_if_better(
                t,
                dawg_index,
                K_BEAM_WIDTHS[0],
                code,
                unichar_id,
                permuter,
                false,
                word_start,
                valid_end,
                false,
                cert,
                prev,
                Some(updated.into_boxed_slice()),
            );
            if valid_end && !self.space_delimited {
                // Non-space-delimited language: a new word can start right away.
                self.push_initial_dawg_if_better(
                    t, code, unichar_id, permuter, word_start, true, cert, cont, prev,
                );
                self.push_heap_if_better(
                    t,
                    nodawg_index,
                    K_BEAM_WIDTHS[0],
                    code,
                    unichar_id,
                    permuter,
                    false,
                    word_start,
                    true,
                    false,
                    cert,
                    prev,
                    None,
                );
            }
        }
    }

    /// `PushInitialDawgIfBetter` (`recodebeam.cpp:1141-1160`) — D1.3: keeps the
    /// single best initial-dawg candidate per continuation for this timestep in
    /// [`Self::best_initial_dawgs`] (pushed to the real dawg heap only once, by
    /// [`Self::decode_step`]'s end-of-step special case).
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful transcode of one C++ method"
    )]
    fn push_initial_dawg_if_better(
        &mut self,
        t: usize,
        code: i32,
        unichar_id: i32,
        permuter: PermuterType,
        start: bool,
        end: bool,
        cert: f32,
        cont: usize,
        prev: Option<u32>,
    ) {
        let prev_score = prev.map_or(0.0, |p| self.arena[p as usize].score);
        let score = cert + prev_score;
        let better = match &self.best_initial_dawgs[t][cont] {
            None => true,
            Some(node) => score > node.score,
        };
        if !better {
            return;
        }
        let dict = self
            .dict
            .as_ref()
            .expect("dict present whenever push_initial_dawg_if_better is reached");
        let initial_dawgs = dict.default_dawgs(false).into_boxed_slice();
        let code_hash = self.compute_code_hash(code, false, prev);
        self.best_initial_dawgs[t][cont] = Some(RecodeNode {
            code,
            unichar_id,
            permuter,
            start_of_dawg: true,
            start_of_word: start,
            end_of_word: end,
            duplicate: false,
            certainty: cert,
            score,
            prev,
            dawgs: Some(initial_dawgs),
            code_hash,
        });
    }

    /// `PushDupOrNoDawgIfBetter` (`recodebeam.cpp:1166-1185`): the non-dawg branch
    /// scales by `dict_ratio` and drops below the certainty floor (unless null);
    /// the dawg branch (D1.3) instead gates on `worst_dict_cert` with no scaling
    /// (the dict-side certainty is already comparable across the dawg beams).
    /// Either way, pushes to the length-`length` beam for `cont`.
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful transcode of one C++ method"
    )]
    fn push_dup_or_no_dawg_if_better(
        &mut self,
        t: usize,
        length: usize,
        dup: bool,
        code: i32,
        unichar_id: i32,
        cert: f32,
        use_dawgs: bool,
        cont: usize,
        prev: Option<u32>,
    ) {
        if length >= K_NUM_LENGTHS {
            // No beam exists past `kMaxCodeLen`. Unreachable with a Core-validated
            // recoder — codes are capped at `kMaxCodeLen = 9`, so `get_next_codes`
            // is empty for a prefix of length >= 8 and the `length + 1` from the
            // next-codes path never reaches 10. This guards the theoretical index
            // over libtesseract's unchecked `kBeamWidths[length]`.
            return;
        }
        let index = beam_index(use_dawgs, cont, length);
        if use_dawgs {
            if cert > self.worst_dict_cert {
                let permuter = prev.map_or(NO_PERM, |p| self.arena[p as usize].permuter);
                self.push_heap_if_better(
                    t,
                    index,
                    K_BEAM_WIDTHS[length],
                    code,
                    unichar_id,
                    permuter,
                    false,
                    false,
                    false,
                    dup,
                    cert,
                    prev,
                    None,
                );
            }
        } else {
            let cert = cert * self.dict_ratio;
            if cert >= K_MIN_CERTAINTY || code == self.null_char {
                let permuter = prev.map_or(TOP_CHOICE_PERM, |p| self.arena[p as usize].permuter);
                self.push_heap_if_better(
                    t,
                    index,
                    K_BEAM_WIDTHS[length],
                    code,
                    unichar_id,
                    permuter,
                    false,
                    false,
                    false,
                    dup,
                    cert,
                    prev,
                    None,
                );
            }
        }
    }

    /// `PushHeapIfBetter(max_size, code, unichar_id, permuter, dawg_start,
    /// word_start, end, dup, cert, prev, d, heap)` (`recodebeam.cpp:1190-1216`):
    /// builds the node (`score = cert + prev.score`) and delegates the
    /// heap-insertion decision to [`Self::push_node_if_better`].
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful transcode of one C++ method"
    )]
    fn push_heap_if_better(
        &mut self,
        t: usize,
        index: usize,
        max_size: usize,
        code: i32,
        unichar_id: i32,
        permuter: PermuterType,
        start_of_dawg: bool,
        start_of_word: bool,
        end_of_word: bool,
        duplicate: bool,
        cert: f32,
        prev: Option<u32>,
        dawgs: Option<Box<[DawgPosition]>>,
    ) {
        let prev_score = prev.map_or(0.0, |p| self.arena[p as usize].score);
        let score = cert + prev_score;
        let code_hash = self.compute_code_hash(code, duplicate, prev);
        let node = RecodeNode {
            code,
            unichar_id,
            permuter,
            start_of_dawg,
            start_of_word,
            end_of_word,
            duplicate,
            certainty: cert,
            score,
            prev,
            dawgs,
            code_hash,
        };
        self.push_node_if_better(t, index, max_size, node);
    }

    /// `PushHeapIfBetter(max_size, RecodeNode *node, heap)`
    /// (`recodebeam.cpp:1220-1233`) — the ready-made-node overload: the "best
    /// initial dawg" special case ([`Self::decode_step`]) uses this directly (the
    /// node's `score` is already final), while
    /// [`Self::push_heap_if_better`] builds a node from its arguments and
    /// delegates here.
    fn push_node_if_better(&mut self, t: usize, index: usize, max_size: usize, node: RecodeNode) {
        let heap_len = self.beams[t][index].len();
        let better = if heap_len < max_size {
            true
        } else {
            let top_idx = self.beams[t][index].peek_top().idx;
            node.score > self.arena[top_idx as usize].score
        };
        if !better {
            return;
        }
        if self.update_heap_if_matched(t, index, &node) {
            return;
        }
        let score = node.score;
        let new_idx = self.arena.len() as u32;
        self.arena.push(node);
        self.beams[t][index].push(HeapPair {
            key: f64::from(score),
            idx: new_idx,
        });
        if self.beams[t][index].len() > max_size {
            self.beams[t][index].pop();
        }
    }

    /// `UpdateHeapIfMatched` (`recodebeam.cpp:1237`): if a heap node has the same
    /// `(code, code_hash, permuter, start_of_dawg)`, keep the better score
    /// (updating + reshuffling in place) and report the match.
    fn update_heap_if_matched(&mut self, t: usize, index: usize, new_node: &RecodeNode) -> bool {
        let heap_len = self.beams[t][index].len();
        for j in 0..heap_len {
            let arena_idx = self.beams[t][index].get(j).idx as usize;
            let node = &self.arena[arena_idx];
            if node.code == new_node.code
                && node.code_hash == new_node.code_hash
                && node.permuter == new_node.permuter
                && node.start_of_dawg == new_node.start_of_dawg
            {
                if new_node.score > node.score {
                    self.arena[arena_idx] = new_node.clone();
                    self.beams[t][index].update_and_reshuffle(j, f64::from(new_node.score));
                }
                return true;
            }
        }
        false
    }

    /// `ComputeCodeHash` (`recodebeam.cpp:1262`): a rolling `u64` mix of the prefix
    /// codes (dups and nulls do not advance it), used for duplicate-path removal.
    fn compute_code_hash(&self, code: i32, dup: bool, prev: Option<u32>) -> u64 {
        let mut hash = prev.map_or(0, |p| self.arena[p as usize].code_hash);
        if !dup && code != self.null_char {
            let num_classes = self.recoder.code_range() as u64;
            let carry = ((hash >> 32).wrapping_mul(num_classes)) >> 32;
            hash = hash.wrapping_mul(num_classes);
            hash = hash.wrapping_add(carry);
            hash = hash.wrapping_add(code as u64);
        }
        hash
    }

    /// `ExtractBestPaths` (`recodebeam.cpp:1279-1326`): the highest-scoring node in
    /// the last beam's NC_ANYTHING + NC_NO_DUP heaps (NC_ONLY_DUP skipped), over
    /// both the non-dawg AND dawg beams. A dawg-beam node is only a valid
    /// candidate if, scanning back past intermediate/duplicate codes, the last
    /// real unichar is a completed word (`end_of_word`) or a space — exactly
    /// [`Self::extract_best_node`]'s dawg-validity walk. On the non-dict path the
    /// dawg beams are always empty (never populated), so this is unchanged from
    /// `E-OCR-RECODEBEAM-1`. Returns the best node's arena index, or `None` on an
    /// empty decode.
    fn extract_best_node(&self) -> Option<u32> {
        if self.beam_size == 0 {
            return None;
        }
        let last = self.beam_size - 1;
        let mut best: Option<u32> = None;
        let mut second: Option<u32> = None;
        for c in 0..NC_COUNT {
            if c == NC_ONLY_DUP {
                continue;
            }
            for is_dawg in [false, true] {
                let bindex = beam_index(is_dawg, c, 0);
                let sz = self.beams[last][bindex].len();
                for h in 0..sz {
                    let node_idx = self.beams[last][bindex].get(h).idx;
                    if is_dawg && !self.is_valid_dawg_end(node_idx) {
                        continue;
                    }
                    let score = self.arena[node_idx as usize].score;
                    let better_than_best =
                        best.is_none_or(|b| score > self.arena[b as usize].score);
                    if better_than_best {
                        second = best;
                        best = Some(node_idx);
                    } else if second.is_none_or(|s| score > self.arena[s as usize].score) {
                        second = Some(node_idx);
                    }
                }
            }
        }
        let _ = second;
        best
    }

    /// The dawg-beam candidate filter inlined in `ExtractBestPaths`
    /// (`recodebeam.cpp:1296-1311`): scan back past `INVALID_UNICHAR_ID`/duplicate
    /// nodes to the last real unichar, then accept iff it is a completed word
    /// (`end_of_word`) or a space.
    fn is_valid_dawg_end(&self, node_idx: u32) -> bool {
        let mut cur = Some(node_idx);
        while let Some(pi) = cur {
            let node = &self.arena[pi as usize];
            if node.unichar_id == INVALID_UNICHAR_ID || node.duplicate {
                cur = node.prev;
            } else {
                break;
            }
        }
        match cur {
            None => false,
            Some(pi) => {
                let node = &self.arena[pi as usize];
                node.end_of_word || node.unichar_id == UNICHAR_SPACE
            }
        }
    }

    /// Backtrack the lattice from `node` to the root, returning arena indices in
    /// forward (root→leaf) order (`ExtractPath`, `recodebeam.cpp:1330`).
    fn extract_path(&self, node: Option<u32>) -> Vec<u32> {
        let mut path = Vec::new();
        let mut cur = node;
        while let Some(idx) = cur {
            path.push(idx);
            cur = self.arena[idx as usize].prev;
        }
        path.reverse();
        path
    }

    /// `ExtractBestPathAsLabels` (`recodebeam.cpp:201`): the best path run through
    /// CTC — drop nulls, fold adjacent equal codes — as `(labels, xcoords)`. The
    /// labels are individual recoded **codes** (faithful to the C++, which likewise
    /// returns per-position codes here).
    ///
    /// For a **single-code recoder** (the eng.lstm pass-through, every code is a
    /// complete char), each label is a full `RecodedCharId::from_codes(&[label])`,
    /// so `labels → recoded_to_text` yields the string directly. For a
    /// **multi-code recoder** (Han/Hangul, code length > 1) the completing code's
    /// `unichar_id` must be recovered by grouping consecutive codes back into
    /// complete `RecodedCharId`s and calling `decode` on each group — the job of
    /// the C++ `ExtractBestPathAsUnicharIds` (deferred, alongside the multi-code
    /// `next_codes_` trie). Do NOT feed multi-code labels one-at-a-time to
    /// `recoded_to_text`: a partial one-code id decodes to `INVALID` and drops the
    /// char.
    #[must_use]
    pub fn extract_best_path_as_labels(&self) -> (Vec<i32>, Vec<i32>) {
        let best_nodes = self.extract_path(self.extract_best_node());
        let mut labels = Vec::new();
        let mut xcoords = Vec::new();
        let width = best_nodes.len();
        let mut t = 0;
        while t < width {
            let label = self.arena[best_nodes[t] as usize].code;
            if label != self.null_char {
                labels.push(label);
                xcoords.push(t as i32);
            }
            t += 1;
            while t < width
                && !self.is_simple_text
                && self.arena[best_nodes[t] as usize].code == label
            {
                t += 1;
            }
        }
        xcoords.push(width as i32);
        (labels, xcoords)
    }

    /// `ExtractBestPathAsUnicharIds` (`recodebeam.cpp:224-236` →
    /// `ExtractPathAsUnicharIds`, `recodebeam.cpp:567-632`) — recognizer **C2**,
    /// the general text extract: walk the best path skipping duplicates, nulls
    /// and (multi-code) intermediate parts, returning per-character
    /// `(unichar_ids, certs, ratings, xcoords)`. This is the words-with-certs
    /// surface `RecognizeLine` consumes (B3 milestone ii); unlike
    /// [`extract_best_path_as_labels`](Self::extract_best_path_as_labels) it
    /// groups multi-code sequences (only the completing code carries a valid
    /// `unichar_id`; intermediates are `INVALID` and fold into the rating).
    ///
    /// Float contract (`recodebeam.cpp:582-624`): the running `certainty`/
    /// `rating` accumulate in **f64** from the nodes' f32 certainties and are
    /// narrowed to f32 on push — including the space-merge back-writes
    /// (`certs.back()`/`ratings.back()`), where the comparison promotes the
    /// stored f32 to f64, exactly as C++.
    ///
    /// Space handling: a `UNICHAR_SPACE` whose node is not `NO_PERM` donates
    /// its accumulated leading-null certainty/rating to the PREVIOUS character
    /// (`recodebeam.cpp:594-604`); a `NO_PERM` space (dict-path only — never
    /// produced by this non-dict beam, kept for fidelity) resets the certainty
    /// to its own (`recodebeam.cpp:609-613`).
    #[must_use]
    pub fn extract_best_path_as_unichar_ids(&self) -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<i32>) {
        let best_nodes = self.extract_path(self.extract_best_node());
        let mut unichar_ids: Vec<i32> = Vec::new();
        let mut certs: Vec<f32> = Vec::new();
        let mut ratings: Vec<f32> = Vec::new();
        let mut xcoords: Vec<i32> = Vec::new();
        let width = best_nodes.len();
        let mut t = 0usize;
        while t < width {
            let mut certainty = 0.0_f64;
            let mut rating = 0.0_f64;
            // Leading nulls / intermediate codes: fold into the accumulators.
            while t < width && self.arena[best_nodes[t] as usize].unichar_id == INVALID_UNICHAR_ID {
                let cert = f64::from(self.arena[best_nodes[t] as usize].certainty);
                t += 1;
                if cert < certainty {
                    certainty = cert;
                }
                rating -= cert;
            }
            if t < width {
                let unichar_id = self.arena[best_nodes[t] as usize].unichar_id;
                if unichar_id == UNICHAR_SPACE
                    && !certs.is_empty()
                    && self.arena[best_nodes[t] as usize].permuter != NO_PERM
                {
                    // The rating/certainty accumulated so far go on the
                    // PREVIOUS character, not the space itself.
                    let back = certs.len() - 1;
                    if certainty < f64::from(certs[back]) {
                        certs[back] = certainty as f32;
                    }
                    ratings[back] = (f64::from(ratings[back]) + rating) as f32;
                    certainty = 0.0;
                    rating = 0.0;
                }
                unichar_ids.push(unichar_id);
                xcoords.push(t as i32);
                loop {
                    let node = &self.arena[best_nodes[t] as usize];
                    let cert = f64::from(node.certainty);
                    let no_perm_space = unichar_id == UNICHAR_SPACE && node.permuter == NO_PERM;
                    t += 1;
                    // A NO_PERM space forgets the preceding nulls' certainty.
                    if cert < certainty || no_perm_space {
                        certainty = cert;
                    }
                    rating -= cert;
                    if !(t < width && self.arena[best_nodes[t] as usize].duplicate) {
                        break;
                    }
                }
                certs.push(certainty as f32);
                ratings.push(rating as f32);
            } else if !certs.is_empty() {
                // Trailing nulls: fold into the last character.
                let back = certs.len() - 1;
                if certainty < f64::from(certs[back]) {
                    certs[back] = certainty as f32;
                }
                ratings[back] = (f64::from(ratings[back]) + rating) as f32;
            }
        }
        xcoords.push(width as i32);
        (unichar_ids, certs, ratings, xcoords)
    }

    /// `ExtractPathAsUnicharIds` with the `character_boundaries` out-parameter
    /// populated (`recodebeam.cpp:567-632`, the `character_boundaries != nullptr`
    /// branch) — the variant [`Self::extract_best_path_as_words`] needs for box
    /// construction. Identical certainty/rating/space-merge walk to
    /// [`Self::extract_best_path_as_unichar_ids`] (see that method's docs for
    /// the float contract); additionally tracks the `starts`/`ends` xcoord
    /// bookkeeping (`recodebeam.cpp:591,617,627`) and feeds it to
    /// [`calculate_char_boundaries`]. Kept as a separate method (rather than a
    /// shared-with-boundaries refactor of the public 4-tuple extractor) so the
    /// existing byte-parity-proven `extract_best_path_as_unichar_ids` is
    /// untouched by this addition.
    #[must_use]
    fn extract_path_as_unichar_ids_with_boundaries(
        &self,
        best_nodes: &[u32],
    ) -> UnicharIdsWithBoundaries {
        let mut unichar_ids: Vec<i32> = Vec::new();
        let mut certs: Vec<f32> = Vec::new();
        let mut ratings: Vec<f32> = Vec::new();
        let mut xcoords: Vec<i32> = Vec::new();
        let mut starts: Vec<i32> = Vec::new();
        let mut ends: Vec<i32> = Vec::new();
        let width = best_nodes.len();
        let mut t = 0usize;
        while t < width {
            let mut certainty = 0.0_f64;
            let mut rating = 0.0_f64;
            while t < width && self.arena[best_nodes[t] as usize].unichar_id == INVALID_UNICHAR_ID {
                let cert = f64::from(self.arena[best_nodes[t] as usize].certainty);
                t += 1;
                if cert < certainty {
                    certainty = cert;
                }
                rating -= cert;
            }
            starts.push(t as i32);
            if t < width {
                let unichar_id = self.arena[best_nodes[t] as usize].unichar_id;
                if unichar_id == UNICHAR_SPACE
                    && !certs.is_empty()
                    && self.arena[best_nodes[t] as usize].permuter != NO_PERM
                {
                    let back = certs.len() - 1;
                    if certainty < f64::from(certs[back]) {
                        certs[back] = certainty as f32;
                    }
                    ratings[back] = (f64::from(ratings[back]) + rating) as f32;
                    certainty = 0.0;
                    rating = 0.0;
                }
                unichar_ids.push(unichar_id);
                xcoords.push(t as i32);
                loop {
                    let node = &self.arena[best_nodes[t] as usize];
                    let cert = f64::from(node.certainty);
                    let no_perm_space = unichar_id == UNICHAR_SPACE && node.permuter == NO_PERM;
                    t += 1;
                    if cert < certainty || no_perm_space {
                        certainty = cert;
                    }
                    rating -= cert;
                    if !(t < width && self.arena[best_nodes[t] as usize].duplicate) {
                        break;
                    }
                }
                ends.push(t as i32);
                certs.push(certainty as f32);
                ratings.push(rating as f32);
            } else if !certs.is_empty() {
                let back = certs.len() - 1;
                if certainty < f64::from(certs[back]) {
                    certs[back] = certainty as f32;
                }
                ratings[back] = (f64::from(ratings[back]) + rating) as f32;
            }
        }
        starts.push(width as i32);
        let character_boundaries = calculate_char_boundaries(&starts, &ends, width as i32);
        xcoords.push(width as i32);
        (unichar_ids, certs, ratings, xcoords, character_boundaries)
    }

    /// `ExtractBestPathAsWords` (`recodebeam.cpp:239-322`) — the word/box output
    /// surface `Tesseract::LSTMRecognizeWord` consumes to build `WERD_RES`. We
    /// do not port `WERD_RES`/`MATRIX`/`BLOB_CHOICE` (a higher layer that
    /// actually needs Tesseract's `PAGE_RES` tree would own that); [`WordResult`]
    /// carries the same information content per word instead. The C++ `debug`
    /// flag, `lstm_choice_mode`, and the second-choice path (`second_nodes`,
    /// debug-only) are not ported — `tesseract-rs` never runs the interactive
    /// choice UI those feed.
    ///
    /// `line_box` is `(left, bottom, right, top)` — `TBOX`'s constructor
    /// argument order (`recodebeam.cpp:647-654`). `scale_factor` un-does any
    /// `pixScale` pre-processing so boxes land in the ORIGINAL image's pixel
    /// space.
    #[must_use]
    pub fn extract_best_path_as_words(
        &self,
        line_box: (i32, i32, i32, i32),
        scale_factor: f32,
        charset: &UniCharSet,
    ) -> Vec<WordResult> {
        let best_nodes = self.extract_path(self.extract_best_node());
        let (unichar_ids, certs, ratings, xcoords, character_boundaries) =
            self.extract_path_as_unichar_ids_with_boundaries(&best_nodes);
        let num_ids = unichar_ids.len();
        let (line_left, line_bottom, _line_right, line_top) = line_box;

        let mut words = Vec::new();
        let mut word_start = 0usize;
        let mut prev_space_cert = 0.0_f32;
        while word_start < num_ids {
            // A word is terminated when a space character or start_of_word flag
            // is hit. We also want to force a separate word for every non
            // space-delimited character when not in a dictionary context.
            let mut word_end = word_start + 1;
            while word_end < num_ids {
                if unichar_ids[word_end] == UNICHAR_SPACE {
                    break;
                }
                let index = xcoords[word_end] as usize;
                if self.arena[best_nodes[index] as usize].start_of_word {
                    break;
                }
                let perm = self.arena[best_nodes[index] as usize].permuter;
                if perm == TOP_CHOICE_PERM
                    && (!is_space_delimited(charset, unichar_ids[word_end])
                        || !is_space_delimited(charset, unichar_ids[word_end - 1]))
                {
                    break;
                }
                word_end += 1;
            }

            let space_cert = if word_end < num_ids && unichar_ids[word_end] == UNICHAR_SPACE {
                certs[word_end]
            } else {
                0.0_f32
            };
            let leading_space = word_start > 0 && unichar_ids[word_start - 1] == UNICHAR_SPACE;
            let space_certainty = space_cert.min(prev_space_cert);

            let mut char_boxes = Vec::with_capacity(word_end - word_start);
            for i in word_start..word_end {
                if i + 1 < character_boundaries.len() {
                    let left = i32::from(clip_to_i16(
                        (character_boundaries[i] as f32 * scale_factor).floor(),
                    )) + line_left;
                    let right = i32::from(clip_to_i16(
                        (character_boundaries[i + 1] as f32 * scale_factor).ceil(),
                    )) + line_left;
                    char_boxes.push((left, line_bottom, right, line_top));
                }
            }

            let last_index = xcoords[word_end - 1] as usize;
            let permuter = self.arena[best_nodes[last_index] as usize].permuter;

            words.push(WordResult {
                unichar_ids: unichar_ids[word_start..word_end].to_vec(),
                certs: certs[word_start..word_end].to_vec(),
                ratings: ratings[word_start..word_end].to_vec(),
                char_boxes,
                permuter,
                space_certainty,
                leading_space,
            });

            prev_space_cert = space_cert;
            if word_end < num_ids && unichar_ids[word_end] == UNICHAR_SPACE {
                word_end += 1;
            }
            word_start = word_end;
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pass-through recoder: id `i` → code `[i]`, for `i` in `0..n`. `null_char`
    /// is one of these codes (as in a real model, where the network width is
    /// `code_range` and the null is one of the classes), so `get_final_codes(empty)`
    /// includes it and the beam can emit — and drop — null nodes.
    fn passthrough_recoder(n: i32) -> UnicharCompress {
        let mut bytes = (n as u32).to_le_bytes().to_vec();
        for code in 0..n {
            bytes.push(1); // self_normalized
            bytes.extend_from_slice(&1_i32.to_le_bytes()); // length
            bytes.extend_from_slice(&code.to_le_bytes());
        }
        UnicharCompress::from_le_bytes(&bytes).expect("valid recoder")
    }

    /// A **tie-free** softmax row: a tiny per-code base (breaks ties so the best
    /// path is unambiguous), `winner` bumped by `0.8`, `null` by `0.1`, normalized
    /// — the same shape the byte-parity oracle is fed.
    fn row(n: usize, winner: usize, null_char: usize) -> Vec<f32> {
        let mut row = vec![0.0_f32; n];
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = 0.001 + (c as f32) * 1e-5;
        }
        row[winner] += 0.8;
        if winner != null_char {
            row[null_char] += 0.1;
        }
        let sum: f32 = row.iter().sum();
        for slot in &mut row {
            *slot /= sum;
        }
        row
    }

    fn decode(recoder: &UnicharCompress, null_char: i32, rows: &[Vec<f32>]) -> Vec<i32> {
        let mut beam = RecodeBeamSearch::new(recoder, null_char, false);
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        beam.decode(&refs, 1.0, 0.0);
        beam.extract_best_path_as_labels().0
    }

    #[test]
    fn decodes_a_clean_sequence() {
        // codes 0..4, null=4. Distinct winners 1,2,3 over 3 timesteps → [1,2,3],
        // no folding (this exact shape is byte-parity green vs libtesseract).
        let recoder = passthrough_recoder(5);
        let rows = [row(5, 1, 4), row(5, 2, 4), row(5, 3, 4)];
        assert_eq!(decode(&recoder, 4, &rows), vec![1, 2, 3]);
    }

    #[test]
    fn folds_duplicates_and_drops_nulls() {
        // "1,1,null,1" → the CTC fold collapses the adjacent 1s but the null
        // between the pair and the last 1 preserves both runs → [1, 1].
        let recoder = passthrough_recoder(5);
        let rows = [row(5, 1, 4), row(5, 1, 4), row(5, 4, 4), row(5, 1, 4)];
        assert_eq!(decode(&recoder, 4, &rows), vec![1, 1]);
    }

    #[test]
    fn all_null_decodes_empty() {
        // Every timestep favours the null class → no labels.
        let recoder = passthrough_recoder(5);
        let rows = [row(5, 4, 4), row(5, 4, 4)];
        assert!(decode(&recoder, 4, &rows).is_empty(), "all-null → empty");
    }

    #[test]
    fn labels_feed_recoded_to_text() {
        // The decode → RecodedCharId → recoded_to_text → string chain composes.
        // codes 0..4, null=4; winners 1,2 → labels [1,2] → decode → ids [1,2] →
        // "a","b" (id 0 = NULL→space, id 1 = a, id 2 = b in this charset).
        let recoder = passthrough_recoder(5);
        let charset = crate::UniCharSet::load_from_str(
            "5\nNULL 0 Common 0\na 3 0 a Left a a\nb 3 0 b Left b b\nc 3 0 c Left c c\nd 3 0 d Left d d\n",
        )
        .expect("valid unicharset");
        let rows = [row(5, 1, 4), row(5, 2, 4)];
        let labels = decode(&recoder, 4, &rows);
        assert_eq!(labels, vec![1, 2], "clean two-code decode");
        let codes: Vec<RecodedCharId> = labels
            .iter()
            .map(|&c| RecodedCharId::from_codes(&[c]))
            .collect();
        assert_eq!(crate::recoded_to_text(&recoder, &charset, &codes), "ab");
    }

    // ---- D1.3: dict-enabled beam arms ----

    fn load_real_dict() -> Option<DictLite> {
        let word = std::fs::read("/tmp/eng.lstm-word-dawg").ok()?;
        let punc = std::fs::read("/tmp/eng.lstm-punc-dawg").ok()?;
        let number = std::fs::read("/tmp/eng.lstm-number-dawg").ok()?;
        DictLite::from_components(&word, &punc, &number).ok()
    }

    fn load_real_charset() -> Option<UniCharSet> {
        UniCharSet::load_from_file(std::path::Path::new("/tmp/eng.lstm-unicharset")).ok()
    }

    fn load_real_recoder() -> Option<UnicharCompress> {
        let bytes = std::fs::read("/tmp/eng.lstm-recoder").ok()?;
        UnicharCompress::from_le_bytes(&bytes).ok()
    }

    /// eng's null/blank class (`E-OCR-RECOGNIZER-LOAD-1`).
    const ENG_NULL_CHAR: i32 = 110;

    /// A row with two competing dominant codes (`code_a` at `prob_a`, `code_b` at
    /// `prob_b`) plus a tiny strictly-increasing baseline (tie-breaker + keeps
    /// every other class, including null, non-zero), normalized to a probability
    /// distribution. Mirrors the module's `row()` helper but for a genuine
    /// two-candidate beam competition instead of a single clear winner.
    fn row_two(n: usize, code_a: usize, prob_a: f32, code_b: usize, prob_b: f32) -> Vec<f32> {
        let mut row = vec![0.0_f32; n];
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = (c as f32) * 1e-7;
        }
        row[code_a] = prob_a;
        row[code_b] = prob_b;
        let sum: f32 = row.iter().sum();
        for slot in &mut row {
            *slot /= sum;
        }
        row
    }

    #[test]
    fn dict_dawgs_attach_and_transfer_through_the_beam() {
        let Some(dict) = load_real_dict() else {
            eprintln!("skipping: /tmp/eng.lstm-*-dawg not present");
            return;
        };
        let Some(charset) = load_real_charset() else {
            eprintln!("skipping: /tmp/eng.lstm-unicharset not present");
            return;
        };
        let Some(recoder) = load_real_recoder() else {
            eprintln!("skipping: /tmp/eng.lstm-recoder not present");
            return;
        };
        let n = recoder.code_range() as usize;
        // "the" — ids 91/97/92 per the D1.2 seed-decision finding's fixture
        // (`.claude/plans/pdf-to-text-ocr-v1.md` §"D1.2 seed decision"). Each is a
        // single-code pass-through on eng.lstm's recoder.
        let ids = [91_u32, 97, 92];
        let codes: Vec<usize> = ids
            .iter()
            .map(|&id| {
                let rc = recoder.encode(id).expect("id in range");
                assert_eq!(rc.length(), 1, "eng.lstm recoder is pass-through");
                usize::try_from(rc.codes()[0]).expect("non-negative code")
            })
            .collect();
        let rows: Vec<Vec<f32>> = codes
            .iter()
            .map(|&c| row(n, c, ENG_NULL_CHAR as usize))
            .collect();
        let mut beam =
            RecodeBeamSearch::new_with_dict(&recoder, ENG_NULL_CHAR, false, dict, charset);
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        beam.decode_with_dict(&refs, 2.25, -0.085, K_MIN_CERTAINTY);

        // At least one node in the dawg beams at every timestep after the first
        // should carry attached dawg positions: 't' seeds a fresh word (attaches
        // `default_dawgs()`), and each successive letter's `def_letter_is_okay`
        // continuation reattaches the updated set — this exercises both the
        // seed (`ContinueDawg`'s `uni_prev == None` arm) and the transfer
        // (`uni_prev.dawgs` arm) the brief asked to verify.
        for t in 0..beam.beam_size {
            let mut found = false;
            for cont in 0..NC_COUNT {
                let index = beam_index(true, cont, 0);
                let sz = beam.beams[t][index].len();
                for h in 0..sz {
                    let node_idx = beam.beams[t][index].get(h).idx;
                    if beam.arena[node_idx as usize].dawgs.is_some() {
                        found = true;
                    }
                }
            }
            assert!(
                found,
                "timestep {t} should have a dawg-beam node with attached dawg positions"
            );
        }

        // And the decode actually recognizes "the" end-to-end through the dict
        // path (a real, if secondary, byte-parity-relevant sanity check).
        let (uids, _certs, _ratings, _xcoords) = beam.extract_best_path_as_unichar_ids();
        assert_eq!(uids, vec![91, 97, 92]);
    }

    #[test]
    fn dict_word_beats_higher_raw_probability_non_word_under_dict_ratio() {
        // A genuine flip: WITHOUT the dictionary, per-step "junk" beats "the"
        // outright (junk has the higher raw per-step probability at every
        // timestep, so the plain CTC argmax-of-the-beam favours it). WITH the
        // dictionary (kDictRatio = 2.25 scaling every NON-dawg continuation's
        // certainty), "the"'s unscaled dawg-path score overtakes junk's scaled
        // non-dawg score — this is exactly the production
        // `Tesseract::LSTMRecognizeWord` dict-ratio mechanism (`ContinueUnichar`'s
        // unconditional `cert * dict_ratio` on the non-dawg push vs `ContinueDawg`'s
        // unscaled dawg push).
        let Some(dict) = load_real_dict() else {
            eprintln!("skipping: /tmp/eng.lstm-*-dawg not present");
            return;
        };
        let Some(charset) = load_real_charset() else {
            eprintln!("skipping: /tmp/eng.lstm-unicharset not present");
            return;
        };
        let Some(recoder) = load_real_recoder() else {
            eprintln!("skipping: /tmp/eng.lstm-recoder not present");
            return;
        };
        let n = recoder.code_range() as usize;
        let word_ids = [91_u32, 97, 92]; // "the"
                                         // "xqz" — three letters that don't spell "the" nor (checked below) form a
                                         // valid dict word themselves; chosen only to be a DIFFERENT plain-text
                                         // sequence than "the", not to be linguistically meaningful.
        let junk_chars = ['x', 'q', 'z'];
        let junk_ids: Vec<u32> = junk_chars
            .iter()
            .map(|&c| {
                charset
                    .unichar_to_id(&c.to_string())
                    .unwrap_or_else(|| panic!("charset has {c:?}"))
            })
            .collect();

        // Self-verifying precondition: "xqz" must not be a valid, complete dict
        // word (so `extract_best_node`'s dawg-validity filter would reject it as
        // a dawg-path candidate even if one existed) — the flip below must come
        // from the dict_ratio mechanism, not from an accidental real word.
        let mut active = dict.default_dawgs(false);
        let mut junk_valid_end = false;
        for (i, &id) in junk_ids.iter().enumerate() {
            let word_end = i + 1 == junk_ids.len();
            let (updated, _perm, valid_end) =
                dict.def_letter_is_okay(&active, &charset, id, word_end, PermuterType::NoPerm);
            active = updated;
            junk_valid_end = valid_end;
            if active.is_empty() {
                break;
            }
        }
        assert!(!junk_valid_end, "\"xqz\" must not be a valid dict word");

        let word_codes: Vec<usize> = word_ids
            .iter()
            .map(|&id| {
                let rc = recoder.encode(id).expect("id in range");
                assert_eq!(rc.length(), 1);
                usize::try_from(rc.codes()[0]).unwrap()
            })
            .collect();
        let junk_codes: Vec<usize> = junk_ids
            .iter()
            .map(|&id| {
                let rc = recoder.encode(id).expect("id in range");
                assert_eq!(rc.length(), 1);
                usize::try_from(rc.codes()[0]).unwrap()
            })
            .collect();

        // junk's raw per-step probability (0.85) is HIGHER than the word's
        // (0.75) — junk wins a plain (non-dict) decode.
        let rows: Vec<Vec<f32>> = (0..3)
            .map(|i| row_two(n, junk_codes[i], 0.85, word_codes[i], 0.75))
            .collect();
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();

        // Without dict: junk wins.
        let mut plain = RecodeBeamSearch::new(&recoder, ENG_NULL_CHAR, false);
        plain.decode(&refs, 1.0, 0.0);
        let (plain_uids, ..) = plain.extract_best_path_as_unichar_ids();
        let junk_ids_i32: Vec<i32> = junk_ids.iter().map(|&id| id as i32).collect();
        assert_eq!(
            plain_uids, junk_ids_i32,
            "without the dictionary, the higher raw-probability sequence wins"
        );

        // With dict (kDictRatio = 2.25): "the" overtakes junk.
        let mut dicted =
            RecodeBeamSearch::new_with_dict(&recoder, ENG_NULL_CHAR, false, dict, charset);
        dicted.decode_with_dict(&refs, 2.25, 0.0, K_MIN_CERTAINTY);
        let (dict_uids, ..) = dicted.extract_best_path_as_unichar_ids();
        let word_ids_i32: Vec<i32> = word_ids.iter().map(|&id| id as i32).collect();
        assert_eq!(
            dict_uids, word_ids_i32,
            "with the dictionary active, dict_ratio scaling lets the dict word overtake \
             the higher raw-probability non-word sequence"
        );
    }

    // ---- Word/box output surface: character_boundaries + ExtractBestPathAsWords ----

    /// `calculateCharBoundaries` (`recodebeam.cpp:187-198`) on hand-picked
    /// `starts`/`ends`, matching the shape a two-character decode of overall
    /// width 10 would produce.
    #[test]
    fn char_boundaries_bracket_each_character() {
        let starts = [0_i32, 4, 10];
        let ends = [3_i32, 7];
        let bounds = calculate_char_boundaries(&starts, &ends, 10);
        // char 0: middle = (starts[1] - ends[0]) / 2 = (4 - 3) / 2 = 0 -> 3.
        // char 1: middle = (starts[2] - ends[1]) / 2 = (10 - 7) / 2 = 1 -> 8, but
        // the last computed boundary is dropped and replaced by max_width (the
        // C++ pop_back + push(maxWidth) dance, `recodebeam.cpp:196-197`).
        assert_eq!(bounds, vec![0, 3, 10]);
    }

    /// `character_boundaries_` from a real (non-dict) decode: one more entry
    /// than there are characters, starting at 0 and ending at the decode
    /// width, non-decreasing throughout.
    #[test]
    fn boundary_vector_length_matches_unichar_count_on_a_synthetic_decode() {
        let recoder = passthrough_recoder(5);
        let rows = [row(5, 1, 4), row(5, 2, 4), row(5, 3, 4)];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let mut beam = RecodeBeamSearch::new(&recoder, 4, false);
        beam.decode(&refs, 1.0, 0.0);
        let best_nodes = beam.extract_path(beam.extract_best_node());
        let (unichar_ids, _certs, _ratings, _xcoords, boundaries) =
            beam.extract_path_as_unichar_ids_with_boundaries(&best_nodes);
        assert_eq!(unichar_ids, vec![1, 2, 3]);
        assert_eq!(
            boundaries.len(),
            unichar_ids.len() + 1,
            "one more boundary than characters"
        );
        assert_eq!(boundaries[0], 0);
        assert_eq!(
            *boundaries.last().expect("non-empty"),
            best_nodes.len() as i32,
            "last boundary = decode width"
        );
        assert!(
            boundaries.windows(2).all(|w| w[0] <= w[1]),
            "boundaries are non-decreasing"
        );
    }

    /// Hand-builds a `RecodeBeamSearch` whose "decode" is a single fixed chain
    /// of nodes, bypassing `decode`/`decode_with_dict` entirely — used to test
    /// [`RecodeBeamSearch::extract_best_path_as_words`]'s SPLIT LOGIC in
    /// isolation from beam-search dynamics (already covered by
    /// `decodes_a_clean_sequence`,
    /// `dict_dawgs_attach_and_transfer_through_the_beam`, and the space-split
    /// test below, which all exercise a real `decode`). `nodes` is `(code,
    /// unichar_id, permuter, start_of_word, duplicate, certainty)` in
    /// ROOT→LEAF order; each becomes one `RecodeNode` chained via `prev`, and
    /// the last one is placed as the sole entry of the final timestep's
    /// `beam_index(false, NC_ANYTHING, 0)` heap so
    /// `extract_best_node`/`extract_path` walk exactly this chain.
    fn hand_built_beam<'a>(
        recoder: &'a UnicharCompress,
        null_char: i32,
        nodes: &[(i32, i32, PermuterType, bool, bool, f32)],
    ) -> RecodeBeamSearch<'a> {
        let mut beam = RecodeBeamSearch::new(recoder, null_char, false);
        let mut prev = None;
        let mut score = 0.0_f32;
        for &(code, unichar_id, permuter, start_of_word, duplicate, certainty) in nodes {
            score += certainty;
            let code_hash = beam.compute_code_hash(code, duplicate, prev);
            let node = RecodeNode {
                code,
                unichar_id,
                permuter,
                start_of_dawg: false,
                start_of_word,
                end_of_word: false,
                duplicate,
                certainty,
                score,
                prev,
                dawgs: None,
                code_hash,
            };
            let idx = beam.arena.len() as u32;
            beam.arena.push(node);
            prev = Some(idx);
        }
        let last_idx = prev.expect("hand_built_beam needs at least one node");
        let width = nodes.len();
        beam.beam_size = width;
        beam.beams = (0..width)
            .map(|_| core::array::from_fn(|_| MinHeap::default()))
            .collect();
        let index = beam_index(false, NC_ANYTHING, 0);
        beam.beams[width - 1][index].push(HeapPair {
            key: f64::from(score),
            idx: last_idx,
        });
        beam
    }

    #[test]
    fn word_split_on_space_via_real_decode() {
        // codes 0..5, null=5; charset ids: NULL(space)=0, a=1, b=2, c=3, d=4.
        let recoder = passthrough_recoder(6);
        let charset = crate::UniCharSet::load_from_str(
            "5\nNULL 0 Common\na 0 Latin\nb 0 Latin\nc 0 Latin\nd 0 Latin\n",
        )
        .expect("valid unicharset");
        let mut beam = RecodeBeamSearch::new(&recoder, 5, false);
        let rows = [row(6, 1, 5), row(6, 2, 5), row(6, 0, 5), row(6, 3, 5)];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        beam.decode(&refs, 1.0, 0.0);
        let words = beam.extract_best_path_as_words((0, 0, 1000, 36), 1.0, &charset);
        assert_eq!(words.len(), 2, "the space splits the run into two words");
        assert_eq!(
            words[0].unichar_ids,
            vec![1, 2],
            "\"a\", \"b\" before the space"
        );
        assert!(!words[0].leading_space);
        assert_eq!(words[1].unichar_ids, vec![3], "\"c\" after the space");
        assert!(
            words[1].leading_space,
            "the second word is preceded by the space character"
        );
        // Boxes are non-empty and left-to-right monotone within each word.
        assert_eq!(words[0].char_boxes.len(), 2);
        assert!(words[0].char_boxes[0].0 <= words[0].char_boxes[1].0);
    }

    #[test]
    fn word_split_on_start_of_word_flag_without_a_space() {
        let recoder = passthrough_recoder(5);
        let charset = crate::UniCharSet::load_from_str("3\nNULL 0 Common\na 0 Latin\nb 0 Latin\n")
            .expect("valid unicharset");
        // Both "a" and "b" are Latin (space-delimited) and TOP_CHOICE_PERM, so
        // the ONLY thing that can force a split between them is the
        // start_of_word flag on b's node — the non-space-delimited-language
        // dawg-restart case (`recodebeam.cpp:283-284`), isolated here from the
        // TOP_CHOICE/script branch of the same condition (next test).
        let nodes = [
            (1_i32, 1_i32, TOP_CHOICE_PERM, false, false, -0.1_f32), // "a"
            (2_i32, 2_i32, TOP_CHOICE_PERM, true, false, -0.1_f32),  // "b", start_of_word
        ];
        let beam = hand_built_beam(&recoder, 99, &nodes);
        let words = beam.extract_best_path_as_words((0, 0, 1000, 36), 1.0, &charset);
        assert_eq!(
            words.len(),
            2,
            "start_of_word forces a split even with no space"
        );
        assert_eq!(words[0].unichar_ids, vec![1]);
        assert_eq!(words[1].unichar_ids, vec![2]);
        assert!(!words[0].leading_space);
        assert!(
            !words[1].leading_space,
            "no actual space character was involved in this split"
        );
    }

    #[test]
    fn word_split_on_non_space_delimited_script_without_dict() {
        let recoder = passthrough_recoder(5);
        let charset =
            crate::UniCharSet::load_from_str("4\nNULL 0 Common\na 0 Latin\nb 0 Han\nc 0 Latin\n")
                .expect("valid unicharset");
        // "b" is Han (not space-delimited); TOP_CHOICE_PERM (the always-active
        // non-dict permuter) plus a non-space-delimited neighbor on EITHER
        // side forces a break (`recodebeam.cpp:286-290`), so a Han character
        // between two Latin ones splits into three one-character words.
        let nodes = [
            (1_i32, 1_i32, TOP_CHOICE_PERM, false, false, -0.1_f32), // "a" (Latin)
            (2_i32, 2_i32, TOP_CHOICE_PERM, false, false, -0.1_f32), // "b" (Han)
            (3_i32, 3_i32, TOP_CHOICE_PERM, false, false, -0.1_f32), // "c" (Latin)
        ];
        let beam = hand_built_beam(&recoder, 99, &nodes);
        let words = beam.extract_best_path_as_words((0, 0, 1000, 36), 1.0, &charset);
        assert_eq!(
            words.len(),
            3,
            "a non-space-delimited (Han) neighbor forces a split on both sides"
        );
        assert_eq!(words[0].unichar_ids, vec![1]);
        assert_eq!(words[1].unichar_ids, vec![2]);
        assert_eq!(words[2].unichar_ids, vec![3]);
        assert!(words.iter().all(|w| !w.leading_space));
    }
}
