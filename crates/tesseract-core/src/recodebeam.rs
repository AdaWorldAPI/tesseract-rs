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

use lance_graph_contract::unicharcompress::{RecodedCharId, UnicharCompress};

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

/// `NodeContinuation::NC_ANYTHING` (`recodebeam.h:73`): this node used only its
/// own score, so anything may follow.
const NC_ANYTHING: usize = 0;
/// `NC_ONLY_DUP` (`recodebeam.h:74`): combined a score without a stand-alone
/// duplicate before, so must be followed by a stand-alone duplicate.
const NC_ONLY_DUP: usize = 1;
/// `NC_NO_DUP` (`recodebeam.h:77`): combined a score after a stand-alone, so can
/// only be followed by a non-duplicate.
const NC_NO_DUP: usize = 2;

/// `PermuterType::TOP_CHOICE_PERM` (`ratngs.h`). The non-dict path never uses any
/// other permuter, so the concrete value only ever participates in the (always
/// equal) [`RecodeBeamSearch::update_heap_if_matched`] identity check.
const TOP_CHOICE_PERM: u8 = 2;

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
/// becomes safe indices. Dawg-only fields (`start_of_word`/`end_of_word`/`dawgs`)
/// are omitted — the non-dict path never sets them.
#[derive(Clone, Debug)]
struct RecodeNode {
    /// The re-encoded code = index into the network output.
    code: i32,
    /// The decoded unichar-id (valid only at the final code of a sequence).
    unichar_id: i32,
    /// The permuter (always [`TOP_CHOICE_PERM`] in the non-dict path).
    permuter: u8,
    /// True if this is the initial dawg state (always `false` in the non-dict path;
    /// retained for the [`RecodeBeamSearch::update_heap_if_matched`] identity).
    start_of_dawg: bool,
    /// True if `code` is a duplicate of `prev.code` (CTC fold-on-the-fly).
    duplicate: bool,
    /// Total certainty of the path to this position. (The per-position `certainty`
    /// of `RecodeNode` is dropped — the labels path never reads it; it returns when
    /// `ExtractBestPathAsUnicharIds` lands.)
    score: f32,
    /// The previous node in the chain, as an arena index.
    prev: Option<u32>,
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
    /// Every node across all timesteps (the lattice backing store); `prev` indexes
    /// here.
    arena: Vec<RecodeNode>,
    /// `beam_[t].beams_[k]` — `K_NUM_BEAMS` heaps per timestep.
    beams: Vec<[MinHeap; K_NUM_BEAMS]>,
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
            arena: Vec::new(),
            beams: Vec::new(),
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
    /// and `cert_offset = 0.0` give a plain decode.
    pub fn decode(&mut self, outputs: &[&[f32]], dict_ratio: f32, cert_offset: f32) {
        self.dict_ratio = dict_ratio;
        self.cert_offset = cert_offset;
        self.beam_size = 0;
        self.arena.clear();
        self.beams.clear();
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
        }
        self.beam_size = t + 1;
        for heap in &mut self.beams[t] {
            heap.clear();
        }
        if t == 0 {
            // The first step can only use singles and initials (dict null → no dawg).
            self.continue_context(
                None,
                beam_index(false, NC_ANYTHING, 0),
                outputs,
                TopN::Top2,
                t,
            );
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
        // The dawg best-initial special case is dict-only; skipped (dict null).
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

    /// `ContinueUnichar` (`recodebeam.cpp:1012`), non-dawg branch: push a new
    /// unichar node to the length-0 beam for `cont` (the dict/dawg branch is skipped).
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
            return; // ContinueDawg — dict-only, never reached.
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
            cert * self.dict_ratio,
            prev,
        );
    }

    /// `PushDupOrNoDawgIfBetter` (`recodebeam.cpp:1166`), non-dawg branch: scale by
    /// `dict_ratio`, drop below the certainty floor (unless null), then push to the
    /// length-`length` beam for `cont`.
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
        if use_dawgs {
            return; // dict-only.
        }
        let index = beam_index(false, cont, length);
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
                dup,
                cert,
                prev,
            );
        }
    }

    /// `PushHeapIfBetter` (`recodebeam.cpp:1190`): `score = cert + prev.score`; if
    /// there is room or it beats the heap's worst, either update a matching node
    /// (same code/hash/permuter/dawg-start) or push and evict the worst when full.
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
        permuter: u8,
        start_of_dawg: bool,
        duplicate: bool,
        cert: f32,
        prev: Option<u32>,
    ) {
        let prev_score = prev.map_or(0.0, |p| self.arena[p as usize].score);
        let score = cert + prev_score;
        let heap_len = self.beams[t][index].len();
        let better = if heap_len < max_size {
            true
        } else {
            let top_idx = self.beams[t][index].peek_top().idx;
            score > self.arena[top_idx as usize].score
        };
        if !better {
            return;
        }
        let code_hash = self.compute_code_hash(code, duplicate, prev);
        let node = RecodeNode {
            code,
            unichar_id,
            permuter,
            start_of_dawg,
            duplicate,
            score,
            prev,
            code_hash,
        };
        if self.update_heap_if_matched(t, index, &node) {
            return;
        }
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

    /// `ExtractBestPaths` (`recodebeam.cpp:1279`), non-dict: the highest-scoring
    /// node in the last beam's NC_ANYTHING + NC_NO_DUP heaps (NC_ONLY_DUP skipped;
    /// dawg beams empty). Returns its arena index, or `None` on an empty decode.
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
                    if is_dawg {
                        continue; // dawg validity — non-dict beams are empty.
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
    /// labels are recoded codes; feed them (via [`RecodedCharId::from_codes`]) to
    /// [`recoded_to_text`](crate::recoded_to_text) for the string.
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
}
