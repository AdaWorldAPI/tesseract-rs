//! The Dict-lite walker — `Dict::default_dawgs` + `Dict::def_letter_is_okay`
//! (`dict/dict.{h,cpp}`), the character-by-character DAWG-beam-step primitive
//! `RecodeBeamSearch::ContinueDawg` consumes in production `recognize_line`.
//!
//! This is **beam-coupled compute-free logic**, like [`crate::recodebeam`] — it
//! rides on top of the Core's dawg TABLE ([`SquishedDawg`], already
//! byte-parity-proven load + `edge_char_of` traversal), but the walker itself
//! lives here, not in the Core, because it is a *use* of the table rather than
//! table content.
//!
//! # Byte-parity surface
//!
//! Transcodes:
//! - `Dict::GetStartingNode` (`dict.h:397-406`)
//! - `Dict::char_for_dawg` (`dict.h:411-421`)
//! - `Dict::default_dawgs` (`dict.cpp:625-647`)
//! - `Dict::def_letter_is_okay` (`dict.cpp:407-571`)
//! - `kDawgSuccessors` (`dawg.h:87-92`) + the `FinishLoad` successor-list build
//!   (`dict.cpp:363-375`, specialised to a single-language load so the
//!   `dawg->lang() == other->lang()` guard is always true)
//!
//! `DawgArgs` (`dawg.h:414-419`) is not carried as a struct — its four fields
//! become the walker's parameters/return: `active_dawgs` is `active: &[DawgPosition]`,
//! `updated_dawgs`/`permuter`/`valid_end` are the returned tuple.
//!
//! `ProcessPatternEdges` (`dict.cpp:552-571`, the `DAWG_TYPE_PATTERN` arm) is
//! **unreachable for `eng`**: `eng.lstm` ships no pattern dawg (only word, punc,
//! number), so [`DictLite`] never holds a [`DawgType::Pattern`] entry and the
//! branch below is a documented no-op — kept as a structural marker of the gap,
//! not silently dropped, per the D1.2b brief.

use lance_graph_contract::dawg::{
    DawgError, DawgType, NodeRef, PermuterType, SquishedDawg, NO_EDGE,
};
use lance_graph_contract::unicharset::UniCharSet;

/// `Dawg::kPatternUnicharID` (`dawg.h:113-117`) — the sentinel `UNICHAR_ID`
/// used both to guard against pattern-poisoned input and to probe whether a
/// punctuation dawg has a "some word may start here" pattern edge.
const K_PATTERN_UNICHAR_ID: u32 = 0;

/// `kDawgSuccessors` (`dawg.h:87-92`) — which dawg types may continue directly
/// out of a given dawg type, indexed `[from][to]` by [`DawgType`] ordinal.
/// Punctuation subsumes word and number; word and number each fall back to
/// punctuation; pattern has no successors.
const DAWG_SUCCESSORS: [[bool; 4]; 4] = [
    [false, true, true, false],   // DAWG_TYPE_PUNCTUATION
    [true, false, false, false],  // DAWG_TYPE_WORD
    [true, false, false, false],  // DAWG_TYPE_NUMBER
    [false, false, false, false], // DAWG_TYPE_PATTERN
];

/// `tesseract::DawgPosition` (`dawg.h:355-376`) — a single active position in
/// one dawg (and, optionally, a shadow position in the punctuation dawg it is
/// currently paired with). Field order matches the C++ struct layout, NOT the
/// constructor argument order (`dawg_ref, punc_ref, dawg_index, punc_index,
/// back_to_punc` vs constructor `dawg_idx, dawgref, punc_idx, puncref,
/// backtopunc`) — see [`DawgPosition::new`] for the constructor-order
/// convenience that mirrors every C++ call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DawgPosition {
    /// `EDGE_REF dawg_ref` — the edge just taken in the primary dawg, or
    /// [`NO_EDGE`] if no primary dawg edge has been taken yet.
    pub dawg_ref: NodeRef,
    /// `EDGE_REF punc_ref` — the edge just taken in the punctuation dawg, or
    /// [`NO_EDGE`].
    pub punc_ref: NodeRef,
    /// `int8_t dawg_index` — index into [`DictLite`]'s dawg vector, or `-1` if
    /// no primary dawg is chosen yet (pure punctuation-dawg position).
    pub dawg_index: i8,
    /// `int8_t punc_index` — index into [`DictLite`]'s dawg vector for the
    /// paired punctuation dawg, or `-1` if none is paired.
    pub punc_index: i8,
    /// Whether the main word has already ended and this position has
    /// returned to the punctuation dawg to continue matching trailing
    /// punctuation.
    pub back_to_punc: bool,
}

impl DawgPosition {
    /// Constructs a position using the C++ constructor's argument order
    /// (`DawgPosition(dawg_idx, dawgref, punc_idx, puncref, backtopunc)`,
    /// `dawg.h:357-363`) so every call site below can be transcoded
    /// positionally without re-deriving the field permutation each time.
    #[must_use]
    pub fn new(
        dawg_index: i8,
        dawg_ref: NodeRef,
        punc_index: i8,
        punc_ref: NodeRef,
        back_to_punc: bool,
    ) -> Self {
        Self {
            dawg_ref,
            punc_ref,
            dawg_index,
            punc_index,
            back_to_punc,
        }
    }
}

/// `DawgPositionVector::add_unique` (`dawg.h:383-397`) — linear dedup on full
/// structural equality; pushes and returns `true` if `pos` is not already
/// present, otherwise leaves `positions` untouched and returns `false`. The
/// C++ debug-print argument is dropped (log-only, no behavioural effect).
fn add_unique(positions: &mut Vec<DawgPosition>, pos: DawgPosition) -> bool {
    if positions.contains(&pos) {
        return false;
    }
    positions.push(pos);
    true
}

/// `Dict::GetStartingNode` (`dict.h:397-406`) — maps an `EDGE_REF` already
/// taken in `dawg` to the `NODE_REF` to search from next: [`NO_EDGE`] means
/// "beginning to explore the dawg" (node 0); landing back on node 0 (a dawg's
/// root, reachable only as an end-of-word wraparound) means "end of word",
/// reported as [`NO_EDGE`] so the caller's own `edge_char_of` calls fail
/// closed instead of silently re-walking the root.
fn get_starting_node(dawg: &SquishedDawg, edge_ref: NodeRef) -> NodeRef {
    if edge_ref == NO_EDGE {
        return 0;
    }
    let node = dawg.next_node(edge_ref as usize) as NodeRef;
    if node == 0 {
        NO_EDGE
    } else {
        node
    }
}

/// `Dict::char_for_dawg` (`dict.h:411-421`) — the unichar substitution a given
/// dawg type expects: the number dawg matches any digit against
/// [`K_PATTERN_UNICHAR_ID`], every other dawg type matches the unichar
/// verbatim. Both call sites in [`DictLite::def_letter_is_okay`] always pass a
/// real dawg (the C++ `!dawg` early-return arm is never reached from either
/// call site), so this takes `dawg` directly rather than `Option`.
fn char_for_dawg(charset: &UniCharSet, ch: u32, dawg: &SquishedDawg) -> u32 {
    match dawg.dawg_type() {
        DawgType::Number if charset.get_isdigit(ch) => K_PATTERN_UNICHAR_ID,
        _ => ch,
    }
}

/// A minimal, in-memory `Dict` — just enough of Tesseract's dictionary layer
/// to walk word/punctuation/number dawgs one `UNICHAR_ID` at a time, matching
/// the DAWG beam step `RecodeBeamSearch::ContinueDawg` drives in production.
#[derive(Debug, Clone)]
pub struct DictLite {
    /// The loaded dawgs, in Tesseract's own `LoadLSTM` order:
    /// `[punctuation, word, number]` (`dict.cpp:292-313`, no user-words/
    /// user-patterns dawgs in this lite loader). Positions' `dawg_index`/
    /// `punc_index` fields are indices into this vector, so the ORDER here is
    /// load-bearing: it must match libtesseract's own `dawgs_` order for a
    /// byte-parity dump to agree on which index names which dawg.
    dawgs: Vec<SquishedDawg>,
    /// `Dict::successors_[i]` (`dict.cpp:363-375`) — for each dawg `i`, the
    /// indices of dawgs it may hand off to, derived from [`DAWG_SUCCESSORS`]
    /// over the actual loaded [`DawgType`]s (never hardcoded by position).
    successors: Vec<Vec<usize>>,
}

impl DictLite {
    /// Loads the word/punctuation/number dawgs from their raw little-endian
    /// bytes (each a standalone `SquishedDawg` component, `dawg.cpp:313-352`),
    /// tags each with its [`DawgType`]/[`PermuterType`] role (a constructor
    /// argument in real Tesseract, not stored on disk — `dawg.h:410-420`), and
    /// derives the successor lists ([`DAWG_SUCCESSORS`] over the loaded types).
    ///
    /// # Errors
    ///
    /// Propagates [`DawgError`] from any of the three component loads.
    pub fn from_components(word: &[u8], punc: &[u8], number: &[u8]) -> Result<Self, DawgError> {
        let (punc_dawg, _) = SquishedDawg::from_le_bytes(punc)?;
        let punc_dawg = punc_dawg
            .with_type(DawgType::Punctuation)
            .with_permuter(PermuterType::PuncPerm);
        let (word_dawg, _) = SquishedDawg::from_le_bytes(word)?;
        let word_dawg = word_dawg
            .with_type(DawgType::Word)
            .with_permuter(PermuterType::SystemDawgPerm);
        let (number_dawg, _) = SquishedDawg::from_le_bytes(number)?;
        let number_dawg = number_dawg
            .with_type(DawgType::Number)
            .with_permuter(PermuterType::NumberPerm);

        // Load order matches `Dict::LoadLSTM` (`dict.cpp:292-313`): punc,
        // system(word), number.
        let dawgs = vec![punc_dawg, word_dawg, number_dawg];
        let successors = Self::compute_successors(&dawgs);
        Ok(Self { dawgs, successors })
    }

    /// `Dict::FinishLoad`'s successor-list build (`dict.cpp:363-375`),
    /// specialised to a single-language load (the `dawg->lang() ==
    /// other->lang()` guard is always true here, so it is omitted).
    fn compute_successors(dawgs: &[SquishedDawg]) -> Vec<Vec<usize>> {
        dawgs
            .iter()
            .map(|dawg| {
                let from = dawg.dawg_type() as usize;
                dawgs
                    .iter()
                    .enumerate()
                    .filter_map(|(j, other)| {
                        DAWG_SUCCESSORS[from][other.dawg_type() as usize].then_some(j)
                    })
                    .collect()
            })
            .collect()
    }

    /// `Dict::default_dawgs` (`dict.cpp:625-647`) — the dawg positions that
    /// could contain the beginning of a word: production `RecodeBeamSearch::
    /// ContinueDawg` seeds word-start from exactly this (not
    /// `init_active_dawgs`, which is the legacy-recognizer-only seed; see the
    /// D1.2 seed-decision finding this walker was built against). Punctuation
    /// subsumes word/number whenever the punctuation dawg itself has a
    /// "pattern" edge at the root (i.e. it can represent "a word may start
    /// here"); `suppress_patterns` additionally drops any [`DawgType::Pattern`]
    /// entries (always a no-op here — `eng.lstm` has none).
    #[must_use]
    pub fn default_dawgs(&self, suppress_patterns: bool) -> Vec<DawgPosition> {
        let punc_index = self
            .dawgs
            .iter()
            .position(|d| d.dawg_type() == DawgType::Punctuation);
        let punc_dawg_available = punc_index.is_some_and(|i| {
            self.dawgs[i]
                .edge_char_of(0, K_PATTERN_UNICHAR_ID, true)
                .is_some()
        });

        let mut out = Vec::new();
        for (i, dawg) in self.dawgs.iter().enumerate() {
            let dawg_ty = dawg.dawg_type();
            if suppress_patterns && dawg_ty == DawgType::Pattern {
                continue;
            }
            let subsumed_by_punc =
                DAWG_SUCCESSORS[DawgType::Punctuation as usize][dawg_ty as usize];
            if dawg_ty == DawgType::Punctuation {
                out.push(DawgPosition::new(-1, NO_EDGE, i as i8, NO_EDGE, false));
            } else if !punc_dawg_available || !subsumed_by_punc {
                out.push(DawgPosition::new(i as i8, NO_EDGE, -1, NO_EDGE, false));
            }
        }
        out
    }

    /// `Dict::def_letter_is_okay` (`dict.cpp:407-571`) — advances every active
    /// dawg position by one `unichar_id`, returning the surviving positions,
    /// the winning [`PermuterType`], and whether any surviving position is a
    /// valid word end. `permuter_in` is the walker's own `dawg_args->permuter`
    /// carried in (production `RecodeBeamSearch` resets this fresh per beam
    /// node — see the byte-parity oracle, which constructs a fresh
    /// `DawgArgs(..., NO_PERM)` every step).
    #[must_use]
    pub fn def_letter_is_okay(
        &self,
        active: &[DawgPosition],
        charset: &UniCharSet,
        unichar_id: u32,
        word_end: bool,
        permuter_in: PermuterType,
    ) -> (Vec<DawgPosition>, PermuterType, bool) {
        // Do not accept words that contain kPatternUnicharID (otherwise
        // pattern dawgs would not function correctly). The C++ additionally
        // guards `unichar_id == INVALID_UNICHAR_ID` (-1); that sentinel is not
        // representable under this function's `u32` signature, so only the
        // pattern-id guard is reachable here.
        if unichar_id == K_PATTERN_UNICHAR_ID {
            return (Vec::new(), PermuterType::NoPerm, false);
        }

        let mut curr_perm = PermuterType::NoPerm;
        let mut updated: Vec<DawgPosition> = Vec::new();
        let mut valid_end = false;

        for pos in active {
            let punc_dawg = (pos.punc_index >= 0).then(|| &self.dawgs[pos.punc_index as usize]);
            let dawg = (pos.dawg_index >= 0).then(|| &self.dawgs[pos.dawg_index as usize]);

            let Some(dawg) = dawg else {
                let Some(punc_dawg) = punc_dawg else {
                    // "Received DawgPosition with no dawg or punc_dawg." —
                    // shouldn't happen; skip defensively.
                    continue;
                };
                // We're in the punctuation dawg. A core dawg has not been
                // chosen.
                let punc_node = get_starting_node(punc_dawg, pos.punc_ref);
                let punc_transition_edge =
                    punc_dawg.edge_char_of(punc_node, K_PATTERN_UNICHAR_ID, word_end);
                if let Some(punc_transition_edge) = punc_transition_edge {
                    // Find all successors, and see which can transition.
                    for &sdawg_index in &self.successors[pos.punc_index as usize] {
                        let sdawg = &self.dawgs[sdawg_index];
                        let ch = char_for_dawg(charset, unichar_id, sdawg);
                        if let Some(dawg_edge) = sdawg.edge_char_of(0, ch, word_end) {
                            add_unique(
                                &mut updated,
                                DawgPosition::new(
                                    sdawg_index as i8,
                                    dawg_edge as NodeRef,
                                    pos.punc_index,
                                    punc_transition_edge as NodeRef,
                                    false,
                                ),
                            );
                            if sdawg.permuter() > curr_perm {
                                curr_perm = sdawg.permuter();
                            }
                            if sdawg.end_of_word(dawg_edge)
                                && punc_dawg.end_of_word(punc_transition_edge)
                            {
                                valid_end = true;
                            }
                        }
                    }
                }
                let punc_edge = punc_dawg.edge_char_of(punc_node, unichar_id, word_end);
                if let Some(punc_edge) = punc_edge {
                    add_unique(
                        &mut updated,
                        DawgPosition::new(-1, NO_EDGE, pos.punc_index, punc_edge as NodeRef, false),
                    );
                    if PermuterType::PuncPerm > curr_perm {
                        curr_perm = PermuterType::PuncPerm;
                    }
                    if punc_dawg.end_of_word(punc_edge) {
                        valid_end = true;
                    }
                }
                continue;
            };

            if let Some(punc_dawg) = punc_dawg {
                if dawg.end_of_word(pos.dawg_ref as usize) {
                    // We can end the main word here. If we can continue on
                    // the punc ref, add that possibility.
                    let punc_node = get_starting_node(punc_dawg, pos.punc_ref);
                    let punc_edge = if punc_node == NO_EDGE {
                        None
                    } else {
                        punc_dawg.edge_char_of(punc_node, unichar_id, word_end)
                    };
                    if let Some(punc_edge) = punc_edge {
                        add_unique(
                            &mut updated,
                            DawgPosition::new(
                                pos.dawg_index,
                                pos.dawg_ref,
                                pos.punc_index,
                                punc_edge as NodeRef,
                                true,
                            ),
                        );
                        if dawg.permuter() > curr_perm {
                            curr_perm = dawg.permuter();
                        }
                        if punc_dawg.end_of_word(punc_edge) {
                            valid_end = true;
                        }
                    }
                }
            }

            if pos.back_to_punc {
                continue;
            }

            // `ProcessPatternEdges` (dict.cpp:552-571): UNREACHABLE FOR ENG.
            // `eng.lstm` ships no pattern dawg, so `dawg.dawg_type()` is never
            // `Pattern` here in practice. Kept as an explicit branch (rather
            // than silently folded into the fallthrough) to mark the known
            // falsifier gap: a legacy config with a user pattern dawg would
            // need this arm transcoded for correctness.
            if dawg.dawg_type() == DawgType::Pattern {
                continue;
            }

            // Find the edge out of the node for the unichar_id.
            let node = get_starting_node(dawg, pos.dawg_ref);
            let edge = if node == NO_EDGE {
                None
            } else {
                dawg.edge_char_of(node, char_for_dawg(charset, unichar_id, dawg), word_end)
            };

            if let Some(edge) = edge {
                if word_end && punc_dawg.is_some_and(|p| !p.end_of_word(pos.punc_ref as usize)) {
                    continue;
                }
                if dawg.permuter() > curr_perm {
                    curr_perm = dawg.permuter();
                }
                if dawg.end_of_word(edge)
                    && punc_dawg.is_none_or(|p| p.end_of_word(pos.punc_ref as usize))
                {
                    valid_end = true;
                }
                add_unique(
                    &mut updated,
                    DawgPosition::new(
                        pos.dawg_index,
                        edge as NodeRef,
                        pos.punc_index,
                        pos.punc_ref,
                        false,
                    ),
                );
            }
        }

        // Update the permuter if it used to be NO_PERM or became NO_PERM, or
        // if we found the current letter in a non-punctuation dawg. Keep the
        // old value if it is COMPOUND_PERM (dict.cpp:559-566).
        let mut out_permuter = permuter_in;
        if out_permuter == PermuterType::NoPerm
            || curr_perm == PermuterType::NoPerm
            || (curr_perm != PermuterType::PuncPerm && out_permuter != PermuterType::CompoundPerm)
        {
            out_permuter = curr_perm;
        }

        (updated, out_permuter, valid_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_real_dawgs() -> Option<DictLite> {
        let word = std::fs::read("/tmp/eng.lstm-word-dawg").ok()?;
        let punc = std::fs::read("/tmp/eng.lstm-punc-dawg").ok()?;
        let number = std::fs::read("/tmp/eng.lstm-number-dawg").ok()?;
        DictLite::from_components(&word, &punc, &number).ok()
    }

    fn load_real_charset() -> Option<UniCharSet> {
        UniCharSet::load_from_file(std::path::Path::new("/tmp/eng.lstm-unicharset")).ok()
    }

    #[test]
    fn default_dawgs_seeds_only_punc_when_available() {
        let Some(dict) = load_real_dawgs() else {
            eprintln!("skipping: /tmp/eng.lstm-*-dawg not present");
            return;
        };
        let seeds = dict.default_dawgs(false);
        // Punctuation subsumes word+number on eng.lstm (kDawgSuccessors[PUNC]
        // = [_, true, true, _], and the punc dawg has a pattern edge), so the
        // only seed is a pure punctuation position at the punc dawg's index.
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].dawg_index, -1);
        assert_eq!(seeds[0].dawg_ref, NO_EDGE);
        assert_eq!(seeds[0].punc_ref, NO_EDGE);
        assert!(!seeds[0].back_to_punc);
        // Punc dawg is loaded first (Dict::LoadLSTM order), so index 0.
        assert_eq!(seeds[0].punc_index, 0);
    }

    #[test]
    fn add_unique_dedups() {
        let mut v = Vec::new();
        let pos = DawgPosition::new(1, 5, -1, NO_EDGE, false);
        assert!(add_unique(&mut v, pos));
        assert!(!add_unique(&mut v, pos));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn char_for_dawg_maps_digits_on_number_dawg_only() {
        let Some(dict) = load_real_dawgs() else {
            eprintln!("skipping: /tmp/eng.lstm-*-dawg not present");
            return;
        };
        let Some(charset) = load_real_charset() else {
            eprintln!("skipping: /tmp/eng.lstm-unicharset not present");
            return;
        };
        let number_dawg = dict
            .dawgs
            .iter()
            .find(|d| d.dawg_type() == DawgType::Number)
            .expect("number dawg loaded");
        let word_dawg = dict
            .dawgs
            .iter()
            .find(|d| d.dawg_type() == DawgType::Word)
            .expect("word dawg loaded");
        // Find some digit id in the real charset to exercise the mapping.
        if let Some(digit_id) = (0..charset.size() as u32).find(|&id| charset.get_isdigit(id)) {
            assert_eq!(
                char_for_dawg(&charset, digit_id, number_dawg),
                K_PATTERN_UNICHAR_ID
            );
            // Word dawg never remaps, regardless of digit-ness.
            assert_eq!(char_for_dawg(&charset, digit_id, word_dawg), digit_id);
        }
    }

    #[test]
    fn def_letter_is_okay_walks_the_word_the() {
        let Some(dict) = load_real_dawgs() else {
            eprintln!("skipping: /tmp/eng.lstm-*-dawg not present");
            return;
        };
        let Some(charset) = load_real_charset() else {
            eprintln!("skipping: /tmp/eng.lstm-unicharset not present");
            return;
        };
        let mut active = dict.default_dawgs(false);
        let ids = [91_u32, 97, 92];
        let mut last_perm = PermuterType::NoPerm;
        let mut last_valid_end = false;
        for (i, &id) in ids.iter().enumerate() {
            let word_end = i + 1 == ids.len();
            let (updated, perm, valid_end) =
                dict.def_letter_is_okay(&active, &charset, id, word_end, PermuterType::NoPerm);
            assert!(
                !updated.is_empty(),
                "step {i} should keep at least one position"
            );
            active = updated;
            last_perm = perm;
            last_valid_end = valid_end;
        }
        assert_eq!(last_perm, PermuterType::SystemDawgPerm);
        assert!(last_valid_end);
    }
}
