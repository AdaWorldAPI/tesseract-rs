//! OPTIONAL dictionary correction over a recognized [`DocPage`] — Levenshtein
//! nearest-neighbour against a supplied lexicon.
//!
//! **Opt-in, and deliberately so.** Nothing in the recognition path calls
//! this; a caller reaches for it AFTER getting a `DocPage`, exactly like
//! [`crate::sentences`] and [`crate::reasoning`]. Same footing as the rest of
//! this crate's post-processing: consumer-side synthesis, NOT a Tesseract
//! transcode, so no parity claim applies or is made.
//!
//! # Why this is dangerous, and what bounds it
//!
//! Snapping OCR output to a dictionary can *invent* text that was never on
//! the page. Every guard below exists because of a specific, measured way
//! that goes wrong on this repo's own fixtures:
//!
//! 1. **Never touch a token containing a digit.** Measured on
//!    `corpus/lab/lab_table_ruled.pgm`: the degraded cells are values like
//!    `14.2 -> $142` and `0.9 -> O09`. A lab result or an invoice total has
//!    NO lexical answer — there is no dictionary entry that "should" be
//!    there, so nearest-neighbour would fabricate a plausible number. This is
//!    the single most important guard in the module and the reason a numeric
//!    cell is safer left visibly wrong than silently corrected.
//! 2. **Never "correct" a word the lexicon already knows.** A hit is a hit.
//! 3. **Length floor.** Edit distance carries almost no information on 1-3
//!    character tokens (`a`/`I`/`in`/`is` are all within one edit of each
//!    other), so short tokens are left alone.
//! 4. **Length-scaled distance budget.** A token 7 edits from everything is
//!    not a typo, it is garbage; snapping it produces confident nonsense.
//! 5. **Deterministic tie-break by corpus frequency**, so the same input
//!    always yields the same output — a corrector that varies run to run
//!    cannot be regression-tested.
//! 6. **Every change is REPORTED, never silent.** [`correct_page`] returns a
//!    [`Correction`] per edit carrying the original text. A caller that wants
//!    an audit trail, a confidence penalty, or a "show me what you changed"
//!    review gets it for free, and a caller that ignores the return value at
//!    least had to ignore it explicitly.
//!
//! # The lexicon is YOURS, not this module's
//!
//! [`Lexicon`] is built from any `(word, weight)` iterator. That is
//! deliberate: deepnsm's COCA vocabulary is **English**, and the words that
//! actually need rescuing on a German lab report (`Haemoglobin`, `Kreatinin`,
//! `Einheit`, `Referenzbereich`) are not in it. Correcting German text
//! against an English lexicon is worse than not correcting it — every
//! in-domain term becomes a "typo" one edit from some unrelated English word.
//! Supply the lexicon that matches the document, or do not run this pass.
//! [`Lexicon::from_deepnsm_vocab_dir`] is one convenience adapter, not the
//! contract.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use tesseract_ocr::structured::DocPage;

/// A word list to correct against, plus a weight per word used only to break
/// ties deterministically (corpus frequency is the natural choice; any
/// consistent ordering works).
///
/// Words are stored lowercase; matching is case-insensitive and the original
/// token's capitalisation is restored on output.
pub struct Lexicon {
    /// Lowercased word -> weight, for O(1) "does the lexicon know this".
    known: HashSet<String>,
    /// Candidates bucketed by character length, so a lookup only ever
    /// compares against words whose length is within the distance budget.
    /// Without this, every token would be diffed against the whole 18k+
    /// vocabulary.
    by_len: HashMap<usize, Vec<(String, u64)>>,
}

impl Lexicon {
    /// Build from any `(word, weight)` source. Empty and non-alphabetic
    /// entries are skipped — a lexicon containing digits would defeat guard 1
    /// by making numeric-looking tokens correctable.
    #[must_use]
    pub fn from_words<I: IntoIterator<Item = (String, u64)>>(words: I) -> Self {
        let mut known = HashSet::new();
        let mut by_len: HashMap<usize, Vec<(String, u64)>> = HashMap::new();
        for (w, weight) in words {
            let lw = w.to_lowercase();
            if lw.is_empty() || lw.chars().any(|c| c.is_numeric()) {
                continue;
            }
            let n = lw.chars().count();
            if known.insert(lw.clone()) {
                by_len.entry(n).or_default().push((lw, weight));
            }
        }
        // Sort each bucket by descending weight so the frequency tie-break is
        // just "first match wins at equal distance".
        for v in by_len.values_mut() {
            v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }
        Self { known, by_len }
    }

    /// Convenience adapter over deepnsm's COCA vocabulary, **including
    /// inflected forms**.
    ///
    /// **English.** See the module docs before using it on anything else.
    /// Weight is the corpus frequency, so ties resolve toward the commoner
    /// word.
    ///
    /// # Why this reads the CSV instead of only walking `Vocabulary`
    ///
    /// `Vocabulary::word(rank)` enumerates the **canonical lemma ranks only**
    /// (measured: 3558 distinct words). The inflected forms live in a private
    /// `forms` map reachable through `lookup_word` but not iterable, and there
    /// are **11,461** of them. Building the lexicon from ranks alone was a
    /// real, measured bug: `pictures` is a perfectly good word that is not a
    /// canonical rank, so guard 2 never fired and the corrector "fixed" it to
    /// the singular `picture`. **A lexicon missing inflections does not merely
    /// fail to correct — it actively corrupts correct text**, because every
    /// absent form looks like a typo one edit from its own lemma.
    ///
    /// So both files are read, per the format documented on
    /// `Vocabulary::load`: `word_rank_lookup.csv` (`rank,word,pos,freq`) and
    /// `word_forms.csv` (`lemRank,lemma,PoS,lemFreq,wordFreq,word`).
    /// `Vocabulary::load` still runs first, so a malformed directory fails
    /// here exactly as it would anywhere else in this crate.
    ///
    /// # Errors
    /// Propagates `Vocabulary::load`'s error string when the directory is
    /// missing or malformed, or an IO error string when a CSV cannot be read.
    pub fn from_deepnsm_vocab_dir(dir: &Path) -> Result<Self, String> {
        // Validate the directory the same way every other consumer does, so
        // this adapter cannot accept input the rest of the crate rejects.
        let vocab = deepnsm::vocabulary::Vocabulary::load(dir)?;
        let mut words: Vec<(String, u64)> = Vec::new();

        // Canonical ranks (also the source of truth for weights).
        for i in 0..vocab.len() {
            let Ok(rank) = u16::try_from(i) else { break };
            let w = vocab.word(rank);
            if !w.is_empty() {
                words.push((w.to_string(), vocab.freq(rank)));
            }
        }

        // Inflected forms: column 5 (0-based) is `word`, column 4 `wordFreq`.
        // A plain split is sufficient for these files (single-token words, no
        // quoting); a malformed row is skipped rather than failing the load,
        // since a partial lexicon is still safe — every guard still applies.
        let forms_path = dir.join("word_forms.csv");
        let text = std::fs::read_to_string(&forms_path)
            .map_err(|e| format!("read {}: {e}", forms_path.display()))?;
        for line in text.lines().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 6 {
                continue;
            }
            let w = cols[5].trim();
            if w.is_empty() {
                continue;
            }
            let freq = cols[4].trim().parse::<u64>().unwrap_or(0);
            words.push((w.to_string(), freq));
        }

        Ok(Self::from_words(words))
    }

    /// Number of distinct words.
    #[must_use]
    pub fn len(&self) -> usize {
        self.known.len()
    }

    /// Whether the lexicon is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.known.is_empty()
    }

    /// Case-insensitive membership.
    #[must_use]
    pub fn contains(&self, word: &str) -> bool {
        self.known.contains(&word.to_lowercase())
    }
}

/// How aggressive the correction is allowed to be.
///
/// The defaults are deliberately conservative: this pass should fix obvious
/// single-glyph misreads and decline everything else, because a wrong
/// correction is strictly worse than no correction (it is confident AND
/// wrong, and it destroys the evidence that anything was uncertain).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorrectionPolicy {
    /// Tokens with fewer alphabetic characters than this are never touched.
    /// At 1-3 characters nearly every word is within one edit of many others.
    pub min_len: usize,
    /// Distance budget for tokens shorter than [`Self::long_len`].
    pub max_distance_short: usize,
    /// Length at which the larger budget applies.
    pub long_len: usize,
    /// Distance budget for tokens at least [`Self::long_len`] long.
    pub max_distance_long: usize,
}

impl Default for CorrectionPolicy {
    fn default() -> Self {
        Self {
            min_len: 4,
            max_distance_short: 1,
            long_len: 8,
            // MEASURED default, not a guess. Sweeping this over the real 10k
            // COCA lexicon (`examples/correction_probe.rs`): a budget of 2
            // bought ZERO additional correct fixes over a budget of 1 (all six
            // genuine OCR repairs — beginnlng/thlnking/conversatlon/slster/
            // rabblt/wondet — are distance 1) while introducing ONE corruption,
            // the German `Referenz -> Refered`. Strictly worse on the evidence,
            // so the conservative value is the default and a caller who
            // measures 2-edit wins on their own corpus raises it deliberately.
            max_distance_long: 1,
        }
    }
}

impl CorrectionPolicy {
    /// The distance budget for a token of `len` characters.
    #[must_use]
    pub fn budget(&self, len: usize) -> usize {
        if len >= self.long_len {
            self.max_distance_long
        } else {
            self.max_distance_short
        }
    }
}

/// One applied edit, carrying the original so the change is auditable and
/// reversible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Correction {
    /// Index into [`DocPage::lines`].
    pub line: usize,
    /// Index into that line's `words`.
    pub word: usize,
    /// The recognized text BEFORE correction.
    pub from: String,
    /// The text written in its place.
    pub to: String,
    /// Levenshtein distance between the two alphabetic cores.
    pub distance: usize,
}

/// Levenshtein edit distance over CHARACTERS, with an early exit once the
/// whole row exceeds `budget`.
///
/// Characters, not bytes: the German words this is most useful for carry
/// multi-byte `ä ö ü ß`, and a byte-wise distance would count one substituted
/// umlaut as two edits and blow the budget on a correct answer.
fn levenshtein_within(a: &[char], b: &[char], budget: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > budget {
        return None;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        let mut row_min = cur[0];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            row_min = row_min.min(cur[j]);
        }
        // No later row can go below this row's minimum, so once the whole row
        // is over budget the answer is out of range.
        if row_min > budget {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[b.len()];
    (d <= budget).then_some(d)
}

/// Split a recognized token into (leading punctuation, alphabetic core,
/// trailing punctuation).
///
/// Real OCR words arrive as `Referenz,` or `(Kalium)`; correcting the whole
/// string including its punctuation would blow the distance budget on an
/// otherwise perfect match.
fn split_core(token: &str) -> (&str, &str, &str) {
    let s = token;
    let start = s
        .char_indices()
        .find(|(_, c)| c.is_alphabetic())
        .map_or(s.len(), |(i, _)| i);
    let end = s
        .char_indices()
        .rfind(|(_, c)| c.is_alphabetic())
        .map_or(start, |(i, c)| i + c.len_utf8());
    (&s[..start], &s[start..end], &s[end..])
}

/// Restore `model`'s capitalisation pattern onto `replacement`.
///
/// Only the two patterns that actually occur are handled — all-caps and
/// leading-capital — because anything more elaborate (camel case, small caps)
/// is not something a lexicon lookup should be reshaping.
fn match_case(model: &str, replacement: &str) -> String {
    let mut chars = model.chars();
    let first_upper = chars.next().is_some_and(char::is_uppercase);
    let rest_upper = model.chars().skip(1).any(char::is_alphabetic)
        && model
            .chars()
            .skip(1)
            .filter(|c| c.is_alphabetic())
            .all(char::is_uppercase);
    if first_upper && rest_upper {
        replacement.to_uppercase()
    } else if first_upper {
        let mut c = replacement.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => replacement.to_string(),
        }
    } else {
        replacement.to_string()
    }
}

/// The best in-budget replacement for one token, or `None` to leave it alone.
///
/// Returns `None` for every guard case in the module docs: digits present,
/// core too short, already known, or nothing inside the distance budget.
#[must_use]
pub fn suggest(token: &str, lex: &Lexicon, policy: &CorrectionPolicy) -> Option<(String, usize)> {
    // Guard 1: anything with a digit is data, not vocabulary.
    if token.chars().any(|c| c.is_numeric()) {
        return None;
    }
    let (pre, core, post) = split_core(token);
    let core_chars: Vec<char> = core.to_lowercase().chars().collect();
    // Guard 3: length floor.
    if core_chars.len() < policy.min_len {
        return None;
    }
    // Guard 2: a known word is never "corrected".
    if lex.known.contains(&core.to_lowercase()) {
        return None;
    }
    let budget = policy.budget(core_chars.len());

    // Only lengths within the budget can possibly be in range.
    let mut best: Option<(usize, &str)> = None;
    let lo = core_chars.len().saturating_sub(budget);
    for n in lo..=core_chars.len() + budget {
        let Some(bucket) = lex.by_len.get(&n) else {
            continue;
        };
        for (cand, _weight) in bucket {
            let cand_chars: Vec<char> = cand.chars().collect();
            let Some(d) = levenshtein_within(&core_chars, &cand_chars, budget) else {
                continue;
            };
            // Buckets are pre-sorted by descending weight, so a STRICT
            // improvement is required to displace an earlier (commoner)
            // candidate — that is the deterministic frequency tie-break.
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, cand));
                if d == 0 {
                    break;
                }
            }
        }
    }
    let (d, cand) = best?;
    Some((format!("{pre}{}{post}", match_case(core, cand)), d))
}

/// Apply [`suggest`] to every word of `page`, in place, returning one
/// [`Correction`] per change.
///
/// The page is left untouched wherever no correction was warranted, so an
/// empty return value means "nothing was changed" and not "nothing was
/// examined".
pub fn correct_page(
    page: &mut DocPage,
    lex: &Lexicon,
    policy: &CorrectionPolicy,
) -> Vec<Correction> {
    let mut out = Vec::new();
    for (li, line) in page.lines.iter_mut().enumerate() {
        for (wi, word) in line.words.iter_mut().enumerate() {
            if let Some((fixed, d)) = suggest(&word.text, lex, policy) {
                if fixed != word.text {
                    out.push(Correction {
                        line: li,
                        word: wi,
                        from: word.text.clone(),
                        to: fixed.clone(),
                        distance: d,
                    });
                    word.text = fixed;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::from_words(
            [
                ("Haemoglobin", 100),
                ("Kreatinin", 90),
                ("Einheit", 80),
                ("Ergebnis", 70),
                ("Referenz", 60),
                ("Glukose", 50),
                ("beginning", 40),
                ("cat", 30),
                ("car", 31),
            ]
            .into_iter()
            .map(|(w, f)| (w.to_string(), f)),
        )
    }

    #[test]
    fn corrects_a_single_glyph_misread() {
        // One substitution inside a long word: exactly what this exists for.
        let (fixed, d) = suggest("Haemoglobln", &lex(), &CorrectionPolicy::default())
            .expect("a 1-edit neighbour must be found");
        assert_eq!(fixed, "Haemoglobin");
        assert_eq!(d, 1);
    }

    /// GUARD 1, the load-bearing one. A lab value or invoice total has no
    /// lexical answer; correcting it would FABRICATE data. Measured motivation:
    /// `14.2 -> $142` and `0.9 -> O09` on the ruled lab fixture.
    ///
    /// **The first version of this test was VACUOUS and shipped that way.** It
    /// used group A alone — and every one of those tokens has an alphabetic
    /// core BELOW `min_len`, so guard 3 (the length floor) declined them and
    /// guard 1 was never reached. Deleting the digit guard left the test
    /// passing; verified by running exactly that. Group B exists so the guard
    /// has something only *it* can decline.
    #[test]
    fn never_touches_a_token_containing_a_digit() {
        let l = lex();
        let p = CorrectionPolicy::default();

        // GROUP A — the real measured tokens. They ARE declined, but say
        // plainly why: their alphabetic cores sit below the length floor. This
        // group cannot falsify guard 1, and the assertion below pins that
        // honestly rather than letting it read as evidence.
        for t in ["$142", "O09", "13.5", "4mm", "2.4", "136-145"] {
            let (_, core, _) = split_core(t);
            assert!(
                core.chars().count() < p.min_len,
                "group A is the length-floor group by construction, but {t:?} \
                 has core {core:?} at or above the floor — it belongs in group B"
            );
            assert!(
                suggest(t, &l, &p).is_none(),
                "numeric token {t:?} must never be corrected — a fabricated \
                 number is worse than a visibly wrong one"
            );
        }

        // GROUP B — a long, unknown, IN-BUDGET alphabetic core plus a digit.
        // Guards 2, 3 and 4 all pass these through, so guard 1 is the only
        // thing that can decline them. Delete guard 1 and this loop fails.
        for (t, would_become) in [
            ("Haemoglobln2", "Haemoglobin"),
            ("Glukos3", "Glukose"),
            ("2Kreatlnin", "Kreatinin"),
        ] {
            // Prove the premise instead of asserting it: the SAME core with the
            // digit removed IS corrected, so the digit is the only difference.
            let (_, core, _) = split_core(t);
            assert_eq!(
                suggest(core, &l, &p).map(|(s, _)| s).as_deref(),
                Some(would_become),
                "the digit-free core {core:?} must be correctable, or {t:?} \
                 cannot falsify guard 1"
            );
            assert!(
                suggest(t, &l, &p).is_none(),
                "{t:?} carries a digit and must be left alone — a fabricated \
                 number is worse than a visibly wrong one"
            );
        }
    }

    /// GUARD 2: a word the lexicon knows is returned untouched, even though a
    /// distance-1 neighbour exists in the lexicon ("car"). Without this, every
    /// correct word would be at risk of being "corrected" to a commoner one.
    #[test]
    fn never_corrects_a_word_the_lexicon_already_knows() {
        let l = lex();
        assert!(l.contains("cat"));
        assert!(l.contains("car"));
        assert!(
            suggest("cat", &l, &CorrectionPolicy::default()).is_none(),
            "a known word must be left alone even with a 1-edit neighbour present"
        );
    }

    /// GUARD 3: below the length floor nothing is touched. Two-letter tokens
    /// sit within one edit of most of the alphabet, so a correction there is
    /// a coin flip wearing a confident face.
    #[test]
    fn leaves_short_tokens_alone() {
        let l = lex();
        let p = CorrectionPolicy::default();
        assert!(suggest("ca", &l, &p).is_none());
        assert!(suggest("qx", &l, &p).is_none());
    }

    /// GUARD 4, two-sided on the budget itself. `Ergebnis` (8 chars) takes the
    /// LONG budget, so a misread within it is corrected; a token that is far
    /// from everything is left as-is rather than snapped to nonsense. The long
    /// budget's default VALUE is 1, not 2 — measured, 2 bought no extra correct
    /// fixes and cost one cross-language corruption — and the assertions below
    /// pin both that value and the fact that raising it still works.
    #[test]
    fn respects_the_distance_budget_in_both_directions() {
        let l = lex();
        let p = CorrectionPolicy::default();
        assert_eq!(
            p.budget(8),
            1,
            "the DEFAULT long budget is 1 — measured, 2 bought no extra correct \
             fixes and cost one cross-language corruption"
        );
        assert_eq!(p.budget(5), 1, "5 chars gets the short budget");
        // The knob still WORKS when a caller raises it deliberately, or this
        // would be a dead field rather than a policy.
        let wide = CorrectionPolicy {
            max_distance_long: 2,
            ..CorrectionPolicy::default()
        };
        assert_eq!(wide.budget(8), 2, "a raised long budget must take effect");
        assert_eq!(
            wide.budget(5),
            1,
            "raising the long budget must not move the short one"
        );

        let (fixed, d) = suggest("Ergebnls", &l, &p).expect("1 edit, in budget");
        assert_eq!(fixed, "Ergebnis");
        assert_eq!(d, 1);

        assert!(
            suggest("zzzzzzzzz", &l, &p).is_none(),
            "garbage far from every entry must be LEFT ALONE, not snapped"
        );
    }

    /// Punctuation is preserved around the corrected core — otherwise a
    /// trailing comma would consume the whole distance budget and a perfectly
    /// correctable word would be declined.
    #[test]
    fn preserves_surrounding_punctuation() {
        let (fixed, _) = suggest("(Kreatlnin),", &lex(), &CorrectionPolicy::default())
            .expect("core is 1 edit from Kreatinin");
        assert_eq!(fixed, "(Kreatinin),");
    }

    /// Capitalisation of the original token is restored, so correcting does
    /// not silently case-fold a proper noun or a heading.
    #[test]
    fn restores_the_original_capitalisation() {
        let l = lex();
        let p = CorrectionPolicy::default();
        assert_eq!(suggest("EINHEIU", &l, &p).unwrap().0, "EINHEIT");
        assert_eq!(suggest("Einhelt", &l, &p).unwrap().0, "Einheit");
    }

    /// The tie-break is deterministic and frequency-ordered: with two lexicon
    /// entries at the SAME distance, the commoner one wins, every run.
    #[test]
    fn ties_break_toward_the_commoner_word_deterministically() {
        // "caz" is 1 edit from both "cat" (30) and "car" (31).
        let l = Lexicon::from_words(
            [("cat", 30u64), ("car", 31u64)]
                .into_iter()
                .map(|(w, f)| (w.to_string(), f)),
        );
        let p = CorrectionPolicy {
            min_len: 3,
            ..CorrectionPolicy::default()
        };
        let first = suggest("caz", &l, &p).expect("in budget");
        assert_eq!(first.0, "car", "the commoner entry wins the tie");
        for _ in 0..8 {
            assert_eq!(
                suggest("caz", &l, &p).expect("in budget").0,
                first.0,
                "the tie-break must be stable across runs"
            );
        }
    }

    /// A lexicon must never admit digit-bearing entries, or guard 1 could be
    /// defeated from the data side rather than the token side.
    #[test]
    fn lexicon_rejects_entries_containing_digits() {
        let l = Lexicon::from_words(
            [("abc123", 1u64), ("plain", 1)]
                .into_iter()
                .map(|(w, f)| (w.to_string(), f)),
        );
        assert!(!l.contains("abc123"));
        assert!(l.contains("plain"));
        assert_eq!(l.len(), 1);
    }

    /// The REAL vocabulary must contain INFLECTED FORMS, not just canonical
    /// lemma ranks — the falsifier for the `word_forms.csv` half of
    /// [`Lexicon::from_deepnsm_vocab_dir`].
    ///
    /// Measured bug this pins: building the lexicon from `Vocabulary::word(rank)`
    /// alone yields 3558 words and omits `pictures`, so guard 2 never fired and
    /// the corrector rewrote a correct plural to its singular. Reading the forms
    /// file as well takes it past 11k and makes `pictures` known.
    ///
    /// Skips gracefully without the sibling checkout, matching this crate's
    /// established pattern for tests that need real data.
    #[test]
    fn real_vocabulary_knows_inflected_forms_not_only_lemma_ranks() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../lance-graph/crates/deepnsm/word_frequency");
        if !dir.join("word_forms.csv").exists() {
            eprintln!("skipping: deepnsm word_frequency not present");
            return;
        }
        let l = Lexicon::from_deepnsm_vocab_dir(&dir).expect("load the real vocabulary");
        assert!(
            l.len() > 8000,
            "a forms-inclusive lexicon must be far larger than the 3558 canonical \
             ranks; got {}",
            l.len()
        );
        // The exact word the rank-only lexicon corrupted.
        assert!(
            l.contains("pictures"),
            "an inflected form present in word_forms.csv must be KNOWN, or the \
             corrector will rewrite it to its lemma"
        );
        assert!(
            suggest("pictures", &l, &CorrectionPolicy::default()).is_none(),
            "a known inflected form must be left alone (guard 2)"
        );
    }

    /// `correct_page` reports every change and leaves everything else byte
    /// identical — the audit-trail contract.
    #[test]
    fn correct_page_reports_changes_and_touches_nothing_else() {
        use tesseract_ocr::structured::{DocLine, DocWord};
        let mk = |t: &str| DocWord {
            text: t.to_string(),
            bbox: (0, 0, 10, 10),
            conf: 90.0,
            leading_space: false,
            numeric_norm: None,
        };
        let mut page = DocPage {
            width: 100,
            height: 100,
            lines: vec![DocLine {
                bbox: (0, 0, 100, 10),
                words: vec![mk("Haemoglobln"), mk("14.2"), mk("Glukose")],
                metrics: None,
            }],
        };
        let changes = correct_page(&mut page, &lex(), &CorrectionPolicy::default());
        assert_eq!(
            changes.len(),
            1,
            "exactly one word was correctable: {changes:?}"
        );
        assert_eq!(changes[0].from, "Haemoglobln");
        assert_eq!(changes[0].to, "Haemoglobin");
        assert_eq!(page.lines[0].words[0].text, "Haemoglobin");
        assert_eq!(page.lines[0].words[1].text, "14.2", "numeric untouched");
        assert_eq!(
            page.lines[0].words[2].text, "Glukose",
            "known word untouched"
        );
    }
}
