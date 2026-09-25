//! S-8: the rule attaches to the thing being recognised.
//!
//! ```text
//!   USERS NEVER MEET A RULE ENGINE.  THEY MEET "THIS SUPPLIER, MATCHED THIS WAY."
//! ```
//!
//! paperless-ngx gives every correspondent, tag and document type its own
//! `matching_algorithm` + `match` string (`documents/models.py`,
//! `MatchingModel`), evaluated against a document's text by
//! `documents/matching.py::matches`. This module is that evaluator, transcribed
//! rule for rule: a [`MatchRule`] is what a tag or correspondent carries, and
//! [`CompiledRule::matches`] answers the same question `matches()` does.
//! `OGAR-DOC-INGESTION-SPINE` S-8 names the vocabulary; this is it, with the
//! rule living on the object rather than in a separate engine.
//!
//! | algorithm | paperless-ngx | here |
//! |---|---|---|
//! | [`MatchAlgorithm::None`] | never matches | same |
//! | [`MatchAlgorithm::Any`] | any whole word/phrase of `match` | same |
//! | [`MatchAlgorithm::All`] | every whole word/phrase of `match` | same |
//! | [`MatchAlgorithm::Literal`] | `match` as one whole-word string | same |
//! | [`MatchAlgorithm::Regex`] | `match` as a regular expression | same, see below |
//! | [`MatchAlgorithm::Fuzzy`] | `rapidfuzz.partial_ratio >= 90` | [`partial_ratio`], same cutoff |
//! | [`MatchAlgorithm::Auto`] | the learned classifier, "done elsewhere" | never matches here |
//!
//! Words and phrases split exactly as `_split_match` does: a double-quoted
//! run is one phrase whose inner whitespace matches any whitespace, anything
//! else splits on whitespace. Word boundaries are Unicode `\b`, as in Python.
//!
//! # Where this differs from paperless-ngx, and why
//!
//! * **Regex engine.** paperless-ngx runs the Python `regex` module under a
//!   0.1 s timeout, because a backtracking engine can take exponential time on
//!   a pattern like `(a+)+$`. The `regex` crate guarantees linear time, so
//!   there is no timeout to set. The cost of that guarantee: look-around and
//!   back-references are not supported. A pattern using them fails to compile
//!   and never matches — the same outcome paperless-ngx gives any pattern that
//!   fails its own validation.
//! * **`Auto`** is the classifier tier (S-10). It never matches here; a
//!   learned rule is meant to materialise into one of the explicit algorithms
//!   above, not to run as a separate mechanism.

use regex::{Regex, RegexBuilder};

/// How a rule's `pattern` is applied. Discriminants are paperless-ngx's own
/// `MatchingModel.MATCH_*` values, so a stored rule round-trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MatchAlgorithm {
    /// Never matches.
    None = 0,
    /// Any one word or quoted phrase of the pattern appears as a whole word.
    Any = 1,
    /// Every word and quoted phrase of the pattern appears as a whole word.
    All = 2,
    /// The whole pattern appears as one whole-word string.
    Literal = 3,
    /// The pattern is a regular expression.
    Regex = 4,
    /// The pattern approximately appears (`partial_ratio >= 90`).
    Fuzzy = 5,
    /// The learned classifier. Never matches here.
    Auto = 6,
}

impl MatchAlgorithm {
    /// The algorithm with paperless-ngx's number `v`, or `None` for a value
    /// it does not define.
    #[must_use]
    pub fn from_paperless(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::None,
            1 => Self::Any,
            2 => Self::All,
            3 => Self::Literal,
            4 => Self::Regex,
            5 => Self::Fuzzy,
            6 => Self::Auto,
            _ => return None,
        })
    }
}

/// The rule a tag, correspondent or document type carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchRule {
    /// How [`Self::pattern`] is applied.
    pub algorithm: MatchAlgorithm,
    /// paperless-ngx's `match` string.
    pub pattern: String,
    /// paperless-ngx's `is_insensitive` (its default is `true`).
    pub case_insensitive: bool,
}

/// Score a fuzzy rule must reach, as in paperless-ngx (`score_cutoff=90`).
pub const FUZZY_CUTOFF: f64 = 90.0;

impl MatchRule {
    /// Compile once, match many documents.
    #[must_use]
    pub fn compile(&self) -> CompiledRule {
        // paperless-ngx: `if not matching_model.match.strip(): return False`.
        if self.pattern.trim().is_empty() {
            return CompiledRule(Inner::Never);
        }
        let ci = self.case_insensitive;
        let inner = match self.algorithm {
            MatchAlgorithm::None | MatchAlgorithm::Auto => Inner::Never,
            MatchAlgorithm::Any => words(&self.pattern, ci).map_or(Inner::Never, Inner::Any),
            MatchAlgorithm::All => words(&self.pattern, ci).map_or(Inner::Never, Inner::All),
            MatchAlgorithm::Literal => build(&format!(r"\b{}\b", regex::escape(&self.pattern)), ci)
                .map_or(Inner::Never, Inner::One),
            MatchAlgorithm::Regex => build(&self.pattern, ci).map_or(Inner::Never, Inner::One),
            MatchAlgorithm::Fuzzy => {
                let needle = fuzzy_clean(&self.pattern, ci);
                Inner::Fuzzy { needle, ci }
            }
        };
        CompiledRule(inner)
    }

    /// Compile and match in one call. Prefer [`Self::compile`] when the same
    /// rule is tried against many documents.
    #[must_use]
    pub fn matches(&self, content: &str) -> bool {
        self.compile().matches(content)
    }
}

/// A [`MatchRule`] ready to run.
#[derive(Debug, Clone)]
pub struct CompiledRule(Inner);

#[derive(Debug, Clone)]
enum Inner {
    Never,
    One(Regex),
    Any(Vec<Regex>),
    All(Vec<Regex>),
    Fuzzy { needle: Vec<char>, ci: bool },
}

impl CompiledRule {
    /// Whether `content` — a document's text — satisfies the rule.
    #[must_use]
    pub fn matches(&self, content: &str) -> bool {
        match &self.0 {
            Inner::Never => false,
            Inner::One(re) => re.is_match(content),
            Inner::Any(res) => res.iter().any(|r| r.is_match(content)),
            Inner::All(res) => res.iter().all(|r| r.is_match(content)),
            Inner::Fuzzy { needle, ci } => {
                partial_ratio(needle, &fuzzy_clean(content, *ci)) >= FUZZY_CUTOFF
            }
        }
    }
}

/// The subset of `rules` whose rule matches `content`, in input order — what
/// `match_tags` / `match_correspondents` return, minus the classifier tier.
pub fn matching<'a, T>(
    rules: impl IntoIterator<Item = (T, &'a CompiledRule)>,
    content: &str,
) -> Vec<T> {
    rules
        .into_iter()
        .filter(|(_, r)| r.matches(content))
        .map(|(t, _)| t)
        .collect()
}

fn build(pattern: &str, case_insensitive: bool) -> Option<Regex> {
    RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .ok()
}

/// `_split_match`: a `"quoted phrase"` is one term, anything else splits on
/// whitespace; a term's inner whitespace is normalised and then matches any
/// whitespace run. Each term becomes `\bterm\b`.
fn words(pattern: &str, ci: bool) -> Option<Vec<Regex>> {
    let mut terms = Vec::new();
    let mut rest = pattern;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        // `"([^"]+)"` first: a quote followed by at least one non-quote and a
        // closing quote. Otherwise `(\S+)`.
        let quoted = rest.strip_prefix('"').and_then(|after| {
            let end = after.find('"')?;
            (end > 0).then(|| (&after[..end], &after[end + 1..]))
        });
        let (term, next) = quoted.unwrap_or_else(|| {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            (&rest[..end], &rest[end..])
        });
        let normalised = term.split_whitespace().collect::<Vec<_>>().join(" ");
        let escaped = regex::escape(&normalised).replace(' ', r"\s+");
        terms.push(build(&format!(r"\b{escaped}\b"), ci)?);
        rest = next;
    }
    Some(terms)
}

/// `re.sub(r"[^\w\s]", "", s)`, lower-cased when insensitive, as chars.
fn fuzzy_clean(s: &str, case_insensitive: bool) -> Vec<char> {
    let kept = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || c.is_whitespace());
    if case_insensitive {
        kept.flat_map(char::to_lowercase).collect()
    } else {
        kept.collect()
    }
}

/// rapidfuzz's `fuzz.partial_ratio`, 0..=100: the best normalised Indel
/// similarity between the shorter string and any window of the longer one of
/// the same length, including the shorter windows that overhang either end.
///
/// Indel similarity is `2·LCS / (|a| + |b|)`. Cost is one LCS per window —
/// `O(|long| · |short|²)` — which is fine for rule strings of a few dozen
/// characters against a page of text.
#[must_use]
pub fn partial_ratio(a: &[char], b: &[char]) -> f64 {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.is_empty() {
        return 0.0;
    }
    let m = short.len();
    // Lengths are counts of chars in one document; u32 is ample and converts
    // to f64 exactly.
    let count = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let score = |w: &[char]| 200.0 * count(lcs(short, w)) / count(m + w.len());
    if long.len() <= m {
        return score(long);
    }
    let mut best = 0.0f64;
    for k in 1..m {
        best = best
            .max(score(&long[..k]))
            .max(score(&long[long.len() - k..]));
    }
    for start in 0..=long.len() - m {
        best = best.max(score(&long[start..start + m]));
        if best >= 100.0 {
            break;
        }
    }
    best
}

fn lcs(a: &[char], b: &[char]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for &x in a {
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                cur[j].max(prev[j + 1])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    //! The cases are paperless-ngx's own, from
    //! `src/documents/tests/test_matchables.py`, so a pass here means the
    //! same verdicts as upstream on upstream's fixtures.
    use super::*;

    fn rule(algorithm: MatchAlgorithm, pattern: &str, case_sensitive: bool) -> CompiledRule {
        MatchRule {
            algorithm,
            pattern: pattern.to_string(),
            case_insensitive: !case_sensitive,
        }
        .compile()
    }

    fn check(algorithm: MatchAlgorithm, pattern: &str, yes: &[&str], no: &[&str], cs: bool) {
        let r = rule(algorithm, pattern, cs);
        for c in yes {
            assert!(r.matches(c), "{algorithm:?} {pattern:?} should match {c:?}");
        }
        for c in no {
            assert!(
                !r.matches(c),
                "{algorithm:?} {pattern:?} should not match {c:?}"
            );
        }
    }

    #[test]
    fn none_never_matches() {
        check(MatchAlgorithm::None, "", &[], &["no", "match"], false);
        check(MatchAlgorithm::None, "no", &[], &["no", "match"], false);
    }

    #[test]
    fn an_empty_or_blank_pattern_never_matches() {
        for alg in [
            MatchAlgorithm::Any,
            MatchAlgorithm::All,
            MatchAlgorithm::Literal,
            MatchAlgorithm::Regex,
            MatchAlgorithm::Fuzzy,
        ] {
            check(alg, "   ", &[], &["anything at all", "   "], false);
        }
    }

    #[test]
    fn all_words() {
        check(
            MatchAlgorithm::All,
            "alpha charlie gamma",
            &["I have alpha, charlie, and gamma in me"],
            &[
                "I have alpha in me",
                "I have charlie in me",
                "I have gamma in me",
                "I have alpha and charlie in me",
                "I have alphas, charlie, and gamma in me",
                "I have alphas in me",
                "I have bravo in me",
            ],
            false,
        );
        check(
            MatchAlgorithm::All,
            "12 34 56",
            &["I have 12 34, and 56 in me"],
            &[
                "I have 12 in me",
                "I have 34 in me",
                "I have 56 in me",
                "I have 12 and 34 in me",
                "I have 120, 34, and 56 in me",
                "I have 123456 in me",
                "I have 01234567 in me",
            ],
            false,
        );
        check(
            MatchAlgorithm::All,
            r#"brown fox "lazy dogs""#,
            &[
                "the quick brown fox jumped over the lazy dogs",
                "the quick brown fox jumped over the lazy  dogs",
            ],
            &[
                "the quick fox jumped over the lazy dogs",
                "the quick brown wolf jumped over the lazy dogs",
                "the quick brown fox jumped over the fat dogs",
                "the quick brown fox jumped over the lazy... dogs",
            ],
            false,
        );
    }

    #[test]
    fn any_word() {
        check(
            MatchAlgorithm::Any,
            "alpha charlie gamma",
            &[
                "I have alpha in me",
                "I have charlie in me",
                "I have gamma in me",
                "I have alpha, charlie, and gamma in me",
                "I have alpha and charlie in me",
            ],
            &["I have alphas in me", "I have bravo in me"],
            false,
        );
        check(
            MatchAlgorithm::Any,
            "12 34 56",
            &[
                "I have 12 in me",
                "I have 34 in me",
                "I have 56 in me",
                "I have 12 and 34 in me",
                "I have 12, 34, and 56 in me",
                "I have 120, 34, and 56 in me",
            ],
            &["I have 123456 in me", "I have 01234567 in me"],
            false,
        );
        check(
            MatchAlgorithm::Any,
            r#""brown fox" " lazy  dogs ""#,
            &["the quick brown fox", "jumped over the lazy  dogs."],
            &["the lazy fox jumped over the brown dogs"],
            false,
        );
    }

    #[test]
    fn literal() {
        check(
            MatchAlgorithm::Literal,
            "alpha charlie gamma",
            &["I have 'alpha charlie gamma' in me"],
            &[
                "I have alpha in me",
                "I have charlie in me",
                "I have gamma in me",
                "I have alpha and charlie in me",
                "I have alpha, charlie, and gamma in me",
                "I have alphas, charlie, and gamma in me",
                "I have alphas in me",
                "I have bravo in me",
            ],
            false,
        );
        check(
            MatchAlgorithm::Literal,
            "12 34 56",
            &["I have 12 34 56 in me"],
            &[
                "I have 12 in me",
                "I have 34 in me",
                "I have 56 in me",
                "I have 12 and 34 in me",
                "I have 12 34, and 56 in me",
                "I have 120, 34, and 560 in me",
                "I have 120, 340, and 560 in me",
                "I have 123456 in me",
                "I have 01234567 in me",
            ],
            false,
        );
    }

    #[test]
    fn regex() {
        check(
            MatchAlgorithm::Regex,
            r"alpha\w+gamma",
            &[
                "I have alpha_and_gamma in me",
                "I have alphas_and_gamma in me",
            ],
            &[
                "I have alpha in me",
                "I have gamma in me",
                "I have alpha and charlie in me",
                "I have alpha,and,gamma in me",
                "I have alpha and gamma in me",
                "I have alpha, charlie, and gamma in me",
                "I have alphas, charlie, and gamma in me",
                "I have alphas in me",
            ],
            false,
        );
    }

    #[test]
    fn an_invalid_regex_never_matches() {
        check(
            MatchAlgorithm::Regex,
            "[",
            &[],
            &["Don't match this"],
            false,
        );
    }

    /// paperless-ngx needs a timeout for this pattern; a linear-time engine
    /// answers it directly, and still answers `false`.
    #[test]
    fn a_catastrophic_backtracking_pattern_is_answered_not_timed_out() {
        let content = format!("{}X", "a".repeat(5000));
        check(MatchAlgorithm::Regex, r"(a+)+$", &[], &[&content], false);
    }

    #[test]
    fn fuzzy() {
        check(
            MatchAlgorithm::Fuzzy,
            "Springfield, Miss.",
            &[
                "1220 Main Street, Springf eld, Miss.",
                "1220 Main Street, Spring field, Miss.",
                "1220 Main Street, Springfeld, Miss.",
                "1220 Main Street Springfield Miss",
            ],
            &["1220 Main Street, Springfield, Mich."],
            false,
        );
    }

    #[test]
    fn case_sensitive_all() {
        check(
            MatchAlgorithm::All,
            "alpha charlie gamma",
            &[
                "I have alpha, charlie, and gamma in me",
                "I have gamma, charlie, and alpha in me",
            ],
            &[
                "I have Alpha, charlie, and gamma in me",
                "I have gamma, Charlie, and alpha in me",
                "I have alpha, charlie, and Gamma in me",
                "I have gamma, charlie, and ALPHA in me",
            ],
            true,
        );
        check(
            MatchAlgorithm::All,
            r#"brown fox "lazy dogs""#,
            &[
                "the quick brown fox jumped over the lazy dogs",
                "the quick brown fox jumped over the lazy  dogs",
            ],
            &[
                "the quick Brown fox jumped over the lazy dogs",
                "the quick brown Fox jumped over the lazy  dogs",
                "the quick brown fox jumped over the Lazy dogs",
                "the quick brown fox jumped over the lazy  Dogs",
            ],
            true,
        );
    }

    /// The default is insensitive; the same rule with case sensitivity on
    /// must reject what the insensitive one accepts, so the flag is live.
    #[test]
    fn the_case_flag_changes_the_verdict() {
        let text = "I have ALPHA in me";
        assert!(rule(MatchAlgorithm::Any, "alpha", false).matches(text));
        assert!(!rule(MatchAlgorithm::Any, "alpha", true).matches(text));
        assert!(rule(MatchAlgorithm::Fuzzy, "alpha", false).matches(text));
        assert!(!rule(MatchAlgorithm::Fuzzy, "alpha", true).matches(text));
    }

    /// Punctuation is stripped from both sides before scoring, as
    /// paperless-ngx does with `re.sub(r"[^\w\s]", "", ...)`: a word spelled
    /// out with dots still matches its rule.
    #[test]
    fn fuzzy_ignores_punctuation_between_letters() {
        let rule = MatchRule {
            algorithm: MatchAlgorithm::Fuzzy,
            pattern: "Springfield".into(),
            case_insensitive: true,
        };
        assert!(rule.matches("posted from S.p.r.i.n.g.f.i.e.l.d yesterday"));
        assert!(!rule.matches("posted from S.h.e.l.b.y.v.i.l.l.e yesterday"));
    }

    #[test]
    fn partial_ratio_reference_values() {
        let c = |s: &str| s.chars().collect::<Vec<_>>();
        assert!((partial_ratio(&c("abc"), &c("xxabcxx")) - 100.0).abs() < 1e-9);
        assert!((partial_ratio(&c("abcd"), &c("abcd")) - 100.0).abs() < 1e-9);
        assert!(partial_ratio(&c(""), &c("abc")).abs() < 1e-9);
        // One substitution in four: best window LCS 3 of 4 -> 2*3/8 = 75.
        assert!((partial_ratio(&c("abcd"), &c("zzabxdzz")) - 75.0).abs() < 1e-9);
        // A match overhanging the start of the text scores through the short
        // prefix window "cd" (2*2/(4+2)), not the best full-length window
        // "cdxx" (2*2/(4+4) = 50).
        assert!((partial_ratio(&c("abcd"), &c("cdxxxxxx")) - 200.0 / 3.0).abs() < 1e-9);
        assert!((partial_ratio(&c("abcd"), &c("xxxxxxab")) - 200.0 / 3.0).abs() < 1e-9);
        // Symmetric in its arguments.
        assert!(
            (partial_ratio(&c("hello"), &c("say hello there"))
                - partial_ratio(&c("say hello there"), &c("hello")))
            .abs()
                < 1e-9
        );
    }

    #[test]
    fn matching_returns_the_rules_that_fire_in_order() {
        let rules = [
            (
                "invoice",
                rule(MatchAlgorithm::Any, "Rechnung invoice", false),
            ),
            ("bank", rule(MatchAlgorithm::Literal, "Sparkasse", false)),
            ("tax", rule(MatchAlgorithm::All, "Steuer Bescheid", false)),
        ];
        let got = matching(
            rules.iter().map(|(t, r)| (*t, r)),
            "Rechnung Nr. 1042 von der Sparkasse",
        );
        assert_eq!(got, vec!["invoice", "bank"]);
    }

    #[test]
    fn algorithm_numbers_are_paperless_ngx_numbers() {
        for v in 0..=6u8 {
            let a = MatchAlgorithm::from_paperless(v).expect("defined");
            assert_eq!(a as u8, v);
        }
        assert_eq!(MatchAlgorithm::from_paperless(7), None);
    }
}
