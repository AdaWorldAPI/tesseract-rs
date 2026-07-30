//! Sentence assembly over a recognized [`DocPage`] — turns per-line OCR
//! output into sentence-shaped text units, the input unit
//! [`deepnsm::parser::parse`] actually wants (it operates per-sentence, not
//! per-line: `tesseract-rs/CLAUDE.md`'s "AS-IS BOUNDARY" section named this
//! exact gap — lines are a typographic artifact, a sentence spans several of
//! them, hyphenated at the wraps).
//!
//! **Consumer-side synthesis, NOT a Tesseract transcode** — same footing as
//! `tesseract_ocr::structured`'s `doc.v1` or `rectify.rs`'s `auto_rectify`:
//! built on top of the proven recognizer output, no parity claim applies.
//! The heuristics below are this module's own, documented as such.
//!
//! ## Known limitations (by design, for a first cut)
//!
//! - **Dehyphenation has no lookahead.** A line ending in `-` after a letter
//!   is treated as a wrap-hyphen and spliced to the next line with no space
//!   — this is right for "compli-" + "cated" but wrong for a genuine
//!   compound word that happens to wrap right after its own hyphen (e.g.
//!   "well-" + "known" loses the hyphen it should have kept). A dictionary
//!   check against both the split and joined forms would fix this; that is
//!   future work, not built here.
//! - **Sentence-boundary attribution is line-granular, not word-granular.**
//!   When one physical line contains the tail of one sentence and the head
//!   of the next, BOTH sentences get that line's full bbox/confidence
//!   attributed to them (no text is lost — the split point in the text
//!   itself is exact — only the bbox/conf bookkeeping over-attributes on
//!   that line).
//! - **Abbreviation/decimal guarding is narrow.** Only a `.` flanked by
//!   digits on both sides is protected (e.g. `13.5`) — "Dr." / "z.B." style
//!   abbreviations are NOT guarded and will incorrectly end a sentence.

use tesseract_ocr::{DocLine, DocPage};

/// One assembled sentence: its joined text, the union bbox of every
/// contributing line, which line indices (into the source [`DocPage`])
/// contributed, and the mean OCR word confidence over those lines.
#[derive(Clone, Debug, PartialEq)]
pub struct AssembledSentence {
    /// The sentence's joined, dehyphenated text.
    pub text: String,
    /// Top-down image bbox — the union of every contributing line's bbox.
    pub bbox: (i32, i32, i32, i32),
    /// Indices into the source `DocPage::lines` that contributed text to
    /// this sentence (may overlap with the next sentence's list — see the
    /// module docs' line-granular-attribution limitation).
    pub line_indices: Vec<usize>,
    /// Mean OCR word confidence (0-100) over the contributing lines' words.
    pub mean_conf: f32,
}

/// Join one line's words into text, respecting [`DocWord::leading_space`]
/// (`crate::tesseract_ocr::DocWord`) — the same per-word space rule
/// `tesseract_ocr::render_text` uses (`renderer.rs`), just applied to the
/// already-resolved `DocWord::text` instead of decoding unichar ids.
fn line_text(line: &DocLine) -> String {
    let mut s = String::new();
    for w in &line.words {
        if w.leading_space {
            s.push(' ');
        }
        s.push_str(&w.text);
    }
    s
}

/// Does `s` end in a line-wrap hyphen — an ASCII `-` immediately preceded by
/// a letter? A standalone `-` (preceded by whitespace, e.g. a bullet or a
/// spaced em-dash-as-punctuation) returns `false`.
fn ends_with_wrap_hyphen(s: &str) -> bool {
    let mut chars = s.chars().rev();
    match chars.next() {
        Some('-') => matches!(chars.next(), Some(c) if c.is_alphabetic()),
        _ => false,
    }
}

/// Find the byte offset immediately after the first sentence-terminal
/// punctuation run (`.`/`!`/`?`, possibly repeated as in `"?!"` or `"..."`)
/// that is followed by whitespace or end-of-string. Returns `None` when `s`
/// has no complete sentence yet.
///
/// Guards exactly one false-positive: a `.` flanked by digits on both sides
/// (a decimal point, e.g. `"13.5"`) never counts as a sentence end — the
/// numeric-token shape this crate's own numeric hardening
/// (`tesseract_ocr::harden_numeric_tokens`) already treats as one unit.
fn find_sentence_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'.' || c == b'!' || c == b'?' {
            if c == b'.' {
                let prev_digit = i > 0 && bytes[i - 1].is_ascii_digit();
                let next_digit = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
                if prev_digit && next_digit {
                    i += 1;
                    continue;
                }
            }
            let mut j = i + 1;
            while j < bytes.len() && matches!(bytes[j], b'.' | b'!' | b'?') {
                j += 1;
            }
            if j >= bytes.len() || bytes[j].is_ascii_whitespace() {
                return Some(j);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    None
}

/// Build an [`AssembledSentence`] from its text plus the `DocPage` line
/// indices that contributed to it: bbox = union of those lines' bboxes,
/// `mean_conf` = mean over their words' confidences.
fn build_sentence(page: &DocPage, text: String, contributing: &[usize]) -> AssembledSentence {
    let mut bbox: Option<(i32, i32, i32, i32)> = None;
    let mut conf_sum = 0.0_f32;
    let mut conf_n = 0_usize;
    for &idx in contributing {
        let line = &page.lines[idx];
        bbox = Some(match bbox {
            None => line.bbox,
            Some(b) => (
                b.0.min(line.bbox.0),
                b.1.min(line.bbox.1),
                b.2.max(line.bbox.2),
                b.3.max(line.bbox.3),
            ),
        });
        for w in &line.words {
            conf_sum += w.conf;
            conf_n += 1;
        }
    }
    AssembledSentence {
        text,
        bbox: bbox.unwrap_or((0, 0, 0, 0)),
        line_indices: contributing.to_vec(),
        mean_conf: if conf_n > 0 {
            conf_sum / conf_n as f32
        } else {
            0.0
        },
    }
}

/// Assemble a [`DocPage`]'s lines into sentence-shaped text units — see the
/// module docs for the dehyphenation and sentence-boundary heuristics, and
/// their documented limitations. No text is ever dropped: a page whose
/// final line carries no terminal punctuation still emits one last
/// (unterminated) sentence for whatever text remains.
#[must_use]
pub fn assemble_sentences(page: &DocPage) -> Vec<AssembledSentence> {
    let mut sentences = Vec::new();
    let mut buf = String::new();
    let mut contributing: Vec<usize> = Vec::new();
    let mut hyphen_pending = false;

    for (idx, line) in page.lines.iter().enumerate() {
        if line.words.is_empty() {
            continue;
        }
        let text = line_text(line);
        if text.is_empty() {
            continue;
        }

        if hyphen_pending {
            buf.push_str(&text);
            hyphen_pending = false;
        } else if !buf.is_empty() {
            buf.push(' ');
            buf.push_str(&text);
        } else {
            buf.push_str(&text);
        }
        if !contributing.contains(&idx) {
            contributing.push(idx);
        }

        if ends_with_wrap_hyphen(&buf) {
            buf.pop();
            hyphen_pending = true;
            continue;
        }

        while let Some(cut) = find_sentence_end(&buf) {
            let sent_text = buf[..cut].trim().to_string();
            let remainder = buf[cut..].trim_start().to_string();
            if !sent_text.is_empty() {
                sentences.push(build_sentence(page, sent_text, &contributing));
            }
            if remainder.is_empty() {
                contributing.clear();
            }
            buf = remainder;
        }
    }

    if !buf.trim().is_empty() {
        sentences.push(build_sentence(page, buf.trim().to_string(), &contributing));
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;
    use tesseract_ocr::DocWord;

    fn word(text: &str, leading_space: bool, x: i32, conf: f32) -> DocWord {
        DocWord {
            text: text.to_string(),
            bbox: (x, 0, x + 10, 10),
            conf,
            leading_space,
            numeric_norm: None,
        }
    }

    fn line(words: Vec<DocWord>, top: i32) -> DocLine {
        let l = words.first().map(|w| w.bbox.0).unwrap_or(0);
        let r = words.last().map(|w| w.bbox.2).unwrap_or(0);
        DocLine {
            bbox: (l, top, r, top + 12),
            words,
            metrics: None,
        }
    }

    fn page(lines: Vec<DocLine>) -> DocPage {
        DocPage {
            width: 1000,
            height: 1000,
            lines,
        }
    }

    #[test]
    fn assemble_sentences_splits_on_terminal_punctuation() {
        let p = page(vec![
            line(
                vec![
                    word("The", false, 0, 90.0),
                    word("dog", true, 20, 90.0),
                    word("runs.", true, 40, 90.0),
                ],
                0,
            ),
            line(
                vec![
                    word("The", false, 0, 90.0),
                    word("cat", true, 20, 90.0),
                    word("sleeps.", true, 40, 90.0),
                ],
                20,
            ),
        ]);
        let sentences = assemble_sentences(&p);
        assert_eq!(
            sentences.len(),
            2,
            "two terminated lines must split into two sentences, not one"
        );
        assert_eq!(sentences[0].text, "The dog runs.");
        assert_eq!(sentences[1].text, "The cat sleeps.");
    }

    #[test]
    fn assemble_sentences_joins_wrapped_lines_without_terminator() {
        let p = page(vec![
            line(
                vec![word("The", false, 0, 90.0), word("dog", true, 20, 90.0)],
                0,
            ),
            line(
                vec![word("runs", false, 0, 90.0), word("today.", true, 20, 90.0)],
                20,
            ),
        ]);
        let sentences = assemble_sentences(&p);
        assert_eq!(
            sentences.len(),
            1,
            "an unterminated line must join, not split"
        );
        assert_eq!(sentences[0].text, "The dog runs today.");
        assert_eq!(
            sentences[0].line_indices,
            vec![0, 1],
            "the joined sentence must attribute both lines"
        );
    }

    #[test]
    fn assemble_sentences_dehyphenates_line_wrap() {
        let p = page(vec![
            line(
                vec![
                    word("This", false, 0, 90.0),
                    word("is", true, 20, 90.0),
                    word("a", true, 40, 90.0),
                    word("compli-", true, 60, 90.0),
                ],
                0,
            ),
            line(
                vec![
                    word("cated", false, 0, 90.0),
                    word("example.", true, 20, 90.0),
                ],
                20,
            ),
        ]);
        let sentences = assemble_sentences(&p);
        assert_eq!(sentences.len(), 1);
        assert_eq!(
            sentences[0].text, "This is a complicated example.",
            "dehyphenation must splice 'compli-' + 'cated' -> 'complicated', \
             not 'compli-cated' nor 'compli cated'"
        );
        assert!(!sentences[0].text.contains('-'));
    }

    #[test]
    fn assemble_sentences_does_not_dehyphenate_a_standalone_hyphen() {
        let p = page(vec![
            line(
                vec![
                    word("See", false, 0, 90.0),
                    word("below", true, 20, 90.0),
                    word("-", true, 40, 90.0),
                ],
                0,
            ),
            line(
                vec![
                    word("the", false, 0, 90.0),
                    word("answer", true, 20, 90.0),
                    word("is", true, 40, 90.0),
                    word("5.", true, 60, 90.0),
                ],
                20,
            ),
        ]);
        let sentences = assemble_sentences(&p);
        assert_eq!(sentences.len(), 1);
        assert_eq!(
            sentences[0].text, "See below - the answer is 5.",
            "a standalone '-' word must join with a space, not dehyphenate"
        );
    }

    #[test]
    fn assemble_sentences_guards_decimal_points_from_splitting() {
        let p = page(vec![line(
            vec![
                word("The", false, 0, 90.0),
                word("total", true, 20, 90.0),
                word("is", true, 40, 90.0),
                word("13.5", true, 60, 90.0),
                word("units.", true, 80, 90.0),
            ],
            0,
        )]);
        let sentences = assemble_sentences(&p);
        assert_eq!(
            sentences.len(),
            1,
            "a decimal point inside '13.5' must not split the sentence"
        );
        assert_eq!(sentences[0].text, "The total is 13.5 units.");
    }

    #[test]
    fn assemble_sentences_keeps_unterminated_trailing_text() {
        let p = page(vec![line(
            vec![word("No", false, 0, 90.0), word("period", true, 20, 90.0)],
            0,
        )]);
        let sentences = assemble_sentences(&p);
        assert_eq!(
            sentences.len(),
            1,
            "an unterminated trailing line must not be silently dropped"
        );
        assert_eq!(sentences[0].text, "No period");
    }

    #[test]
    fn assemble_sentences_empty_page_returns_empty() {
        assert!(assemble_sentences(&page(vec![])).is_empty());
    }

    #[test]
    fn assemble_sentences_computes_bbox_union_and_mean_conf() {
        let p = page(vec![
            line(
                vec![
                    word("Wide", false, 0, 80.0),
                    word("text.", true, 500, 100.0),
                ],
                0,
            ),
            line(vec![word("More.", false, 0, 60.0)], 20),
        ]);
        let sentences = assemble_sentences(&p);
        assert_eq!(sentences.len(), 2);
        // First sentence: bbox is the FIRST line's bbox (l=0, r=510), and
        // mean_conf is the mean of its two words (80, 100) = 90.
        assert_eq!(sentences[0].bbox, (0, 0, 510, 12));
        assert!((sentences[0].mean_conf - 90.0).abs() < 1e-6);
    }
}
