//! Page furniture detection — header / footer / page-number lines.
//!
//! **Consumer-side heuristic layer — NOT a Tesseract transcode.** Tesseract 5
//! has no notion of "page furniture": it emits words/lines/boxes and nothing
//! more. This module is this crate's own heuristic pass over the structured
//! DOM ([`crate::structured::DocPage`]), built to help downstream consumers
//! (e.g. a document-cleanup step) tell running-header / running-footer /
//! page-number lines apart from real body content. Nothing here feeds the
//! recognizer or changes recognition output, so no byte-parity claim applies
//! or is made — these are ordinary, falsifiable heuristics, tuned and
//! documented like any other application logic.
//!
//! ## The heuristics, in order
//!
//! 1. **Band membership.** A line is only ever *considered* furniture if its
//!    bbox lies **fully** inside the top [`HEADER_BAND_FRAC`] or bottom
//!    [`FOOTER_BAND_FRAC`] slice of the page height. A line that merely dips
//!    into a band (its bbox straddles the band boundary) is never furniture —
//!    that shape is exactly what a normal body paragraph that happens to sit
//!    near the edge looks like, and a partial-overlap rule would misfire on
//!    it.
//! 2. **Shape gate.** Within a band, a line only qualifies if it is *short*
//!    (joined text ≤ [`MAX_FURNITURE_TEXT_CHARS`] characters — running
//!    headers/footers are terse by nature) **or** its text matches one of the
//!    recognized [page-number shapes](#page-number-shapes). The shape check
//!    exists so that a page-number line is recognized even in the rare case
//!    it is phrased longer than the short-text cutoff (e.g. a verbose
//!    `"Seite 3 von 129"` on a fine-print footer).
//! 3. **Amount-label guard.** A line that would otherwise qualify but
//!    contains an amount label (`"netto"`, `"brutto"`, `"summe"`,
//!    case-insensitive) is explicitly excluded. Invoices routinely print
//!    their totals line in the bottom 8% of the page — without this guard a
//!    short `"Summe: 1.234,00"` sitting in the footer band would be
//!    misclassified as page furniture and stripped from the document, when
//!    it is exactly the content a downstream consumer wants kept.
//! 4. **Page-number extraction.** Among the qualifying lines, the value is
//!    read from whichever line's text matches a page-number shape (see
//!    below), preferring a **footer** match over a **header** match (page
//!    numbers live in the footer far more often than the header), and the
//!    first match within a band wins (top-to-bottom line order). Parsed
//!    numbers `> 9999` are rejected as spurious (nobody prints a five-digit
//!    page number; that's a false-positive digit run, not a page number).
//!
//! ### Page-number shapes
//!
//! Matched case-insensitively against the line's whitespace-joined, trimmed
//! text (hand-written parsers — this crate carries no regex dependency):
//!
//! - `^\d{1,4}$` — a bare number (`"7"`, `"128"`).
//! - `- \d{1,4} -` (spaces around the hyphens optional) — dash-bracketed
//!   (`"-42-"`, `"- 42 -"`).
//! - `Seite \d+( von \d+)?` — German "page N (of M)".
//! - `Page \d+( of \d+)?` — English "page N (of M)".
//! - `\d+ / \d+` — slash form; the first number is taken.
//!
//! ## Whitespace normalization
//!
//! Line text is built by joining word text with a single space
//! ([`line_text`]); `DocWord::leading_space` (which records the recognizer's
//! *own* inter-word gap for faithful text reconstruction elsewhere) is
//! deliberately ignored here — furniture matching only cares about the
//! sequence of words, not the exact recognized whitespace, so normalizing to
//! single spaces makes the shape parsers simpler without losing anything the
//! heuristics need.

use crate::structured::{DocLine, DocPage};

/// Top slice of the page height treated as the header band (8%).
pub const HEADER_BAND_FRAC: f64 = 0.08;

/// Bottom slice of the page height treated as the footer band (8%).
pub const FOOTER_BAND_FRAC: f64 = 0.08;

/// A line's joined text at or under this length is considered "short" —
/// eligible as furniture on length alone, independent of shape.
pub const MAX_FURNITURE_TEXT_CHARS: usize = 60;

/// Parsed page numbers above this are rejected as spurious.
pub const MAX_PAGE_NUMBER: u32 = 9999;

/// Amount labels that veto furniture classification even inside a band —
/// invoices print totals near the page bottom; without this guard a short
/// totals line reads as a footer and gets stripped.
const AMOUNT_LABEL_KEYWORDS: [&str; 3] = ["netto", "brutto", "summe"];

/// The result of a page-furniture scan: which lines look like running
/// header/footer content, and — if found — the page number and which line
/// carried it. All indices are into [`DocPage::lines`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PageFurniture {
    /// Indices of lines classified as header furniture.
    pub header_lines: Vec<usize>,
    /// Indices of lines classified as footer furniture.
    pub footer_lines: Vec<usize>,
    /// The extracted page number, if any qualifying line matched a
    /// page-number shape.
    pub page_number: Option<u32>,
    /// Index of the line [`page_number`](Self::page_number) was read from.
    pub page_number_line: Option<usize>,
}

/// Which band a line's bbox falls fully inside, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Band {
    Header,
    Footer,
}

/// Detect header/footer furniture lines and a page number on `page`. See the
/// module docs for the full heuristic chain.
#[must_use]
pub fn detect_page_furniture(page: &DocPage) -> PageFurniture {
    let page_bottom = page.height as i32;
    let header_bottom = band_extent(page.height, HEADER_BAND_FRAC);
    let footer_top = page_bottom - band_extent(page.height, FOOTER_BAND_FRAC);

    let mut header_lines = Vec::new();
    let mut footer_lines = Vec::new();
    // (line index, parsed page number if the line's text matched a shape)
    let mut header_candidates: Vec<(usize, Option<u32>)> = Vec::new();
    let mut footer_candidates: Vec<(usize, Option<u32>)> = Vec::new();

    for (idx, line) in page.lines.iter().enumerate() {
        let Some(band) = classify_band(line.bbox, header_bottom, footer_top, page_bottom) else {
            continue;
        };

        let text = line_text(line);
        let trimmed = text.trim();
        let lower = trimmed.to_ascii_lowercase();

        let is_short = trimmed.chars().count() <= MAX_FURNITURE_TEXT_CHARS;
        let shape_number = page_number_shape(&lower);
        if !is_short && shape_number.is_none() {
            continue;
        }
        if has_amount_label(&lower) {
            continue;
        }

        match band {
            Band::Header => {
                header_lines.push(idx);
                header_candidates.push((idx, shape_number));
            }
            Band::Footer => {
                footer_lines.push(idx);
                footer_candidates.push((idx, shape_number));
            }
        }
    }

    // Footer preference; first match wins within a band (line order).
    let picked = footer_candidates
        .iter()
        .find_map(|&(idx, n)| n.filter(|&n| n <= MAX_PAGE_NUMBER).map(|n| (idx, n)))
        .or_else(|| {
            header_candidates
                .iter()
                .find_map(|&(idx, n)| n.filter(|&n| n <= MAX_PAGE_NUMBER).map(|n| (idx, n)))
        });
    let (page_number, page_number_line) = match picked {
        Some((idx, n)) => (Some(n), Some(idx)),
        None => (None, None),
    };

    PageFurniture {
        header_lines,
        footer_lines,
        page_number,
        page_number_line,
    }
}

/// Round `page_height * frac` to the nearest pixel — the vertical extent of
/// one band.
fn band_extent(page_height: u32, frac: f64) -> i32 {
    ((page_height as f64) * frac).round() as i32
}

/// Classify a line's bbox as fully inside the header band, fully inside the
/// footer band, or neither. "Fully inside" (both `top` and `bottom` within
/// the band) is deliberate: a line whose bbox merely straddles a band
/// boundary is the shape of an ordinary body paragraph running near the page
/// edge, not a running header/footer, and must never be swept up.
fn classify_band(
    bbox: (i32, i32, i32, i32),
    header_bottom: i32,
    footer_top: i32,
    page_bottom: i32,
) -> Option<Band> {
    let (_, top, _, bottom) = bbox;
    if top >= 0 && bottom <= header_bottom {
        Some(Band::Header)
    } else if top >= footer_top && bottom <= page_bottom {
        Some(Band::Footer)
    } else {
        None
    }
}

/// Join a line's words with single spaces. `DocWord::leading_space` is
/// ignored on purpose — see the module docs' "Whitespace normalization"
/// section: furniture matching only needs the word sequence, and
/// single-space joining keeps the shape parsers simple.
fn line_text(line: &DocLine) -> String {
    line.words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Does `lower` (already-lowercased text) contain an amount label that
/// vetoes furniture classification?
fn has_amount_label(lower: &str) -> bool {
    AMOUNT_LABEL_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Try every page-number shape against `lower` (already trimmed +
/// lowercased); returns the parsed number of the first shape that matches
/// the **entire** string. `None` if nothing matches.
fn page_number_shape(lower: &str) -> Option<u32> {
    if lower.is_empty() {
        return None;
    }
    parse_bare_number(lower)
        .or_else(|| parse_dashed_number(lower))
        .or_else(|| parse_labeled_page(lower, "seite", "von"))
        .or_else(|| parse_labeled_page(lower, "page", "of"))
        .or_else(|| parse_slash_number(lower))
}

/// `^\d{1,4}$` — the whole (already-trimmed) string is 1-4 ASCII digits.
fn parse_bare_number(lower: &str) -> Option<u32> {
    if lower.is_empty() || lower.len() > 4 || !lower.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    lower.parse().ok()
}

/// `- \d{1,4} -` with spaces around either hyphen optional — strip all
/// whitespace, then require a leading and trailing `-` wrapping 1-4 digits.
fn parse_dashed_number(lower: &str) -> Option<u32> {
    let stripped: String = lower.chars().filter(|c| !c.is_whitespace()).collect();
    let inner = stripped.strip_prefix('-')?.strip_suffix('-')?;
    if inner.is_empty() || inner.len() > 4 || !inner.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    inner.parse().ok()
}

/// `<label> \d+( <connector> \d+)?`, anchored to the whole string —
/// `"seite"/"von"` for German, `"page"/"of"` for English. The optional
/// `<connector> \d+` tail must either be entirely absent or fully well
/// formed; any other trailing text fails the match (this is a full-line
/// shape, not a prefix search).
fn parse_labeled_page(lower: &str, label: &str, connector: &str) -> Option<u32> {
    let prefix = format!("{label} ");
    let rest = lower.strip_prefix(&prefix)?;
    let (num_str, remainder) = take_leading_digits(rest);
    if num_str.is_empty() {
        return None;
    }
    let num: u32 = num_str.parse().ok()?;

    let remainder = remainder.trim();
    if remainder.is_empty() {
        return Some(num);
    }

    let conn_prefix = format!("{connector} ");
    let after = remainder.strip_prefix(&conn_prefix)?;
    let (total_str, total_rest) = take_leading_digits(after);
    if total_str.is_empty() || !total_rest.trim().is_empty() {
        return None;
    }
    Some(num)
}

/// `\d+ / \d+`, anchored to the whole string — the first number is
/// returned (the second is required for the shape to match but is
/// otherwise unused, per spec: "take the first number").
fn parse_slash_number(lower: &str) -> Option<u32> {
    let mut parts = lower.splitn(2, '/');
    let left = parts.next()?.trim();
    let right = parts.next()?.trim();
    if left.is_empty()
        || right.is_empty()
        || !left.chars().all(|c| c.is_ascii_digit())
        || !right.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    left.parse().ok()
}

/// Split `s` into its leading run of ASCII digits and the remainder.
fn take_leading_digits(s: &str) -> (&str, &str) {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    (&s[..end], &s[end..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured::DocWord;

    /// Build a [`DocWord`] with the fields this module's tests never vary.
    fn dw(text: &str, bbox: (i32, i32, i32, i32)) -> DocWord {
        DocWord {
            text: text.to_string(),
            bbox,
            conf: 95.0,
            leading_space: true,
            numeric_norm: None,
        }
    }

    fn dl(bbox: (i32, i32, i32, i32), words: Vec<DocWord>) -> DocLine {
        DocLine { bbox, words }
    }

    /// A 800x1000 page: header band = y in [0, 80], footer band = y in
    /// [920, 1000] (8% of 1000 each way).
    fn page_with_lines(lines: Vec<DocLine>) -> DocPage {
        DocPage {
            width: 800,
            height: 1000,
            lines,
        }
    }

    #[test]
    fn footer_seite_von_line_is_furniture_and_yields_page_number() {
        let line = dl(
            (300, 930, 500, 960),
            vec![
                dw("Seite", (300, 930, 340, 960)),
                dw("3", (345, 930, 355, 960)),
                dw("von", (360, 930, 390, 960)),
                dw("12", (395, 930, 415, 960)),
            ],
        );
        let page = page_with_lines(vec![line]);
        let furniture = detect_page_furniture(&page);

        assert_eq!(furniture.footer_lines, vec![0]);
        assert!(furniture.header_lines.is_empty());
        assert_eq!(furniture.page_number, Some(3));
        assert_eq!(furniture.page_number_line, Some(0));
    }

    #[test]
    fn bare_number_centered_in_footer_is_page_number() {
        let line = dl((380, 940, 420, 960), vec![dw("7", (390, 940, 400, 960))]);
        let page = page_with_lines(vec![line]);
        let furniture = detect_page_furniture(&page);

        assert_eq!(furniture.footer_lines, vec![0]);
        assert_eq!(furniture.page_number, Some(7));
        assert_eq!(furniture.page_number_line, Some(0));
    }

    #[test]
    fn footer_page_number_is_preferred_over_header() {
        // Header: dash-bracketed "- 42 -".
        let header_line = dl(
            (350, 10, 450, 40),
            vec![
                dw("-", (350, 10, 360, 40)),
                dw("42", (365, 10, 385, 40)),
                dw("-", (390, 10, 400, 40)),
            ],
        );
        // Footer: bare "8".
        let footer_line = dl((390, 950, 410, 970), vec![dw("8", (390, 950, 400, 970))]);
        let page = page_with_lines(vec![header_line, footer_line]);
        let furniture = detect_page_furniture(&page);

        assert_eq!(furniture.header_lines, vec![0]);
        assert_eq!(furniture.footer_lines, vec![1]);
        // Header line alone would say 42, but the footer wins.
        assert_eq!(furniture.page_number, Some(8));
        assert_eq!(furniture.page_number_line, Some(1));
    }

    #[test]
    fn line_only_partially_inside_footer_band_is_not_furniture() {
        // A long body-text line whose bbox starts above the footer band
        // (top = 900 < footer_top = 920) and ends inside it (bottom = 950):
        // it straddles the boundary, so the fully-inside test must reject
        // it regardless of its own length.
        let text = "This is a long paragraph line of body text that runs \
                     well past sixty characters in total length.";
        let words: Vec<DocWord> = text
            .split(' ')
            .enumerate()
            .map(|(i, w)| dw(w, (10 + i as i32 * 20, 900, 30 + i as i32 * 20, 950)))
            .collect();
        let line = dl((10, 900, 500, 950), words);
        let page = page_with_lines(vec![line]);
        let furniture = detect_page_furniture(&page);

        assert!(furniture.header_lines.is_empty());
        assert!(furniture.footer_lines.is_empty());
        assert_eq!(furniture.page_number, None);
    }

    #[test]
    fn amount_label_in_footer_band_is_never_furniture() {
        let line = dl(
            (300, 940, 500, 960),
            vec![
                dw("Summe:", (300, 940, 360, 960)),
                dw("1.234,00", (365, 940, 440, 960)),
            ],
        );
        let page = page_with_lines(vec![line]);
        let furniture = detect_page_furniture(&page);

        assert!(furniture.footer_lines.is_empty());
        assert!(furniture.header_lines.is_empty());
        assert_eq!(furniture.page_number, None);
    }

    #[test]
    fn page_of_line_yields_page_number() {
        let line = dl(
            (300, 935, 520, 960),
            vec![
                dw("Page", (300, 935, 340, 960)),
                dw("12", (345, 935, 365, 960)),
                dw("of", (370, 935, 390, 960)),
                dw("30", (395, 935, 415, 960)),
            ],
        );
        let page = page_with_lines(vec![line]);
        let furniture = detect_page_furniture(&page);

        assert_eq!(furniture.footer_lines, vec![0]);
        assert_eq!(furniture.page_number, Some(12));
        assert_eq!(furniture.page_number_line, Some(0));
    }

    // --- shape parser unit coverage -----------------------------------

    #[test]
    fn page_number_shape_matches_all_documented_forms() {
        assert_eq!(page_number_shape("7"), Some(7));
        assert_eq!(page_number_shape("128"), Some(128));
        assert_eq!(page_number_shape("-42-"), Some(42));
        assert_eq!(page_number_shape("- 42 -"), Some(42));
        assert_eq!(page_number_shape("seite 3"), Some(3));
        assert_eq!(page_number_shape("seite 3 von 12"), Some(3));
        assert_eq!(page_number_shape("page 12"), Some(12));
        assert_eq!(page_number_shape("page 12 of 30"), Some(12));
        assert_eq!(page_number_shape("12 / 30"), Some(12));
        assert_eq!(page_number_shape("not a page number"), None);
        assert_eq!(page_number_shape(""), None);
        // Malformed connector tail fails the (anchored) shape entirely.
        assert_eq!(page_number_shape("seite 3 von"), None);
        assert_eq!(page_number_shape("seite 3 vonx 12"), None);
    }

    #[test]
    fn parsed_page_numbers_above_9999_are_rejected_in_selection() {
        let line = dl(
            (300, 935, 520, 960),
            vec![
                dw("Seite", (300, 935, 340, 960)),
                dw("12345", (345, 935, 400, 960)),
            ],
        );
        let page = page_with_lines(vec![line]);
        let furniture = detect_page_furniture(&page);

        // Still short enough to be furniture-eligible on length, so it is
        // still classified as a footer line...
        assert_eq!(furniture.footer_lines, vec![0]);
        // ...but its out-of-range number is rejected as the page number.
        assert_eq!(furniture.page_number, None);
        assert_eq!(furniture.page_number_line, None);
    }
}
