//! Structured document output — a JSON DOM (`doc.v1`) over the recognizer's
//! word results, plus a typed-field harvest for invoice/form pages
//! (numeric hardening, IBAN checksum, label-proximity field extraction).
//!
//! **Consumer-side layer — NOT a Tesseract transcode.** Tesseract 5 has no
//! JSON renderer and no field-extraction stage; this module is this crate's
//! own output surface, built ON TOP of the proven pipeline (word text via
//! `ids_to_text`, boxes/confidence exactly as the TSV/hOCR renderers derive
//! them — see `renderer.rs` for those transcodes). Nothing here feeds back
//! into recognition, so no parity claim applies or is made.
//!
//! ## The JSON shape (`schema: "tesseract-rs/doc.v1"`)
//!
//! ```json
//! {
//!   "schema": "tesseract-rs/doc.v1",
//!   "pages": [{
//!     "page": 1, "width": 2480, "height": 3508,
//!     "quality": {"mean_conf": 96.10, "low_confidence": false},
//!     "regions": [{
//!       "type": "paragraph",
//!       "bbox": [l, t, r, b],
//!       "lines": [{
//!         "bbox": [l, t, r, b],
//!         "words": [{"text": "…", "bbox": [l, t, r, b], "conf": 96.5,
//!                     "leading_space": true, "numeric_norm": "250"}]
//!       }]
//!     }],
//!     "fields": [{"key": "netto", "label": "Netto:", "value": "1.250,00",
//!                  "value_cents": 125000, "bbox": [l, t, r, b],
//!                  "conf": 96.1, "checks": ["arithmetic_ok"]}]
//!   }]
//! }
//! ```
//!
//! - `bbox` is always top-down image coordinates `[left, top, right, bottom]`
//!   (the hOCR convention, via the same `PageIterator::BoundingBox` transcode
//!   the other renderers use).
//! - `conf` is the same 0–100 word confidence as TSV/hOCR
//!   (`ClipToRange(100 + 5·min(cert))`).
//! - `regions`: [`render_json`] emits ONE `"paragraph"` region (the plain
//!   default, byte-stable); [`render_json_with_regions`] emits CLASSIFIED
//!   regions built by [`build_regions`] from the layout stack — `"text"`
//!   (XY-cut blocks, reading order), `"figure"` (halftone-mask components),
//!   `"header"`/`"footer"` (page furniture). Additive `type` values;
//!   consumers must ignore unknown ones.
//! - `quality.mean_conf` is the mean word confidence 0–100 (`null` when no
//!   words), and `low_confidence` flags a page below
//!   [`LOW_CONFIDENCE_THRESHOLD`] — the honesty signal that the input is
//!   likely handwriting / low-resolution / not printed text (`eng.lstm` is
//!   print-trained). See [`mean_word_confidence`].
//! - `numeric_norm` appears only on words the numeric hardening pass changed;
//!   `fields` only when a harvest ran. Consumers must ignore unknown keys.
//!
//! ## Numeric hardening — "eine 0 kann nie ein O sein"
//!
//! In a digit-dominated token, confusable LETTERS are OCR misreads of digits
//! and are normalized: `O/o→0`, `I/l/|→1`, `Z/z→2`, `S/s→5`, `B→8`, `G→6`.
//! Guards keep legitimately-mixed identifiers untouched: GUIDs (hex+dash
//! shape), IBANs (checksum-validated instead), and any token where digits do
//! not strictly dominate letters (so `Summe`, `B8`-style codes, part numbers
//! survive). The original text is never destroyed — the normalized form goes
//! to `numeric_norm` alongside `text`.

use tesseract_core::CharSet;

use crate::renderer::LineWords;

/// One word in the structured DOM: rendered text, top-down image box,
/// 0–100 confidence, and the optional numeric-hardened form.
#[derive(Clone, Debug, PartialEq)]
pub struct DocWord {
    /// The word's text (`ids_to_text` over its unichar ids).
    pub text: String,
    /// Top-down image box `(left, top, right, bottom)`.
    pub bbox: (i32, i32, i32, i32),
    /// Word confidence 0–100 (same derivation as the TSV/hOCR renderers).
    pub conf: f32,
    /// Whether the recognizer emitted a leading space before this word.
    pub leading_space: bool,
    /// Set by [`harden_numeric_tokens`] iff the hardening changed the text.
    pub numeric_norm: Option<String>,
}

/// One recognized line: its top-down image box plus its words.
#[derive(Clone, Debug, PartialEq)]
pub struct DocLine {
    /// Top-down image box `(left, top, right, bottom)` of the line band.
    pub bbox: (i32, i32, i32, i32),
    /// Words in reading order.
    pub words: Vec<DocWord>,
}

/// One page of structured output — the unit [`render_json`] serializes.
#[derive(Clone, Debug, PartialEq)]
pub struct DocPage {
    /// Page width in pixels.
    pub width: u32,
    /// Page height in pixels.
    pub height: u32,
    /// Non-empty lines, top-to-bottom (empty lines are skipped, same rule as
    /// every other renderer in this crate).
    pub lines: Vec<DocLine>,
}

impl DocPage {
    /// Build the structured DOM from recognizer line output — the same
    /// `LineWords` unit the TSV/hOCR renderers consume, converted through the
    /// same box/confidence derivations (`to_image_box`, `word_confidence`,
    /// `word_text` — see `renderer.rs` for those transcodes). Empty lines are
    /// skipped entirely.
    #[must_use]
    pub fn from_line_words(
        lines: &[LineWords],
        charset: &CharSet,
        page_w: u32,
        page_h: u32,
    ) -> Self {
        let pw = page_w as i32;
        let ph = page_h as i32;
        let doc_lines = lines
            .iter()
            .filter(|l| !l.words.is_empty())
            .map(|line| DocLine {
                bbox: crate::renderer::to_image_box(line.line_box, pw, ph),
                words: line
                    .words
                    .iter()
                    .map(|w| DocWord {
                        text: crate::renderer::word_text(charset, w),
                        bbox: crate::renderer::to_image_box(
                            crate::renderer::union_boxes(w.char_boxes.iter().copied()),
                            pw,
                            ph,
                        ),
                        conf: crate::renderer::word_confidence(w),
                        leading_space: w.leading_space,
                        numeric_norm: None,
                    })
                    .collect(),
            })
            .collect();
        DocPage {
            width: page_w,
            height: page_h,
            lines: doc_lines,
        }
    }
}

/// Escape a string for a JSON string literal (RFC 8259 §7): `"` and `\` get
/// backslash escapes, the C0 controls use the short forms where they exist
/// (`\n` `\r` `\t` `\b` `\f`) and `\u00XX` otherwise. Everything else —
/// including all non-ASCII UTF-8 — passes through verbatim (JSON strings are
/// Unicode; no `\uXXXX` escaping of printable text).
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Format a bbox as the JSON array `[l,t,r,b]`.
fn json_bbox(b: (i32, i32, i32, i32)) -> String {
    format!("[{},{},{},{}]", b.0, b.1, b.2, b.3)
}

/// The mean word confidence (0–100) over all words on the page, or `None`
/// when the page has no words. This is the page-level quality signal: the
/// recognizer's per-word confidence is the min over its character CTC
/// certainties (`100 + 5·min_cert`, clamped) — on clean printed text it sits
/// in the high 80s–100s; on OUT-OF-DISTRIBUTION input (handwriting,
/// low-resolution, or non-text) the softmax flattens, the certainties
/// collapse, and the mean drops sharply. It is a signal, NOT a proof — see
/// [`LOW_CONFIDENCE_THRESHOLD`].
#[must_use]
pub fn mean_word_confidence(page: &DocPage) -> Option<f32> {
    let (sum, n) = page
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .fold((0.0_f32, 0_usize), |(s, n), w| (s + w.conf, n + 1));
    (n > 0).then(|| sum / n as f32)
}

/// The mean-confidence floor below which a page is flagged `low_confidence`
/// in `doc.v1` — a **heuristic**, deliberately conservative, NOT calibrated
/// against a labelled handwriting corpus. `eng.lstm` is a PRINT-trained model
/// (Tesseract's CTC-LSTM has no handwriting support in the standard tessdata),
/// so a handwritten or otherwise unreadable page produces confidently-shaped
/// but low-certainty garbage; this floor lets a consumer surface "the model is
/// not confident — this may be handwriting / low-res / not printed text"
/// instead of returning the garbage silently. Tune per deployment; the raw
/// `mean_conf` value is always emitted so a consumer can apply its own gate.
pub const LOW_CONFIDENCE_THRESHOLD: f32 = 65.0;

/// Internal emit unit shared by both renderers: the `type` string, the
/// region bbox, and the owned line indices.
type EmitRegion<'a> = (&'a str, (i32, i32, i32, i32), Vec<usize>);

/// Serialize one page (plus an optional field harvest) as a `doc.v1` JSON
/// document — see the module docs for the schema. `fields` may be empty
/// (serialized as `"fields":[]` so the key is always present and consumers
/// never need an existence check).
///
/// Confidences print with two decimals (`{:.2}`) — enough to preserve the
/// 0.5-steps the `100 + 5·cert` formula produces, without float noise.
#[must_use]
pub fn render_json(page: &DocPage, fields: &[HarvestedField]) -> String {
    // Default region synthesis: one "paragraph" over all lines — bbox = union
    // of the line boxes, same APPROX policy as TSV/hOCR's block/par rows. An
    // all-empty page emits an empty regions array. (The typed-region surface
    // is [`render_json_with_regions`]; this default keeps the plain renderer's
    // output byte-stable.)
    let default_regions: Vec<EmitRegion> = if page.lines.is_empty() {
        Vec::new()
    } else {
        let region_box = page.lines.iter().skip(1).fold(page.lines[0].bbox, |a, l| {
            let b = l.bbox;
            (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
        });
        vec![("paragraph", region_box, (0..page.lines.len()).collect())]
    };
    render_doc(page, &default_regions, fields)
}

/// The kind of a classified [`DocRegion`] — the `type` value it serializes
/// as in `doc.v1`. Additive to the schema: `"paragraph"` (the
/// [`render_json`] default) and these four coexist; consumers must ignore
/// unknown values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Body text (an XY-cut block's lines).
    Text,
    /// An image / halftone region (from the halftone mask; carries no lines).
    Figure,
    /// Page-furniture header lines.
    Header,
    /// Page-furniture footer lines.
    Footer,
}

impl RegionKind {
    /// The `doc.v1` `type` string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RegionKind::Text => "text",
            RegionKind::Figure => "figure",
            RegionKind::Header => "header",
            RegionKind::Footer => "footer",
        }
    }
}

/// One classified page region: its kind, its top-down bbox, and the indices
/// of the [`DocPage::lines`] it owns (empty for [`RegionKind::Figure`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocRegion {
    /// The region's kind (serialized as the `type` value).
    pub kind: RegionKind,
    /// Top-down image bbox `(left, top, right, bottom)`.
    pub bbox: (i32, i32, i32, i32),
    /// Indices into [`DocPage::lines`], in reading order.
    pub line_indices: Vec<usize>,
}

/// Union of two top-down boxes.
fn union_bbox(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
}

/// Assemble typed regions from the classifier outputs:
///
/// - `header_lines` / `footer_lines` — line indices from the page-furniture
///   detector (`crate::page_furniture`).
/// - `blocks` — layout blocks in READING ORDER (e.g. `crate::xy_cut` leaves as
///   top-down `(l, t, r, b)`); each remaining line joins the FIRST block
///   containing its bbox center.
/// - `figures` — image-region bboxes (e.g. halftone-mask components,
///   `crate::pageseg`); they own no lines.
///
/// Emission order: header, blocks (with their lines, block order), a
/// catch-all `Text` region for body lines no block claimed (only if any),
/// figures, footer. Line-bearing regions get the union of their lines'
/// bboxes; empty blocks are dropped.
#[must_use]
pub fn build_regions(
    page: &DocPage,
    header_lines: &[usize],
    footer_lines: &[usize],
    blocks: &[(i32, i32, i32, i32)],
    figures: &[(i32, i32, i32, i32)],
) -> Vec<DocRegion> {
    let mut block_members: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    let mut header: Vec<usize> = Vec::new();
    let mut footer: Vec<usize> = Vec::new();
    let mut orphans: Vec<usize> = Vec::new();

    for (i, line) in page.lines.iter().enumerate() {
        if header_lines.contains(&i) {
            header.push(i);
            continue;
        }
        if footer_lines.contains(&i) {
            footer.push(i);
            continue;
        }
        let cx = (line.bbox.0 + line.bbox.2) / 2;
        let cy = (line.bbox.1 + line.bbox.3) / 2;
        match blocks
            .iter()
            .position(|&(l, t, r, b)| cx >= l && cx < r && cy >= t && cy < b)
        {
            Some(bi) => block_members[bi].push(i),
            None => orphans.push(i),
        }
    }

    let lines_region = |kind: RegionKind, members: &[usize]| -> Option<DocRegion> {
        let first = *members.first()?;
        let bbox = members
            .iter()
            .skip(1)
            .fold(page.lines[first].bbox, |a, &i| {
                union_bbox(a, page.lines[i].bbox)
            });
        Some(DocRegion {
            kind,
            bbox,
            line_indices: members.to_vec(),
        })
    };

    let mut out: Vec<DocRegion> = Vec::new();
    out.extend(lines_region(RegionKind::Header, &header));
    for members in &block_members {
        out.extend(lines_region(RegionKind::Text, members));
    }
    out.extend(lines_region(RegionKind::Text, &orphans));
    out.extend(figures.iter().map(|&bbox| DocRegion {
        kind: RegionKind::Figure,
        bbox,
        line_indices: Vec::new(),
    }));
    out.extend(lines_region(RegionKind::Footer, &footer));
    out
}

/// Serialize with CLASSIFIED regions (see [`build_regions`]) instead of the
/// single-`"paragraph"` default. Same `doc.v1` shape; `type` takes the
/// [`RegionKind`] values.
#[must_use]
pub fn render_json_with_regions(
    page: &DocPage,
    regions: &[DocRegion],
    fields: &[HarvestedField],
) -> String {
    let mapped: Vec<EmitRegion> = regions
        .iter()
        .map(|r| (r.kind.as_str(), r.bbox, r.line_indices.clone()))
        .collect();
    render_doc(page, &mapped, fields)
}

/// The shared `doc.v1` emitter over `(type, bbox, line indices)` regions —
/// both public renderers route through here so the schema cannot fork.
fn render_doc(page: &DocPage, regions: &[EmitRegion], fields: &[HarvestedField]) -> String {
    let mut out = String::new();
    out.push_str("{\"schema\":\"tesseract-rs/doc.v1\",\"pages\":[{");
    out.push_str(&format!(
        "\"page\":1,\"width\":{},\"height\":{},",
        page.width, page.height
    ));

    // Page-level quality signal (the honesty layer): mean word confidence +
    // the low-confidence flag. `mean_conf` is `null` on a page with no words.
    match mean_word_confidence(page) {
        Some(mc) => out.push_str(&format!(
            "\"quality\":{{\"mean_conf\":{:.2},\"low_confidence\":{}}},",
            mc,
            mc < LOW_CONFIDENCE_THRESHOLD
        )),
        None => out.push_str("\"quality\":{\"mean_conf\":null,\"low_confidence\":false},"),
    }

    out.push_str("\"regions\":[");
    for (ri, (kind, bbox, line_indices)) in regions.iter().enumerate() {
        if ri > 0 {
            out.push(',');
        }
        out.push_str(&format!("{{\"type\":\"{kind}\",\"bbox\":"));
        out.push_str(&json_bbox(*bbox));
        out.push_str(",\"lines\":[");
        for (li, &line_idx) in line_indices.iter().enumerate() {
            let line = &page.lines[line_idx];
            if li > 0 {
                out.push(',');
            }
            out.push_str("{\"bbox\":");
            out.push_str(&json_bbox(line.bbox));
            out.push_str(",\"words\":[");
            for (wi, w) in line.words.iter().enumerate() {
                if wi > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"text\":\"{}\",\"bbox\":{},\"conf\":{:.2},\"leading_space\":{}",
                    json_escape(&w.text),
                    json_bbox(w.bbox),
                    w.conf,
                    w.leading_space
                ));
                if let Some(norm) = &w.numeric_norm {
                    out.push_str(&format!(",\"numeric_norm\":\"{}\"", json_escape(norm)));
                }
                out.push('}');
            }
            out.push_str("]}");
        }
        out.push_str("]}");
    }
    out.push_str("],");

    out.push_str("\"fields\":[");
    for (fi, f) in fields.iter().enumerate() {
        if fi > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"key\":\"{}\",\"label\":\"{}\",\"value\":\"{}\"",
            json_escape(&f.key),
            json_escape(&f.label_text),
            json_escape(&f.value)
        ));
        if let Some(cents) = f.value_cents {
            out.push_str(&format!(",\"value_cents\":{cents}"));
        }
        out.push_str(&format!(
            ",\"bbox\":{},\"conf\":{:.2},\"checks\":[",
            json_bbox(f.bbox),
            f.conf
        ));
        for (ci, c) in f.checks.iter().enumerate() {
            if ci > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\"", json_escape(c)));
        }
        out.push_str("]}");
    }
    out.push_str("]}]}");
    out
}

// ---------------------------------------------------------------------------
// Numeric hardening — typed constraint over digit-context tokens
// ---------------------------------------------------------------------------

/// Map one confusable letter to the digit it is a misread of, if any.
/// The six classic OCR confusion pairs; anything else returns `None`.
fn confusable_digit(ch: char) -> Option<char> {
    match ch {
        'O' | 'o' => Some('0'),
        'I' | 'l' | '|' => Some('1'),
        'Z' | 'z' => Some('2'),
        'S' | 's' => Some('5'),
        'B' => Some('8'),
        'G' => Some('6'),
        _ => None,
    }
}

/// Characters that may appear inside a numeric token without disqualifying
/// it: digit separators, decimal marks, currency, sign, percent.
fn is_numeric_furniture(ch: char) -> bool {
    matches!(
        ch,
        '.' | ',' | '\'' | '-' | '+' | '%' | '€' | '$' | '£' | ' '
    )
}

/// GUID shape (`8-4-4-4-12` hex groups, any case) — such tokens legitimately
/// mix letters into digit runs and must never be "corrected".
#[must_use]
pub fn looks_like_guid(token: &str) -> bool {
    let groups: Vec<&str> = token.split('-').collect();
    groups.len() == 5
        && [8usize, 4, 4, 4, 12]
            .iter()
            .zip(&groups)
            .all(|(&len, g)| g.len() == len && g.chars().all(|c| c.is_ascii_hexdigit()))
}

/// IBAN shape: 2 ASCII letters + 2 digits + 11..=30 alphanumerics (total
/// 15..=34). Shape only — [`iban_mod97_ok`] is the actual validation.
#[must_use]
pub fn looks_like_iban(token: &str) -> bool {
    let bytes = token.as_bytes();
    (15..=34).contains(&bytes.len())
        && bytes[0].is_ascii_alphabetic()
        && bytes[1].is_ascii_alphabetic()
        && bytes[2].is_ascii_digit()
        && bytes[3].is_ascii_digit()
        && bytes[4..].iter().all(u8::is_ascii_alphanumeric)
}

/// ISO 13616 / ISO 7064 mod-97-10 IBAN checksum: move the first four chars to
/// the end, map `A..Z → 10..35`, and the resulting decimal number must be
/// `≡ 1 (mod 97)`. Case-insensitive; the input must already be shape-valid
/// ([`looks_like_iban`]) — spaces are NOT accepted here (join groups first).
#[must_use]
pub fn iban_mod97_ok(iban: &str) -> bool {
    if !looks_like_iban(iban) {
        return false;
    }
    let upper = iban.to_ascii_uppercase();
    let rearranged = format!("{}{}", &upper[4..], &upper[..4]);
    let mut rem: u32 = 0;
    for ch in rearranged.chars() {
        if let Some(d) = ch.to_digit(10) {
            rem = (rem * 10 + d) % 97;
        } else {
            // A=10 .. Z=35 — two decimal digits, folded in incrementally.
            let v = (ch as u32) - ('A' as u32) + 10;
            rem = (rem * 100 + v) % 97;
        }
    }
    rem == 1
}

/// Harden one token: if it is digit-DOMINATED (≥ 2 digits, strictly more
/// digits than letters, and every letter confusable + every other char
/// numeric furniture) — and NOT a GUID/IBAN — replace each confusable letter
/// with its digit. Returns `Some(normalized)` only when something changed.
///
/// The dominance gate is deliberately conservative: a token with as many
/// letters as digits (`B8`, `A1`) or any non-confusable letter (`Summe`,
/// `Rechnung`, part numbers like `XK-250`) is left untouched. One misread in
/// a real amount (`1.O50` → `1.050`, `2S0,00` → `250,00`) passes; a token
/// that would need half its characters "fixed" does not.
#[must_use]
pub fn harden_numeric_token(token: &str) -> Option<String> {
    if looks_like_guid(token) || looks_like_iban(token) {
        return None;
    }
    let mut digits = 0usize;
    let mut letters = 0usize;
    for ch in token.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else if ch.is_alphabetic() || ch == '|' {
            // A genuine (non-confusable) letter → not a numeric token.
            confusable_digit(ch)?;
            letters += 1;
        } else if !is_numeric_furniture(ch) {
            return None; // something structural (slash, colon, …) — leave it
        }
    }
    if letters == 0 || digits < 2 || digits <= letters {
        return None;
    }
    Some(
        token
            .chars()
            .map(|c| confusable_digit(c).unwrap_or(c))
            .collect(),
    )
}

/// Run [`harden_numeric_token`] over every word of a page, filling
/// [`DocWord::numeric_norm`] where the hardening fired. The original `text`
/// is never modified — consumers choose which form to trust.
pub fn harden_numeric_tokens(page: &mut DocPage) {
    for line in &mut page.lines {
        for word in &mut line.words {
            word.numeric_norm = harden_numeric_token(&word.text);
        }
    }
}

/// Parse a printed amount into cents. Handles the German and English
/// conventions: `1.250,00` / `1,250.00` (grouped + 2-digit decimals),
/// `1250,00` / `1250.00`, bare integers (`99` → `9900`), and currency/sign
/// furniture (`€ 99,50`, `-12,00`). A single separator followed by exactly
/// two digits at the end is the decimal mark; otherwise separators group
/// thousands. Returns `None` for anything that doesn't parse cleanly.
#[must_use]
pub fn parse_amount_cents(token: &str) -> Option<i64> {
    let cleaned: String = token
        .chars()
        .filter(|c| !matches!(c, '€' | '$' | '£' | ' ' | '\'' | '+'))
        .collect();
    let (neg, body) = match cleaned.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, cleaned.as_str()),
    };
    if body.is_empty()
        || !body
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
    {
        return None;
    }
    // Decide the decimal separator: the LAST '.' or ',' — but only if exactly
    // two digits follow it (printed money always has 2 decimals; a trailing
    // 3-digit group is a thousands group: "1.250" = 1250,00).
    let last_sep = body.rfind(['.', ',']);
    let (int_part, frac_part): (String, i64) = match last_sep {
        Some(pos) if body.len() - pos - 1 == 2 => {
            let frac: i64 = body[pos + 1..].parse().ok()?;
            (
                body[..pos].chars().filter(char::is_ascii_digit).collect(),
                frac,
            )
        }
        _ => (body.chars().filter(char::is_ascii_digit).collect(), 0),
    };
    if int_part.is_empty() && frac_part == 0 && body.chars().all(|c| !c.is_ascii_digit()) {
        return None;
    }
    let int_val: i64 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().ok()?
    };
    let cents = int_val.checked_mul(100)?.checked_add(frac_part)?;
    Some(if neg { -cents } else { cents })
}

// ---------------------------------------------------------------------------
// Field harvest — label proximity + arithmetic cross-check
// ---------------------------------------------------------------------------

/// What kind of value a field expects — drives candidate filtering and
/// which checks run on the harvested value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// A money amount — candidate must satisfy [`parse_amount_cents`]
    /// (after numeric hardening).
    Amount,
    /// An identifier (invoice no, tax no, IBAN, …) — candidate is taken as
    /// text; IBAN-shaped values get the mod-97 check.
    Id,
}

/// One field to look for: a stable output key, the label spellings that
/// mark it on the page, and the expected value kind.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    /// Stable key in the JSON output (`"netto"`, `"iban"`, …).
    pub key: &'static str,
    /// Lowercase label prefixes that identify this field on the page
    /// (matched against a lowercased, `:`-stripped label word).
    pub labels: &'static [&'static str],
    /// Expected value kind.
    pub kind: FieldKind,
}

/// The default German invoice field set: net/tax/gross amounts (wired to the
/// arithmetic cross-check), invoice number, tax numbers, IBAN.
#[must_use]
pub fn german_invoice_fields() -> Vec<FieldSpec> {
    vec![
        FieldSpec {
            key: "netto",
            labels: &["netto", "nettobetrag", "zwischensumme"],
            kind: FieldKind::Amount,
        },
        FieldSpec {
            key: "ust",
            labels: &[
                "ust",
                "ust.",
                "mwst",
                "mwst.",
                "umsatzsteuer",
                "mehrwertsteuer",
            ],
            kind: FieldKind::Amount,
        },
        FieldSpec {
            key: "brutto",
            labels: &[
                "brutto",
                "bruttobetrag",
                "gesamt",
                "gesamtbetrag",
                "summe",
                "endbetrag",
                "total",
            ],
            kind: FieldKind::Amount,
        },
        FieldSpec {
            key: "rechnungsnummer",
            labels: &[
                "rechnungsnr",
                "rechnungsnr.",
                "rechnungsnummer",
                "rechnung-nr",
                "re-nr",
                "re-nr.",
            ],
            kind: FieldKind::Id,
        },
        FieldSpec {
            key: "steuernummer",
            labels: &[
                "steuernr",
                "steuernr.",
                "steuernummer",
                "st-nr",
                "st-nr.",
                "ust-idnr",
                "ust-idnr.",
                "ust-id",
            ],
            kind: FieldKind::Id,
        },
        FieldSpec {
            key: "iban",
            labels: &["iban"],
            kind: FieldKind::Id,
        },
    ]
}

/// One harvested field: which spec matched, the label word as printed, the
/// value (hardened form where hardening fired), parsed cents for amounts,
/// the value word's bbox + confidence, and the validation checks that passed.
#[derive(Clone, Debug, PartialEq)]
pub struct HarvestedField {
    /// The matching [`FieldSpec::key`].
    pub key: String,
    /// The label word as printed on the page (`"Netto:"`).
    pub label_text: String,
    /// The harvested value (numeric-hardened form when it fired).
    pub value: String,
    /// Parsed cents for `Amount` fields (`None` for `Id` fields).
    pub value_cents: Option<i64>,
    /// Top-down image bbox of the value word(s).
    pub bbox: (i32, i32, i32, i32),
    /// Confidence of the value word(s) — the MINIMUM over joined words.
    pub conf: f32,
    /// Names of the checks that passed (`"iban_mod97_ok"`,
    /// `"arithmetic_ok"`). Empty = harvested but unverified.
    pub checks: Vec<String>,
}

/// Vertical overlap test: do two boxes share at least half of the shorter
/// box's height? (Same-line test for label→value pairing.)
fn same_band(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> bool {
    let overlap = a.3.min(b.3) - a.1.max(b.1);
    let min_h = (a.3 - a.1).min(b.3 - b.1);
    min_h > 0 && overlap * 2 >= min_h
}

/// Normalize a printed label word for matching: lowercase, trailing `:`/`.`
/// stripped (`"Netto:"` → `"netto"`).
fn normalize_label(word: &str) -> String {
    word.trim_end_matches([':', '.']).to_ascii_lowercase()
}

/// The value form of a word: the hardened text when hardening fired, the raw
/// text otherwise.
fn value_text(w: &DocWord) -> &str {
    w.numeric_norm.as_deref().unwrap_or(&w.text)
}

/// Harvest typed fields from a page by label proximity:
///
/// 1. Find label words matching a [`FieldSpec`] (lowercased, `:`-stripped).
/// 2. Take the nearest suitable word to the RIGHT on the same line band
///    (`Amount`: must parse as an amount; `Id`: the next word). For `Id`
///    fields the value continues over following alphanumeric words (an IBAN
///    printed as `DE89 3704 …` groups) up to 34 chars.
/// 3. Amounts get [`parse_amount_cents`]; IBAN-shaped ids get
///    [`iban_mod97_ok`] → check `"iban_mod97_ok"`.
/// 4. If `netto`, `ust` and `brutto` were all harvested and
///    `netto + ust == brutto` (exact, in cents), all three get
///    `"arithmetic_ok"` — the cross-check that disambiguates which number is
///    which better than any single read.
///
/// First match per spec wins (top-to-bottom, left-to-right page order).
/// Run [`harden_numeric_tokens`] first so amounts see hardened text.
#[must_use]
pub fn harvest_fields(page: &DocPage, specs: &[FieldSpec]) -> Vec<HarvestedField> {
    let mut out: Vec<HarvestedField> = Vec::new();

    for spec in specs {
        if out.iter().any(|f| f.key == spec.key) {
            continue;
        }
        'search: for line in &page.lines {
            for (wi, word) in line.words.iter().enumerate() {
                let norm = normalize_label(&word.text);
                if !spec.labels.contains(&norm.as_str()) {
                    continue;
                }
                // Candidates: words to the right of the label, same band,
                // nearest first (reading order within the line suffices).
                let mut candidates = line.words[wi + 1..]
                    .iter()
                    .filter(|c| c.bbox.0 >= word.bbox.2 && same_band(word.bbox, c.bbox));
                match spec.kind {
                    FieldKind::Amount => {
                        for cand in candidates {
                            let text = value_text(cand);
                            if let Some(cents) = parse_amount_cents(text) {
                                out.push(HarvestedField {
                                    key: spec.key.to_string(),
                                    label_text: word.text.clone(),
                                    value: text.to_string(),
                                    value_cents: Some(cents),
                                    bbox: cand.bbox,
                                    conf: cand.conf,
                                    checks: Vec::new(),
                                });
                                break 'search;
                            }
                        }
                    }
                    FieldKind::Id => {
                        if let Some(first) = candidates.next() {
                            // Join following alnum words (IBAN groups etc.)
                            // up to the 34-char IBAN ceiling.
                            let mut value = value_text(first).to_string();
                            let mut bbox = first.bbox;
                            let mut conf = first.conf;
                            for extra in candidates {
                                let t = value_text(extra);
                                if value.len() + t.len() > 34
                                    || !t.chars().all(|c| c.is_ascii_alphanumeric())
                                {
                                    break;
                                }
                                value.push_str(t);
                                bbox = (
                                    bbox.0.min(extra.bbox.0),
                                    bbox.1.min(extra.bbox.1),
                                    bbox.2.max(extra.bbox.2),
                                    bbox.3.max(extra.bbox.3),
                                );
                                conf = conf.min(extra.conf);
                            }
                            let mut checks = Vec::new();
                            if iban_mod97_ok(&value) {
                                checks.push("iban_mod97_ok".to_string());
                            }
                            out.push(HarvestedField {
                                key: spec.key.to_string(),
                                label_text: word.text.clone(),
                                value,
                                value_cents: None,
                                bbox,
                                conf,
                                checks,
                            });
                            break 'search;
                        }
                    }
                }
            }
        }
    }

    // Arithmetic cross-check: netto + ust == brutto (exact cents).
    let cents = |key: &str| -> Option<i64> {
        out.iter()
            .find(|f| f.key == key)
            .and_then(|f| f.value_cents)
    };
    if let (Some(n), Some(u), Some(b)) = (cents("netto"), cents("ust"), cents("brutto")) {
        if n + u == b {
            for f in &mut out {
                if matches!(f.key.as_str(), "netto" | "ust" | "brutto") {
                    f.checks.push("arithmetic_ok".to_string());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dw(text: &str, bbox: (i32, i32, i32, i32), conf: f32) -> DocWord {
        DocWord {
            text: text.to_string(),
            bbox,
            conf,
            leading_space: false,
            numeric_norm: None,
        }
    }

    fn dl(bbox: (i32, i32, i32, i32), words: Vec<DocWord>) -> DocLine {
        DocLine { bbox, words }
    }

    // --- numeric hardening -------------------------------------------------

    #[test]
    fn hardening_fixes_single_misreads_in_amounts() {
        assert_eq!(harden_numeric_token("2S0,00").as_deref(), Some("250,00"));
        assert_eq!(harden_numeric_token("1.O50").as_deref(), Some("1.050"));
        assert_eq!(
            harden_numeric_token("l.250,00").as_deref(),
            Some("1.250,00")
        );
        assert_eq!(harden_numeric_token("12Z").as_deref(), Some("122"));
    }

    #[test]
    fn hardening_leaves_words_ids_and_balanced_tokens_alone() {
        // Genuine words: a non-confusable letter short-circuits.
        assert_eq!(harden_numeric_token("Summe"), None);
        assert_eq!(harden_numeric_token("Rechnung"), None);
        // Digits must strictly dominate letters: "B8" is 1:1 -- could be a
        // legitimate code, stays.
        assert_eq!(harden_numeric_token("B8"), None);
        // Pure digits: nothing to change.
        assert_eq!(harden_numeric_token("1250"), None);
        // Structural chars (slash) disqualify: dates/fractions untouched.
        assert_eq!(harden_numeric_token("1/2"), None);
        // GUID shape survives even though hex letters are digit-dominated.
        assert_eq!(
            harden_numeric_token("a1b2c3d4-0000-4111-8222-333344445555"),
            None
        );
        // IBAN shape is guarded (validated by checksum instead).
        assert_eq!(harden_numeric_token("DE89370400440532013000"), None);
    }

    #[test]
    fn harden_numeric_tokens_fills_only_changed_words() {
        let mut page = DocPage {
            width: 100,
            height: 100,
            lines: vec![dl(
                (0, 0, 100, 10),
                vec![
                    dw("Netto:", (0, 0, 30, 10), 95.0),
                    dw("2S0,00", (40, 0, 80, 10), 90.0),
                ],
            )],
        };
        harden_numeric_tokens(&mut page);
        assert_eq!(page.lines[0].words[0].numeric_norm, None);
        assert_eq!(
            page.lines[0].words[1].numeric_norm.as_deref(),
            Some("250,00")
        );
    }

    // --- IBAN --------------------------------------------------------------

    #[test]
    fn iban_mod97_accepts_the_canonical_example_and_rejects_corruption() {
        // The ISO 13616 documentation example IBAN.
        assert!(iban_mod97_ok("DE89370400440532013000"));
        // Case-insensitive.
        assert!(iban_mod97_ok("de89370400440532013000"));
        // One corrupted digit -> checksum fails.
        assert!(!iban_mod97_ok("DE89370400440532013001"));
        // Shape violations are rejected before any math.
        assert!(!iban_mod97_ok("89DE370400440532013000"));
        assert!(!iban_mod97_ok("DE8937"));
        assert!(!iban_mod97_ok(""));
    }

    // --- amount parsing ----------------------------------------------------

    #[test]
    fn parse_amount_cents_handles_german_and_english_conventions() {
        assert_eq!(parse_amount_cents("1.250,00"), Some(125_000));
        assert_eq!(parse_amount_cents("1,250.00"), Some(125_000));
        assert_eq!(parse_amount_cents("1250,00"), Some(125_000));
        assert_eq!(parse_amount_cents("1250.00"), Some(125_000));
        // A trailing 3-digit group is thousands, not decimals.
        assert_eq!(parse_amount_cents("1.250"), Some(125_000));
        assert_eq!(parse_amount_cents("12,345"), Some(1_234_500));
        assert_eq!(parse_amount_cents("0,50"), Some(50));
        assert_eq!(parse_amount_cents("99"), Some(9_900));
        assert_eq!(parse_amount_cents("€ 99,50"), Some(9_950));
        assert_eq!(parse_amount_cents("-12,00"), Some(-1_200));
        assert_eq!(parse_amount_cents("offen"), None);
        assert_eq!(parse_amount_cents(""), None);
        assert_eq!(parse_amount_cents("-"), None);
        assert_eq!(parse_amount_cents(","), None);
    }

    // --- harvest -----------------------------------------------------------

    /// A synthetic German invoice page: amounts (one with an OCR misread),
    /// an invoice number, and an IBAN printed in groups.
    fn invoice_page() -> DocPage {
        let mut page = DocPage {
            width: 600,
            height: 200,
            lines: vec![
                dl(
                    (0, 0, 400, 20),
                    vec![
                        dw("Rechnungsnr.:", (0, 0, 120, 20), 96.0),
                        dw("2024-0815", (130, 0, 220, 20), 97.0),
                    ],
                ),
                dl(
                    (0, 30, 400, 50),
                    vec![
                        dw("Netto:", (0, 30, 60, 50), 95.0),
                        // OCR misread: S for 5 -- hardening fixes it pre-parse.
                        dw("1.2S0,00", (100, 30, 200, 50), 88.0),
                    ],
                ),
                dl(
                    (0, 60, 400, 80),
                    vec![
                        dw("MwSt:", (0, 60, 60, 80), 95.0),
                        dw("237,50", (100, 60, 200, 80), 94.0),
                    ],
                ),
                dl(
                    (0, 90, 400, 110),
                    vec![
                        dw("Brutto:", (0, 90, 60, 110), 95.0),
                        dw("1.487,50", (100, 90, 200, 110), 93.0),
                    ],
                ),
                dl(
                    (0, 120, 600, 140),
                    vec![
                        dw("IBAN:", (0, 120, 50, 140), 96.0),
                        dw("DE89", (60, 120, 100, 140), 92.0),
                        dw("3704", (110, 120, 150, 140), 91.0),
                        dw("0044", (160, 120, 200, 140), 93.0),
                        dw("0532", (210, 120, 250, 140), 92.0),
                        dw("0130", (260, 120, 300, 140), 90.0),
                        dw("00", (310, 120, 330, 140), 94.0),
                    ],
                ),
            ],
        };
        harden_numeric_tokens(&mut page);
        page
    }

    #[test]
    fn harvest_extracts_typed_fields_and_cross_checks_arithmetic() {
        let page = invoice_page();
        let fields = harvest_fields(&page, &german_invoice_fields());

        let get = |key: &str| fields.iter().find(|f| f.key == key).unwrap();

        // The hardened amount parsed: 1.2S0,00 -> 1.250,00 -> 125000 cents.
        let netto = get("netto");
        assert_eq!(netto.value, "1.250,00");
        assert_eq!(netto.value_cents, Some(125_000));

        assert_eq!(get("ust").value_cents, Some(23_750));
        assert_eq!(get("brutto").value_cents, Some(148_750));

        // 125000 + 23750 == 148750 -> all three carry arithmetic_ok.
        for key in ["netto", "ust", "brutto"] {
            assert!(
                get(key).checks.iter().any(|c| c == "arithmetic_ok"),
                "{key} missing arithmetic_ok: {:?}",
                get(key).checks
            );
        }

        // Id field: taken verbatim.
        assert_eq!(get("rechnungsnummer").value, "2024-0815");

        // IBAN groups joined and checksum-verified; conf = min over groups.
        let iban = get("iban");
        assert_eq!(iban.value, "DE89370400440532013000");
        assert!(iban.checks.iter().any(|c| c == "iban_mod97_ok"));
        assert_eq!(iban.conf, 90.0);
        assert_eq!(iban.bbox, (60, 120, 330, 140));
    }

    #[test]
    fn harvest_without_arithmetic_consistency_adds_no_check() {
        let mut page = DocPage {
            width: 400,
            height: 100,
            lines: vec![
                dl(
                    (0, 0, 400, 20),
                    vec![
                        dw("Netto:", (0, 0, 60, 20), 95.0),
                        dw("100,00", (100, 0, 200, 20), 95.0),
                    ],
                ),
                dl(
                    (0, 30, 400, 50),
                    vec![
                        dw("MwSt:", (0, 30, 60, 50), 95.0),
                        dw("19,00", (100, 30, 200, 50), 95.0),
                    ],
                ),
                dl(
                    (0, 60, 400, 80),
                    vec![
                        dw("Brutto:", (0, 60, 60, 80), 95.0),
                        // WRONG total: 100 + 19 != 120.
                        dw("120,00", (100, 60, 200, 80), 95.0),
                    ],
                ),
            ],
        };
        harden_numeric_tokens(&mut page);
        let fields = harvest_fields(&page, &german_invoice_fields());
        assert!(fields
            .iter()
            .all(|f| !f.checks.iter().any(|c| c == "arithmetic_ok")));
    }

    #[test]
    fn harvest_skips_labels_with_no_parseable_value() {
        let page = DocPage {
            width: 400,
            height: 40,
            lines: vec![dl(
                (0, 0, 400, 20),
                vec![
                    dw("Netto:", (0, 0, 60, 20), 95.0),
                    dw("offen", (100, 0, 200, 20), 95.0),
                ],
            )],
        };
        let fields = harvest_fields(&page, &german_invoice_fields());
        assert!(fields.is_empty());
    }

    // --- JSON --------------------------------------------------------------

    #[test]
    fn json_escape_covers_rfc8259_musts_and_passes_unicode() {
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("a\nb\tc\rd"), "a\\nb\\tc\\rd");
        assert_eq!(json_escape("\u{0008}\u{000C}"), "\\b\\f");
        assert_eq!(json_escape("\u{0001}"), "\\u0001");
        assert_eq!(json_escape("äöü€ß"), "äöü€ß");
        assert_eq!(json_escape(""), "");
    }

    #[test]
    fn render_json_golden_one_line_one_field() {
        let page = DocPage {
            width: 10,
            height: 10,
            lines: vec![dl((0, 0, 10, 10), vec![dw("a", (0, 0, 4, 10), 100.0)])],
        };
        let field = HarvestedField {
            key: "netto".to_string(),
            label_text: "Netto:".to_string(),
            value: "1,00".to_string(),
            value_cents: Some(100),
            bbox: (1, 2, 3, 4),
            conf: 99.5,
            checks: vec!["arithmetic_ok".to_string()],
        };
        let json = render_json(&page, &[field]);
        let expected = concat!(
            "{\"schema\":\"tesseract-rs/doc.v1\",\"pages\":[{",
            "\"page\":1,\"width\":10,\"height\":10,",
            "\"quality\":{\"mean_conf\":100.00,\"low_confidence\":false},",
            "\"regions\":[{\"type\":\"paragraph\",\"bbox\":[0,0,10,10],",
            "\"lines\":[{\"bbox\":[0,0,10,10],\"words\":[",
            "{\"text\":\"a\",\"bbox\":[0,0,4,10],\"conf\":100.00,\"leading_space\":false}",
            "]}]}],",
            "\"fields\":[{\"key\":\"netto\",\"label\":\"Netto:\",\"value\":\"1,00\"",
            ",\"value_cents\":100,\"bbox\":[1,2,3,4],\"conf\":99.50,",
            "\"checks\":[\"arithmetic_ok\"]}]}]}",
        );
        assert_eq!(json, expected);
    }

    #[test]
    fn render_json_empty_page_keeps_stable_shape() {
        let page = DocPage {
            width: 5,
            height: 5,
            lines: vec![],
        };
        let json = render_json(&page, &[]);
        assert_eq!(
            json,
            "{\"schema\":\"tesseract-rs/doc.v1\",\"pages\":[{\"page\":1,\
             \"width\":5,\"height\":5,\"quality\":{\"mean_conf\":null,\
             \"low_confidence\":false},\"regions\":[],\"fields\":[]}]}"
        );
    }

    #[test]
    fn low_confidence_flags_garbled_output_but_not_clean_text() {
        // Clean text: high conf → not flagged.
        let clean = DocPage {
            width: 100,
            height: 20,
            lines: vec![dl(
                (0, 0, 100, 20),
                vec![dw("Rechnung", (0, 0, 80, 20), 96.0)],
            )],
        };
        assert_eq!(mean_word_confidence(&clean), Some(96.0));
        assert!(render_json(&clean, &[]).contains("\"low_confidence\":false"));

        // Garble (e.g. handwriting): low conf across words → flagged.
        let garble = DocPage {
            width: 100,
            height: 20,
            lines: vec![dl(
                (0, 0, 100, 20),
                vec![
                    dw("xq", (0, 0, 20, 20), 41.0),
                    dw("z,", (30, 0, 50, 20), 38.0),
                ],
            )],
        };
        let mc = mean_word_confidence(&garble).unwrap();
        assert!(
            mc < LOW_CONFIDENCE_THRESHOLD,
            "garble mean {mc} must be below floor"
        );
        assert!(render_json(&garble, &[]).contains("\"low_confidence\":true"));
    }

    #[test]
    fn render_json_emits_numeric_norm_only_when_present() {
        let mut page = DocPage {
            width: 100,
            height: 20,
            lines: vec![dl(
                (0, 0, 100, 20),
                vec![dw("2S0,00", (0, 0, 60, 20), 88.0)],
            )],
        };
        harden_numeric_tokens(&mut page);
        let json = render_json(&page, &[]);
        assert!(json.contains("\"text\":\"2S0,00\""));
        assert!(json.contains("\"numeric_norm\":\"250,00\""));
    }

    // --- regions -----------------------------------------------------------

    /// Synthetic classified page: header line, two body lines in two blocks,
    /// an orphan body line outside every block, a footer line, one figure.
    #[test]
    fn build_regions_assigns_kinds_blocks_and_orphans_in_order() {
        let page = DocPage {
            width: 400,
            height: 300,
            lines: vec![
                dl((10, 5, 200, 15), vec![dw("Kopf", (10, 5, 60, 15), 95.0)]), // 0 header
                dl((10, 50, 180, 70), vec![dw("links", (10, 50, 80, 70), 95.0)]), // 1 block A
                dl(
                    (210, 50, 380, 70),
                    vec![dw("rechts", (210, 50, 300, 70), 95.0)],
                ), // 2 block B
                dl(
                    (10, 150, 180, 170),
                    vec![dw("verwaist", (10, 150, 100, 170), 95.0)],
                ), // 3 orphan
                dl(
                    (10, 280, 200, 295),
                    vec![dw("Seite", (10, 280, 60, 295), 95.0)],
                ), // 4 footer
            ],
        };
        let blocks = [(0, 40, 200, 100), (200, 40, 400, 100)];
        let figures = [(250, 150, 380, 250)];
        let regions = build_regions(&page, &[0], &[4], &blocks, &figures);

        let kinds: Vec<&str> = regions.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(
            kinds,
            ["header", "text", "text", "text", "figure", "footer"],
            "order: header, blocks, orphan catch-all, figures, footer"
        );
        assert_eq!(regions[0].line_indices, [0]);
        assert_eq!(regions[1].line_indices, [1]);
        assert_eq!(regions[2].line_indices, [2]);
        assert_eq!(regions[3].line_indices, [3], "orphan catch-all");
        assert!(regions[4].line_indices.is_empty(), "figures own no lines");
        assert_eq!(regions[4].bbox, (250, 150, 380, 250));
        assert_eq!(regions[5].line_indices, [4]);
        // Line-bearing region bbox = union of member line bboxes.
        assert_eq!(regions[1].bbox, (10, 50, 180, 70));
    }

    #[test]
    fn build_regions_drops_empty_blocks_and_skips_missing_sections() {
        let page = DocPage {
            width: 100,
            height: 100,
            lines: vec![dl(
                (10, 10, 90, 30),
                vec![dw("nur", (10, 10, 40, 30), 95.0)],
            )],
        };
        // Two blocks, only the first is populated; no furniture, no figures.
        let blocks = [(0, 0, 100, 50), (0, 50, 100, 100)];
        let regions = build_regions(&page, &[], &[], &blocks, &[]);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].kind, RegionKind::Text);
        assert_eq!(regions[0].line_indices, [0]);
    }

    #[test]
    fn render_json_with_regions_emits_typed_regions() {
        let page = DocPage {
            width: 50,
            height: 50,
            lines: vec![dl((0, 0, 50, 10), vec![dw("a", (0, 0, 10, 10), 100.0)])],
        };
        let regions = vec![
            DocRegion {
                kind: RegionKind::Text,
                bbox: (0, 0, 50, 10),
                line_indices: vec![0],
            },
            DocRegion {
                kind: RegionKind::Figure,
                bbox: (5, 20, 45, 45),
                line_indices: vec![],
            },
        ];
        let json = render_json_with_regions(&page, &regions, &[]);
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("{\"type\":\"figure\",\"bbox\":[5,20,45,45],\"lines\":[]}"));
        assert!(!json.contains("\"type\":\"paragraph\""));
        // The plain renderer still emits the byte-stable default.
        let plain = render_json(&page, &[]);
        assert!(plain.contains("\"type\":\"paragraph\""));
    }

    // --- from_line_words ---------------------------------------------------

    #[test]
    fn from_line_words_converts_boxes_and_skips_empty_lines() {
        use crate::renderer::LineWords;
        use tesseract_core::dawg::PermuterType;
        use tesseract_core::WordResult;

        let charset = tesseract_core::CharSet::load_from_str(
            "3\nNULL 0 Common 0\na 3 0 a Left a a\nb 3 0 b Left b b\n",
        )
        .expect("valid unicharset");

        let word = |ids: &[i32], cert: f32, box_: (i32, i32, i32, i32)| WordResult {
            unichar_ids: ids.to_vec(),
            certs: ids.iter().map(|_| cert).collect(),
            ratings: ids.iter().map(|_| 0.0).collect(),
            char_boxes: ids.iter().map(|_| box_).collect(),
            permuter: PermuterType::TopChoicePerm,
            space_certainty: 0.0,
            leading_space: false,
        };

        let lines = vec![
            LineWords {
                words: vec![],
                line_box: (0, 0, 10, 10),
            },
            LineWords {
                // Bottom-up TBOX (0,0,4,10) on a 10-high page -> top-down (0,0,4,10).
                words: vec![word(&[1], -0.2, (0, 0, 4, 10))],
                line_box: (0, 0, 10, 10),
            },
        ];
        let page = DocPage::from_line_words(&lines, &charset, 10, 10);
        assert_eq!(page.lines.len(), 1, "empty line skipped");
        let w = &page.lines[0].words[0];
        assert_eq!(w.text, "a");
        assert_eq!(w.bbox, (0, 0, 4, 10));
        assert_eq!(w.conf, 99.0); // 100 + 5*(-0.2)
        assert_eq!(w.numeric_norm, None);
    }
}
