//! Decode arbitrary image bytes → grey → recognize → text/JSON + stats.

use std::io::Cursor;
use std::time::Instant;

use image::{ImageReader, Limits};
use tesseract_ocr::{
    german_invoice_fields, mean_word_confidence, DocPage, LOW_CONFIDENCE_THRESHOLD,
};

use crate::state::AppState;

/// Hard ceiling on a single decoded dimension (guards a degenerate aspect that
/// slips under the pixel budget, e.g. `1 × 400_000_000`).
const MAX_DIM: u32 = 20_000;
/// Pixel budget (width × height). Bounds the grey buffer + all downstream OCR
/// allocation. 40 MP comfortably covers a 300 dpi A3 scan while a hostile
/// "22000×22000" bomb is rejected before it can allocate ~500 MB.
const MAX_PIXELS: u64 = 40_000_000;
/// Cap the decoder's own intermediate allocation (a compressed bomb can inflate
/// far past its byte size). Applies during `decode()`, before our pixel check.
const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
/// Smallest dimension the recognizer's proven line path accepts (`E-OCR-FROMPIX-1`
/// floor is 3 px); anything narrower cannot hold a glyph.
const MIN_DIM: usize = 3;

/// The output format the client asked for, from the upload form's `format`
/// multipart field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain recognized text (the default).
    Text,
    /// The structured `tesseract-rs/doc.v1` JSON DOM plus a German-invoice
    /// field harvest (see `tesseract_ocr::structured`).
    Json,
}

impl OutputFormat {
    /// Parse the multipart `format` field value. `"json"` selects
    /// [`OutputFormat::Json`]; anything else — including an absent field, an
    /// empty string, or an unrecognized value — falls back to
    /// [`OutputFormat::Text`]. Never errors: an unknown format is not a user
    /// mistake worth rejecting the upload over.
    #[must_use]
    pub fn from_field(value: Option<&str>) -> Self {
        match value {
            Some("json") => Self::Json,
            _ => Self::Text,
        }
    }
}

/// The result of OCR-ing one uploaded/fetched image in text mode.
pub struct OcrOutcome {
    /// The recognized text (lines joined with `\n`).
    pub text: String,
    /// Decoded image width in pixels.
    pub width: usize,
    /// Decoded image height in pixels.
    pub height: usize,
    /// Number of characters in the recognized text.
    pub char_count: usize,
    /// Number of non-empty recognized lines.
    pub line_count: usize,
    /// Mean word confidence 0–100 (`-1` sentinel when no words).
    pub mean_conf: f32,
    /// `true` when the recognizer is not confident — likely handwriting /
    /// low-resolution / non-printed input (`eng.lstm` is print-trained).
    pub low_confidence: bool,
    /// Wall-clock recognition time in milliseconds (decode excluded).
    pub elapsed_ms: f64,
    /// `true` when [`tesseract_ocr::rectify::auto_rectify`] actually changed
    /// the page (it was requested AND the fitted ramp cleared
    /// [`tesseract_ocr::rectify::ShearRamp::is_significant`]) — mirrors
    /// [`OcrDebugOutcome::rectified`]'s honesty signal, since `auto_rectify`
    /// is a safe no-op on an already-straight page.
    pub rectified: bool,
}

/// The result of OCR-ing one uploaded/fetched image in JSON mode: the
/// rendered `tesseract-rs/doc.v1` document (structure + harvested fields) plus
/// the same stats shape as [`OcrOutcome`], but word/line counts instead of a
/// (meaningless, for JSON) character count.
pub struct OcrJsonOutcome {
    /// The rendered `doc.v1` JSON document.
    pub json: String,
    /// Decoded image width in pixels.
    pub width: usize,
    /// Decoded image height in pixels.
    pub height: usize,
    /// Total recognized words across all lines.
    pub word_count: usize,
    /// Number of non-empty recognized lines.
    pub line_count: usize,
    /// Mean word confidence 0–100 (`-1` sentinel when no words), rounded.
    pub mean_conf: f32,
    /// `true` when the recognizer is not confident — the image is likely
    /// handwriting / low-resolution / not printed text (`eng.lstm` is a
    /// print-trained model).
    pub low_confidence: bool,
    /// Wall-clock recognition time in milliseconds (decode excluded).
    pub elapsed_ms: f64,
    /// `true` when [`tesseract_ocr::rectify::auto_rectify`] actually changed
    /// the page — mirrors [`OcrDebugOutcome::rectified`]'s honesty signal.
    /// Consumed by the HTML `/ocr?format=json` result page; the machine API
    /// (`POST /api/v1/recognize`) returns ONLY [`Self::json`] as its response
    /// body, so this field does not (yet) reach API callers on the wire —
    /// surfacing it there would mean either changing the `doc.v1` schema (a
    /// cross-crate change, out of scope here) or adding a response header,
    /// neither of which this pass does.
    pub rectified: bool,
}

/// Decode `bytes` (PNG / JPEG / WebP / TIFF / GIF / BMP / PNM — via the `image`
/// crate, all pure-Rust decoders) to 8-bit grey, bounded against
/// decompression / pixel bombs: the decoder is capped at [`MAX_DECODE_ALLOC`]
/// and [`MAX_DIM`], and the decoded pixel count is rejected above
/// [`MAX_PIXELS`] before the grey buffer (and the larger OCR working set) is
/// ever materialized. Shared by both [`ocr_image_bytes`] and
/// [`ocr_image_bytes_json`] so the two output modes decode identically.
/// Shared opt-in preprocessing chain for every entry point below: DESKEW
/// (rotation) runs BEFORE RECTIFY (keystone), never the reverse.
///
/// The order is not a convention, it is load-bearing: a purely-rotated page
/// that skipped deskew would leak its rotational component into
/// [`tesseract_ocr::rectify::detect_row_shears`]'s per-row line fit, and
/// `auto_rectify`'s single-shear model would then apply a spurious
/// correction trying to explain a distortion that is not keystone at all.
/// Running deskew first is what makes
/// `deskew::tests::deskew_then_rectify_measures_near_zero_shear_on_a_purely_rotated_page`
/// true: after deskew, a purely-rotated page's own residual shear sits below
/// `ShearRamp::is_significant`'s gate, so `auto_rectify` correctly no-ops.
///
/// Both passes are independently a documented no-op when nothing significant
/// is detected (see their own doc comments), so `deskewed`/`rectified` report
/// what ACTUALLY changed — never just an echo of the request flags.
fn preprocess(
    decoded: &[u8],
    w: usize,
    decoded_h: usize,
    deskew: bool,
    rectify: bool,
) -> (Vec<u8>, usize, bool, bool) {
    let (after_deskew, dh, deskewed) = if deskew {
        let (out, out_h) = tesseract_ocr::deskew::auto_deskew(decoded, w, decoded_h);
        let changed = out_h != decoded_h || out != decoded;
        (out, out_h, changed)
    } else {
        (decoded.to_vec(), decoded_h, false)
    };
    let (raw, h, rectified) = if rectify {
        let (out, out_h) = tesseract_ocr::rectify::auto_rectify(&after_deskew, w, dh);
        let changed = out_h != dh || out != after_deskew;
        (out, out_h, changed)
    } else {
        (after_deskew, dh, false)
    };
    (raw, h, deskewed, rectified)
}

fn decode_grey(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    // Sniff the format from the bytes, then decode under explicit limits — the
    // `image` defaults set only a 512 MiB alloc cap and NO dimension cap, so a
    // tiny compressed file can still decode to a gigapixel raster.
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("could not read image: {e}"))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DIM);
    limits.max_image_height = Some(MAX_DIM);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    reader.limits(limits);

    let dynimg = reader.decode().map_err(|e| {
        format!("could not decode image (PNG, JPEG, WebP, TIFF, GIF, BMP, PNM supported): {e}")
    })?;

    let (w, h) = (dynimg.width() as usize, dynimg.height() as usize);
    if w == 0 || h == 0 {
        return Err("decoded image has zero size".to_string());
    }
    if w < MIN_DIM || h < MIN_DIM {
        return Err(format!("image is too small to contain text ({w}x{h})"));
    }
    // Reject an oversized pixel count BEFORE `to_luma8` allocates a second
    // full-resolution buffer and before the recognizer's larger working set.
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(format!(
            "image too large: {w}x{h} exceeds the {} megapixel limit",
            MAX_PIXELS / 1_000_000
        ));
    }

    let grey = dynimg.to_luma8();
    drop(dynimg); // free the decoded source before the recognizer's working set
    Ok((grey.into_raw(), w, h)) // Vec<u8>, row-major, len == w*h
}

/// Decode `bytes` to grey and run the page recognition via the WORD path
/// ([`LstmRecognizer::recognize_page_makerow_words`]), rendering the plain
/// text with [`render_text`]. This is byte-identical to the old string surface
/// (`render_text(words).trim_end() == recognize_page_makerow`, a proven
/// property) but ALSO yields per-word confidence, so text mode can report the
/// same honesty signal as JSON mode — a low mean confidence flags likely
/// handwriting / low-resolution / non-printed input. Returns text + stats, or
/// a user-safe error string.
///
/// `lang` selects the model via [`AppState::model`] (`Some("deu")` → German,
/// anything else → English, the pre-existing default). `rectify`, when
/// `true`, runs [`tesseract_ocr::rectify::auto_rectify`] on the decoded grey
/// page BEFORE recognition — the same opt-in preprocessing pass
/// [`ocr_image_bytes_debug`] already exposes; see that function's doc comment
/// for what it corrects and why it is always a safe no-op.
///
/// This is heavy synchronous CPU work — callers MUST run it off the async
/// runtime (via `spawn_blocking`); see [`crate::routes`].
pub fn ocr_image_bytes(
    state: &AppState,
    bytes: &[u8],
    lang: Option<&str>,
    deskew: bool,
    rectify: bool,
) -> Result<OcrOutcome, String> {
    let (decoded, w, decoded_h) = decode_grey(bytes)?;
    let (raw, h, _deskewed, rectified) = preprocess(&decoded, w, decoded_h, deskew, rectify);
    let (_lang, model) = state.model(lang);

    let t0 = Instant::now();
    // Block-aware surface: multi-column pages read column-by-column in
    // XY-cut order; a single-column page takes the identical whole-page
    // path (see recognize_page_blocks_words' docs).
    let lines = model
        .recognizer
        .recognize_page_blocks_words(&raw, w, h, model.dict.as_ref())
        .map_err(|e| format!("recognition failed: {e}"))?;
    // Paragraph-gap detection is strictly additive (only ever inserts a blank
    // line at a raw-pixel-measured structural break; never removes or
    // reorders content) and default-on, not gated behind a checkbox like
    // deskew/rectify: those alter what gets RECOGNIZED, this is pure
    // post-processing of an already-final line list. See
    // `renderer::detect_paragraph_gaps`'s doc comment for the guard that
    // keeps it silent on un-deskewed rotated pages.
    let binary =
        tesseract_ocr::xy_cut::binarize_page_with(&raw, w, h, tesseract_ocr::BinarizeMode::Otsu);
    let para_gaps = tesseract_ocr::renderer::detect_paragraph_gaps(&binary, w, h, &lines);
    let text = tesseract_ocr::renderer::render_text_with_gaps(
        &lines,
        &model.recognizer.charset,
        &para_gaps,
    )
    .trim_end()
    .to_string();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let page = DocPage::from_line_words(&lines, &model.recognizer.charset, w as u32, h as u32);
    let mean = mean_word_confidence(&page);
    let char_count = text.chars().count();
    let line_count = text.lines().filter(|l| !l.trim().is_empty()).count();
    Ok(OcrOutcome {
        text,
        width: w,
        height: h,
        char_count,
        line_count,
        mean_conf: mean.unwrap_or(-1.0),
        low_confidence: mean.is_some_and(|mc| mc < LOW_CONFIDENCE_THRESHOLD),
        elapsed_ms,
        rectified,
    })
}

/// Decode `bytes` to grey and run the canonical one-shot structured-document
/// path ([`LstmRecognizer::recognize_document`]): word/box recognition →
/// `doc.v1` DOM → numeric hardening → German-invoice field harvest → region
/// classification (page furniture + XY-cut blocks + halftone figures) →
/// `doc.v1` JSON. The composition itself lives in `tesseract-ocr` so this
/// demo and the `tesseract-ogar` executor share ONE source of truth. Same
/// off-runtime contract as [`ocr_image_bytes`] — heavy synchronous CPU work,
/// callers MUST run it via `spawn_blocking`.
///
/// `lang` selects the model via [`AppState::model`] (`Some("deu")` → German,
/// anything else → English, the pre-existing default). The German-invoice
/// field harvest itself runs regardless of `lang` — its label keys/regexes
/// are a fixed spec, not a language switch. `rectify`, when `true`, runs
/// [`tesseract_ocr::rectify::auto_rectify`] on the decoded grey page BEFORE
/// recognition — the same opt-in preprocessing pass [`ocr_image_bytes_debug`]
/// already exposes. This function is also the machine API's own entry point
/// for it: `crate::api`'s `recognize` handler for `POST /api/v1/recognize`
/// calls straight through to it. See [`ocr_image_bytes_debug`]'s doc comment
/// for what the pass corrects and why it is always a safe no-op.
pub fn ocr_image_bytes_json(
    state: &AppState,
    bytes: &[u8],
    lang: Option<&str>,
    deskew: bool,
    rectify: bool,
) -> Result<OcrJsonOutcome, String> {
    let (decoded, w, decoded_h) = decode_grey(bytes)?;
    let (raw, h, _deskewed, rectified) = preprocess(&decoded, w, decoded_h, deskew, rectify);
    let (_lang, model) = state.model(lang);

    let t0 = Instant::now();
    let specs = german_invoice_fields();
    let doc = model
        .recognizer
        .recognize_document(&raw, w, h, model.dict.as_ref(), Some(&specs))
        .map_err(|e| format!("recognition failed: {e}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(OcrJsonOutcome {
        json: doc.json,
        width: w,
        height: h,
        word_count: doc.word_count,
        line_count: doc.line_count,
        mean_conf: doc.mean_confidence.unwrap_or(-1.0),
        low_confidence: doc.low_confidence,
        elapsed_ms,
        rectified,
    })
}

/// Everything the `/debug` and `/pdf` routes need from ONE recognition pass:
/// the `doc.v1` JSON (the canonical structured output), the decoded grey raster
/// (needed to build the searchable-PDF "A" background and to crop `doc.v1`
/// figure regions for "B"), the page dimensions, and the same honesty stats the
/// other modes report. A and B are BOTH derived from this single pass, so the
/// two panels cannot drift.
pub struct OcrDebugOutcome {
    /// Decoded image width in pixels.
    pub width: usize,
    /// Decoded image height in pixels.
    pub height: usize,
    /// The decoded 8-bit grey raster, row-major, `width * height` bytes.
    pub grey: Vec<u8>,
    /// The rendered `tesseract-rs/doc.v1` JSON document.
    pub doc_json: String,
    /// Total recognized words across all lines.
    pub word_count: usize,
    /// Number of non-empty recognized lines.
    pub line_count: usize,
    /// Mean word confidence 0–100 (`-1` sentinel when no words).
    pub mean_conf: f32,
    /// `true` when the recognizer is not confident (likely handwriting /
    /// low-resolution / non-printed input).
    pub low_confidence: bool,
    /// Wall-clock recognition time in milliseconds (decode excluded).
    pub elapsed_ms: f64,
    /// `true` when the production dict beam was used (DAWGs loaded).
    pub dict_on: bool,
    /// The model actually selected — `"eng"` or `"deu"` — from
    /// [`AppState::model`]. Reflects reality even when the caller asked for a
    /// language that isn't loaded (falls back to `"eng"`), so callers report
    /// the truth rather than echoing the request.
    pub lang: &'static str,
    /// The selected model's network spec string (e.g.
    /// `[1,36,0,1Ct3,3,16Mp3,3Lfys48Lfx96Lrx96Lfx192O1c1]`), for the debug
    /// stats panel. Carried here (rather than re-selecting the model from
    /// `state` again) so the caller can't accidentally report a DIFFERENT
    /// model's spec than the one that actually ran.
    pub network_spec: String,
    /// The selected model's null/blank CTC label id, for the debug stats panel.
    pub null_char: i32,
    /// `true` when [`tesseract_ocr::rectify::auto_rectify`] actually changed
    /// the page (it was requested AND the fitted ramp cleared
    /// [`tesseract_ocr::rectify::ShearRamp::is_significant`]) — the debug
    /// panel's honest "did rectification do anything" signal, since
    /// `auto_rectify` is a safe no-op on an already-straight page.
    pub rectified: bool,
    /// `true` when [`tesseract_ocr::deskew::auto_deskew`] actually changed
    /// the page (it was requested AND [`tesseract_ocr::deskew::deskew_general`]'s
    /// own confidence/angle gate cleared) — same honesty contract as
    /// [`Self::rectified`], reported separately since the two passes correct
    /// different distortions (rotation vs keystone) and can each fire
    /// independently.
    pub deskewed: bool,
}

/// Decode `bytes` to grey and run the canonical one-shot structured-document
/// path ([`LstmRecognizer::recognize_document`]) with the German-invoice field
/// harvest — the SAME composition [`ocr_image_bytes_json`] uses — but ALSO
/// return the grey raster so the caller can build the searchable-PDF "A"
/// facsimile (scan + invisible word layer) and the `doc.v1` reconstruction "B"
/// from one recognition. Same off-runtime contract as the other entry points —
/// heavy synchronous CPU work, callers MUST run it via `spawn_blocking`.
///
/// `lang` selects the model via [`AppState::model`] (`Some("deu")` → German,
/// anything else → English, the pre-existing default). `rectify`, when
/// `true`, runs [`tesseract_ocr::rectify::auto_rectify`] on the decoded grey
/// page BEFORE recognition — a NEW, non-Tesseract preprocessing pass that
/// corrects rotational skew and (a first-order approximation of) keystone/
/// trapezoid distortion from a photographed page; see that module's docs.
/// Opt-in (default `false` at every call site today) — same "available, not
/// yet the default" positioning `binarize::sauvola_binarize` already has in
/// this crate.
pub fn ocr_image_bytes_debug(
    state: &AppState,
    bytes: &[u8],
    lang: Option<&str>,
    deskew: bool,
    rectify: bool,
) -> Result<OcrDebugOutcome, String> {
    let (decoded, w, decoded_h) = decode_grey(bytes)?;
    let (raw, h, deskewed, rectified) = preprocess(&decoded, w, decoded_h, deskew, rectify);
    let (lang, model) = state.model(lang);

    let t0 = Instant::now();
    let specs = german_invoice_fields();
    let doc = model
        .recognizer
        .recognize_document(&raw, w, h, model.dict.as_ref(), Some(&specs))
        .map_err(|e| format!("recognition failed: {e}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(OcrDebugOutcome {
        width: w,
        height: h,
        grey: raw,
        doc_json: doc.json,
        word_count: doc.word_count,
        line_count: doc.line_count,
        mean_conf: doc.mean_confidence.unwrap_or(-1.0),
        low_confidence: doc.low_confidence,
        elapsed_ms,
        dict_on: model.dict.is_some(),
        lang,
        network_spec: model.recognizer.network_str.clone(),
        null_char: model.recognizer.null_char,
        rectified,
        deskewed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_from_field_defaults_to_text() {
        assert_eq!(OutputFormat::from_field(None), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("")), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("text")), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("bogus")), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("JSON")), OutputFormat::Text); // case-sensitive
    }

    #[test]
    fn output_format_from_field_recognizes_json() {
        assert_eq!(OutputFormat::from_field(Some("json")), OutputFormat::Json);
    }

    // -----------------------------------------------------------------
    // preprocess() — D8's actual wiring, tested directly against a
    // synthetic ROTATED page. No model load needed: this exercises only
    // the deskew/rectify composition, not recognition.
    // -----------------------------------------------------------------

    /// The same hollow-rectangle-blob text-like fixture
    /// `tesseract_ocr::deskew`'s own D8 falsifier uses (`detect_row_shears`
    /// needs connected-component blob geometry, not real glyphs).
    fn straight_text_like_page(w: usize, h: usize, n_lines: usize) -> Vec<u8> {
        let mut page = vec![255u8; w * h];
        let margin = h / (n_lines + 2);
        let bar_h = 16usize.min(margin.saturating_sub(4).max(9));
        let seg_w = w / 12;
        for i in 0..n_lines {
            let y0 = margin + i * margin;
            for seg in 0..8 {
                let x0 = seg * (w / 8) + seg_w / 4;
                let x1 = (x0 + seg_w).min(w);
                let border = 2usize;
                for y in y0..(y0 + bar_h).min(h) {
                    for x in x0..x1.min(w) {
                        let on_border = y < y0 + border
                            || y + border >= (y0 + bar_h).min(h)
                            || x < x0 + border
                            || x + border >= x1;
                        if on_border {
                            page[y * w + x] = 0;
                        }
                    }
                }
            }
        }
        page
    }

    #[test]
    fn preprocess_deskew_runs_before_rectify_and_actually_straightens_a_rotated_page() {
        let (w, h) = (300usize, 220usize);
        let straight = straight_text_like_page(w, h, 5);
        let rotated =
            tesseract_ocr::deskew::rotate_am_gray(&straight, w, h, 4.0_f32.to_radians(), 255);

        // Sanity: the fixture must carry a measurable shear before ANY
        // preprocessing, or this test measures nothing.
        let before = tesseract_ocr::rectify::fit_shear_ramp(
            &tesseract_ocr::rectify::detect_row_shears(&rotated, w, h),
        );
        assert!(
            before.is_some_and(|r| r.is_significant(h)),
            "fixture must carry a significant shear before preprocessing"
        );

        // deskew=true, rectify=true — the wired production path.
        let (out, out_h, deskewed, rectified) = preprocess(&rotated, w, h, true, true);
        assert!(
            deskewed,
            "a 4-degree rotation must clear auto_deskew's gate"
        );

        // THE ORDER-DISCRIMINATING CLAIM (measured, not assumed — an earlier
        // version of this test checked only "is there residual shear left",
        // which stayed GREEN even with the two passes swapped: at this small
        // angle a pure rotation approximates a shear well enough that
        // auto_rectify ALONE can also straighten it, so "ended up straight"
        // does not distinguish the orders). What DOES distinguish them:
        // `rectified` itself. Measured on the swapped (rectify-first) order,
        // `rectified` comes back `true` — rectify fires for real on the raw
        // rotation, then deskew has nothing left to do. With the correct
        // order, deskew consumes the rotation FIRST, so rectify's own
        // shear-fit finds nothing significant and must report a genuine
        // no-op: `rectified == false`.
        assert!(
            !rectified,
            "with deskew running FIRST, rectify must be a documented no-op \
             on a purely-rotated page (rectified=true means deskew did NOT \
             run before rectify's scan)"
        );

        // Secondary confirmation: no significant shear left in the OUTPUT
        // either way (this alone is not order-discriminating, per the note
        // above, but it is still a real property of the composed result).
        let after = tesseract_ocr::rectify::fit_shear_ramp(
            &tesseract_ocr::rectify::detect_row_shears(&out, w, out_h),
        );
        assert!(
            after.is_none_or(|r| !r.is_significant(out_h)),
            "the composed output must not carry a significant residual shear"
        );
    }

    #[test]
    fn preprocess_with_both_flags_off_is_a_byte_identical_no_op() {
        let (w, h) = (40usize, 30usize);
        let page: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let (out, out_h, deskewed, rectified) = preprocess(&page, w, h, false, false);
        assert_eq!(out, page, "no-op preprocessing must not touch the buffer");
        assert_eq!(out_h, h);
        assert!(!deskewed);
        assert!(!rectified);
    }
}
