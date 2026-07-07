//! D5.2 (pragmatic variant) — scanned-page image extraction.
//!
//! ## Rationale
//!
//! A "scanned PDF" is, in the overwhelming majority of real-world documents,
//! **one full-page embedded image XObject per page**: a scanner or
//! phone-camera app photographs/scans a physical page and drops the
//! resulting bitmap straight into a PDF page with no vector content at all.
//! The three filters that carry that bitmap are `DCTDecode` (JPEG),
//! `FlateDecode` (raw/deflated bitmap samples), and `CCITTFaxDecode` (G3/G4
//! fax-style bilevel compression, common for black-and-white document
//! scanners).
//!
//! Extracting and decoding *that one image* is a pure-Rust, dependency-light
//! path that covers the large majority of scanned PDFs without needing a
//! full page rasterizer (no PDF content-stream interpreter for vector
//! drawing operators, no `pdfium`/`mupdf` C++ dependency). This module
//! implements exactly that: find the largest image XObject on a page,
//! decode it to 8-bit greyscale, and hand it to
//! [`crate::OcrPipeline::ocr_grey_page`].
//!
//! Pages that are genuinely vector-only (a scanned-*looking* page that was
//! actually produced by "print to PDF" from a scanned-image-only source
//! *without* re-embedding the bitmap, or a page assembled from many small
//! image tiles/vector paths rather than one full-page scan) are out of
//! scope for this module and surface as `Ok(None)` from
//! [`extract_page_image`] — the real fix for those is a full page
//! rasterizer (content-stream interpretation + compositing), which remains
//! explicitly future work (see `.claude/plans/pdf-to-text-ocr-v1.md` Phase 5,
//! D5.2-full).
//!
//! ## Filter / colour-space coverage (v1)
//!
//! | Filter            | Colour space           | BitsPerComponent | Support |
//! |--------------------|------------------------|-------------------|---------|
//! | `DCTDecode` (JPEG) | grey or RGB (from JPEG)| n/a (JPEG-native) | yes     |
//! | `FlateDecode` / none | `DeviceGray`          | 8                 | yes     |
//! | `FlateDecode` / none | `DeviceGray`          | 1                 | yes (default `Decode` only) |
//! | `FlateDecode` / none | `DeviceRGB`           | 8                 | yes     |
//! | `CCITTFaxDecode`    | any                    | any               | no (typed error, named future leaf) |
//! | any                 | `Indexed`              | any               | no (typed error) |
//! | any (has `/SMask`)  | any                    | any               | no (typed error) |
//! | any                 | non-default `/Decode` array | any         | no (typed error, "exotic Decode array") |
//!
//! Spec citations (ISO 32000-1:2008, "PDF 32000-1:2008"):
//! - §7.4 "Filters" (Table 6) — the standard filter names, including
//!   `DCTDecode`, `CCITTFaxDecode`, `FlateDecode`.
//! - §8.9.5.1 "Color Key Masking" / Table 89 "Additional Entries Specific to
//!   an Image Dictionary" — `ColorSpace`, `BitsPerComponent`, `ImageMask`,
//!   `SMask` keys.
//! - §8.9.5.2 "Decode Arrays" (Table 90) — default `Decode` array per colour
//!   space/bit depth (`DeviceGray` 1bpc default `[0 1]`, `DeviceGray`/
//!   `DeviceRGB` 8bpc default `[0 1]`/`[0 1 0 1 0 1]`) and the general sample
//!   interpretation formula
//!   `value = Dmin + sample · (Dmax − Dmin) / (2^BitsPerComponent − 1)`.
//! - §7.4.3 "Image Data" — row byte-alignment (`each row of image data shall
//!   begin on a byte boundary`) and MSB-first bit packing within a byte,
//!   which [`expand_1bit_gray`] implements.

use lopdf::{Dictionary, Document, Object};
use tesseract_ocr::image_input::rgb_to_luminance;

use crate::PdfError;

/// An 8-bit greyscale image extracted from a PDF page's largest image
/// XObject, row-major, `w`×`h` pixels — the same shape
/// [`crate::OcrPipeline::ocr_grey_page`] expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreyImage {
    /// Row-major 8-bit grey samples, `w * h` bytes.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
}

/// Extract the largest image XObject on `page` (1-based, matching the page
/// numbers [`crate::extract_text_layer`] and [`lopdf::Document::get_pages`]
/// use) as an 8-bit grey buffer.
///
/// Returns `Ok(None)` when the page has no image XObjects at all (or only
/// `ImageMask` stencils, which are not photographic scans). Returns
/// `Err(_)` when an image XObject *is* present but uses an unsupported
/// filter, colour space, bit depth, or `Decode` array — callers should treat
/// that as "this page needs the (unimplemented) full rasterizer", not as
/// "no image on this page".
///
/// # Errors
///
/// [`PdfError::Load`] if `pdf_bytes` is not a parseable PDF;
/// [`PdfError::ImageObject`] if the page's XObject dictionary or the chosen
/// image stream cannot be read; [`PdfError::UnsupportedFilter`],
/// [`PdfError::UnsupportedColorSpace`], [`PdfError::UnsupportedFeature`], or
/// [`PdfError::UnsupportedDecodeArray`] for a recognized-but-out-of-scope
/// image encoding; [`PdfError::TruncatedImageData`] if the decoded/raw
/// sample buffer is shorter than `Width * Height * <components>` demands;
/// [`PdfError::Jpeg`] if a `DCTDecode` stream fails to decode as JPEG.
pub fn extract_page_image(pdf_bytes: &[u8], page: u32) -> Result<Option<GreyImage>, PdfError> {
    let doc = Document::load_mem(pdf_bytes).map_err(PdfError::Load)?;
    let pages = doc.get_pages();
    let Some(&page_id) = pages.get(&page) else {
        return Ok(None);
    };
    // `lopdf::Document::get_page_images` propagates a `DictKey` error (via
    // its internal `?`) when the page has no `/Resources` or no
    // `/Resources/XObject` entry at all — which is simply "this page has no
    // image XObjects" (the normal shape of a vector-only page), not a
    // malformed-document error worth surfacing. Treat any error here the
    // same way: no supported image on this page.
    let Ok(images) = doc.get_page_images(page_id) else {
        return Ok(None);
    };

    let best = images
        .iter()
        .filter(|img| !is_image_mask(img.origin_dict))
        .max_by_key(|img| area(img.width, img.height));
    let Some(image) = best else {
        return Ok(None);
    };

    if image.color_space.as_deref() == Some("Indexed") {
        return Err(PdfError::UnsupportedColorSpace("Indexed".to_string()));
    }
    if image.origin_dict.has(b"SMask") {
        return Err(PdfError::UnsupportedFeature(
            "image has an /SMask soft mask".to_string(),
        ));
    }

    let w = usize::try_from(image.width).map_err(|_| PdfError::InvalidDimensions)?;
    let h = usize::try_from(image.height).map_err(|_| PdfError::InvalidDimensions)?;

    let filters = image.filters.clone().unwrap_or_default();
    match filters.first().map(String::as_str) {
        Some("DCTDecode") => decode_dct(image.content, w, h).map(Some),
        Some("CCITTFaxDecode") => Err(PdfError::UnsupportedFilter(
            "CCITTFaxDecode (future leaf: G3/G4 fax bilevel decode)".to_string(),
        )),
        Some("FlateDecode") | None => decode_flate_or_raw(&doc, image, w, h).map(Some),
        Some(other) => Err(PdfError::UnsupportedFilter(other.to_string())),
    }
}

/// `Width * Height` as `i64` for max-by-area comparison, saturating instead
/// of panicking on the (spec-illegal) case of a negative dimension.
fn area(width: i64, height: i64) -> i64 {
    width.max(0).saturating_mul(height.max(0))
}

/// `/ImageMask true` images are 1-bit stencils painted with the current
/// fill colour (PDF 32000-1:2008 §8.9.6.2) — never the photographic scan
/// this module looks for, so they're excluded from the "largest image"
/// search entirely (an image-mask page falls through to `Ok(None)`, the
/// same as a page with no images at all).
fn is_image_mask(dict: &Dictionary) -> bool {
    dict.get(b"ImageMask")
        .and_then(Object::as_bool)
        .unwrap_or(false)
}

/// Checks that an image's `/Decode` array (if present) matches the PDF
/// default for its colour space/bit depth (§8.9.5.2, Table 90). Absent is
/// equivalent to default. A present-but-different array is scoped out as
/// "exotic" per the D5.2 pragmatic-variant plan.
fn check_default_decode(dict: &Dictionary, default: &[f32]) -> Result<(), PdfError> {
    let Ok(array) = dict.get(b"Decode").and_then(Object::as_array) else {
        return Ok(());
    };
    let values: Result<Vec<f32>, _> = array.iter().map(Object::as_float).collect();
    let Ok(values) = values else {
        return Err(PdfError::UnsupportedDecodeArray(
            "non-numeric /Decode entry".to_string(),
        ));
    };
    let matches_default = values.len() == default.len()
        && values
            .iter()
            .zip(default)
            .all(|(a, b)| (a - b).abs() <= f32::EPSILON);
    if matches_default {
        Ok(())
    } else {
        Err(PdfError::UnsupportedDecodeArray(format!("{values:?}")))
    }
}

/// `FlateDecode` (or a truly filter-less/raw image stream) → grey bytes.
/// Reads the decompressed (or raw) sample bytes, then interprets them per
/// `ColorSpace`/`BitsPerComponent`.
fn decode_flate_or_raw(
    doc: &Document,
    image: &lopdf::xobject::PdfImage<'_>,
    w: usize,
    h: usize,
) -> Result<GreyImage, PdfError> {
    let stream = doc
        .get_object(image.id)
        .and_then(Object::as_stream)
        .map_err(PdfError::ImageObject)?;
    let is_filterless = image.filters.as_ref().is_none_or(|f| f.is_empty());
    let raw = if is_filterless {
        stream.content.clone()
    } else {
        stream
            .decompressed_content()
            .map_err(PdfError::ImageObject)?
    };

    let bpc = image
        .bits_per_component
        .ok_or(PdfError::MissingBitsPerComponent)?;
    let color_space = image
        .color_space
        .as_deref()
        .ok_or_else(|| PdfError::UnsupportedColorSpace("<no ColorSpace entry>".to_string()))?;

    match (color_space, bpc) {
        ("DeviceGray", 8) => {
            check_default_decode(image.origin_dict, &[0.0, 1.0])?;
            let need = w * h;
            require_len(&raw, need)?;
            Ok(GreyImage {
                data: raw[..need].to_vec(),
                w,
                h,
            })
        }
        ("DeviceRGB", 8) => {
            check_default_decode(image.origin_dict, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
            let need = 3 * w * h;
            require_len(&raw, need)?;
            Ok(GreyImage {
                data: rgb_to_luminance(&raw[..need], w, h),
                w,
                h,
            })
        }
        ("DeviceGray", 1) => {
            check_default_decode(image.origin_dict, &[0.0, 1.0])?;
            let stride = w.div_ceil(8);
            let need = stride * h;
            require_len(&raw, need)?;
            Ok(GreyImage {
                data: expand_1bit_gray(&raw, w, h, stride),
                w,
                h,
            })
        }
        (cs, other_bpc) => Err(PdfError::UnsupportedColorSpace(format!(
            "{cs} at {other_bpc} bits/component"
        ))),
    }
}

fn require_len(data: &[u8], need: usize) -> Result<(), PdfError> {
    if data.len() < need {
        Err(PdfError::TruncatedImageData {
            expected: need,
            got: data.len(),
        })
    } else {
        Ok(())
    }
}

/// 1-bit `DeviceGray`, default `Decode` array (`[0 1]`) → 0/255 bytes.
///
/// PDF 32000-1:2008 §7.4.3: "each row of image data shall begin on a byte
/// boundary" and samples are packed MSB-first within each byte. With the
/// default `Decode` array, sample value formula (§8.9.5.2) collapses to
/// `value = sample` for 1 bit/component (`Dmin=0`, `Dmax=1`,
/// `2^1 − 1 = 1`), so a `0` bit is black (byte `0`) and a `1` bit is white
/// (byte `255`).
fn expand_1bit_gray(data: &[u8], w: usize, h: usize, stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let row = &data[y * stride..(y + 1) * stride];
        for (x, px) in out[y * w..(y + 1) * w].iter_mut().enumerate() {
            let byte = row[x / 8];
            let bit = (byte >> (7 - (x % 8))) & 1;
            *px = if bit == 1 { 255 } else { 0 };
        }
    }
    out
}

/// `DCTDecode` (JPEG) → grey bytes. `jpeg_bytes` is the raw embedded JPEG
/// bitstream (a `DCTDecode` stream's content bytes ARE the JPEG file, so no
/// filter decompression step is needed — `lopdf`'s `decompressed_content`
/// deliberately does not implement `DCTDecode`; see its filter-dispatch
/// table, which only lists `FlateDecode`/`LZWDecode`/`ASCII85Decode`).
fn decode_dct(jpeg_bytes: &[u8], w: usize, h: usize) -> Result<GreyImage, PdfError> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::JpegDecoder;

    let mut decoder = JpegDecoder::new(ZCursor::new(jpeg_bytes));
    let pixels = decoder.decode().map_err(PdfError::Jpeg)?;
    let info = decoder
        .info()
        .expect("decoder.info() is Some after a successful decode()");
    let out_cs = decoder
        .output_colorspace()
        .expect("decoder.output_colorspace() is Some after a successful decode()");

    let (jw, jh) = (info.width as usize, info.height as usize);
    if jw != w || jh != h {
        return Err(PdfError::JpegDimensionMismatch {
            pdf: (w, h),
            jpeg: (jw, jh),
        });
    }

    match out_cs {
        ColorSpace::Luma => {
            require_len(&pixels, w * h)?;
            Ok(GreyImage {
                data: pixels[..w * h].to_vec(),
                w,
                h,
            })
        }
        ColorSpace::RGB => {
            require_len(&pixels, 3 * w * h)?;
            Ok(GreyImage {
                data: rgb_to_luminance(&pixels[..3 * w * h], w, h),
                w,
                h,
            })
        }
        other => Err(PdfError::UnsupportedJpegColorspace(format!("{other:?}"))),
    }
}
