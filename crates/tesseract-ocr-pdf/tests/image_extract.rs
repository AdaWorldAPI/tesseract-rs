//! D5.2 (pragmatic variant) tests for
//! [`tesseract_ocr_pdf::extract_page_image`].
//!
//! Test PDFs are built in-test with `lopdf`, mirroring `tests/text_layer.rs`
//! (a minimal Catalog → Pages → Page → Resources/XObject tree), so these
//! tests have no external file dependency except the one pre-made JPEG
//! fixture used for the `DCTDecode` case (see its doc comment below for
//! provenance).

use std::io::Write as _;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use lopdf::{dictionary, Document, Object, Stream};
use tesseract_ocr_pdf::{extract_page_image, PdfError};

const W: usize = 24;
const H: usize = 36;

/// The session-standard deterministic test grid:
/// `((x*37 + y*11) ^ (x*y)) % 256`, row-major, `W`x`H`.
fn test_grid() -> Vec<u8> {
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let v = ((x * 37 + y * 11) ^ (x * y)) % 256;
            out.push(v as u8);
        }
    }
    out
}

/// Minimal 1-page PDF whose only content is a full-page image XObject named
/// `Im0`. `image_dict` supplies the XObject's `Width`/`Height`/`ColorSpace`/
/// `BitsPerComponent`/`Filter`/etc keys; `content` is the (already
/// filter-encoded, e.g. zlib-deflated or raw JPEG) stream bytes. No `Do`
/// content-stream operator is emitted — `extract_page_image` only walks the
/// Resources/XObject dictionary, it does not interpret content streams.
fn build_image_pdf(image_dict: lopdf::Dictionary, content: Vec<u8>) -> Vec<u8> {
    let mut doc = Document::new();
    let pages_id = doc.new_object_id();

    let image_id = doc.add_object(Stream::new(image_dict, content));
    let resources_id = doc.add_object(dictionary! {
        "XObject" => dictionary! {
            "Im0" => image_id,
        },
    });

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), (W as i64).into(), (H as i64).into()],
    });

    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).expect("save in-memory PDF");
    bytes
}

fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).expect("zlib write");
    encoder.finish().expect("zlib finish")
}

#[test]
fn flate_decode_device_gray_8bit_round_trips_exactly() {
    let grid = test_grid();
    let compressed = zlib_compress(&grid);

    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
    };
    let bytes = build_image_pdf(image_dict, compressed);

    let image = extract_page_image(&bytes, 1)
        .expect("extract_page_image")
        .expect("page 1 has an image XObject");

    assert_eq!(image.w, W);
    assert_eq!(image.h, H);
    assert_eq!(
        image.data, grid,
        "FlateDecode/DeviceGray/8bpc must round-trip byte-for-byte (lossless path)"
    );
}

#[test]
fn dct_decode_jpeg_decodes_within_lossy_tolerance() {
    // `tests/fixtures/grid_24x36.jpg`: the same 24x36 test grid as
    // `test_grid()`, JPEG-encoded at quality 90. Provenance: generated
    // once via `python3 -c "from PIL import Image; ..."` (Pillow), NOT
    // hand-written — zune-jpeg is decode-only so an external encoder was
    // needed to produce a real DCTDecode fixture. See the crate-level
    // `image_extract` module doc for why DCTDecode is decode-only in this
    // crate (JPEG *encoding* is not part of the OCR input path).
    let jpeg_bytes = include_bytes!("fixtures/grid_24x36.jpg");

    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "DCTDecode",
    };
    let bytes = build_image_pdf(image_dict, jpeg_bytes.to_vec());

    let image = extract_page_image(&bytes, 1)
        .expect("extract_page_image")
        .expect("page 1 has an image XObject");

    assert_eq!(image.w, W);
    assert_eq!(image.h, H);

    let grid = test_grid();
    assert_eq!(image.data.len(), grid.len());
    let max_delta = image
        .data
        .iter()
        .zip(&grid)
        .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_delta <= 20,
        "JPEG (quality 90) per-pixel delta should be small, got max_delta={max_delta}"
    );
}

#[test]
fn no_image_xobject_yields_none() {
    // A page with Resources but no XObject dict at all (an empty
    // Resources, mirroring `tests/text_layer.rs`'s pattern for the
    // "nothing here" case).
    let mut doc = Document::new();
    let pages_id = doc.new_object_id();
    let resources_id = doc.add_object(dictionary! {});
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).expect("save in-memory PDF");

    let result = extract_page_image(&bytes, 1).expect("extract_page_image");
    assert!(result.is_none());
}

#[test]
fn ccitt_fax_decode_is_a_typed_unsupported_filter_error() {
    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 1,
        "Filter" => "CCITTFaxDecode",
    };
    // Content doesn't need to be valid G4 data; the filter name alone
    // routes to the typed error before any decode is attempted.
    let bytes = build_image_pdf(image_dict, vec![0u8; 16]);

    let err = extract_page_image(&bytes, 1).unwrap_err();
    assert!(matches!(err, PdfError::UnsupportedFilter(f) if f.contains("CCITTFaxDecode")));
}

#[test]
fn indexed_color_space_is_a_typed_unsupported_error() {
    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "Indexed",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
    };
    let bytes = build_image_pdf(image_dict, zlib_compress(&vec![0u8; W * H]));

    let err = extract_page_image(&bytes, 1).unwrap_err();
    assert!(matches!(err, PdfError::UnsupportedColorSpace(cs) if cs == "Indexed"));
}

#[test]
fn smask_soft_mask_is_a_typed_unsupported_feature_error() {
    let mut doc = Document::new();
    let pages_id = doc.new_object_id();

    let smask_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
    };
    let smask_id = doc.add_object(Stream::new(smask_dict, zlib_compress(&vec![255u8; W * H])));

    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
        "SMask" => smask_id,
    };
    let image_id = doc.add_object(Stream::new(image_dict, zlib_compress(&test_grid())));

    let resources_id = doc.add_object(dictionary! {
        "XObject" => dictionary! {
            "Im0" => image_id,
        },
    });
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), (W as i64).into(), (H as i64).into()],
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).expect("save in-memory PDF");

    let err = extract_page_image(&bytes, 1).unwrap_err();
    assert!(matches!(err, PdfError::UnsupportedFeature(f) if f.contains("SMask")));
}

#[test]
fn one_bit_device_gray_default_decode_expands_to_black_and_white() {
    // A 3x2 checkerboard: row0 = [1,0,1] -> packed MSB-first into 1 byte
    // (bits 3-7 don't-care/padding since row must byte-align per
    // PDF32000-1:2008 §7.4.3); row1 = [0,1,0].
    let w = 3_i64;
    let h = 2_i64;
    // bit7..bit0: 1 0 1 x x x x x -> 0b1010_0000 = 0xA0
    // bit7..bit0: 0 1 0 x x x x x -> 0b0100_0000 = 0x40
    let packed = vec![0xA0u8, 0x40u8];

    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => w,
        "Height" => h,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 1,
        "Filter" => "FlateDecode",
    };
    let bytes = build_image_pdf(image_dict, zlib_compress(&packed));

    let image = extract_page_image(&bytes, 1)
        .expect("extract_page_image")
        .expect("page 1 has an image XObject");

    assert_eq!(image.w, 3);
    assert_eq!(image.h, 2);
    assert_eq!(image.data, vec![255, 0, 255, 0, 255, 0]);
}

#[test]
fn non_default_decode_array_is_a_typed_unsupported_error() {
    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
        // Inverted Decode array (exotic, not the [0 1] default).
        "Decode" => vec![1.into(), 0.into()],
    };
    let bytes = build_image_pdf(image_dict, zlib_compress(&test_grid()));

    let err = extract_page_image(&bytes, 1).unwrap_err();
    assert!(matches!(err, PdfError::UnsupportedDecodeArray(_)));
}

#[test]
fn missing_page_yields_none() {
    let grid = test_grid();
    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => W as i64,
        "Height" => H as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
    };
    let bytes = build_image_pdf(image_dict, zlib_compress(&grid));

    let result = extract_page_image(&bytes, 2).expect("extract_page_image");
    assert!(result.is_none());
}
