//! D5.1 byte-level tests for [`tesseract_ocr_pdf::extract_text_layer`].
//!
//! Both PDFs are built in-test with `lopdf` (mirroring the pattern in
//! lopdf's own `tests/unicode.rs`: a minimal Catalog → Pages → Page →
//! Contents tree), so the test has no external file dependency.

use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Document, Object};
use tesseract_ocr_pdf::extract_text_layer;

/// Build a minimal 1-page PDF whose content stream either does or does not
/// contain a `BT ... Tj ... ET` text-showing sequence.
fn build_pdf(text: Option<&str>) -> Vec<u8> {
    let mut doc = Document::new();

    let pages_id = doc.new_object_id();

    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });

    let operations = match text {
        Some(s) => vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![72.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal(s)]),
            Operation::new("ET", vec![]),
        ],
        // An image-only page: no text-showing operator at all. A real
        // scanned PDF would draw an XObject here (`/Im0 Do`); the
        // classifier only cares that no Tj/TJ text is present, so an
        // empty operation list already exercises that branch faithfully.
        None => vec![],
    };
    let content = Content { operations };
    let content_id = doc.add_object(lopdf::Stream::new(
        dictionary! {},
        content.encode().unwrap(),
    ));

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
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

#[test]
fn text_layer_page_extracts_hello_world() {
    let bytes = build_pdf(Some("Hello, world!"));
    let pages = extract_text_layer(&bytes).expect("extract_text_layer");
    assert_eq!(pages.len(), 1);
    let text = pages[0]
        .as_deref()
        .expect("page 1 should have a text layer");
    assert_eq!(text.trim_end(), "Hello, world!");
}

#[test]
fn image_only_page_yields_none() {
    let bytes = build_pdf(None);
    let pages = extract_text_layer(&bytes).expect("extract_text_layer");
    assert_eq!(pages.len(), 1);
    assert!(
        pages[0].is_none(),
        "an image-only (no text operators) page must classify as None, got {:?}",
        pages[0]
    );
}

#[test]
fn not_a_pdf_is_a_load_error() {
    let err = extract_text_layer(b"not a pdf at all").unwrap_err();
    assert!(matches!(err, tesseract_ocr_pdf::PdfError::Load(_)));
}
