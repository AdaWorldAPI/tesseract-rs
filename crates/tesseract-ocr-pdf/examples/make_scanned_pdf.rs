//! Builds a synthetic "scanned PDF": a single page whose only content is one
//! full-page `FlateDecode`/`DeviceGray` image XObject, embedding the grey
//! bytes of a `.pgm` file verbatim (no lossy step, unlike the DCTDecode
//! test fixture) — this is the D5.2 E2E demo input.
//!
//! ```sh
//! cargo run -q -p tesseract-ocr-pdf --example make_scanned_pdf -- \
//!     /tmp/line36.pgm /tmp/scanned_line36.pdf
//! ```

use std::io::Write as _;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use lopdf::{dictionary, Document, Object, Stream};

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "/tmp/line36.pgm".to_string());
    let output = args
        .next()
        .unwrap_or_else(|| "/tmp/scanned_line36.pdf".to_string());

    let pgm_bytes = std::fs::read(&input).unwrap_or_else(|e| panic!("reading {input}: {e}"));
    let (grey, w, h) =
        tesseract_ocr::parse_pgm(&pgm_bytes).unwrap_or_else(|e| panic!("parsing {input}: {e}"));

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&grey).expect("zlib-compress grey bytes");
    let compressed = encoder.finish().expect("finish zlib stream");

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => w as i64,
        "Height" => h as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
    };
    let image_id = doc.add_object(Stream::new(image_dict, compressed));

    let resources_id = doc.add_object(dictionary! {
        "XObject" => dictionary! {
            "Im0" => image_id,
        },
    });

    // A minimal content stream that actually paints the image (`Do`), so
    // the PDF is also valid for viewers/rasterizers, even though
    // `extract_page_image` only walks the Resources/XObject dict and
    // doesn't interpret this stream.
    let content = lopdf::content::Content {
        operations: vec![
            lopdf::content::Operation::new("q", vec![]),
            lopdf::content::Operation::new(
                "cm",
                vec![
                    (w as f32).into(),
                    0.into(),
                    0.into(),
                    (h as f32).into(),
                    0.into(),
                    0.into(),
                ],
            ),
            lopdf::content::Operation::new("Do", vec!["Im0".into()]),
            lopdf::content::Operation::new("Q", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(
        dictionary! {},
        content.encode().expect("encode content stream"),
    ));

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), (w as i64).into(), (h as i64).into()],
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

    doc.save(&output)
        .unwrap_or_else(|e| panic!("saving {output}: {e}"));
    println!("wrote {output} ({w}x{h} DeviceGray image page)");
}
