//! Does our structure know where the paragraph break is?
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]
fn main() {
    let p = std::env::args().nth(1).expect("pgm");
    let b = std::fs::read(&p).expect("read");
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&b).expect("pgm");
    let params = tesseract_ocr::xy_cut::XyCutParams::default();
    let blocks = tesseract_ocr::xy_cut::xy_cut(&grey, w, h, &params);
    println!("page {w}x{h}  xy_cut blocks = {}", blocks.len());
    for (i, r) in blocks.iter().enumerate() {
        println!(
            "  block {i}: l={} t={} r={} b={}",
            r.left, r.top, r.right, r.bottom
        );
    }
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let rec = tesseract_ocr::LstmRecognizer::from_components(
        &std::fs::read(c.join("eng.lstm")).unwrap(),
        &std::fs::read_to_string(c.join("eng.lstm-unicharset")).unwrap(),
        &std::fs::read(c.join("eng.lstm-recoder")).unwrap(),
    )
    .unwrap();
    let doc = rec
        .recognize_document(&grey, w, h, None, None)
        .expect("doc");
    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();
    for (pi, pg) in v["pages"].as_array().unwrap().iter().enumerate() {
        for (ri, rg) in pg["regions"].as_array().unwrap().iter().enumerate() {
            let n = rg["lines"].as_array().map_or(0, |l| l.len());
            println!(
                "  page{pi} region{ri}: type={} lines={n} bbox={:?}",
                rg["type"], rg["bbox"]
            );
        }
    }
}
