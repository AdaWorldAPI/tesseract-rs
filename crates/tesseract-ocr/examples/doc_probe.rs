//! Dump doc.v1 structure + per-line metrics for an arbitrary grey PGM.
//! Diagnostic only: reports regions, table grids, and the font-size spread
//! that drives renderer placement.
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (pgm, model) = (&a[1], a.get(2).map_or("eng", |s| s.as_str()));
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let r = LstmRecognizer::from_components(
        &std::fs::read(c.join(format!("{model}.lstm"))).unwrap(),
        &std::fs::read_to_string(c.join(format!("{model}.lstm-unicharset"))).unwrap(),
        &std::fs::read(c.join(format!("{model}.lstm-recoder"))).unwrap(),
    )
    .unwrap();
    let dict = match (
        std::fs::read(c.join(format!("{model}.lstm-word-dawg"))),
        std::fs::read(c.join(format!("{model}.lstm-punc-dawg"))),
        std::fs::read(c.join(format!("{model}.lstm-number-dawg"))),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };
    let bytes = std::fs::read(pgm).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let doc = r
        .recognize_document(&grey, w, h, dict.as_ref(), None)
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();

    let mut fonts: Vec<(f64, String)> = Vec::new();
    for page in v["pages"].as_array().unwrap() {
        println!("page {}x{}", page["width"], page["height"]);
        for reg in page["regions"].as_array().unwrap() {
            let kind = reg["type"].as_str().unwrap_or("?");
            let n = reg["lines"].as_array().map_or(0, Vec::len);
            print!("  region {kind} lines={n}");
            if let (Some(rr), Some(cc)) = (reg["rows"].as_u64(), reg["cols"].as_u64()) {
                print!(
                    "  GRID {rr}x{cc} cells={}",
                    reg["cells"].as_array().map_or(0, Vec::len)
                );
            }
            println!();
            for line in reg["lines"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                let t = line["text"].as_str().unwrap_or("");
                let words: String = line["words"]
                    .as_array()
                    .map(|ws| {
                        ws.iter()
                            .filter_map(|w| w["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_else(|| t.to_string());
                if let Some(xh) = line["xheight"].as_f64() {
                    let (asc, desc) = (
                        line["ascrise"].as_f64().unwrap_or(0.0),
                        line["descdrop"].as_f64().unwrap_or(0.0),
                    );
                    let measured = xh + asc - desc;
                    // the OTHER formula the renderer uses when metrics are None
                    let bb = line["bbox"].as_array().unwrap();
                    let box_h = bb[3].as_f64().unwrap() - bb[1].as_f64().unwrap();
                    let guessed = box_h * 0.5;
                    fonts.push((
                        measured,
                        format!(
                            "guess={guessed:5.1} ratio={:.2}x  {}",
                            guessed / measured,
                            words.chars().take(30).collect::<String>()
                        ),
                    ));
                }
            }
        }
    }
    fonts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let med = fonts[fonts.len() / 2].0;
    println!(
        "\nper-line font_px (xheight+ascrise-descdrop), median={med:.1}, n={}",
        fonts.len()
    );
    for (f, t) in fonts.iter().take(3) {
        println!("   MIN {f:6.1}  ({:.2}x median)  {t}", f / med);
    }
    for (f, t) in fonts.iter().rev().take(3) {
        println!("   MAX {f:6.1}  ({:.2}x median)  {t}", f / med);
    }
}
