//! Measures whether `global_max_local_stddev` (Wolf's own `max_s` — the
//! page's best local contrast anywhere) separates "Sauvola will collapse"
//! from "Sauvola will succeed" across the FULL real corpus, not just the
//! two anchor fixtures already named in `PLAUSIBLE_INK_FRAC_LO`'s doc
//! comment. If it separates cleanly (with real spread margin, not just two
//! endpoints), it becomes the predictor for a gamma-gated early exit in
//! `binarize_page_escalating` that skips Sauvola's own binarization pass
//! on pages it can already tell will collapse.

use std::path::Path;
use tesseract_ocr::binarize::global_max_local_stddev;
use tesseract_ocr::xy_cut::{binarize_page_with, BinarizeMode};

// Crate convention (`local_adaptive`'s doc comment): 0 = ink, 255 = background.
fn ink_frac(binary: &[u8]) -> f32 {
    let ink = binary.iter().filter(|&&p| p == 0).count();
    ink as f32 / binary.len() as f32
}

fn main() {
    let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/quality");
    let sweep_dir = std::env::var("GAMMA_SWEEP_DIR")
        .ok()
        .map(std::path::PathBuf::from);
    let whsize = 16usize;
    let k = 0.34f32;

    let mut dirs = vec![corpus_dir];
    if let Some(d) = sweep_dir {
        dirs.push(d);
    }

    let mut rows: Vec<(String, f32, f32, bool)> = Vec::new();
    for dir in &dirs {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("pgm") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let bytes = std::fs::read(&path).unwrap();
            let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();

            let max_s = global_max_local_stddev(&grey, w, h, whsize);
            let sauvola = binarize_page_with(&grey, w, h, BinarizeMode::Sauvola { whsize, k });
            let frac = ink_frac(&sauvola);
            let collapsed = frac < 0.005; // PLAUSIBLE_INK_FRAC_LO
            rows.push((name, max_s, frac, collapsed));
        }
    }
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!(
        "{:>28} {:>10} {:>14} {:>10}",
        "fixture", "max_s", "sauvola_ink_frac", "collapsed"
    );
    for (name, max_s, frac, collapsed) in &rows {
        println!("{name:>28} {max_s:>10.2} {frac:>14.4} {:>10}", collapsed);
    }

    let collapsed_max_s: Vec<f32> = rows.iter().filter(|r| r.3).map(|r| r.1).collect();
    let healthy_max_s: Vec<f32> = rows.iter().filter(|r| !r.3).map(|r| r.1).collect();
    let hi_of_collapsed = collapsed_max_s.iter().cloned().fold(f32::MIN, f32::max);
    let lo_of_healthy = healthy_max_s.iter().cloned().fold(f32::MAX, f32::min);
    println!(
        "\ncollapsed fixtures: max_s in [{:.2}, {:.2}] (n={})",
        collapsed_max_s.iter().cloned().fold(f32::MAX, f32::min),
        hi_of_collapsed,
        collapsed_max_s.len()
    );
    println!(
        "healthy   fixtures: max_s in [{:.2}, {:.2}] (n={})",
        lo_of_healthy,
        healthy_max_s.iter().cloned().fold(f32::MIN, f32::max),
        healthy_max_s.len()
    );
    if hi_of_collapsed < lo_of_healthy {
        println!(
            "CLEAN SEPARATION: gap = [{:.2}, {:.2}], width {:.2}",
            hi_of_collapsed,
            lo_of_healthy,
            lo_of_healthy - hi_of_collapsed
        );
    } else {
        println!("NO CLEAN SEPARATION: ranges overlap");
    }
}
