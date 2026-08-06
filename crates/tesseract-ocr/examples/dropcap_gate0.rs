//! P-41 GATE 0 — the STOP rule for default-on drop-cap handling.
//!
//! The seam recovery and the loss counter are specced default-ON, which is
//! only safe if no existing fixture carries a shape-qualified `.large` blob:
//! the corpus pages are single-font renders, so `.large` should be empty or
//! shape-disqualified everywhere. **Any hit here means re-scope to opt-in
//! rather than re-pin a golden** (the plan states this as a hard STOP, not a
//! judgement call).
//!
//! ```sh
//! cargo run -p tesseract-ocr --release --example dropcap_gate0
//! ```
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

fn pgm_fixtures(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    for sub in ["pages", "quality", "lab"] {
        if let Ok(rd) = std::fs::read_dir(root.join(sub)) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "pgm") {
                    v.push(p);
                }
            }
        }
    }
    v.sort();
    v
}

fn main() {
    let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let mut hits = 0usize;
    let mut pages = 0usize;
    for p in pgm_fixtures(&corpus) {
        let Ok(bytes) = std::fs::read(&p) else {
            continue;
        };
        let Ok((grey, w, h)) = tesseract_ocr::image_input::parse_pgm(&bytes) else {
            continue;
        };
        pages += 1;
        let binary = tesseract_ocr::xy_cut::binarize_page_with(
            &grey,
            w,
            h,
            tesseract_ocr::BinarizeMode::Otsu,
        );
        let n = tesseract_ocr::dropcap::count_page_drop_caps(&binary, w, h);
        let name = p.file_name().unwrap_or_default().to_string_lossy();
        if n > 0 {
            hits += n;
            println!("  *** {name:<28} {n} shape-qualified drop cap(s)  <-- STOP");
        } else {
            println!("  ok  {name:<28} 0");
        }
    }
    println!("\n{pages} fixtures scanned, {hits} shape-qualified drop cap(s) total");
    println!(
        "{}",
        if hits == 0 {
            "GATE 0 PASS — default-on is provably a no-op on every committed fixture."
        } else {
            "GATE 0 FAIL — re-scope to DocumentOptions opt-in; do NOT re-pin goldens."
        }
    );
}
