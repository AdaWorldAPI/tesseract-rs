//! Block counts across every fixture — the before/after for a cut-rule change.
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]
fn main() {
    let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let mut extra: Vec<std::path::PathBuf> = std::env::args().skip(1).map(Into::into).collect();
    for sub in ["pages", "quality", "lab"] {
        if let Ok(rd) = std::fs::read_dir(corpus.join(sub)) {
            let mut v: Vec<_> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "pgm"))
                .collect();
            v.sort();
            extra.extend(v);
        }
    }
    let params = tesseract_ocr::xy_cut::XyCutParams::default();
    for p in extra {
        let Ok(b) = std::fs::read(&p) else { continue };
        let Ok((g, w, h)) = tesseract_ocr::image_input::parse_pgm(&b) else {
            continue;
        };
        let n = tesseract_ocr::xy_cut::xy_cut(&g, w, h, &params);
        let widest = n.iter().map(|r| r.right - r.left).max().unwrap_or(0);
        println!(
            "{:<28} {w}x{h:<5} blocks={:<4} widest={widest}",
            p.file_name().unwrap().to_string_lossy(),
            n.len()
        );
    }
}
