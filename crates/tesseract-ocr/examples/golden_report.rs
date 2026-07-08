//! D6.2 -- parity/CER report: our transcoded output vs the C++ `tesseract`
//! CLI, plus an inventory of where the hard byte-parity gates actually live.
//!
//! This is a REPORT, not a gate. **Tier A** documents the fixtures whose
//! byte-for-byte equality is already asserted by `cargo test -p tesseract-ocr`
//! (the `golden_lines`/`golden_pages`/`golden_pdfs` suites) -- this example
//! does not re-assert them, only lists what's there. **Tier B** compares
//! against the untranscoded C++ CLI, which wraps framing this crate does not
//! (and by design does not try to) reproduce byte-for-byte: DPI estimation,
//! invert-retry, PSM-13 row normalization, PSM-6 ColumnFinder/tospace. Tier B
//! is CER/WER, never pass/fail.
//!
//! Reads the fixtures banked under `corpus/` -- see
//! `corpus/gen/run_cli_golden.py` for how the `corpus/golden/cli/**` half
//! (the CLI's own output) is produced from the real `tesseract` binary.
//! Never panics on a missing fixture: prints a `SKIP <path> (absent)` line
//! and continues. Exit code is always 0 -- this is a report, not a gate.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example golden_report
//! ```

use std::fs;
use std::path::{Path, PathBuf};

/// The corpus root, resolved relative to this crate's manifest dir -- NOT
/// canonicalized (matching every other corpus-relative path convention used
/// in this repo's examples/tests).
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Generic two-row Levenshtein edit distance over any equatable element type
/// (`char`s for CER, words for WER).
fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Character error rate: `levenshtein(chars) / max(1, reference chars)`.
fn cer(ours: &str, reference: &str) -> f64 {
    let a: Vec<char> = ours.chars().collect();
    let b: Vec<char> = reference.chars().collect();
    levenshtein(&a, &b) as f64 / (b.len().max(1)) as f64
}

/// Word error rate: same as [`cer`] but over whitespace-split words.
fn wer(ours: &str, reference: &str) -> f64 {
    let a: Vec<&str> = ours.split_whitespace().collect();
    let b: Vec<&str> = reference.split_whitespace().collect();
    levenshtein(&a, &b) as f64 / (b.len().max(1)) as f64
}

/// Trims ONLY trailing newline characters (no other whitespace) -- the
/// Tier-B lines-table normalization.
fn trim_trailing_newlines(s: &str) -> &str {
    s.trim_end_matches('\n')
}

/// The Tier-B pages-table normalization: trim trailing whitespace from each
/// line, drop empty lines (the CLI's `--psm 6` output emits blank separator
/// lines between blocks), then rejoin with `"\n"`. Applied identically to
/// ours/cli/gt before CER/WER -- see the footnote this prints alongside the
/// pages table.
fn normalize_page_text(s: &str) -> String {
    s.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Reads a file to a `String`, treating ANY failure (missing, not UTF-8,
/// permission, ...) as "absent" -- this report never panics on a fixture.
fn read_to_string_opt(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// Announces a missing fixture without panicking. Still part of the markdown
/// report (stdout), not a separate diagnostic channel.
fn skip(path: &Path) {
    println!("SKIP {} (absent)", path.display());
}

/// The stem before the FIRST `.` in a filename -- e.g. `"page_01.gt.txt"`,
/// `"page_01.txt"`, and `"page_01.psm6.tsv"` all yield `"page_01"`. Every
/// filename in the fixed corpus layout has at least one segment before its
/// first dot, so `split('.').next()` always yields `Some`.
fn first_dot_stem(filename: &str) -> String {
    filename.split('.').next().unwrap_or(filename).to_string()
}

/// Escapes a string for safe embedding as a single markdown table cell:
/// backslashes, pipes, and embedded newlines. Line fixtures shouldn't carry
/// any post-trim, but a CLI framing quirk producing one is exactly what this
/// report exists to surface -- don't let it also break the table.
fn md_cell(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
}

/// Lists every file directly under `dir`, sorted by path. Empty (not an
/// error) if `dir` doesn't exist yet.
fn list_files_sorted(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files
}

/// The sorted, de-duplicated set of [`first_dot_stem`] values for every file
/// directly under `dir` (empty if `dir` doesn't exist).
fn stems_in(dir: &Path) -> Vec<String> {
    let mut stems: Vec<String> = list_files_sorted(dir)
        .into_iter()
        .filter_map(|p| p.file_name().map(|n| first_dot_stem(&n.to_string_lossy())))
        .collect();
    stems.sort();
    stems.dedup();
    stems
}

/// **Tier A** -- inventory of the byte-parity-gated fixtures. The actual gate
/// lives in `cargo test`, not here; this only documents what's on disk.
fn tier_a(root: &Path) {
    println!(
        "## Tier A — transcoded chain, byte gates (regression goldens; parity proven vs libtesseract API oracles at landing time)"
    );
    println!();
    println!("| fixture | golden bytes | note |");
    println!("|---|---|---|");
    let mut any = false;
    for sub in ["lines", "pages", "pdfs"] {
        let dir = root.join("golden").join(sub);
        let files = list_files_sorted(&dir);
        if files.is_empty() {
            println!(
                "<!-- no {sub} goldens present under {} yet -->",
                dir.display()
            );
            continue;
        }
        for f in files {
            any = true;
            let len = fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
            let name = f
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            println!(
                "| {sub}/{name} | {len} | gate lives in cargo test golden_lines/golden_pages/golden_pdfs |"
            );
        }
    }
    if !any {
        println!("| _(none present yet)_ | | |");
    }
    println!();
}

/// **Tier B, lines** -- our `.txt` vs the CLI's `--psm 13` `.txt`, both
/// trimmed of trailing newline(s) only before display/compare.
fn tier_b_lines(root: &Path) {
    println!("### Lines (--psm 13)");
    println!();

    let ours_dir = root.join("golden").join("lines");
    let cli_dir = root.join("golden").join("cli").join("lines");

    let mut stems = stems_in(&ours_dir);
    for s in stems_in(&cli_dir) {
        if !stems.contains(&s) {
            stems.push(s);
        }
    }
    stems.sort();

    // (stem, ours-trimmed, cli-trimmed, byte-equal, CER(ours vs cli))
    let mut rows: Vec<(String, String, String, bool, f64)> = Vec::new();

    for stem in &stems {
        let ours_path = ours_dir.join(format!("{stem}.txt"));
        let cli_path = cli_dir.join(format!("{stem}.psm13.txt"));
        let ours = read_to_string_opt(&ours_path);
        let cli = read_to_string_opt(&cli_path);
        match (ours, cli) {
            (Some(o), Some(c)) => {
                let ot = trim_trailing_newlines(&o).to_string();
                let ct = trim_trailing_newlines(&c).to_string();
                let eq = ot == ct;
                let cer_val = cer(&ot, &ct);
                rows.push((stem.clone(), ot, ct, eq, cer_val));
            }
            (o, c) => {
                if o.is_none() {
                    skip(&ours_path);
                }
                if c.is_none() {
                    skip(&cli_path);
                }
            }
        }
    }

    println!("| fixture | ours | CLI | byte-equal? | CER(ours vs CLI) |");
    println!("|---|---|---|---|---|");
    for (stem, ours, cli, eq, cer_val) in &rows {
        println!(
            "| {stem} | {} | {} | {} | {cer_val:.4} |",
            md_cell(ours),
            md_cell(cli),
            if *eq { "yes" } else { "no" }
        );
    }
    if rows.is_empty() {
        println!("| _(none available yet)_ | | | | |");
    }
    println!();
    println!(
        "_Footnote: \"ours\" and \"CLI\" above are trimmed of trailing newline(s) \
         ONLY before display and comparison -- no other normalization is applied \
         to the lines table._"
    );
    println!();
}

/// **Tier B, pages** -- our `.txt` / the CLI's `--psm 6` `.txt` / authored
/// ground truth, each normalized identically (see [`normalize_page_text`])
/// before CER/WER.
fn tier_b_pages(root: &Path) {
    println!("### Pages (--psm 6)");
    println!();

    let ours_dir = root.join("golden").join("pages");
    let cli_dir = root.join("golden").join("cli").join("pages");
    let gt_dir = root.join("pages");

    let mut stems: Vec<String> = list_files_sorted(&gt_dir)
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().into_owned();
            name.ends_with(".gt.txt").then(|| first_dot_stem(&name))
        })
        .collect();
    stems.sort();
    stems.dedup();

    // (page, CER ours-vs-gt, WER ours-vs-gt, CER cli-vs-gt, WER cli-vs-gt, CER ours-vs-cli)
    let mut rows: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();

    for stem in &stems {
        let gt_path = gt_dir.join(format!("{stem}.gt.txt"));
        let ours_path = ours_dir.join(format!("{stem}.txt"));
        let cli_path = cli_dir.join(format!("{stem}.psm6.txt"));

        let Some(gt) = read_to_string_opt(&gt_path) else {
            skip(&gt_path);
            continue;
        };
        let ours = read_to_string_opt(&ours_path);
        let cli = read_to_string_opt(&cli_path);
        if ours.is_none() {
            skip(&ours_path);
        }
        if cli.is_none() {
            skip(&cli_path);
        }
        let (Some(ours), Some(cli)) = (ours, cli) else {
            continue;
        };

        let gt_n = normalize_page_text(&gt);
        let ours_n = normalize_page_text(&ours);
        let cli_n = normalize_page_text(&cli);

        rows.push((
            stem.clone(),
            cer(&ours_n, &gt_n),
            wer(&ours_n, &gt_n),
            cer(&cli_n, &gt_n),
            wer(&cli_n, &gt_n),
            cer(&ours_n, &cli_n),
        ));
    }

    println!(
        "| page | CER(ours vs gt) | WER(ours vs gt) | CER(cli vs gt) | WER(cli vs gt) | CER(ours vs cli) |"
    );
    println!("|---|---|---|---|---|---|");
    for (page, cer_og, wer_og, cer_cg, wer_cg, cer_oc) in &rows {
        println!(
            "| {page} | {cer_og:.4} | {wer_og:.4} | {cer_cg:.4} | {wer_cg:.4} | {cer_oc:.4} |"
        );
    }
    if rows.is_empty() {
        println!("| _(none available yet)_ | | | | | |");
    }
    println!();
    println!(
        "_Footnote: CER/WER in the pages table normalizes each of ours/cli/gt \
         identically before comparing: trim trailing whitespace from every \
         line, drop empty lines (the CLI's --psm 6 output emits blank \
         separator lines between blocks), then rejoin with \"\\n\"._"
    );
    println!();

    if rows.is_empty() {
        println!("Summary: no pages with complete ours+cli+gt data yet.");
    } else {
        let n = rows.len() as f64;
        let mean_og: f64 = rows.iter().map(|r| r.1).sum::<f64>() / n;
        let mean_cg: f64 = rows.iter().map(|r| r.3).sum::<f64>() / n;
        println!(
            "Summary: mean CER(ours vs gt) = {mean_og:.4}, mean CER(cli vs gt) = {mean_cg:.4} (n={})",
            rows.len()
        );
    }
}

fn main() {
    let root = corpus_root();
    println!("# tesseract-rs golden parity report (D6.2)");
    println!();
    println!("Corpus root: {}", root.display());
    println!();

    tier_a(&root);

    println!(
        "## Tier B — vs C++ tesseract CLI (report only; CLI wraps untranscoded framing: DPI estimation, invert-retry, PSM-13 row normalization, PSM-6 ColumnFinder/tospace)"
    );
    println!();
    tier_b_lines(&root);
    tier_b_pages(&root);
}
