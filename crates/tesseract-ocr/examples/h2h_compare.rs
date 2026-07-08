//! P-H2H W2 -- box-matched recognition-only comparison: OUR recognizer vs the
//! C++ `tesseract` CLI vs ground truth, on the SAME identical line crop.
//!
//! This is a REPORT, not a byte-parity gate (contrast the `golden_*` suite,
//! which asserts byte-for-byte equality). Both engines are fed the exact same
//! GT-boxed line image, so this isolates the RECOGNIZER from the line finder:
//! any gap measured here is a recognition gap, not a layout/segmentation gap.
//!
//! Reads `corpus/hiertext/lines_manifest.jsonl` (one JSON object per line:
//! `crop`, `image_id`, `line_idx`, `gt_text`, `w`, `h`, `rotation_deg`,
//! `n_words` -- only `crop` and `gt_text` are used here), produced by
//! `corpus/hiertext/gen_hiertext_lines.py`. `crop` is a path RELATIVE to
//! `corpus/hiertext/`.
//!
//! Never panics on ordinary absence: a missing manifest is a `SKIP` + exit 0
//! (the generator hasn't run yet); a missing `tesseract` CLI degrades every
//! CLI-dependent column to `n/a`; a bad crop is counted and skipped.
//!
//! ```sh
//! cargo run --release -p tesseract-ocr --example h2h_compare
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tesseract_core::DictLite;
use tesseract_ocr::{parse_pgm, LstmRecognizer};

/// One `lines_manifest.jsonl` record -- only the two string fields this
/// comparison needs. `image_id`/`line_idx`/`w`/`h`/`rotation_deg`/`n_words`
/// are present in the manifest but unused here.
struct ManifestLine {
    /// Path to the line crop PGM, relative to `corpus/hiertext/`.
    crop: String,
    /// The ground-truth transcription for this line.
    gt_text: String,
}

/// Parses 4 hex digits starting at byte offset `at` into a `u32` codepoint
/// (used for `\uXXXX` JSON escapes).
fn parse_hex4(bytes: &[u8], at: usize) -> Option<u32> {
    let slice = bytes.get(at..at + 4)?;
    let hex_str = std::str::from_utf8(slice).ok()?;
    u32::from_str_radix(hex_str, 16).ok()
}

/// Parses a JSON string literal starting at byte offset `start` in `s`
/// (which must point at the opening `"`), decoding `\"`, `\\`, `\/`, `\n`,
/// `\t`, `\r`, `\b`, `\f`, and `\uXXXX` (including surrogate pairs) escapes.
/// Returns the decoded value and the byte offset just past the closing `"`.
///
/// This is a small hand-rolled extractor for exactly the string fields this
/// tool needs -- not a general JSON parser (no numbers/arrays/objects/bools).
fn parse_json_string(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut i = start + 1;
    let mut chunk_start = i;
    let mut out = String::new();
    loop {
        let b = *bytes.get(i)?;
        match b {
            b'"' => {
                out.push_str(&s[chunk_start..i]);
                return Some((out, i + 1));
            }
            b'\\' => {
                out.push_str(&s[chunk_start..i]);
                i += 1;
                let esc = *bytes.get(i)?;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'u' => {
                        let cp = parse_hex4(bytes, i + 1)?;
                        i += 4;
                        if (0xD800..=0xDBFF).contains(&cp)
                            && bytes.get(i + 1) == Some(&b'\\')
                            && bytes.get(i + 2) == Some(&b'u')
                        {
                            let cp2 = parse_hex4(bytes, i + 3)?;
                            if (0xDC00..=0xDFFF).contains(&cp2) {
                                let combined = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                                out.push(char::from_u32(combined).unwrap_or('\u{FFFD}'));
                                i += 6;
                            } else {
                                out.push('\u{FFFD}');
                            }
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                    }
                    _ => return None,
                }
                i += 1;
                chunk_start = i;
            }
            _ => {
                i += 1;
            }
        }
    }
}

/// Finds `"field":` in a JSON object line and decodes the string value that
/// follows (skipping any whitespace between the colon and the opening
/// quote). Returns `None` if the field is absent or isn't a JSON string.
fn extract_json_string_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\":");
    let key_at = line.find(&needle)?;
    let after = key_at + needle.len();
    let bytes = line.as_bytes();
    let mut i = after;
    while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }
    let (value, _end) = parse_json_string(line, i)?;
    Some(value)
}

/// Parses every non-empty line of `text` as a manifest record, skipping (and
/// warning to stderr about) any line missing `crop` or `gt_text`.
fn parse_manifest(text: &str) -> Vec<ManifestLine> {
    let mut out = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let crop = extract_json_string_field(raw, "crop");
        let gt_text = extract_json_string_field(raw, "gt_text");
        match (crop, gt_text) {
            (Some(crop), Some(gt_text)) => out.push(ManifestLine { crop, gt_text }),
            _ => eprintln!(
                "warning: manifest line {} missing crop/gt_text, skipping",
                idx + 1
            ),
        }
    }
    out
}

/// The corpus root, resolved relative to this crate's manifest dir -- the
/// same convention `golden_report.rs`/`golden_bench.rs` use.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Prints a diagnostic to stderr and exits 1 -- for setup failures (a
/// missing/corrupt model component) where continuing would only produce a
/// misleading report.
fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    std::process::exit(1);
}

/// Generic two-row Levenshtein edit distance over any equatable element type
/// (`char`s for CER, words for WER). Compact (two rolling rows) -- use
/// [`levenshtein_ops`] when the alignment itself (not just the distance) is
/// needed.
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

/// Escapes a string for safe embedding as a single markdown table cell:
/// backslashes, pipes, and embedded newlines.
fn md_cell(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
}

/// Backtracked character-level edit operations for one ours-vs-gt pair.
struct EditOps {
    /// Aligned differing characters.
    subs: usize,
    /// Extra characters OUR recognizer produced that aren't in GT.
    ins: usize,
    /// GT characters our recognizer failed to produce (dropped).
    dels: usize,
    /// `(gt_char, our_char)` for every substitution, in backtrack order.
    sub_pairs: Vec<(char, char)>,
}

/// The full (non-rolling) Levenshtein DP table over `ours` (rows) vs `gt`
/// (columns), then backtracks the minimal alignment from `(n, m)` to
/// `(0, 0)` to recover substitution/insertion/deletion counts -- unlike the
/// compact two-row [`levenshtein`] used for CER/WER means, this keeps the
/// whole matrix so the alignment can be walked back.
///
/// Direction (ASR convention, relative to transforming GT into OURS): a
/// diagonal move is a match (cost 0) or substitution (cost 1); a move that
/// consumes only an `ours` character is an INSERTION (ours has an extra
/// character not in GT); a move that consumes only a `gt` character is a
/// DELETION (a GT character ours failed to produce). On a tie, a
/// substitution is preferred over an insert+delete pair.
fn levenshtein_ops(ours: &[char], gt: &[char]) -> EditOps {
    let (n, m) = (ours.len(), gt.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(ours[i - 1] != gt[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    let (mut i, mut j) = (n, m);
    let mut subs = 0usize;
    let mut ins = 0usize;
    let mut dels = 0usize;
    let mut sub_pairs = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = usize::from(ours[i - 1] != gt[j - 1]);
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                if cost == 1 {
                    subs += 1;
                    sub_pairs.push((gt[j - 1], ours[i - 1]));
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            // Consumed an `ours` char with no `gt` counterpart: extra output
            // not present in the reference -> an insertion (by OUR recognizer).
            ins += 1;
            i -= 1;
            continue;
        }
        if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            // Consumed a `gt` char with no `ours` counterpart: a reference
            // char OUR recognizer failed to produce -> a deletion (dropped).
            dels += 1;
            j -= 1;
            continue;
        }
        break; // unreachable for a correctly-built DP table; avoids a hang
    }
    EditOps {
        subs,
        ins,
        dels,
        sub_pairs,
    }
}

/// Running totals of character-level edit operations, accumulated across
/// every processed line (see [`levenshtein_ops`]).
struct EditOpTotals {
    subs: usize,
    ins: usize,
    dels: usize,
}

/// Attribution bucket for a single line, comparing OURS and the C++ CLI
/// against ground truth via exact, case-sensitive string equality.
#[derive(Debug, Clone, Copy)]
enum Bucket {
    /// `ours == gt && cli == gt`.
    BothCorrect,
    /// `ours != gt && cli == gt` -- a recognition gap on OUR side.
    OursOnlyWrong,
    /// `cli != gt && ours == gt` -- we beat the CLI on this line.
    CliOnlyWrong,
    /// `ours == cli && ours != gt` -- both wrong the SAME way: a shared
    /// model/scene limit, not a transcode bug.
    BothWrongSame,
    /// Both wrong, and differently from each other.
    BothWrongDiff,
}

impl Bucket {
    /// Classifies a line by exact, case-sensitive string equality against
    /// `gt` -- see the definitions printed alongside the attribution table.
    fn classify(ours: &str, cli: &str, gt: &str) -> Self {
        let ours_ok = ours == gt;
        let cli_ok = cli == gt;
        match (ours_ok, cli_ok) {
            (true, true) => Bucket::BothCorrect,
            (false, true) => Bucket::OursOnlyWrong,
            (true, false) => Bucket::CliOnlyWrong,
            (false, false) => {
                if ours == cli {
                    Bucket::BothWrongSame
                } else {
                    Bucket::BothWrongDiff
                }
            }
        }
    }
}

/// Probes whether the C++ `tesseract` CLI is runnable on PATH (spawns
/// `tesseract --version`, discarding its output). A spawn failure (most
/// commonly "binary not found") is the only thing this distinguishes; a
/// found-but-misbehaving binary is still reported as available and left to
/// fail per-invocation in [`run_cli`].
fn detect_cli() -> bool {
    Command::new("tesseract")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Runs the C++ `tesseract` CLI on `crop_path`, pinned for determinism:
/// single-text-line PSM, English, LSTM-only engine, single-threaded. Returns
/// its stdout trimmed of trailing whitespace/newlines, or `None` if the
/// process could not be spawned or run at all.
fn run_cli(crop_path: &Path, tessdata_dir: Option<&Path>) -> Option<String> {
    let mut cmd = Command::new("tesseract");
    cmd.arg(crop_path).arg("stdout").arg("--psm").arg("7");
    // codex P2: pin the CLI to the committed model bytes, not the host's
    // installed eng.traineddata — `--tessdata-dir` points at the recombined
    // corpus/model components so both engines run the SAME model.
    if let Some(dir) = tessdata_dir {
        cmd.arg("--tessdata-dir").arg(dir);
    }
    cmd.arg("-l")
        .arg("eng")
        .arg("--oem")
        .arg("1")
        .env("OMP_THREAD_LIMIT", "1");
    let output = cmd.output().ok()?;
    // codex P2: a non-zero exit (e.g. the model failed to load) is a FAILURE,
    // not an empty OCR result — returning Some("") here would corrupt every
    // CER/attribution bucket. Report it as n/a instead.
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .to_string(),
    )
}

/// codex P2 — pin the CLI to the EXACT committed model: recombine the split
/// `corpus/model/eng.lstm*` components into an `eng.traineddata` in a temp dir
/// (via `combine_tessdata`), which `run_cli` passes through `--tessdata-dir`.
/// Returns the dir on success, or `None` (caller warns + falls back to the
/// host `-l eng`) when `combine_tessdata` is absent or fails.
fn pin_cli_model(model_dir: &Path) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join("tesseract_h2h_pinned_tessdata");
    fs::create_dir_all(&dir).ok()?;
    for comp in [
        "eng.lstm",
        "eng.lstm-unicharset",
        "eng.lstm-recoder",
        "eng.lstm-word-dawg",
        "eng.lstm-punc-dawg",
        "eng.lstm-number-dawg",
    ] {
        fs::copy(model_dir.join(comp), dir.join(comp)).ok()?;
    }
    let status = Command::new("combine_tessdata")
        .arg(dir.join("eng."))
        .status()
        .ok()?;
    (status.success() && dir.join("eng.traineddata").exists()).then_some(dir)
}

/// Everything computed for one manifest line: the three texts, the CER/WER
/// metrics against ground truth, and the attribution bucket.
struct LineRecord {
    /// The manifest's `crop` path (relative to `corpus/hiertext/`), used as
    /// this line's display identifier.
    crop: String,
    gt: String,
    ours: String,
    /// `None` when the CLI is unavailable or its invocation failed to spawn.
    cli: Option<String>,
    /// CER(ours vs gt), case-sensitive -- the primary metric.
    cer_cs: f64,
    /// CER(ours vs gt) with both sides lowercased first.
    cer_ci: f64,
    cer_cli_gt: Option<f64>,
    cer_ours_cli: Option<f64>,
    wer_ours_gt: f64,
    wer_cli_gt: Option<f64>,
    bucket: Option<Bucket>,
}

/// Mean of `vals`, formatted to 4 decimals, or `"n/a"` if `vals` is empty --
/// the uniform way every CLI-dependent column degrades when the CLI is
/// unavailable or produced no comparable data for any line.
fn mean_or_na(vals: &[f64]) -> String {
    if vals.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.4}", vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

/// Formats `100 * count / total` to one decimal place plus `%`, or `"n/a"`
/// if `total` is 0.
fn pct(count: usize, total: usize) -> String {
    if total == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", 100.0 * count as f64 / total as f64)
    }
}

/// Renders a single char as a markdown table cell, spelling out whitespace
/// that would otherwise be invisible or break the table.
fn fmt_char(c: char) -> String {
    match c {
        ' ' => "<space>".to_string(),
        '\t' => "<tab>".to_string(),
        '\n' => "<newline>".to_string(),
        '\r' => "<cr>".to_string(),
        _ => c.to_string(),
    }
}

/// Prints the report header: what this comparison isolates and which corpus
/// subset it runs over.
fn print_header() {
    println!("# H2H recognition-only comparison (P-H2H W2)");
    println!();
    println!("This is a RECOGNITION-ONLY comparison: both engines are fed the IDENTICAL ground-truth-boxed line crop for every row, so line-finding is not exercised on either side and any gap measured below is a recognizer gap, not a layout gap.");
    println!("The corpus is the HierText legible-horizontal-Latin line subset -- see `corpus/hiertext/gen_hiertext_lines.py` for the exact filter.");
    println!();
}

/// Prints the aggregate metrics table: sample size plus mean CER/WER across
/// every successfully-processed line.
fn print_aggregate_table(records: &[LineRecord], skipped: usize, manifest_entries: usize) {
    println!("## Aggregate");
    println!();
    let n = records.len();
    let cer_cs_vals: Vec<f64> = records.iter().map(|r| r.cer_cs).collect();
    let cer_ci_vals: Vec<f64> = records.iter().map(|r| r.cer_ci).collect();
    let cer_cli_gt_vals: Vec<f64> = records.iter().filter_map(|r| r.cer_cli_gt).collect();
    let cer_ours_cli_vals: Vec<f64> = records.iter().filter_map(|r| r.cer_ours_cli).collect();
    let wer_ours_gt_vals: Vec<f64> = records.iter().map(|r| r.wer_ours_gt).collect();
    let wer_cli_gt_vals: Vec<f64> = records.iter().filter_map(|r| r.wer_cli_gt).collect();

    println!("| metric | value |");
    println!("|---|---|");
    println!("| manifest entries (parsed) | {manifest_entries} |");
    println!("| lines skipped (crop read / PGM parse / recognize failure) | {skipped} |");
    println!("| N lines compared | {n} |");
    println!(
        "| mean CER(ours vs gt), case-sensitive | {} |",
        mean_or_na(&cer_cs_vals)
    );
    println!(
        "| mean CER(ours vs gt), case-folded (`to_lowercase`) | {} |",
        mean_or_na(&cer_ci_vals)
    );
    println!("| mean CER(cli vs gt) | {} |", mean_or_na(&cer_cli_gt_vals));
    println!(
        "| mean CER(ours vs cli) | {} |",
        mean_or_na(&cer_ours_cli_vals)
    );
    println!(
        "| mean WER(ours vs gt) | {} |",
        mean_or_na(&wer_ours_gt_vals)
    );
    println!("| mean WER(cli vs gt) | {} |", mean_or_na(&wer_cli_gt_vals));
    println!();
}

/// Bucket counts across every line -- lines without CLI data (CLI
/// unavailable, or that invocation failed to spawn) contribute to no
/// bucket, so all counts are 0 in that case.
#[derive(Default)]
struct BucketCounts {
    both_correct: usize,
    ours_only_wrong: usize,
    cli_only_wrong: usize,
    both_wrong_same: usize,
    both_wrong_diff: usize,
}

impl BucketCounts {
    fn total(&self) -> usize {
        self.both_correct
            + self.ours_only_wrong
            + self.cli_only_wrong
            + self.both_wrong_same
            + self.both_wrong_diff
    }
}

/// Prints the 5-bucket attribution table: which side (ours / the CLI /
/// neither / both) got each line right, relative to ground truth. This is
/// the KEY diagnostic: `ours-only-wrong` is a recognition gap on OUR side,
/// `both-wrong-*` is a shared model/scene limit neither engine clears, and
/// `cli-only-wrong` is a line where we beat the CLI.
fn print_attribution_table(records: &[LineRecord]) {
    println!("## Attribution (ours vs CLI vs ground truth)");
    println!();
    println!("Definitions (exact, case-sensitive string equality against gt): both-correct = ours == gt && cli == gt; ours-only-wrong = ours != gt && cli == gt (a recognition gap on OUR side); cli-only-wrong = cli != gt && ours == gt (we beat the CLI on this line); both-wrong-same-way = ours == cli != gt (a shared model/scene limit, not a transcode bug); both-wrong-diff = both wrong, differently.");
    println!();

    let mut counts = BucketCounts::default();
    for r in records {
        match r.bucket {
            Some(Bucket::BothCorrect) => counts.both_correct += 1,
            Some(Bucket::OursOnlyWrong) => counts.ours_only_wrong += 1,
            Some(Bucket::CliOnlyWrong) => counts.cli_only_wrong += 1,
            Some(Bucket::BothWrongSame) => counts.both_wrong_same += 1,
            Some(Bucket::BothWrongDiff) => counts.both_wrong_diff += 1,
            None => {}
        }
    }
    let total = counts.total();

    println!("| bucket | count | % |");
    println!("|---|---|---|");
    for (label, count) in [
        ("both-correct", counts.both_correct),
        ("ours-only-wrong", counts.ours_only_wrong),
        ("cli-only-wrong", counts.cli_only_wrong),
        ("both-wrong-same-way", counts.both_wrong_same),
        ("both-wrong-diff", counts.both_wrong_diff),
    ] {
        println!("| {label} | {count} | {} |", pct(count, total));
    }
    println!();
}

/// Prints the substitution/insertion/deletion totals (backtracked from the
/// full ours-vs-gt Levenshtein DP table, see [`levenshtein_ops`]) plus the
/// top-5 most frequent single-character substitutions.
fn print_edit_op_table(totals: &EditOpTotals, sub_pair_tally: &HashMap<(char, char), usize>) {
    println!("## Edit operations (ours vs gt, character-level)");
    println!();
    println!("| op | total |");
    println!("|---|---|");
    println!("| substitutions | {} |", totals.subs);
    println!(
        "| insertions (extra char in ours, not in gt) | {} |",
        totals.ins
    );
    println!(
        "| deletions (gt char missing from ours) | {} |",
        totals.dels
    );
    println!();

    let mut pairs: Vec<((char, char), usize)> =
        sub_pair_tally.iter().map(|(&k, &v)| (k, v)).collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    println!("Top-5 most-frequent single-character substitutions (ours vs gt):");
    println!();
    println!("| gt char | our char | count |");
    println!("|---|---|---|");
    if pairs.is_empty() {
        println!("| _(none)_ | | |");
    }
    for &((gt_ch, our_ch), count) in pairs.iter().take(5) {
        let gt_disp = md_cell(&fmt_char(gt_ch));
        let our_disp = md_cell(&fmt_char(our_ch));
        println!("| {gt_disp} | {our_disp} | {count} |");
    }
    println!();
}

/// Prints the 10 lines with the highest CER(ours vs gt, case-sensitive) --
/// an eyeball list of failure modes, not a statistical sample.
fn print_worst10(records: &[LineRecord]) {
    println!("## Worst 10 (ours vs gt, case-sensitive CER)");
    println!();
    let mut sorted: Vec<&LineRecord> = records.iter().collect();
    sorted.sort_by(|a, b| {
        b.cer_cs
            .total_cmp(&a.cer_cs)
            .then_with(|| a.crop.cmp(&b.crop))
    });

    println!("| crop | gt | ours | cli | CER |");
    println!("|---|---|---|---|---|");
    if sorted.is_empty() {
        println!("| _(none)_ | | | | |");
    }
    for r in sorted.iter().take(10) {
        let crop_disp = md_cell(&r.crop);
        let gt_disp = md_cell(&r.gt);
        let ours_disp = md_cell(&r.ours);
        let cli_disp = md_cell(r.cli.as_deref().unwrap_or("n/a"));
        let cer = r.cer_cs;
        println!("| {crop_disp} | {gt_disp} | {ours_disp} | {cli_disp} | {cer:.4} |");
    }
    println!();
}

/// Prints the closing footnotes: the pinned CLI invocation, the case-fold
/// note, the empty-ours/guard-skip note, and (when relevant) the CLI-missing
/// note.
fn print_footnotes(cli_ok: bool) {
    println!("## Footnotes");
    println!();
    println!("- CLI invocation (pinned for determinism): `tesseract <crop> stdout --psm 7 -l eng --oem 1` with `OMP_THREAD_LIMIT=1` in the environment. `--psm 7` = \"treat the image as a single text line\", matching what the crop actually is.");
    println!("- Case-folded CER lowercases both `ours` and `gt` before comparing. HierText ground truth is case-sensitive, so the case-sensitive column is the primary metric; the case-folded column shows how much of the error is attributable to case alone.");
    println!("- An empty `ours` can mean the recognizer produced nothing, OR that the line tripped the min-size guard (`Input::PrepareLSTMInputs`'s \"Image too small to scale!!\" check) and was skipped -- both cases surface as an empty string here, never a crash.");
    if !cli_ok {
        println!("- The C++ `tesseract` CLI was not found on PATH: every CLI-dependent column above is `n/a`, and the attribution table's bucket counts are all 0 by construction (no CLI text existed to compare against).");
    }
    println!("- Attribution buckets use exact, case-sensitive string equality (not a CER threshold): both-correct / ours-only-wrong / cli-only-wrong / both-wrong-same-way / both-wrong-diff, per the definitions printed above the attribution table.");
    println!("- Insertion/deletion follow the ASR convention relative to ground truth: an insertion is an extra character OUR recognizer produced that isn't in GT; a deletion is a GT character our recognizer failed to produce (dropped). Substitutions/insertions/deletions are backtracked from a full (non-rolling) Levenshtein DP table over ours-vs-gt characters, preferring a substitution over an insert+delete pair whenever both reach the minimum edit distance.");
}

fn main() {
    let root = corpus_root();
    let hiertext_dir = root.join("hiertext");
    let manifest_path = hiertext_dir.join("lines_manifest.jsonl");

    if !manifest_path.exists() {
        println!("SKIP: run corpus/hiertext/gen_hiertext_lines.py first");
        return;
    }

    let manifest_text = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| fail(&format!("read {}", manifest_path.display()), e));
    let entries = parse_manifest(&manifest_text);

    let model_dir = root.join("model");
    let lstm_path = model_dir.join("eng.lstm");
    let uni_path = model_dir.join("eng.lstm-unicharset");
    let rec_path = model_dir.join("eng.lstm-recoder");
    let word_path = model_dir.join("eng.lstm-word-dawg");
    let punc_path = model_dir.join("eng.lstm-punc-dawg");
    let number_path = model_dir.join("eng.lstm-number-dawg");

    let lstm =
        fs::read(&lstm_path).unwrap_or_else(|e| fail(&format!("read {}", lstm_path.display()), e));
    let uni = fs::read_to_string(&uni_path)
        .unwrap_or_else(|e| fail(&format!("read {}", uni_path.display()), e));
    let rec =
        fs::read(&rec_path).unwrap_or_else(|e| fail(&format!("read {}", rec_path.display()), e));
    let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
        .unwrap_or_else(|e| fail("LstmRecognizer::from_components", e));

    let word =
        fs::read(&word_path).unwrap_or_else(|e| fail(&format!("read {}", word_path.display()), e));
    let punc =
        fs::read(&punc_path).unwrap_or_else(|e| fail(&format!("read {}", punc_path.display()), e));
    let number = fs::read(&number_path)
        .unwrap_or_else(|e| fail(&format!("read {}", number_path.display()), e));
    let dict = DictLite::from_components(&word, &punc, &number)
        .unwrap_or_else(|e| fail("DictLite::from_components", e));

    let cli_ok = detect_cli();
    let tessdata_dir = if cli_ok {
        match pin_cli_model(&model_dir) {
            Some(d) => {
                eprintln!(
                    "CLI pinned to committed model: --tessdata-dir {}",
                    d.display()
                );
                Some(d)
            }
            None => {
                eprintln!(
                    "WARNING: combine_tessdata unavailable — CLI uses the HOST eng.traineddata, \
                     NOT guaranteed the same model bytes (codex P2). Attribution is only a \
                     same-model comparison if the host tessdata matches corpus/model."
                );
                None
            }
        }
    } else {
        eprintln!("SKIP: tesseract CLI not found");
        None
    };

    let mut records: Vec<LineRecord> = Vec::with_capacity(entries.len());
    let mut skipped = 0usize;
    let mut totals = EditOpTotals {
        subs: 0,
        ins: 0,
        dels: 0,
    };
    let mut sub_pair_tally: HashMap<(char, char), usize> = HashMap::new();

    for entry in &entries {
        let crop_path = hiertext_dir.join(&entry.crop);
        let bytes = match fs::read(&crop_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {}: read failed: {e}", crop_path.display());
                skipped += 1;
                continue;
            }
        };
        let (grey, w, h) = match parse_pgm(&bytes) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skip {}: pgm parse failed: {e}", crop_path.display());
                skipped += 1;
                continue;
            }
        };
        let ours = match recognizer.recognize_grey_line(&grey, w, h, Some(dict.clone())) {
            Ok((_, text)) => text,
            Err(e) => {
                eprintln!("skip {}: recognize failed: {e}", crop_path.display());
                skipped += 1;
                continue;
            }
        };
        let cli = if cli_ok {
            run_cli(&crop_path, tessdata_dir.as_deref())
        } else {
            None
        };

        let gt = entry.gt_text.clone();
        let cer_cs = cer(&ours, &gt);
        let cer_ci = cer(&ours.to_lowercase(), &gt.to_lowercase());
        let wer_ours_gt = wer(&ours, &gt);
        let (cer_cli_gt, wer_cli_gt, cer_ours_cli) = match &cli {
            Some(c) => (Some(cer(c, &gt)), Some(wer(c, &gt)), Some(cer(&ours, c))),
            None => (None, None, None),
        };
        let bucket = cli.as_deref().map(|c| Bucket::classify(&ours, c, &gt));

        let ours_chars: Vec<char> = ours.chars().collect();
        let gt_chars: Vec<char> = gt.chars().collect();
        let ops = levenshtein_ops(&ours_chars, &gt_chars);
        totals.subs += ops.subs;
        totals.ins += ops.ins;
        totals.dels += ops.dels;
        for pair in &ops.sub_pairs {
            *sub_pair_tally.entry(*pair).or_insert(0) += 1;
        }

        records.push(LineRecord {
            crop: entry.crop.clone(),
            gt,
            ours,
            cli,
            cer_cs,
            cer_ci,
            cer_cli_gt,
            cer_ours_cli,
            wer_ours_gt,
            wer_cli_gt,
            bucket,
        });
    }

    print_header();
    print_aggregate_table(&records, skipped, entries.len());
    print_attribution_table(&records);
    print_edit_op_table(&totals, &sub_pair_tally);
    print_worst10(&records);
    print_footnotes(cli_ok);
}
