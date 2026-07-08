//! `par_bench` — parallel-speedup bench for the page-chunk `rayon` driver
//! ([`tesseract_ocr_pdf::parallel`]).
//!
//! Mirrors `tesseract-ocr/examples/h2h_speed.rs`'s methodology: one untimed
//! warm-up pass, then [`TIMED_PASSES`] timed passes, keeping the MIN wall
//! time per side (the steadiest reading). Unlike `h2h_speed`, this bench
//! compares two of THIS repo's own paths against each other (serial vs
//! `OcrPipeline::ocr_pages_parallel`), not our side vs the C++ CLI.
//!
//! ## Topology under test
//!
//! The unit of parallel work is one **page-chunk job**
//! (`(doc_id, page_no)` + an owned grey page buffer) — see
//! `tesseract-ocr-pdf/src/parallel.rs`'s module doc comment for the full
//! rationale (page-chunk jobs, not whole-document lanes; no nested
//! parallelism; determinism as a hard invariant). This bench builds its job
//! set as `R` replicas of the 10 committed `corpus/pages/page_NN.pgm`
//! fixtures (`doc_id` = replica index, `page_no` = the page number), so by
//! default there are more jobs than cores and `rayon`'s work-stealing is
//! genuinely exercised.
//!
//! **Invariant this bench documents, not merely benchmarks:** the
//! recognizer's forward pass (`ndarray::simd_runtime::matmul_i8_to_i32`) is
//! single-threaded by design. `rayon` owns *all* of the CPU parallelism in
//! this crate at the outer page-job level — do not add an inner
//! `par_iter`/threaded-GEMM under this driver; that would stack a second
//! parallel scheduler underneath `rayon`'s with no throughput benefit and
//! real contention cost.
//!
//! ```sh
//! cargo run --release -p tesseract-ocr-pdf --example par_bench -- 3
//! RAYON_NUM_THREADS=4 cargo run --release -p tesseract-ocr-pdf --example par_bench -- 3
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tesseract_ocr::{parse_pgm, LstmRecognizer, RecognizerError};
use tesseract_ocr_pdf::{OcrPipeline, PageJob, PageResult};

/// Number of timed passes on each side; the reported wall time per side is
/// the MINIMUM across these (the steadiest reading), mirroring
/// `h2h_speed.rs`.
const TIMED_PASSES: usize = 3;
/// Default number of replicas of the 10-page corpus to build the job set
/// from (default 3 => 30 jobs, comfortably more than typical core counts so
/// work-stealing is actually exercised); overridable by the first CLI arg.
const DEFAULT_REPLICAS: usize = 3;
/// The 10 committed page fixtures, in page-number order.
const PAGE_NUMBERS: [u32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/// The corpus root, resolved relative to this crate's manifest dir — the
/// repo-root `corpus/` directory, same convention `h2h_speed.rs` uses
/// (`tesseract-ocr-pdf` and `tesseract-ocr` are both one level under
/// `crates/`, so the relative path is identical: `../../corpus`).
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Prints a diagnostic to stderr and exits 1. A benchmark that silently
/// produced a partial/misleading number would be worse than one that stops.
fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    std::process::exit(1);
}

/// One decoded page fixture, loaded once and reused to build every
/// replica's owned buffer without re-touching disk inside the timed region.
struct PageFixture {
    page_no: usize,
    grey: Vec<u8>,
    width: usize,
    height: usize,
}

fn load_fixtures(root: &Path) -> Vec<PageFixture> {
    let pages_dir = root.join("pages");
    PAGE_NUMBERS
        .iter()
        .map(|&n| {
            let path = pages_dir.join(format!("page_{n:02}.pgm"));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| fail(&format!("read {}", path.display()), e));
            let (grey, width, height) = parse_pgm(&bytes)
                .unwrap_or_else(|e| fail(&format!("parse_pgm {}", path.display()), e));
            PageFixture {
                page_no: n as usize,
                grey,
                width,
                height,
            }
        })
        .collect()
}

/// Build `replicas` copies of the job set from the loaded fixtures. Each
/// replica gets its own `doc_id`; `page_no` is the fixture's real page
/// number within its (replicated) document. Buffers are cloned here, OUTSIDE
/// any timed region, so the timed passes only pay for recognition.
fn build_jobs(fixtures: &[PageFixture], replicas: usize) -> Vec<PageJob> {
    let mut jobs = Vec::with_capacity(fixtures.len() * replicas);
    for doc_id in 0..replicas {
        for f in fixtures {
            jobs.push(PageJob {
                doc_id,
                page_no: f.page_no,
                grey: f.grey.clone(),
                width: f.width,
                height: f.height,
            });
        }
    }
    jobs
}

/// The serial baseline: recognize every job in `jobs`, IN ORDER, via the
/// same sequential page path `ocr_pages_parallel` calls per job
/// ([`LstmRecognizer::recognize_page_makerow`]), sorted by `(doc_id,
/// page_no)` to match `ocr_pages_parallel`'s output ordering exactly (so
/// the determinism self-check below is a plain `==`).
///
/// `OcrPipeline`'s `recognizer`/`dict` fields are private to that crate, so
/// this bench assembles its OWN recognizer + dictionary from the same
/// on-disk components via the same public API `OcrPipeline::from_data_paths`
/// uses internally (`LstmRecognizer::from_components` /
/// `DictLite::from_components`) — never reaching into `OcrPipeline`'s
/// internals, and never touching `tesseract-ocr-pdf/src/lib.rs`.
fn recognize_serial(
    recognizer: &LstmRecognizer,
    dict: Option<&tesseract_core::DictLite>,
    jobs: &[PageJob],
) -> Result<Vec<PageResult>, RecognizerError> {
    let mut results: Vec<PageResult> = jobs
        .iter()
        .map(|job| {
            let text = recognizer.recognize_page_makerow(&job.grey, job.width, job.height, dict)?;
            Ok(PageResult {
                doc_id: job.doc_id,
                page_no: job.page_no,
                text,
            })
        })
        .collect::<Result<Vec<_>, RecognizerError>>()?;
    results.sort_by_key(|r| (r.doc_id, r.page_no));
    Ok(results)
}

/// Peak resident set size (`VmHWM` in `/proc/self/status`), or a diagnostic
/// string when `/proc` isn't available on this platform. Mirrors
/// `h2h_speed.rs::peak_rss_kb`.
fn peak_rss_kb() -> String {
    match std::fs::read_to_string("/proc/self/status") {
        Ok(status) => status
            .lines()
            .find_map(|l| l.strip_prefix("VmHWM:"))
            .map(str::trim)
            .map(str::to_string)
            .unwrap_or_else(|| "VmHWM not present in /proc/self/status".to_string()),
        Err(_) => "unavailable (/proc/self/status not readable)".to_string(),
    }
}

fn main() {
    let replicas: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPLICAS);
    if replicas == 0 {
        fail("args", "replica count must be >= 1");
    }

    let root = corpus_root();
    let model_dir = root.join("model");
    let lstm_path = model_dir.join("eng.lstm");
    let uni_path = model_dir.join("eng.lstm-unicharset");
    let rec_path = model_dir.join("eng.lstm-recoder");

    // COLD: model load, once, apart from every timed pass (h2h_speed.rs
    // convention). Loaded TWICE deliberately: once as a standalone
    // recognizer for the serial baseline (see `recognize_serial`'s doc
    // comment for why), once via `OcrPipeline::from_data_paths` for the
    // parallel path under test. Both loads read the same on-disk
    // components, so the two recognizers are behaviourally identical.
    let lstm_bytes = std::fs::read(&lstm_path)
        .unwrap_or_else(|e| fail(&format!("read {}", lstm_path.display()), e));
    let uni_text = std::fs::read_to_string(&uni_path)
        .unwrap_or_else(|e| fail(&format!("read {}", uni_path.display()), e));
    let rec_bytes = std::fs::read(&rec_path)
        .unwrap_or_else(|e| fail(&format!("read {}", rec_path.display()), e));

    let serial_recognizer = LstmRecognizer::from_components(&lstm_bytes, &uni_text, &rec_bytes)
        .unwrap_or_else(|e| fail("LstmRecognizer::from_components", e));

    let word_path = model_dir.join("eng.lstm-word-dawg");
    let punc_path = model_dir.join("eng.lstm-punc-dawg");
    let number_path = model_dir.join("eng.lstm-number-dawg");
    let has_dawgs = word_path.exists() && punc_path.exists() && number_path.exists();
    let serial_dict = if has_dawgs {
        let word = std::fs::read(&word_path)
            .unwrap_or_else(|e| fail(&format!("read {}", word_path.display()), e));
        let punc = std::fs::read(&punc_path)
            .unwrap_or_else(|e| fail(&format!("read {}", punc_path.display()), e));
        let number = std::fs::read(&number_path)
            .unwrap_or_else(|e| fail(&format!("read {}", number_path.display()), e));
        Some(
            tesseract_core::DictLite::from_components(&word, &punc, &number)
                .unwrap_or_else(|e| fail("DictLite::from_components", e)),
        )
    } else {
        None
    };

    let pipeline = OcrPipeline::from_data_paths(
        &lstm_path,
        &uni_path,
        &rec_path,
        has_dawgs.then_some(word_path.as_path()),
        has_dawgs.then_some(punc_path.as_path()),
        has_dawgs.then_some(number_path.as_path()),
    )
    .unwrap_or_else(|e| fail("OcrPipeline::from_data_paths", e));

    let fixtures = load_fixtures(&root);
    let job_template = build_jobs(&fixtures, replicas);
    let jobs_len = job_template.len();

    eprintln!(
        "corpus: {} page fixtures x {replicas} replicas = {jobs_len} jobs, rayon threads = {}",
        fixtures.len(),
        rayon::current_num_threads()
    );

    // Fresh job buffers per pass, built OUTSIDE the timed region (each job
    // owns its grey buffer and is consumed by `ocr_pages_parallel`).
    let fresh_jobs = |n: usize| -> Vec<Vec<PageJob>> {
        (0..n).map(|_| build_jobs(&fixtures, replicas)).collect()
    };

    // One untimed warm-up pass on each side (the exact code paths that get
    // timed below).
    recognize_serial(&serial_recognizer, serial_dict.as_ref(), &job_template)
        .unwrap_or_else(|e| fail("serial warm-up", e));
    pipeline
        .ocr_pages_parallel(build_jobs(&fixtures, replicas))
        .unwrap_or_else(|e| fail("parallel warm-up", e));

    // SERIAL: TIMED_PASSES timed passes, MIN wall kept; every pass's output
    // is kept too (the first is used as the determinism-check reference —
    // all passes are deterministic in-order recognition, so any of them
    // would do).
    let mut serial_wall = Duration::MAX;
    let mut serial_outputs: Vec<Vec<PageResult>> = Vec::with_capacity(TIMED_PASSES);
    for jobs in fresh_jobs(TIMED_PASSES) {
        let t0 = Instant::now();
        let out = recognize_serial(&serial_recognizer, serial_dict.as_ref(), &jobs)
            .unwrap_or_else(|e| fail("serial timed pass", e));
        let elapsed = t0.elapsed();
        serial_wall = serial_wall.min(elapsed);
        serial_outputs.push(out);
    }

    // PARALLEL: TIMED_PASSES timed passes, MIN wall kept.
    let mut parallel_wall = Duration::MAX;
    let mut last_parallel_out: Option<Vec<PageResult>> = None;
    for jobs in fresh_jobs(TIMED_PASSES) {
        let t0 = Instant::now();
        let out = pipeline
            .ocr_pages_parallel(jobs)
            .unwrap_or_else(|e| fail("parallel timed pass", e));
        let elapsed = t0.elapsed();
        parallel_wall = parallel_wall.min(elapsed);
        last_parallel_out = Some(out);
    }
    let parallel_out = last_parallel_out.expect("TIMED_PASSES >= 1");

    // DETERMINISM SELF-CHECK: parallel output must be byte-identical to
    // serial output (same (doc_id, page_no) ordering, same text per page).
    let serial_out = serial_outputs.first().expect("TIMED_PASSES >= 1");
    if serial_out.len() != parallel_out.len() {
        eprintln!(
            "determinism self-check: FAILED — serial produced {} pages, parallel produced {}",
            serial_out.len(),
            parallel_out.len()
        );
        std::process::exit(1);
    }
    let mut mismatches = 0usize;
    for (s, p) in serial_out.iter().zip(parallel_out.iter()) {
        if s.doc_id != p.doc_id || s.page_no != p.page_no || s.text != p.text {
            mismatches += 1;
            eprintln!(
                "determinism self-check: mismatch at doc_id={}/{} page_no={}/{}",
                s.doc_id, p.doc_id, s.page_no, p.page_no
            );
        }
    }
    if mismatches > 0 {
        eprintln!("determinism self-check: FAILED ({mismatches} mismatching pages)");
        std::process::exit(1);
    }
    println!("determinism self-check: OK ({} pages)", serial_out.len());

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let pages_per_sec = |d: Duration| {
        if d.is_zero() {
            0.0
        } else {
            jobs_len as f64 / d.as_secs_f64()
        }
    };
    let ms_per_page = |d: Duration| {
        if jobs_len == 0 {
            0.0
        } else {
            d.as_secs_f64() * 1000.0 / jobs_len as f64
        }
    };
    let speedup = if parallel_wall.is_zero() {
        0.0
    } else {
        serial_wall.as_secs_f64() / parallel_wall.as_secs_f64()
    };

    println!();
    println!("## par_bench — page-chunk parallel-speedup bench (tesseract-ocr-pdf)");
    println!();
    println!(
        "Corpus: {} page fixtures x {replicas} replicas = {jobs_len} jobs. rayon threads: {} (`RAYON_NUM_THREADS=k` to sweep). Release recommended (`cargo run --release -p tesseract-ocr-pdf --example par_bench -- {replicas}`).",
        fixtures.len(),
        rayon::current_num_threads()
    );
    println!();
    println!("| path | wall (min of {TIMED_PASSES}) | pages/sec | ms/page |");
    println!("|---|---|---|---|");
    println!(
        "| serial (in-order, {} jobs) | {:.3} ms | {:.2} | {:.3} |",
        jobs_len,
        ms(serial_wall),
        pages_per_sec(serial_wall),
        ms_per_page(serial_wall)
    );
    println!(
        "| parallel (`ocr_pages_parallel`, {} jobs) | {:.3} ms | {:.2} | {:.3} |",
        jobs_len,
        ms(parallel_wall),
        pages_per_sec(parallel_wall),
        ms_per_page(parallel_wall)
    );
    println!();
    println!("| metric | value |");
    println!("|---|---|");
    println!("| speedup (serial / parallel) | {speedup:.2}x |");
    println!(
        "| rayon::current_num_threads() | {} |",
        rayon::current_num_threads()
    );
    println!("| peak RSS (VmHWM) | {} |", peak_rss_kb());
    println!();
    println!("---");
    println!();
    println!(
        "Footnote: the recognizer's forward pass (`ndarray::simd_runtime::matmul_i8_to_i32`) is single-threaded by design — `rayon` owns ALL of this crate's CPU parallelism at the outer page-chunk-job level; there is no inner `par_iter` or threaded GEMM stacked underneath it in either path timed above."
    );
    println!(
        "Methodology mirrors `tesseract-ocr/examples/h2h_speed.rs`: one untimed warm-up pass per side (the exact code path that gets timed), then {TIMED_PASSES} timed passes per side with the MINIMUM wall time kept (the steadiest reading); model load is a one-time cost excluded from every timed pass."
    );
    println!(
        "`RAYON_NUM_THREADS=k cargo run --release -p tesseract-ocr-pdf --example par_bench -- {replicas}` lets a caller sweep thread counts against the same job set to find the scaling curve, including `RAYON_NUM_THREADS=1` as an alternative single-threaded cross-check of the parallel code path itself (distinct from the `serial` row above, which never goes through `ocr_pages_parallel`/rayon at all)."
    );
    println!(
        "The determinism self-check above compares the parallel path's `Vec<PageResult>` against the serial path's element-by-element (same `(doc_id, page_no)` order per `ocr_pages_parallel`'s own sort, same recognized text per page) — a byte-identical match is the hard invariant this bench exists to police, not just the speed numbers."
    );
}
