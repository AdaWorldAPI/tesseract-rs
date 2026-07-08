//! Page-chunk parallel OCR (`parallel` feature).
//!
//! ## Topology — page-chunk jobs, not whole-document lanes
//!
//! The unit of parallel work is a single page: `(doc_id, page_no)` plus its
//! owned grey pixel buffer ([`PageJob`]). `rayon` fans these jobs out across
//! its global thread pool. This is deliberately **not** "one lane per
//! document" — a single large document (e.g. a 200-page PDF) must not
//! serialize onto one thread while other documents' pages sit idle on a
//! lane-per-document design. Page-chunk jobs keep every core fed regardless
//! of how work is distributed across documents.
//!
//! It is also deliberately **not** a global line/crop bucket pool: there is
//! no gather/bucket/scatter phase that collects every line crop across every
//! page before recognizing anything. Each job owns its page end-to-end,
//! running the existing **sequential** page path
//! ([`LstmRecognizer::recognize_page_makerow`]) exactly as the single-page
//! caller would. Width-bucketed intra-job batching (grouping same-width line
//! crops for fuller GEMM tiles) is explicitly out of scope for this wave —
//! it is a later optimization, gated on benchmarks actually showing
//! underfilled tiles.
//!
//! ## No nested parallelism
//!
//! `rayon` owns **all** of this crate's CPU parallelism, at the outer
//! page-job level only. The recognizer's forward pass
//! (`ndarray::simd_runtime::matmul_i8_to_i32`) is single-threaded by design.
//! Do **not** add an inner `par_iter` inside a job's recognition path, and do
//! not enable a threaded GEMM/BLAS backend underneath this driver — either
//! would oversubscribe the CPU by stacking a second parallel scheduler under
//! rayon's, with no throughput benefit and real contention cost.
//!
//! ## Determinism is a hard invariant
//!
//! Parallel output must be **byte-identical** to the serial path,
//! regardless of job submission or completion order. This holds because:
//! - each job calls `recognize_page_makerow(&self, ...)` — a shared
//!   `&LstmRecognizer` borrow, never `&mut self` — so there is no shared
//!   mutable state between jobs;
//! - each page's own recognition seeds its own randomizer internally from
//!   `sample_iteration` (see `LstmRecognizer::seeded_randomizer`), so the
//!   Convolve noise a page gets does not depend on which other jobs ran
//!   concurrently or in what order;
//! - reassembly ([`OcrPipeline::ocr_pages_parallel`]) sorts results by
//!   `(doc_id, page_no)` before returning, so the *order* of the returned
//!   `Vec` is independent of completion order too.
//!
//! [`tests/parallel_determinism.rs`] enforces this by comparing the parallel
//! path's output against the serial path's, and by feeding jobs in shuffled
//! order and asserting the returned `Vec` is still correctly sorted.
//!
//! [`LstmRecognizer::recognize_page_makerow`]: tesseract_ocr::LstmRecognizer::recognize_page_makerow
//! [`tests/parallel_determinism.rs`]: https://github.com/AdaWorldAPI/tesseract-rs

use rayon::prelude::*;
use tesseract_ocr::RecognizerError;

use crate::OcrPipeline;

/// One page's worth of parallel OCR work: an owned grey pixel buffer plus
/// enough addressing (`doc_id`, `page_no`) to sort results back into
/// document/page order after the fact. `page_no` is caller-defined (this
/// module never assumes 0- vs 1-based numbering); it only needs to sort
/// correctly within a `doc_id`.
#[derive(Debug, Clone)]
pub struct PageJob {
    /// Which document this page belongs to.
    pub doc_id: usize,
    /// The page's position within its document (used only for sorting the
    /// output; caller-defined numbering).
    pub page_no: usize,
    /// Row-major 8-bit grey pixel buffer, `width` × `height` pixels.
    pub grey: Vec<u8>,
    /// Page width in pixels.
    pub width: usize,
    /// Page height in pixels.
    pub height: usize,
}

/// One page's recognized text, tagged with the same `(doc_id, page_no)`
/// addressing as the [`PageJob`] it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResult {
    /// Which document this page belongs to.
    pub doc_id: usize,
    /// The page's position within its document.
    pub page_no: usize,
    /// The recognized text for this page.
    pub text: String,
}

impl OcrPipeline {
    /// Recognize a batch of pages in parallel, one `rayon` task per
    /// [`PageJob`], each running the existing sequential page path
    /// ([`LstmRecognizer::recognize_page_makerow`](tesseract_ocr::LstmRecognizer::recognize_page_makerow))
    /// unchanged. See the module doc comment for the full topology
    /// rationale (page-chunk jobs, no nested parallelism) and the
    /// determinism guarantee this method upholds.
    ///
    /// The returned `Vec<PageResult>` is always sorted by
    /// `(doc_id, page_no)`, regardless of the order `jobs` were submitted
    /// in or the order they finish recognizing in — callers do not need to
    /// pre-sort `jobs` or re-sort the result themselves.
    ///
    /// # Errors
    ///
    /// Returns the first [`RecognizerError`] encountered across the batch
    /// (which job "first" refers to is unspecified under concurrent
    /// execution — on any failure, treat the whole batch as failed and
    /// retry/inspect at the job level rather than relying on partial
    /// results).
    pub fn ocr_pages_parallel(
        &self,
        jobs: Vec<PageJob>,
    ) -> Result<Vec<PageResult>, RecognizerError> {
        let mut results: Vec<PageResult> = jobs
            .into_par_iter()
            .map(|job| {
                let text = self.recognizer.recognize_page_makerow(
                    &job.grey,
                    job.width,
                    job.height,
                    self.dict.as_ref(),
                )?;
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

    /// The sequential twin of [`Self::ocr_pages_parallel`] — identical
    /// per-job body (`recognize_page_makerow` + the same sort-by-key
    /// reassembly), run via a plain iterator instead of `rayon`. This
    /// exists so callers/tests have a public, independent reference to
    /// assert byte-identical output against (the determinism invariant
    /// documented at the module level); it is not expected to be faster
    /// than calling [`tesseract_ocr::LstmRecognizer::recognize_page_makerow`]
    /// directly, since [`OcrPipeline`]'s `recognizer` field is private to
    /// this crate.
    ///
    /// # Errors
    ///
    /// Returns the first [`RecognizerError`] encountered, in job order.
    pub fn ocr_pages_serial(&self, jobs: Vec<PageJob>) -> Result<Vec<PageResult>, RecognizerError> {
        let mut results: Vec<PageResult> = jobs
            .into_iter()
            .map(|job| {
                let text = self.recognizer.recognize_page_makerow(
                    &job.grey,
                    job.width,
                    job.height,
                    self.dict.as_ref(),
                )?;
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
}
