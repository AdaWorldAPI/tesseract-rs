//! Recognizer **B2**: `LSTMRecognizer::DeSerialize` (`lstmrecognizer.cpp:133-177`)
//! — assemble a runnable recognizer from the serialized `lstm` component plus
//! the separate `lstm-unicharset` and `lstm-recoder` components.
//!
//! ## What B2 is (and is NOT)
//!
//! B2 is **assembly of already-proven pieces** + a thin trailing-field parse:
//! - the network (B1 `Network::from_le_bytes`, `E-OCR-NETWORK-FORWARD-1`),
//! - the character set (`E-CPP-PARITY-1..6`, `UniCharSet::load_from_str`),
//! - the recoder (`E-CPP-PARITY-7`, `UnicharCompress::from_le_bytes`),
//! - the `null_char` the CTC beam (`E-OCR-RECODEBEAM-1`) needs.
//!
//! The only NEW byte-parity content is the 8 trailing fields the lstm component
//! carries after the network. When a model is split from a `.traineddata` (as
//! `/tmp/eng.lstm` was, via `combine_tessdata -u`) the unicharset + recoder live
//! in SEPARATE components, so `include_charsets` was `false` on the wire and the
//! lstm component's tail is exactly: `network_str_` then 4×`i32`
//! (`training_flags_`, `training_iteration_`, `sample_iteration_`, `null_char_`)
//! then 3×`f32` (`adam_beta_`, `learning_rate_`, `momentum_`). The unicharset +
//! recoder are then pulled from their own components (`LoadCharsets`, the
//! `!include_charsets` branch).

use std::path::Path;

use tesseract_core::{
    ids_to_text, DictLite, RecodeBeamSearch, RecoderError, UniCharSet, UniCharSetError,
    UnicharCompress, WordResult,
};
use tesseract_recognizer::{from_grey_pix, NetworkIo, TRand};

use crate::image_input::{parse_pgm, prescale_grey_to_height, PgmError};
use crate::network::{NetError, Network};
use crate::xy_cut::BinarizeMode;

/// `TF_COMPRESS_UNICHARSET` (`lstmrecognizer.h` `TrainingFlags`): the recoder is
/// present (recoding on) rather than a pass-through identity codec.
const TF_COMPRESS_UNICHARSET: i32 = 64;

/// `kDictRatio` (`lstmrecognizer.cpp:46`) — the production certainty scale for
/// dict-path continuations, passed to `RecodeBeamSearch::Decode`.
const K_DICT_RATIO: f32 = 2.25;
/// `kCertOffset` (`lstmrecognizer.cpp:47`) — the production certainty offset.
const K_CERT_OFFSET: f32 = -0.085;
/// `kWorstDictCertainty / kCertaintyScale` (`ccmain/linerec.cpp:33,35,253-254`) —
/// the dawg-continuation certainty floor `Tesseract::LSTMRecognizeWord` passes to
/// `RecognizeLine`. The division happens in the CALLER, not in
/// `lstmrecognizer.cpp` — kept as a division here (not a pre-rounded decimal
/// literal) so the float result is bit-for-bit the expression libtesseract
/// evaluates.
const K_WORST_DICT_CERT: f32 = -25.0_f32 / 7.0_f32;

/// A failure assembling the recognizer from its components, or recognizing.
#[derive(Debug)]
pub enum RecognizerError {
    /// The network (B1) failed to load, or the trailing fields were truncated.
    Network(NetError),
    /// The unicharset text component failed to parse.
    Charset(UniCharSetError),
    /// The recoder binary component failed to parse.
    Recoder(RecoderError),
    /// An image file could not be read.
    Io(std::io::Error),
    /// An image file could not be parsed.
    Pgm(PgmError),
}

impl std::fmt::Display for RecognizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "network/tail load: {e}"),
            Self::Charset(e) => write!(f, "unicharset load: {e:?}"),
            Self::Recoder(e) => write!(f, "recoder load: {e:?}"),
            Self::Io(e) => write!(f, "image read: {e}"),
            Self::Pgm(e) => write!(f, "image parse: {e}"),
        }
    }
}

impl std::error::Error for RecognizerError {}

impl From<NetError> for RecognizerError {
    fn from(e: NetError) -> Self {
        Self::Network(e)
    }
}

/// Opt-in switches for [`LstmRecognizer::recognize_document_with_options`].
///
/// [`Default`] reproduces [`LstmRecognizer::recognize_document`] exactly:
/// `Otsu`, no border stripping. Every field here is off-by-default on purpose
/// — each changes what the recognizer sees, so each needs its own
/// measurement before it could become a default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DocumentOptions {
    /// Binarization mode for every internal pass (see
    /// [`LstmRecognizer::recognize_document_with_mode`]).
    pub binarize_mode: BinarizeMode,
    /// Paint printed borders out to background before recognition
    /// ([`crate::pageseg::strip_borders_grey`]).
    ///
    /// # Why this is a recognizer option and not a caller pre-processing step
    ///
    /// Border stripping cannot simply be applied to the page before calling in,
    /// the way [`crate::rectify::auto_rectify`] can — and the reason is a
    /// measured coupling between this crate's two table defects, worth
    /// stating because it is not obvious and it cost a wrong first attempt:
    ///
    /// - **Defect A:** printed borders are recognized AS GLYPHS (`|`, `=`, `—`,
    ///   `‘`), which corrupts cell text and — worse — fills the inter-column
    ///   gutters, so `crate::structured::extract_table_grid` has no
    ///   whitespace gap left to split columns on.
    /// - **Defect B:** a BORDERLESS table is not classified a table at all.
    ///   Two of `pageseg::decide_if_table`'s four score conditions
    ///   (`nhb > 1`, `nvb > 2`) COUNT BORDERS, and the whitespace pair alone
    ///   does not clear the threshold in practice.
    ///
    /// So stripping the borders from the page a caller passes in fixes A by
    /// causing B: the table stops being detected. Measured on the ruled
    /// four-column lab fixture — 3 recovered columns became **zero table
    /// regions** (`tests/lab_table_columns.rs`).
    ///
    /// **The printed borders are simultaneously what ruins the columns and what proves
    /// it is a table.** The two consumers therefore need DIFFERENT inputs,
    /// which only the recognizer can arrange: layout, `decide_if_table` and
    /// figure detection all read the ORIGINAL binarization, while only word
    /// and line recognition reads the stripped page. A caller outside this
    /// function cannot express that split, because by the time it has a
    /// `Document` both decisions are already made.
    pub strip_borders: bool,
}

/// The result of [`LstmRecognizer::recognize_document`] — the rendered
/// `tesseract-rs/doc.v1` JSON, the harvested typed fields, and word/line
/// counts for callers that want stats without re-parsing the JSON.
#[derive(Clone, Debug)]
pub struct Document {
    /// The rendered `doc.v1` JSON document (structure + classified regions).
    pub json: String,
    /// The harvested typed fields (empty when no harvest profile was given).
    pub fields: Vec<crate::structured::HarvestedField>,
    /// Total recognized words across all lines.
    pub word_count: usize,
    /// Number of non-empty recognized lines.
    pub line_count: usize,
    /// Mean word confidence (0–100), or `None` when no words were recognized.
    /// The honesty signal — see [`crate::structured::mean_word_confidence`].
    pub mean_confidence: Option<f32>,
    /// `true` when `mean_confidence` is below
    /// [`LOW_CONFIDENCE_THRESHOLD`](crate::structured::LOW_CONFIDENCE_THRESHOLD)
    /// — the page is likely handwriting / low-resolution / not printed text
    /// (`eng.lstm` is a print-trained model). A conservative heuristic, not a
    /// proof.
    pub low_confidence: bool,
    /// Shape-qualified drop caps on the page ([`crate::dropcap`]).
    ///
    /// **This is a LOSS counter, not a feature counter.** A drop cap is
    /// rejected by `filter_blobs` into `.large`, never reaches `make_rows`, and
    /// so contributes NO text — "Alice" reads as "ice". Measured, the fix that
    /// worked for the period bug (widen the row crop) is HARMFUL here: a cap
    /// spans ~2 line heights, so including it inflates the band ~3x and the
    /// prescale then wrecks every other glyph on the line (the whole line
    /// degrades to `"ye hewn to eet very tired of"`). The seam in
    /// [`Self::makerow_row_crops`] therefore moves the crop's left edge by at
    /// most ONE glyph width — enough to recover the cap/text seam, structurally
    /// unable to recover an 81 px cap. So this count exists to make the loss
    /// LOUD rather than silent: non-zero means the page dropped an initial that
    /// no current path recovers. Recovering it needs the cap recognized as its
    /// OWN unit and reattached — filed, not attempted.
    pub drop_caps: usize,
}

/// A loaded LSTM recognizer — the network plus the char-set / recoder tissue and
/// the scalar fields `LSTMRecognizer::DeSerialize` reads. This is the object
/// `RecognizeLine` (B3) drives; the training-only scalars are carried for
/// byte-parity + `null_char`/`is_recoding` fidelity, unused at inference.
#[derive(Debug)]
pub struct LstmRecognizer {
    /// The runnable network tree (B1).
    pub network: Network,
    /// The VGSL-ish spec string (`[1,36,0,1Ct3,3,16Mp3,3...O1c1]`).
    pub network_str: String,
    /// `TrainingFlags` bitset (`TF_INT_MODE` | `TF_COMPRESS_UNICHARSET` | ...).
    pub training_flags: i32,
    /// Training iteration counter (inference-irrelevant; carried for parity).
    pub training_iteration: i32,
    /// Sample iteration counter (also the recognizer's random seed source).
    pub sample_iteration: i32,
    /// The CTC null/blank class id (eng: 110) — the beam's `null_char`.
    pub null_char: i32,
    /// Adam β (training-only).
    pub adam_beta: f32,
    /// Learning rate (training-only).
    pub learning_rate: f32,
    /// Momentum (training-only).
    pub momentum: f32,
    /// The character set (`E-CPP-PARITY-1..6`).
    pub charset: UniCharSet,
    /// The unichar recoder (`E-CPP-PARITY-7`).
    pub recoder: UnicharCompress,
}

/// Half the average CENTRE-TO-CENTRE distance between a row's blobs — the
/// yardstick that decides whether a blob `filter_blobs` rejected as noise is
/// nonetheless plainly part of this line (see the call site in
/// [`LstmRecognizer::makerow_row_crops`] for the defect this closes).
///
/// `blob_spans` are the row's `(left, right)` pairs in any order. Returns
/// `None` when the row has fewer than two blobs, i.e. when there is no
/// spacing to average and therefore no basis to judge anything by.
///
/// # Why centre-to-centre, and why half
///
/// Edge-to-edge GAPS do not work: within-word gaps are 1-3 px and dominate the
/// mean, so half the average gap lands at ~2 px and rejects a period sitting
/// 3 px past the last letter (measured on a real scan: 1 of 7 recovered).
/// The centre-to-centre step is the glyph ADVANCE — "one character along" as
/// the eye reads it — and half of it is the same order of yardstick word-space
/// detection uses, which is also what bounds the worst case: misjudge it and
/// you pay a word space, not a lost glyph.
///
/// Crucially this is MEASURED from the row's own blobs and never derived from
/// a font-size or x-height estimate, so it normalizes across DPI, point size
/// and typeface without any absolute pixel constant.
fn noise_readmit_reach(blob_spans: &[(i32, i32)]) -> Option<f32> {
    if blob_spans.len() < 2 {
        return None;
    }
    let mut spans = blob_spans.to_vec();
    spans.sort_unstable();
    let steps: Vec<i32> = spans
        .windows(2)
        .map(|p| (((p[1].0 + p[1].1) - (p[0].0 + p[0].1)) / 2).max(0))
        .collect();
    if steps.is_empty() {
        return None;
    }
    Some(steps.iter().sum::<i32>() as f32 / steps.len() as f32 / 2.0)
}

impl LstmRecognizer {
    /// `IsRecoding()` (`lstmrecognizer.h:91`): the recoder is a real compress
    /// codec, not a pass-through. eng: true (`training_flags & 64 != 0`).
    #[must_use]
    pub fn is_recoding(&self) -> bool {
        self.training_flags & TF_COMPRESS_UNICHARSET != 0
    }

    /// `IsIntMode()` (`lstmrecognizer.h:88`, `TF_INT_MODE = 1`): the int8
    /// forward path (eng: true). The B1 forward is int8; this is the flag that
    /// says so.
    #[must_use]
    pub fn is_int_mode(&self) -> bool {
        self.training_flags & 1 != 0
    }

    /// **B3-core** — recognize an already-prepared int8 feature grid → text (the
    /// A6b-independent core of `LSTMRecognizer::RecognizeLine`,
    /// `lstmrecognizer.cpp:247-291`). Threads the proven pieces: `network.forward`
    /// (B1) → the softmax logits → `RecodeBeamSearch::decode` (`E-OCR-RECODEBEAM-1`)
    /// → `extract_best_path_as_unichar_ids` (C2) → `ids_to_text`
    /// (`E-CPP-PARITY-1`). Returns `(unichar_ids, text)`.
    ///
    /// `input` is the network's Input-shaped grid (e.g. from A6a
    /// [`from_grey_pix`](tesseract_recognizer::from_grey_pix) for a grey image;
    /// B3-core proves the grid→text seam independently of A6b's image decode).
    /// `rng` feeds `Convolve`'s out-of-image noise; seed it as the recognizer
    /// does. Decode uses `dict_ratio = 1.0`, `cert_offset = 0.0` — the best path
    /// is invariant to a uniform certainty transform, so this matches
    /// `RecognizeLine`'s `kDictRatio`/`kCertOffset` result on the non-dict path.
    ///
    /// # Errors
    ///
    /// [`RecognizerError::Network`] on a forward-pass failure, or if the output
    /// is int-mode (a non-softmax network — this path expects the softmax float
    /// logits the beam consumes).
    pub fn recognize_grid(
        &self,
        input: &NetworkIo,
        rng: &mut TRand,
    ) -> Result<(Vec<i32>, String), RecognizerError> {
        let outputs = self.network.forward(input, rng)?;
        if outputs.int_mode() {
            return Err(RecognizerError::Network(NetError::Forward(
                "recognize_grid expects softmax float logits (int-mode output)",
            )));
        }
        // SimpleTextOutput() == (OutputLossType() == LT_SOFTMAX) — derived
        // from the loaded tree (Network::simple_text_output). eng.lstm's
        // O1c111 head is NT_SOFTMAX = softmax activation with CTC LOSS, so
        // this is FALSE and the beam runs full CTC dup-collapse semantics.
        // (Softmax activation does NOT imply LT_SOFTMAX loss.)
        let simple = self.network.simple_text_output();
        let rows: Vec<&[f32]> = (0..outputs.width()).map(|t| outputs.f(t)).collect();
        let mut beam = RecodeBeamSearch::new(&self.recoder, self.null_char, simple);
        beam.decode(&rows, 1.0, 0.0);
        let (uids, _certs, _ratings, _xcoords) = beam.extract_best_path_as_unichar_ids();
        let ids: Vec<u32> = uids.iter().map(|&i| i as u32).collect();
        let text = ids_to_text(&self.charset, &ids);
        Ok((uids, text))
    }

    /// **B3-core, dict-enabled (D1.3)** — the dict-path counterpart of
    /// [`Self::recognize_grid`]: same `network.forward` → softmax logits walk,
    /// but decodes via [`RecodeBeamSearch::new_with_dict`] +
    /// [`RecodeBeamSearch::decode_with_dict`] with the production
    /// `kDictRatio`/`kCertOffset`/`worst_dict_cert` constants
    /// (`Tesseract::LSTMRecognizeWord`, `linerec.cpp:253-254`). `dict` is
    /// consumed (matches `RecodeBeamSearch` borrowing it for exactly one decode);
    /// `self.charset` is cloned into the beam (the beam needs an owned copy for
    /// `IsSpaceDelimited` lookups; `self.charset` is also needed afterward for
    /// `ids_to_text`).
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_grid`].
    pub fn recognize_grid_with_dict(
        &self,
        input: &NetworkIo,
        rng: &mut TRand,
        dict: DictLite,
    ) -> Result<(Vec<i32>, String), RecognizerError> {
        let outputs = self.network.forward(input, rng)?;
        if outputs.int_mode() {
            return Err(RecognizerError::Network(NetError::Forward(
                "recognize_grid_with_dict expects softmax float logits (int-mode output)",
            )));
        }
        let simple = self.network.simple_text_output();
        let rows: Vec<&[f32]> = (0..outputs.width()).map(|t| outputs.f(t)).collect();
        let mut beam = RecodeBeamSearch::new_with_dict(
            &self.recoder,
            self.null_char,
            simple,
            dict,
            self.charset.clone(),
        );
        beam.decode_with_dict(&rows, K_DICT_RATIO, K_CERT_OFFSET, K_WORST_DICT_CERT);
        let (uids, _certs, _ratings, _xcoords) = beam.extract_best_path_as_unichar_ids();
        let ids: Vec<u32> = uids.iter().map(|&i| i as u32).collect();
        let text = ids_to_text(&self.charset, &ids);
        Ok((uids, text))
    }

    /// Shared plumbing behind every `recognize_image_file*`/`recognize_grey_line`
    /// entry point: pre-scale a raw grey buffer to the network's input height
    /// (A6b) and build the int8 feature grid (A6a), seeding the randomizer
    /// exactly as `RecognizeLine` does ([`seeded_randomizer`]). Returns the
    /// prepared grid plus the randomizer at the post-warm-up, post-`from_grey_pix`
    /// state the forward pass expects.
    ///
    /// Pure extraction of the steps every `recognize_image_file*` method already
    /// performed inline — no behavior change.
    ///
    /// [`seeded_randomizer`]: LstmRecognizer::seeded_randomizer
    ///
    /// Returns [`None`] when the line is too small to recognize — the
    /// transcribed `Input::PrepareLSTMInputs` min-size gate (`input.cpp:92-96`,
    /// "Image too small to scale!!"; `RecognizeLine` then reports the line as
    /// not recognized and the caller skips it). The gate is checked on the
    /// **actual** prescaled width `sw` (the value `from_grey_pix` builds the
    /// grid from — floored for exact 2⁻ⁿ halvings via `scale_gray_area_map2`,
    /// so it is byte-faithful to the C++ `width`), NOT an independent
    /// `round(w·f)` estimate: on an odd-width exact halving (e.g. 5×72 → width
    /// 2) a rounded estimate reads 3 and would let a width-2 grid reach
    /// `Maxpool`'s ragged window off the grid. Gating here covers every
    /// forward call site.
    fn prepare_grid(&self, grey: &[u8], w: usize, h: usize) -> Option<(NetworkIo, TRand)> {
        let target_h = self
            .network
            .input_shape
            .map_or(36, |s| s.height.max(1) as usize);
        let (scaled, sw) = prescale_grey_to_height(grey, w, h, target_h);
        let min_width = self.network.x_scale_factor().max(1) as usize;
        if sw < min_width || target_h < min_width {
            return None;
        }
        // Seed exactly as RecognizeLine (SetRandomSeed) — the Convolve noise
        // depends on it. from_grey_pix makes no draws for a full-width image, so
        // the randomizer enters the forward pass at the post-warm-up state.
        let mut rng = self.seeded_randomizer();
        let grid = from_grey_pix(&scaled, sw, target_h, target_h as i32, 0, &mut rng);
        Some((grid, rng))
    }

    /// **D3.0 plumbing** — recognize a single already-cropped grey line strip
    /// (in memory, not a file on disk) → text, optionally through the dict
    /// beam. This is the [`prepare_grid`] + [`recognize_grid`]/
    /// [`recognize_grid_with_dict`] composition factored out of
    /// `recognize_image_file`/`recognize_image_file_with_dict` so a caller that
    /// already has a grey buffer (e.g. a cropped page band from
    /// [`find_text_lines`](crate::line_segment::find_text_lines), `seg-approx`
    /// feature) doesn't need to round-trip through a temporary PGM file.
    ///
    /// Lines whose PRE-SCALED dimensions fall below the network's
    /// [`x_scale_factor`](Network::x_scale_factor) are unrecognizable and
    /// return empty — the transcribed `Input::PrepareLSTMInputs` guard
    /// (`input.cpp:92-96`, "Image too small to scale!!"; `RecognizeLine`
    /// then reports the line as not recognized and the caller skips it).
    /// Without the guard, degenerate scene-text bands (scaled width 1-2 px)
    /// walk `Maxpool`'s ragged window off the grid.
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_grid`] / [`Self::recognize_grid_with_dict`].
    pub fn recognize_grey_line(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<DictLite>,
    ) -> Result<(Vec<i32>, String), RecognizerError> {
        // PrepareLSTMInputs' min-size gate lives in prepare_grid, on the
        // actual prescaled width — None means the line is too small to
        // recognize, so RecognizeLine skips it (empty result).
        let Some((grid, mut rng)) = self.prepare_grid(grey, w, h) else {
            return Ok((Vec::new(), String::new()));
        };
        match dict {
            Some(dict) => self.recognize_grid_with_dict(&grid, &mut rng, dict),
            None => self.recognize_grid(&grid, &mut rng),
        }
    }

    /// **D3.0 — page-level recognition composition (Batch 3-alt).**
    ///
    /// **APPROXIMATION — not a Tesseract transcode; replaced by the textord
    /// batches (plan §P3).** Segments a full GREY page into candidate text-line
    /// bands via [`find_text_lines`](crate::line_segment::find_text_lines) (the
    /// D3.0 projection-profile line finder — itself an approximation of the
    /// real textord layout pipeline), crops each band (full page width, the
    /// band's row range), and recognizes each crop via [`recognize_grey_line`]
    /// (the SAME proven line-recognition path `recognize_image_file` uses).
    /// Non-empty line texts are joined with `'\n'`; empty results (e.g. a band
    /// that decodes to nothing) are dropped rather than emitting a blank line.
    ///
    /// `dict`, if given, is cloned per line (each line gets an independent
    /// dict-beam decode) — the whole-page equivalent of choosing between
    /// [`Self::recognize_grey_line`]'s `None`/`Some(DictLite)` branches per
    /// line.
    ///
    /// # Errors
    ///
    /// The first [`RecognizerError`] hit while recognizing any band (from
    /// [`Self::recognize_grey_line`]); recognition stops at that band.
    ///
    /// [`recognize_grey_line`]: LstmRecognizer::recognize_grey_line
    #[cfg(feature = "seg-approx")]
    pub fn recognize_page(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
    ) -> Result<String, RecognizerError> {
        let bands = crate::line_segment::find_text_lines(grey, w, h);
        let mut lines: Vec<String> = Vec::with_capacity(bands.len());
        for band in bands {
            let band_h = band.height();
            if band_h == 0 {
                continue;
            }
            let crop = &grey[band.top * w..band.bottom * w];
            let (_ids, text) = self.recognize_grey_line(crop, w, band_h, dict.cloned())?;
            if !text.is_empty() {
                lines.push(text);
            }
        }
        Ok(lines.join("\n"))
    }

    /// **3F₂ — page recognition through the REAL makerow line finder.**
    ///
    /// The parity-component composition that supersedes the `seg-approx`
    /// projection-profile [`Self::recognize_page`]: every stage below is a
    /// byte-parity-proven transcode; the one documented boundary is the blob
    /// SOURCE (`pixConnComp` seedfill components vs the real pipeline's
    /// edge-traced `C_BLOB`s — see `conncomp.rs`'s island-in-hole note).
    ///
    /// Chain: Otsu binarize (P2) → [`conn_comp_areas`] (3B + 3F₂ leaf 1) →
    /// [`filter_blobs`] (3F₂ leaf 2: the `line_size`/`line_spacing`/
    /// `max_blob_size` seed, `tordmain.cpp:238-360`) → [`make_rows`]
    /// (waves 1+2) → [`compute_block_xheight`] (wave 3) → each row fed as
    /// the TYPOGRAPHIC line box of `Tesseract::LSTMRecognizeWord`
    /// (`linerec.cpp:239-246`: the row's ink bounding box EXTENDED — never
    /// shrunk — to `[baseline + descdrop, baseline + xheight + ascrise]`,
    /// baseline evaluated at the box x-midpoint from the wave-2 parallel
    /// fit) → `GetRectImage`'s `kImagePadding = 4` pad on all sides + clip
    /// to the image → [`Self::recognize_grey_line`] (the proven A6b line
    /// path). In LSTM mode the real pipeline recognizes a whole textline
    /// per call (the row's words are merged before `LSTMRecognizeWord`), so
    /// feeding the row box IS the real feeding semantics.
    ///
    /// Coordinate note: components come out in raster space (`y` down);
    /// makerow runs in Tesseract's y-UP page space, so boxes are flipped
    /// (`bottom = h - (y + bh)`, `top = h - y`) on the way in and the padded
    /// typographic box flipped back on the way out. Rows are kept in the
    /// `TO_ROW_LIST` order make_rows maintains (descending `min_y` = top of
    /// page first), so the joined text reads top-to-bottom.
    ///
    /// Feeding is position-invariant when nothing clips: identical ink at
    /// different page positions yields pixel-identical crops (the roomy
    /// stacked fixture asserts this). Near the image edges the pad+clip
    /// truncates faithfully, exactly as `GetRectImage` does. Remaining
    /// documented approximations: the blob source (above) and the
    /// straight-baseline case (`baseline = m·x + parallel_c`; the real
    /// `row->base_line()` consults the quadratic spline where one exists).
    ///
    /// # Errors
    ///
    /// The first [`RecognizerError`] from any line's recognition.
    pub fn recognize_page_makerow(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
    ) -> Result<String, RecognizerError> {
        self.recognize_page_makerow_with_mode(grey, w, h, dict, BinarizeMode::default())
    }

    /// The same [`Self::recognize_page_makerow`] composition, with the
    /// makerow line-finder's own segmentation binarization selectable via an
    /// explicit [`BinarizeMode`](crate::xy_cut::BinarizeMode).
    /// [`Self::recognize_page_makerow`] is exactly this method called with
    /// `BinarizeMode::default()` (i.e. [`BinarizeMode::Otsu`]) — byte-identical
    /// to its pre-existing behaviour, so every existing caller of
    /// [`Self::recognize_page_makerow`] is unaffected by this addition.
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_page_makerow`].
    pub fn recognize_page_makerow_with_mode(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        binarize_mode: BinarizeMode,
    ) -> Result<String, RecognizerError> {
        // The shared makerow layout prefix (binarize → components → y-flip →
        // filter → make_rows → xheight → per-row typographic crop) lives in
        // [`Self::makerow_row_crops`] so this string method and
        // [`Self::recognize_page_makerow_words_with_mode`] cannot drift. The
        // tail below is byte-identical to the pre-refactor loop: each
        // non-empty row's crop → [`recognize_grey_line`] → keep the
        // non-empty text → join with `'\n'`. (Rows the prefix skipped —
        // empty `row.blobs` or a degenerate padded box — never surface as
        // crops, exactly as the old `continue`s.)
        //
        // [`recognize_grey_line`]: LstmRecognizer::recognize_grey_line
        let crops = self.makerow_row_crops(grey, w, h, binarize_mode);
        let mut lines: Vec<String> = Vec::with_capacity(crops.len());
        for row in &crops {
            let (_ids, text) =
                self.recognize_grey_line(&row.crop, row.band_w, row.band_h, dict.cloned())?;
            if !text.is_empty() {
                lines.push(text);
            }
        }
        Ok(lines.join("\n"))
    }

    /// The shared makerow layout prefix behind both
    /// [`Self::recognize_page_makerow_with_mode`] and
    /// [`Self::recognize_page_makerow_words_with_mode`]: binarize the page
    /// (under the given [`BinarizeMode`]), find components, run the real
    /// make_rows line finder, and emit — per recognizable row, in
    /// top-of-page-first order — the padded typographic crop plus that
    /// row's bottom-up `TBOX` line box in PAGE space.
    ///
    /// Factored out verbatim from the original `recognize_page_makerow` body so
    /// the string and word surfaces feed the recognizer IDENTICAL crops in
    /// IDENTICAL order; see [`Self::recognize_page_makerow`]'s doc comment for
    /// the full chain rationale (binarize → [`conn_comp_areas`] → [`filter_blobs`]
    /// → [`make_rows`] → [`compute_block_xheight`] → `linerec.cpp:239-246`
    /// typographic band + `GetRectImage` `kImagePadding = 4` pad/clip).
    /// `binarize_mode` reaches this binarization step via
    /// `crate::segment::segment_rows_with_mode` — see that module's docs for
    /// why `crate::rectify` (a NEW, non-Tesseract preprocessing addition)
    /// needs a DIFFERENT sibling entry point (`segment_rows_independent`)
    /// rather than sharing this one.
    ///
    /// Rows the pipeline cannot feed produce NO entry (mirroring the original
    /// inline `continue`s): a row with empty `row.blobs`, or one whose
    /// padded+clipped box is degenerate (`img_bottom <= img_top` or
    /// `img_right <= img_left`).
    ///
    /// [`conn_comp_areas`]: crate::conncomp::conn_comp_areas
    /// [`filter_blobs`]: crate::blob_filter::filter_blobs
    /// [`make_rows`]: crate::textline::make_rows
    /// [`compute_block_xheight`]: crate::textline::compute_block_xheight
    fn makerow_row_crops(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        binarize_mode: BinarizeMode,
    ) -> Vec<MakerowRowCrop> {
        // The binarize→components→make_rows→xheight prefix is factored out
        // as crate::segment::segment_rows_with_mode (segment.rs threads
        // BinarizeMode through what used to be extracted verbatim from this
        // function's original body; no OTHER behaviour change) — see that
        // module's docs for why crate::rectify (a NEW, non-Tesseract
        // preprocessing addition) needs a DIFFERENT sibling entry point
        // (segment_rows_independent) rather than sharing this one.
        let block = crate::segment::segment_rows_with_mode(grey, w, h, binarize_mode);

        // Rows (top-of-page first) → the TYPOGRAPHIC line box → the proven
        // line path. This is the real pipeline's feeding, not the expanded
        // TO_ROW band: `Tesseract::LSTMRecognizeWord` (`linerec.cpp:239-246`)
        // starts from the ink's bounding box and EXTENDS it (never shrinks)
        // to at least `[baseline + descenders, baseline + x_height +
        // ascenders]` — the baseline evaluated at the box's x-midpoint from
        // the row's fitted line (straight-baseline case: `m·x + parallel_c`,
        // our wave-2 parallel fit) — then `GetRectImage` pads by
        // `kImagePadding = 4` on ALL sides (`imagedata.h:39`) and clips to
        // the image, cropping x AND y. descdrop/xheight/ascrise come from
        // wave 3's `compute_block_xheight`. The recognizer input is then the
        // proven prescale+FromPix path, exactly as `RecognizeLine` does.
        const K_IMAGE_PADDING: i32 = 4;
        // Shape-qualified drop caps, computed ONCE against the page's own blob
        // population (never per row — the population is a page property).
        let caps = crate::dropcap::detect_drop_caps(&block.large, &block.blobs);
        let mut out: Vec<MakerowRowCrop> = Vec::with_capacity(block.rows.len());
        for row in &block.rows {
            if row.blobs.is_empty() {
                continue;
            }
            // Ink bounding box of the row (y-up space).
            let mut left = i32::MAX;
            let mut right = i32::MIN;
            let mut bottom = f32::MAX;
            let mut top = f32::MIN;
            for &(l, b, r, t) in &row.blobs {
                left = left.min(l);
                right = right.max(r);
                bottom = bottom.min(b as f32);
                top = top.max(t as f32);
            }

            // ── "must-consider" noise re-admission ────────────────────────
            // `filter_blobs` rejects any component shorter than
            // `textord_max_noise_size` (7 px) as noise, and this port then
            // DROPPED that list. A period is definitionally a tiny solid dot
            // — 5-6 px tall at ordinary book scale — so it fails that
            // ABSOLUTE bar at any normal scan resolution, forever. Measured
            // consequence on a real two-column scan: 2 periods against 42
            // commas, 0 of 72 lines ending in one, where libtesseract gets 9.
            // (Commas descend below the baseline, clear h >= 7, and are
            // mid-line anyway — which is exactly why they survived.)
            //
            // The fix judges a blob by WHAT IS AROUND IT instead of by an
            // absolute size — the same correction the `xy_cut` column-gutter
            // fallback makes one layer up. A rejected blob is re-admitted to
            // this row's crop when it sits inside the row's vertical ink band
            // AND lies within HALF THE AVERAGE INTER-BLOB DISTANCE of the
            // row's ink. That distance is MEASURED from this row's own blob
            // gaps, never derived from a font-size/x-height estimate, so it
            // normalizes across DPI, point size and typeface on its own.
            //
            // Asymmetric by design: admitting a stray speck costs at most one
            // spurious mark the CTC decoder usually swallows, and in the worst
            // case reads as a word space — while rejecting a period deletes
            // real content from every sentence on the page. This only widens
            // the CROP; `make_rows` still never sees these blobs, so row
            // assignment, x-height and baseline fitting are untouched.
            // Hoisted out of the noise guard below: the drop-cap seam uses the
            // SAME population-relative yardstick, so both consumers must read
            // one `reach`, not two independently-derived ones.
            let spans: Vec<(i32, i32)> = row.blobs.iter().map(|&(l, _, r, _)| (l, r)).collect();
            let reach = noise_readmit_reach(&spans);

            if !block.noise.is_empty() && row.blobs.len() > 1 {
                if let Some(reach) = reach {
                    for &(nl, nb, nr, nt) in &block.noise {
                        let vcenter = (nb + nt) as f32 / 2.0;
                        if vcenter < bottom || vcenter > top {
                            continue;
                        }
                        let dist = if nr < left {
                            (left - nr) as f32
                        } else if nl > right {
                            (nl - right) as f32
                        } else {
                            0.0
                        };
                        if dist <= reach {
                            left = left.min(nl);
                            right = right.max(nr);
                            bottom = bottom.min(nb as f32);
                            top = top.max(nt as f32);
                        }
                    }
                }
            }

            // Drop-cap seam. A cap spans several line heights, so
            // `filter_blobs` rejects it into `.large` and `make_rows` never
            // sees it — which is why "Alice" recognized as "ice". Measured,
            // the two size-extreme defects have OPPOSITE correct treatments:
            // a line-final period IS part of its line and only needed the
            // crop to reach it, but including a whole cap forces the band to
            // ~3x its true height and the prescale then wrecks every other
            // glyph on the line (measured: the full line degrades to
            // "ye hewn to eet very tired of"). So the seam moves the crop's
            // LEFT edge only, by at most one glyph width, and never into the
            // cap's body — it recovers the seam between cap and text, not the
            // cap itself. Left edge only: `bottom`/`top` stay untouched, so
            // the band height — the thing that broke — cannot change.
            if !caps.is_empty() {
                if let Some(r) = reach {
                    if let Some(new_left) =
                        crate::dropcap::seam_left_extension(left, bottom, top, &spans, &caps, r)
                    {
                        left = new_left;
                    }
                }
            }

            // linerec.cpp:240-246 — extend to the typographic band.
            let mid_x = (left + right) as f32 / 2.0;
            let baseline = row.line_m() * mid_x + row.parallel_c();
            if baseline + row.descdrop < bottom {
                bottom = baseline + row.descdrop;
            }
            if baseline + row.xheight + row.ascrise > top {
                top = baseline + row.xheight + row.ascrise;
            }
            // GetRectImage: pad 4 all sides, clip to the image (x AND y),
            // then flip y-up → raster rows for the crop.
            let img_left = (left - K_IMAGE_PADDING).max(0) as usize;
            let img_right = ((right + K_IMAGE_PADDING) as usize).min(w);
            let img_top = ((h as f32 - (top + K_IMAGE_PADDING as f32)).floor()).max(0.0) as usize;
            let img_bottom =
                ((h as f32 - (bottom - K_IMAGE_PADDING as f32)).ceil()).min(h as f32) as usize;
            if img_bottom <= img_top || img_right <= img_left {
                continue;
            }
            let band_w = img_right - img_left;
            let band_h = img_bottom - img_top;
            let mut crop = Vec::with_capacity(band_w * band_h);
            for y in img_top..img_bottom {
                crop.extend_from_slice(&grey[y * w + img_left..y * w + img_right]);
            }
            // The raster crop rectangle → the bottom-up TBOX line box in PAGE
            // space (renderer.rs:88-95 `LineWords::line_box` order). The prefix
            // built img_* in raster (y-down) space; flip y back to the page's
            // y-UP `TBOX` frame: bottom = h - img_bottom, top = h - img_top. x
            // is unchanged (left = img_left, right = img_right). This is the box
            // `extract_best_path_as_words` (recodebeam.rs:1663-1718) offsets
            // char boundaries into, so char x lands at `img_left + boundary`.
            let line_box = (
                img_left as i32,
                (h - img_bottom) as i32,
                img_right as i32,
                (h - img_top) as i32,
            );
            out.push(MakerowRowCrop {
                crop,
                band_w,
                band_h,
                line_box,
                metrics: crate::renderer::LineMetrics {
                    xheight: row.xheight,
                    ascrise: row.ascrise,
                    descdrop: row.descdrop,
                    baseline,
                },
            });
        }
        out
    }

    /// **The word/box page surface** — the [`WordResult`]-returning counterpart
    /// of [`Self::recognize_page_makerow`]. Runs the SAME makerow layout prefix
    /// ([`Self::makerow_row_crops`] — so the two cannot drift and every crop is
    /// byte-identical to the string path), then per row decodes to WORDS the way
    /// [`Self::recognize_image_file_words`] does (`prepare_grid` →
    /// `network.forward` → [`RecodeBeamSearch`] with the `None`/`Some(dict)`
    /// arm's constants → [`RecodeBeamSearch::extract_best_path_as_words`],
    /// recodebeam.rs:1663-1718). Returns one [`LineWords`](crate::renderer::LineWords)
    /// per recognized row, top-of-page first — ready for
    /// [`render_text`](crate::renderer::render_text) / `render_tsv` / `render_hocr`.
    ///
    /// ## The two precision points (both proven against the cited source)
    ///
    /// 1. **`line_box`** is the row's crop rectangle in bottom-up `TBOX` PAGE
    ///    space `(left, bottom, right, top)` — [`makerow_row_crops`] flips the
    ///    raster crop (`bottom = h - img_bottom`, `top = h - img_top`; x
    ///    unchanged), matching `LineWords::line_box` (renderer.rs:88-95).
    ///    `extract_best_path_as_words` (recodebeam.rs:1712-1718) offsets every
    ///    char boundary by `line_box.left` and stamps `line_box.bottom` /
    ///    `line_box.top` as the char box's vertical extent, so the words land in
    ///    the row's page rectangle.
    /// 2. **`scale_factor`** un-does the crop's prescale so char x lands in
    ///    ORIGINAL page pixels. The crop (height `band_h`) is scaled to the
    ///    network input height `target_h` by `im_factor = target_h / band_h`
    ///    (`prescale_grey_to_height`, image_input.rs:137; the eng network is 1:1
    ///    in x, so an output timestep column maps to one scaled-crop pixel
    ///    column). `extract_best_path_as_words` MULTIPLIES char boundaries by
    ///    `scale_factor` (recodebeam.rs:1713/1716), so the inverse
    ///    `scale_factor = band_h / target_h = 1 / im_factor` recovers original
    ///    crop pixels — exactly `ImageData::PreScale`'s `*scale_factor =
    ///    1 / im_factor`. `target_h` is read as in [`prepare_grid`]
    ///    (lstm_recognizer.rs:244-247); at model height (`band_h == target_h`)
    ///    the factor is `1.0`, matching [`Self::recognize_image_file_words`].
    ///
    /// ## Skips (mirroring the string path)
    ///
    /// A row is dropped (nothing pushed) when its crop is too small to scale
    /// (`prepare_grid` returns [`None`] — the `PrepareLSTMInputs` min-size gate,
    /// input.cpp:92-96) OR when the beam yields no words — the word-surface
    /// analogue of [`Self::recognize_page_makerow`] skipping an empty-text row,
    /// so both surfaces report the same set of recognized lines.
    ///
    /// # Errors
    ///
    /// The first [`RecognizerError`] from any row's forward pass (or an
    /// unexpected int-mode output — this path needs the softmax float logits).
    ///
    /// [`makerow_row_crops`]: LstmRecognizer::makerow_row_crops
    /// [`prepare_grid`]: LstmRecognizer::prepare_grid
    pub fn recognize_page_makerow_words(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
    ) -> Result<Vec<crate::renderer::LineWords>, RecognizerError> {
        self.recognize_page_makerow_words_with_mode(grey, w, h, dict, BinarizeMode::default())
    }

    /// The same [`Self::recognize_page_makerow_words`] composition, with the
    /// makerow line-finder's own segmentation binarization selectable via an
    /// explicit [`BinarizeMode`](crate::xy_cut::BinarizeMode).
    /// [`Self::recognize_page_makerow_words`] is exactly this method called
    /// with `BinarizeMode::default()` (i.e. [`BinarizeMode::Otsu`]) —
    /// byte-identical to its pre-existing behaviour, so every existing
    /// caller of [`Self::recognize_page_makerow_words`] is unaffected by
    /// this addition.
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_page_makerow_words`].
    pub fn recognize_page_makerow_words_with_mode(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        binarize_mode: BinarizeMode,
    ) -> Result<Vec<crate::renderer::LineWords>, RecognizerError> {
        // Same target height prepare_grid derives (lstm_recognizer.rs:244-247).
        let target_h = self
            .network
            .input_shape
            .map_or(36, |s| s.height.max(1) as usize);
        // `RecognizeLine`'s "Reduction factor from image to coords"
        // (`lstmrecognizer.cpp:327,344`):
        //     min_width      = network_->XScaleFactor()
        //     *scale_factor  = im_factor        (set by PrepareLSTMInputs)
        //     *scale_factor  = min_width / *scale_factor
        // i.e. scale_factor = XScaleFactor() / im_factor. The beam's
        // character_boundaries are DECODER TIMESTEP indices, and the network
        // reduces width by XScaleFactor() (eng.lstm: 3, from `Mp3,3`), so the
        // pooling factor is as load-bearing as the prescale — omitting it
        // compresses every word box toward the line's left edge by exactly
        // that factor while leaving the TEXT correct.
        let x_scale = self.network.x_scale_factor().max(1) as f32;
        let crops = self.makerow_row_crops(grey, w, h, binarize_mode);
        let mut out: Vec<crate::renderer::LineWords> = Vec::with_capacity(crops.len());
        for row in &crops {
            // PrepareLSTMInputs min-size gate (input.cpp:92-96): a crop too
            // small to scale is unrecognizable — RecognizeLine skips it, so we
            // push nothing (the word analogue of the string path's empty-text
            // skip).
            let Some((grid, mut rng)) = self.prepare_grid(&row.crop, row.band_w, row.band_h) else {
                continue;
            };
            let outputs = self.network.forward(&grid, &mut rng)?;
            if outputs.int_mode() {
                return Err(RecognizerError::Network(NetError::Forward(
                    "recognize_page_makerow_words_with_mode expects softmax float logits (int-mode output)",
                )));
            }
            let simple = self.network.simple_text_output();
            let logits: Vec<&[f32]> = (0..outputs.width()).map(|t| outputs.f(t)).collect();
            let scale_factor = x_scale * (row.band_h as f32 / target_h as f32);
            let words = if let Some(dict) = dict.cloned() {
                let mut beam = RecodeBeamSearch::new_with_dict(
                    &self.recoder,
                    self.null_char,
                    simple,
                    dict,
                    self.charset.clone(),
                );
                beam.decode_with_dict(&logits, K_DICT_RATIO, K_CERT_OFFSET, K_WORST_DICT_CERT);
                beam.extract_best_path_as_words(row.line_box, scale_factor, &self.charset)
            } else {
                let mut beam = RecodeBeamSearch::new(&self.recoder, self.null_char, simple);
                beam.decode(&logits, 1.0, 0.0);
                beam.extract_best_path_as_words(row.line_box, scale_factor, &self.charset)
            };
            // Mirror the string path's `if !text.is_empty()`: a row that decodes
            // to no words contributes nothing.
            if words.is_empty() {
                continue;
            }
            out.push(crate::renderer::LineWords {
                words,
                line_box: row.line_box,
                metrics: Some(row.metrics),
            });
        }
        Ok(out)
    }

    /// **Multi-column reading order** — consumer-side composition (NOT a
    /// Tesseract transcode; same footing as `structured.rs`): run
    /// [`xy_cut`](crate::xy_cut::xy_cut) layout analysis FIRST, then the
    /// proven makerow row finder WITHIN each block, concatenating the
    /// blocks' lines in XY-cut reading order.
    ///
    /// Why: whole-page makerow finds rows by projection across the FULL page
    /// width, so side-by-side columns merge into single full-width rows read
    /// ACROSS the gutter — a real repro (an 8-column resolution test sheet)
    /// produced 26 full-width lines where ~176 per-column lines exist, each
    /// "line" concatenating all 8 columns left-to-right. Real Tesseract's own
    /// pipeline runs layout analysis (textord blocks) before per-block line
    /// finding; this composition mirrors that ordering with the pieces this
    /// crate already has.
    ///
    /// A page where XY-cut finds no split (0 or 1 leaf — the common
    /// single-column case) takes the EXACT whole-page
    /// [`Self::recognize_page_makerow_words`] path: byte-identical output,
    /// no behaviour change for existing single-column consumers.
    ///
    /// Per-block outputs are translated back into full-page coordinates
    /// (`x += crop_left`; bottom-up `y += page_h - crop_bottom`), covering
    /// `line_box`, every word's `char_boxes`, and the line metrics' baseline
    /// — so downstream consumers (doc.v1, renderers, region assignment) see
    /// one coherent page space regardless of which path ran.
    ///
    /// # Errors
    ///
    /// The first [`RecognizerError`] from any block's recognition.
    pub fn recognize_page_blocks_words(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
    ) -> Result<Vec<crate::renderer::LineWords>, RecognizerError> {
        self.recognize_page_blocks_words_with_mode(grey, w, h, dict, BinarizeMode::default())
    }

    /// The same [`Self::recognize_page_blocks_words`] composition, with the
    /// column-layout `xy_cut` split AND the makerow line-finder's own
    /// segmentation binarization both selectable via a single explicit
    /// [`BinarizeMode`](crate::xy_cut::BinarizeMode) — the fix for the gap
    /// `.claude/harvest/sauvola-vs-otsu-probe.md` measured: `binarize_mode`
    /// used to reach only the LAYOUT `xy_cut` call inside
    /// [`Self::recognize_document_with_mode`], never the word/line
    /// recognition this method performs.
    /// [`Self::recognize_page_blocks_words`] is exactly this method called
    /// with `BinarizeMode::default()` (i.e. [`BinarizeMode::Otsu`]) —
    /// byte-identical to its pre-existing behaviour, so every existing
    /// caller of [`Self::recognize_page_blocks_words`] is unaffected by
    /// this addition.
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_page_blocks_words`].
    pub fn recognize_page_blocks_words_with_mode(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        binarize_mode: BinarizeMode,
    ) -> Result<Vec<crate::renderer::LineWords>, RecognizerError> {
        let xy_params = crate::xy_cut::XyCutParams {
            binarize_mode,
            ..crate::xy_cut::XyCutParams::default()
        };
        let blocks = crate::xy_cut::xy_cut(grey, w, h, &xy_params);
        if blocks.len() <= 1 {
            return self.recognize_page_makerow_words_with_mode(grey, w, h, dict, binarize_mode);
        }
        self.recognize_blocks_words(grey, w, h, dict, binarize_mode, &blocks)
    }

    /// The per-block crop → makerow → shift-back → no-content-loss-guard
    /// body shared by [`Self::recognize_page_blocks_words_with_mode`] (plain
    /// [`crate::xy_cut::xy_cut`] blocks) and
    /// [`Self::recognize_document_with_options`]'s table-aware recognition
    /// path ([`crate::xy_cut::xy_cut_table_aware`] blocks) — factored out so
    /// the two callers cannot drift on this logic while differing only in
    /// WHICH block list they hand it.
    ///
    /// `blocks` is assumed non-empty and pre-filtered to `len() > 1` by the
    /// caller (the `<= 1` whole-page special case lives at each call site,
    /// not here, since the two callers reach it via different block-finding
    /// calls).
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_page_blocks_words_with_mode`].
    fn recognize_blocks_words(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        binarize_mode: BinarizeMode,
        blocks: &[crate::xy_cut::PageRect],
    ) -> Result<Vec<crate::renderer::LineWords>, RecognizerError> {
        let mut any_block_empty = false;

        // GetRectImage's kImagePadding (imagedata.h:39) — the same slack the
        // per-row crops get, so a block's edge glyphs keep their context.
        const PAD: usize = 4;
        let mut out: Vec<crate::renderer::LineWords> = Vec::new();
        for blk in blocks {
            let left = blk.left.saturating_sub(PAD);
            let top = blk.top.saturating_sub(PAD);
            let right = (blk.right + PAD).min(w);
            let bottom = (blk.bottom + PAD).min(h);
            if right <= left || bottom <= top {
                continue;
            }
            let (cw, ch) = (right - left, bottom - top);
            let mut crop = Vec::with_capacity(cw * ch);
            for y in top..bottom {
                crop.extend_from_slice(&grey[y * w + left..y * w + right]);
            }

            let lines =
                self.recognize_page_makerow_words_with_mode(&crop, cw, ch, dict, binarize_mode)?;
            if lines.is_empty() {
                any_block_empty = true;
            }

            // Crop space → page space. x shifts by the crop's left edge; the
            // bottom-up y frames relate by dy = page_h - crop_bottom (a crop
            // y-up coordinate yc sits at page raster row `top + (ch - yc)`,
            // i.e. page y-up `h - top - ch + yc = yc + (h - bottom)`).
            let dx = left as i32;
            let dy = (h - bottom) as i32;
            for mut line in lines {
                let (l, b, r, t) = line.line_box;
                line.line_box = (l + dx, b + dy, r + dx, t + dy);
                for word in &mut line.words {
                    for cb in &mut word.char_boxes {
                        cb.0 += dx;
                        cb.1 += dy;
                        cb.2 += dx;
                        cb.3 += dy;
                    }
                }
                if let Some(m) = &mut line.metrics {
                    m.baseline += dy as f32;
                }
                out.push(line);
            }
        }

        // No-content-loss guard: a block that recognized to NOTHING is either
        // a genuine non-text block (a figure — fine) or a degenerate
        // over-split crop the row finder cannot handle (e.g. XY-cut carving a
        // sparse page into per-glyph micro-blocks) — which would silently
        // LOSE text the whole-page reading finds. Arbitrate by total
        // recognized words: only when a block came back empty, run the
        // whole-page surface too and keep whichever recognized MORE words
        // (ties → the blocked reading, whose column order is strictly
        // better). The extra whole-page pass costs one recognition, paid only
        // in the suspicious case, never on a cleanly-split page.
        if any_block_empty || out.is_empty() {
            let whole =
                self.recognize_page_makerow_words_with_mode(grey, w, h, dict, binarize_mode)?;
            let words_of =
                |ls: &[crate::renderer::LineWords]| ls.iter().map(|l| l.words.len()).sum::<usize>();
            if words_of(&whole) > words_of(&out) {
                return Ok(whole);
            }
        }
        Ok(out)
    }

    /// **The one-shot structured-document path** — consumer-side composition
    /// (NOT a Tesseract transcode; see the `structured.rs` / `xy_cut.rs` /
    /// `pageseg.rs` / `page_furniture.rs` banners). Recognizes the grey page
    /// to word/box output, builds the `doc.v1` DOM, hardens numeric tokens,
    /// optionally harvests typed fields, classifies regions (page furniture +
    /// XY-cut layout blocks + halftone-mask figures), and renders the
    /// `tesseract-rs/doc.v1` JSON.
    ///
    /// **This is the single canonical composition** the web demo's JSON arm
    /// and the `tesseract-ogar` `recognize_document` executor arm both call,
    /// so the two consumers cannot drift.
    ///
    /// `harvest` selects the field harvest: `Some(specs)` runs
    /// [`harvest_fields`](crate::structured::harvest_fields) (e.g. with
    /// [`german_invoice_fields`](crate::structured::german_invoice_fields));
    /// `None` harvests nothing (`"fields":[]`).
    ///
    /// # Errors
    ///
    /// [`RecognizerError`] from the underlying word recognition.
    pub fn recognize_document(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        harvest: Option<&[crate::structured::FieldSpec]>,
    ) -> Result<Document, RecognizerError> {
        self.recognize_document_with_mode(grey, w, h, dict, harvest, BinarizeMode::default())
    }

    /// The same one-shot structured-document composition as
    /// [`Self::recognize_document`], with binarization mode selectable via
    /// an explicit [`BinarizeMode`](crate::xy_cut::BinarizeMode).
    /// [`Self::recognize_document`] is exactly this method called with
    /// `BinarizeMode::default()` (i.e. [`BinarizeMode::Otsu`]) — byte-identical
    /// to its pre-existing behaviour, so every existing caller of
    /// [`Self::recognize_document`] is unaffected by this addition.
    ///
    /// `binarize_mode` governs EVERY internal binarization pass this
    /// composition runs: word/line text recognition
    /// ([`Self::recognize_page_blocks_words_with_mode`], called first,
    /// below — which threads `binarize_mode` through its own layout
    /// `xy_cut` split AND the makerow line-finder's segmentation
    /// binarization), the layout-block [`xy_cut`](crate::xy_cut::xy_cut)
    /// segmentation (reading order fed to
    /// [`build_regions`](crate::structured::build_regions)), and the region
    /// classifier's own pass feeding [`Self::region_figures`] /
    /// [`Self::block_is_table`].
    ///
    /// **This closes a gap this method used to have** (measured, see
    /// `examples/binarize_ab.rs` and
    /// `.claude/harvest/sauvola-vs-otsu-probe.md`): `binarize_mode` used to
    /// reach only the layout `xy_cut` + region/table classification pass,
    /// never word/line text recognition — which ran its own separate,
    /// always-Otsu binarization internally, untouched by this parameter.
    /// `Document::word_count`, `Document::line_count`, and
    /// `Document::mean_confidence` CAN now differ between modes on the same
    /// input, in addition to the emitted `doc.v1` JSON's region/table/figure
    /// classification (which was already mode-aware).
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_document`].
    pub fn recognize_document_with_mode(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        harvest: Option<&[crate::structured::FieldSpec]>,
        binarize_mode: BinarizeMode,
    ) -> Result<Document, RecognizerError> {
        self.recognize_document_with_options(
            grey,
            w,
            h,
            dict,
            harvest,
            DocumentOptions {
                binarize_mode,
                ..DocumentOptions::default()
            },
        )
    }

    /// [`Self::recognize_document_with_mode`] plus the opt-in pre-processing
    /// switches — see [`DocumentOptions`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_document`].
    pub fn recognize_document_with_options(
        &self,
        grey: &[u8],
        w: usize,
        h: usize,
        dict: Option<&DictLite>,
        harvest: Option<&[crate::structured::FieldSpec]>,
        opts: DocumentOptions,
    ) -> Result<Document, RecognizerError> {
        let binarize_mode = opts.binarize_mode;
        // The ORIGINAL page's binarization — hoisted above recognition
        // because layout, table detection and figure detection must all see
        // the page AS PRINTED. `strip_borders` (below) changes only what the
        // RECOGNIZER sees; see `DocumentOptions::strip_borders` for why those
        // two must not be the same buffer.
        let binary = Self::binarize_page_with(grey, w, h, binarize_mode);
        let stripped = if opts.strip_borders {
            crate::pageseg::strip_borders_grey(grey, &binary, w, h)
        } else {
            None
        };
        let recog_grey: &[u8] = stripped.as_deref().unwrap_or(grey);
        // Glyph-ink measurement must read the page that was actually
        // RECOGNIZED — a char box overlapping a removed rule would otherwise
        // measure the rule's ink and report a wildly oversized glyph.
        let recog_binary: std::borrow::Cow<'_, [u8]> = if stripped.is_some() {
            std::borrow::Cow::Owned(Self::binarize_page_with(recog_grey, w, h, binarize_mode))
        } else {
            std::borrow::Cow::Borrowed(&binary)
        };

        let xy_params = crate::xy_cut::XyCutParams {
            binarize_mode,
            ..crate::xy_cut::XyCutParams::default()
        };

        // Table-aware recognition (table ingredient 3): a plain `xy_cut` over
        // `recog_grey` would fragment a table into one leaf PER COLUMN
        // (internal gutters read as ordinary layout breaks), and each column
        // then gets recognized independently — one line per printed ROW but
        // only within that one column, so `extract_table_grid`'s "rows ARE
        // the recognized lines" assumption breaks (measured: a real 7×4
        // table becomes ~28×1, column-major). `xy_cut_table_aware` vetoes
        // that fragmentation: a candidate rect that `region_is_table`
        // classifies a table (checked against `binary` — the ORIGINAL page,
        // never `recog_grey`, since a border-stripped page has no rules left
        // for the table decision to count) is kept as ONE full-width leaf.
        //
        // GATED on `opts.strip_borders`, NOT unconditional — measured
        // regression (`quality_resolution_grid.rs`): `decide_if_table`'s
        // borderless (`nvw`-only) path, the SAME fragile regime
        // `lab_table_grid.rs` already knew about, ALSO fires on an ordinary
        // WIDE multi-column TEXT layout (the resolution grid's 8 columns —
        // no table anywhere) purely because 8 separated columns produce
        // enough long whitespace corridors to clear the same threshold a
        // real table would. Running this unconditionally turned the 8×2
        // grid's ~48 per-cell lines into 6 full-width-merged ones — the
        // EXACT failure mode `recognize_page_blocks_words_with_mode`
        // (module docs, `.claude/harvest/sauvola-vs-otsu-probe.md`) exists
        // to prevent. Scoping this to `strip_borders` restores the plain,
        // proven `xy_cut` for every caller that has not opted into
        // table-specific handling — which is every existing caller of
        // `recognize_document`/`recognize_document_with_mode` — while
        // keeping the fix live for its actual, tested scenario: a caller
        // who already opted in to border-stripping for tables also wants
        // the resulting bare table treated as one region, not fragmented.
        let rec_blocks = if opts.strip_borders {
            crate::xy_cut::xy_cut_table_aware(recog_grey, w, h, &xy_params, &binary)
        } else {
            crate::xy_cut::xy_cut(recog_grey, w, h, &xy_params)
        };
        let lines = if rec_blocks.len() <= 1 {
            self.recognize_page_makerow_words_with_mode(recog_grey, w, h, dict, binarize_mode)?
        } else {
            self.recognize_blocks_words(recog_grey, w, h, dict, binarize_mode, &rec_blocks)?
        };
        let mut page =
            crate::structured::DocPage::from_line_words(&lines, &self.charset, w as u32, h as u32);
        crate::structured::harden_numeric_tokens(&mut page);
        let fields = match harvest {
            Some(specs) => crate::structured::harvest_fields(&page, specs),
            None => Vec::new(),
        };

        // Region classification over the SAME grey page: furniture
        // (header/footer), XY-cut layout blocks (reading order), and image
        // ("figure") regions from the byte-parity leptonica pixGetRegionsBinary
        // composition. All are parity-proven library layers.
        let furniture = crate::page_furniture::detect_page_furniture(&page);

        // Layout blocks — ALSO table-aware when `opts.strip_borders` is on
        // (same gate as `rec_blocks` above, same reason — see its comment),
        // and MUST agree with `rec_blocks` on where a table's bbox falls
        // whenever both run table-aware: `build_regions` assigns each
        // recognized line to a region by testing whether the line's CENTROID
        // lands inside one of THESE `blocks`. A full-width recognized table
        // line's centroid sits near the table's horizontal middle — if this
        // list still fragmented the table into per-column blocks, that
        // centroid would land in whichever single narrow column-block
        // happens to contain the page's midpoint, silently dropping the
        // other columns' worth of region-membership. Both calls share the
        // same `binary` (original page) as the table-decision reference, so
        // both make the identical table/no-table call at every candidate
        // rect — `grey` (not `recog_grey`) is used here deliberately,
        // unchanged from before this fix, since classification block
        // BOUNDARIES for non-table regions are not part of this fix's scope.
        let blocks: Vec<(i32, i32, i32, i32)> = if opts.strip_borders {
            crate::xy_cut::xy_cut_table_aware(grey, w, h, &xy_params, &binary)
        } else {
            crate::xy_cut::xy_cut(grey, w, h, &xy_params)
        }
        .into_iter()
        .map(|r| (r.left as i32, r.top as i32, r.right as i32, r.bottom as i32))
        .collect();

        // MEASURED font sizing. Replaces the statistical `xheight + ascrise -
        // descdrop` fit, which is unstable on short rows — two table rows of
        // the SAME printed size measured 24.7 vs 14.2 px (1.74x apart), a
        // visible size jump in the rendered output. `attach_glyph_px` instead
        // measures each glyph's real ink extent inside its char box against
        // this binarized page and takes the per-line p90, which held at 12-13
        // across every same-size body line of a real German document while
        // still reading genuinely small text (`Kleine Schriftgröße`) at 9.
        //
        // Runs here, after `binary` exists, and writes ONLY into lines that
        // already carry `DocLineMetrics` — it never fabricates metrics for a
        // line that had none. Additive: the existing band-derived fields are
        // untouched, and the renderer falls back to them when `glyph_px` is
        // absent.
        crate::structured::attach_glyph_px(&mut page, &lines, &recog_binary, w as u32, h as u32);

        let figures = Self::region_figures(&binary, w, h);
        // A block is a TABLE when its FULL bbox (rules + column corridors, not
        // just the text-line union the region carries) clears decide_if_table.
        //
        // `require_ruled = !opts.strip_borders` — the SAME opt-in rule that
        // gates the two `xy_cut_table_aware` calls above, applied to the
        // LABELLING half of table handling, which that fix left ungated. A
        // caller who has not asked for table-specific handling must not have
        // ordinary multi-column prose stamped `type=table` by the
        // whitespace-only path (measured: 69 of 72 lines of a real two-column
        // scan) — see `Self::block_is_table` for the full mechanism and why
        // this narrows the EVIDENCE rather than switching detection off.
        let table_blocks: Vec<bool> = blocks
            .iter()
            .map(|&blk| Self::block_is_table(&binary, w, h, blk, !opts.strip_borders))
            .collect();
        let regions = crate::structured::build_regions(
            &page,
            &furniture.header_lines,
            &furniture.footer_lines,
            &blocks,
            &table_blocks,
            &figures,
        );
        let json = crate::structured::render_json_with_regions(&page, &regions, &fields);
        let word_count = page.lines.iter().map(|l| l.words.len()).sum();
        let line_count = page.lines.len();
        let mean_confidence = crate::structured::mean_word_confidence(&page);
        let low_confidence =
            mean_confidence.is_some_and(|mc| mc < crate::structured::LOW_CONFIDENCE_THRESHOLD);
        let drop_caps = crate::dropcap::count_page_drop_caps(&binary, w, h);
        Ok(Document {
            json,
            fields,
            word_count,
            line_count,
            mean_confidence,
            low_confidence,
            drop_caps,
        })
    }

    /// Image-region ("figure") bboxes for [`Self::recognize_document`]'s region
    /// classification — the byte-parity leptonica `pixGetRegionsBinary`
    /// composition ([`get_regions_binary`](crate::pageseg::get_regions_binary)).
    /// Binarize (Otsu, with the same fixed-128 fallback the library segmenters
    /// use when Otsu declines), run the composition, and return the halftone
    /// (image) mask's 8-connected components as page-space `(l, t, r, b)` rects.
    /// Empty when the page is too small or holds no image region.
    ///
    /// This runs the "is it a picture?" half of the classifier through the REAL
    /// leptonica leaf — the 2×-reduce → seed → seedfill-fill-back → expand chain
    /// — rather than the previous full-resolution `generate_halftone_mask`
    /// approximation. Text-block reading order stays with
    /// [`xy_cut`](crate::xy_cut::xy_cut); the textblock mask
    /// ([`Regions::textblock`](crate::pageseg::Regions::textblock)) is the
    /// pixel-level text-region witness the same composition also yields.
    ///
    /// Takes the already-binarized page ([`Self::binarize_page_with`]) — the
    /// caller binarizes once and shares it with the table pass.
    fn region_figures(binary: &[u8], w: usize, h: usize) -> Vec<(i32, i32, i32, i32)> {
        if w < crate::pageseg::MIN_WIDTH || h < crate::pageseg::MIN_HEIGHT {
            return Vec::new();
        }
        let Some(regions) = crate::pageseg::get_regions_binary(binary, w, h) else {
            return Vec::new();
        };
        if !regions.halftone.contains(&0) {
            return Vec::new(); // no image region on the page
        }
        crate::conncomp::conn_comp_bb(&regions.halftone, regions.halftone_w, regions.halftone_h, 8)
            .into_iter()
            .map(|b| (b.x, b.y, b.x + b.w, b.y + b.h))
            .collect()
    }

    /// Binarize the page once → this crate's 1bpp `0` = ON, under an
    /// explicit [`BinarizeMode`]. Shared by the region classifier's figure
    /// ([`Self::region_figures`]) and table ([`Self::block_is_table`]) paths
    /// — the caller binarizes once. Delegates to
    /// [`crate::xy_cut::binarize_page_with`] rather than re-implementing the
    /// Otsu-with-fixed-128-fallback logic a second time in this file;
    /// [`BinarizeMode::Otsu`] (the default — see
    /// [`Self::recognize_document`]) reproduces this method's pre-existing
    /// (pre-[`BinarizeMode`]) Otsu-only body byte-for-byte.
    fn binarize_page_with(grey: &[u8], w: usize, h: usize, mode: BinarizeMode) -> Vec<u8> {
        crate::xy_cut::binarize_page_with(grey, w, h, mode)
    }

    /// Whether an XY-cut layout `block` is a TABLE — the byte-parity leptonica
    /// `decide_if_table` ([`crate::pageseg::decide_if_table`]) over the block's
    /// FULL bbox, cropped from the binarized page. A score ≥
    /// [`TABLE_SCORE_THRESHOLD`](crate::pageseg::TABLE_SCORE_THRESHOLD) marks a
    /// table.
    ///
    /// **Cropping the block, not the emitted text-region bbox, is the point.**
    /// The region bbox `build_regions` produces is the union of the OCR line
    /// boxes; it excludes the borders, borders, and empty column corridors that
    /// live *between/around* the text — exactly the structure
    /// `decide_if_table` counts. Feeding it the text-line union would strip that
    /// signal and miss ruled tables (per the #39 review). The block bbox keeps
    /// the whole layout region intact.
    ///
    /// **Scope note:** `decide_if_table`'s brick sizes target leptonica's
    /// ~75-300 ppi structural-line scale; the `pixPrepare1bpp` ppi-normalization
    /// plus `pixDeskewBoth` front-end is deferred to the deskew wave, so this
    /// runs on the block crop at the page's own resolution — robust for typical
    /// document scales (table rules span whole columns, text words are short),
    /// but not yet ppi-exact. Blocks under 100 px in either dimension can hold
    /// no `o100` structural line, so they score 0 and are skipped.
    ///
    /// # `require_ruled` — the false-positive gate on the default path
    ///
    /// When `true`, a block must ALSO carry ruled evidence
    /// ([`TableDecision::has_ruled_evidence`](crate::pageseg::TableDecision::has_ruled_evidence):
    /// `nhb > 1` or `nvb > 2`) before it is called a table, on top of clearing
    /// the score threshold.
    ///
    /// **Why, measured.** `decide_if_table` is byte-parity with leptonica and
    /// stays untouched, but two of its four conditions count only VERTICAL
    /// WHITESPACE, and that pair cannot discriminate a table from wide
    /// multi-column prose — both genuinely have many long corridors. Measured
    /// on a real 2550×3300 two-column scan through the plain
    /// [`Self::recognize_document`] path: **2 of 5 regions and 69 of 72 lines
    /// of ordinary flowing prose were stamped `type=table`**, because each
    /// tall column block scores on `nvw` alone (`nhb = nvb = 0` — there is not
    /// a single printed rule on the page). `corpus/quality/resgrid.pgm` is the
    /// same shape at unit scale: 8 columns of ORDINARY TEXT, zero rules,
    /// `nvw=17 score=2`.
    ///
    /// The ruled pair is the RELIABLE half — a block holding 2+ horizontal or
    /// 3+ vertical `o100` structural lines really is tabular — so requiring it
    /// kills the prose false positive **without disabling table detection**,
    /// which is why this is a discrimination gate and not an on/off switch.
    /// Turning classification off wholesale would also have broken
    /// `tests/lab_table_columns.rs::naive_pre_strip_destroys_table_detection`,
    /// whose precondition (`!plain_shapes.is_empty()`) legitimately depends on
    /// a genuinely ruled fixture still being detected on the default path.
    ///
    /// `false` restores the bare `score >= TABLE_SCORE_THRESHOLD` verdict, and
    /// is used when the caller has ALREADY signalled table intent via
    /// [`DocumentOptions::strip_borders`] — the same "opt in before the
    /// unreliable path is trusted" rule that gates `xy_cut_table_aware`. Note
    /// that a `strip_borders` caller NEEDS the whitespace-only path: stripping
    /// removes the very rules the ruled conditions count (see
    /// `naive_pre_strip_destroys_table_detection`), so requiring ruled evidence
    /// there would defeat the feature.
    fn block_is_table(
        binary: &[u8],
        w: usize,
        h: usize,
        block: (i32, i32, i32, i32),
        require_ruled: bool,
    ) -> bool {
        match crate::pageseg::region_table_decision(binary, w, h, block) {
            Some(d) => {
                d.score >= crate::pageseg::TABLE_SCORE_THRESHOLD
                    && (!require_ruled || d.has_ruled_evidence())
            }
            None => false,
        }
    }

    /// `LSTMRecognizer::SetRandomSeed` (`lstmrecognizer.h:287-291`): the exact
    /// randomizer seeding `RecognizeLine` uses before the forward pass —
    /// `seed = (i64)sample_iteration · 0x10000001`, `minstd` seed, one warm-up
    /// draw. Reproducing it makes [`recognize_image_file`] bit-match real
    /// libtesseract (not just "correct for an arbitrary seed"): the `Convolve`
    /// out-of-image noise depends on this seed.
    ///
    /// [`recognize_image_file`]: LstmRecognizer::recognize_image_file
    #[must_use]
    fn seeded_randomizer(&self) -> TRand {
        let seed = i64::from(self.sample_iteration).wrapping_mul(0x1000_0001) as u64;
        let mut rng = TRand::default();
        rng.set_seed(seed);
        rng.int_rand(); // the warm-up draw
        rng
    }

    /// **A6b — image FILE on disk → text.** The full pure-Rust
    /// `RecognizeLine`-equivalent (`lstmrecognizer.cpp:321-360`): read a P5 PGM →
    /// pre-scale to the network input height (A6b) → `from_grey_pix` (A6a) →
    /// `recognize_grid` (B3-core), seeding the randomizer exactly as
    /// `RecognizeLine` does ([`seeded_randomizer`]). Returns `(unichar_ids, text)`.
    ///
    /// **Byte-parity vs libtesseract holds when the image is at the model input
    /// height** (leptonica `pixScale` at factor 1.0 is a copy, so the scale is
    /// identity and every downstream step is proven). Other heights use the
    /// marked bilinear approximation in
    /// [`prescale_grey_to_height`](crate::image_input::prescale_grey_to_height)
    /// (functional, NOT leptonica-`pixScale`-exact).
    ///
    /// # Errors
    ///
    /// [`RecognizerError::Io`] / [`RecognizerError::Pgm`] on a bad image;
    /// [`RecognizerError::Network`] on a forward failure.
    ///
    /// [`seeded_randomizer`]: LstmRecognizer::seeded_randomizer
    pub fn recognize_image_file(&self, path: &Path) -> Result<(Vec<i32>, String), RecognizerError> {
        let bytes = std::fs::read(path).map_err(RecognizerError::Io)?;
        let (grey, w, h) = parse_pgm(&bytes).map_err(RecognizerError::Pgm)?;
        self.recognize_grey_line(&grey, w, h, None)
    }

    /// The dict-enabled counterpart of [`Self::recognize_image_file`] (D1.3):
    /// same P5-PGM read → pre-scale → `from_grey_pix` pipeline, but decodes via
    /// [`Self::recognize_grid_with_dict`]. See that method for the dict-path
    /// constants; see [`Self::recognize_image_file`] for the byte-parity scope
    /// (model-input-height images only — other heights use the marked
    /// approximation in [`prescale_grey_to_height`](crate::image_input::prescale_grey_to_height)).
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_image_file`].
    pub fn recognize_image_file_with_dict(
        &self,
        path: &Path,
        dict: DictLite,
    ) -> Result<(Vec<i32>, String), RecognizerError> {
        let bytes = std::fs::read(path).map_err(RecognizerError::Io)?;
        let (grey, w, h) = parse_pgm(&bytes).map_err(RecognizerError::Pgm)?;
        self.recognize_grey_line(&grey, w, h, Some(dict))
    }

    /// **The word/box output surface** — the counterpart of
    /// [`Self::recognize_image_file`] / [`Self::recognize_image_file_with_dict`]
    /// that returns [`WordResult`]s (`RecodeBeamSearch::extract_best_path_as_words`,
    /// `recodebeam.cpp:239-322`) instead of a flat unichar-id run. Same P5-PGM
    /// read → pre-scale → `from_grey_pix` → `network.forward` pipeline as the
    /// other `recognize_image_file*` methods; `dict` selects the beam variant
    /// exactly as [`Self::recognize_grid`]/[`Self::recognize_grid_with_dict`]
    /// do (`None` → the plain `TOP_CHOICE_PERM`-only beam; `Some` → the
    /// production `kDictRatio`/`kCertOffset`/`worst_dict_cert` dict beam).
    ///
    /// `line_box` is `(left, bottom, right, top)` — `TBOX`'s constructor
    /// argument order. `scale_factor` un-does any `pixScale` pre-processing so
    /// boxes land in the ORIGINAL image's pixel space (`1.0` for a model-height
    /// image, matching [`Self::recognize_image_file`]'s byte-parity scope).
    ///
    /// **The caller supplies the IMAGE factor only.** The network's own
    /// horizontal reduction ([`Network::x_scale_factor`], eng.lstm: 3 from
    /// `Mp3,3`) is folded in here, mirroring the C++ split: `PrepareLSTMInputs`
    /// yields the image's `im_factor` and `RecognizeLine` then applies
    /// `*scale_factor = XScaleFactor() / im_factor` (`lstmrecognizer.cpp:344`,
    /// "Reduction factor from image to coords"). Callers cannot reasonably be
    /// expected to know the loaded model's pooling, and every in-repo caller
    /// previously passed a bare `1.0` — which silently compressed all word
    /// boxes into the left third of each line.
    ///
    /// # Errors
    ///
    /// Same as [`Self::recognize_image_file`].
    pub fn recognize_image_file_words(
        &self,
        path: &Path,
        dict: Option<DictLite>,
        line_box: (i32, i32, i32, i32),
        scale_factor: f32,
    ) -> Result<Vec<WordResult>, RecognizerError> {
        let bytes = std::fs::read(path).map_err(RecognizerError::Io)?;
        let (grey, w, h) = parse_pgm(&bytes).map_err(RecognizerError::Pgm)?;
        // Same min-size gate as recognize_grey_line: a line too small to scale
        // yields no words (RecognizeLine skips it) rather than walking Maxpool
        // off a degenerate grid.
        let Some((grid, mut rng)) = self.prepare_grid(&grey, w, h) else {
            return Ok(Vec::new());
        };
        let outputs = self.network.forward(&grid, &mut rng)?;
        if outputs.int_mode() {
            return Err(RecognizerError::Network(NetError::Forward(
                "recognize_image_file_words expects softmax float logits (int-mode output)",
            )));
        }
        let simple = self.network.simple_text_output();
        let rows: Vec<&[f32]> = (0..outputs.width()).map(|t| outputs.f(t)).collect();
        // Fold in the network's horizontal reduction (see the doc comment):
        // the beam's character_boundaries are decoder TIMESTEP indices.
        let scale_factor = scale_factor * self.network.x_scale_factor().max(1) as f32;
        let words = if let Some(dict) = dict {
            let mut beam = RecodeBeamSearch::new_with_dict(
                &self.recoder,
                self.null_char,
                simple,
                dict,
                self.charset.clone(),
            );
            beam.decode_with_dict(&rows, K_DICT_RATIO, K_CERT_OFFSET, K_WORST_DICT_CERT);
            beam.extract_best_path_as_words(line_box, scale_factor, &self.charset)
        } else {
            let mut beam = RecodeBeamSearch::new(&self.recoder, self.null_char, simple);
            beam.decode(&rows, 1.0, 0.0);
            beam.extract_best_path_as_words(line_box, scale_factor, &self.charset)
        };
        Ok(words)
    }

    /// Assemble from the three split `.traineddata` components (the
    /// `include_charsets == false` path): the `lstm` component bytes (network +
    /// trailing scalars), the `lstm-unicharset` TEXT, and the `lstm-recoder`
    /// binary bytes.
    ///
    /// # Errors
    ///
    /// [`RecognizerError`] if the network/tail parse fails, or either component
    /// fails to load.
    pub fn from_components(
        lstm: &[u8],
        unicharset_text: &str,
        recoder: &[u8],
    ) -> Result<Self, RecognizerError> {
        let (network, consumed) = Network::from_le_bytes(lstm)?;
        let mut tail = TailReader {
            bytes: lstm,
            pos: consumed,
        };
        let network_str = tail.string()?;
        let training_flags = tail.i32()?;
        let training_iteration = tail.i32()?;
        let sample_iteration = tail.i32()?;
        let null_char = tail.i32()?;
        let adam_beta = tail.f32()?;
        let learning_rate = tail.f32()?;
        let momentum = tail.f32()?;

        let charset =
            UniCharSet::load_from_str(unicharset_text).map_err(RecognizerError::Charset)?;
        let recoder = UnicharCompress::from_le_bytes(recoder).map_err(RecognizerError::Recoder)?;

        Ok(Self {
            network,
            network_str,
            training_flags,
            training_iteration,
            sample_iteration,
            null_char,
            adam_beta,
            learning_rate,
            momentum,
            charset,
            recoder,
        })
    }
}

/// One recognizable row emitted by [`LstmRecognizer::makerow_row_crops`]: the
/// padded typographic crop (raster grey, `band_h × band_w`) plus that row's
/// bottom-up `TBOX` line box in PAGE space `(left, bottom, right, top)`. The
/// shared unit consumed by both `recognize_page_makerow` (string) and
/// `recognize_page_makerow_words` (words) so the two feed identical crops.
struct MakerowRowCrop {
    /// Raster-order grey crop, `band_h` rows of `band_w` bytes.
    crop: Vec<u8>,
    /// Crop width in original page pixels.
    band_w: usize,
    /// Crop height in original page pixels.
    band_h: usize,
    /// The crop rectangle in bottom-up `TBOX` PAGE space `(left, bottom, right,
    /// top)` — the `line_box` passed to `extract_best_path_as_words`.
    line_box: (i32, i32, i32, i32),
    /// The row's typographic metrics (`TO_ROW` xheight/ascrise/descdrop from
    /// wave 3 + the fitted mid-line baseline) — the same numbers the band
    /// extension above the crop was computed FROM, preserved instead of
    /// discarded so renderers can size text from measurement rather than
    /// guessing from the (deliberately generous) band height. Bottom-up PAGE
    /// space, matching `line_box`.
    metrics: crate::renderer::LineMetrics,
}

/// Reads the lstm component's trailing scalar fields (`TFile` LE encoding,
/// starting where `Network::from_le_bytes` stopped).
struct TailReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl TailReader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], NetError> {
        let end = self.pos.checked_add(n).ok_or(NetError::UnexpectedEof)?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or(NetError::UnexpectedEof)?;
        self.pos = end;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32, NetError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, NetError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// A `TFile` `std::string`: `u32 len` then `len` raw bytes.
    fn string(&mut self) -> Result<String, NetError> {
        let len = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trailing-field parse on hand-built bytes: an empty "network" is not
    /// valid, so test the TailReader-shaped parse via a minimal synthetic
    /// component whose tail matches the real eng.lstm field layout. (Full
    /// real-file parity is the `lstm_recognizer_dump` example vs the oracle.)
    #[test]
    fn tail_reader_reads_the_field_block() {
        // network_str "AB" + 4 i32 + 3 f32.
        let mut b = Vec::new();
        b.extend_from_slice(&2_u32.to_le_bytes());
        b.extend_from_slice(b"AB");
        for v in [65_i32, 100, 200, 110] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.999_f32, 0.001, 0.5] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let mut r = TailReader { bytes: &b, pos: 0 };
        assert_eq!(r.string().unwrap(), "AB");
        assert_eq!(r.i32().unwrap(), 65);
        assert_eq!(r.i32().unwrap(), 100);
        assert_eq!(r.i32().unwrap(), 200);
        assert_eq!(r.i32().unwrap(), 110);
        assert!((r.f32().unwrap() - 0.999).abs() < 1e-6);
        assert!((r.f32().unwrap() - 0.001).abs() < 1e-6);
        assert!((r.f32().unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(r.pos, b.len(), "consumes the whole field block");
    }

    /// `is_recoding` / `is_int_mode` read the flag bits (eng training_flags=65 =
    /// TF_INT_MODE | TF_COMPRESS_UNICHARSET).
    #[test]
    fn flag_predicates() {
        // Build a minimal recognizer by hand is awkward (needs a real network);
        // test the bit logic directly against the eng flag value.
        assert_eq!(65 & TF_COMPRESS_UNICHARSET, 64, "eng recodes");
        assert_eq!(65 & 1, 1, "eng is int-mode");
        // TF_INT_MODE(1) only, no TF_COMPRESS_UNICHARSET(64) → pass-through codec.
        assert_eq!(
            1 & TF_COMPRESS_UNICHARSET,
            0,
            "int-mode-only model doesn't recode"
        );
    }
}

#[cfg(test)]
mod makerow_page_tests {
    use super::*;

    /// 3F₂/feeding E2E anchor on the stacked-line synthetic (hermetic: reads
    /// the committed `corpus/` fixtures; regenerate with
    /// `.claude/harvest/oracles/gen_page_fixture.py`): the REAL makerow line
    /// finder must segment the two stacked copies into exactly two rows, and
    /// — because the typographic feeding (`linerec.cpp:239-246` band +
    /// `GetRectImage` pad-4) is position-invariant when nothing clips at the
    /// image edges — the roomy fixture's two rows must recognize to
    /// IDENTICAL text. (On the legacy tight 24×88 layout
    /// (`corpus/lines/page_tight.pgm`) the padded band clips at the top edge
    /// for row A and the bottom edge for row B — faithful `GetRectImage`
    /// clipping — so the lines legitimately differ there; that layout is not
    /// asserted.)
    #[test]
    fn stacked_page_finds_two_deterministic_rows() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let a = r.recognize_page_makerow(&grey, w, h, None).unwrap();
        let b = r.recognize_page_makerow(&grey, w, h, None).unwrap();
        assert_eq!(a, b, "must be deterministic");
        let lines: Vec<&str> = a.split('\n').collect();
        assert_eq!(lines.len(), 2, "two stacked lines -> two rows: {a:?}");
        assert!(lines.iter().all(|l| !l.is_empty()));
        // Position invariance: identical ink + identical typographic band
        // (unclipped, the committed roomy layout) => identical crops =>
        // identical text.
        assert_eq!(
            lines[0], lines[1],
            "roomy fixture: typographic feeding must be position-invariant"
        );
    }

    /// The min-size gate must fire on the actual (floored) prescaled width, not
    /// a rounded estimate. Codex P2: an odd source width at an EXACT 2⁻ⁿ halving
    /// scales through `scale_gray_area_map2`, whose width is FLOORED — a 5×72
    /// eng-model line prescales to width `floor(5/2) = 2`, below `XScaleFactor`
    /// (3, from `Mp3,3`), so the line must be skipped (empty). A `round(5·36/72)
    /// = 3` estimate would wrongly pass it and walk `Maxpool` off the width-2
    /// grid. This drives the exact geometry the guard exists for; it must NOT
    /// panic and must return empty.
    #[test]
    fn odd_width_exact_halving_below_min_width_is_skipped_not_crashed() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        // 5 wide × 72 tall → target 36 → factor 0.5 (exact halving) → width
        // floor(5/2) = 2 < XScaleFactor 3. A mid-grey strip; content is
        // irrelevant, the geometry is the point.
        let (w, h) = (5usize, 72usize);
        let grey = vec![128u8; w * h];
        let (ids, text) = r
            .recognize_grey_line(&grey, w, h, None)
            .expect("too-small line returns Ok, not a panic/error");
        assert!(
            ids.is_empty() && text.is_empty(),
            "too-small line yields nothing"
        );
    }

    /// The word page surface must report the SAME set of recognized lines as the
    /// string surface (both run [`LstmRecognizer::makerow_row_crops`], so a row
    /// that produces text must produce ≥1 word and vice-versa), and every word's
    /// char box must sit inside its row's `line_box` — the second precision
    /// point. Char boxes come back in bottom-up PAGE space `(left, bottom,
    /// right, top)` from `extract_best_path_as_words`; `line_box` bottom/top were
    /// derived from the raster crop via `h` (`bottom = h - img_bottom`,
    /// `top = h - img_top`). Allow the `kImagePadding = 4` slack on all sides.
    /// Word boxes must SPAN their line, not merely sit inside it.
    ///
    /// The sibling test above only asserts char boxes are *within* `line_box`
    /// — which stayed trivially true while every box was compressed toward
    /// the line's left edge by the network's `XScaleFactor()` (eng.lstm: 3,
    /// from `Mp3,3`), because the beam's `character_boundaries` are decoder
    /// TIMESTEP indices and `scale_factor` originally carried only the image
    /// prescale (`lstmrecognizer.cpp:344` folds in `XScaleFactor()` too:
    /// "Reduction factor from image to coords"). That shipped: on a wide page
    /// word boxes covered exactly 33.3% of the content width, so the
    /// searchable-PDF text layer bunched into the left third instead of
    /// stretching across the page. Recognized TEXT was unaffected, which is
    /// why nothing caught it. This test pins the absolute-position invariant
    /// the other one cannot see.
    #[test]
    fn word_boxes_span_the_line_not_just_its_left_third() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        // Graceful skip when the (gitignored) model / fixtures are absent.
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let lines = r.recognize_page_makerow_words(&grey, w, h, None).unwrap();
        let non_empty: Vec<_> = lines.iter().filter(|l| !l.words.is_empty()).collect();
        assert!(!non_empty.is_empty(), "expected recognized lines");

        // For each line, the rightmost char box must reach most of the way to
        // the line box's right edge. Under the 3x compression this ratio sat
        // near 1/3; a correct mapping puts it near 1.0.
        let mut worst = f32::MAX;
        for line in &non_empty {
            let (lb_left, _, lb_right, _) = line.line_box;
            let span = (lb_right - lb_left) as f32;
            if span <= 0.0 {
                continue;
            }
            let max_r = line
                .words
                .iter()
                .flat_map(|word| word.char_boxes.iter().map(|b| b.2))
                .max()
                .unwrap_or(lb_left);
            worst = worst.min((max_r - lb_left) as f32 / span);
        }
        assert!(
            worst > 0.6,
            "rightmost word box only reaches {:.1}% of the line width \
             (a 3x-compressed mapping lands near 33%)",
            worst * 100.0
        );
    }

    #[test]
    fn words_page_matches_string_lines_and_boxes_within_line_box() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        // Graceful skip when the (gitignored) model / fixtures are absent.
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let string_out = r.recognize_page_makerow(&grey, w, h, None).unwrap();
        let string_lines = string_out.split('\n').count();
        let words = r.recognize_page_makerow_words(&grey, w, h, None).unwrap();

        assert_eq!(
            words.len(),
            string_lines,
            "word surface must report the same non-empty line count as the string surface: \
             string={string_out:?} words={} lines",
            words.len()
        );

        const PAD: i32 = 4;
        for line in &words {
            let (lb_left, lb_bottom, lb_right, lb_top) = line.line_box;
            for word in &line.words {
                for &(cl, cb, cr, ct) in &word.char_boxes {
                    // Char boxes carry line_box's own bottom/top verbatim
                    // (extract_best_path_as_words stamps them), so the vertical
                    // extent is exact; x is offset by line_box.left and must land
                    // within [left - PAD, right + PAD].
                    assert_eq!(cb, lb_bottom, "char box bottom = line_box bottom");
                    assert_eq!(ct, lb_top, "char box top = line_box top");
                    assert!(
                        cl >= lb_left - PAD && cr <= lb_right + PAD,
                        "char x [{cl},{cr}] must lie within line_box x \
                         [{lb_left},{lb_right}] (±{PAD})"
                    );
                }
            }
        }
    }

    /// `render_text` over the word surface must reconstruct the string surface's
    /// page text (bar the renderer's trailing `line_separator_` newline, hence
    /// `trim_end`). Both walk the same beam output per row; the word split drops
    /// the `UNICHAR_SPACE` separators and `render_text` re-inserts them from
    /// [`WordResult::leading_space`], so the per-line concatenation equals
    /// `ids_to_text` over the row's full id run — the string path's text.
    #[test]
    fn render_text_of_words_equals_string_surface() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let string_out = r.recognize_page_makerow(&grey, w, h, None).unwrap();
        let words = r.recognize_page_makerow_words(&grey, w, h, None).unwrap();
        let rendered = crate::renderer::render_text(&words, &r.charset);
        assert_eq!(
            rendered.trim_end(),
            string_out,
            "render_text(words).trim_end() must equal the string surface"
        );
    }

    /// `recognize_document` composes the word surface into `doc.v1` JSON: the
    /// canonical one-shot both the web demo and the tesseract-ogar executor
    /// call. It must produce well-formed doc.v1, report the same line count as
    /// the word surface, and — with the German-invoice harvest profile — carry
    /// a `fields` array (possibly empty on a non-invoice fixture, but the key
    /// is always present). The no-harvest arm must emit `"fields":[]`.
    #[test]
    fn recognize_document_composes_docv1_and_counts() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let words = r.recognize_page_makerow_words(&grey, w, h, None).unwrap();

        // No-harvest arm: valid doc.v1, empty fields, counts consistent.
        let plain = r.recognize_document(&grey, w, h, None, None).unwrap();
        assert!(plain
            .json
            .starts_with("{\"schema\":\"tesseract-rs/doc.v1\""));
        assert!(
            plain.json.contains("\"fields\":[]"),
            "no-harvest → empty fields"
        );
        assert_eq!(
            plain.line_count,
            words.len(),
            "line count matches word surface"
        );
        assert_eq!(
            plain.word_count,
            words.iter().map(|l| l.words.len()).sum::<usize>()
        );

        // Harvest arm: the fields key is present (array may be empty on a
        // non-invoice fixture, but the harvest ran without panicking).
        let specs = crate::structured::german_invoice_fields();
        let harvested = r
            .recognize_document(&grey, w, h, None, Some(&specs))
            .unwrap();
        assert!(harvested.json.contains("\"fields\":"));
        assert_eq!(harvested.line_count, words.len());
    }

    /// **The default-preservation invariant.** `recognize_document` must
    /// produce byte-identical `Document`s to
    /// `recognize_document_with_mode(.., BinarizeMode::default())` — the
    /// whole point of the `_with_mode` sibling pattern is that adding the
    /// new parameter never changes a single existing caller's behaviour.
    #[test]
    fn recognize_document_matches_with_mode_default() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let via_plain = r.recognize_document(&grey, w, h, None, None).unwrap();
        let via_with_mode = r
            .recognize_document_with_mode(&grey, w, h, None, None, BinarizeMode::default())
            .unwrap();
        assert_eq!(via_plain.json, via_with_mode.json, "json must match");
        assert_eq!(via_plain.word_count, via_with_mode.word_count);
        assert_eq!(via_plain.line_count, via_with_mode.line_count);
        assert_eq!(via_plain.mean_confidence, via_with_mode.mean_confidence);
        assert_eq!(via_plain.low_confidence, via_with_mode.low_confidence);
    }

    /// **Mode-threading safety net for the NEW `_with_mode` siblings added
    /// alongside `crate::segment::segment_rows_with_mode` /
    /// `segment_rows_independent_with_mode`.** Mirrors
    /// [`recognize_document_matches_with_mode_default`] (the pre-existing
    /// invariant for `recognize_document`/`recognize_document_with_mode`),
    /// extended to the three new pairs this pass introduces:
    /// `recognize_page_makerow`/`_with_mode`,
    /// `recognize_page_makerow_words`/`_with_mode`, and
    /// `recognize_page_blocks_words`/`_with_mode`. Each `_with_mode` sibling
    /// called with `BinarizeMode::default()` MUST reproduce its plain
    /// counterpart byte-for-byte — the regression this pass exists to
    /// prevent is an accidental default-behaviour change silently re-pinning
    /// every golden anchor and the 8+7+0 CER fence, both of which run
    /// through `crate::segment::segment_rows_with_mode` underneath these methods.
    #[test]
    fn new_with_mode_siblings_match_their_plain_counterparts_at_default() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let string_plain = r.recognize_page_makerow(&grey, w, h, None).unwrap();
        let string_with_mode = r
            .recognize_page_makerow_with_mode(&grey, w, h, None, BinarizeMode::default())
            .unwrap();
        assert_eq!(
            string_plain, string_with_mode,
            "recognize_page_makerow(default) must match recognize_page_makerow_with_mode(default)"
        );

        let words_plain = r.recognize_page_makerow_words(&grey, w, h, None).unwrap();
        let words_with_mode = r
            .recognize_page_makerow_words_with_mode(&grey, w, h, None, BinarizeMode::default())
            .unwrap();
        assert_eq!(
            words_plain, words_with_mode,
            "recognize_page_makerow_words(default) must match _with_mode(default)"
        );

        let blocks_plain = r.recognize_page_blocks_words(&grey, w, h, None).unwrap();
        let blocks_with_mode = r
            .recognize_page_blocks_words_with_mode(&grey, w, h, None, BinarizeMode::default())
            .unwrap();
        assert_eq!(
            blocks_plain, blocks_with_mode,
            "recognize_page_blocks_words(default) must match _with_mode(default)"
        );
    }

    /// `recognize_document_with_mode` must accept the non-default
    /// [`BinarizeMode::Sauvola`] end-to-end without panicking and still
    /// produce well-formed `doc.v1` JSON.
    ///
    /// **This test previously asserted `otsu.word_count == sauvola.word_count`
    /// as a "regression guard" — that assertion encoded the very gap this
    /// change fixes.** `binarize_mode` used to reach only region/table
    /// classification, never word/line text recognition, so the two modes
    /// were guaranteed to agree on `word_count`/`mean_confidence` regardless
    /// of input. Now that `binarize_mode` also governs the makerow line
    /// finder's own segmentation binarization
    /// (`crate::segment::segment_rows_with_mode`), Otsu and Sauvola CAN
    /// legitimately disagree on word/line output — measuring whether/how
    /// much they actually do on a given input is exactly what
    /// `examples/binarize_ab.rs` is for (see
    /// `.claude/harvest/sauvola-vs-otsu-probe.md`'s "clean" row for a case
    /// where the two still agree closely on an evenly-lit page). Pinning
    /// numeric equality here would just re-assert the old bug, so this test
    /// now only guards against a panic or malformed-output regression when
    /// Sauvola is selected — not equality with Otsu.
    #[test]
    fn recognize_document_with_mode_sauvola_is_well_formed() {
        let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        if !corpus.join("model/eng.lstm").exists() || !corpus.join("lines/page_roomy.pgm").exists()
        {
            return;
        }
        let lstm = std::fs::read(corpus.join("model/eng.lstm")).unwrap();
        let uni = std::fs::read_to_string(corpus.join("model/eng.lstm-unicharset")).unwrap();
        let rec = std::fs::read(corpus.join("model/eng.lstm-recoder")).unwrap();
        let img = std::fs::read(corpus.join("lines/page_roomy.pgm")).unwrap();
        let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
        let (grey, w, h) = crate::image_input::parse_pgm(&img).unwrap();

        let otsu = r
            .recognize_document_with_mode(&grey, w, h, None, None, BinarizeMode::Otsu)
            .unwrap();
        let sauvola = r
            .recognize_document_with_mode(
                &grey,
                w,
                h,
                None,
                None,
                BinarizeMode::Sauvola {
                    whsize: 16,
                    k: 0.34,
                },
            )
            .unwrap();
        assert!(sauvola
            .json
            .starts_with("{\"schema\":\"tesseract-rs/doc.v1\""));
        // A sanity floor, not an equality pin: Sauvola must still recognize
        // *something* on a fixture Otsu recognizes cleanly, even if the
        // exact count now legitimately differs.
        assert!(
            otsu.word_count > 0,
            "fixture sanity: Otsu must find words on page_roomy.pgm"
        );
        assert!(
            sauvola.word_count > 0,
            "Sauvola must still recognize words on the same fixture, not silently zero"
        );
    }

    /// The region-classifier figure path (`region_figures`) is corpus-free —
    /// it takes a grey page and runs the byte-parity `get_regions_binary`
    /// composition. A page with a solid image block + text columns must yield
    /// exactly one image ("figure") region boxing the block, proving the
    /// recognize_document wiring consumes the real leptonica leaf (not the old
    /// full-res halftone approximation).
    #[test]
    fn region_figures_boxes_the_image_block() {
        let (w, h) = (320usize, 280usize);
        let mut grey = vec![255u8; w * h];
        // Solid dark 100×80 image block.
        for y in 30..110 {
            for x in 30..130 {
                grey[y * w + x] = 0;
            }
        }
        // Two text columns (thin stripes) — must NOT be classified as image.
        for c0 in [160usize, 250] {
            let mut yb = 20;
            while yb + 5 <= 260 {
                for y in yb..yb + 5 {
                    for x in c0..c0 + 60 {
                        if (x - c0) % 24 < 18 {
                            grey[y * w + x] = 0;
                        }
                    }
                }
                yb += 12;
            }
        }

        let figures = LstmRecognizer::region_figures(&grey, w, h);
        assert_eq!(figures.len(), 1, "exactly the one solid image block");
        let (l, t, r, b) = figures[0];
        // The seedfill-fill-back recovers the full 100×80 block; the bbox hugs
        // its extent (±2 for the 2×-reduction flooring).
        assert!(
            (28..=32).contains(&l) && (28..=32).contains(&t),
            "origin {l},{t}"
        );
        assert!(
            (128..=132).contains(&r) && (108..=112).contains(&b),
            "extent {r},{b}"
        );
    }

    /// Table classification (`block_is_table`) is corpus-free and decides on
    /// the BLOCK bbox: a ruled grid block flips to a table (via the byte-parity
    /// `decide_if_table`) while a plain-paragraph block does not, and a block
    /// under the 100 px structural-scale floor is skipped. Cropping the block —
    /// not the text-line union — is what keeps the borders and column corridors
    /// the decision keys on (the #39 fix).
    #[test]
    fn block_is_table_detects_grid_not_paragraph() {
        let (w, h) = (480usize, 280usize);
        let mut grey = vec![255u8; w * h];
        // Ruled grid (left block): 4 horizontal + 4 vertical black lines.
        for &r in &[20usize, 90, 160, 230] {
            for y in r..r + 2 {
                for x in 20..220 {
                    grey[y * w + x] = 0;
                }
            }
        }
        for &c in &[20usize, 90, 160, 220] {
            for x in c..c + 2 {
                for y in 20..232 {
                    grey[y * w + x] = 0;
                }
            }
        }
        // Plain paragraph (right block): char stripes, no rules.
        let mut yb = 20;
        while yb + 5 <= 260 {
            for y in yb..yb + 5 {
                for x in 260..460 {
                    if (x - 260) % 24 < 18 {
                        grey[y * w + x] = 0;
                    }
                }
            }
            yb += 14;
        }

        let binary = LstmRecognizer::binarize_page_with(&grey, w, h, BinarizeMode::Otsu);
        // Checked under BOTH strictness settings: a genuinely RULED grid has
        // real `nhb`/`nvb`, so requiring ruled evidence must not cost it.
        for require_ruled in [false, true] {
            assert!(
                LstmRecognizer::block_is_table(&binary, w, h, (18, 18, 224, 236), require_ruled),
                "ruled grid block → table (require_ruled={require_ruled})"
            );
            assert!(
                !LstmRecognizer::block_is_table(&binary, w, h, (258, 18, 462, 262), require_ruled),
                "paragraph block → not table (require_ruled={require_ruled})"
            );
            assert!(
                !LstmRecognizer::block_is_table(&binary, w, h, (18, 18, 70, 70), require_ruled),
                "block under 100 px is skipped (require_ruled={require_ruled})"
            );
        }
    }

    /// The `require_ruled` gate must actually DO something, and the input that
    /// proves it is the measured false positive itself: a block of ordinary
    /// multi-column TEXT with **zero printed rules** that nonetheless clears
    /// `decide_if_table`'s score on the whitespace-only path.
    ///
    /// This is the two-sided falsifier for the default-path fix. `require_ruled
    /// = false` (a `strip_borders` caller, who has signalled table intent) must
    /// still call it a table — otherwise the gate is not a discrimination fix
    /// but a disabled feature. `require_ruled = true` (the default path) must
    /// not — otherwise the fix does nothing.
    ///
    /// The fixture reproduces the mechanism measured on a real 2550×3300
    /// two-column scan (`nhb = nvb = 0`, high `nvw`, 69 of 72 prose lines
    /// stamped `type=table`) at unit scale: 8 tall text columns separated by
    /// wide, full-height whitespace corridors. The `assert_eq!(nhb, 0)` /
    /// `assert_eq!(nvb, 0)` guards are load-bearing — without them a fixture
    /// whose ink accidentally aliased to rules under the `o100.1` opening (a
    /// real, previously-hit mistake: solid blocks measured `nhb = 14`) would
    /// make this pass for entirely the wrong reason.
    #[test]
    fn require_ruled_rejects_rule_free_multi_column_text_that_scores_on_whitespace_alone() {
        let (w, h) = (900usize, 400usize);
        let mut grey = vec![255u8; w * h];
        // 8 tall columns of glyph-sized marks, wide empty gutters between them.
        // Sizes are picked against `decide_if_table`'s own morphology, not by
        // eye — a first attempt with 6 px marks measured `nvw = 1` because the
        // chain's `o8.1` noise-clean opening ERASED every mark and left a blank
        // page (one big region, no corridors). Each constant below is chosen so
        // the fixture survives the step that would otherwise destroy it:
        //   - mark width 12 > the `o8.1` open (8) so the ink survives cleaning
        //   - mark run 12 << the `o100.1` open (100) so it never aliases to a
        //     horizontal RULE (keeps nhb = 0, asserted below)
        //   - mark height 6 << the `o1.100` open (100) so it never aliases to a
        //     vertical rule either (keeps nvb = 0)
        //   - gutter 36 px, full page height: after the chain's `r1` 2x reduce
        //     that is 18 wide (>= the width-5 bar) and 200 tall (>= the o1.100
        //     bar), so each gutter counts toward nvw
        //   - intra-column gaps 6 px -> 3 after reduce, BELOW the width-5 bar,
        //     so they do not inflate nvw
        for col in 0..8usize {
            let x0 = 20 + col * 108;
            let mut yb = 20;
            while yb + 6 <= 380 {
                for y in yb..yb + 6 {
                    for x in x0..x0 + 72 {
                        if (x - x0) % 18 < 12 {
                            grey[y * w + x] = 0;
                        }
                    }
                }
                yb += 12;
            }
        }
        let binary = LstmRecognizer::binarize_page_with(&grey, w, h, BinarizeMode::Otsu);
        let block = (0i32, 0i32, w as i32, h as i32);
        let d = crate::pageseg::region_table_decision(&binary, w, h, block)
            .expect("block is well above the 100 px floor");

        // The fixture must be genuinely RULE-FREE, or this test proves nothing.
        assert_eq!(d.nhb, 0, "fixture aliased to horizontal rules: {d:?}");
        assert_eq!(d.nvb, 0, "fixture aliased to vertical rules: {d:?}");
        // ...and it must genuinely clear the score on whitespace alone, or the
        // gate is never even consulted.
        assert!(
            d.score >= crate::pageseg::TABLE_SCORE_THRESHOLD,
            "fixture does not reproduce the false positive ({d:?}) — it must \
             clear the threshold on nvw alone for this test to measure the gate"
        );
        assert!(
            !d.has_ruled_evidence(),
            "a rule-free fixture must report no ruled evidence: {d:?}"
        );

        assert!(
            LstmRecognizer::block_is_table(&binary, w, h, block, false),
            "require_ruled=false must keep the bare leptonica verdict — a \
             strip_borders caller NEEDS the whitespace-only path, since \
             stripping removes the very rules the ruled conditions count"
        );
        assert!(
            !LstmRecognizer::block_is_table(&binary, w, h, block, true),
            "require_ruled=true must reject rule-free multi-column TEXT ({d:?}) \
             — this is the 69-of-72-prose-lines false positive"
        );
    }
}

#[cfg(test)]
mod noise_readmit_tests {
    use super::noise_readmit_reach;

    /// The rule NORMALIZES: the same layout at 2x scale must yield exactly 2x
    /// the reach, so no absolute pixel constant leaks in. This is the whole
    /// point — `filter_blobs`'s `textord_max_noise_size = 7` is an ABSOLUTE
    /// bar, which is why a period (definitionally a tiny solid dot) fails it
    /// at every ordinary scan resolution.
    #[test]
    fn reach_scales_with_the_layout_not_with_an_absolute_constant() {
        let small: Vec<(i32, i32)> = (0..8).map(|i| (i * 20, i * 20 + 14)).collect();
        let large: Vec<(i32, i32)> = (0..8).map(|i| (i * 40, i * 40 + 28)).collect();
        let rs = noise_readmit_reach(&small).expect("small");
        let rl = noise_readmit_reach(&large).expect("large");
        assert!(
            (rl - rs * 2.0).abs() < 0.01,
            "doubling the layout must double the reach: {rs} -> {rl}"
        );
        // 20 px pitch -> half-average = 10 px.
        assert!(
            (rs - 10.0).abs() < 0.01,
            "20px pitch -> 10px reach, got {rs}"
        );
    }

    /// Can-it-fire: a period sitting a few px past the last letter of a line
    /// whose glyph advance is ~20 px must be INSIDE the reach. These are the
    /// real measured numbers from the reference scan (last word ends at 800,
    /// period ink at 803..808).
    #[test]
    fn a_real_line_final_period_is_within_reach() {
        let spans: Vec<(i32, i32)> = (0..20).map(|i| (600 + i * 10, 600 + i * 10 + 7)).collect();
        let reach = noise_readmit_reach(&spans).expect("reach");
        let gap_to_period = 3.0_f32;
        assert!(
            gap_to_period <= reach,
            "a period {gap_to_period} px past the last glyph must be within \
             reach {reach} on a 10 px-advance line"
        );
    }

    /// Can-it-stay-silent: a speck far out in the margin must NOT be admitted,
    /// or the rule would drag page noise into every crop. Uses a genuinely
    /// distant blob, not a degenerate empty input.
    #[test]
    fn a_distant_margin_speck_is_out_of_reach() {
        let spans: Vec<(i32, i32)> = (0..20).map(|i| (600 + i * 10, 600 + i * 10 + 7)).collect();
        let reach = noise_readmit_reach(&spans).expect("reach");
        let far = 200.0_f32;
        assert!(
            far > reach,
            "a speck {far} px out must be beyond reach {reach}"
        );
    }

    /// No basis to judge by => no re-admission. A single-blob row has no
    /// spacing to average, so the rule must decline rather than invent a
    /// threshold.
    #[test]
    fn a_row_with_too_few_blobs_yields_no_reach() {
        assert!(noise_readmit_reach(&[]).is_none());
        assert!(noise_readmit_reach(&[(10, 20)]).is_none());
    }
}
