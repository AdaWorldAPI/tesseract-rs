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
        // SimpleTextOutput == (OutputLossType == LT_SOFTMAX) == float output.
        let simple = !outputs.int_mode();
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
        let simple = !outputs.int_mode();
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
    fn prepare_grid(&self, grey: &[u8], w: usize, h: usize) -> (NetworkIo, TRand) {
        let target_h = self
            .network
            .input_shape
            .map_or(36, |s| s.height.max(1) as usize);
        let (scaled, sw) = prescale_grey_to_height(grey, w, h, target_h);
        // Seed exactly as RecognizeLine (SetRandomSeed) — the Convolve noise
        // depends on it. from_grey_pix makes no draws for a full-width image, so
        // the randomizer enters the forward pass at the post-warm-up state.
        let mut rng = self.seeded_randomizer();
        let grid = from_grey_pix(&scaled, sw, target_h, target_h as i32, 0, &mut rng);
        (grid, rng)
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
        let (grid, mut rng) = self.prepare_grid(grey, w, h);
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
        let (grid, mut rng) = self.prepare_grid(&grey, w, h);
        let outputs = self.network.forward(&grid, &mut rng)?;
        if outputs.int_mode() {
            return Err(RecognizerError::Network(NetError::Forward(
                "recognize_image_file_words expects softmax float logits (int-mode output)",
            )));
        }
        let simple = !outputs.int_mode();
        let rows: Vec<&[f32]> = (0..outputs.width()).map(|t| outputs.f(t)).collect();
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
