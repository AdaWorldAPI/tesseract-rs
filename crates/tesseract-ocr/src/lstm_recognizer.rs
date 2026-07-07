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

use tesseract_core::{RecoderError, UniCharSet, UniCharSetError, UnicharCompress};

use crate::network::{NetError, Network};

/// `TF_COMPRESS_UNICHARSET` (`lstmrecognizer.h` `TrainingFlags`): the recoder is
/// present (recoding on) rather than a pass-through identity codec.
const TF_COMPRESS_UNICHARSET: i32 = 64;

/// A failure assembling the recognizer from its components.
#[derive(Debug)]
pub enum RecognizerError {
    /// The network (B1) failed to load, or the trailing fields were truncated.
    Network(NetError),
    /// The unicharset text component failed to parse.
    Charset(UniCharSetError),
    /// The recoder binary component failed to parse.
    Recoder(RecoderError),
}

impl std::fmt::Display for RecognizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "network/tail load: {e}"),
            Self::Charset(e) => write!(f, "unicharset load: {e:?}"),
            Self::Recoder(e) => write!(f, "recoder load: {e:?}"),
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
        assert_eq!(1 & TF_COMPRESS_UNICHARSET, 0, "int-mode-only model doesn't recode");
    }
}
