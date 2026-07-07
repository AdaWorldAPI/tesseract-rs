//! # tesseract-ocr — the assembly tier of the pure-Rust Tesseract transcode
//!
//! The two foundations meet here: the **compute** tier (`tesseract-recognizer`
//! — the SIMD/int8 NetworkIo grid + layer forwards) and the **content/decode**
//! tier (`tesseract-core` — charset, recoder, CTC beam). Only the assembly
//! tier sees both, so it hosts the network-tree loader (B1), the recognizer
//! load (B2), and `RecognizeLine` (B3) — the steps that need the network AND
//! the recoder/charset together.
//!
//! Per Core-First, the network **structure** vocabulary comes from the Core
//! (`tesseract_core::network::{NetworkType, NetworkHeader}`,
//! `E-OCR-NETWORK-SINK-1`, whose subclass/override manifest is the
//! `ruff_cpp_spo` C++ SPO harvest); the compute **payloads** come from the
//! recognizer's proven leaves. See `.claude/plans/recognizer-image-to-text-v2.md`.

pub mod conncomp;
pub mod image_input;
#[cfg(feature = "seg-approx")]
pub mod line_segment;
pub mod lstm_recognizer;
pub mod network;
pub mod renderer;
pub mod threshold;

pub use image_input::{parse_pgm, prescale_grey_to_height, PgmError};
#[cfg(feature = "seg-approx")]
pub use line_segment::{find_text_lines, LineBand};
pub use lstm_recognizer::{LstmRecognizer, RecognizerError};
pub use network::{InputShape, NetError, Network, Node, ReverseKind};
pub use threshold::{
    histogram_rect_gray, histogram_rect_multi, histogram_rect_rgb, otsu_stats,
    otsu_threshold_channels, otsu_threshold_gray, threshold_rect_to_binary,
    threshold_rect_to_binary_multi, OtsuChannel, OtsuResult, OtsuStatsResult, HISTOGRAM_SIZE,
};
