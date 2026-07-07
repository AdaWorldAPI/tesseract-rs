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

pub mod image_input;
pub mod lstm_recognizer;
pub mod network;

pub use image_input::{parse_pgm, prescale_grey_to_height, PgmError};
pub use lstm_recognizer::{LstmRecognizer, RecognizerError};
pub use network::{InputShape, NetError, Network, Node, ReverseKind};
