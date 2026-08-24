//! The token seam — ONE TOKENIZATION RECEIPT, MANY BORROWED CONSUMERS.
//!
//! The bounded seam this crate exists to test: can a single versioned BPE
//! tokenization of one source span drive lexical retrieval (Tantivy),
//! lexical/grammar projection (DeepNSM-v2) and a forward-prediction input
//! surface at the same time, WITHOUT re-tokenizing, without a `DataFrame`, and
//! without a second cognitive population?
//!
//! ```text
//!   ONE SOURCE SPAN -> ONE TOKENIZATION RECEIPT.
//!   TOKENIZE ONCE.  PROJECT MANY TIMES.
//!   THE POPULATION DOES NOT MOVE.  THE VIEW DOES.
//!   TANTIVY IS AN INDEX, NOT MEMORY OWNERSHIP.
//!   BPE IS TOKENIZATION, NOT ONTOLOGY.
//!   CONTENT NEVER TRAVELS IN CLASS ADDRESSES.
//! ```
//!
//! Layering, and what each layer is forbidden from doing:
//!
//! | module | owns | never |
//! |---|---|---|
//! | [`contract`] | the codebook + its identity | reads the lane |
//! | [`lane`] | resident particles + framing | owns text |
//! | [`lexical`] | the `DeepNSM` projection | reads the source |
//! | [`seam_tantivy`] | the index seam | owns offsets |
//! | [`forward`] | the prediction input surface | owns the sequence |
//!
//! Status: PROBE. Nothing here is a production carrier. #1012 returned
//! CAN-FIT, NOT YET BUY at fixture scale; this crate answers the integration
//! half of the question that verdict left open, and the probe reports what it
//! measured rather than what it hoped.

pub mod contract;
pub mod docir;
pub mod forward;
pub mod lane;
pub mod lexical;
pub mod seam_tantivy;
