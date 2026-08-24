//! The span population, taken from the document layer's own IR.
//!
//! ```text
//!   THE DOCUMENT LAYER ALREADY HAS AN IDENTITY.  DO NOT MINT A SECOND ONE.
//! ```
//!
//! A tokenization receipt needs three things from upstream: WHICH document,
//! WHICH span of it, and the span's canonical text. All three already exist in
//! [`ogar_doc_ir::DocIr`], and an earlier cut of this crate invented parallel
//! `source_id`/`span_id` integers instead of reading them — a second
//! population wearing the document layer's job.
//!
//! - **which document** — [`ogar_doc_ir::DocIr::content_sha256`], the sha256 of
//!   the ORIGINAL bytes. The crate's own docs correct the plan's first sketch
//!   here and the correction matters for this seam: that hash is a
//!   **per-acquisition dedup key**, NOT a cross-retina semantic identity (a
//!   scan and an HTML page of the same invoice have different bytes). For a
//!   TOKENIZATION receipt the per-acquisition reading is exactly the right
//!   one: you tokenize bytes, and different bytes are a different tokenization.
//!   Cross-retina convergence is a facts question
//!   (`ogar_doc_ir::converges_on_facts`) and is not this seam's business.
//! - **which span** — a [`ogar_doc_ir::Region`], addressed by its page number
//!   and its `reading_order`, which the IR documents as "the reading-order the
//!   temporal stream (and `DeepNSM`) consumes". The seam does not choose an
//!   order; it inherits the one the document layer already fixed.
//! - **the text** — [`ogar_doc_ir::Region::text`]. Each region owns its own
//!   canonical text, so a byte offset is REGION-LOCAL. This is what dissolves
//!   the "the OCR boundary supplies no byte offsets" gap: the boundary supplies
//!   no PAGE-wide offsets, and does not need to.
//!
//! Because the IR is source-agnostic, so is the seam: a crawled page
//! (`Provenance::Dom`) and a scan (`Provenance::Ocr`) present the same span
//! population to the tokenizer, and this module contains not one line that
//! knows which retina produced it.

use ogar_doc_ir::{DocIr, Region};

/// Where a span sits in the document layer's own address space.
///
/// The document half is an INDEX into [`crate::token::lane::TokenLane`]'s document
/// table rather than a repeated 32-byte hash: at these span sizes a receipt is
/// already a third of the resident bytes, and stamping the sha256 on every one
/// of them would more than double that for no addressing gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpanKey {
    /// Index into the lane's document table (which holds the `content_sha256`).
    pub doc: u16,
    /// [`ogar_doc_ir::DocPage::number`].
    pub page: u16,
    /// [`ogar_doc_ir::Region::reading_order`].
    pub reading_order: u16,
}

/// One span the tokenizer will consume: its address and its canonical text.
#[derive(Clone, Copy, Debug)]
pub struct SpanRef<'a> {
    /// The document-layer address.
    pub key: SpanKey,
    /// The region's own canonical text — the bytes that get tokenized.
    pub text: &'a str,
}

/// Walk a [`DocIr`] into its text-bearing spans, in page then reading order.
///
/// Container regions are descended into; a region with no `text` contributes
/// nothing itself (a `Figure` has no text, a `Main` holding tables is a
/// container). Table cells are deliberately NOT flattened into text here —
/// a cell is typed, addressed `(row, col)` data that the structured path
/// consumes directly, and pouring it into a token stream is the
/// flatten-the-table mistake the ingestion doctrine names.
#[must_use]
pub fn spans(ir: &DocIr, doc: u16) -> Vec<SpanRef<'_>> {
    let mut out = Vec::new();
    for page in &ir.pages {
        for r in &page.regions {
            walk(r, doc, page.number, &mut out);
        }
    }
    out
}

fn walk<'a>(r: &'a Region, doc: u16, page: u16, out: &mut Vec<SpanRef<'a>>) {
    // (Edition 2021 here, so no let-chain: the workspace pins edition 2021.)
    if let Some(t) = r.text.as_deref().filter(|t| !t.is_empty()) {
        out.push(SpanRef {
            key: SpanKey {
                doc,
                page,
                reading_order: r.reading_order,
            },
            text: t,
        });
    }
    for c in &r.children {
        walk(c, doc, page, out);
    }
}
