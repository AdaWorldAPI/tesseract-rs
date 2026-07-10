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

pub mod binreduce;
pub mod blob_filter;
pub mod conncomp;
pub mod image_input;
#[cfg(feature = "seg-approx")]
pub mod line_segment;
pub mod lstm_recognizer;
pub mod morph;
pub mod morphapp;
pub mod network;
pub mod page_furniture;
pub mod pageseg;
pub mod renderer;
pub mod seedfill;
pub mod stats;
pub mod structured;
pub mod textline;
pub mod threshold;
pub mod xy_cut;

pub use binreduce::{
    expand_binary_power2, expand_replicate, reduce_rank_binary2, reduce_rank_binary_cascade,
};
pub use blob_filter::{filter_blobs, FilteredBlobs};
pub use conncomp::{conn_comp_areas, conn_comp_bb, ConnComp, ConnCompBox};
pub use image_input::{parse_pgm, prescale_grey_to_height, PgmError};
#[cfg(feature = "seg-approx")]
pub use line_segment::{find_text_lines, LineBand};
pub use lstm_recognizer::{LstmRecognizer, RecognizerError};
pub use morph::{
    close_brick, close_safe_brick, dilate_brick, erode_brick, morph_sequence, open_brick,
};
pub use morphapp::{
    morph_sequence_by_component, select_by_size, SelectRelation, SelectType, SizeFilter,
};
pub use network::{InputShape, NetError, Network, Node, ReverseKind};
pub use page_furniture::{detect_page_furniture, PageFurniture};
pub use pageseg::{
    gen_textblock_mask, gen_textline_mask, generate_halftone_mask, HalftoneMask, TextlineMask,
    MIN_HEIGHT, MIN_WIDTH,
};
pub use renderer::{render_hocr, render_text, render_tsv, LineWords};
pub use seedfill::seedfill_binary;
pub use stats::Stats;
pub use structured::{
    build_regions, german_invoice_fields, harden_numeric_token, harden_numeric_tokens,
    harvest_fields, iban_mod97_ok, looks_like_guid, looks_like_iban, parse_amount_cents,
    render_json, render_json_with_regions, DocLine, DocPage, DocRegion, DocWord, FieldKind,
    FieldSpec, HarvestedField, RegionKind,
};
pub use textline::{
    adjust_row_limits, assign_blobs_to_rows, cleanup_rows_making, compute_block_xheight,
    compute_dropout_distances, compute_height_modes, compute_line_occupation,
    compute_occupation_threshold, compute_page_skew, compute_row_stats, compute_row_xheight,
    correct_row_xheight, delete_non_dropout_rows, expand_rows, fill_heights, fit_lms_line,
    fit_parallel_lms, fit_parallel_rows, get_row_category, make_initial_textrows, make_rows,
    DetLineFit, FCoord, ICoord, OverlapState, RowCategory, ToBlockCtx, ToRow,
};
pub use threshold::{
    histogram_rect_gray, histogram_rect_multi, histogram_rect_rgb, otsu_stats,
    otsu_threshold_channels, otsu_threshold_gray, threshold_rect_to_binary,
    threshold_rect_to_binary_multi, OtsuChannel, OtsuResult, OtsuStatsResult, HISTOGRAM_SIZE,
};
pub use xy_cut::{xy_cut, PageRect, XyCutParams};
