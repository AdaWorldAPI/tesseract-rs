//! Dump the [`FacetCascade`] bytes for a handful of synthetic `ccstruct`/
//! `textord` shapes (`TBOX`/`BLOBNBOX`/`ROW`/`TO_ROW`/`BLOCK`/`TO_BLOCK`/
//! `POLY_BLOCK`) — the future oracle seam for this batch, sibling to
//! `network_dump`.
//!
//! Unlike `network_dump` (which diffs against a real `network_spec_oracle`
//! linking libtesseract), no C++ oracle exists yet for this leaf — there is
//! no `TBOX::Serialize`-shaped on-disk blob to parse a real file from
//! (`textord_facet.rs`'s shapes are built directly from in-memory field
//! values, not deserialized). This example exists so that a future oracle
//! (a tiny C++ program that constructs the same synthetic values via the
//! real `TBOX`/`BLOBNBOX`/… constructors and prints their field values) has
//! a Rust-side hex dump to diff the *carving* against, the same way
//! `network_dump`'s `facet:` line is diffed today.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example textord_facet_dump
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use lance_graph_contract::facet::FacetCascade;
use lance_graph_contract::textord_facet::{
    blobnbox_facet, block_facet, poly_block_facet, row_facet, tbox_facet, to_block_facet,
    to_row_facet,
};

fn dump(name: &str, f: FacetCascade) {
    let hex: String = f.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
    println!("{name:<12} classid={:#010x} bytes={hex}", f.facet_classid);
}

fn main() {
    // TBOX: a plain bounding box (left, bottom, right, top).
    dump("tbox", tbox_facet(-100, 5, 200, 300));

    // BLOBNBOX: a text blob with a box + textord classification.
    dump(
        "blobnbox",
        blobnbox_facet(
            10, 20, 130, 240,    // box
            7,      // region_type = BRT_TEXT
            4,      // left_tab_type = TT_CONFIRMED
            0,      // right_tab_type = TT_NONE
            false,  // joined
            true,   // vert_possible
            true,   // horz_possible
            false,  // leader_on_left
            false,  // leader_on_right
            13_200, // area
        ),
    );

    // ROW: a finished text line.
    dump("row", row_facet(40, 480, 23.6, 2, 15, 7.4, 5.2));

    // TO_ROW: the same line mid-textord, before the baseline fit finalizes.
    dump(
        "to_row",
        to_row_facet(0.0031, 462.0, 23.0, 7.0, 5.0, 3.0, 12.0),
    );

    // BLOCK: a finished page block.
    dump(
        "block",
        block_facet(
            0, 0, 620, 30, -7, 12, 3, /* proportional */ true, false,
        ),
    );

    // TO_BLOCK: the same block mid-textord.
    dump(
        "to_block",
        to_block_facet(18.0, 24.0, 20.0, 3.0, 10.0, 40.0, 6, -2.0),
    );

    // POLY_BLOCK: a polygonal flowing-text region.
    dump("poly_block", poly_block_facet(0, 0, 620, 800, 1)); // PT_FLOWING_TEXT
}
