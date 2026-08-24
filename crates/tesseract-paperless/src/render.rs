//! Plain-text projection of a [`DocIr`] — the ONE join every consumer that
//! wants "the document as text" needs, so it exists once rather than once
//! per consumer.
//!
//! Not a Tesseract transcode and not part of the token seam: it needs no
//! `token` dependency, walks the tree exactly the way
//! [`crate::token::docir::spans`] does (kept independent on purpose — that
//! module is feature-gated behind `token`; this one is `default`), and is
//! meant for search-index bodies and human-facing previews, not tokenization.

use ogar_doc_ir::{DocIr, Region};

/// Join every text-bearing region's text, depth-first, one line per region.
/// Container regions contribute nothing themselves; a [`ogar_doc_ir::Region`]
/// with no `text` (a `Figure`, a `Main` holding only tables) is skipped, not
/// blanked, so the join has no empty lines from structure alone.
#[must_use]
pub fn plain_text(ir: &DocIr) -> String {
    let mut out = String::new();
    for page in &ir.pages {
        for r in &page.regions {
            walk(r, &mut out);
        }
    }
    out
}

fn walk(r: &Region, out: &mut String) {
    if let Some(t) = r.text.as_deref() {
        if !t.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(t);
        }
    }
    for cell in &r.cells {
        if !cell.text.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&cell.text);
        }
    }
    for c in &r.children {
        walk(c, out);
    }
}

/// A short preview — the first `max_chars` characters of [`plain_text`], cut
/// on a char boundary (never inside a multi-byte UTF-8 sequence) and marked
/// with an ellipsis when truncated.
#[must_use]
pub fn preview(ir: &DocIr, max_chars: usize) -> String {
    let text = plain_text(ir);
    if text.chars().count() <= max_chars {
        return text;
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogar_doc_ir::{BBoxRail, DocPage, Provenance, Rail, RegionKind, TableCell, DOC_IR_VERSION};

    fn bbox() -> BBoxRail {
        BBoxRail {
            tl: Rail { x: 0, y: 0 },
            br: Rail { x: 10, y: 10 },
        }
    }

    fn region(kind: RegionKind, order: u16, text: Option<&str>) -> Region {
        Region {
            kind,
            bbox: bbox(),
            reading_order: order,
            text: text.map(str::to_string),
            cells: Vec::new(),
            children: Vec::new(),
        }
    }

    fn doc(pages: Vec<DocPage>) -> DocIr {
        DocIr {
            version: DOC_IR_VERSION.to_string(),
            source: Provenance::Ocr,
            geometry: ogar_doc_ir::Geometry::DomOrder,
            content_sha256: [0u8; 32],
            mime: "image/png".to_string(),
            pages,
            fields: Vec::new(),
        }
    }

    #[test]
    fn joins_text_bearing_regions_in_reading_order_and_skips_containers() {
        let ir = doc(vec![DocPage {
            number: 0,
            width: 100,
            height: 100,
            regions: vec![
                region(RegionKind::Header, 0, Some("Acme GmbH")),
                Region {
                    // A container with no text of its own — must contribute
                    // NOTHING, not an empty line, or every container in a
                    // real document would pad the preview with blanks.
                    kind: RegionKind::Main,
                    bbox: bbox(),
                    reading_order: 1,
                    text: None,
                    cells: Vec::new(),
                    children: vec![region(RegionKind::Text, 2, Some("Invoice body"))],
                },
                region(RegionKind::Footer, 3, Some("Seite 1")),
            ],
        }]);
        assert_eq!(plain_text(&ir), "Acme GmbH\nInvoice body\nSeite 1");
    }

    #[test]
    fn table_cells_join_too_since_this_is_a_preview_not_the_token_seam() {
        // Deliberately the OPPOSITE rule from `token::docir::spans`, which
        // skips cells on purpose (a cell is typed data, not tokenizable
        // prose). A human-facing preview wants the numbers on the page.
        let mut table = region(RegionKind::Table, 0, None);
        table.cells = vec![
            TableCell {
                row: 0,
                col: 0,
                text: "14.2".to_string(),
                bbox: bbox(),
                confidence: 90,
            },
            TableCell {
                row: 0,
                col: 1,
                text: "mg/dl".to_string(),
                bbox: bbox(),
                confidence: 90,
            },
        ];
        let ir = doc(vec![DocPage {
            number: 0,
            width: 10,
            height: 10,
            regions: vec![table],
        }]);
        assert_eq!(plain_text(&ir), "14.2\nmg/dl");
    }

    #[test]
    fn preview_truncates_on_a_char_boundary_and_marks_the_cut() {
        // "café" — the 'é' is 2 UTF-8 bytes; a byte-indexed cut at a bad
        // offset would panic or split the character. Truncate by CHAR count.
        let ir = doc(vec![DocPage {
            number: 0,
            width: 10,
            height: 10,
            regions: vec![region(RegionKind::Text, 0, Some("café society"))],
        }]);
        let p = preview(&ir, 4);
        assert_eq!(p, "café…");
        assert_eq!(preview(&ir, 100), "café society");
    }

    #[test]
    fn an_empty_document_previews_to_an_empty_string_not_a_panic() {
        let ir = doc(Vec::new());
        assert_eq!(plain_text(&ir), "");
        assert_eq!(preview(&ir, 10), "");
    }
}
