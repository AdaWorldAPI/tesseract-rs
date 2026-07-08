#!/usr/bin/env python3
"""Deterministic generator for `corpus/pdfs/` — the 5 digital-PDF
golden-corpus fixtures (plan D6.1, `.claude/plans/pdf-to-text-ocr-v1.md`
Phase 6).

Each PDF is hand-assembled as raw PDF 1.4 bytes, with zero third-party PDF
libraries involved in *generation* (the repo's extractor,
`tesseract_ocr_pdf::extract_text_layer`, is `lopdf`-based and reads standard
content streams, so a minimal hand-built PDF exercises it the same way a
real digital PDF would). All document text is authored directly in this
file (see `DOCS` below): no external document corpus, so the corpus stays
license-clean by construction.

PDF shape (deliberately minimal, per D6.1 spec):
  * header `%PDF-1.4\\n`
  * object graph: Catalog -> Pages -> Page(s), each Page has its own
    `/Contents` stream and an inline (non-indirect) Helvetica `/Resources`
    dict -- `<< /Font << /F1 << /Type /Font /Subtype /Type1
    /BaseFont /Helvetica >> >> >>`.
  * content stream per non-empty page: `BT /F1 12 Tf 72 720 Td 16 TL` then,
    per line, `(escaped text) Tj T*`, then `ET`.
  * a correct xref table (20-byte entries) + trailer with `/Size` + `/Root`
    only -- no `/Info`, no `/ID`, no dates, so the bytes are fully
    deterministic across runs.

Ground truth (`doc_NN.gt.txt`) is the SEMANTIC ground truth (what a human
reader should get back), not a byte-for-byte mirror of whatever
`extract_text_layer` happens to return -- the two can differ in whitespace
normalization, and the corpus README documents that distinction explicitly
rather than blurring it. `doc_05` has an intentionally EMPTY content stream
(no text at all, not even `BT`/`ET`) to exercise the no-text-layer arm; its
`.gt.txt` is the literal sentinel `<NO-TEXT-LAYER>\\n`.

Usage:
    python3 gen_pdfs.py [outdir]

`outdir` defaults to `corpus/pdfs` resolved relative to this script's own
location (`corpus/gen/gen_pdfs.py` -> `corpus/pdfs`), independent of the
caller's current working directory.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PAGE_W = 612
PAGE_H = 792

RESOURCES = "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>"


def pdf_escape(s: str) -> str:
    """Escape a line for use inside a PDF literal string `(...)`.

    Backslash MUST be escaped before parentheses -- escaping in the other
    order would double-escape the backslashes just inserted by the
    parenthesis step.
    """
    assert s.isascii(), f"non-ASCII text: {s!r}"
    assert "\n" not in s and "\r" not in s, f"embedded newline in line: {s!r}"
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_content_stream(lines: list[str]) -> bytes:
    """Build one page's content stream. An empty `lines` list produces a
    genuinely empty stream (no BT/ET at all) -- the doc_05 no-text-layer arm.
    """
    if not lines:
        return b""
    parts = [b"BT\n", b"/F1 12 Tf\n", b"72 720 Td\n", b"16 TL\n"]
    for line in lines:
        parts.append(f"({pdf_escape(line)}) Tj\n".encode("ascii"))
        parts.append(b"T*\n")
    parts.append(b"ET")
    return b"".join(parts)


def build_pdf(pages: list[list[str]]) -> bytes:
    """Assemble a minimal PDF 1.4 file for `pages` (one inner list per page).

    Object numbering: 1=Catalog, 2=Pages, then for each page a (Page,
    Contents) object pair in order. No /Info, no /ID, no timestamps -> the
    output is byte-identical across runs for the same `pages` input.
    """
    objects: dict[int, bytes] = {}
    next_num = 3
    page_obj_nums: list[int] = []

    for lines in pages:
        page_num = next_num
        next_num += 1
        content_num = next_num
        next_num += 1

        stream = build_content_stream(lines)
        objects[content_num] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources {RESOURCES} /Contents {content_num} 0 R >>"
        ).encode("ascii")
        page_obj_nums.append(page_num)

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>".encode("ascii")

    total_objs = next_num - 1

    buf = bytearray()
    buf += b"%PDF-1.4\n"

    offsets: dict[int, int] = {}
    for n in range(1, total_objs + 1):
        offsets[n] = len(buf)
        buf += f"{n} 0 obj\n".encode("ascii")
        buf += objects[n]
        buf += b"\nendobj\n"

    xref_offset = len(buf)
    buf += b"xref\n"
    buf += f"0 {total_objs + 1}\n".encode("ascii")
    buf += b"0000000000 65535 f \n"  # the mandatory free-list head, 20 bytes
    for n in range(1, total_objs + 1):
        entry = f"{offsets[n]:010d} 00000 n \n".encode("ascii")
        assert len(entry) == 20, f"xref entry for obj {n} is not 20 bytes: {entry!r}"
        buf += entry

    buf += b"trailer\n"
    buf += f"<< /Size {total_objs + 1} /Root 1 0 R >>\n".encode("ascii")
    buf += b"startxref\n"
    buf += f"{xref_offset}\n".encode("ascii")
    buf += b"%%EOF"

    return bytes(buf)


def self_check(pdf_bytes: bytes) -> None:
    """Sanity-check the assembled PDF: `%%EOF` present, and `startxref`'s
    offset actually points at the literal `xref` keyword.
    """
    assert pdf_bytes.endswith(b"%%EOF"), "missing trailing %%EOF"

    idx = pdf_bytes.rfind(b"startxref")
    assert idx != -1, "missing startxref keyword"

    rest = pdf_bytes[idx + len(b"startxref") :]
    i = 0
    while rest[i : i + 1] in b"\r\n ":
        i += 1
    j = i
    while rest[j : j + 1].isdigit():
        j += 1
    assert j > i, "startxref has no numeric offset following it"

    offset = int(rest[i:j])
    assert pdf_bytes[offset : offset + 4] == b"xref", (
        f"startxref offset {offset} does not point at the 'xref' keyword "
        f"(found {pdf_bytes[offset : offset + 4]!r})"
    )


def write_gt(path: Path, pages: list[list[str]], sentinel: str | None) -> None:
    if sentinel is not None:
        text = sentinel + "\n"
    else:
        text = "\f\n".join("\n".join(page) for page in pages) + "\n"
    path.write_text(text, encoding="ascii", newline="\n")


def report(path: Path) -> None:
    data = path.read_bytes()
    print(f"{path}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")


# --- Document content (authored in-script; ASCII only) --------------------

DOC_01 = [
    "Invoice number 4471 is now due.",
    "Please remit payment within 30 days.",
    "Contact billing with any questions.",
    "Thank you for your continued business.",
]

DOC_02 = [
    "Meeting minutes for the March review.",
    "Attendees discussed the quarterly budget.",
    "Action items were assigned to each team.",
    "The next meeting is set for April.",
    "Please review the attached summary.",
    "Send corrections before the deadline.",
]

# Line 2 deliberately carries one '(' + one ')' + one '\' to exercise the
# PDF literal-string escaping path in build_content_stream/pdf_escape.
DOC_03 = [
    "Project status report, week twelve.",
    r"Note (see appendix): path is C:\data",
    "All milestones remain on schedule.",
]

DOC_04_PAGE_1 = [
    "Shipment tracking summary, page one.",
    "Order 8823 left the warehouse today.",
    "Expected delivery in three days.",
]

DOC_04_PAGE_2 = [
    "Shipment tracking summary, page two.",
    "Order 9014 is currently in transit.",
    "Support was notified by email.",
]

# name -> (pages: list of per-page line lists, gt sentinel or None)
DOCS: list[tuple[str, list[list[str]], str | None]] = [
    ("doc_01", [DOC_01], None),
    ("doc_02", [DOC_02], None),
    ("doc_03", [DOC_03], None),
    ("doc_04", [DOC_04_PAGE_1, DOC_04_PAGE_2], None),
    ("doc_05", [[]], "<NO-TEXT-LAYER>"),
]


def main(argv: list[str]) -> int:
    outdir = Path(argv[1]) if len(argv) > 1 else (Path(__file__).resolve().parent.parent / "pdfs")
    outdir.mkdir(parents=True, exist_ok=True)

    for name, pages, sentinel in DOCS:
        pdf_bytes = build_pdf(pages)
        self_check(pdf_bytes)

        pdf_path = outdir / f"{name}.pdf"
        gt_path = outdir / f"{name}.gt.txt"
        pdf_path.write_bytes(pdf_bytes)
        write_gt(gt_path, pages, sentinel)

        report(pdf_path)
        report(gt_path)

    print(f"OK: {len(DOCS)} PDFs, outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
