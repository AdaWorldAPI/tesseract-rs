#!/usr/bin/env python3
"""Deterministic generator for `corpus/lab/` — a RULED four-column lab-report
table rendered with REAL text, for the table-column-splitting regression.

Why this exists separately from `tests/lab_table_grid.rs`'s own fixtures: that
test draws glyph-shaped ink marks, which is exactly right for exercising
`decide_if_table` (a morphology leaf that only sees ink), but those marks do
not RECOGNIZE — they produce no words. `structured::extract_table_grid` splits
columns by the whitespace gaps *between recognized words*, so measuring it
needs a fixture the LSTM can actually read.

Shape mirrors a real lab report: a header row plus six results, four columns
(test | value | unit | reference range). ASCII-only on purpose — the column
splitter is charset-independent, and ASCII keeps this runnable against the
`eng` model without a `deu` dependency. Real German reports carry umlauts
(`Hamoglobin`); that is a charset concern proven elsewhere (the deu model is
byte-parity green), not a column-splitting one.

**The gutters are deliberately generous** (~3x the glyph height). The defect
under test is that a four-column table collapses to one column; if the gutters
were tight, a collapse would be ambiguous between "the splitter is broken" and
"the columns really were too close to separate". Wide gutters remove that
excuse — a collapse here is unambiguously the splitter.

Determinism: no RNG, no timestamps, fixed font path. Running twice is
byte-identical; each write prints byte count + sha256.

Usage:
    python3 gen_lab_table.py [outdir]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_PX = 26

# Four columns, ASCII-only. Row 0 is the header a real report prints.
ROWS = [
    ("Parameter", "Ergebnis", "Einheit", "Referenz"),
    ("Haemoglobin", "14.2", "g/dl", "13.5 - 17.5"),
    ("Glukose", "98", "mg/dl", "74 - 106"),
    ("Kalium", "4.1", "mmol/l", "3.5 - 5.1"),
    ("Natrium", "141", "mmol/l", "136 - 145"),
    ("Kreatinin", "0.9", "mg/dl", "0.7 - 1.3"),
    ("Calcium", "2.4", "mmol/l", "2.2 - 2.6"),
]

# Column left edges. Gutters between columns are ~90 px against a ~26 px glyph
# height — far wider than the "one median word-height" the splitter needs, so a
# collapse cannot be blamed on tight spacing.
COL_X = [60, 420, 700, 980]
ROW_TOP = 70
ROW_PITCH = 64
PAGE_W = 1400
PAGE_H = ROW_TOP + len(ROWS) * ROW_PITCH + 70

# Rule geometry (this fixture is the RULED variant — it must be classified a
# table at all before its grid can be measured).
RULE_PX = 3
TABLE_L, TABLE_R = 30, PAGE_W - 30


def write_confirmed(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"wrote {path}  bytes={len(data)}  sha256={hashlib.sha256(data).hexdigest()}")


def main() -> None:
    outdir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "lab"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype(str(FONT_PATH), FONT_PX)
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)

    # Horizontal rules: above the header, under the header, and under the last
    # row — what a boxed report actually prints.
    for y in (ROW_TOP - 14, ROW_TOP + ROW_PITCH - 14, ROW_TOP + len(ROWS) * ROW_PITCH - 14):
        draw.rectangle([TABLE_L, y, TABLE_R, y + RULE_PX - 1], fill=0)
    # Vertical rules: the table's outer edges plus one between each column.
    v_edges = [TABLE_L] + [x - 30 for x in COL_X[1:]] + [TABLE_R]
    top, bot = ROW_TOP - 14, ROW_TOP + len(ROWS) * ROW_PITCH - 14 + RULE_PX - 1
    for x in v_edges:
        draw.rectangle([x, top, x + RULE_PX - 1, bot], fill=0)

    gt_rows = []
    for r, row in enumerate(ROWS):
        y = ROW_TOP + r * ROW_PITCH
        for c, cell in enumerate(row):
            draw.text((COL_X[c], y), cell, font=font, fill=0)
        gt_rows.append(list(row))

    pgm = b"P5\n%d %d\n255\n" % (PAGE_W, PAGE_H) + img.tobytes()
    write_confirmed(outdir / "lab_table_ruled.pgm", pgm)
    write_confirmed(
        outdir / "lab_table_ruled.gt.json",
        (
            json.dumps(
                {
                    "schema": "tesseract-rs/lab-table-gt.v1",
                    "width": PAGE_W,
                    "height": PAGE_H,
                    "rows": len(ROWS),
                    "cols": len(COL_X),
                    "col_x": COL_X,
                    "row_top": ROW_TOP,
                    "row_pitch": ROW_PITCH,
                    "font_px": FONT_PX,
                    "cells": gt_rows,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )


if __name__ == "__main__":
    main()
