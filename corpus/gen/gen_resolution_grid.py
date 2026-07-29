#!/usr/bin/env python3
"""Deterministic generator for `corpus/quality/` — the 8+7+0 resolution-grid
quality fixture (CI quality fence) plus its typography ground truth.

Shape mirrors the public "resolution testset" style sheet that surfaced the
multi-column + word-box findings (an 8x2 grid of the SAME paragraph at
descending effective resolution), but every byte here is generated from text
authored in this file with the system DejaVu font — no third-party image, so
the corpus stays license-clean by construction (same rule as gen_pages.py).

Grid: 8 columns x 2 rows = 16 cells, row-major "ladder" of downscale factors
(LADDER below). Each cell renders the SAME authored paragraph at the SAME
geometry, then degrades it by downscale->upscale (LANCZOS, deterministic) —
resolution loss at CONSTANT geometry, so ONE typography ground truth (line
baselines, pitch, word x-spans) holds for every cell. The ladder is tuned so
the ENGINE's measured outcome is the "8+7+0" pattern:

    top row    8/8 cells legible
    bottom row 7/8 cells legible
    last cell  dead (no recognizable words)

Those numbers are pinned by `crates/tesseract-ocr/tests/quality_resolution_grid.rs`
(the quality fence) and the typography numbers by
`crates/tesseract-ocr-pdf/tests/typography_overlay.rs`.

Outputs (both committed):
  * corpus/quality/resgrid.pgm     — P5 8-bit grey grid.
  * corpus/quality/resgrid.gt.json — geometry ground truth: cell rects,
    ladder factors, font ascent/descent/nominal size, line pitch, per-line
    baseline y + text + per-word x-spans (all CELL-RELATIVE pixels).

Determinism: no timestamps, no RNG (pure downscale ladder needs none), fixed
font paths. Running twice is byte-identical; each write prints byte count +
sha256 (the cheap "did anything change" check).

Usage:
    python3 gen_resolution_grid.py [outdir]
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_PX = 22

# Authored here (license-clean): three short dictionary-word lines, ASCII-only
# (WinAnsi-safe for the searchable-PDF overlay CI).
# Three lines keeps the whole 16-cell grid inside a CI-viable runtime
# (recognition cost is ~linear in total lines: 16 cells x 3 lines).
LINES = [
    "Optical character readers turn",
    "printed pages into machine",
    "readable text for search and",
]

# Cell-internal layout (pixels).
PAD_X = 10
PAD_TOP = 12
PAD_BOTTOM = 12
PITCH = 30  # baseline-to-baseline

# Grid layout.
COLS = 8
ROWS = 2
GUTTER_X = 48
GUTTER_Y = 56
MARGIN = 24

# The resolution ladder (downscale factor per cell, row-major). Tuned against
# the actual engine (see the quality CI): the whole top row and bottom cells
# 0-6 stay legible; the final cell falls off the recognition cliff.
LADDER = [
    1.00, 0.92, 0.85, 0.78, 0.72, 0.66, 0.60, 0.55,
    0.50, 0.46, 0.42, 0.38, 0.34, 0.30, 0.26, 0.11,
]


def render_reference_cell(font: ImageFont.FreeTypeFont) -> tuple[Image.Image, dict]:
    """The crisp reference cell + its ground-truth geometry (cell-relative)."""
    ascent, descent = font.getmetrics()
    max_w = max(font.getlength(line) for line in LINES)
    cell_w = int(math.ceil(max_w)) + 2 * PAD_X
    cell_h = PAD_TOP + (len(LINES) - 1) * PITCH + ascent + descent + PAD_BOTTOM

    img = Image.new("L", (cell_w, cell_h), 255)
    draw = ImageDraw.Draw(img)
    gt_lines = []
    for i, line in enumerate(LINES):
        top = PAD_TOP + i * PITCH  # PIL default anchor: top of the ascender box
        draw.text((PAD_X, top), line, font=font, fill=0)
        words = []
        cursor = 0.0
        for word in line.split(" "):
            w_len = font.getlength(word)
            words.append(
                {
                    "text": word,
                    "x0": round(PAD_X + cursor, 2),
                    "x1": round(PAD_X + cursor + w_len, 2),
                }
            )
            cursor += w_len + font.getlength(" ")
        gt_lines.append(
            {
                "text": line,
                "baseline_y": top + ascent,
                "x": PAD_X,
                "words": words,
            }
        )

    gt = {
        "font_px": FONT_PX,
        "ascent": ascent,
        "descent": descent,
        "pitch": PITCH,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "lines": gt_lines,
    }
    return img, gt


def degrade(cell: Image.Image, factor: float) -> Image.Image:
    """Resolution loss at constant geometry: downscale then upscale back."""
    if factor >= 0.999:
        return cell.copy()
    w, h = cell.size
    dw = max(1, round(w * factor))
    dh = max(1, round(h * factor))
    small = cell.resize((dw, dh), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def write_confirmed(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"wrote {path}  bytes={len(data)}  sha256={digest}")


def main() -> None:
    outdir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "quality"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype(str(FONT_PATH), FONT_PX)
    ref, gt = render_reference_cell(font)
    cell_w, cell_h = ref.size

    grid_w = 2 * MARGIN + COLS * cell_w + (COLS - 1) * GUTTER_X
    grid_h = 2 * MARGIN + ROWS * cell_h + (ROWS - 1) * GUTTER_Y
    grid = Image.new("L", (grid_w, grid_h), 255)

    cells = []
    for k, factor in enumerate(LADDER):
        row, col = divmod(k, COLS)
        x = MARGIN + col * (cell_w + GUTTER_X)
        y = MARGIN + row * (cell_h + GUTTER_Y)
        grid.paste(degrade(ref, factor), (x, y))
        cells.append({"index": k, "row": row, "col": col, "x": x, "y": y, "factor": factor})

    gt_doc = {
        "schema": "tesseract-rs/resgrid-gt.v1",
        "grid_w": grid_w,
        "grid_h": grid_h,
        "cols": COLS,
        "rows": ROWS,
        "cells": cells,
        **gt,
    }

    pgm = b"P5\n%d %d\n255\n" % (grid_w, grid_h) + grid.tobytes()
    write_confirmed(outdir / "resgrid.pgm", pgm)
    write_confirmed(
        outdir / "resgrid.gt.json",
        (json.dumps(gt_doc, indent=1, sort_keys=True) + "\n").encode(),
    )


if __name__ == "__main__":
    main()
