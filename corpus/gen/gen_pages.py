#!/usr/bin/env python3
"""Deterministic generator for `corpus/pages/` — the 10-page digital-render
golden-corpus fixture set (plan D6.1, `.claude/plans/pdf-to-text-ocr-v1.md`
Phase 6).

These are clean digital renders of hand-authored English text (plus one
noisy variant), NOT scans of a real document — the corpus README documents
that distinction explicitly so nobody mistakes them for real scanned pages.
All sentences are authored directly in this file (see `PAGES` below): no
external text corpus, no scraped content, so the corpus stays license-clean
by construction.

Output per page `page_NN` (1-indexed, zero-padded to 2 digits):
  * `page_NN.pgm`     — 512x720 8-bit grey, P5 (binary) PGM, header
                        `P5\\n{w} {h}\\n255\\n` followed by raw pixel bytes.
  * `page_NN.gt.txt`  — the authored lines, joined with "\\n", plus a single
                        trailing "\\n". This is ground truth for the whole
                        page (single "column", top-to-bottom reading order).

Determinism: no timestamps, no non-deterministic ordering, and the one
noisy page (`page_10`) draws its Gaussian noise from a fixed `random.seed`.
Running this script twice produces byte-identical output; a diff means an
intentional content change (add/edit a sentence, adjust layout), not
generator drift. Each write is confirmed at the end with a byte count and
sha256, which is the operator's/CI's cheap way to check "did anything
actually change".

Usage:
    python3 gen_pages.py [outdir]

`outdir` defaults to `corpus/pages` resolved relative to this script's own
location (`corpus/gen/gen_pages.py` -> `corpus/pages`), independent of the
caller's current working directory.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Raster + layout constants -------------------------------------------

IMG_W = 512
IMG_H = 720
LEFT_MARGIN = 48
TOP_MARGIN = 56
# "No line may render wider than 512-96 px" (D6.1 spec): a 48px margin on
# each side, expressed as a single width cap so the assertion in
# render_page() is the one source of truth for the constraint.
MAX_TEXT_WIDTH = IMG_W - 96  # 416 px

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_SANS = FONT_DIR / "DejaVuSans.ttf"
FONT_SERIF = FONT_DIR / "DejaVuSerif.ttf"
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"

NOISE_SEED = 42
NOISE_SIGMA = 6.0

# --- Page content (authored in-script; ASCII only) ------------------------
#
# 10 pages x 7 lines = 70 distinct sentences, plain everyday English, no
# punctuation beyond a trailing period (the eng.lstm charset covers ASCII;
# smart quotes/dashes are avoided entirely). Grouped by page below; the font
# plan per page is fixed by the D6.1 spec:
#   page_01-04  DejaVuSans   20px
#   page_05-06  DejaVuSerif  20px
#   page_07     DejaVuSansMono 18px
#   page_08     DejaVuSans   16px
#   page_09     DejaVuSans   24px
#   page_10     DejaVuSans   20px + deterministic Gaussian noise

PAGE_01 = [
    "The old clock ticked all night.",
    "She drinks coffee every morning.",
    "A cool wind moved past the door.",
    "He tied his boots for the hike.",
    "The garden smelled of cut grass.",
    "Birds sang softly before dawn.",
    "Fresh bread cooled on the rack.",
]

PAGE_02 = [
    "The train left the station on time.",
    "Rain tapped gently on the roof.",
    "A cat slept on the warm window.",
    "The map showed a narrow trail.",
    "Waves rolled onto the quiet shore.",
    "He fixed the fence before noon.",
    "The kettle whistled in the kitchen.",
]

PAGE_03 = [
    "Leaves fell across the empty path.",
    "She read a book by the fireplace.",
    "The market opened at seven sharp.",
    "Snow covered the small wooden shed.",
    "The dog barked at the passing cart.",
    "A lantern glowed near the old barn.",
    "Children played in the shaded yard.",
]

PAGE_04 = [
    "The baker sliced a warm loaf.",
    "Clouds drifted over the hills.",
    "He painted the fence bright blue.",
    "The river ran fast after rain.",
    "She planted seeds in neat rows.",
    "The lamp flickered twice at dusk.",
    "Wind chimes rang near the porch.",
]

PAGE_05 = [
    "The library closed at nine sharp.",
    "A fox crossed the frosty field.",
    "The bridge creaked in the wind.",
    "Smoke rose from the tall chimney.",
    "The farmer counted his sheep twice.",
    "Stars filled the clear night sky.",
    "The kettle boiled for morning tea.",
]

PAGE_06 = [
    "The old boat rocked on the tide.",
    "A squirrel darted up the oak tree.",
    "The chef stirred a pot of soup.",
    "Frost covered the quiet meadow.",
    "The mail arrived a bit early.",
    "Thunder rumbled far to the east.",
    "The candle burned low by ten.",
]

PAGE_07 = [
    "The clock struck noon.",
    "A dog ran past the gate.",
    "Snow fell all night long.",
    "The bus was five minutes late.",
    "She locked the front door.",
    "The oven timer went off.",
    "Two birds sat on the wire.",
]

PAGE_08 = [
    "The orchard bloomed early this spring.",
    "A narrow road wound through the valley.",
    "The teacher wrote notes on the board.",
    "Ships passed slowly beyond the pier.",
    "The candle flame swayed in the draft.",
    "Workers repaired the old stone wall.",
    "The orchard smelled of ripe apples.",
]

PAGE_09 = [
    "The gate creaked shut.",
    "Rain began just after noon.",
    "The fire crackled softly.",
    "A ship left the harbor.",
    "The road curved past the farm.",
    "Leaves covered the old bench.",
    "The moon rose over the hill.",
]

PAGE_10 = [
    "The market buzzed with voices.",
    "A cool fog settled on the bay.",
    "The old mill still turned slowly.",
    "Children raced along the shore.",
    "The porch light flickered on.",
    "Crickets chirped through the night.",
    "The bridge lights reflected below.",
]

# (page number, font path, size, lines, apply_noise)
PAGES: list[tuple[int, Path, int, list[str], bool]] = [
    (1, FONT_SANS, 20, PAGE_01, False),
    (2, FONT_SANS, 20, PAGE_02, False),
    (3, FONT_SANS, 20, PAGE_03, False),
    (4, FONT_SANS, 20, PAGE_04, False),
    (5, FONT_SERIF, 20, PAGE_05, False),
    (6, FONT_SERIF, 20, PAGE_06, False),
    (7, FONT_MONO, 18, PAGE_07, False),
    (8, FONT_SANS, 16, PAGE_08, False),
    (9, FONT_SANS, 24, PAGE_09, False),
    (10, FONT_SANS, 20, PAGE_10, True),
]


def render_page(font_path: Path, size: int, lines: list[str]) -> Image.Image:
    """Render `lines` onto a fresh white 512x720 8-bit grey page.

    Left-aligned single column, left margin 48px, top margin 56px, line
    pitch = 2x font size. Asserts every line actually fits inside
    MAX_TEXT_WIDTH at this font/size -- a failure here is an authoring bug
    (the sentence is too long for this page's font), not something to paper
    over by shrinking the font.
    """
    font = ImageFont.truetype(str(font_path), size)
    img = Image.new("L", (IMG_W, IMG_H), color=255)
    draw = ImageDraw.Draw(img)
    pitch = size * 2

    for i, line in enumerate(lines):
        assert line.isascii(), f"non-ASCII line: {line!r}"
        width = draw.textlength(line, font=font)
        assert width <= MAX_TEXT_WIDTH, (
            f"{font_path.name}@{size}px: line too wide "
            f"({width:.1f}px > {MAX_TEXT_WIDTH}px): {line!r}"
        )
        y = TOP_MARGIN + i * pitch
        assert y + pitch <= IMG_H, (
            f"{font_path.name}@{size}px: line {i} at y={y} overflows the "
            f"{IMG_H}px page height"
        )
        draw.text((LEFT_MARGIN, y), line, font=font, fill=0)

    return img


def apply_noise(img: Image.Image, seed: int = NOISE_SEED, sigma: float = NOISE_SIGMA) -> Image.Image:
    """Deterministic per-pixel Gaussian noise, applied AFTER text rendering.

    `random.seed(seed)` then one `random.gauss(0, sigma)` draw per pixel, in
    row-major `Image.get_flattened_data()` order, so the same seed always
    reproduces the same noise field regardless of platform.
    """
    random.seed(seed)
    pixels = img.get_flattened_data()
    noisy = [min(255, max(0, v + int(round(random.gauss(0, sigma))))) for v in pixels]
    out = Image.new("L", img.size)
    out.putdata(noisy)
    return out


def write_pgm(path: Path, img: Image.Image) -> None:
    w, h = img.size
    data = img.tobytes()
    assert len(data) == w * h, f"unexpected pixel buffer size for {path}"
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(data)


def write_gt(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def report(path: Path) -> None:
    data = path.read_bytes()
    print(f"{path}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")


def main(argv: list[str]) -> int:
    outdir = Path(argv[1]) if len(argv) > 1 else (Path(__file__).resolve().parent.parent / "pages")
    outdir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for num, font_path, size, lines, noise in PAGES:
        assert len(lines) == 7, f"page_{num:02d}: expected 7 lines, got {len(lines)}"
        for line in lines:
            assert line not in seen, f"duplicate line across pages: {line!r}"
            seen.add(line)

        img = render_page(font_path, size, lines)
        if noise:
            img = apply_noise(img)

        pgm_path = outdir / f"page_{num:02d}.pgm"
        gt_path = outdir / f"page_{num:02d}.gt.txt"
        write_pgm(pgm_path, img)
        write_gt(gt_path, lines)

        report(pgm_path)
        report(gt_path)

    assert len(seen) == 70, f"expected 70 distinct lines total, got {len(seen)}"
    print(f"OK: {len(PAGES)} pages, {len(seen)} distinct lines, outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
