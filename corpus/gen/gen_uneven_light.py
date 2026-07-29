#!/usr/bin/env python3
"""Deterministic generator for `corpus/quality/uneven_*.pgm` — the
uneven-illumination A/B fixture set that falsifies whether Sauvola adaptive
binarization actually survives what a single global Otsu threshold cannot.

The existing rendered corpus (`corpus/pages/*.pgm`, `corpus/quality/resgrid.pgm`)
is cleanly and evenly lit by construction, so Sauvola's claimed advantage is
invisible there. This script takes ONE existing clean corpus page
(`corpus/pages/page_01.pgm` — read, not regenerated; see gen_pages.py for its
own provenance) and, alongside a copy of it unchanged, emits illumination-
degraded variants under a multiplicative illumination field: a linear
left->right gradient and a radial vignette, each at two strengths.

Model: `new = clamp(round(old * illum(x, y)), 0, 255)` -- a physically
motivated "uneven light source" simulation. `illum` ranges from `1.0` (full
original brightness) down to `1.0 - strength` at the most-dimmed point
(gradient: the far edge; vignette: the farthest corner). At the higher
strength this pushes the dimmed region's background value down far enough
that a SINGLE global pixel-value threshold -- calibrated mostly by the
page's brighter majority -- cannot also correctly separate that region's own
ink from its own (now much darker) background; a human reading the page (or
an adaptive/local threshold) can still track ink-darker-than-its-own-local-
background throughout, since the illumination field preserves EVERY pixel's
value relative to its immediate neighbours.

Outputs (all committed under `corpus/quality/`):
  * uneven_clean.pgm          -- the source page, copied through unchanged
                                  (the CER reference in binarize_ab.rs is
                                  THIS page's own Otsu recognition, not an
                                  external transcript).
  * uneven_linear_060.pgm     -- linear gradient, strength 0.60 (far edge
                                  retains 40% brightness).
  * uneven_linear_085.pgm     -- linear gradient, strength 0.85 (far edge
                                  retains 15% brightness).
  * uneven_vignette_060.pgm   -- radial vignette, strength 0.60 (farthest
                                  corner retains 40% brightness).
  * uneven_vignette_085.pgm   -- radial vignette, strength 0.85 (farthest
                                  corner retains 15% brightness).

Determinism: pure arithmetic over the existing page's own bytes -- no RNG,
no timestamps. Running twice is byte-identical; each write reports its byte
count + sha256 (the same convention gen_pages.py / gen_resolution_grid.py
use -- the cheap "did anything actually change" check).

Usage:
    python3 gen_uneven_light.py [outdir]

`outdir` defaults to `corpus/quality` resolved relative to this script's own
location (`corpus/gen/gen_uneven_light.py` -> `corpus/quality`), independent
of the caller's current working directory. The source page is always read
from `corpus/pages/page_01.pgm`, resolved the same way.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

SOURCE_PAGE = "page_01.pgm"

# (strength) values used for BOTH the linear gradient and the radial
# vignette: the fraction of original brightness LOST at the most-dimmed
# pixel. 0.60 -> darkest point retains 40% brightness; 0.85 -> retains 15%.
# Tuned to be strong enough that a single global Otsu split cannot also
# correctly separate the dimmed region's own ink from its own background
# (see `crates/tesseract-ocr/examples/binarize_ab.rs` for the measured
# consequence), while remaining a physically plausible "uneven light
# source" a human reads through easily.
STRENGTHS = (0.60, 0.85)


def parse_pgm(data: bytes) -> tuple[int, int, bytes]:
    """Minimal P5 (binary grey) PGM parser: whitespace-separated
    `P5 W H MAXVAL` header (`#`-comments tolerated though never written by
    this repo's generators), then exactly W*H raw pixel bytes. Mirrors the
    writer in gen_pages.py (`P5\\n{w} {h}\\n255\\n` + raw bytes) exactly.
    """
    if not data.startswith(b"P5"):
        raise ValueError("not a P5 PGM")
    pos = 2
    fields: list[int] = []
    while len(fields) < 3:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if pos < len(data) and data[pos : pos + 1] == b"#":
            while pos < len(data) and data[pos : pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1  # the single mandatory whitespace byte separating maxval from pixel data
    w, h, maxval = fields
    assert maxval == 255, f"unsupported maxval {maxval}"
    pixels = data[pos : pos + w * h]
    assert len(pixels) == w * h, f"truncated pixel data: {len(pixels)} != {w * h}"
    return w, h, pixels


def write_pgm(path: Path, w: int, h: int, pixels: bytes) -> None:
    assert len(pixels) == w * h, "pixel buffer does not match w*h"
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(pixels)


def linear_illum(w: int, h: int, strength: float) -> list[float]:
    """Bright at the left edge (factor 1.0) fading to dim at the right edge
    (factor `1 - strength`), constant down each column. Row-major, w*h
    entries."""
    col_factor = [1.0 - strength * (x / (w - 1)) for x in range(w)]
    return [col_factor[x] for _y in range(h) for x in range(w)]


def vignette_illum(w: int, h: int, strength: float) -> list[float]:
    """Bright at the centre (factor 1.0) fading radially to dim at the
    farthest corner (factor `1 - strength`). Row-major, w*h entries."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    max_dist = math.hypot(cx, cy)
    out = [0.0] * (w * h)
    for y in range(h):
        row = y * w
        dy = y - cy
        for x in range(w):
            dist = math.hypot(x - cx, dy)
            out[row + x] = 1.0 - strength * (dist / max_dist)
    return out


def apply_illum(pixels: bytes, factors: list[float]) -> bytes:
    """`new = clamp(round(old * factor), 0, 255)`, pixel-for-pixel."""
    assert len(pixels) == len(factors), "factor field size mismatch"
    return bytes(min(255, max(0, round(p * f))) for p, f in zip(pixels, factors))


def write_confirmed(path: Path, w: int, h: int, pixels: bytes) -> None:
    write_pgm(path, w, h, pixels)
    data = path.read_bytes()
    print(f"{path}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")


def main(argv: list[str]) -> int:
    script_dir = Path(__file__).resolve().parent
    outdir = Path(argv[1]) if len(argv) > 1 else (script_dir.parent / "quality")
    outdir.mkdir(parents=True, exist_ok=True)

    source_path = script_dir.parent / "pages" / SOURCE_PAGE
    w, h, pixels = parse_pgm(source_path.read_bytes())

    # Baseline: the clean page copied through unchanged.
    write_confirmed(outdir / "uneven_clean.pgm", w, h, pixels)

    for strength in STRENGTHS:
        tag = f"{round(strength * 100):03d}"

        linear = apply_illum(pixels, linear_illum(w, h, strength))
        write_confirmed(outdir / f"uneven_linear_{tag}.pgm", w, h, linear)

        vignette = apply_illum(pixels, vignette_illum(w, h, strength))
        write_confirmed(outdir / f"uneven_vignette_{tag}.pgm", w, h, vignette)

    print(f"OK: source={source_path} ({w}x{h}), outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
