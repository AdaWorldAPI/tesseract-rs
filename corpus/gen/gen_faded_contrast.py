#!/usr/bin/env python3
"""Deterministic generator for `corpus/quality/faded_*.pgm` — the FADED
low-contrast fixture set `.claude/harvest/binarization-roadmap.md` filed as
the missing evidence for Wolf-Jolion's actual claim.

## Why this is a DIFFERENT degradation from `gen_uneven_light.py`

`uneven_*.pgm` multiplies by an illumination FIELD: `new = old * illum(x, y)`.
That preserves every pixel's value RELATIVE TO ITS IMMEDIATE NEIGHBOURS — a
window's local standard deviation `s` stays close to its original value even
where the field has dimmed the page, because ink and background inside one
small window are dimmed by almost the same factor. That is exactly Sauvola's
home turf (local mean/std adapt to the dimming), and the measured result
confirmed it: Sauvola and Wolf were byte-identical on every `uneven_*`
fixture (`binarize_ab.rs`, 2026-07-29).

Wolf-Jolion's own claim is about a DIFFERENT failure mode: Sauvola's
threshold uses a FIXED `R = 128` denominator (`t = m*(1 - k*(1 - s/R))`), so
on a page where the ACTUAL local contrast is uniformly low (`s << 128`
everywhere, not just in one dimmed region), the `s/R` term collapses toward
zero everywhere and the threshold degenerates to `m*(1-k)` — a fixed
fraction of the local mean, insensitive to how much real contrast a window
actually has. Wolf replaces the fixed `R` with the image's own maximum local
std (`max_s`), restoring sensitivity to WHATEVER contrast the page has, low
or high.

So the fixture this method's claim needs is not a shifted illumination field
but a COMPRESSED DYNAMIC RANGE: ink and background pulled toward the same
mid-grey everywhere, which lowers `s` uniformly across the whole page rather
than only in one region.

## The model

`new = clamp(round(128 + b * (old - 128)), 0, 255)`, applied uniformly
(no `(x, y)` dependence — unlike the illumination field, contrast loss from
a faded print / worn toner / sun-bleached scan is a property of the MEDIUM,
not the lighting, so it has no spatial pattern here).

- `b = 1.0`: identity (no fade).
- `b < 1.0`: every pixel is pulled toward 128 by that fraction. A page whose
  ink sits at ~40 and background at ~230 (spread 190) becomes, at `b=0.40`:
  ink -> 128+0.40*(40-128) = 92.8 -> 93; background -> 128+0.40*(230-128) =
  168.8 -> 169 (spread 76). At `b=0.15`: ink -> 114.8 -> 115; background ->
  143.3 -> 143 (spread 28) — close to the 20-level stripe
  (`binarize::tests::wolf_recovers_faint_ink_that_sauvola_misses`) that
  already proved Wolf's threshold catches what Sauvola's misses at unit
  scale. This fixture asks the same question at page/recognition scale.

`b` values are chosen to MIRROR `gen_uneven_light.py`'s `STRENGTHS = (0.60,
0.85)` exactly (`b = 1 - strength`), so the two fixture families read as
siblings differing only in which axis they degrade — same magnitude
vocabulary, same "_060"/"_085" tag meaning "60% / 85% of the range is lost".

Outputs (all committed under `corpus/quality/`):
  * faded_060.pgm   -- b = 0.40 (60% of contrast range lost)
  * faded_085.pgm   -- b = 0.15 (85% of contrast range lost)

No `faded_clean.pgm`: at `b=1.0` this transform is the identity, which is
already covered by `uneven_clean.pgm` (same source page, unchanged) — a
redundant fixture would test nothing new.

Determinism: pure arithmetic over the existing page's own bytes -- no RNG,
no timestamps. Running twice is byte-identical; each write reports its byte
count + sha256 (same convention as `gen_uneven_light.py`).

Usage:
    python3 gen_faded_contrast.py [outdir]

`outdir` defaults to `corpus/quality` resolved relative to this script's own
location, independent of the caller's current working directory. The source
page is always read from `corpus/pages/page_01.pgm` — the SAME source
`gen_uneven_light.py` uses, so `binarize_ab.rs`'s existing `uneven_clean`
row (`FIXTURES[0]`) remains the valid CER reference for these fixtures too;
no second reference page is needed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

SOURCE_PAGE = "page_01.pgm"

# b = 1 - strength, for strength in gen_uneven_light.py's STRENGTHS (0.60,
# 0.85) -- same magnitude vocabulary as the illumination-field sibling, so
# "_060" / "_085" mean the same fraction-lost across both fixture families.
FADE_B = (0.40, 0.15)

MIDPOINT = 128.0


def parse_pgm(data: bytes) -> tuple[int, int, bytes]:
    """Minimal P5 (binary grey) PGM parser -- identical to
    `gen_uneven_light.py`'s, duplicated rather than imported since these
    generators are each self-contained single-file scripts by convention."""
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


def apply_fade(pixels: bytes, b: float) -> bytes:
    """`new = clamp(round(128 + b*(old - 128)), 0, 255)`, pixel-for-pixel --
    uniform (no spatial dependence), unlike `gen_uneven_light.py`'s
    per-pixel illumination FIELD. That is the entire point of this
    generator: the degradation this fixture models has no `(x, y)` pattern.
    """
    return bytes(
        min(255, max(0, round(MIDPOINT + b * (p - MIDPOINT)))) for p in pixels
    )


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

    for b in FADE_B:
        strength_pct = round((1.0 - b) * 100)
        tag = f"{strength_pct:03d}"
        faded = apply_fade(pixels, b)
        write_confirmed(outdir / f"faded_{tag}.pgm", w, h, faded)

    print(f"OK: source={source_path} ({w}x{h}), outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
