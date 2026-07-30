#!/usr/bin/env python3
"""Deterministic generator for `corpus/quality/skew_*.pgm` — the ROTATED
fixtures the D4 (`pixFindSkewSweepAndSearchScorePivot`) / D6
(`pixDeskewGeneral`) / D7 (`pixDeskewBoth`) byte-parity harness needs.

## Why D4/D6/D7 need MULTIPLE fixtures, not just parameter variation

D5 (`pixRotateAMGray`) is a rotation that takes an explicit angle argument, so
its own harness (`run_skew_parity.sh`) densely sweeps that argument on ONE
source page and catches the f32/f64 precision trap that way.

D4 is a DETECTOR: it takes no angle argument at all, it measures whatever
skew is already present in the page. `page_01.pgm` itself has only a tiny
inherent skew (`.claude/plans/deskew-wave-v1.md`'s own falsifier table:
angle=-0.141 degrees, conf=3.36) — barely enough to exercise the search's
interval-halving bisection, which converges almost immediately near zero.
Real coverage of the search/bisection path needs pages with SUBSTANTIALLY
different actual skew magnitudes, which only exists if the pages themselves
are rotated by different known amounts.

## The angles

Exactly the ones the plan document already used for its own manual falsifier
table (`deskew-wave-v1.md` lines 30-41) — this generator turns that one-off
manual demonstration into a committed, reproducible fixture set, closing the
exact gap the falsifiability rule (`CLAUDE.md`) exists to catch: a doc-comment
claim ("verified") that was never backed by a repeatable artifact.

    0.0, +1.5, -2.5, +5.0    (degrees)

## Why PIL, and why generation fidelity is NOT part of the parity chain

The byte-parity SUBJECT here is the DETECTOR (D4) and the deskew COMPOSITIONS
(D6/D7) — not the injection method. Once a rotated `.pgm` is written to disk
and committed, both the C++ oracle and the Rust transcode read the IDENTICAL
bytes; PIL's own resampling kernel never enters the comparison. Using PIL
(already the plan's own documented method, and already a dependency of this
repo's other `corpus/gen/*.py` scripts' toolchain) is therefore no different,
parity-wise, from using a scanner or a camera — it just needs to be
deterministic so the fixture is reproducible byte-for-byte across runs.

`expand=False` (fixed canvas, matching `page_01.pgm`'s own 512x720) with
`fillcolor=255` (white) mirrors `L_BRING_IN_WHITE`, this crate's own grey
convention (white background), and the crop-not-expand behaviour
`rotate_am_gray` itself reproduces from leptonica -- so the fixture's own
off-canvas fill matches what the detector's internal binarization + the
compositions' own rotation will assume elsewhere in the pipeline.

Determinism: `Image.rotate` with fixed inputs and no RNG is deterministic
across runs on the same Pillow version; each write reports byte count +
sha256 (same convention as every other `corpus/gen/*.py` script).

Usage:
    python3 gen_skew_fixtures.py [outdir]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

SOURCE_PAGE = "page_01.pgm"

# Degrees. Matches deskew-wave-v1.md's own falsifier table exactly (0.0 is
# skipped as a separate fixture -- page_01.pgm itself already IS the angle=0
# case and is already committed; regenerating it here would be a byte-for-
# byte-different but semantically redundant copy).
ANGLES = (1.5, -2.5, 5.0)


def write_confirmed(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"wrote {path}  bytes={len(data)}  sha256={hashlib.sha256(data).hexdigest()}")


def tag_for(angle: float) -> str:
    # e.g. 1.5 -> "p015", -2.5 -> "m025", 5.0 -> "p050" -- sign + 3-digit
    # tenths-of-a-degree, sortable and collision-free for this angle set.
    sign = "p" if angle >= 0 else "m"
    return f"{sign}{round(abs(angle) * 10):03d}"


def main() -> None:
    outdir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "quality"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    source_path = Path(__file__).resolve().parent.parent / "pages" / SOURCE_PAGE
    src = Image.open(source_path)
    assert src.mode == "L", f"expected 8bpp grey, got {src.mode}"
    w, h = src.size

    for angle in ANGLES:
        # PIL's rotate is counter-clockwise for positive angles; sign
        # convention is not asserted here (that is the DETECTOR's job to
        # measure and the harness's job to check against expectation) -- this
        # generator only needs to produce SOME real, known, non-trivial
        # rotation, deterministically.
        rotated = src.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
        assert rotated.size == (w, h), "expand=False must preserve the canvas size"
        pgm = b"P5\n%d %d\n255\n" % (w, h) + rotated.tobytes()
        tag = tag_for(angle)
        write_confirmed(outdir / f"skew_{tag}.pgm", pgm)

    print(f"OK: source={source_path} ({w}x{h}), outdir={outdir}")


if __name__ == "__main__":
    main()
