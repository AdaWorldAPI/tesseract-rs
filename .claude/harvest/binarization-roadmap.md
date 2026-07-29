# Local adaptive binarization — the family roadmap

Operator direction (2026-07-29): *"after sauvola is wolf"* →
<https://github.com/chriswolfvision/local_adaptive_binarization> ·
<https://chriswolfvision.github.io/www/>, then
*"A New Local Adaptive Thresholding Technique in Binarization"* — T. Romen
Singh, Sudipta Roy, O. Imocha Singh, Tejmani Sinam, Kh. Manglem Singh,
<https://arxiv.org/pdf/1201.5227>.

These are **one family with one shape**: a per-pixel threshold computed from a
local window's statistics, differing only in *which* statistics and *how they
are normalized*. That matters for this repo because the expensive machinery —
the mirrored border, the integral images, the windowed accumulators — is
**already built and byte-parity green** in `binarize.rs` for Sauvola. Each
successive method is a different closing formula over the same windows, not a
new pipeline.

## The ladder

| | Method | Threshold | What it fixes | Cost |
|---|---|---|---|---|
| 0 | **Otsu** (shipped, `threshold.rs`) | one global `t` | — | histogram only |
| 1 | Niblack | `m + k·s` | local contrast | windowed mean + stddev |
| 2 | **Sauvola** (shipped, byte-parity) | `m·(1 − k·(1 − s/R))`, `R=128` | Niblack's noise explosion in blank regions | same windows |
| 3 | **Wolf-Jolion** (next) | normalizes against **global** min grey `M` and **global max** local stddev `S`: `(1−a)·m + a·M + a·s/S·(m−M)` | Sauvola's weakness on **low-contrast / faded** regions — Sauvola's `s/R` uses a *fixed* `R=128`, so on a washed-out scan where `s ≪ 128` the term collapses and `t → m`, under-thresholding. Wolf rescales by the image's ACTUAL max stddev. | + one global pass for `M`, `S` |
| 4 | **Singh et al.** (arXiv 1201.5227) | local **mean deviation** instead of standard deviation | mainly a **cost** win — mean deviation needs no squared integral image, so it drops the f64 `pixMeanSquareAccum` half of the Sauvola chain | cheapest of 2-4 |

## Why this is cheap to build here

`binarize.rs` already carries, byte-parity-proven against liblept 1.82.0:

- `pixAddMirroredBorder(whsize+1)` — the border handling,
- `pixWindowedMean` — the u32 **wrapping** integral (`blockconvAccumLow`),
- `pixWindowedMeanSquare` — the f64 integral (`pixMeanSquareAccum`),
- `pixApplyLocalThreshold` — the `grey < t` application.

Wolf-Jolion reuses **all four** and changes only `pixSauvolaGetThreshold`'s
formula plus one extra global reduction (min grey, max local stddev). Singh
reuses the first and last and swaps the mean-square integral for a mean-deviation
one.

## The parity question — read before starting

Sauvola was byte-parity-able because **leptonica implements it**
(`pixSauvolaBinarize`), so an oracle existed. **Leptonica does NOT implement
Wolf-Jolion or Singh.** So neither can be a byte-parity leaf against liblept.

Two honest options, decide before writing code:

1. **Oracle against the reference implementation.** Chris Wolf's repo is the
   canonical Wolf-Jolion source (`NiblackSauvolaWolfJolion.cpp`, which
   dispatches Niblack / Sauvola / Wolf / NICK from one file). Build it as the
   oracle the same way `skew_oracle.cpp` links liblept. This keeps the repo's
   byte-parity discipline intact and is the preferred path — with the caveat
   that it is an OpenCV-based reference, so the harness differs from the
   existing liblept oracles.
2. **Quality fence, not parity** — the `structured.rs` / `rectify.rs` / quality-CI
   footing: no oracle exists, so pin *measured* behaviour against generated
   ground truth instead. The `binarize_ab` probe + `corpus/quality/uneven_*.pgm`
   fixtures already built for the Sauvola measurement are exactly the harness
   this would extend (add a mode column; `ink_frac` is the discriminating
   metric — see `sauvola-vs-otsu-probe.md`).

Prefer (1) if the reference builds cleanly; fall back to (2) and **say so in the
module docs**, per the repo's rule that a non-parity leaf must never be
presented as a parity leaf.

## The wiring caveat that applies to every method in this family

Measured 2026-07-29 (`sauvola-vs-otsu-probe.md`): `BinarizeMode` reaches the
layout / region / table pass **but not the text path** — `segment.rs` holds a
third, independent, always-Otsu binarizer that word and line recognition runs
through. So *any* new binarization mode improves region classification only
until that threading is done. Adding Wolf without it would produce the same
"CER identical between modes" result Sauvola just produced, for the same
structural reason. Thread `segment.rs` first, or state the limit explicitly.
