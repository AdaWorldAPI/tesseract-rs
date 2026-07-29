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
| 3 | **Wolf-Jolion** (next) | `m + k·(s/max_s − 1)·(m − min_I)` — **verified from Wolf's own source**, see below | Sauvola's weakness on **low-contrast / faded** regions — Sauvola's `s/R` uses a *fixed* `R=128`, so on a washed-out scan where `s ≪ 128` the term collapses and `t → m`, under-thresholding. Wolf rescales by the image's ACTUAL max stddev. | + one global reduction for `max_s`, `min_I` |
| 4 | **Singh et al.** (arXiv 1201.5227) | local **mean deviation** instead of standard deviation — **exact equation not yet obtained**, see below | mainly a **cost** win: `O(n²)` vs `O(w²·n²)`, and independent of window size | cheapest of 2-4 |

> **⚠ The Wolf row above was CORRECTED 2026-07-29.** It previously carried
> `(1−a)·m + a·M + a·s/S·(m−M)`, written from recollection. The verified form
> from `binarizewolfjolion.cpp:183` is `m + k·(s/max_s − 1)·(m − min_I)`. The
> two are not equivalent, and the *reason* Wolf helps (§"What Wolf actually
> changes") only reads correctly from the verified one. Left visible rather
> than silently edited, because a plausible-looking wrong formula in a
> roadmap is precisely what gets implemented six weeks later.

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

---

# Verified formulas (researched 2026-07-29, from primary sources)

Everything below is from the **reference implementation or the paper itself**,
not recollection. Where a source did not yield something, that is said plainly
rather than filled in.

## The three threshold formulas — from Wolf's own C++

`chriswolfvision/local_adaptive_binarization`, `binarizewolfjolion.cpp:174-183`
— the canonical implementation, cited by the author as the one to use:

```cpp
case NIBLACK:     th = m + k*s;
case SAUVOLA:     th = m * (1 + k*(s/dR - 1));           // dR = 128, FIXED
case WOLFJOLION:  th = m + k * (s/max_s - 1) * (m - min_I);
```

| symbol | meaning |
|---|---|
| `m` | local mean over the `winx × winy` window |
| `s` | local standard deviation |
| `k` | sensitivity; **default 0.5** for Sauvola/Wolf, but Niblack wants **−0.2** (the repo warns the parameter is NOT transferable between methods) |
| `dR` | Sauvola's **fixed** dynamic-range constant, 128 |
| `max_s` | **maximum local stddev over the WHOLE image** (`calcLocalStats` returns it) |
| `min_I` | **global minimum grey** (`minMaxLoc`) |

### What Wolf actually changes, precisely

Two substitutions, and both matter:

1. **`s/dR` → `s/max_s`.** Sauvola divides the local stddev by a *constant*
   128. On a faded or low-contrast scan every `s ≪ 128`, so the term collapses
   and `t → m` — Sauvola under-thresholds exactly where contrast is worst.
   Wolf divides by the image's **actual** maximum local stddev, so the term
   spans its full range whatever the document's contrast.
2. **`m·(…)` → `(m − min_I)·(…)`.** Sauvola scales the correction by the local
   mean; Wolf scales it by the distance from the **darkest pixel in the
   image**, anchoring the correction to the real ink level rather than to
   however bright this particular window happens to be.

### The implementation consequence that shapes our port

**Wolf is inherently two-pass over the whole image.** `max_s` and `min_I` are
global reductions, so the entire local-stddev map must exist before a single
threshold can be computed. Sauvola can stream. Our `binarize.rs` already
builds the windowed mean and mean-square integrals, so `s` per pixel is
available — Wolf needs one added sweep to reduce for `max_s`/`min_I`, then the
existing threshold sweep. No new machinery, one extra pass.

Provenance: Wolf, Jolion & Chassaing, *Text Localization, Enhancement and
Binarization in Multimedia Documents*, ICPR 2002, vol. 4, pp. 1037-1040. The
repo reports the method placed **5th in DIBCO 2009**.

## Singh et al. — arXiv 1201.5227

*A New Local Adaptive Thresholding Technique in Binarization*, T. Romen Singh,
Sudipta Roy, O. Imocha Singh, Tejmani Sinam, Kh. Manglem Singh. IJCSI vol. 8,
Nov 2011.

**⚠ The exact equation could NOT be extracted.** The PDF's formulas are set in
a math font with custom encoding and come out of the text layer as glyph soup
(`10),(yxb ifotherwiseyxTyxI`). Everything below is from the paper's *prose*,
which is unambiguous; the closing formula must be read off the rendered PDF or
a reference implementation before coding. **Do not reconstruct it from
memory — that is exactly the kind of paraphrase this repo has been burned by.**

What the prose establishes, verbatim in substance:

- Thresholds on **local mean and mean deviation** — explicitly "does not
  involve calculations of standard deviations as in other local adaptive
  techniques".
- Uses an **integral sum image** for the local mean, so mean computation is a
  single pass **independent of window size**.
- **Complexity: O(n²)**, against **O(w²·n²)** for the others on an `n×n` image.
  This is the paper's headline claim and the real reason to care.
- **Works at very small windows** (5×5) where Niblack and Bernsen fail
  outright; the authors note Niblack/Bernsen "require large window size".
- Parameter regime differs sharply per figure: `k = 0.06` in the document
  comparison (Sauvola also at 0.06 there), but `k = 15` vs Sauvola's `k = 5`
  in the window-size figure. **`k` is not comparable across methods** — the
  same warning Wolf's repo gives.

One citation from the paper worth carrying, because it bears on our use case:
the authors cite Sezgin & Sankur's comparative analysis finding Sauvola "is
the best on non documental images, **but somewhat poor in document images**".
Ours are document images.

## Parity status — unchanged and still the deciding constraint

leptonica implements **Sauvola only**. Neither Wolf nor Singh can be a
byte-parity leaf against `liblept`. Options remain as recorded above: oracle
against Wolf's OpenCV reference (preferred — keeps the discipline, at the cost
of a different harness), or drop to the quality-fence footing and **say so in
the module docs**. Singh has no reference implementation identified yet, which
pushes it toward the fence.

## Ordering, revised by this research

**Wolf first, and by a wider margin than the roadmap assumed.** Its formula is
verified, its reference implementation is in hand, its failure mode is exactly
the one our corpus has (faded/low-contrast where `s ≪ 128`), and it reuses
every integral already built — one extra reduction pass.

**Singh second, and gated on obtaining the equation.** Its win is asymptotic
cost, not quality, and cost is not currently a measured problem for us. It is
the right rung when window size becomes a tuning burden; it is the wrong rung
to build from a half-extracted PDF.
