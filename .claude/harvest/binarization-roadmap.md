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
| 4 | **Singh et al.** (arXiv 1201.5227) | `m·[1 + k·(∂/(1−∂) − 1)]` where **`∂ = I(x,y) − m(x,y)`** is a **PER-PIXEL** deviation — **verified from the paper, eq. (13)**, see below | mainly a **cost** win: `O(n²)` vs `O(w²·n²)`, **flat across window size** (measured 0.19-0.25 s where Sauvola goes 7.1→13.3) | **no second integral at all** — `∂` is per-pixel, so nothing is accumulated |

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
formula plus one extra global reduction (min grey, max local stddev).

Singh reuses only the **mean** half and the application step — and this is
sharper than first assumed: because `∂ = I(x,y) − m(x,y)` is **per-pixel**,
it does not swap the mean-square integral for a different accumulator, it
**deletes that half outright**. No `pixWindowedMeanSquare`, no f64 integral,
no sqrt. Just the mean it already has, minus the pixel.

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

**✅ EQUATION OBTAINED** (operator supplied page screenshots, 2026-07-29 —
the PDF text layer could not yield it; see the note at the end of this
section for what failed and why).

### Eq. (13) — the proposed threshold

```
T(x,y) = m(x,y) · [ 1 + k · ( ∂(x,y) / (1 − ∂(x,y)) − 1 ) ]
```

with **`∂(x,y) = I(x,y) − m(x,y)`**, the *local mean deviation*, and `k` a
bias with range **[0,1] only**.

**The thing that could not have been guessed, and that IS the cost win:**
`∂` is **per-pixel** — the pixel minus its own local mean — **not a windowed
statistic**. There is nothing to accumulate, so no second integral image
exists. Sauvola needs the mean-square integral to get `δ`; Singh needs
nothing beyond the mean it already has.

Supporting definitions from the same paper (all now verified, not recalled):

| eq | formula | notes |
|---|---|---|
| (1) | `g(x,y) = ΣΣ I(i,j)` | integral sum image (Viola-Jones) |
| (2-4) | `g(x,y) = I(x,y) + g(x,y−1) + g(x−1,y) − g(x−1,y−1)` | single-pass build |
| (6) | `s(x,y) = [g(x+d−1,y+d−1) + g(x−d,y−d)] − [g(x−d,y+d−1) + g(x+d−1,y−d)]`, `d = round(w/2)` | window sum, 2 adds + 1 sub, **window-size independent** |
| (7) | `m(x,y) = s(x,y) / w²` | local mean |
| (8) | `b(x,y) = 0 if I(x,y) ≤ T(x,y) else 1` | **note `I(x,y) ∈ [0,1]`** |
| (9) | `T = m + k·δ` | Niblack, `k = −0.2`, `w = 15` |
| (10) | `T = m·[1 + k·(δ/R − 1)]` | Sauvola, `R = 128`, `k ∈ [0.2, 0.5]` |
| (11-12) | `T = 0.5·(Imax + Imin)`, contrast `C = Imax − Imin ≥ 15` | Bernsen, `w = 31` |

Eq. (10) **exactly matches** what Wolf's C++ has for Sauvola — two
independent sources agreeing, which is the cross-check worth having.

### Behaviour the paper states explicitly (useful for the port's tests)

- `k = 0` ⟹ `T = m`, the plain local mean.
- Uniform window ⟹ `I(x,y) = m(x,y)` ⟹ `∂ = 0` ⟹ `T < m` ⟹ pixel is
  **background**. This is the blank-region behaviour Niblack gets wrong.
- `m = 0` ⟹ `T = 0` ⟹ pixel is background.
- Lower `k` raises the threshold; higher `k` lowers it.

### ⚠ Implementation gotcha the formula hides

`∂/(1−∂)` has a **singularity at `∂ = 1`**, reachable when `I = 1` and
`m = 0` (a white pixel in a fully black window). Intensities MUST be
normalized to `[0,1]` per eq. (8) — on a raw `0..=255` buffer the term is
meaningless. Guard the denominator.

Note also that `∂ = I − m` is **signed**: a pixel darker than its own local
mean gives `∂ < 0`, so `∂/(1−∂)` is negative and the bracket drops below
`1 − k`. That is the intended discrimination, but it means the term is NOT
bounded in `[0,1]` the way `s/R` is in Sauvola — an implementation that
clamps `∂` to non-negative would silently delete half the method.

### ⚠ The paper contradicts itself on `k` — do not trust one reading

Eq. (13)'s own text gives **`k ∈ [0,1]`**, and the document-comparison figure
uses `k = 0.06`. But the window-size figure states **`k = 15`** (against
Sauvola's `k = 5` in the same figure). Those cannot both be the `k` of eq.
(13) — at `k = 15` the bracket `1 + 15·(∂/(1−∂) − 1)` is wildly negative for
any `∂ < 1`, making `T` negative and every pixel foreground.

**Consequence for the port: `k` must be swept empirically, not taken from the
paper.** Treat `[0, 1]` as the live range (it is the one attached to the
equation itself) and treat the `k = 15` figure as either a different
parameterization or an erratum. This is exactly the "`k` is not comparable
across methods" warning Wolf's repo gives, showing up *within* one paper.

### Measured timing (Table 1, Lena 512×512, seconds)

| window | **Proposed** | Bernsen | Niblack | Sauvola |
|---|---|---|---|---|
| 3 | 0.2496 | 0.8112 | 7.176 | 7.1448 |
| 15 | 0.234 | 3.4164 | 8.5177 | 8.5489 |
| 35 | **0.1872** | 12.0589 | 13.2913 | 13.3225 |

Singh is **flat** (~0.19-0.25 s) across a 12× window-size range while Sauvola
nearly doubles. That is the `O(n²)` vs `O(w²·n²)` claim, measured — and it
does not get *faster* than Sauvola at small windows by much, so the win is
specifically **window-size independence**, not raw speed.

One more datum from the Sauvola discussion worth carrying: the paper reports
`k = 0.5` as used by Sauvola/Sezgin/Badekas but finds **`k = 0.34` gives the
best results**, adding that "the algorithm is not very sensitive to the value
of k".

### What the PDF text layer could not give (kept as a method note)

The PDF's formulas are set in a math font with custom encoding and extract as
glyph soup (`10),(yxb ifotherwiseyxTyxI`). Recording it as *unobtained*
rather than reconstructing from memory was the right call: the same file had
just carried a wrong Wolf formula written from recollection, and the real
Singh equation — a **per-pixel** `∂`, not a windowed one — is not what a
plausible reconstruction would have produced.

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

---

# SHIPPED + MEASURED (2026-07-29)

Both rungs are implemented in `crates/tesseract-ocr/src/binarize.rs`
(`wolf_binarize` / `singh_binarize`, sharing Sauvola's parity-proven
`windowed_stats` front half) and selectable end-to-end via
`BinarizeMode::{Wolf, Singh}` through BOTH the layout path (`xy_cut`) and the
text path (`segment.rs`). Option (2) was taken: **quality fence, not parity**,
stated in the module docs as the repo rule requires.

## The measurement — `examples/binarize_ab.rs`, 4 modes × 5 fixtures

| mode | mean_cer | mean_ink_frac |
|---|---|---|
| otsu | 0.3041 | 0.3898 |
| **sauvola** | **0.0045** | 0.0279 |
| wolf | 0.0054 | 0.0285 |
| singh | 0.0090 | 0.0303 |

**All three adaptive methods recover the full 42-word text on every degraded
fixture**; Otsu drops to 18-32 words. That part is unambiguous.

**But Wolf did not beat Sauvola — and the reason is the fixtures, not Wolf.**
On all four *degraded* fixtures Wolf and Sauvola are byte-identical
(0.0045/0.0045, 0.0181/0.0181, 0.0000/0.0000, 0.0000/0.0000). Wolf's only
difference is on the CLEAN page, where it introduces one character of error
Sauvola does not.

That is the expected result once stated plainly: **`uneven_*.pgm` is uneven
ILLUMINATION, not FADED contrast.** Those pages have full local contrast and
merely a shifting background — Sauvola's home turf, the case its local mean
already handles. Wolf's claim is specifically about low-contrast source, where
`s ≪ 128` collapses the fixed `R` and `t → m`. **The probe never exercised the
failure mode Wolf exists to fix.**

Wolf's claim IS measured, at unit scale, by
`binarize::tests::wolf_recovers_faint_ink_that_sauvola_misses`: a 20-grey-level
stripe on a 200-grey field. Measured thresholds — Sauvola `t(ink)=136` (ink at
180 sails over it, MISSED), Wolf `t(ink)=191` (ink at 180 falls under it,
CAUGHT), neither flooding the background. Two-sided on the same fixture, so it
cannot pass if `wolf_binarize` were merely a second name for Sauvola.

**Consequences, in priority order (§1-3 below are the ORIGINAL 2026-07-29
findings; the faded-arm result that closes §1 follows immediately after):**

1. ~~**The fixture set needs a FADED arm.**~~ **CLOSED — see below.**
   `gen_uneven_light.py` multiplies by an illumination field; a faded page
   needs the dynamic range COMPRESSED (`grey → a + b·grey`, `b ≪ 1`).
2. **Sauvola remains the default-flip candidate for MODERATE degradation** —
   see below for why this needed a qualifier it didn't have before.
3. **Singh's claim is COST, and cost was not measured here.** It is ~2× Sauvola's
   CER on these fixtures (0.0090 vs 0.0045 — about one character in 226) and
   loses one word on the clean page. Its selling point is window-size
   independence (paper Table 1: flat 0.19-0.25 s where Sauvola goes
   7.1→13.3 s), which this probe does not time. Judging Singh on CER alone
   would be judging it on the axis it does not compete on.

---

## The faded arm — built, measured, and the result is SHARPER than expected (2026-07-29)

`corpus/gen/gen_faded_contrast.py` closes §1: `new = clamp(round(128 +
b·(old−128)), 0, 255)`, a UNIFORM compression toward mid-grey (no `(x,y)`
dependence — unlike the illumination field, print fade has no spatial
pattern). `b` values mirror `gen_uneven_light.py`'s `STRENGTHS` exactly
(`b = 1 − strength`), so `faded_060`/`faded_085` mean the same "60%/85% of
range lost" as their `uneven_*` siblings. Verified: global pixel stdev drops
36.5 (clean) → 14.6 (`faded_060`) → 5.4 (`faded_085`); deterministic
(byte-identical across repeated runs, checked by hash).

Extended `binarize_ab.rs`'s existing `FIXTURES` array (same harness, same
CER reference — no new probe written). Full re-run, all 7 fixtures × 4 modes:

| fixture | mode | words | CER |
|---|---|---|---|
| faded_060 | otsu / sauvola / wolf / singh | 42/42/42/42 | 0.0000/0.0000/0.0045/0.0045 |
| **faded_085** | **otsu** | **42/42** | **0.0000** |
| faded_085 | **sauvola** | **0/42** | **1.0000** |
| faded_085 | **wolf** | **42/42** | **0.0045** |
| faded_085 | **singh** | **42/42** | **0.0000** |

**`faded_060` (moderate) breaks nothing** — all four modes read the full
page. Only `faded_085` (severe, spread compressed to 38 grey levels)
produces a real split, and the split is not the one a first guess predicts.

### Otsu is FINE here — the naive "faded breaks global thresholding" story is wrong

Measured page histogram (`faded_085.pgm`, 368,640 px): a spike of 355,298 px
(96.4%) at grey 147 (background), a cluster of 4,624 px (1.25%) near grey 109
(ink), antialiasing between. **A uniform monotonic compression preserves the
histogram's bimodal SHAPE** — Otsu's threshold depends on that shape, not on
the absolute contrast, so it still finds the valley and reads the page
perfectly. Faded contrast alone does not defeat a global histogram threshold.

### Sauvola specifically fails, and by exactly the mechanism the formula predicts

Ink is SPARSE (1.25% of pixels), so almost every local window is nearly all
background. Local mean `m` sits near 147; with `s ≪ 128` UNIFORMLY (not just
regionally, as in the illumination fixtures), `t = m·(1−k·(1−s/128))`
collapses toward `m·(1−k)` regardless of what a window actually contains. At
`m≈147, k=0.34`: `t ≈ 97` — comfortably ABOVE the real ink value (109), so
every ink pixel fails `grey < t` and reads as background. Not "degraded
recognition" — the WHOLE PAGE returns zero words, confirmed independently by
`ink_frac = 0.0000` (computed directly from `binarize_page_with`, entirely
outside the OCR pipeline).

### Wolf and Singh both recover, by two INDEPENDENT mechanisms

Wolf's `max_s` renormalization was built for exactly this; Singh's per-pixel
`∂ = I−m` has no fixed `R` to collapse in the first place, so it isn't even
susceptible to the same failure mode. Two different fixes agreeing on one
page is corroboration, not redundancy.

### Pinned as a fast deterministic unit test, not left as a corpus-gated finding

`binarize::tests::sauvola_fails_on_sparse_ink_at_faded_contrast_but_wolf_and_singh_recover`
reproduces the same qualitative shape at unit scale (sparse ink block, not a
full-height stripe — density is what drives the collapse here, contrast
alone is not sufficient) at the REAL measured cluster centres (147/109, not
round numbers). Measured thresholds confirm the hand-derivation exactly:
`sauvola t(ink)=97` (above 109 → miss), `wolf t(ink)=135` (below → catch),
`singh t(ink)=126` (below → catch). This complements
`wolf_recovers_faint_ink_that_sauvola_misses` (a DENSE full-height stripe at
a different mean level, 200/180) rather than duplicating it — sparse-vs-dense
matters and both are now covered.

### What this changes about "Sauvola remains the default-flip candidate"

That conclusion isn't reversed, but it loses its unconditional phrasing. On
MODERATE degradation (both `uneven_*` fixtures and `faded_060`) Sauvola is at
least as good as everything else and beats Otsu by a wide margin — nothing
here disturbs that. On SEVERE, uniformly low-contrast pages, Sauvola is not
merely worse than Wolf/Singh, it is **catastrophically, silently wrong** —
zero output with no error, which is a worse failure mode than Otsu's own
partial-word degradation on the same class of input. A production default
that could hit `faded_085`-shaped input would need Wolf or Singh as a
fallback, or at minimum a "zero words recognized" sentinel — not a decision
to make from this measurement alone, but the asymmetry is now on record
rather than assumed away.

## Implementation notes worth keeping

- **Singh's pole cancels — do not transcribe eq. (13) literally.** `∂/(1−∂)`
  diverges at `∂ = 1`, and the outer `m·` factor then gives `0 · inf = NaN`.
  Rearranged as `T = m + k·(m·∂/(1−∂) − m)` with `den = 1 − I + m`, the risky
  product is `m·(I−m)/den`, whose numerator carries a matching factor of `m`;
  the limit is `1`, so `T → k`. Pinned by
  `singh_singularity_yields_the_cancelled_limit_not_nan`, which asserts the
  exact limit value (`15 == (0.06·255) as u8`) rather than merely "not NaN" —
  a NaN guard maps to `0` and would otherwise look valid.
- **Wolf's `u8` threshold clamp is exact, not lossy.** `s/max_s − 1 ≤ 0` and
  `m − min_I ≥ 0`, so for `k ≥ 0` the correction is non-positive and
  `t ≤ m ≤ 255` — it can never overflow upward. `t < 0` is reachable, and
  clamping to `0` is decision-identical (no `u8` grey satisfies `grey < 0`).
- **`max_s == 0`** (a flat page) is `0/0`; defined as `0`, which lands on
  `t = m` and gives all-background — the right answer for a blank page and
  the `0/0` limit anyway.
- **`k` is per-method and NOT transferable.** The probe holds `whsize = 16`
  constant across modes (isolating the closing formula) but deliberately uses
  each method's own default `k`. So a losing row means "did not win untuned",
  never "is worse".

---

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

> **⊘ HISTORICAL — both are now shipped, this ordering plan is obsolete.**
> Written before implementation; kept for provenance (append-only), not as
> current guidance. Both Wolf and Singh landed together (`crates/
> tesseract-ocr/src/binarize.rs`), the equation was obtained (see above), and
> the faded-arm measurement found Singh performing AT LEAST as well as Wolf
> on the fixture that mattered (`faded_085`: wolf 42/42 words CER 0.0045,
> singh 42/42 words CER 0.0000) — so "Singh second" never described a real
> quality gap once both existed. For current status, defaults, and the
> measured findings, see "SHIPPED + MEASURED" and "The faded arm" sections
> above, both dated 2026-07-29.
