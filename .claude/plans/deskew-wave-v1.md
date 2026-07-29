# Deskew wave v1 — `pixFindSkew` + `pixRotate`, byte-parity

> **Read the manifest first:** `.claude/harvest/leptonica-skew-callgraph.txt`
> (1039 lines) is the SPEC — call graphs, exact constants, the leaf
> classification, and the per-site precision audit. This file is the
> *sequencing and design* layer on top of it, not a restatement. Where the
> two disagree, the manifest wins (it was read from source; this was
> reasoned from it).

## Why this wave

Two things are blocked on it, and one is user-visible:

1. **The rotational half of page geometry.** `rectify.rs` corrects *keystone*
   (a height-dependent shear ramp). It does not correct *rotation*, and its
   module docs say so. A photographed or copied page has both. Deskew is the
   missing half.
2. **`decide_if_table` steps 1-4.** `pixPrepare1bpp` (ppi-normalize) +
   `pixDeskewBoth` is the documented front-end gap that keeps table detection
   running at the page's own resolution instead of ppi-exact.

## Provenance + why parity is achievable here

`/tmp/leptonica-src` is at tag **1.82.0**, an exact match for the installed
`liblept` — **zero ABI/version skew**, the same clean footing the Sauvola leaf
had, not the 5.5.0-header/5.3.4-lib skew the older waves fought.

Oracle: `.claude/harvest/oracles/skew_oracle.cpp` — built, and all five arms
exercised on real corpus data before any Rust was written. Falsifier evidence
(`page_01.pgm` rotated by PIL to known angles, then detected):

| applied | detected | conf |
|---|---|---|
| 0.0° | −0.141° (the page's own baseline) | 3.36 |
| +1.5° | +1.609° | 3.94 |
| −2.5° | −2.594° | 3.56 |
| +5.0° | +5.156° | 2.40 |

Correct sign, right magnitude, confidence degrading with skew. This fixture
can *fail* a bad transcode — unlike the uniform-input diffs that made the CTC
`simple_text` bug unfalsifiable for a whole arc.

## Two design facts that shape everything

**1. Leptonica's rotation CROPS.** `pixRotateAMGray` and `pixDeskew` both
return the same `w × h` they were given (measured: 512×720 → 512×720), corners
filled with `grayval`. That is precisely the eager-cropping failure
`rectify.rs` was rebuilt to avoid — relocated one layer down.

*Resolution:* byte-parity the leaf **as it is** (parity means reproducing the
crop, not improving on it), and reach expansion through the parameters
leptonica already provides — `pixRotate` takes explicit `width, height`, and
`pixEmbedForRotation` is the sizing path behind them. So the expanding
behaviour lives *inside* the faithful transcode, not in a non-parity wrapper.
v1 passes `0, 0` (the no-op clone-through the deskew compositions use); the
expanding path is a deliberate v2, gated on v1 being green.

**2. Grey is raw, binary is flipped.** Verified, because the harvest flagged it
as the top unknown: `image_input::parse_pgm` reads P5 samples straight through,
and `xy_cut` documents grey as "white background ≈ 255, dark ink ≈ 0" — i.e.
**8bpp grey is leptonica-native, no inversion**. Only this crate's *binary*
buffers use `0 = ON`, against leptonica's `1 = ON`. Off-edge fill
(`L_BRING_IN_WHITE` → 255 in grey) therefore transcribes directly. Getting this
backwards would invert fill semantics without necessarily failing a fixture
whose rotation exposes no edges — so it is pinned here.

## Sub-leaf sequence

Ordered so each leaf has its own oracle arm and nothing depends on an unproven
predecessor. **One leaf per commit**, each byte-parity green before the next.

| | Leaf | C source | Oracle arm | Notes |
|---|---|---|---|---|
| **D1** | `pixRotate90` (d==8) | `rotateorth.c:222-232` CW / `:310-320` CCW | (new arm) | Pure index remap, **no float at all**. Cheapest leaf; needed by D7. Warm-up that proves the harness. |
| **D2** | `pixFindDifferentialSquareSum` | `skew.c:1138-1160` | `dss` | The score the sweep maximizes. **f32 sequential accumulation, same row order** — an f64 accumulator will NOT match (audit §10). `skiph = (0.05_f64 * w as f64) as i32` (§9). |
| **D3** | `pixVShearCorner` / `pixVShearCenter` | `shear.c` | (via `dss`) | The sweep's own rotation, on 1bpp. Lives in `shear.c`, which was **not** in the original scope — the harvest caught that. |
| **D4** | `pixFindSkewSweepAndSearchScorePivot` | `skew.c:708+` | `findskew`, `sweep` | The detector. `deg2rad = (3.1415926535_f64/180.0) as f32` (§1); `minthresh` all-f32 **left-to-right, do not reorder** (§8). **Sweep-edge max returns SUCCESS with angle=0/conf=0, not an error** — a careless `Result`/`Option` port gets this wrong. |
| **D5** | `rotateAMGrayLow` / `pixRotateAMGray` | `rotateam.c:406-423`, `:698-709` | `rotamgray` | **The precision trap.** `sina/cosa = (16.0_f64 * angle.sin()) as f32` — computed in f64, truncated **once** — then the entire per-pixel loop in **f32** (§3). All-f32 or all-f64 both diverge on a subset of angles. `(l_int32)` casts truncate toward zero → plain `as i32`, **never** `+0.5` (§4). |
| **D6** | `pixDeskewGeneral` | `skew.c:289-351` | `deskew` | Composition + defaults (redsweep 4, sweeprange 7.0, sweepdelta 1.0, redsearch 2, **thresh 130**) + the gate: `|angle| < 0.1°` **or** `conf < 3.0` → return the original unrotated. Rotates the **original grey**, not the binarized copy. |
| **D7** | `pixDeskewBoth` | `skew.c:166-189` | `deskew` | deskew → rot90 → deskew → rot90⁻¹. The 90° round-trip exists because the differential-square-sum detector is directional; one pass cannot see skew in both axes. Needs D1. |
| **D8** | Pipeline wiring | — | end-to-end | Same page-geometry slot `auto_rectify` occupies. |

## The pi-literal rule, refined by measurement (2026-07-29)

Audit §1/§2 says each C pi literal must be reproduced with **its own exact
literal**, never a shared `std::f*::consts::PI`, "which would be MORE precise
than either C literal and could diverge in the last bit." That is correct as a
general rule and the reason the audit flags the sites at all.

It also collides head-on with this repo's **no `#[allow(...)]`** rule, because
clippy's `approx_constant` rejects the literals outright. Measured, so the
conflict is resolved by evidence rather than by whoever edits last:

| | f64 bits | `== f64::consts::PI` |
|---|---|---|
| `shear.c` `3.14159265` | `0x400921fb53c8d4f1` | **no** |
| `skew.c` `3.1415926535` | `0x400921fb54411744` | **no** |
| `f64::consts::PI` | `0x400921fb54442d18` | — |

So in **f64** the literals genuinely differ from `PI` — three distinct values.
The audit's caution is real, not theoretical.

**But both sites narrow to f32 immediately**, and after that narrowing the
difference vanishes: `(3.14159265_f64 / 2.0) as f32` is bit-identical to
`(PI / 2.0) as f32`, and `(3.1415926535_f64 / 180.0) as f32` is bit-identical
to `(PI / 180.0) as f32`. The extra f64 precision is discarded by the cast
before it can reach anything observable.

**The rule, therefore:**

- Substituting `consts::PI` is safe **only where the result is narrowed to
  f32**, and only after checking the bits. Two sites qualify:
  `normalize_angle_for_shear`'s `pi2` and the sweep's `deg2rad`.
- It is **NOT** safe anywhere the value stays f64. If a future change removes
  an `as f32` — e.g. widening the sweep's arithmetic — the substitution
  silently becomes wrong and the parity diff is the only thing that would
  catch it.
- The f32-side precedent is different and simpler:
  `deskew_dump.rs`'s `oracle_deg_to_rad` uses `f32::consts::PI` because the C++
  side's literal is itself a `float` literal, so both round to `0x40490fdb` at
  parse time. That one needs no narrowing argument.

Record the bits in the code comment at each site. A bare "verified identical"
is not enough — the next reader needs to know *which* condition made it true.

## D8 — should deskew default ON? (decide when D6 lands, not before)

`rectify` and `sauvola` are both **opt-in**, and the reflex is to make deskew
opt-in too. But deskew is structurally different from both, in a way that
argues the other direction:

**Leptonica already gates it.** `pixDeskewGeneral` refuses to rotate unless
`|angle| >= 0.1°` **and** `conf >= 3.0` — below either it returns the original,
unrotated. So on an already-straight page the leaf is a **no-op by the C's own
design**, not by our restraint. That is a much stronger safety property than
`auto_rectify`'s (which is a no-op only because *we* checked
`ShearRamp::is_significant`) and than Sauvola's (which is not a no-op at all —
it produces different pixels on every page).

Measured evidence the gate is real and tight: `page_01.pgm` — a cleanly
rendered, nominally straight page — measures `angle = −0.141°`, `conf = 3.36`.
That sits **just inside both thresholds**, so even our own straight corpus page
would be rotated. Which means "no-op on straight pages" is NOT automatic in
practice, and the reflex to default it ON is premature.

So the decision procedure, once D6 exists:

1. Run the corpus under deskew-on vs deskew-off through the **8+7+0 CER fence**
   and the golden anchors. The fence is already two-sided, so it detects
   improvement as well as regression.
2. Default ON only if the goldens are unchanged **or** deliberately re-pinned
   with the CER moving the right way. `page_01` sitting at conf 3.36 says the
   goldens probably DO move — treat a re-pin as the expected outcome, not a
   surprise.
3. If they move ambiguously, ship opt-in like rectify and say why.

Do not decide this from the gate constants alone — they look conservative and
the one page we measured says otherwise.

## Composition with `rectify` — order matters

Deskew and rectify are complementary, not redundant, but they **overlap in
`m0`**: `fit_shear_ramp`'s constant term absorbs part of a pure rotation (a
vertical shear and a rotation agree to first order at small angles; they differ
in that rotation also moves x).

**Order: deskew FIRST, rectify on the residual.** Deskew is the
byte-parity-faithful global correction; rectify then sees a page whose
rotation is already removed and fits only the height-dependent keystone.

That ordering also hands us a free falsifier: **after deskew, a purely-rotated
page must measure `m0 ≈ 0` under `fit_shear_ramp`.** If it does not, the two
stages are fighting and the composition is wrong.

## Non-goals for v1 (explicit, so they are not silently attempted)

- `pixEmbedForRotation` (the expanding canvas) — SKIP. Audit §6/§7 flag a
  *different* rounding convention (round-half-up on f64, vs D5's
  truncate-toward-zero) and a dormant `i32` overflow at side > ~46340 that
  would **panic** a Rust debug build where C silently misbehaves. Its own wave.
- Colour (`d == 32`) rotation paths — eng/deu are grey.
- `L_ROTATE_SAMPLING` — only reachable above 20° skew, never a real page.
- The standalone sweep-only and orthogonal-range entry points — not on
  `pixFindSkew`'s actual call path.
- `pixPrepare1bpp` ppi-normalization — pairs with D7 for the table front-end,
  but it is a separate leaf and does not block D1-D8.

## Gates

Per leaf: byte-parity diff vs the oracle arm across **multiple angles** (a
single angle will match either way for the f32/f64 traps — that is the whole
reason §3 is dangerous), then `cargo fmt -p <crate>`,
`cargo clippy -p <crate> -- -D warnings`, `cargo test -p <crate>` — **scoped**,
never `--all` (iron rule 1: `--all` follows the path dep into the lance-graph
workspace). Orchestrator runs the gates centrally, once, after edits land.
