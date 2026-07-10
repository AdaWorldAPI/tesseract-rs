# Leptonica pageseg harvest — region-classifier wave inventory

> Source: `/tmp/leptonica` @ tag **1.82.0** (pinned to the installed liblept
> 1.82.0 so oracle and transcode read the SAME source; `/tmp` is ephemeral —
> re-clone with `git clone --depth 1 --branch 1.82.0
> https://github.com/DanBloomberg/leptonica.git /tmp/leptonica`).
>
> Operator directive (2026-07-09): **feature parity for the leptonica parts
> that are relevant** — where a brick computes something leptonica computes,
> transcode + byte-parity-prove the leptonica leaf instead of hand-rolling a
> heuristic. This manifest is the leaf list for the Tabelle/Bild/Textbox
> region classifier and its supporting primitives.

## Already proven (banked this session)

| Leaf | Source | Oracle | Status |
|---|---|---|---|
| `pixCountPixelsByRow` | `pix3.c:2143` | `oracles/counts_oracle.cpp` + `counts_oracle_out.txt` | **convention pinned**: lept 1bpp ON-count == our "grey<128" ink count on the formula fixture (w=97,h=61, grey=(7x+13y)%251), rows+cols exact — independent Python cross-check True/True. The xy_cut profile test asserts against the banked numbers. |
| `pixCountPixelsByColumn` | `pix3.c:2177` | same | same |

## The region-classifier leaf list (pageseg.c 1.82.0)

| Leaf | Source | Role | Deps (transcode order) |
|---|---|---|---|
| `pixGetRegionsBinary` | `pageseg.c:113` | THE composition: halftone mask + textline mask + textblock mask → region split (Bild vs Text) | the three below |
| `pixGenHalftoneMask` | `pageseg.c:281` | **Bild/Halbton-Regionen** detector | `pixReduceRankBinaryCascade` (`binreduce.c:152`), `pixExpandBinaryPower2` (`binexpand.c:135`), seedfill (**already have**: conncomp seedfill core), morph (**already have**: open/close/dilate/erode byte-parity) |
| `pixGenTextlineMask` | `pageseg.c:389` | Textzeilen-Maske | morph bricks (have) + the halftone mask (subtracted) |
| `pixGenTextblockMask` | `pageseg.c:481` | **Textbox-Regionen** (blocks from line mask) | morph bricks (have) |
| `pixReduceRankBinaryCascade` | `binreduce.c:152` | rank-2× binary reduction cascade (1-4 levels) | `pixReduceRankBinary2` (`binreduce.c:227`) — NEW leaf, table-driven |
| `pixExpandBinaryPower2` | `binexpand.c:135` | 2^n binary expansion (mask back to full res) | NEW leaf, mechanical |

Missing-primitive summary: only **rank binary reduce/expand** are new leaves;
everything else composes already-parity-green bricks (Otsu, morph, conncomp).
Table detection (Tabelle vs Textbox) is NOT in pageseg.c — leptonica has
`pixDecideIfTable` (`pageseg.c`, later in file) as a candidate follow-up;
line-grid detection via our morph bricks (long-thin open) is the interim.

## Deskew (the known gap — separate wave, listed for completeness)

| Leaf | Source | Note |
|---|---|---|
| `pixFindSkew` | `skew.c:375` | sweep+search score maximization over binarized page |
| `pixFindSkewSweepAndSearch(Score)` | `skew.c:563/617` | the real worker; depends on `pixFindDifferentialSquareSum` |
| `pixDeskew` | `skew.c:210` | find + rotate; rotation needs `pixRotate` family (NOT yet scoped) |

## Oracle method reminder

Installed liblept **1.82.0** links directly (no ABI skew here, unlike
libtesseract 5.3.4-vs-5.5.0) — oracles compile with
`g++ -std=c++17 oracle.cpp -llept`. Fixtures: deterministic formula images
(never random), formula duplicated in the Rust test, counts/masks dumped and
diffed byte-for-byte.
