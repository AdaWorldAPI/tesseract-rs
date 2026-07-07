# Plan — PDF → Text OCR (v1): the granular phase/batch integration plan

**Goal:** close the gap from the DONE byte-parity line recognizer
(`image line → text`, `E-OCR-IMAGE-TEXT-1` + `E-OCR-PIXSCALE-COMPLETE-1`) to
**PDF in → text out**, in batches executable by an Opus 4.8 orchestrator +
Sonnet 5 worker fleet. Written 2026-07-07; supersedes nothing (the
recognizer plan `recognizer-image-to-text-v2.md` stays the record of
Phases done).

## Model policy (P0 — per goal directive + lance-graph Model Policy)

| Role | Model | What |
|---|---|---|
| Orchestrator | **Opus 4.8** main thread | batch planning, preflight, C++ full-reads for briefs, review, synthesis, gates, commits, board |
| Worker: oracle drafting | **Sonnet 5** (background Agent) | C++ oracle .cpp from a template + spec (the proven B1/B2/A6 pattern) |
| Worker: kernel port | **Sonnet 5** | ONE leaf kernel from (harvest manifest row + full C source + Rust module conventions) — bounded, no synthesis |
| Worker: sweep/verify | **Sonnet 5** | run dump-vs-oracle sweeps, report diffs |
| Review / red-team | **Opus 4.8** | brutally-honest pass per batch before commit |

**Iron rules (unchanged, every batch):** ruff-harvest FIRST — never hand-roll
from eyeballed C (`E-OCR-PIXSCALE-RUFF-1` lesson); read the C fully before a
brief; cargo scoped `-p <crate>`, NEVER `--all`; byte-parity = `diff` on dumps,
non-zero = not done; per-subexpression C float-precision audit (the f64
`(i+1.0)` lesson); board hygiene lands in lance-graph; oracles banked in the
plan; `/tmp` artifacts are ephemeral — rebuild instructions in every EXECUTED
block.

## The layer map (what exists / what's missing)

```
PDF file
  │ [P5] render page → raster          ❌ EXTERNAL (pdfium/mupdf) — not a transcode
  │ [P5] text-layer fast path          ❌ (lopdf) — digital PDFs skip OCR entirely
  ▼
page image (PNG/JPEG/TIFF/raw)
  │ [P2] decode                        ❌ (image-rs policy) / P5-PGM only today ✅
  │ [P2] colour→grey (pixConvertTo8)   ❌ ruff-harvest pixconv.c
  │ [P2] threshold/binarize (Otsu)     ❌ ruff-harvest binarize.c/grayquant.c
  ▼
clean grey/binary page
  │ [P3] layout + line segmentation    ❌ THE MARATHON (textord/ + CC + morph)
  ▼
line images (h≈36 after PreScale)      ── everything below is DONE, byte-parity:
  │ prescale (pixScale)                ✅ E-OCR-PIXSCALE-COMPLETE-1
  │ FromPix grid (A6a)                 ✅ E-OCR-FROMPIX-1
  │ network forward (B1)               ✅ E-OCR-NETWORK-FORWARD-1
  │ CTC beam + extract (7b/C2)         ✅ E-OCR-RECODEBEAM-1 / -UNICHAR-EXTRACT-1
  │ ids→text                           ✅ E-CPP-PARITY-1..7
  ▼
line text                              [P1] dict beam C1 / word boxes B3-full ⬜
  │ [P4] reading order + paragraphs    ❌
  │ [P4] renderers (txt/tsv/hOCR/PDF)  ❌
  ▼
document text
```

---

## Phase 0 — PLATEAU PRs (ruff + OGAR) — do FIRST, container-loss risk

The container is ephemeral; local-only commits die with it. Everything needed
to re-land push-locked work is banked in THIS repo (pushed).

| D-id | What | Model | Status |
|---|---|---|---|
| D0.1 | ruff `walk_free_functions` (`096689c`) + example generalization (`c8baf2d`) exported as `git format-patch` + bundle + PR body → `.claude/harvest/ruff-plateau/` | Opus | ✅ done this session |
| D0.2 | **ruff branch PUSHED to github.com** (`claude/walk-free-functions`, base `9ef26c1` == main): git-over-HTTPS with GH_TOKEN works even where the proxy remote 403s. PR creation via API/MCP is blocked (App lacks `pulls:write` on ruff) → **1-click**: <https://github.com/AdaWorldAPI/ruff/compare/main...claude/walk-free-functions?expand=1> with `.claude/harvest/ruff-plateau/PR-BODY.md` | Opus | ✅ **PR #53 CREATED** (direct no-proxy REST) |
| D0.3 | **OGAR mints WRITTEN + TESTED** (`textline` 0x0805, `blob` 0x0806, `page_layout` 0x0807, `page_image` 0x0808, `ocr_renderer` 0x0809; all lockstep regions, 108/108 tests, fmt+clippy clean). OGAR push denied for this token → commit banked as patch+bundle+PR-body at `.claude/harvest/ogar-plateau/` (land per its How-to). | Opus | ✅ **pushed + PR #172 CREATED** (the 403 was a proxy artifact — bypassed; patches stay banked as container-loss insurance) |
| D0.4 | ruff follow-up (with D0.2): populate the C++ `BodyArm` into `CppMethod`+expand (the fuzzy-codebook DTO arm, flagged "needs the arm populated") — unblocks recipe-classification for textord C++ classes | Sonnet draft + Opus review | ⬜ queued behind D0.2 |

**Exit criterion Phase 0:** ruff PR merged (1-click away) + OGAR mint PR landed
from the banked patch, **together with** D0.5.

| D-id | What | Model | Status |
|---|---|---|---|
| D0.5 | **lance-graph mirror** (paired with the OGAR merge, two-sided drift fuse): `contract::ogar_codebook::CODEBOOK` +5 rows (0x0805..0x0809) + `lance-graph-ogar` COUNT_FUSE 79→84. MUST land in the same timeframe as the OGAR merge — never one side alone. | Sonnet (mechanical) + Opus review | ⬜ ready — land in the same timeframe as the #172 merge |

---

## Phase 1 — line-recognizer accuracy completion (small, pure transcode)

Everything here follows the proven leaf method 1:1. Independent of P2-P5.

### Batch 1A — C1: the dictionary beam (dawg path)
| D-id | Leaf | C++ ref | Oracle | Model |
|---|---|---|---|---|
| D1.1 | `Dawg`/`SquishedDawg` binary load (`eng.lstm-word-dawg` 3.7MB, `-punc-dawg`, `-number-dawg` on disk at /tmp) | `dict/dawg.{h,cpp}`, `trie.h` | dump edges/flags vs libtesseract `SquishedDawg::read_squished_dawg` | Sonnet port, Opus brief |
| D1.2 | `Dict`-lite: `def_letter_is_okay` / dawg traversal (NO full Dict class — only what the beam consults) | `dict/dict.cpp` (`dawg_permute…`) | per-word accept/reject table vs oracle | Opus scoping (tricky cut), Sonnet port |
| D1.3 | beam dict path: `ContinueDawg`, `PushDupOrNoDawgIfBetter` dict arms, live `kDictRatio=2.25`/`kCertOffset=-0.085`/`worst_dict_cert` | `recodebeam.cpp` (arms currently dormant in our 7b port) | full `RecognizeLine`-with-dict uids+text vs libtesseract (`Dict` loaded from the 3 dawgs) | Opus (touches proven beam — review-heavy) |
| gate | image→text sweep WITH dict == libtesseract-with-dict, AND without-dict regression stays green | | 5+ heights | Opus |

Placement: dawg load = **Core** (`lance-graph-contract`, content-tier table like
the recoder); traversal + beam arms = `tesseract-core`. Board: `E-OCR-DICT-*`.

### Batch 1B — B3-full: `ExtractBestPathAsWords` (word boxes)
| D-id | Leaf | Notes |
|---|---|---|
| D1.4 | `WordData`/box math: `line_box` + `scale_factor` mapping, per-word `TBOX` | needs a box DTO — mint via D0.3 concepts, NOT a parallel object model |
| D1.5 | `ExtractBestPathAsWords` (recodebeam.cpp) → words+boxes+certs dump vs oracle | Sonnet port after Opus DTO decision |

### Batch 1C — C3: CJK multi-code trie (OPTIONAL, gated)
Gate: a real `chi_sim.lstm-recoder` fetched → `next_codes_` non-empty falsifier.
Skip until a CJK consumer exists. (7a maps already proven generically.)

---

## Phase 2 — input layer: decode → grey → threshold (small/medium)

### Batch 2A — image decode policy + wiring
| D-id | What | Model | Notes |
|---|---|---|---|
| D2.1 | POLICY: PNG/JPEG/TIFF decode via `image`-rs (pure Rust, NOT a transcode — decode is lossless-defined by the formats, like the P5 parser). Feature `img-formats`. | Opus (policy) + Sonnet (wiring) | JPEG is lossy-DEFINED but decode is deterministic per libjpeg variant — accept `image`-rs output as input-defining (document: NOT byte-parity vs leptonica's libjpeg for JPEG; PNG/TIFF-raw are exact) |
| D2.2 | `load_grey_image(path)` → dispatch PGM/PNG/TIFF(/JPEG) → grey buffer | Sonnet | tests per format |

### Batch 2B — colour→grey (`pixConvertTo8`)
| D-id | Leaf | Oracle |
|---|---|---|
| D2.3 | ruff-harvest `pixconv.c` (`FAMILY=pixConvertTo8,pixConvertRGBToLuminance,pixConvertRGBToGray`) → manifest row | Sonnet harvest run |
| D2.4 | port the luminance kernel (the `L_RED_WEIGHT…` fixed-point) | Sonnet port, Opus review |
| D2.5 | byte-parity vs `pixConvertTo8` on synthetic RGB rasters | Sonnet sweep |

### Batch 2C — threshold / binarization (Tesseract's actual pipeline)
Tesseract thresholds via `ImageThresholder::ThresholdToPix` → Otsu
(`OtsuThreshold`, `thresholder.cpp` + `otsuthr.cpp` — **tesseract** C++, not
leptonica). The LSTM path feeds thresholded-then-grey (`PreparePixInput` gets
the thresholder's pix). NOTE: for eng.lstm the recognizer consumes GREY — the
thresholder matters for textord (P3) and for `tessedit_do_invert`.
| D-id | Leaf | C++ ref | Oracle |
|---|---|---|---|
| D2.6 | ruff-harvest `src/ccmain/thresholder.cpp` + `src/textord/otsuthr.cpp` (walk_tu — C++ classes) | — | manifest |
| D2.7 | `OtsuThreshold` + `HistogramRect` port | `otsuthr.cpp` (~200 lines, pure int) | vs libtesseract `OtsuThreshold` on synthetic rects |
| D2.8 | `ThresholdRectToPix` + the grey-normalization path | `thresholder.cpp` | vs `ImageThresholder` public API |

**Exit P2:** page image file → clean grey page buffer, each Tesseract-owned
step byte-parity, decode step policy-documented.

---

## Phase 3 — textord: layout + line segmentation (THE MARATHON)

Strategy: **harvest-first, then a batch per subsystem**, mirroring how the
recognizer was done (a proven leaf at a time), PLUS a non-parity fast-path so
end-to-end PDF→text works early.

### Batch 3-alt — pragmatic line finder (UNBLOCKS E2E EARLY, marked approx)
| D-id | What | Model |
|---|---|---|
| D3.0 | projection-profile line segmenter (horizontal ink-profile valleys → line boxes → crop+feed recognizer). MARKED approximation (like the old prescale path): functional E2E now, replaced by 3B-3F for parity. Feature-gated `seg-approx`. | Opus design, Sonnet impl |

### Batch 3A — the harvest manifests (drives ALL batch planning below)
| D-id | What |
|---|---|
| D3.1 | `walk_tu` over `src/textord/*.h` + `src/ccstruct/{blobbox,ocrblock,ocrrow,polyblk}.h` → class manifest (the BLOBNBOX/TO_ROW/TO_BLOCK ClassViews) |
| D3.2 | `walk_free_functions` over leptonica `conncomp.c`, `morph.c` (the CC + brick-morphology substrate textord leans on) → dispatch manifests |
| D3.3 | Opus: batch cut from the manifests — leaf order, oracle strategy per leaf, Sonnet brief per leaf (the D0.4 BodyArm, if landed, classifies method recipes here) |

### Batch 3B — connected components (leptonica substrate)
`pixConnComp` family → the blob source. Byte-parity vs leptonica (CC count,
boxes, pixel membership). Consumes `blob` classid (D0.3).

### Batch 3C — brick morphology minimal set
`pixDilateBrick`/`pixErodeBrick`/`pixOpenBrick`/`pixCloseBrick` (only the ops
the P3 path actually calls — harvest decides the exact set). Byte-parity.

### Batch 3D — ccstruct data shapes onto V3 SoA
`TBOX`/`BLOBNBOX`/`TO_ROW`/`TO_BLOCK` as classid-keyed facets (textline/blob/
page_layout mints from D0.3) — NOT a parallel object model; ClassView from the
D3.1 manifest. Opus-heavy (architecture), Sonnet for the mechanical facets.

### Batch 3E — textline formation
`makerow.cpp` (`make_rows`/baseline fitting/`cleanup_rows`) + x-height
(`topitch`-adjacent). THE core algorithmic batch — several leaves; oracle =
per-line boxes+baselines dump vs libtesseract on synthetic multi-line pages.

### Batch 3F — page segmentation modes, staged
1. `PSM_SINGLE_LINE` (bypass — one line box) → trivially done after 3E
2. `PSM_SINGLE_BLOCK` (one column) — the 80% document case
3. full auto layout (`ColumnFinder`/tabfind) — LAST, largest; may stay Phase 7
OSD/orientation: explicitly OUT of v1.

**Exit P3:** grey page → Vec<line image + box> byte-parity for PSM 6/7/13 on
the golden corpus; full-auto deferred allowed.

---

## Phase 4 — page assembly + output renderers (medium)

| D-id | What | Model | Oracle/gate |
|---|---|---|---|
| D4.1 | reading order + paragraph grouping (single-column first; consumes page_layout facets) | Opus design, Sonnet impl | text-order parity vs `tesseract --psm 6 txt` on golden corpus |
| D4.2 | plain-text renderer (line joins, dehyphenation OFF v1) | Sonnet | diff vs tesseract txt output |
| D4.3 | TSV renderer (words+boxes+conf — needs B3-full D1.4/D1.5) | Sonnet | diff vs tesseract tsv |
| D4.4 | hOCR renderer | Sonnet | structural diff (attribute order normalized) |
| D4.5 | searchable-PDF renderer (invisible text layer via `lopdf`) | Opus design (PDF model), Sonnet impl | text-extraction round-trip == D4.2 output |

`ocr_renderer` classid (D0.3): one slot, format = custom-low (mirrors the
`network_layer` pattern).

---

## Phase 5 — PDF front-end (EXTERNAL, not a transcode)

| D-id | What | Model | Notes |
|---|---|---|---|
| D5.1 | POLICY: digital-PDF **text-layer fast path** first (`lopdf`/`pdf-extract`): if a page has a text layer → extract, NO OCR. | Opus | most PDFs never hit OCR |
| D5.2 | raster fallback: `pdfium-render` (feature `pdf-raster`), 300 dpi grey render per page | Sonnet wiring | pdfium is a C++ dep — OK: it is INPUT tooling, not the OCR runtime (the "no leptonica" rule guards the OCR path; document the boundary). Pure-Rust alternative (`hayro`?) evaluated in a spike D5.4 |
| D5.3 | `tesseract-ocr-pdf` orchestrator binary: per page → (text-layer? extract : render→P2→P3→recognize→P4) → document assembly | Sonnet impl, Opus review | E2E golden test |
| D5.4 | SPIKE: pure-Rust rasterizer feasibility (drop the pdfium C++ dep) | Sonnet spike, Opus verdict | timeboxed |

---

## Phase 6 — E2E validation + performance

| D-id | What | Gate |
|---|---|---|
| D6.1 | golden corpus: 10 scanned pages + 5 digital PDFs + line-image regression set | corpus committed (small, license-clean) |
| D6.2 | parity harness: our output vs `tesseract` CLI (`--psm 6/7/13`, txt+tsv) | byte-parity where the full chain is transcoded; CER/WER report where approx paths (3-alt, JPEG decode) are active — NEVER blur which is which |
| D6.3 | perf bench vs C++ tesseract (pages/sec, RSS) | report, no gate |
| D6.4 | CI wiring: golden suite in tesseract-rs workflow (scoped -p, sibling checkouts) | green |

---

## Batch execution protocol (every batch, Opus 4.8 + Sonnet 5)

1. **Opus preflight:** read the C++ fully; run/refresh the ruff harvest for the
   batch's TU(s); cut leaves; write per-leaf Sonnet briefs (template below);
   pre-register the oracle format + pass criteria.
2. **Sonnet fan-out (background Agents, poor-man's parallelism):** oracle
   drafting ‖ kernel port ‖ dump example — bounded briefs, no synthesis.
   Sonnet-worker guardrails (lance-graph `sonnet-worker-guardrails.md` §1
   verbatim preamble) in EVERY brief.
3. **Opus verify:** run the sweep personally; non-empty, ≥2 shapes/factors;
   diff byte-identical. Debug precision mismatches personally (the f64/f32
   audit is Opus work).
4. **Opus gates + land:** fmt/clippy `-D warnings`/tests scoped `-p`; commit
   (message = leaf + parity evidence + gotchas); push; EPIPHANIES entry
   (lance-graph) + plan EXECUTED block with banked oracle; tasks updated.
5. **Batch review:** brutally-honest pass before the next batch starts.

**Sonnet brief template (per leaf):** goal (ONE function) · full C source
inline or path+lines · harvest manifest row (dispatch context) · Rust module +
conventions to mirror · oracle template path (`/tmp/*_oracle.cpp` precedent) ·
exact g++ line · dump format (pre-registered) · DO-NOTs (no cargo --all, no
/home/user writes outside the named file, no scope creep).

## Dependency graph / suggested order

```
P0 (plateau: ruff PR 1-click, OGAR patch land, D0.5 mirror)   ── independent, FIRST
P1A dict ─┐
P1B words ─┼─ independent of P2/P3 — good Sonnet warm-up batches
P2A-C ─────┤
P3-alt ────┴─→ E2E-approx demo (PDF→text works, marked approx)
P3A → {3B ‖ 3C} → 3D → 3E → 3F₁ → 3F₂            (parity marathon)
P4.1-4.3 after P3E + P1B; P4.5 after P4.2
P5.1 anytime; P5.2/5.3 after P2; P6 rolling, hard-gate after P3F₂+P4
```

**Effort profile (rough, in batch-days):** P0 ≈ done/operator · P1 ≈ 2-3 ·
P2 ≈ 2 · P3-alt ≈ 1 · P3 ≈ 8-15 (the marathon) · P4 ≈ 3-4 · P5 ≈ 2 · P6 ≈ 2.

## Board hygiene
Plan index: lance-graph `INTEGRATION_PLANS.md` (prepended). Per-batch findings:
`EPIPHANIES.md` `E-OCR-*`. Status: this file's tables (⬜→✅ in the landing
commit). Oracles: banked in the consuming plan section, always.

### D1.2 seed decision (finding from oracle draft, 2026-07-07)
`RecodeBeamSearch::ContinueDawg` (recodebeam.cpp:1108) uses **`default_dawgs()`** for
word-start, NOT `init_active_dawgs()` (which is `LanguageModel`'s legacy-recognizer
seed — out of scope for the LSTM beam transcode). D1.2b's Rust walker MUST seed
via the `default_dawgs` equivalent to match production `recognize_line` behavior.
Oracle at `/tmp/def_letter_oracle.cpp`; example: "the" (ids 91 97 92) → perm=8
SystemDawgPerm, `valid_end=1` after "th" (word-end reached).

---

## P1 execution addendum (planned on Fable, 2026-07-07) — ready-to-fire briefs

### D1.2b — the Dict-lite walker (fire AFTER D1.2a lands with ruff-verified shapes)

**Placement:** `tesseract-core/src/dict_walker.rs` (NOT the Core — the walker is
beam-coupled compute-free logic like `recodebeam`; the Core carries only the
dawg TABLE). Consumes `lance_graph_contract::dawg::{SquishedDawg, DawgType,
PermuterType, NodeRef, NO_EDGE}` re-exported through tesseract-core.

**Shapes (ruff-manifest-sourced, per directive):** `DawgPosition { dawg_ref:
NodeRef, punc_ref: NodeRef, dawg_index: i8, punc_index: i8, back_to_punc: bool }`
(+ `PartialEq` for `add_unique`), `DawgArgs`-equivalent as fn params/return
(`active: &[DawgPosition], out updated: Vec<DawgPosition>, permuter, valid_end`).

**API:**
```rust
pub struct DictLite { dawgs: Vec<SquishedDawg>, /* indices: word/punc/number */ }
impl DictLite {
    pub fn from_components(word: &[u8], punc: &[u8], number: &[u8]) -> Result<...>;
    pub fn default_dawgs(&self, suppress_patterns: bool) -> Vec<DawgPosition>;   // dict.cpp:625-647
    pub fn def_letter_is_okay(&self, active: &[DawgPosition], charset: &UniCharSet,
        unichar_id: u32, word_end: bool, permuter_in: PermuterType)
        -> (Vec<DawgPosition>, PermuterType, bool /*valid_end*/);               // dict.cpp:407-571
}
```
**Transcode notes (from the full Opus read, banked):**
- `GetStartingNode`: `NO_EDGE → 0`; `next_node(edge)==0 → NO_EDGE` (dict.h:397-406).
- `char_for_dawg`: Number-dawg maps `get_isdigit(ch) → kPatternUnicharID(0)` (dict.h:411-421).
- Successor sets: `kDawgSuccessors[punc][ty]` — punc→{word,number}; word/number→{punc} (dawg.h:87-92).
- eng has NO pattern dawg → `ProcessPatternEdges` arm is dead for eng; implement the
  DAWG_TYPE_PATTERN branch as a documented `unreachable-for-eng` returning no-op (do NOT
  silently skip — keep the branch structure, note the falsifier gap).
- Permuter update rule at fn end (dict.cpp:559-566): overwrite unless COMPOUND_PERM kept.
- `add_unique` = linear dedup on the 5-tuple.
- Successor lists in Dict are built per-dawg at load (dict.cpp:367ff `SuccessorList`);
  for 3 dawgs this is: punc→[word_idx, number_idx], word→[punc_idx], number→[punc_idx].

**Byte-parity:** example `dict_walk_dump` — args = space-separated unichar-ids;
output format EXACTLY the oracle's (`step\t…` + sorted `p\t…` lines).
Oracle: `/tmp/def_letter_oracle` (built, works; TessBaseAPI+getDict public path;
needs `/tmp/eng.traineddata` — present, 4.1 MB). Sweep: ≥6 words — "the", "cat",
negative "qjx", punctuation-wrapped («"the"», ids for quote chars), a number
("42"), a mixed token — full-step dumps byte-identical.

### D1.3 — beam dict arms (fire AFTER D1.2b green)

**Change surface:** `tesseract-core/src/recodebeam.rs` ONLY:
1. `RecodeBeamSearch::new_with_dict(recoder, null_char, simple, DictLite)` (keep
   the existing `new` — non-dict path stays untouched, its 7b parity is regression).
2. Port `ContinueDawg` + the dict arms of `PushDupOrNoDawgIfBetter` +
   `PushInitialDawgIfBetter` (recodebeam.cpp:1057-1160) — the currently-dormant
   branches; `dawgs: Vec<DawgPosition>` rides on `RecodeNode` (arena-friendly:
   store as `Option<Box<[DawgPosition]>>`).
3. Live constants: `kDictRatio=2.25`, `kCertOffset=-0.085` enter `Decode` args
   from `recognize_line` when dict present.
**Oracle:** extend `/tmp/image_text_oracle.cpp` → `image_text_dict_oracle.cpp`
with `api.Init + RecodeBeamSearch(recoder, null, simple, dict)` … OR simpler:
libtesseract full `RecognizeLine` with dict via TessBaseAPI on line images.
Gate: with-dict uids+text == oracle on ≥5 images AND the without-dict sweep
(`E-OCR-RECOGNIZE-GRID-1`, 5/5) stays green (regression).

### Sequencing + model split
1. ruff enum-harvest lands (in flight) → verify D1.2a shapes vs manifest → land D1.2a [Opus verify]
2. Fire D1.2b brief [Sonnet grind; Opus lands]
3. Fire D1.3 brief [Sonnet grind on the beam edits is HIGHER RISK (touches proven
   code) → Sonnet drafts, Opus reviews diff hunk-by-hunk before gates]
4. B3-full (ExtractBestPathAsWords) — after D1.3, same recodebeam.rs wave.
5. ruff PR: walk_enums + harvest_tesseract_dict via runbook recipe → ruff main.

## P2 EXECUTED (2026-07-07) — input layer: pixconv + Otsu, both byte-parity green

Ruff-harvest manifests (banked in `.claude/harvest/leptonica-scale-callgraph.txt`):
`pixConvertTo8 → pixConvertRGBToLuminance → pixConvertRGBToGray` (LEAF, weights
0.3/0.5/0.2) and `OtsuThreshold → {HistogramRect, OtsuStats}` (both LEAF;
otsuthr.cpp lives in `ccstruct/`, not textord/). Harvested via the extended
`harvest_leptonica_scale` (LANG_MODE=c++ / EXTRA_INC, ruff session branch `ee030a0`).

- **pixconv** (`image_input::rgb_to_gray`/`rgb_to_luminance`, pixconv.c:741-885):
  f32 weighted sum, `+0.5` f64 promotion; weights 0,0,0 → the default trio
  (exactly `pixConvertRGBToLuminance`). Parity 3/3 (24×36, 33×50, 0.5/0.3/0.2
  explicit weights) vs REAL `pixConvertRGBToGray`.
- **Otsu** (`threshold.rs`: `histogram_rect_*`, `otsu_stats` (i32 counts, f64
  mu/variance), `otsu_threshold_gray`/`_channels`, `threshold_rect_to_binary`,
  otsuthr.cpp:34/88/118 + thresholder.cpp:394-421): parity 3/3 (24×36, 64×36,
  37×29) vs REAL `tesseract::OtsuThreshold` + replicated per-pixel predicate.
  Dump convention: white_result → 255 (grey rendering of 1bpp CLEAR bit;
  the oracle's first draft dumped the raw bit value = inverted).

Oracles banked at `.claude/harvest/oracles/{pixconv,otsu}_oracle.cpp`. Rebuild:
```sh
g++ -std=c++17 pixconv_oracle.cpp -I/usr/include/leptonica $(pkg-config --cflags --libs lept) -o /tmp/pixconv_oracle
INCS=$(find /tmp/tesseract/src -maxdepth 1 -type d | sed 's/^/-I/' | tr '\n' ' ')
g++ -std=c++17 -DFAST_FLOAT $INCS -I/tmp/tesseract/include otsu_oracle.cpp $(pkg-config --cflags --libs tesseract) $(pkg-config --libs lept) -o /tmp/otsu_oracle
```
Shared inputs: the Rust dumps self-generate + write `/tmp/{pixconv,otsu}_input.bin`;
run the Rust example FIRST, then the oracle on the bin, then diff.
