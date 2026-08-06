#!/usr/bin/env bash
# run_skew_parity.sh — byte-parity harness for the deskew wave (D1-D7).
#
# Diffs `.claude/harvest/oracles/skew_oracle.cpp` (liblept 1.82.0, the truth)
# against `cargo run -p tesseract-ocr --example deskew_dump` (the transcode).
# Sibling of run_unicharset_parity.sh; same contract — exit 0 == every arm
# byte-identical, non-zero == a real divergence with the diff printed.
#
# History: this script originally covered D1/D2/D5 only. D3/D4/D6/D7 had
# real byte-parity evidence — `deskew.rs`'s own doc comments narrate specific
# verified diffs ("REAL D2+D3 end-to-end", "Verified byte-identical 10/10")
# — but none of it was captured HERE, so CLAUDE.md's summary claim ("only
# D1/D2/D5 green") went stale the moment those doc-comment diffs were run and
# never re-run again. Per this repo's own falsifiability rule
# ("a doc-comment claim is not a behaviour... a test must exercise the claim
# or the claim must be labelled claimed, unverified"), that gap is now
# closed: every leaf below is exercised by this ONE script, on committed
# fixtures, every time it runs.
#
# Usage:
#   .claude/harvest/oracles/run_skew_parity.sh [fixture.pgm]
#
# Requires: g++, liblept-dev, the repo's pinned toolchain (rust-toolchain.toml
# — bare `cargo` resolves to it). Run from the repo root.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

FIXTURE="${1:-corpus/pages/page_01.pgm}"
ORACLE=/tmp/skew_oracle
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

THRESH=130   # pixDeskewGeneral's own default (skew.c) — NOT Otsu.
GRAYVAL=255  # L_BRING_IN_WHITE in 8bpp grey.

fail=0
pass=0

note() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { pass=$((pass+1)); printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()  { fail=$((fail+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

# ── build both sides ────────────────────────────────────────────────────────
note "building the oracle (liblept $(pkg-config --modversion lept))"
g++ -std=c++17 .claude/harvest/oracles/skew_oracle.cpp \
    -I/usr/include/leptonica -lleptonica -o "$ORACLE" || exit 1

note "building the Rust dump"
# Scoped -p ALWAYS: `--all` follows the path dep into the lance-graph
# workspace and rebuilds ~30 unrelated files (CLAUDE.md iron rule 1).
cargo build -q -p tesseract-ocr --example deskew_dump || exit 1
RUST="$(cargo metadata --format-version 1 --no-deps \
        --manifest-path Cargo.toml 2>/dev/null \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')/debug/examples/deskew_dump"

diff_arm() {  # $1 = label, rest = argv for BOTH sides
  local label="$1"; shift
  "$ORACLE" "$@" > "$WORK/o.txt" 2>"$WORK/o.err"
  "$RUST"   "$@" > "$WORK/r.txt" 2>"$WORK/r.err"
  if [ ! -s "$WORK/o.txt" ]; then bad "$label (oracle produced nothing: $(head -1 "$WORK/o.err"))"; return; fi
  if [ ! -s "$WORK/r.txt" ]; then bad "$label (rust produced nothing: $(head -1 "$WORK/r.err"))"; return; fi
  if diff -q "$WORK/o.txt" "$WORK/r.txt" >/dev/null; then
    ok "$label ($(wc -l < "$WORK/o.txt") lines)"
  else
    bad "$label"
    diff "$WORK/o.txt" "$WORK/r.txt" | head -12
  fi
}

# ── D1 — pixRotate90, both directions. Lossless index remap, no float. ──────
note "D1  rotate90 (exact: pure index remap)"
for dir in 1 -1; do diff_arm "rot90 dir=$dir" rot90 "$FIXTURE" "$dir"; done

# ── D2+D3 — pixFindDifferentialSquareSum + pixVShearCorner/Center. ──────────
#
# The oracle's `dss` arm is binarize -> vshear -> score; the Rust `dss` arm
# (via [`crate::deskew::v_shear_corner`]/[`v_shear_center`] then
# [`find_differential_square_sum`]) reproduces that SAME composition, so this
# is real end-to-end evidence for D2 AND D3 together, at ANY angle — not just
# angle=0. (Angle 0 alone would only prove the shear is a correctly-behaving
# no-op there, which every implementation gets right trivially; it says
# nothing about the shear's indexing at a real angle.) Dense: ±7° in 0.5°
# steps, both pivots — 58 comparisons.
note "D2+D3  differential square sum after vertical shear (dense angle sweep, both pivots)"
for deg in $(python3 -c "
for i in range(-14, 15):          # -7.0 .. +7.0 in 0.5 steps
    print(f'{i*0.5:g}')
"); do
  for pivot in 1 2; do diff_arm "dss deg=$deg pivot=$pivot" dss "$FIXTURE" "$THRESH" "$deg" "$pivot"; done
done

# ── D5 — rotateAMGray, the precision trap. ──────────────────────────────────
#
# A DENSE sweep, deliberately. The f64-computed-then-narrowed-once sina/cosa
# (audit §3) diverges from an all-f32 or all-f64 port only on a SUBSET of
# angles — wherever the rounding boundary of xpm/ypm's low 4 bits falls
# differently. A handful of round numbers can pass while the port is wrong;
# that is the entire reason the audit flags this site.
note "D5  rotate_am_gray — dense sweep (the f32/f64 trap)"
for deg in $(python3 -c "
for i in range(-40, 41):          # -20.0 .. +20.0 in 0.5 steps
    print(f'{i*0.5:g}')
"); do
  diff_arm "rotamgray deg=$deg" rotamgray "$FIXTURE" "$deg" "$GRAYVAL"
done

note "D5c rotate_am_gray_corner — same trap, different per-pixel formula"
for deg in 1 2.5 5 7.5 10 12.5 15 17.5 19; do
  diff_arm "rotamgraycorner deg=$deg" rotamgraycorner "$FIXTURE" "$deg" "$GRAYVAL"
done

# ── D4/D6/D7 fixtures — genuinely different skew magnitudes ─────────────────
#
# `$FIXTURE` alone is nearly straight (page_01.pgm's own baseline is
# angle≈-0.14°), which barely exercises D4's interval-halving search — it
# converges almost immediately near zero. `corpus/gen/gen_skew_fixtures.py`
# (committed, deterministic, PIL-rotated from the SAME source page at
# +1.5°/-2.5°/+5.0°, mirroring `deskew-wave-v1.md`'s own falsifier table)
# supplies the magnitude spread the search logic actually needs to prove
# itself against. Generation fidelity is not part of the parity chain —
# once written to disk, both oracle and Rust read the identical committed
# bytes; only the DETECTOR's output on those bytes is compared.
SKEW_FIXTURES=("$FIXTURE")
for f in corpus/quality/skew_p015.pgm corpus/quality/skew_m025.pgm corpus/quality/skew_p050.pgm; do
  if [ -f "$f" ]; then
    SKEW_FIXTURES+=("$f")
  else
    bad "missing $f -- run \`python3 corpus/gen/gen_skew_fixtures.py\` first"
  fi
done

# ── D4 — pixFindSkewSweepAndSearchScorePivot, at pixFindSkew's OWN defaults.
#
# Parameters below are NOT invented: read directly from
# `/tmp/leptonica-src/src/skew.c`'s `pixFindSkew` -> `...SweepAndSearch` ->
# `...SweepAndSearchScore` -> `...SweepAndSearchScorePivot` call chain —
# `DefaultSweepReduction=4`, `DefaultBsReduction=2`, `sweepcenter=0.0`,
# `DefaultSweepRange=7.0`, `DefaultSweepDelta=1.0`, `DefaultMinbsDelta=0.01`,
# `pivot=L_SHEAR_ABOUT_CORNER` (1). So this arm IS `pixFindSkew` itself, not
# an approximation of it — verified by cross-checking the oracle's OWN
# `findskew` output against its `sweep` output at these exact parameters
# (bit-identical, both sides, every fixture below) before ever touching Rust.
note "D4  find_skew_sweep_and_search_score_pivot — real skew magnitudes, pixFindSkew's own defaults"
for f in "${SKEW_FIXTURES[@]}"; do
  diff_arm "sweep $(basename "$f")" sweep "$f" "$THRESH" 4 2 0.0 7.0 1.0 0.01 1
done

# ── D6 — pixDeskewGeneral, via pixDeskew's own all-default call. ────────────
note "D6  deskew_general — real skew magnitudes × redsearch {2,4}"
for f in "${SKEW_FIXTURES[@]}"; do
  for rs in 2 4; do diff_arm "deskew $(basename "$f") rs=$rs" deskew "$f" "$rs"; done
done

# ── D7 — pixDeskewBoth, the two-pass orthogonal composition. ────────────────
note "D7  deskew_both — real skew magnitudes × redsearch {2,4}"
for f in "${SKEW_FIXTURES[@]}"; do
  for rs in 2 4; do diff_arm "deskewboth $(basename "$f") rs=$rs" deskewboth "$f" "$rs"; done
done

# ── verdict ─────────────────────────────────────────────────────────────────
note "verdict"
printf '  %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
printf '  \033[32mall arms byte-identical\033[0m\n'
