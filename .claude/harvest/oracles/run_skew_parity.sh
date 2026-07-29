#!/usr/bin/env bash
# run_skew_parity.sh — byte-parity harness for the deskew wave (D1 / D2 / D5).
#
# Diffs `.claude/harvest/oracles/skew_oracle.cpp` (liblept 1.82.0, the truth)
# against `cargo run -p tesseract-ocr --example deskew_dump` (the transcode).
# Sibling of run_unicharset_parity.sh; same contract — exit 0 == every arm
# byte-identical, non-zero == a real divergence with the diff printed.
#
# Usage:
#   .claude/harvest/oracles/run_skew_parity.sh [fixture.pgm]
#
# Requires: g++, liblept-dev, the 1.95 toolchain. Run from the repo root.

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
cargo +1.95 build -q -p tesseract-ocr --example deskew_dump || exit 1
RUST="$(cargo +1.95 metadata --format-version 1 --no-deps \
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

# ── D2 — pixFindDifferentialSquareSum. ──────────────────────────────────────
#
# ONLY diffable at angle 0 until D3 (the vertical shear) lands. The oracle's
# dss arm is binarize -> vshear -> score; the Rust side is binarize -> score.
# At angle 0 the shear is a verified no-op (both pivots return the identical
# sum), so this isolates the SCORING function exactly. A nonzero angle here
# would be comparing two different operations and is NOT D2 evidence.
note "D2  differential square sum (angle 0 only — see comment)"
for pivot in 1 2; do diff_arm "dss angle=0 pivot=$pivot" dss "$FIXTURE" "$THRESH" 0.0 "$pivot"; done

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

# ── verdict ─────────────────────────────────────────────────────────────────
note "verdict"
printf '  %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
printf '  \033[32mall arms byte-identical\033[0m\n'
