#!/usr/bin/env bash
# run_unicharset_parity.sh — byte-parity the lance_graph_contract::unicharset::UniCharSet
# transcode against libtesseract for ANY model. The `bijection` half self-validates
# the object layout (E-CPP-PARITY-1); the five field halves are trusted once it is 0-diff.
#
# Prereqs (step 1 of the two-step method — install the oracle):
#   apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev tesseract-ocr-{eng,deu}
#   git clone --depth 1 --branch 5.3.4 https://github.com/tesseract-ocr/tesseract /tmp/tesseract-src
#   combine_tessdata -u /usr/share/tesseract-ocr/5/tessdata/<lang>.traineddata corpus/model/<lang>.
# Usage:
#   run_unicharset_parity.sh <path/to/X.lstm-unicharset> <label>
set -euo pipefail

UNI="${1:?usage: run_unicharset_parity.sh <unicharset> <label>}"
LABEL="${2:-model}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LG="${LANCE_GRAPH:-/home/user/lance-graph}"
ORACLE=/tmp/unicharset_oracle

# Build oracle once (source headers 5.3.4 + installed lib 5.3.4 → no ABI skew).
if [ ! -x "$ORACLE" ]; then
  g++ -std=c++17 "$HERE/unicharset_oracle.cpp" \
    -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/src/ccstruct -I/tmp/tesseract-src/include \
    -ltesseract -lleptonica -o "$ORACLE"
fi

# Build the Rust dump once, reuse the binary.
BIN="$LG/target/debug/examples/unicharset_dump"
( cd "$LG" && cargo build -q -p lance-graph-contract --example unicharset_dump )

pass=0; fail=0
for mode in bijection properties script other_case direction mirror; do
  [ "$mode" = bijection ] && rarg="" || rarg="$mode"
  "$ORACLE" "$UNI" "$mode"      > "/tmp/o_${LABEL}_${mode}.tsv"
  "$BIN"    "$UNI" $rarg        > "/tmp/r_${LABEL}_${mode}.tsv"
  if diff -q "/tmp/o_${LABEL}_${mode}.tsv" "/tmp/r_${LABEL}_${mode}.tsv" >/dev/null; then
    printf '  OK  %-11s %s: %s rows byte-identical\n' "$mode" "$LABEL" "$(wc -l < /tmp/o_${LABEL}_${mode}.tsv)"
    pass=$((pass+1))
  else
    printf '  XX  %-11s %s: DIFF\n' "$mode" "$LABEL"
    diff "/tmp/o_${LABEL}_${mode}.tsv" "/tmp/r_${LABEL}_${mode}.tsv" | head -6
    fail=$((fail+1))
  fi
done
echo "== $LABEL unicharset: $pass/6 byte-parity ($fail fail) =="
[ "$fail" -eq 0 ]
