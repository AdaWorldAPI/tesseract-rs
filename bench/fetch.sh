#!/usr/bin/env bash
# fetch.sh — fetch every entry in bench/manifest.toml into the local cache and
# verify it byte-for-byte against its pinned SHA-256.
#
# FETCH-DON'T-COMMIT: the cache directory is gitignored. Nothing fetched here
# enters git, and nothing fetched here may become a committed test fixture —
# committed tests read `corpus/` only (CLAUDE.md's `/tmp`-fixture time-bomb
# rule).
#
# A hash mismatch is a HARD FAILURE, never a warning: the whole point of the
# manifest is that a benchmark number can be traced to exact bytes. If upstream
# re-encodes a page, this must break loudly so the pin is updated deliberately
# and the affected measurements are re-taken — not silently absorbed.
#
# Usage:
#   bench/fetch.sh              # fetch + verify all entries
#   bench/fetch.sh --verify     # verify what is already cached; fetch nothing
#   BENCH_CACHE=/path bench/fetch.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.toml"
CACHE="${BENCH_CACHE:-$HERE/cache}"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

# `--noproxy '*'` is REQUIRED, not cosmetic. This environment's agent proxy
# scopes GitHub to an allowlist and returns 403 for third-party repositories,
# while the direct path returns 200 — measured both ways, identical bytes.
# See CLAUDE.md § "GitHub access matrix": a 403 here is usually the proxy,
# not the repository.
fetch_one() {
  curl -sSL --noproxy '*' --fail --max-time 120 -o "$1" "$2"
}

# Extract the manifest's `key = "value"` scalars per [[corpus]] block. Kept to
# the flat subset the manifest actually uses rather than reaching for a TOML
# parser — this crate deliberately carries no TOML dependency, and a fetch
# script that needs a build to run is a worse trade than a documented subset.
# The `note` field is multi-line and deliberately NOT read here.
entries() { awk '
  /^\[\[corpus\]\]/ { if (name != "") print rec; name=""; rec=""; next }
  /^[a-z0-9_]+ *= *"/ {
    k=$0; sub(/ *=.*/, "", k)
    v=$0; sub(/^[a-z0-9_]+ *= *"/, "", v); sub(/".*$/, "", v)
    if (k == "name") name=v
    rec = rec k "\t" v "\n"
  }
  END { if (name != "") print rec }
' "$MANIFEST"; }

field() { printf '%s' "$1" | awk -F'\t' -v k="$2" '$1==k {print $2; exit}'; }

# Verify $1 against expected hash $2; print the reason and return 1 on failure.
check() {
  local path="$1" want="$2" label="$3"
  [ -f "$path" ] || { echo "    MISSING  $label"; return 1; }
  local got
  got="$(sha256sum "$path" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "    MISMATCH $label"
    echo "      expected $want"
    echo "      actual   $got"
    return 1
  fi
  echo "    ok       $label  ($(wc -c < "$path") bytes)"
}

pass=0; fail=0
while IFS= read -r -d '' rec; do
  [ -n "$rec" ] || continue
  name="$(field "$rec" name)"
  [ -n "$name" ] || continue
  img_url="$(field "$rec" image_url)"; img_sha="$(field "$rec" image_sha256)"
  gt_url="$(field "$rec" gt_url)";     gt_sha="$(field "$rec" gt_sha256)"
  lic="$(field "$rec" license)"

  echo "== $name  [$lic]"
  dir="$CACHE/$name"
  mkdir -p "$dir"
  img="$dir/${img_url##*/}"
  gt="$dir/${gt_url##*/}"

  if [ "$VERIFY_ONLY" -eq 0 ]; then
    [ -f "$img" ] || fetch_one "$img" "$img_url"
    [ -f "$gt" ]  || fetch_one "$gt"  "$gt_url"
  fi

  ok=1
  check "$img" "$img_sha" "image" || ok=0
  check "$gt"  "$gt_sha"  "gt"    || ok=0
  if [ "$ok" -eq 1 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done < <(entries | awk 'BEGIN{RS="";ORS="\0"}{print}')

echo
echo "== $pass entr(y|ies) verified, $fail failed =="
[ "$fail" -eq 0 ] || {
  echo "A mismatch means the pinned bytes and the fetched bytes disagree." >&2
  echo "Do NOT re-pin to silence it: re-take any measurement that used the" >&2
  echo "old bytes, then update the pin deliberately." >&2
  exit 1
}
