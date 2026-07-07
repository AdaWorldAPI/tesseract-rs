# ruff_cpp_spo: walk_free_functions — the C-library free-function + call-graph harvest arm

## Plateau PR — commits carried as patches (branch push-locked at authoring time)

Branch `claude/happy-hamilton-0azlw4`, base = merge of #51 (`9ef26c1`). Two commits:

1. `096689c` — `walk_free_functions`: the **C-library** harvest arm. `walk_tu`
   harvests C++ *classes* (the classid→ClassView manifest); a C library
   (leptonica, zlib, …) is free functions on pointer buffers, where the AR/OO
   member body-arm captures nothing. `walk_free_functions` parses WITH bodies
   and collects every `FunctionDecl` definition + its **general call graph**
   (every `CallExpr` callee, not just the persistence-mutator set). New type
   `CppFunction { namespace, name, calls }`.
2. `c8baf2d` — `harvest_leptonica_scale` example generalized to the intra-TU
   dispatch graph (`[]` = LEAF kernel; optional `FAMILY=` root filter).

## Proven in production (tesseract-rs pixScale transcode)

The arm carried an entire multi-file C-library subsystem end-to-end:
harvested leptonica `scale1.c` + `enhance.c` → dispatch manifest → classified
`scaleGrayLILow`/`scaleGrayAreaMapLow`/`scaleAreaMapLow2`/`pixUnsharpMaskingGray2D`
as leaf kernels → hand-ported → **byte-parity vs the real `pixScale`**
(12/12 factors + 4/4 exact 2⁻ⁿ; image→text 5/5 non-model heights).
Boards: lance-graph `E-OCR-PIXSCALE-RUFF-1`, `E-OCR-PIXSCALE-COMPLETE-1`.

## Gates

clippy `-D warnings` clean; 17/17 crate tests pass (`--test-threads=1`; the
parallel run trips the documented `Clang` process-singleton — pre-existing).

## How to land from this folder

```sh
cd ruff && git checkout -b claude/walk-free-functions main   # or the current default
git am 0001-*.patch 0002-*.patch        # or: git fetch <path>/walk-free-functions.bundle
git push -u origin claude/walk-free-functions
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_016b33swuXE23hKtqxsHu9p1
