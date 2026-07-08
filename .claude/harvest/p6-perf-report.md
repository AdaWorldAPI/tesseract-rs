# D6.3 — perf report (report only, NO gate per plan)

Machine: this arc's dev container (x86_64, AVX-512 VNNI, no AMX — the int8 GEMM
dispatches ndarray's Tier-2 VPDPBUSD path). Corpus: the 10 committed 512×720
rendered pages. Both sides measured 2026-07-08, idle machine (an earlier CLI
run that overlapped the debug golden-verify was discarded as contaminated —
page_01 read 63 s under contention vs 181 ms clean).

## Ours — `recognize_page_makerow` + dict, release build

In-process: model+dict loaded ONCE, then 1 warm-up + 3 timed full-corpus
passes, best pass reported (`cargo run --release -p tesseract-ocr --example
golden_bench`).

| metric | value |
|---|---|
| per page (best pass) | 779–1098 ms |
| pages/sec | **1.06** |
| peak RSS (VmHWM) | **20 444 kB** |

## C++ tesseract CLI — `--psm 6`, same PGMs

Per-page SUBPROCESS, best of 3 (`python3 corpus/gen/run_cli_golden.py --bench`)
— each timing INCLUDES process startup + traineddata load, so the shapes are
not identical; documented, not blurred.

| metric | value |
|---|---|
| per page (best of 3) | 174–198 ms |
| pages/sec | **5.38** |
| peak child RSS (cumulative) | 36 740 kB |

## Reading

The C++ CLI is ~5× faster end-to-end on this corpus even paying per-process
startup. The gap is in our page path, not the GEMM (the forward is the proven
ndarray VNNI path): the dict beam and the textline stage are single-threaded,
allocation-heavy first-correctness transcodes. Memory: ours peaks at ~56% of
the CLI's. No optimization work was gated on this report (plan: "report, no
gate"); candidate levers, in expected order of yield: beam-node arena reuse,
`recognize_grey_line` buffer reuse across rows, parallel rows, dict-beam
top-n pruning parity check.

Regenerate: `cargo run --release -p tesseract-ocr --example golden_bench` and
`python3 corpus/gen/run_cli_golden.py --bench` (idle machine, or the numbers
are garbage — see the discarded contaminated run above).
