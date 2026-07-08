#!/usr/bin/env python3
"""corpus/gen/run_cli_golden.py

Banks the C++ `tesseract` CLI's outputs for the tesseract-rs golden-parity
harness (D6.2 -- see `crates/tesseract-ocr/examples/golden_report.rs`), and
optionally benchmarks the CLI's own wall-clock/RSS cost (D6.3, the CLI side --
the Rust-side counterpart is `crates/tesseract-ocr/examples/golden_bench.rs`).

Corpus layout this script reads/writes (fixed contract, see
`crates/tesseract-ocr/examples/golden_report.rs` for the consumer side):

    corpus/lines/   img_8.pgm img_16.pgm img_24.pgm img_40.pgm img_64.pgm
                    img_100.pgm line36.pgm page_roomy.pgm page_tight.pgm
    corpus/pages/   page_01.pgm .. page_10.pgm (+ page_NN.gt.txt, untouched
                    by this script)
    corpus/golden/cli/lines/<stem>.psm13.txt
    corpus/golden/cli/pages/page_NN.psm6.txt
    corpus/golden/cli/pages/page_NN.psm6.tsv

Default mode writes:
  corpus/golden/cli/lines/<stem>.psm13.txt   -- `tesseract <pgm> stdout --psm 13`
                                                 for every corpus/lines/*.pgm
                                                 EXCEPT page_roomy/page_tight
                                                 (those two live under lines/
                                                 as page-shaped E2E fixtures,
                                                 not single text lines, so a
                                                 single-line PSM would be
                                                 meaningless for them).
  corpus/golden/cli/pages/page_NN.psm6.txt   -- `tesseract <pgm> stdout --psm 6`
  corpus/golden/cli/pages/page_NN.psm6.tsv   -- `tesseract <pgm> stdout --psm 6 tsv`
                                                 for every corpus/pages/*.pgm

--bench mode (D6.3, CLI side) runs `tesseract <page> stdout --psm 6` 3x per
page (a fresh process each time -- unlike the Rust-side bench, which loads
the model once and repeats a full-corpus pass), keeps the PER-PAGE best wall
time, and reports the CLI's cumulative child RSS.

Usage:
    python3 run_cli_golden.py            # bank the golden CLI outputs
    python3 run_cli_golden.py --bench    # D6.3 CLI-side perf bench instead
"""

from __future__ import annotations

import argparse
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Path(__file__).resolve() = .../corpus/gen/run_cli_golden.py
# .parent                  = .../corpus/gen
# .parent.parent           = .../corpus   (the corpus root)
CORPUS_ROOT = Path(__file__).resolve().parent.parent
LINES_DIR = CORPUS_ROOT / "lines"
PAGES_DIR = CORPUS_ROOT / "pages"
GOLDEN_CLI_LINES_DIR = CORPUS_ROOT / "golden" / "cli" / "lines"
GOLDEN_CLI_PAGES_DIR = CORPUS_ROOT / "golden" / "cli" / "pages"

TESSERACT_BIN = "tesseract"

# page_roomy / page_tight live under corpus/lines/ (page-shaped E2E fixtures
# used by the makerow tests) but are not single text lines, so they are
# excluded from the --psm 13 (single line) lines bank.
LINES_EXCLUDE_STEMS = {"page_roomy", "page_tight"}


def run_tesseract(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run the `tesseract` CLI, capturing stdout/stderr as separate byte
    streams (never mixed together)."""
    return subprocess.run(
        [TESSERACT_BIN, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def report_failure(pgm: Path, label: str, proc: subprocess.CompletedProcess[bytes]) -> None:
    stderr_text = proc.stderr.decode("utf-8", "replace").strip()
    print(
        f"FAIL {pgm.name} ({label}): tesseract exited {proc.returncode}\n  stderr: {stderr_text}",
        file=sys.stderr,
    )


def bank_lines() -> bool:
    """Bank `--psm 13` stdout for every corpus/lines/*.pgm (excluding the two
    page-shaped fixtures). Returns True iff every invocation succeeded."""
    ok = True
    GOLDEN_CLI_LINES_DIR.mkdir(parents=True, exist_ok=True)
    for pgm in sorted(LINES_DIR.glob("*.pgm")):
        if pgm.stem in LINES_EXCLUDE_STEMS:
            continue
        proc = run_tesseract([str(pgm), "stdout", "--psm", "13"])
        if proc.returncode != 0:
            ok = False
            report_failure(pgm, "psm13", proc)
            continue
        out_path = GOLDEN_CLI_LINES_DIR / f"{pgm.stem}.psm13.txt"
        out_path.write_bytes(proc.stdout)
        print(f"{out_path.relative_to(CORPUS_ROOT)}\t{len(proc.stdout)} bytes")
    return ok


def bank_pages() -> bool:
    """Bank `--psm 6` stdout (txt) + tsv for every corpus/pages/*.pgm.
    Returns True iff every invocation succeeded."""
    ok = True
    GOLDEN_CLI_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for pgm in sorted(PAGES_DIR.glob("*.pgm")):
        proc = run_tesseract([str(pgm), "stdout", "--psm", "6"])
        if proc.returncode != 0:
            ok = False
            report_failure(pgm, "psm6 txt", proc)
        else:
            txt_path = GOLDEN_CLI_PAGES_DIR / f"{pgm.stem}.psm6.txt"
            txt_path.write_bytes(proc.stdout)
            print(f"{txt_path.relative_to(CORPUS_ROOT)}\t{len(proc.stdout)} bytes")

        proc = run_tesseract([str(pgm), "stdout", "--psm", "6", "tsv"])
        if proc.returncode != 0:
            ok = False
            report_failure(pgm, "psm6 tsv", proc)
        else:
            tsv_path = GOLDEN_CLI_PAGES_DIR / f"{pgm.stem}.psm6.tsv"
            tsv_path.write_bytes(proc.stdout)
            print(f"{tsv_path.relative_to(CORPUS_ROOT)}\t{len(proc.stdout)} bytes")
    return ok


def bench_pages() -> bool:
    """D6.3 CLI-side bench: run `tesseract <page> stdout --psm 6` 3x per page
    (a fresh process each time), report the PER-PAGE best wall time as a
    markdown table, plus the cumulative child RSS read once at the end."""
    pages = sorted(PAGES_DIR.glob("*.pgm"))
    if not pages:
        print(f"SKIP bench (no pages found under {PAGES_DIR})", file=sys.stderr)
        return True

    ok = True
    results: list[tuple[str, float]] = []
    for pgm in pages:
        best: float | None = None
        for _ in range(3):
            start = time.perf_counter()
            proc = run_tesseract([str(pgm), "stdout", "--psm", "6"])
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                ok = False
                report_failure(pgm, "bench psm6", proc)
                continue
            if best is None or elapsed < best:
                best = elapsed
        if best is not None:
            results.append((pgm.stem, best * 1000.0))

    print()
    print("## D6.3 -- C++ tesseract CLI bench (--psm 6, best of 3, fresh process per run)")
    print()
    print("| page | best (ms) |")
    print("|---|---|")
    total_seconds = 0.0
    for stem, ms in results:
        print(f"| {stem} | {ms:.2f} |")
        total_seconds += ms / 1000.0
    print()
    if results and total_seconds > 0:
        print(f"pages/sec (sum of per-page best times): {len(results) / total_seconds:.3f}")

    # ru_maxrss (KiB on Linux) is CUMULATIVE across every reaped child process
    # in this interpreter's lifetime, not a per-page peak -- report it once,
    # at the end, labelled accordingly.
    peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(f"peak child RSS (KiB, cumulative session): {peak_kib}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run the D6.3 CLI-side perf bench instead of banking golden files",
    )
    args = parser.parse_args()

    if shutil.which(TESSERACT_BIN) is None:
        print(f"error: '{TESSERACT_BIN}' not found on PATH", file=sys.stderr)
        return 1

    if args.bench:
        return 0 if bench_pages() else 1

    ok_lines = bank_lines()
    ok_pages = bank_pages()
    return 0 if (ok_lines and ok_pages) else 1


if __name__ == "__main__":
    sys.exit(main())
