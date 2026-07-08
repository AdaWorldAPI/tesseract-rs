#!/usr/bin/env python3
"""CLI-side companion to the `h2h_speed` Rust example — the MATCHED half of the
speed head-to-head.

The plain per-page-subprocess CLI number (see `.claude/harvest/p6-perf-report.md`)
pays process startup + traineddata load on EVERY page, which our in-process
harness pays ONCE. That is not apples-to-apples. This script fixes it: it runs
the C++ CLI in **image-list (batch) mode** — one process over all N crops, so
the model loads once and the throughput number amortizes load exactly as our
harness does. Single-threaded (`OMP_THREAD_LIMIT=1`) to match our single-thread
path, and peak child RSS via `getrusage(RUSAGE_CHILDREN)` to match our VmHWM
methodology (peak-per-process, both sides).

It also isolates the cold model-load cost: (single-image wall) − (per-image
amortized batch wall) ≈ the one-time load the batch amortizes away.

Usage: python3 run_cli_speed.py [N]   # N = cap on crops (default 100)
Reads corpus/hiertext/lines_manifest.jsonl (run gen_hiertext_lines.py first);
falls back to corpus/lines/*.pgm if absent.
"""

import json
import pathlib
import resource
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE.parent


def crop_list(cap: int) -> list[str]:
    manifest = HERE / "lines_manifest.jsonl"
    if manifest.exists():
        crops = []
        for ln in manifest.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            rel = json.loads(ln)["crop"]
            p = HERE / rel
            if p.exists():
                crops.append(str(p))
        src = "hiertext lines_manifest"
    else:
        crops = sorted(str(p) for p in (CORPUS / "lines").glob("*.pgm"))
        src = "corpus/lines fallback"
    crops = crops[:cap]
    print(f"corpus: {src} ({len(crops)} crops, cap {cap})")
    return crops


def run_batch(flist: pathlib.Path) -> float:
    """One CLI process over the whole image list; returns wall seconds."""
    t0 = time.perf_counter()
    cmd = ["tesseract", str(flist), "stdout", "--psm", "7"]
    # codex P2: pin the CLI to the committed model bytes, not the host's
    # installed eng.traineddata.
    if _PINNED_TESSDATA is not None:
        cmd += ["--tessdata-dir", str(_PINNED_TESSDATA)]
    cmd += ["-l", "eng", "--oem", "1"]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"},
    )
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-500:])
        raise SystemExit("tesseract batch run failed")
    return dt


# Filled in by main(): a temp dir holding an eng.traineddata recombined from the
# committed corpus/model components, so the CLI runs the SAME model the Rust
# side loads (codex P2). None => combine_tessdata unavailable, host model used.
_PINNED_TESSDATA = None


def pin_model() -> "pathlib.Path | None":
    """Recombine corpus/model/eng.lstm* into a temp eng.traineddata via
    combine_tessdata; return its dir for --tessdata-dir, or None on failure."""
    import shutil
    import tempfile

    model = CORPUS / "model"
    comps = [
        "eng.lstm", "eng.lstm-unicharset", "eng.lstm-recoder",
        "eng.lstm-word-dawg", "eng.lstm-punc-dawg", "eng.lstm-number-dawg",
    ]
    if not all((model / c).exists() for c in comps):
        return None
    dst = pathlib.Path(tempfile.mkdtemp(prefix="tesseract_h2h_pin_"))
    for c in comps:
        shutil.copy(model / c, dst / c)
    r = subprocess.run(
        ["combine_tessdata", str(dst / "eng.")],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    if r.returncode == 0 and (dst / "eng.traineddata").exists():
        return dst
    return None


def main() -> None:
    global _PINNED_TESSDATA
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    crops = crop_list(cap)
    if not crops:
        raise SystemExit("no crops — run gen_hiertext_lines.py first")

    _PINNED_TESSDATA = pin_model()
    if _PINNED_TESSDATA is not None:
        print(f"CLI pinned to committed model: --tessdata-dir {_PINNED_TESSDATA}")
    else:
        print("WARNING: combine_tessdata unavailable — CLI uses host eng.traineddata (codex P2)")

    flist = HERE / "_cli_speed_flist.txt"
    flist.write_text("\n".join(crops) + "\n")
    single = HERE / "_cli_speed_single.txt"
    single.write_text(crops[0] + "\n")

    # Warm-up (discard) then 3 timed batch passes, keep the best (steadiest).
    run_batch(flist)
    batch_best = min(run_batch(flist) for _ in range(3))

    # Cold isolation: a single-image process (pays full load for 1 page).
    run_batch(single)  # warm the fs cache
    single_best = min(run_batch(single) for _ in range(3))

    n = len(crops)
    per_image_ms = batch_best / n * 1000.0
    # load ≈ (1-image wall) − (1 image's amortized share)
    load_ms = max(0.0, (single_best - batch_best / n) * 1000.0)
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # KiB on Linux

    print("\n## CLI-side speed (batch/image-list, single-thread, amortized load)\n")
    print(f"| metric | value |")
    print(f"|---|---|")
    print(f"| crops | {n} |")
    print(f"| batch wall (best of 3) | {batch_best * 1000:.1f} ms |")
    print(f"| per image (amortized) | {per_image_ms:.2f} ms |")
    print(f"| images/sec | {n / batch_best:.2f} |")
    print(f"| cold model-load (isolated) | {load_ms:.0f} ms one-time |")
    print(f"| peak child RSS | {peak_rss} KiB |")
    print(
        "\n_Matched to `h2h_speed`: model loads ONCE (batch), single-thread "
        "(`OMP_THREAD_LIMIT=1`), peak-per-process RSS. Compare images/sec + "
        "peak RSS directly; the per-page-subprocess number in the P6 perf "
        "report is NOT matched (it re-loads per page) and must not be used for "
        "the head-to-head ratio._"
    )
    for f in (flist, single):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
