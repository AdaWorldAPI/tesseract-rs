#!/usr/bin/env python3
"""Reproducible-from-scratch HierText line-crop generator (OCR benchmark
infrastructure -- NOT a Tesseract transcode; nothing here harvests or
mirrors any C++ shape).

Fetches the HierText validation ground truth plus a bounded number of
Open-Images-validation JPEGs, extracts the LEGIBLE / HORIZONTAL / LATIN
text-line crops (the subset where a classic-OCR recognizer has a fair
shot -- large-rotation scene text is out of scope for both engines and
would measure the line finder, not recognition), writes each crop as a
grey P5 PGM, and emits a JSONL manifest for the Rust benchmark harnesses.

Data sources (both reachable through this environment's proxy):
  * GT:     https://raw.githubusercontent.com/google-research-datasets/
            hiertext/main/gt/validation.jsonl.gz
            Despite the ".jsonl" name this is a SINGLE gzip-compressed
            JSON object (verified against a real ~12MB download):
              {"info": {...},
               "annotations": [
                 {"image_id": str, "image_width": int, "image_height": int,
                  "paragraphs": [
                    {"vertices": [[x,y], ...], "legible": bool,
                     "lines": [
                       {"vertices": [[x,y]x4], "text": str,
                        "legible": bool, "handwritten": bool,
                        "vertical": bool,
                        "words": [{"vertices": [...], "text": str,
                                   "legible": bool, "handwritten": bool,
                                   "vertical": bool}, ...]},
                       ...
                     ]},
                    ...
                  ]},
                 ...
               ]}
            `vertices` is consistently a 4-point quadrilateral
            [top-left, top-right, bottom-right, bottom-left] for
            horizontal lines. `text` is always present on a line dict
            (never a missing key) but is the empty string whenever
            `legible` is false; whenever `legible` is true, `text` is
            always non-empty and matches the words joined by a single
            space in the overwhelming majority of cases (13145/13149
            sampled).
  * Images: streamed from the Open-Images-validation OCR tarball
            https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz
            (577 MB). Opened as a non-seekable gzip tar stream and read
            member-by-member; the stream is abandoned (not fully
            downloaded) as soon as K ".jpg" members have been read. Each
            member's filename stem (basename, ".jpg" suffix stripped,
            independent of directory depth inside the archive) is the
            `image_id` used to join against the ground truth.

Filter (applied in this order; a line failing more than one filter is
bucketed under the FIRST one it fails -- this order matches the priority
list below):
  1. legible        -- line["legible"] must be true.
  2. vertical       -- line["vertical"] must be false (top-to-bottom
                        character stacking is out of scope here).
  3. rotation       -- |rotation_deg| <= MAX_ROTATION_DEG (see
                        `line_rotation_deg` for the exact heuristic).
  4. latin fraction -- >= MIN_LATIN_FRACTION of the line's gt_text
                        characters must be "printable ASCII or a
                        Latin-1-Supplement letter" (see
                        `is_latinish_char`); this drops CJK/Arabic/etc.
  5. box size       -- the (image-clipped, unpadded) axis-aligned bbox
                        of the line's vertices must be at least
                        MIN_BOX_DIM px in both width and height.

Every drop is tallied and the tally is always printed at the end (never
truncated silently): "dropped: illegible=N vertical=N rotated=N
nonlatin=N tiny=N; kept=M lines from J images".

Crop + manifest:
  * gt_text is line["text"] if non-empty, else the line's words joined
    by a single space (defensive fallback; on real validation data
    `text` is always present and non-empty for every legible line, so
    this fallback is not expected to fire in practice -- see the module
    docstring's "text" note above).
  * The saved crop is the axis-aligned bbox of the line's vertices,
    clipped to the image, THEN padded by CROP_PAD px on each side, THEN
    re-clipped to the image -- converted to 8-bit grey ("L" mode),
    cropped, and written as a binary (P5) PGM:
    `lines/{image_id}_{line_idx}.pgm` (path relative to outdir).
  * `line_idx` numbers every line of an image in flattened
    (paragraph-order, then line-order) GT order, starting at 0 --
    including dropped lines, so a line's idx is stable regardless of
    which other lines in the same image pass or fail the filter.
  * One compact JSON object per KEPT line is appended to
    `lines_manifest.jsonl` (one object per physical line, object built
    via `json.dumps` so quotes/unicode in gt_text are always escaped
    safely):
      {"crop": "lines/{image_id}_{line_idx}.pgm", "image_id": str,
       "line_idx": int, "gt_text": str, "w": int, "h": int,
       "rotation_deg": float, "n_words": int}
    `w`/`h` are the SAVED crop's pixel dimensions (post pad-and-reclip);
    `rotation_deg` is the signed heuristic angle, rounded to 0.1.
  * Iteration order is deterministic: matched image_ids sorted
    ascending, `line_idx` ascending within each image.

Robustness:
  * A GT image_id with no streamed image is silently skipped (most GT
    entries will not be among the first K images fetched from the
    tarball; only images actually fetched ever produce manifest lines).
  * The S3 tar stream is wrapped in try/except -- a mid-stream network
    hiccup is reported cleanly (to stderr) and whatever images were
    already fetched before the hiccup are still processed, rather than
    the whole run dying on a bare traceback.
  * Re-running is idempotent: crops and the manifest are overwritten.

Usage:
    python3 gen_hiertext_lines.py [K] [outdir]

`K` (default 60) is the number of ".jpg" tar members read from the
Open-Images tarball before the stream is abandoned. `outdir` defaults to
this script's own directory (`corpus/hiertext/`), independent of the
caller's current working directory; `<outdir>/lines/` is created if
missing.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

# --- Data sources -----------------------------------------------------------

GT_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "hiertext/main/gt/validation.jsonl.gz"
)
IMAGES_TAR_URL = "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"

DEFAULT_K = 60
FETCH_TIMEOUT_S = 60

# --- Filter + crop constants -------------------------------------------------

MAX_ROTATION_DEG = 8.0
MIN_LATIN_FRACTION = 0.80
MIN_BOX_DIM = 8
CROP_PAD = 4

LINES_SUBDIR = "lines"
MANIFEST_NAME = "lines_manifest.jsonl"

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)


# --- Ground truth -------------------------------------------------------------


def fetch_gt(url: str = GT_URL, timeout: int = FETCH_TIMEOUT_S) -> dict[str, Any]:
    """Download + parse the HierText validation ground truth.

    See the module docstring for the verified JSON shape. The file is
    small enough (~12 MB compressed) to hold fully in memory, unlike the
    577 MB image tarball which is streamed instead (`fetch_images`).
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(gzip.decompress(raw))


def line_rotation_deg(vertices: list[list[int]]) -> float:
    """Rotation heuristic, in degrees: the angle of the vector from the
    polygon's first vertex to its second vertex ("top edge", using
    HierText's consistent per-line vertex ordering of top-left,
    top-right, bottom-right, bottom-left for horizontal lines). 0 means
    perfectly horizontal; the sign follows standard image coordinates
    (y grows downward), so a small positive value is a slight clockwise
    tilt.

    This measures BOX rotation, not text-stacking direction -- vertical
    (top-to-bottom) lines are excluded upstream via the `vertical` flag
    before this heuristic is ever consulted. Empirically (checked
    against ~300 real validation images before writing this filter),
    `vertical=true` boxes are themselves axis-aligned narrow columns, so
    this heuristic and the `vertical` filter never disagree in practice.
    Degenerate near-square single-glyph boxes can produce large angles
    from this heuristic (the "top edge" is ill-defined when width and
    height are comparable); those are caught separately by the
    `MIN_BOX_DIM` size filter in the large majority of cases, and the
    remainder are an accepted rough edge of a heuristic the task brief
    explicitly allows to be approximate.
    """
    (x0, y0), (x1, y1) = vertices[0], vertices[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def is_latinish_char(c: str) -> bool:
    """True for printable ASCII (0x20-0x7E) or a Latin-1-Supplement
    letter (0xC0-0xFF and alphabetic -- this excludes the two non-letter
    code points in that block, U+00D7 MULTIPLICATION SIGN and U+00F7
    DIVISION SIGN)."""
    o = ord(c)
    if 0x20 <= o <= 0x7E:
        return True
    if 0xC0 <= o <= 0xFF and c.isalpha():
        return True
    return False


def latin_fraction(text: str) -> float:
    """Fraction of `text`'s characters that are `is_latinish_char`. Empty
    text scores 0.0 (never passes the >= MIN_LATIN_FRACTION gate)."""
    if not text:
        return 0.0
    return sum(1 for c in text if is_latinish_char(c)) / len(text)


def line_text(line: dict[str, Any]) -> str:
    """gt_text for a line: `line["text"]` if non-empty, else its words
    joined by a single space. See the module docstring's "text" note --
    on real data `text` is always present and, for every legible line,
    always non-empty, so the word-join branch is a defensive fallback
    rather than the common path."""
    text = line.get("text") or ""
    if text:
        return text
    words = line.get("words") or []
    return " ".join(w.get("text", "") for w in words)


def bbox_of(vertices: list[list[int]]) -> Box:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def clip_box(box: Box, w: int, h: int) -> Box:
    x0, y0, x1, y1 = box
    return (
        max(0, min(x0, w)),
        max(0, min(y0, h)),
        max(0, min(x1, w)),
        max(0, min(y1, h)),
    )


def save_crop_pgm(path: Path, gray_img: Image.Image, box: Box) -> tuple[int, int]:
    """Crop `gray_img` (already mode "L") to `box` and write it as a
    binary (P5) grey PGM: header `P5\\n{w} {h}\\n255\\n` followed by raw
    pixel bytes. Returns the saved crop's (width, height)."""
    crop = gray_img.crop(box)
    w, h = crop.size
    data = crop.tobytes()
    assert len(data) == w * h, f"unexpected pixel buffer size for {path}"
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(data)
    return w, h


# --- Image tar streaming ------------------------------------------------------


def fetch_images(
    url: str = IMAGES_TAR_URL, k: int = DEFAULT_K, timeout: int = FETCH_TIMEOUT_S
) -> dict[str, bytes]:
    """Stream `url` (a .tgz of JPEGs) and return up to `k` JPEGs as
    {image_id: raw_bytes}, WITHOUT downloading the whole archive: the
    tar is opened in non-seekable streaming mode ("r|gz") and abandoned
    as soon as `k` ".jpg" members have been read.

    `image_id` is the member's filename stem (basename, ".jpg" suffix
    stripped), independent of the member's directory depth inside the
    archive. Every ".jpg" member encountered counts toward `k`, whether
    or not it ends up matching a ground-truth image_id -- the GT join
    happens afterward in `main()`.

    A mid-stream network/tar error is caught and reported cleanly to
    stderr; whatever was already fetched before the error is returned
    rather than the whole run dying on a bare traceback.
    """
    fetched: dict[str, bytes] = {}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(".jpg"):
                        continue
                    image_id = Path(member.name).stem
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    fetched[image_id] = fh.read()
                    if len(fetched) >= k:
                        break
    except (OSError, urllib.error.URLError, tarfile.TarError) as exc:
        print(
            f"WARNING: image stream interrupted ({exc!r}); "
            f"proceeding with {len(fetched)} image(s) already fetched",
            file=sys.stderr,
        )
    return fetched


# --- Per-image line extraction -------------------------------------------------

DROP_REASONS = ("illegible", "vertical", "rotated", "nonlatin", "tiny")


def process_image(
    image_id: str,
    ann: dict[str, Any],
    gray_img: Image.Image,
    lines_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter + crop every line of one fetched+GT-matched image.

    Returns (manifest_records, counts) where `counts` has one key per
    `DROP_REASONS` entry plus "kept", scoped to this single image (the
    caller aggregates across images).
    """
    img_w, img_h = gray_img.size
    counts = {reason: 0 for reason in DROP_REASONS}
    counts["kept"] = 0
    records: list[dict[str, Any]] = []

    line_idx = 0
    for para in ann.get("paragraphs", []):
        for line in para.get("lines", []):
            idx = line_idx
            line_idx += 1

            if not line.get("legible", False):
                counts["illegible"] += 1
                continue
            if line.get("vertical", False):
                counts["vertical"] += 1
                continue

            vertices = line["vertices"]
            rotation = line_rotation_deg(vertices)
            if abs(rotation) > MAX_ROTATION_DEG:
                counts["rotated"] += 1
                continue

            text = line_text(line)
            if latin_fraction(text) < MIN_LATIN_FRACTION:
                counts["nonlatin"] += 1
                continue

            raw_box = clip_box(bbox_of(vertices), img_w, img_h)
            box_w, box_h = raw_box[2] - raw_box[0], raw_box[3] - raw_box[1]
            if box_w < MIN_BOX_DIM or box_h < MIN_BOX_DIM:
                counts["tiny"] += 1
                continue

            crop_box = clip_box(
                (
                    raw_box[0] - CROP_PAD,
                    raw_box[1] - CROP_PAD,
                    raw_box[2] + CROP_PAD,
                    raw_box[3] + CROP_PAD,
                ),
                img_w,
                img_h,
            )
            crop_name = f"{image_id}_{idx}.pgm"
            crop_w, crop_h = save_crop_pgm(lines_dir / crop_name, gray_img, crop_box)

            counts["kept"] += 1
            records.append(
                {
                    "crop": f"{LINES_SUBDIR}/{crop_name}",
                    "image_id": image_id,
                    "line_idx": idx,
                    "gt_text": text,
                    "w": crop_w,
                    "h": crop_h,
                    "rotation_deg": round(rotation, 1),
                    "n_words": len(line.get("words") or []),
                }
            )

    return records, counts


# --- Orchestration --------------------------------------------------------------


def main(argv: list[str]) -> int:
    k = int(argv[1]) if len(argv) > 1 else DEFAULT_K
    outdir = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parent
    lines_dir = outdir / LINES_SUBDIR
    lines_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching ground truth from {GT_URL} ...")
    gt = fetch_gt()
    annotations = gt.get("annotations", [])
    gt_by_id: dict[str, dict[str, Any]] = {ann["image_id"]: ann for ann in annotations}
    print(f"GT loaded: {len(annotations)} annotated images (info={gt.get('info')})")

    print(f"Streaming up to {k} images from {IMAGES_TAR_URL} ...")
    images = fetch_images(IMAGES_TAR_URL, k)
    print(f"Fetched {len(images)} image(s) from the archive")

    matched_ids = sorted(set(images) & set(gt_by_id))
    unmatched_ids = sorted(set(images) - set(gt_by_id))
    if unmatched_ids:
        preview = unmatched_ids[:5]
        more = "..." if len(unmatched_ids) > 5 else ""
        print(
            f"NOTE: {len(unmatched_ids)} fetched image(s) have no GT entry, "
            f"skipped: {preview}{more}"
        )

    totals = {reason: 0 for reason in DROP_REASONS}
    totals["kept"] = 0
    manifest_records: list[dict[str, Any]] = []
    images_used = 0

    for image_id in matched_ids:
        raw = images[image_id]
        try:
            img = Image.open(io.BytesIO(raw))
            gray = img.convert("L")
        except Exception as exc:  # decode failure -> clean skip, not a crash
            print(
                f"WARNING: failed to decode image {image_id}: {exc!r}; skipping",
                file=sys.stderr,
            )
            continue

        ann = gt_by_id[image_id]
        records, counts = process_image(image_id, ann, gray, lines_dir)
        manifest_records.extend(records)
        for key in totals:
            totals[key] += counts[key]
        images_used += 1

    manifest_path = outdir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    print(
        "dropped: "
        + " ".join(f"{reason}={totals[reason]}" for reason in DROP_REASONS)
        + f"; kept={totals['kept']} lines from {images_used} images"
    )
    print(f"manifest: {manifest_path} ({len(manifest_records)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
