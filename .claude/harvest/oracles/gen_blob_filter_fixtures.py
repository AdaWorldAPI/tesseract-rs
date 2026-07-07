#!/usr/bin/env python3
"""Deterministic synthetic fixture generator for the Batch-3F2 leaf-2 oracle
(Textord::filter_noise_blobs + Textord::filter_blobs, tordmain.cpp:238-360).

Format of each /tmp/blob_filter_input_seed<N>_<clean|mixed>.bin
(little-endian throughout; both readers -- the Rust
`examples/blob_filter_dump.rs` and the C++ `/tmp/blob_filter_oracle.cpp` --
hardcode this order):

  u32 n_blobs
  n_blobs x (i32 left, i32 bottom, i32 right, i32 top, i32 pixel_count)

`pixel_count` stands in for `BLOBNBOX::enclosed_area()` (Batch 3F2 leaf 1's
`conn_comp_areas` pixel_count on the Rust side); (left,bottom,right,top) is
`top - bottom = height` / `right - left = width` with `top > bottom` (the
tesseract-ocr crate's established box-tuple convention, see blob_filter.rs).

Two configs per seed, mirroring the Batch 3E wave-2
`gen_makerow_fixtures.py` clean/noise pairing:
  - "clean": ~30 blobs, a normal text-like low-density population plus a
    handful of near-boundary low-density blobs straddling the
    noise_area_ratio=0.7 threshold. No noise-height or oversized blobs.
  - "mixed": the clean population PLUS noise-height blobs (height <
    textord_max_noise_size=7) and oversized outliers (far taller / far
    wider than the population) -- ~40 blobs total, exercising every branch
    of filter_noise_blobs (noise / small-then-rescued / small-then-stuck /
    large-from-rescue / large-from-repartition / stays-in-pool).
"""
import random
import struct


def normal_population(rng, x, count=24):
    """Text-like low-density population (heights ~16-24, density 15-45%)."""
    blobs = []
    for _ in range(count):
        w = rng.randint(6, 14)
        h = rng.randint(16, 24)
        left = x
        right = x + w
        bottom = 0
        top = h
        area = w * h
        density = rng.uniform(0.15, 0.45)
        pixel_count = max(1, int(area * density))
        blobs.append((left, bottom, right, top, pixel_count))
        x = right + rng.randint(2, 6)
    return blobs, x


def boundary_population(rng, x, count=6):
    """Near-boundary blobs straddling the 0.7 noise_area_ratio (height >= 7
    so the noise-height check never intercepts them first)."""
    blobs = []
    for i in range(count):
        w = rng.randint(8, 12)
        h = rng.randint(8, 16)
        left = x
        right = x + w
        bottom = 0
        top = h
        area = w * h
        density = rng.uniform(0.60, 0.65) if i % 2 == 0 else rng.uniform(0.75, 0.85)
        pixel_count = max(1, min(area, int(area * density)))
        blobs.append((left, bottom, right, top, pixel_count))
        x = right + rng.randint(2, 6)
    return blobs, x


def noise_height_population(rng, x, count=6):
    """height < textord_max_noise_size(7); density is irrelevant here since
    the height check short-circuits before the area-ratio check runs."""
    blobs = []
    for _ in range(count):
        w = rng.randint(2, 6)
        h = rng.randint(1, 6)
        left = x
        right = x + w
        bottom = rng.randint(-5, 40)
        top = bottom + h
        area = max(1, w * h)
        pixel_count = rng.randint(1, area)
        blobs.append((left, bottom, right, top, pixel_count))
        x = right + rng.randint(2, 6)
    return blobs, x


def oversized_population(rng, x, tall_count=2, wide_count=2):
    """Far taller / far wider than the normal population, low density (so
    pass 1 doesn't reroute them to `small` first) -- exercises the
    re-partition pass's height>max_y / width>max_x eviction directly."""
    blobs = []
    for _ in range(tall_count):
        w = rng.randint(6, 14)
        h = rng.randint(150, 220)
        left = x
        right = x + w
        bottom = 0
        top = h
        pixel_count = max(1, int(w * h * 0.2))
        blobs.append((left, bottom, right, top, pixel_count))
        x = right + rng.randint(2, 6)
    for _ in range(wide_count):
        w = rng.randint(100, 160)
        h = rng.randint(16, 24)
        left = x
        right = x + w
        bottom = 0
        top = h
        pixel_count = max(1, int(w * h * 0.2))
        blobs.append((left, bottom, right, top, pixel_count))
        x = right + rng.randint(2, 6)
    return blobs, x


def write_fixture(path, seed, mixed):
    rng = random.Random(seed)
    blobs = []
    x = 0

    part, x = normal_population(rng, x)
    blobs += part
    part, x = boundary_population(rng, x)
    blobs += part

    if mixed:
        part, x = noise_height_population(rng, x)
        blobs += part
        part, x = oversized_population(rng, x)
        blobs += part

    rng.shuffle(blobs)

    out = bytearray()
    out.extend(struct.pack("<I", len(blobs)))
    for (l, b, r, t, pc) in blobs:
        out.extend(struct.pack("<iiiii", l, b, r, t, pc))

    with open(path, "wb") as fh:
        fh.write(out)
    print(f"wrote {path}: {len(blobs)} blobs (mixed={mixed}), {len(out)} bytes")


if __name__ == "__main__":
    write_fixture("/tmp/blob_filter_input_seed1_clean.bin", seed=0xC0FFEE, mixed=False)
    write_fixture("/tmp/blob_filter_input_seed1_mixed.bin", seed=0xC0FFEE, mixed=True)
    write_fixture("/tmp/blob_filter_input_seed2_clean.bin", seed=0xFACADE, mixed=False)
    write_fixture("/tmp/blob_filter_input_seed2_mixed.bin", seed=0xFACADE, mixed=True)
