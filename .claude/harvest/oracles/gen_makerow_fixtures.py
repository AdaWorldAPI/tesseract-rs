#!/usr/bin/env python3
"""Deterministic synthetic fixture generator for the Batch-3E wave-2 oracle
(the makerow.cpp row-assignment + cleanup chain, single-column/single-block
case).

Format of each /tmp/makerow_input_seed<N>_<noise|clean>.bin (little-endian
throughout; both readers -- the Rust `examples/makerow_dump.rs` and the C++
`/tmp/makerow_oracle.cpp` -- hardcode this order):

  u32 n_blobs
  n_blobs x (i32 left, i32 bottom, i32 right, i32 top)   -- the block's
                                                             initial blob pool
  f32 line_spacing   -- TO_BLOCK::line_spacing (initial estimate)
  f32 line_size      -- TO_BLOCK::line_size (initial estimate)
  f32 max_blob_size  -- TO_BLOCK::max_blob_size (initial estimate)
  i32 block_left     -- block->block->pdblk.bounding_box().left() (the
                         assign_blobs_to_rows empty-pool fallback only)

Layout: `n_lines` (3-5) horizontal text lines, each `line_gap` apart
(baseline-to-baseline), each with several variable-width blobs and small
per-blob vertical jitter (baseline + ascender/descender noise), optionally
plus a handful of isolated small "noise" blobs off any line's y-range, then
the whole blob list is shuffled (the real pipeline re-sorts by x internally,
but shuffling also mixes which line a blob nominally "belongs to" in
insertion order, exercising the row-bubble-sort machinery in
assign_blobs_to_rows).
"""
import random
import struct


def write_fixture(path, seed, with_noise):
    rng = random.Random(seed)
    n_lines = rng.choice([3, 4, 5])
    line_height = 20
    line_gap = 40
    blobs = []
    x_cursor_by_line = []
    for li in range(n_lines):
        base_bottom = li * line_gap
        n_blobs_in_line = rng.randint(4, 8)
        x = 0
        for _bi in range(n_blobs_in_line):
            w = rng.randint(6, 16)
            bottom_jitter = rng.randint(-2, 2)
            top_jitter = rng.randint(-3, 5)
            left = x
            right = x + w
            bottom = base_bottom + bottom_jitter
            top = base_bottom + line_height + top_jitter
            blobs.append((left, bottom, right, top))
            x = right + rng.randint(2, 6)
        x_cursor_by_line.append(x)
    max_x = max(x_cursor_by_line)

    if with_noise:
        n_noise = rng.randint(2, 4)
        for _ in range(n_noise):
            nx = rng.randint(0, max_x)
            ny = rng.randint(-30, n_lines * line_gap + 60)
            nw = rng.randint(2, 5)
            nh = rng.randint(2, 5)
            blobs.append((nx, ny, nx + nw, ny + nh))

    rng.shuffle(blobs)

    line_size = float(line_height)
    line_spacing = float(line_gap)
    max_blob_size = line_spacing * 1.3

    out = bytearray()
    out.extend(struct.pack("<I", len(blobs)))
    for (l, b, r, t) in blobs:
        out.extend(struct.pack("<iiii", l, b, r, t))
    out.extend(struct.pack("<fff", line_spacing, line_size, max_blob_size))
    out.extend(struct.pack("<i", 0))

    with open(path, "wb") as fh:
        fh.write(out)
    print(f"wrote {path}: {len(blobs)} blobs ({n_lines} lines, noise={with_noise}), {len(out)} bytes")


if __name__ == "__main__":
    write_fixture("/tmp/makerow_input_seed1_clean.bin", seed=0xC0FFEE, with_noise=False)
    write_fixture("/tmp/makerow_input_seed1_noise.bin", seed=0xC0FFEE, with_noise=True)
    write_fixture("/tmp/makerow_input_seed2_clean.bin", seed=0xFACADE, with_noise=False)
    write_fixture("/tmp/makerow_input_seed2_noise.bin", seed=0xFACADE, with_noise=True)
