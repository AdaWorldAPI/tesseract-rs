#!/usr/bin/env python3
"""Deterministic synthetic fixture generator for the Batch-3E wave-1 oracle.

Format of /tmp/textline_math_input.bin (little-endian throughout), fixed
section order (no tags -- both readers hardcode this order):

SECTION 1 -- STATS histogram
  i32 rangemin, i32 rangemax
  u32 n_entries; n_entries x (i32 value, i32 count)   -- fed via STATS::add
  u32 n_fracs;   n_fracs x f64 frac                    -- fed to STATS::ile

SECTION 2 -- occupation / dropout / threshold arrays
  u32 line_count; line_count x i32 occupation
  i32 low_window, i32 high_window, f64 occupancy_threshold

SECTION 3 -- height modes (separate STATS histogram)
  i32 rangemin, i32 rangemax
  u32 n_entries; n_entries x (i32 value, i32 count)
  i32 min_height, i32 max_height, i32 maxmodes

SECTION 4 -- fill_heights
  u32 n_boxes; n_boxes x (i32 left, i32 bottom, i32 right, i32 top)
  f32 gradient, f32 parallel_c, i32 min_height, i32 max_height,
  f32 min_blob_height_fraction

SECTION 5 -- compute_line_occupation
  u32 n_blobs; n_blobs x (i32 left, i32 bottom, i32 right, i32 top)
  f32 gradient, i32 min_y, i32 max_y

SECTION 6 -- DetLineFit configs
  u32 n_configs; per config:
    u8 kind (0 = Fit(0,0), 1 = ConstrainedFit(direction), 2 = ConstrainedFit(m))
    u32 n_points; n_points x (i32 x, i32 y, i32 halfwidth)
    kind==1: f32 dirx, f32 diry, f64 min_dist, f64 max_dist
    kind==2: f64 m
"""
import struct
import random

out = bytearray()


def i32(v):
    out.extend(struct.pack("<i", v))


def u32(v):
    out.extend(struct.pack("<I", v))


def u8(v):
    out.extend(struct.pack("<B", v))


def f32(v):
    out.extend(struct.pack("<f", v))


def f64(v):
    out.extend(struct.pack("<d", v))


rng = random.Random(0xC0FFEE)

# ---------- Section 1: STATS histogram + ile fracs ----------
i32(0)
i32(50)
entries = []
for _ in range(40):
    entries.append((rng.randint(0, 50), rng.randint(1, 9)))
# guarantee the header's own worked example is embedded via a fresh separate
# histogram below (section 3); this histogram is just a general stress case.
u32(len(entries))
for v, c in entries:
    i32(v)
    i32(c)
fracs = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
u32(len(fracs))
for f in fracs:
    f64(f)

# ---------- Section 2: occupation / dropout / threshold ----------
line_count = 40
occupation = []
val = 0
for i in range(line_count):
    val += rng.randint(-5, 6)
    occupation.append(val)
u32(line_count)
for o in occupation:
    i32(o)
i32(4)  # low_window
i32(3)  # high_window
f64(0.4)  # occupancy_threshold

# ---------- Section 3: height modes histogram (bimodal + header example) ----------
i32(0)
i32(30)
hm_entries = [(6, 2), (13, 1), (14, 1), (10, 5), (20, 3), (21, 1)]
u32(len(hm_entries))
for v, c in hm_entries:
    i32(v)
    i32(c)
i32(0)   # min_height
i32(30)  # max_height
i32(4)   # maxmodes

# ---------- Section 4: fill_heights ----------
fh_boxes = []
for i in range(15):
    left = rng.randint(0, 100)
    right = left + rng.randint(2, 20)
    bottom = rng.randint(0, 50)
    top = bottom + rng.randint(3, 25)
    fh_boxes.append((left, bottom, right, top))
u32(len(fh_boxes))
for (l, b, r, t) in fh_boxes:
    i32(l)
    i32(b)
    i32(r)
    i32(t)
f32(0.05)   # gradient
f32(-2.0)   # parallel_c
i32(0)      # min_height
i32(30)     # max_height
f32(0.75)   # min_blob_height_fraction

# ---------- Section 5: compute_line_occupation ----------
occ_boxes = []
for i in range(12):
    left = rng.randint(0, 200)
    right = left + rng.randint(2, 15)
    bottom = rng.randint(0, 60)
    top = bottom + rng.randint(3, 20)
    occ_boxes.append((left, bottom, right, top))
u32(len(occ_boxes))
for (l, b, r, t) in occ_boxes:
    i32(l)
    i32(b)
    i32(r)
    i32(t)
f32(0.08)  # gradient
i32(-5)    # min_y
i32(90)    # max_y

# ---------- Section 6: DetLineFit configs ----------
configs = []

# Config 0: exact collinear points, plain Fit
pts0 = [(i, 2 * i + 3, 0) for i in range(8)]
configs.append((0, pts0, None))

# Config 1: near-collinear with outliers, plain Fit
pts1 = [(0, 1, 0), (1, 3, 0), (2, 5, 0), (3, 20, 0), (4, 9, 0), (5, 11, 0), (6, 13, 0), (7, 100, 0), (8, 17, 0)]
configs.append((1, pts1, None))

# Config 2: <=2 points
pts2 = [(3, 4, 0), (9, 20, 0)]
configs.append((2, pts2, None))

# Config 3: single point
pts3 = [(5, 5, 0)]
configs.append((3, pts3, None))

# Config 4: ConstrainedFit with a direction vector (near-horizontal), some
# points filtered by [min_dist,max_dist]
pts4 = [(i, (i % 3) - 1 + 10, 0) for i in range(-6, 7)]
configs.append((4, pts4, ("dir", 0.9995, 0.0316, -50.0, 50.0)))

# Config 5: ConstrainedFit(m) backwards-compatible wrapper
pts5 = [(i, 3 * i - 2 + ((i * 37) % 5 - 2), 0) for i in range(-5, 6)]
configs.append((5, pts5, ("m", 3.0)))

# Config 6: points with halfwidths overlapping (diacritic suppression path)
pts6 = [(0, 0, 0), (10, 0, 0), (10, 1, 8), (20, 0, 0), (30, 0, 0)]
configs.append((6, pts6, None))

u32(len(configs))
for idx, pts, extra in configs:
    kind = 0 if extra is None else (1 if extra[0] == "dir" else 2)
    u8(kind)
    u32(len(pts))
    for (x, y, hw) in pts:
        i32(x)
        i32(y)
        i32(hw)
    if kind == 1:
        _, dx, dy, mind, maxd = extra
        f32(dx)
        f32(dy)
        f64(mind)
        f64(maxd)
    elif kind == 2:
        _, m = extra
        f64(m)

with open("/tmp/textline_math_input.bin", "wb") as fh:
    fh.write(out)

print(f"wrote {len(out)} bytes")
