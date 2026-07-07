//! Batch 3E, wave 1 — the pure-math `makerow.cpp` leaves (textline
//! formation, `/tmp/tesseract/src/textord/makerow.cpp`) plus
//! `DetLineFit` (`/tmp/tesseract/src/ccstruct/detlinefit.{h,cpp}`), the
//! deterministic least-upper-quartile-squares line fitter `fit_lms_line`
//! is built on. Structure per `.claude/harvest/makerow-callgraph.txt`.
//!
//! Ported leaves:
//! - [`compute_line_occupation`] — pixel projection onto the (de-skewed) y
//!   axis + its first derivative (`makerow.cpp:799-845`).
//! - [`compute_occupation_threshold`] — sliding-window textline-or-not
//!   thresholds over the occupation array (`makerow.cpp:852-926`).
//! - [`compute_dropout_distances`] — in-place distance-to-nearest-dropout
//!   transform (`makerow.cpp:933-967`).
//! - [`compute_height_modes`] — top-N-mode extraction over a [`Stats`]
//!   histogram (`makerow.cpp:1629-1682`).
//! - [`fill_heights`] — per-blob height histogram construction
//!   (`makerow.cpp:1418-1462`), **the non-baseline-spline variant**: the
//!   real `fill_heights` branches on `textord_fix_xheight_bug` between
//!   `row->baseline.y(xcentre)` (a `QSPLINE` evaluation — out of scope for
//!   a pure-math wave with no row/spline object) and the plain
//!   `gradient * xcentre + row->parallel_c()` line. This port takes the
//!   latter (`parallel_c` as a plain `f32` parameter) on bare
//!   `(left, bottom, right, top)` box tuples, and does not replicate the
//!   `joined_to_prev` / repeated-char-run skip (those need `BLOBNBOX`/
//!   `TO_ROW` state this wave's plain-tuple scope excludes).
//! - [`ICoord`], [`FCoord`], [`DetLineFit`] — the general LMS/constrained
//!   line fitter, including `fit_lms_line`'s `Add`+`Fit(float*,float*)`
//!   path.
//!
//! ## Byte-for-byte fidelity notes
//! - `TDimension` (`ICOORD`/`TBOX` coordinate storage) is `int16_t` in a
//!   default (non-`LARGE_IMAGES`) build (`ccutil/tesstypes.h:29-33`); this
//!   port stores coordinates as `i32` (the C++ *promoted* arithmetic type
//!   for every operator used here — `int16_t OP int16_t` promotes to
//!   `int` in C++) since no overflow-at-i16-boundary behaviour is exercised
//!   by any of these leaves.
//! - `ICOORD::rotate` (`points.h:511-516`) computes `floor(x·cos - y·sin +
//!   0.5)` and `floor(y·cos + x·sin + 0.5)` **both from the pre-rotation
//!   `x,y`** (the C++ stores the new x in a temporary before overwriting
//!   `ycoord`, so `ycoord`'s formula still reads the *original* `xcoord`).
//! - `DetLineFit::ComputeDistances`' `square_length_` is assigned from
//!   `ICOORD::sqlength()`, which returns **`f32`** (`points.h:79-81`, exact
//!   integer sum then a single truncating cast to `float`) even though the
//!   field itself is `f64` — so the intermediate rounding to `f32`
//!   precision happens before the widen, and `line_length =
//!   IntCastRounded(sqrt(square_length_))` takes `sqrt` of that
//!   already-`f32`-rounded `f64`.
//! - `kMaxRealDistance` is declared `const int kMaxRealDistance = 2.0;`
//!   (`detlinefit.cpp:39`) — an `int` narrowed from the literal `2.0`,
//!   value `2`. Since `2.0` truncates exactly to `2` this has no
//!   observable effect and is represented here as the `f64` constant
//!   `2.0` directly.
//! - `STATS::ile`'s target-clamp branch is unconditionally the "live"
//!   (non-`#if 0`) one; see [`super::stats`] module doc.

use crate::stats::Stats;

/// `ICOORD` (`ccstruct/points.h`) — an integer 2-vector, with the operator
/// overloads `DetLineFit` needs: `%` (dot/"scalar product") and `*`
/// (cross product).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ICoord {
    /// x coordinate (`TDimension xcoord`).
    pub x: i32,
    /// y coordinate (`TDimension ycoord`).
    pub y: i32,
}

impl ICoord {
    /// Construct from `(x, y)`.
    #[must_use]
    pub fn new(x: i32, y: i32) -> Self {
        ICoord { x, y }
    }

    /// `operator-(ICOORD, ICOORD)` — vector subtraction.
    #[must_use]
    pub fn minus(self, other: Self) -> Self {
        ICoord {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    /// `operator%(ICOORD, ICOORD)` — scalar (dot) product
    /// (`points.h:416-420`).
    #[must_use]
    pub fn dot(self, other: Self) -> i32 {
        self.x * other.x + self.y * other.y
    }

    /// `operator*(ICOORD, ICOORD)` — cross product (`points.h:428-432`).
    #[must_use]
    pub fn cross(self, other: Self) -> i32 {
        self.x * other.y - self.y * other.x
    }

    /// `ICOORD::sqlength` (`points.h:79-81`) — exact integer sum, then a
    /// single cast to `f32` (matching the C++ `static_cast<float>` on the
    /// already-summed `int`).
    #[must_use]
    pub fn sqlength(self) -> f32 {
        (self.x * self.x + self.y * self.y) as f32
    }

    /// `ICOORD::rotate` (`points.h:511-516`) — rotate by the normalized
    /// `(cos, sin)` vector `(vx, vy)`. Both output components are computed
    /// from the *original* `x, y` (see module doc).
    #[must_use]
    pub fn rotate(self, vx: f32, vy: f32) -> Self {
        let x = self.x as f32;
        let y = self.y as f32;
        let new_x = (x * vx - y * vy + 0.5).floor() as i32;
        let new_y = (y * vx + x * vy + 0.5).floor() as i32;
        ICoord { x: new_x, y: new_y }
    }
}

/// `FCOORD` (`ccstruct/points.h`) — a float 2-vector, used here only for
/// `DetLineFit::ConstrainedFit`'s direction vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FCoord {
    /// x coordinate.
    pub x: f32,
    /// y coordinate.
    pub y: f32,
}

impl FCoord {
    /// Construct from `(x, y)`.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        FCoord { x, y }
    }

    /// `FCOORD(ICOORD)` — exact int-to-float conversion constructor
    /// (`points.h:200-204`).
    #[must_use]
    pub fn from_icoord(pt: ICoord) -> Self {
        FCoord {
            x: pt.x as f32,
            y: pt.y as f32,
        }
    }

    /// `operator*(FCOORD, FCOORD)` — cross product (`points.h:628-632`).
    #[must_use]
    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// `FCOORD::sqlength` (`points.h:222-224`).
    #[must_use]
    pub fn sqlength(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
}

/// `IntCastRounded(double)` (`ccutil/helpers.h:181-186`) — round-half-away-
/// from-zero, truncating cast.
fn int_cast_rounded(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        -((-x + 0.5) as i32)
    }
}

const K_NUM_END_POINTS: i32 = 3;
const K_MIN_POINTS_FOR_ERROR_COUNT: usize = 16;
const K_MAX_REAL_DISTANCE: f64 = 2.0;

/// A stored point + its diacritic-suppression halfwidth
/// (`DetLineFit::PointWidth`, `detlinefit.h:109-115`).
#[derive(Debug, Clone, Copy)]
struct PointWidth {
    pt: ICoord,
    halfwidth: i32,
}

/// `tesseract::DetLineFit` (`ccstruct/detlinefit.{h,cpp}`) — deterministic
/// least-upper-quartile-squares line fitting. See the module doc and
/// `detlinefit.h`'s class doc for the algorithm (fits one of the 9 lines
/// through {first 3} x {last 3} points, picking the one with least upper
/// quartile of squared perpendicular distances).
#[derive(Debug, Clone, Default)]
pub struct DetLineFit {
    pts: Vec<PointWidth>,
    /// `(signed cross-product distance, source point)` pairs
    /// (`DistPointPair = KDPairInc<double, ICOORD>`).
    distances: Vec<(f64, ICoord)>,
    square_length: f64,
}

impl DetLineFit {
    /// `DetLineFit::Clear` (`detlinefit.cpp:44-47`).
    pub fn clear(&mut self) {
        self.pts.clear();
        self.distances.clear();
    }

    /// `DetLineFit::Add(const ICOORD&)` (`detlinefit.cpp:50-52`) — zero
    /// halfwidth.
    pub fn add(&mut self, pt: ICoord) {
        self.pts.push(PointWidth { pt, halfwidth: 0 });
    }

    /// `DetLineFit::Add(const ICOORD&, int)` (`detlinefit.cpp:57-59`).
    pub fn add_with_halfwidth(&mut self, pt: ICoord, halfwidth: i32) {
        self.pts.push(PointWidth { pt, halfwidth });
    }

    /// `DetLineFit::Fit(int skip_first, int skip_last, ICOORD*, ICOORD*)`
    /// (`detlinefit.cpp:64-125`). Returns `(pt1, pt2, error)`.
    pub fn fit(&mut self, skip_first: i32, skip_last: i32) -> (ICoord, ICoord, f64) {
        if self.pts.is_empty() {
            return (ICoord::default(), ICoord::default(), 0.0);
        }
        let pt_count = self.pts.len() as i32;
        let skip_first = if skip_first >= pt_count {
            pt_count - 1
        } else {
            skip_first
        };
        let mut starts: Vec<usize> = Vec::new();
        let end_i = (skip_first + K_NUM_END_POINTS).min(pt_count);
        for i in skip_first..end_i {
            starts.push(i as usize);
        }
        let skip_last = if skip_last >= pt_count {
            pt_count - 1
        } else {
            skip_last
        };
        let mut ends: Vec<usize> = Vec::new();
        let end_i2 = (pt_count - K_NUM_END_POINTS - skip_last).max(0);
        let mut i = pt_count - 1 - skip_last;
        while i >= end_i2 {
            ends.push(i as usize);
            i -= 1;
        }
        if pt_count <= 2 {
            let pt1 = self.pts[starts[0]].pt;
            let pt2 = if pt_count > 1 {
                self.pts[ends[0]].pt
            } else {
                pt1
            };
            return (pt1, pt2, 0.0);
        }
        let mut best_uq = -1.0f64;
        let mut result_pt1 = ICoord::default();
        let mut result_pt2 = ICoord::default();
        for &si in &starts {
            let start = self.pts[si].pt;
            for &ei in &ends {
                let end = self.pts[ei].pt;
                if start != end {
                    self.compute_distances(start, end);
                    let dist = self.evaluate_line_fit();
                    if dist < best_uq || best_uq < 0.0 {
                        best_uq = dist;
                        result_pt1 = start;
                        result_pt2 = end;
                    }
                }
            }
        }
        let error = if best_uq > 0.0 {
            best_uq.sqrt()
        } else {
            best_uq
        };
        (result_pt1, result_pt2, error)
    }

    /// `DetLineFit::Fit(float* m, float* c)` (`detlinefit.cpp:175-186`) —
    /// the backwards-compatible gradient/constant fit `fit_lms_line` uses.
    /// Returns `(m, c, error)`.
    pub fn fit_mc(&mut self) -> (f32, f32, f64) {
        let (start, end, error) = self.fit(0, 0);
        if end.x != start.x {
            let m = (end.y - start.y) as f32 / (end.x - start.x) as f32;
            let c = start.y as f32 - m * start.x as f32;
            (m, c, error)
        } else {
            (0.0, 0.0, error)
        }
    }

    /// `DetLineFit::ConstrainedFit(const FCOORD&, double, double, bool,
    /// ICOORD*)` (`detlinefit.cpp:133-164`), `debug` always `false` (the
    /// `tesserr` dump has no oracle-observable effect). Returns
    /// `(line_pt, error)`.
    pub fn constrained_fit(
        &mut self,
        direction: FCoord,
        min_dist: f64,
        max_dist: f64,
    ) -> (ICoord, f64) {
        self.compute_constrained_distances(direction, min_dist, max_dist);
        if self.pts.is_empty() || self.distances.is_empty() {
            return (ICoord::default(), 0.0);
        }
        let median_index = self.distances.len() / 2;
        self.distances
            .select_nth_unstable_by(median_index, |a, b| a.0.partial_cmp(&b.0).unwrap());
        let line_pt = self.distances[median_index].1;
        let dist_origin = f64::from(direction.cross(FCoord::from_icoord(line_pt)));
        for d in &mut self.distances {
            d.0 -= dist_origin;
        }
        (line_pt, self.evaluate_line_fit().sqrt())
    }

    /// `DetLineFit::ConstrainedFit(double m, float* c)`
    /// (`detlinefit.cpp:191-203`). Returns `(c, error)`.
    pub fn constrained_fit_mc(&mut self, m: f64) -> (f32, f64) {
        if self.pts.is_empty() {
            return (0.0, 0.0);
        }
        let cos = 1.0 / (1.0 + m * m).sqrt();
        let direction = FCoord {
            x: cos as f32,
            y: (m * cos) as f32,
        };
        let (line_pt, error) =
            self.constrained_fit(direction, -f64::from(f32::MAX), f64::from(f32::MAX));
        let c = (f64::from(line_pt.y) - f64::from(line_pt.x) * m) as f32;
        (c, error)
    }

    /// `DetLineFit::SufficientPointsForIndependentFit`
    /// (`detlinefit.cpp:168-170`).
    #[must_use]
    pub fn sufficient_points_for_independent_fit(&self) -> bool {
        self.distances.len() >= K_MIN_POINTS_FOR_ERROR_COUNT
    }

    /// `DetLineFit::EvaluateLineFit` (`detlinefit.cpp:206-217`).
    fn evaluate_line_fit(&mut self) -> f64 {
        let mut dist = self.compute_upper_quartile_error();
        if self.distances.len() >= K_MIN_POINTS_FOR_ERROR_COUNT
            && dist > K_MAX_REAL_DISTANCE * K_MAX_REAL_DISTANCE
        {
            let threshold = K_MAX_REAL_DISTANCE * self.square_length.sqrt();
            dist = f64::from(self.number_of_misfitted_points(threshold));
        }
        dist
    }

    /// `DetLineFit::ComputeUpperQuartileError` (`detlinefit.cpp:221-239`).
    /// Mutates `self.distances` in place: takes `abs()` of every key, then
    /// partitions around the `3n/4` order statistic (`std::nth_element`).
    fn compute_upper_quartile_error(&mut self) -> f64 {
        let num_errors = self.distances.len();
        if num_errors == 0 {
            return 0.0;
        }
        for d in &mut self.distances {
            if d.0 < 0.0 {
                d.0 = -d.0;
            }
        }
        let index = 3 * num_errors / 4;
        self.distances
            .select_nth_unstable_by(index, |a, b| a.0.partial_cmp(&b.0).unwrap());
        let dist = self.distances[index].0;
        if self.square_length > 0.0 {
            dist * dist / self.square_length
        } else {
            0.0
        }
    }

    /// `DetLineFit::NumberOfMisfittedPoints` (`detlinefit.cpp:242-252`).
    fn number_of_misfitted_points(&self, threshold: f64) -> i32 {
        self.distances.iter().filter(|d| d.0 > threshold).count() as i32
    }

    /// `DetLineFit::ComputeDistances` (`detlinefit.cpp:258-286`).
    fn compute_distances(&mut self, start: ICoord, end: ICoord) {
        self.distances.clear();
        let line_vector = end.minus(start);
        let sq = line_vector.sqlength();
        self.square_length = f64::from(sq);
        let line_length = int_cast_rounded(self.square_length.sqrt());
        let pts = self.pts.clone();
        let mut prev_abs_dist: i32 = 0;
        let mut prev_dot: i32 = 0;
        for (i, pw) in pts.iter().enumerate() {
            let pt_vector = pw.pt.minus(start);
            let dot = line_vector.dot(pt_vector);
            let dist = line_vector.cross(pt_vector);
            let abs_dist = dist.abs();
            if abs_dist > prev_abs_dist && i > 0 {
                let separation = (dot - prev_dot).abs();
                if separation < line_length * pw.halfwidth
                    || separation < line_length * pts[i - 1].halfwidth
                {
                    continue;
                }
            }
            self.distances.push((f64::from(dist), pw.pt));
            prev_abs_dist = abs_dist;
            prev_dot = dot;
        }
    }

    /// `DetLineFit::ComputeConstrainedDistances` (`detlinefit.cpp:291-304`).
    fn compute_constrained_distances(&mut self, direction: FCoord, min_dist: f64, max_dist: f64) {
        self.distances.clear();
        self.square_length = f64::from(direction.sqlength());
        for pw in self.pts.clone() {
            let pt_vector = FCoord::from_icoord(pw.pt);
            let dist = f64::from(direction.cross(pt_vector));
            if min_dist <= dist && dist <= max_dist {
                self.distances.push((dist, pw.pt));
            }
        }
    }
}

/// `fit_lms_line` (`makerow.cpp:296-307`) — fit an LMS line through the
/// mid-bottom points of a row's blob boxes, given as
/// `(left, bottom, right, top)` tuples. Returns `(m, c, error)`
/// (`row->set_line(m, c, error)`'s arguments).
#[must_use]
pub fn fit_lms_line(boxes: &[(i32, i32, i32, i32)]) -> (f32, f32, f64) {
    let mut lms = DetLineFit::default();
    for &(left, bottom, right, _top) in boxes {
        lms.add(ICoord::new((left + right) / 2, bottom));
    }
    lms.fit_mc()
}

/// `compute_line_occupation` (`makerow.cpp:799-845`) — project every
/// blob's (de-skewed) bounding box onto the y axis, producing the
/// occupation profile and its first derivative (`deltas`). `blobs` is the
/// flattened set of every row's blob boxes (the row grouping in the C++ is
/// only a traversal detail — the math treats every blob identically
/// regardless of row). Returns `(occupation, deltas)`, each of length
/// `max_y - min_y + 1`.
///
/// # Panics
/// Panics (array index out of bounds) if a de-skewed blob's `bottom`/`top`
/// falls outside `[min_y, max_y]`, mirroring the C++ `ASSERT_HOST` (which
/// aborts under the same condition).
#[must_use]
pub fn compute_line_occupation(
    blobs: &[(i32, i32, i32, i32)],
    gradient: f32,
    min_y: i32,
    max_y: i32,
) -> (Vec<i32>, Vec<i32>) {
    let line_count = (max_y - min_y + 1) as usize;
    let length = (gradient * gradient + 1.0).sqrt();
    let (vx, vy) = (1.0 / length, -gradient / length);
    let mut deltas = vec![0i32; line_count];
    for &(left, bottom, right, top) in blobs {
        let bl = ICoord::new(left, bottom).rotate(vx, vy);
        let tr = ICoord::new(right, top).rotate(vx, vy);
        let width = tr.x - bl.x;
        let idx_b = (bl.y - min_y) as usize;
        assert!(
            bl.y >= min_y && bl.y <= max_y,
            "de-skewed bottom out of [min_y,max_y]"
        );
        deltas[idx_b] += width;
        let idx_t = (tr.y - min_y) as usize;
        assert!(
            tr.y >= min_y && tr.y <= max_y,
            "de-skewed top out of [min_y,max_y]"
        );
        deltas[idx_t] -= width;
    }
    let mut occupation = vec![0i32; line_count];
    occupation[0] = deltas[0];
    for i in 1..line_count {
        occupation[i] = occupation[i - 1] + deltas[i];
    }
    (occupation, deltas)
}

/// `compute_occupation_threshold` (`makerow.cpp:852-926`).
/// `occupancy_threshold` is `textord_occupancy_threshold` (C++ default
/// `0.4`), taken as an explicit parameter for testability.
#[must_use]
pub fn compute_occupation_threshold(
    low_window: i32,
    high_window: i32,
    line_count: i32,
    occupation: &[i32],
    occupancy_threshold: f64,
) -> Vec<i32> {
    let mut thresholds = vec![0i32; line_count as usize];
    let divisor = (f64::from(low_window + high_window) / occupancy_threshold).ceil() as i32;
    let sum;
    let min_occ;
    let mut line_index: i32;

    if low_window + high_window < line_count {
        let mut running_sum = 0i32;
        let mut high_index = 0i32;
        while high_index < low_window {
            running_sum += occupation[high_index as usize];
            high_index += 1;
        }
        let mut low_index = 0i32;
        while low_index < high_window {
            running_sum += occupation[high_index as usize];
            low_index += 1;
            high_index += 1;
        }
        let mut running_min_occ = occupation[0];
        let mut min_index = 0i32;
        let mut test_index = 1;
        while test_index < high_index {
            if occupation[test_index as usize] <= running_min_occ {
                running_min_occ = occupation[test_index as usize];
                min_index = test_index;
            }
            test_index += 1;
        }
        line_index = 0;
        while line_index < low_window {
            thresholds[line_index as usize] =
                (running_sum - running_min_occ) / divisor + running_min_occ;
            line_index += 1;
        }
        low_index = 0;
        while high_index < line_count {
            running_sum -= occupation[low_index as usize];
            running_sum += occupation[high_index as usize];
            if occupation[high_index as usize] <= running_min_occ {
                running_min_occ = occupation[high_index as usize];
                min_index = high_index;
            }
            if min_index <= low_index {
                running_min_occ = occupation[(low_index + 1) as usize];
                min_index = low_index + 1;
                let mut test_index2 = low_index + 2;
                while test_index2 <= high_index {
                    if occupation[test_index2 as usize] <= running_min_occ {
                        running_min_occ = occupation[test_index2 as usize];
                        min_index = test_index2;
                    }
                    test_index2 += 1;
                }
            }
            thresholds[line_index as usize] =
                (running_sum - running_min_occ) / divisor + running_min_occ;
            line_index += 1;
            low_index += 1;
            high_index += 1;
        }
        sum = running_sum;
        min_occ = running_min_occ;
    } else {
        // The reference `min_index` computed in this branch is write-only
        // (never read: the tail loop below uses only `sum`/`min_occ`), so
        // it is not tracked here — an equivalent-behaviour simplification,
        // not a behaviour change.
        let mut running_sum = 0i32;
        let mut running_min_occ = occupation[0];
        let mut low_index = 0i32;
        while low_index < line_count {
            if occupation[low_index as usize] < running_min_occ {
                running_min_occ = occupation[low_index as usize];
            }
            running_sum += occupation[low_index as usize];
            low_index += 1;
        }
        line_index = 0;
        sum = running_sum;
        min_occ = running_min_occ;
    }

    while line_index < line_count {
        thresholds[line_index as usize] = (sum - min_occ) / divisor + min_occ;
        line_index += 1;
    }
    thresholds
}

/// `compute_dropout_distances` (`makerow.cpp:933-967`) — in-place distance-
/// to-nearest-dropout transform. `thresholds` is both input and output, as
/// in the C++ signature.
pub fn compute_dropout_distances(occupation: &[i32], thresholds: &mut [i32], line_count: i32) {
    let mut distance = -line_count;
    let mut line_index: i32 = 0;
    loop {
        let mut prev_threshold;
        loop {
            distance -= 1;
            prev_threshold = thresholds[line_index as usize];
            thresholds[line_index as usize] = distance;
            line_index += 1;
            let cont = line_index < line_count
                && (occupation[line_index as usize] < thresholds[line_index as usize]
                    || occupation[(line_index - 1) as usize] >= prev_threshold);
            if !cont {
                break;
            }
        }
        if line_index < line_count {
            let mut back_index = line_index - 1;
            let mut next_dist = 1;
            while next_dist < -distance && back_index >= 0 {
                thresholds[back_index as usize] = next_dist;
                back_index -= 1;
                next_dist += 1;
                distance += 1;
            }
            distance = 1;
        }
        if line_index >= line_count {
            break;
        }
    }
}

/// `compute_height_modes` (`makerow.cpp:1629-1682`) — find (at most)
/// `maxmodes` of the largest piles in `heights` over `[min_height,
/// max_height]`, in the order they occurred. Returns the found modes
/// (length `<= maxmodes`).
#[must_use]
pub fn compute_height_modes(
    heights: &Stats,
    min_height: i32,
    max_height: i32,
    maxmodes: i32,
) -> Vec<i32> {
    let mut modes = vec![0i32; maxmodes as usize];
    let src_count = max_height + 1 - min_height;
    let mut dest_count: i32 = 0;
    let mut least_count = i32::MAX;
    let mut least_index: i32 = -1;
    for src_index in 0..src_count {
        let pile_count = heights.pile_count(min_height + src_index);
        if pile_count <= 0 {
            continue;
        }
        if dest_count < maxmodes {
            if pile_count < least_count {
                least_count = pile_count;
                least_index = dest_count;
            }
            modes[dest_count as usize] = min_height + src_index;
            dest_count += 1;
        } else if pile_count >= least_count {
            while least_index < maxmodes - 1 {
                modes[least_index as usize] = modes[(least_index + 1) as usize];
                least_index += 1;
            }
            modes[(maxmodes - 1) as usize] = min_height + src_index;
            if pile_count == least_count {
                least_index = maxmodes - 1;
            } else {
                least_count = heights.pile_count(modes[0]);
                least_index = 0;
                let mut scan = 1;
                while scan < maxmodes {
                    let pc = heights.pile_count(modes[scan as usize]);
                    if pc < least_count {
                        least_count = pc;
                        least_index = scan;
                    }
                    scan += 1;
                }
                dest_count = scan;
            }
        }
    }
    modes.truncate(dest_count as usize);
    modes
}

/// `fill_heights` (`makerow.cpp:1418-1462`) — the non-baseline-spline
/// variant; see module doc. `boxes` are `(left, bottom, right, top)`
/// tuples for every non-`joined_to_prev` blob in the row (the caller is
/// responsible for that filtering + any repeated-char-run skip, which are
/// out of scope for this plain-tuple port). `min_blob_height_fraction` is
/// `textord_min_blob_height_fraction` (C++ default `0.75`), taken as an
/// explicit parameter. Returns `(heights, floating_heights)`.
#[must_use]
pub fn fill_heights(
    boxes: &[(i32, i32, i32, i32)],
    gradient: f32,
    parallel_c: f32,
    min_height: i32,
    max_height: i32,
    min_blob_height_fraction: f32,
) -> (Stats, Stats) {
    let mut heights = Stats::new(min_height, max_height);
    let mut floating_heights = Stats::new(min_height, max_height);
    for &(left, bottom, right, top) in boxes {
        let xcentre = (left + right) as f32 / 2.0;
        let height = (top - bottom) as f32;
        let top_adj = top as f32 - (gradient * xcentre + parallel_c);
        if top_adj >= min_height as f32 && top_adj <= max_height as f32 {
            let bucket = (top_adj + 0.5).floor() as i32;
            heights.add(bucket, 1);
            if height / top_adj < min_blob_height_fraction {
                floating_heights.add(bucket, 1);
            }
        }
    }
    (heights, floating_heights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_line_fit_exact_on_collinear_points() {
        let mut lms = DetLineFit::default();
        for i in 0..6 {
            lms.add(ICoord::new(i, 2 * i + 1));
        }
        let (m, c, error) = lms.fit_mc();
        assert!((m - 2.0).abs() < 1e-6);
        assert!((c - 1.0).abs() < 1e-6);
        assert_eq!(error, 0.0);
    }

    #[test]
    fn det_line_fit_two_points() {
        let mut lms = DetLineFit::default();
        lms.add(ICoord::new(0, 0));
        lms.add(ICoord::new(4, 8));
        let (m, c, error) = lms.fit_mc();
        assert!((m - 2.0).abs() < 1e-6);
        assert!((c - 0.0).abs() < 1e-6);
        assert_eq!(error, 0.0);
    }

    #[test]
    fn compute_occupation_threshold_hand_grid() {
        // Uniform occupation of 10 everywhere: thresholds should end up
        // constant too (min_occ==sum/count case).
        let occupation = vec![10i32; 8];
        let thresholds = compute_occupation_threshold(2, 2, 8, &occupation, 0.4);
        assert_eq!(thresholds.len(), 8);
        // All windows see the same 4*10=40 sum, min_occ=10, divisor=ceil(4/0.4)=10.
        for t in &thresholds {
            assert_eq!(*t, (40 - 10) / 10 + 10);
        }
    }

    #[test]
    fn compute_height_modes_bimodal() {
        let mut heights = Stats::new(0, 30);
        for _ in 0..5 {
            heights.add(10, 1);
        }
        for _ in 0..3 {
            heights.add(20, 1);
        }
        let modes = compute_height_modes(&heights, 0, 30, 4);
        assert!(modes.contains(&10));
        assert!(modes.contains(&20));
    }

    #[test]
    fn fit_lms_line_on_boxes() {
        // (left,bottom,right,top) boxes whose (mid-x, bottom) is exactly
        // collinear: y = x.
        let boxes = vec![
            (0, 0, 0, 5),
            (2, 2, 2, 6),
            (4, 4, 4, 7),
            (6, 6, 6, 8),
            (8, 8, 8, 9),
        ];
        let (m, c, error) = fit_lms_line(&boxes);
        assert!((m - 1.0).abs() < 1e-6);
        assert!((c - 0.0).abs() < 1e-6);
        assert_eq!(error, 0.0);
    }
}
