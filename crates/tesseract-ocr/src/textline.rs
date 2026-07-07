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

// ============================================================================
// Batch 3E, wave 2 -- the row-assignment + cleanup chain (`makerow.cpp`,
// single-column case). Structure per
// `.claude/harvest/makerow-callgraph.txt`. Builds on wave 1 above
// ([`fit_lms_line`], the occupation leaves, [`fill_heights`],
// [`compute_height_modes`], [`Stats`]) -- reused directly, not duplicated.
//
// Ported leaves (see per-function docs for exact `makerow.cpp` line
// ranges):
// - [`ToRow`] / [`ToBlockCtx`] -- minimal `TO_ROW` / `TO_BLOCK` carriers
//   (`ccstruct/blobbox.h:555-806`), plain-tuple blobs like wave 1.
// - [`assign_blobs_to_rows`] + [`most_overlapping_row`] -- the row
//   assignment core.
// - [`fit_parallel_rows`] + [`fit_parallel_lms`].
// - [`delete_non_dropout_rows`] + [`deskew_block_coords`] +
//   [`find_best_dropout_row`] -- wires the three wave-1 occupation leaves
//   into the real prune decision.
// - [`expand_rows`] + [`adjust_row_limits`] + [`compute_row_stats`].
// - [`make_initial_textrows`] + [`make_rows`] + [`compute_page_skew`] +
//   [`cleanup_rows_making`] -- the top-level orchestration.
//
// ## Carrier simplification (read before extending)
// The real `TO_BLOCK` splits blobs across FIVE lists: `blobs` (medium),
// `underlines`, `noise_blobs`, `small_blobs`, `large_blobs`. This wave's
// [`ToBlockCtx`] has a single flat pool (`blobs`) standing in for all five
// -- every blob in this wave's fixtures is an ordinary (non-noise,
// non-underline, non-oversized) blob, so the other four lists are always
// empty in the real algorithm's terms. Consequently `cleanup_rows_making`'s
// three `assign_blobs_to_rows` passes -- which in the real C++
// progressively pour `large_blobs` (before pass 2) then
// `noise_blobs`+`small_blobs` (before pass 3) into the pool -- degenerate
// here to three passes over the SAME pool with only the `reject_misses`/
// `make_new_rows` flags differing. This is still behaviourally faithful
// for the "single ordinary blob category" domain these fixtures live in;
// it is not a general-purpose port of the five-list classification (that
// classification -- `ReSetAndReFilterBlobs` and friends -- is out of
// scope for this wave).
//
// Every real blob is also implicitly `!joined_to_prev()` (no outline
// chopping / repeated-char-run state exists in this carrier, mirroring
// wave 1's [`fill_heights`] scoping note), so C++ filters gated on that
// flag (`fit_parallel_lms`, `fill_heights`) are unconditional here.
//
// ## Pinned flags with the dead branch dropped entirely
// A few `textord_*` bools are permanently pinned to their C++ default and,
// rather than carrying a runtime `if` whose other arm can never execute,
// this port -- mirroring wave 1's precedent for `textord_fix_xheight_bug`
// -- only implements the LIVE branch:
// - `textord_straight_baselines` (default `false`): [`fit_parallel_lms`]
//   never re-fits via the unconstrained `DetLineFit::Fit` path; the
//   `textord_lms_line_trials` threshold that gates it is therefore never
//   read and is not declared as a constant here.
// - `textord_new_initial_xheight` (default `true`): [`expand_rows`] always
//   calls [`compute_row_stats`] before the first `assign_blobs_to_rows`
//   pass and never re-calls it after `fit_parallel_rows` (the
//   `!textord_new_initial_xheight` arm).
// - `textord_test_landscape` (default `false`): [`make_rows`] /
//   [`make_initial_textrows`] always use the `FCOORD(1,0)` "no rotation"
//   convention; there is no `FCOORD` parameter to override it in this
//   carrier.
// - `textord_debug_blob` / `textord_debug_xheights` / the
//   `GRAPHICS_DISABLED`-gated `textord_show_*` family: all `tprintf`/
//   `ScrollView` debug output, always inert for a headless port (wave 1
//   precedent). `textord_test_x`/`textord_test_y` (the debug test-point
//   hook) are consequently also unused and not declared.
//
// ## Float vs double promotion (audited per call site -- see inline docs)
// Tesseract mixes `float`-typed row/block members with `double`-typed
// named params (`textord_*`) and `double`-typed `CCStruct::k*Fraction`
// constants throughout this chain. Per C++ arithmetic conversion rules, a
// `float OP double` expression is evaluated in `double` and narrows back
// to `float` only when finally stored into a `float` lvalue (or an
// intermediate C++ `float`-typed local). This port replicates that
// promote-then-narrow behaviour explicitly (`as f64` .. `as f32`) at every
// such site rather than doing the whole computation in one precision --
// see [`adjust_row_limits`], [`expand_rows`], [`compute_row_stats`],
// [`delete_non_dropout_rows`], [`assign_blobs_to_rows`] docs for the
// specific expressions audited. `textord_overlap_x` in particular is a
// `double` constant even though every use site multiplies it against a
// `float` row/block field -- so it is declared `f64` here, not `f32`.

/// `textord_min_blobs_in_row` (`INT_VAR`, default `4`) --
/// [`compute_page_skew`]'s non-biased sample-count gate.
const TEXTORD_MIN_BLOBS_IN_ROW: i32 = 4;
/// `textord_skewsmooth_offset` (`static INT_VAR`, default `4`); the one use
/// site (`makerow.cpp:2396`) is inside a `double` expression, so it is
/// represented directly as `f64` (exact: 4 is losslessly representable).
const TEXTORD_SKEWSMOOTH_OFFSET: f64 = 4.0;
/// `textord_skewsmooth_offset2` (`static INT_VAR`, default `1`); see
/// [`TEXTORD_SKEWSMOOTH_OFFSET`].
const TEXTORD_SKEWSMOOTH_OFFSET2: f64 = 1.0;
/// `textord_skew_lag` (`double_VAR`, default `0.02`).
const TEXTORD_SKEW_LAG: f64 = 0.02;
/// `textord_skew_ile` (`double_VAR`, default `0.5`).
const TEXTORD_SKEW_ILE: f64 = 0.5;
/// `textord_biased_skewcalc` (`static BOOL_VAR`, default `true`).
const TEXTORD_BIASED_SKEWCALC: bool = true;
/// `textord_interpolating_skew` (`static BOOL_VAR`, default `true`).
const TEXTORD_INTERPOLATING_SKEW: bool = true;
/// `textord_overlap_x` (`static double_VAR`, default `0.375`) -- every use
/// site (`makerow.cpp:2374-2375,2529-2530`) multiplies it against a
/// `double`-promoted expression, so it is `f64` here, not `f32`.
const TEXTORD_OVERLAP_X: f64 = 0.375;
/// `textord_fix_makerow_bug` (`BOOL_VAR`, default `true`).
const TEXTORD_FIX_MAKEROW_BUG: bool = true;
/// `textord_expansion_factor` (`static double_VAR`, default `1.0`).
const TEXTORD_EXPANSION_FACTOR: f64 = 1.0;
/// `textord_linespace_iqrlimit` (`double_VAR`, default `0.2`).
const TEXTORD_LINESPACE_IQRLIMIT: f64 = 0.2;
/// `textord_min_xheight` (`INT_VAR`, default `10`) -- the one use site in
/// this wave's scope ([`compute_row_stats`], `makerow.cpp:1235-1237`)
/// compares/assigns it against a lone `float` with no `double` operand
/// present, so the C++ promotion target is `float` (exact: 10 is losslessly
/// representable), represented here as `f32`.
const TEXTORD_MIN_XHEIGHT: f32 = 10.0;
/// `textord_excess_blobsize` (`double_VAR`, default `1.3`).
const TEXTORD_EXCESS_BLOBSIZE: f64 = 1.3;
/// `textord_occupancy_threshold` (`double_VAR`, default `0.4`) -- reused as
/// wave 1's [`compute_occupation_threshold`] `occupancy_threshold` param.
const TEXTORD_OCCUPANCY_THRESHOLD: f64 = 0.4;

/// `tesseract::CCStruct::kDescenderFraction` (`ccstruct.cpp:25`, `= 0.25`).
const K_DESCENDER_FRACTION: f64 = 0.25;
/// `tesseract::CCStruct::kXHeightFraction` (`ccstruct.cpp:26`, `= 0.5`).
const K_XHEIGHT_FRACTION: f64 = 0.5;
/// `tesseract::CCStruct::kAscenderFraction` (`ccstruct.cpp:27`, `= 0.25`).
const K_ASCENDER_FRACTION: f64 = 0.25;

/// `TO_ROW::kErrorWeight` (`blobbox.h:557`, `= 3`).
const TO_ROW_K_ERROR_WEIGHT: f32 = 3.0;

/// `OVERLAP_STATE` (`makerow.h:30-34`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapState {
    /// Assign the blob to the identified row.
    Assign,
    /// Reject the blob -- dual overlap / no acceptable row.
    Reject,
    /// The blob needs a new row.
    NewRow,
}

/// A single text row under construction (`tesseract::TO_ROW`,
/// `ccstruct/blobbox.h:555-695`; scalar mutators from `blobbox.cpp:690-765`).
/// Carries only the fields the wave-2 functions in scope read or write.
/// `blobs` stores plain `(left,bottom,right,top)` box tuples in insertion
/// order, mirroring the C++ `BLOBNBOX_LIST` -- see the module doc's
/// "carrier simplification" note for what this drops.
#[derive(Debug, Clone, Default)]
pub struct ToRow {
    /// Blobs assigned to this row, in insertion (`BLOBNBOX_LIST`) order.
    pub blobs: Vec<(i32, i32, i32, i32)>,
    y_min: f32,
    y_max: f32,
    initial_y_min: f32,
    m: f32,
    c: f32,
    error: f32,
    para_c: f32,
    para_error: f32,
    y_origin: f32,
    credibility: f32,
    /// Spacing to the "next" (upward) row (`compute_row_stats`).
    pub spacing: f32,
    /// `true` when dead (unused by this wave's chain; carried for field
    /// parity with the real `TO_ROW`).
    pub merged: bool,
    /// `TO_ROW::xheight` — x-height estimate (wave 3, `blobbox.h`).
    pub xheight: f32,
    /// `TO_ROW::ascrise` — ascender rise above the x-height.
    pub ascrise: f32,
    /// `TO_ROW::descdrop` — descender drop below the baseline (`<= 0`).
    pub descdrop: f32,
    /// `TO_ROW::xheight_evidence` — mode count backing the x-height.
    pub xheight_evidence: i32,
    /// `TO_ROW::all_caps` — set by [`correct_row_xheight`].
    pub all_caps: bool,
    /// `TO_ROW::rep_chars_marked` — repeated-char run marking done.
    pub rep_chars_marked: bool,
}

impl ToRow {
    /// `TO_ROW::TO_ROW(BLOBNBOX*, float top, float bottom, float row_size)`
    /// (`blobbox.cpp:690-716`).
    #[must_use]
    pub fn new(blob: (i32, i32, i32, i32), top: f32, bottom: f32, row_size: f32) -> Self {
        let mut row = ToRow {
            y_min: bottom,
            y_max: top,
            initial_y_min: bottom,
            ..Default::default()
        };
        row.blobs.push(blob);
        let diff = top - bottom - row_size;
        if diff > 0.0 {
            row.y_max -= diff / 2.0;
            row.y_min += diff / 2.0;
        } else if (top - bottom) * 3.0 < row_size {
            let diff = row_size / 3.0 + bottom - top;
            row.y_max += diff / 2.0;
            row.y_min -= diff / 2.0;
        }
        row
    }

    /// `TO_ROW::max_y` (`blobbox.h:568-570`).
    #[must_use]
    pub fn max_y(&self) -> f32 {
        self.y_max
    }
    /// `TO_ROW::min_y` (`blobbox.h:571-573`).
    #[must_use]
    pub fn min_y(&self) -> f32 {
        self.y_min
    }
    /// `TO_ROW::initial_min_y` (`blobbox.h:577-579`).
    #[must_use]
    pub fn initial_min_y(&self) -> f32 {
        self.initial_y_min
    }
    /// `TO_ROW::line_m` (`blobbox.h:580-582`).
    #[must_use]
    pub fn line_m(&self) -> f32 {
        self.m
    }
    /// `TO_ROW::line_c` (`blobbox.h:583-585`).
    #[must_use]
    pub fn line_c(&self) -> f32 {
        self.c
    }
    /// `TO_ROW::line_error` (`blobbox.h:586-588`).
    #[must_use]
    pub fn line_error(&self) -> f32 {
        self.error
    }
    /// `TO_ROW::parallel_c` (`blobbox.h:589-591`).
    #[must_use]
    pub fn parallel_c(&self) -> f32 {
        self.para_c
    }
    /// `TO_ROW::believability` (`blobbox.h:595-597`).
    #[must_use]
    pub fn believability(&self) -> f32 {
        self.credibility
    }
    /// `TO_ROW::intercept` (`blobbox.h:598-600`) -- the real parallel `c`,
    /// rotated onto the y axis (`set_parallel_line`).
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.y_origin
    }

    /// `TO_ROW::add_blob` (`blobbox.cpp:734-765`).
    pub fn add_blob(&mut self, blob: (i32, i32, i32, i32), top: f32, bottom: f32, row_size: f32) {
        self.blobs.push(blob);
        let allowed = row_size + self.y_min - self.y_max;
        if allowed > 0.0 {
            let mut available = if top > self.y_max {
                top - self.y_max
            } else {
                0.0
            };
            if bottom < self.y_min {
                available += self.y_min - bottom;
            }
            if available > 0.0 {
                available += available; // do it gradually
                if available < allowed {
                    available = allowed;
                }
                if bottom < self.y_min {
                    self.y_min -= (self.y_min - bottom) * allowed / available;
                }
                if top > self.y_max {
                    self.y_max += (top - self.y_max) * allowed / available;
                }
            }
        }
    }

    /// `TO_ROW::set_line` (`blobbox.h:612-618`). `new_error` is `double` in
    /// the caller (every C++ call site passes a `DetLineFit::Fit`/
    /// `ConstrainedFit` result), narrowed to `float` on assignment here,
    /// matching `set_line`'s `float new_error` parameter.
    pub fn set_line(&mut self, new_m: f32, new_c: f32, new_error: f64) {
        self.m = new_m;
        self.c = new_c;
        self.error = new_error as f32;
    }

    /// `TO_ROW::set_parallel_line` (`blobbox.h:619-627`).
    pub fn set_parallel_line(&mut self, gradient: f32, new_c: f32, new_error: f64) {
        self.para_c = new_c;
        self.para_error = new_error as f32;
        self.credibility = self.blobs.len() as f32 - TO_ROW_K_ERROR_WEIGHT * (new_error as f32);
        self.y_origin = new_c / (1.0 + gradient * gradient).sqrt();
    }

    /// `TO_ROW::set_limits` (`blobbox.h:628-633`).
    pub fn set_limits(&mut self, new_min: f32, new_max: f32) {
        self.y_min = new_min;
        self.y_max = new_max;
    }
}

/// Minimal `TO_BLOCK` carrier (`ccstruct/blobbox.h:698-806`). See the
/// module doc's "carrier simplification" note: the real five blob-list
/// categories collapse to a single flat pool here.
#[derive(Debug, Clone, Default)]
pub struct ToBlockCtx {
    /// The unassigned blob pool (stands in for `blobs`/`underlines`/
    /// `noise_blobs`/`small_blobs`/`large_blobs` combined -- see module
    /// doc).
    pub blobs: Vec<(i32, i32, i32, i32)>,
    /// Rows, ALWAYS maintained in descending-`min_y()` order (topmost row
    /// first) by every function below -- the `TO_ROW_LIST` invariant
    /// [`assign_blobs_to_rows`]'s trailing bubble-sort step maintains.
    pub rows: Vec<ToRow>,
    /// `block->block->pdblk.bounding_box().left()` -- only consulted as the
    /// [`assign_blobs_to_rows`] fallback `left_x` when the blob pool is
    /// empty at call time (every other use site in the real C++ is
    /// `#ifndef GRAPHICS_DISABLED`-gated debug drawing, dropped here).
    pub block_left: i32,
    /// `TO_BLOCK::line_spacing`.
    pub line_spacing: f32,
    /// `TO_BLOCK::line_size`.
    pub line_size: f32,
    /// `TO_BLOCK::max_blob_size`.
    pub max_blob_size: f32,
    /// `TO_BLOCK::baseline_offset`.
    pub baseline_offset: f32,
    /// `TO_BLOCK::key_row` -- index into `rows`, set by
    /// [`compute_row_stats`].
    pub key_row: Option<usize>,
    /// `TO_BLOCK::xheight` -- block x-height, set by [`compute_block_xheight`].
    pub xheight: f32,
}

/// `ELIST2_ITERATOR::data_relative` circular-index helper: `idx + offset`,
/// wrapping around a list of length `len` (`ccutil/elst2.h`). Used
/// wherever the real C++ walks a `TO_ROW_IT` by a relative offset without
/// moving the iterator's own position.
fn data_relative(len: usize, idx: usize, offset: isize) -> usize {
    debug_assert!(len > 0);
    (idx as isize + offset).rem_euclid(len as isize) as usize
}

/// `TBOX::major_x_overlap` (`ccstruct/rect.h:419-428`) -- do two boxes
/// overlap by more than half the width of the narrower box, on x. `this`
/// (the receiver in C++) is `lhs`; the argument is `rhs`.
fn major_x_overlap(lhs: (i32, i32, i32, i32), rhs: (i32, i32, i32, i32)) -> bool {
    let (lhs_left, _, lhs_right, _) = lhs;
    let (rhs_left, _, rhs_right, _) = rhs;
    let lhs_width = lhs_right - lhs_left;
    let rhs_width = rhs_right - rhs_left;
    let mut overlap = rhs_width;
    if lhs_left > rhs_left {
        overlap -= lhs_left - rhs_left;
    }
    if lhs_right < rhs_right {
        overlap -= rhs_right - lhs_right;
    }
    overlap >= rhs_width / 2 || overlap >= lhs_width / 2
}

/// `blob_x_order` (`makerow.cpp:2538-2557`) as a sort key: ascending by
/// `left()`.
/// PARITY PIN: real Tesseract's `blob_x_order` compares `left` only and the
/// ELIST sort's tie order is UNSPECIFIED (qsort semantics). Both probe
/// shells pin the same TOTAL order (the full box tuple) so equal-left blobs
/// arrive in the same sequence -- `add_blob` expansion is order-dependent.
fn blob_x_order_total_key(b: &(i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    *b
}

/// `deskew_block_coords` (`makerow.cpp:765-791`) -- bounding box of every
/// row's blobs after de-skewing by `gradient`, without mutating them.
/// Reproduces `TBOX::rotate` (`rect.h:210-214`: rotate both corners via the
/// real `ICOORD::rotate`, then re-derive `(bot_left,top_right)` via the
/// `TBOX(ICOORD,ICOORD)` 2-corner constructor's per-axis min/max,
/// `rect.cpp:35-56`) unioned via `TBOX::operator+=` (`rect.cpp:214-234`,
/// component-wise min/max), seeded from the `TBOX()` null-box default
/// (`bot_left=(INT16_MAX,INT16_MAX)`, `top_right=(-INT16_MAX,-INT16_MAX)`,
/// `rect.h:39-42`). Returns `(left, bottom, right, top)`.
fn deskew_block_coords(block: &ToBlockCtx, gradient: f32) -> (i32, i32, i32, i32) {
    let length = (gradient * gradient + 1.0).sqrt();
    let (vx, vy) = (1.0 / length, -gradient / length);
    const NULL_MAX: i32 = i16::MAX as i32;
    let mut result = (NULL_MAX, NULL_MAX, -NULL_MAX, -NULL_MAX);
    for row in &block.rows {
        for &(left, bottom, right, top) in &row.blobs {
            let bl = ICoord::new(left, bottom).rotate(vx, vy);
            let tr = ICoord::new(right, top).rotate(vx, vy);
            let (rl, rb) = (bl.x.min(tr.x), bl.y.min(tr.y));
            let (rr, rt) = (bl.x.max(tr.x), bl.y.max(tr.y));
            result.0 = result.0.min(rl);
            result.1 = result.1.min(rb);
            result.2 = result.2.max(rr);
            result.3 = result.3.max(rt);
        }
    }
    result
}

/// `most_overlapping_row` (`makerow.cpp:2451-2535`). `rows[start_idx]` is
/// the initial candidate row (the caller has already established it
/// overlaps `[bottom,top]`). May merge `rows[start_idx]` (or whichever row
/// is currently tracked as best) into an adjacent overlapping row,
/// deleting one of the two.
///
/// **Adjacent-merge scope.** The C++ tracks the current scan cursor
/// (`row_it`, here `cursor`) and the best-scoring row seen so far (`row`,
/// here `row_idx`) as SEPARATE variables; a merge always operates between
/// `rows[row_idx]` and `rows[cursor]` (`makerow.cpp:2487-2503`) and then
/// deletes whatever sits at `cursor - 1` (`row_it->backward();
/// delete row_it->extract();`) -- which is `rows[row_idx]` exactly when
/// `row_idx == cursor - 1`. That invariant holds whenever `row_idx` was
/// updated on the immediately-preceding iteration (the common case: a
/// blob overlapping at most two candidate rows). In a 3+-row overlap chain
/// where an intermediate candidate scores lower than an earlier one,
/// `row_idx` can lag behind `cursor` by more than one step, and the C++
/// deletes "whatever is at `cursor - 1`" rather than `rows[row_idx]`
/// specifically -- this port replicates that literal pointer/index
/// arithmetic (not a "delete row_idx" special case), so it is faithful to
/// the real behaviour in that edge case too, but this wave's fixtures do
/// not exercise chains longer than two candidates (the scenario is
/// difficult to hand-verify and not needed for the single-column domain).
///
/// Returns `(state, winning_row_index)`.
fn most_overlapping_row(
    rows: &mut Vec<ToRow>,
    start_idx: usize,
    top: f32,
    bottom: f32,
    rowsize: f32,
) -> (OverlapState, usize) {
    let mut result = OverlapState::Assign;
    let mut row_idx = start_idx;
    let mut bestover = top - bottom;
    if top > rows[row_idx].max_y() {
        bestover -= top - rows[row_idx].max_y();
    }
    if bottom < rows[row_idx].min_y() {
        bestover -= rows[row_idx].min_y() - bottom;
    }

    let mut cursor = start_idx;
    loop {
        if cursor + 1 >= rows.len() {
            break; // !row_it->at_last() was false: the do-while body's
                   // `if (!at_last())` guard skips straight to the
                   // while-condition, which then also reads !at_last() ->
                   // false -> loop ends.
        }
        cursor += 1;
        if rows[cursor].min_y() <= top && rows[cursor].max_y() >= bottom {
            let merge_top = rows[cursor].max_y().max(rows[row_idx].max_y());
            let merge_bottom = rows[cursor].min_y().min(rows[row_idx].min_y());
            if merge_top - merge_bottom <= rowsize {
                rows[cursor].set_limits(merge_bottom, merge_top);
                let mut moved = std::mem::take(&mut rows[row_idx].blobs);
                rows[cursor].blobs.append(&mut moved);
                rows[cursor].blobs.sort_by_key(blob_x_order_total_key);
                let victim = cursor - 1; // row_it->backward()'s target
                rows.remove(victim);
                cursor -= 1; // row_it->forward() after the extract
                bestover = -1.0; // force replacement
            }
            let mut overlap = top - bottom;
            if top > rows[cursor].max_y() {
                overlap -= top - rows[cursor].max_y();
            }
            if bottom < rows[cursor].min_y() {
                overlap -= rows[cursor].min_y() - bottom;
            }
            if bestover >= rowsize - 1.0 && overlap >= rowsize - 1.0 {
                result = OverlapState::Reject;
            }
            if overlap > bestover {
                bestover = overlap;
                row_idx = cursor;
            }
        } else {
            break;
        }
    }
    while cursor != row_idx {
        cursor -= 1;
    }
    if f64::from(top - bottom - bestover) > f64::from(rowsize) * TEXTORD_OVERLAP_X
        && (!TEXTORD_FIX_MAKEROW_BUG
            || f64::from(bestover) < f64::from(rowsize) * TEXTORD_OVERLAP_X)
        && result == OverlapState::Assign
    {
        result = OverlapState::NewRow;
    }
    (result, row_idx)
}

/// `assign_blobs_to_rows` (`makerow.cpp:2272-2444`). `gradient` is the C++
/// `float *gradient` (nullable, `None` only from
/// [`make_initial_textrows`]'s pass 0; never mutated through the pointer,
/// only read). `pass` and `drawing_skew` are dropped: both are debug/
/// `ScrollView`-only in the real signature (`pass` feeds one
/// `textord_debug_blob`-gated `tprintf`; `drawing_skew` gates
/// `#ifndef GRAPHICS_DISABLED` cursor draws) with zero effect on returned
/// state, mirroring wave 1's precedent for dropping debug-only params.
pub fn assign_blobs_to_rows(
    block: &mut ToBlockCtx,
    gradient: Option<f32>,
    reject_misses: bool,
    make_new_rows: bool,
) {
    let g_length = match gradient {
        Some(g) => (1.0 + g * g).sqrt(),
        None => 1.0,
    };

    block.blobs.sort_by_key(blob_x_order_total_key);

    let mut smooth_factor: f32 = 1.0;
    let mut block_skew: f32 = 0.0;
    let mut row_count: i32 = block.rows.len() as i32;

    let left_x = block.blobs.first().map_or(block.block_left, |&(l, ..)| l);
    let mut last_x = left_x;

    let mut i = 0usize;
    while i < block.blobs.len() {
        let (left, bottom_raw, right, top_raw) = block.blobs[i];
        if let Some(g) = gradient {
            block_skew = (1.0 - 1.0 / g_length) * bottom_raw as f32 + g / g_length * left as f32;
        } else if (left - last_x) as f32 > block.line_size / 2.0
            && (last_x - left_x) as f32 > block.line_size * 2.0
            && TEXTORD_INTERPOLATING_SKEW
        {
            block_skew *= (left - left_x) as f32 / (last_x - left_x) as f32;
        }
        last_x = left;
        let top = top_raw as f32 - block_skew;
        let bottom = bottom_raw as f32 - block_skew;

        let mut overlap_result;
        let mut dest_idx: usize = 0;

        if !block.rows.is_empty() {
            let mut r = 0usize;
            while r + 1 < block.rows.len() && block.rows[r].min_y() > top {
                r += 1;
            }
            let mut cursor_pos = r;
            if block.rows[r].min_y() <= top && block.rows[r].max_y() >= bottom {
                let (state, winner) =
                    most_overlapping_row(&mut block.rows, r, top, bottom, block.line_size);
                cursor_pos = winner;
                dest_idx = winner;
                overlap_result = state;
                if overlap_result == OverlapState::NewRow && !reject_misses {
                    overlap_result = OverlapState::Assign;
                }
            } else {
                overlap_result = OverlapState::NewRow;
                dest_idx = r;
                if !make_new_rows {
                    let prev_idx = data_relative(block.rows.len(), r, -1);
                    let near_dist = block.rows[prev_idx].min_y() - top;
                    if bottom < block.rows[r].min_y() {
                        if f64::from(block.rows[r].min_y() - bottom)
                            <= f64::from(block.line_spacing - block.line_size)
                                * K_DESCENDER_FRACTION
                        {
                            overlap_result = OverlapState::Assign;
                            dest_idx = r;
                        }
                    } else if near_dist > 0.0 && near_dist < bottom - block.rows[r].max_y() {
                        dest_idx = prev_idx;
                        if f64::from(block.rows[dest_idx].min_y() - bottom)
                            <= f64::from(block.line_spacing - block.line_size)
                                * K_DESCENDER_FRACTION
                        {
                            overlap_result = OverlapState::Assign;
                        }
                    } else if f64::from(top - block.rows[r].max_y())
                        <= f64::from(block.line_spacing - block.line_size)
                            * (TEXTORD_OVERLAP_X + K_ASCENDER_FRACTION)
                    {
                        overlap_result = OverlapState::Assign;
                        dest_idx = r;
                    }
                }
            }

            match overlap_result {
                OverlapState::Assign => {
                    let blob = block.blobs.remove(i);
                    block.rows[dest_idx].add_blob(blob, top, bottom, block.line_size);
                }
                OverlapState::NewRow => {
                    if make_new_rows && top - bottom < block.max_blob_size {
                        let blob = block.blobs.remove(i);
                        row_count += 1;
                        let new_row = ToRow::new(blob, top, bottom, block.line_size);
                        let insert_at = if bottom > block.rows[cursor_pos].min_y() {
                            cursor_pos
                        } else {
                            cursor_pos + 1
                        };
                        block.rows.insert(insert_at, new_row);
                        dest_idx = insert_at;
                        smooth_factor = (1.0
                            / (f64::from(row_count) * TEXTORD_SKEW_LAG + TEXTORD_SKEWSMOOTH_OFFSET))
                            as f32;
                    } else {
                        overlap_result = OverlapState::Reject;
                    }
                }
                OverlapState::Reject => {}
            }
        } else if make_new_rows && top - bottom < block.max_blob_size {
            overlap_result = OverlapState::NewRow;
            let blob = block.blobs.remove(i);
            row_count += 1;
            block
                .rows
                .push(ToRow::new(blob, top, bottom, block.line_size));
            dest_idx = block.rows.len() - 1;
            smooth_factor = (1.0
                / (f64::from(row_count) * TEXTORD_SKEW_LAG + TEXTORD_SKEWSMOOTH_OFFSET2))
                as f32;
        } else {
            overlap_result = OverlapState::Reject;
        }

        if overlap_result != OverlapState::Reject {
            let mut d = dest_idx;
            while d > 0 && block.rows[d].min_y() > block.rows[d - 1].min_y() {
                block.rows.swap(d, d - 1);
                d -= 1;
            }
            while d + 1 < block.rows.len() && block.rows[d].min_y() < block.rows[d + 1].min_y() {
                block.rows.swap(d, d + 1);
                d += 1;
            }
            let singleton = block.rows[d].blobs.len() == 1;
            let should_update = if singleton {
                true
            } else {
                let n = block.rows[d].blobs.len();
                let prev_box = block.rows[d].blobs[n - 2];
                !major_x_overlap(prev_box, (left, bottom_raw, right, top_raw))
            };
            if should_update {
                block_skew = (1.0 - smooth_factor) * block_skew
                    + smooth_factor * (bottom_raw as f32 - block.rows[d].initial_min_y());
            }
        } else {
            i += 1;
        }
    }

    block.rows.retain(|r| !r.blobs.is_empty());
}

/// `row_y_order` (`makerow.cpp:107-122`) as a sort comparator: rows with a
/// HIGHER `parallel_c()` sort first (descending order).
fn row_y_order_cmp(a: &ToRow, b: &ToRow) -> std::cmp::Ordering {
    b.parallel_c()
        .partial_cmp(&a.parallel_c())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// `fit_parallel_lms` (`makerow.cpp:1970-1991`) -- fit an LMS line to a
/// row, constrained parallel to `gradient`, storing the result into both
/// the parallel fit (`set_parallel_line`) and the general fit
/// (`set_line`). `textord_straight_baselines` (default `false`) is pinned
/// -- see the module doc's "pinned flags" note; the unconstrained re-fit
/// branch it would gate is not ported.
pub fn fit_parallel_lms(gradient: f32, row: &mut ToRow) {
    let mut lms = DetLineFit::default();
    for &(left, bottom, right, _top) in &row.blobs {
        lms.add(ICoord::new((left + right) / 2, bottom));
    }
    let (c, error) = lms.constrained_fit_mc(f64::from(gradient));
    row.set_parallel_line(gradient, c, error);
    row.set_line(gradient, c, error);
}

/// `fit_parallel_rows` (`makerow.cpp:1928-1961`) -- re-fit every row in
/// the block to `gradient`, dropping empty rows first, then re-sort by
/// [`row_y_order_cmp`] (the C++ `ELIST2_IT::sort`, presumed non-stable
/// `qsort`-backed; this port uses Rust's stable `sort_by` -- ties in
/// `parallel_c()` are avoided by this wave's fixture construction, see
/// module doc).
pub fn fit_parallel_rows(block: &mut ToBlockCtx, gradient: f32) {
    let mut i = 0usize;
    while i < block.rows.len() {
        if block.rows[i].blobs.is_empty() {
            block.rows.remove(i);
        } else {
            fit_parallel_lms(gradient, &mut block.rows[i]);
            i += 1;
        }
    }
    block.rows.sort_by(row_y_order_cmp);
}

/// `find_best_dropout_row` (`makerow.cpp:696-757`). `rows[row_idx]` is the
/// row under test; `distance` is its dropout distance (from
/// [`compute_dropout_distances`]); `line_index` is
/// `floor(rows[row_idx].intercept())`. Returns `true` iff the row should
/// be deleted (a neighbour has strictly better dropout characteristics, or
/// ties and is more believable).
fn find_best_dropout_row(
    rows: &[ToRow],
    row_idx: usize,
    distance: i32,
    dist_limit: f32,
    line_index: i32,
) -> bool {
    let n = rows.len();
    let (row_inc, abs_dist): (i32, i32) = if distance < 0 {
        (1, -distance)
    } else {
        (-1, distance)
    };
    if abs_dist as f32 > dist_limit {
        return true;
    }
    let at_last = row_idx == n - 1;
    let at_first = row_idx == 0;
    if (distance < 0 && !at_last) || (distance >= 0 && !at_first) {
        let mut row_offset: i32 = row_inc;
        loop {
            let next_idx = data_relative(n, row_idx, row_offset as isize);
            let next_index = rows[next_idx].intercept().floor() as i32;
            let nearer_neighbour = (distance < 0
                && next_index < line_index
                && next_index > line_index + distance + distance)
                || (distance >= 0
                    && next_index > line_index
                    && next_index < line_index + distance + distance);
            let tied_but_more_believable = (next_index == line_index
                || next_index == line_index + distance + distance)
                && rows[row_idx].believability() <= rows[next_idx].believability();
            if nearer_neighbour || tied_but_more_believable {
                return true;
            }
            row_offset += row_inc;
            let cont = (next_index == line_index || next_index == line_index + distance + distance)
                && row_offset < n as i32;
            if !cont {
                break;
            }
        }
    }
    false
}

/// `delete_non_dropout_rows` (`makerow.cpp:612-688`) -- computes the
/// occupation/dropout profile over every row's blobs (via the wave-1
/// occupation leaves: [`compute_line_occupation`],
/// [`compute_occupation_threshold`], [`compute_dropout_distances`]), then
/// prunes rows whose neighbours have better dropout characteristics
/// ([`find_best_dropout_row`]). Every surviving row's blob list is then
/// unconditionally moved back into the block's pool too (matching the
/// C++'s unconditional second loop, `makerow.cpp:685-687`) -- after this
/// call every row is a position-only marker (`min_y`/`max_y`/`intercept`
/// etc. intact, `blobs` empty) and every blob (from pruned AND surviving
/// rows alike) is back in `block.blobs`, ready for
/// [`assign_blobs_to_rows`] to redistribute.
pub fn delete_non_dropout_rows(block: &mut ToBlockCtx, gradient: f32) {
    if block.rows.is_empty() {
        return;
    }
    let block_box = deskew_block_coords(block, gradient);
    let mut min_y = block_box.1 - 1;
    let mut max_y = block_box.3 + 1;
    for row in &block.rows {
        let line_index = row.intercept().floor() as i32;
        if line_index <= min_y {
            min_y = line_index - 1;
        }
        if line_index >= max_y {
            max_y = line_index + 1;
        }
    }
    let line_count = max_y - min_y + 1;
    if line_count <= 0 {
        return;
    }

    let all_blobs: Vec<(i32, i32, i32, i32)> = block
        .rows
        .iter()
        .flat_map(|r| r.blobs.iter().copied())
        .collect();
    let (occupation, _initial_deltas) = compute_line_occupation(&all_blobs, gradient, min_y, max_y);
    let low_window = (f64::from(block.line_spacing) * (K_DESCENDER_FRACTION + K_ASCENDER_FRACTION))
        .ceil() as i32;
    let high_window =
        (f64::from(block.line_spacing) * (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION)).ceil() as i32;
    let mut deltas = compute_occupation_threshold(
        low_window,
        high_window,
        line_count,
        &occupation,
        TEXTORD_OCCUPANCY_THRESHOLD,
    );
    compute_dropout_distances(&occupation, &mut deltas, line_count);

    let mut i = 0usize;
    while i < block.rows.len() {
        let line_index = block.rows[i].intercept().floor() as i32;
        let distance = deltas[(line_index - min_y) as usize];
        if find_best_dropout_row(
            &block.rows,
            i,
            distance,
            block.line_spacing / 2.0,
            line_index,
        ) {
            let removed = block.rows.remove(i);
            block.blobs.extend(removed.blobs);
        } else {
            i += 1;
        }
    }
    {
        let blobs = &mut block.blobs;
        for row in &mut block.rows {
            blobs.append(&mut row.blobs);
        }
    }
}

/// `adjust_row_limits` (`makerow.cpp:1129-1156`) -- reset every row's
/// `[min_y,max_y]` to the standard fractions of size around its
/// `intercept()`. `size /= (kXHeightFraction+kAscenderFraction+
/// kDescenderFraction)` is a `float /= double` compound assignment (the
/// C++ promotes `size` to `double` for the division, then narrows back);
/// note the divisor is exactly `1.0` given the pinned fraction values, but
/// this port still performs the (bit-exact no-op) division rather than
/// eliding it, matching the source.
pub fn adjust_row_limits(block: &mut ToBlockCtx) {
    for row in &mut block.rows {
        let size0 = row.max_y() - row.min_y();
        let size = (f64::from(size0)
            / (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION + K_DESCENDER_FRACTION))
            as f32;
        let ymax = (f64::from(size) * (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION)) as f32;
        let ymin = (-f64::from(size) * K_DESCENDER_FRACTION) as f32;
        let intercept = row.intercept();
        row.set_limits(intercept + ymin, intercept + ymax);
        row.merged = false;
    }
}

/// `compute_row_stats` (`makerow.cpp:1163-1247`) -- compute inter-row
/// spacing and the block's line-spacing/line-size/`baseline_offset`/
/// `key_row` estimate. `textord_new_initial_xheight` is pinned `true` --
/// see module doc.
pub fn compute_row_stats(block: &mut ToBlockCtx) {
    let n = block.rows.len();
    if n == 0 {
        return;
    }
    let mut spacing_rows: Vec<usize> = Vec::with_capacity(n.saturating_sub(1));
    let mut prev_idx: Option<usize> = None;
    for idx in (0..n).rev() {
        if let Some(p) = prev_idx {
            spacing_rows.push(p);
            let spacing = block.rows[idx].intercept() - block.rows[p].intercept();
            block.rows[p].spacing = if spacing < 0.1 && spacing > -0.1 {
                0.0
            } else {
                spacing
            };
        }
        prev_idx = Some(idx);
    }
    let final_row = prev_idx.expect("n > 0 guarantees at least one iteration");
    block.key_row = Some(final_row);
    block.baseline_offset = block.rows[final_row].parallel_c() % block.line_spacing;

    if !spacing_rows.is_empty() {
        let rowcount = spacing_rows.len();
        let ri = rowcount * 3 / 4;
        spacing_rows.select_nth_unstable_by(ri, |&a, &b| {
            // PARITY PIN: total order (spacing, intercept, min_y) -- see the
            // oracle's matching comparator; nth_element tie order is
            // unspecified in real Tesseract.
            block.rows[a]
                .spacing
                .partial_cmp(&block.rows[b].spacing)
                .unwrap()
                .then(
                    block.rows[a]
                        .intercept()
                        .partial_cmp(&block.rows[b].intercept())
                        .unwrap(),
                )
                .then(
                    block.rows[a]
                        .min_y()
                        .partial_cmp(&block.rows[b].min_y())
                        .unwrap(),
                )
        });
        let mut iqr = block.rows[spacing_rows[ri]].spacing;
        let ri2 = rowcount / 4;
        spacing_rows.select_nth_unstable_by(ri2, |&a, &b| {
            // PARITY PIN: total order (spacing, intercept, min_y) -- see the
            // oracle's matching comparator; nth_element tie order is
            // unspecified in real Tesseract.
            block.rows[a]
                .spacing
                .partial_cmp(&block.rows[b].spacing)
                .unwrap()
                .then(
                    block.rows[a]
                        .intercept()
                        .partial_cmp(&block.rows[b].intercept())
                        .unwrap(),
                )
                .then(
                    block.rows[a]
                        .min_y()
                        .partial_cmp(&block.rows[b].min_y())
                        .unwrap(),
                )
        });
        iqr -= block.rows[spacing_rows[ri2]].spacing;
        let ri3 = rowcount / 2;
        spacing_rows.select_nth_unstable_by(ri3, |&a, &b| {
            // PARITY PIN: total order (spacing, intercept, min_y) -- see the
            // oracle's matching comparator; nth_element tie order is
            // unspecified in real Tesseract.
            block.rows[a]
                .spacing
                .partial_cmp(&block.rows[b].spacing)
                .unwrap()
                .then(
                    block.rows[a]
                        .intercept()
                        .partial_cmp(&block.rows[b].intercept())
                        .unwrap(),
                )
                .then(
                    block.rows[a]
                        .min_y()
                        .partial_cmp(&block.rows[b].min_y())
                        .unwrap(),
                )
        });
        let key_row_idx = spacing_rows[ri3];
        block.key_row = Some(key_row_idx);
        let key_spacing = block.rows[key_row_idx].spacing;
        if rowcount > 2 && f64::from(iqr) < f64::from(key_spacing) * TEXTORD_LINESPACE_IQRLIMIT {
            if key_spacing < block.line_spacing {
                block.line_size = key_spacing;
            } else {
                block.line_size = block.line_spacing;
            }
            if block.line_size < TEXTORD_MIN_XHEIGHT {
                block.line_size = TEXTORD_MIN_XHEIGHT;
            }
            block.line_spacing = key_spacing;
            block.max_blob_size = (f64::from(block.line_spacing) * TEXTORD_EXCESS_BLOBSIZE) as f32;
        }
        block.baseline_offset = block.rows[key_row_idx].intercept() % block.line_spacing;
    }
}

/// `expand_rows` (`makerow.cpp:976-1122`) -- expand each row to the least
/// of its allowed size and touching its neighbours, swallowing a
/// neighbour entirely if the expansion would wholly contain it.
/// `textord_new_initial_xheight` is pinned `true` -- see module doc (the
/// `!textord_new_initial_xheight` branch, a second `compute_row_stats`
/// call, is dropped).
pub fn expand_rows(block: &mut ToBlockCtx, gradient: f32) {
    adjust_row_limits(block);
    if block.rows.is_empty() {
        return;
    }
    compute_row_stats(block);
    assign_blobs_to_rows(block, Some(gradient), true, false); // pass 4
    if block.rows.is_empty() {
        return;
    }
    fit_parallel_rows(block, gradient);
    if block.rows.is_empty() {
        return;
    }

    let mut row_idx = block.rows.len() - 1; // move_to_last()
    loop {
        let y_max0 = block.rows[row_idx].max_y();
        let y_min0 = block.rows[row_idx].min_y();
        let intercept = block.rows[row_idx].intercept();
        let mut y_bottom = (f64::from(intercept)
            - f64::from(block.line_size) * TEXTORD_EXPANSION_FACTOR * K_DESCENDER_FRACTION)
            as f32;
        let mut y_top = (f64::from(intercept)
            + f64::from(block.line_size)
                * TEXTORD_EXPANSION_FACTOR
                * (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION)) as f32;
        let mut y_min = y_min0;
        let mut y_max = y_max0;

        if y_min0 > y_bottom {
            let mut swallowed = true;
            while swallowed && row_idx + 1 < block.rows.len() {
                swallowed = false;
                let test_idx = row_idx + 1;
                if block.rows[test_idx].max_y() > y_bottom {
                    if block.rows[test_idx].min_y() > y_bottom {
                        let mut moved = std::mem::take(&mut block.rows[test_idx].blobs);
                        block.rows[row_idx].blobs.append(&mut moved);
                        block.rows.remove(test_idx);
                        swallowed = true;
                    } else if block.rows[test_idx].max_y() < y_min {
                        y_bottom = block.rows[test_idx].max_y();
                    } else {
                        y_bottom = y_min;
                    }
                }
            }
            y_min = y_bottom;
        }
        if y_max0 < y_top {
            let mut swallowed = true;
            while swallowed && row_idx > 0 {
                swallowed = false;
                let test_idx = row_idx - 1;
                if block.rows[test_idx].min_y() < y_top {
                    if block.rows[test_idx].max_y() < y_top {
                        let mut moved = std::mem::take(&mut block.rows[test_idx].blobs);
                        block.rows[row_idx].blobs.append(&mut moved);
                        block.rows.remove(test_idx);
                        row_idx -= 1; // removed BEFORE row_idx -- shift
                        swallowed = true;
                    } else if block.rows[test_idx].min_y() < y_max {
                        y_top = block.rows[test_idx].min_y();
                    } else {
                        y_top = y_max;
                    }
                }
            }
            y_max = y_top;
        }
        block.rows[row_idx].set_limits(y_min, y_max);

        if row_idx == 0 {
            break;
        }
        row_idx -= 1;
    }
}

/// `cleanup_rows_making` (`makerow.cpp:563-605`) -- the row-refinement
/// orchestration: fit, prune, expand, then three re-assignment passes.
/// See the module doc's "carrier simplification" note for why passes 2
/// and 3 (which in the real C++ pour `large_blobs` then
/// `noise_blobs`+`small_blobs` into the pool) operate on the same pool
/// here as pass 1.
pub fn cleanup_rows_making(block: &mut ToBlockCtx, gradient: f32) {
    fit_parallel_rows(block, gradient);
    delete_non_dropout_rows(block, gradient);
    expand_rows(block, gradient);
    {
        let blobs = &mut block.blobs;
        for row in &mut block.rows {
            blobs.append(&mut row.blobs);
        }
    }
    assign_blobs_to_rows(block, Some(gradient), false, false); // pass 1
    assign_blobs_to_rows(block, Some(gradient), true, true); // pass 2
    assign_blobs_to_rows(block, Some(gradient), false, false); // pass 3
}

/// `make_initial_textrows` (`makerow.cpp:254-289`) -- the very first blob
/// assignment (pass 0, `gradient=None`), then an LMS fit of every row.
/// `ScrollView` debug drawing dropped (headless port, wave 1 precedent).
pub fn make_initial_textrows(block: &mut ToBlockCtx) {
    assign_blobs_to_rows(block, None, true, true);
    for row in &mut block.rows {
        let (m, c, error) = fit_lms_line(&row.blobs);
        row.set_line(m, c, error);
    }
}

/// `compute_page_skew` (`makerow.cpp:315-405`) -- average gradient/error
/// over every block's rows. `blocks` is a slice of per-block row slices
/// (the `TO_BLOCK_LIST` walk, skipping the `pb->IsText()` `POLY_BLOCK`
/// filter -- this carrier has no `POLY_BLOCK`, every block is text).
/// `textord_biased_skewcalc`'s documented double-divide
/// (`makerow.cpp:366-367`) is replicated verbatim (two sequential integer
/// divisions, not `blob_count / (row_err*row_err)`). Returns
/// `(page_m, page_err)`.
#[must_use]
pub fn compute_page_skew(blocks: &[&[ToRow]]) -> (f32, f32) {
    let row_count_total: usize = blocks.iter().map(|b| b.len()).sum();
    if row_count_total == 0 {
        return (0.0, 0.0);
    }
    let mut gradients: Vec<f32> = Vec::new();
    let mut errors: Vec<f32> = Vec::new();
    for &block in blocks {
        for row in block {
            let mut blob_count = row.blobs.len() as i32;
            let mut row_err = row.line_error().ceil() as i32;
            if row_err <= 0 {
                row_err = 1;
            }
            if TEXTORD_BIASED_SKEWCALC {
                blob_count /= row_err;
                blob_count /= row_err; // documented double-divide, makerow.cpp:366-367 -- replicate verbatim
                while blob_count > 0 {
                    gradients.push(row.line_m());
                    errors.push(row.line_error());
                    blob_count -= 1;
                }
            } else if blob_count >= TEXTORD_MIN_BLOBS_IN_ROW {
                gradients.push(row.line_m());
                errors.push(row.line_error());
            }
        }
    }
    if gradients.is_empty() {
        for &block in blocks {
            for row in block {
                gradients.push(row.line_m());
                errors.push(row.line_error());
            }
        }
    }
    let row_count = gradients.len();
    let row_index = (row_count as f64 * TEXTORD_SKEW_ILE) as usize;
    gradients.select_nth_unstable_by(row_index, |a, b| a.partial_cmp(b).unwrap());
    let page_m = gradients[row_index];
    errors.select_nth_unstable_by(row_index, |a, b| a.partial_cmp(b).unwrap());
    let page_err = errors[row_index];
    (page_m, page_err)
}

/// `make_rows` (`makerow.cpp:229-247`) -- the top-level per-page row
/// formation loop: initial assignment on every block, one page-wide skew
/// estimate, then cleanup on every block. `textord_test_landscape` is
/// pinned `false` -- every block uses the `FCOORD(1,0)` convention (no
/// rotation state carried by [`ToBlockCtx`] to override it). Returns the
/// page skew gradient (`port_m`).
pub fn make_rows(blocks: &mut [ToBlockCtx]) -> f32 {
    for block in blocks.iter_mut() {
        make_initial_textrows(block);
    }
    let row_slices: Vec<&[ToRow]> = blocks.iter().map(|b| b.rows.as_slice()).collect();
    let (page_m, _page_err) = compute_page_skew(&row_slices);
    for block in blocks.iter_mut() {
        cleanup_rows_making(block, page_m);
    }
    page_m
}

// ─── Wave 3: x-height chain (makerow.cpp:1276-1462 + makerow.h inlines) ──────
//
// The x-height stage is NOT inside make_rows -- the real Tesseract calls
// `Textord::compute_block_xheight` from baselinedetect.cpp after make_rows.
// Ported here as free fns on the ToBlockCtx/ToRow carriers; the dump example
// calls compute_block_xheight explicitly after make_rows (STAGE7+).

/// `MAX_HEIGHT_MODES` (`makerow.cpp:98`).
const MAX_HEIGHT_MODES: i32 = 12;
/// `textord_minxh` (`makerow.cpp`, default `0.25`).
const TEXTORD_MINXH: f64 = 0.25;
/// `textord_xheight_mode_fraction` (default `0.4`).
const TEXTORD_XHEIGHT_MODE_FRACTION: f32 = 0.4;
/// `textord_ascheight_mode_fraction` (default `0.08`).
const TEXTORD_ASCHEIGHT_MODE_FRACTION: f32 = 0.08;
/// `textord_descheight_mode_fraction` (default `0.08`).
const TEXTORD_DESCHEIGHT_MODE_FRACTION: f32 = 0.08;
/// `textord_ascx_ratio_min` (default `1.25`).
const TEXTORD_ASCX_RATIO_MIN: f32 = 1.25;
/// `textord_ascx_ratio_max` (default `1.8`).
const TEXTORD_ASCX_RATIO_MAX: f32 = 1.8;
/// `textord_descx_ratio_min` (default `0.25`).
const TEXTORD_DESCX_RATIO_MIN: f32 = 0.25;
/// `textord_descx_ratio_max` (default `0.6`).
const TEXTORD_DESCX_RATIO_MAX: f32 = 0.6;
/// `textord_xheight_error_margin` (default `0.1`).
const TEXTORD_XHEIGHT_ERROR_MARGIN: f32 = 0.1;
/// `textord_min_blob_height_fraction` (default `0.75`) -- the `fill_heights`
/// floating-blob threshold.
const TEXTORD_MIN_BLOB_HEIGHT_FRACTION: f32 = 0.75;
/// `textord_single_height_mode` (default `false`, `textord.cpp:40`) -- so
/// `cap_only` is always false on this path; pinned + documented.
const TEXTORD_SINGLE_HEIGHT_MODE: bool = false;
/// `CCStruct::kXHeightCapRatio` = kXHeightFraction / (kXHeightFraction +
/// kAscenderFraction) (`ccstruct.cpp:28`).
const K_XHEIGHT_CAP_RATIO: f64 = K_XHEIGHT_FRACTION / (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION);

/// `ROW_CATEGORY` (`makerow.h:36-41`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowCategory {
    /// `ROW_ASCENDERS_FOUND`.
    AscendersFound,
    /// `ROW_DESCENDERS_FOUND`.
    DescendersFound,
    /// `ROW_UNKNOWN`.
    Unknown,
    /// `ROW_INVALID`.
    Invalid,
}

/// `get_min_max_xheight` (`makerow.h:86-92`). `block_linesize` is `int` in
/// C++ (the caller narrows `TO_BLOCK::line_size` float→int at the call).
fn get_min_max_xheight(block_linesize: i32) -> (i32, i32) {
    let mut min_height = (f64::from(block_linesize) * TEXTORD_MINXH).floor() as i32;
    if (min_height as f32) < TEXTORD_MIN_XHEIGHT {
        min_height = TEXTORD_MIN_XHEIGHT as i32;
    }
    let max_height = (f64::from(block_linesize) * 3.0).ceil() as i32;
    (min_height, max_height)
}

/// `get_row_category` (`makerow.h:94-100`).
#[must_use]
pub fn get_row_category(row: &ToRow) -> RowCategory {
    if row.xheight <= 0.0 {
        return RowCategory::Invalid;
    }
    if row.ascrise > 0.0 {
        RowCategory::AscendersFound
    } else if row.descdrop != 0.0 {
        RowCategory::DescendersFound
    } else {
        RowCategory::Unknown
    }
}

/// `within_error_margin` (`makerow.h:102-104`).
fn within_error_margin(test: f32, num: f32, margin: f32) -> bool {
    test >= num * (1.0 - margin) && test <= num * (1.0 + margin)
}

/// `fill_heights` accumulating variant (`makerow.cpp:1418-1462`). The wave-1
/// [`fill_heights`] returns fresh `Stats`; `compute_block_xheight` instead
/// accumulates cap heights across every `ROW_UNKNOWN` row into block-level
/// `Stats`, so this adds into the caller's stats. PARITY PIN: the real
/// `fill_heights` skips `joined_to_prev()` blobs and repeated-char runs; on
/// this plain-tuple carrier there is no chopping (joined_to_prev always false)
/// and the synthetic fixtures carry no repeated-char runs, so every row blob
/// contributes -- documented, not a silent divergence.
fn fill_heights_into(
    row: &ToRow,
    gradient: f32,
    min_height: i32,
    max_height: i32,
    heights: &mut Stats,
    floating_heights: &mut Stats,
) {
    for &(left, bottom, right, top) in &row.blobs {
        let xcentre = (left + right) as f32 / 2.0;
        let height = (top - bottom) as f32;
        let top_adj = top as f32 - (gradient * xcentre + row.parallel_c());
        if top_adj >= min_height as f32 && top_adj <= max_height as f32 {
            let bucket = (top_adj + 0.5).floor() as i32;
            heights.add(bucket, 1);
            if height / top_adj < TEXTORD_MIN_BLOB_HEIGHT_FRACTION {
                floating_heights.add(bucket, 1);
            }
        }
    }
}

/// `compute_xheight_from_modes` (`makerow.cpp:1480-1562`). Returns
/// `xheight_evidence` (`best_count`) and writes `(xheight, ascrise)` via the
/// out tuple. `cap_only` is pinned `false` (single_height_mode default).
fn compute_xheight_from_modes(
    heights: &mut Stats,
    floating_heights: &Stats,
    cap_only: bool,
    min_height: i32,
    max_height: i32,
) -> (i32, f32, f32) {
    let blob_index = heights.mode();
    let blob_count = heights.pile_count(blob_index);
    if blob_count == 0 {
        return (0, 0.0, 0.0);
    }
    let modes = compute_height_modes(heights, min_height, max_height, MAX_HEIGHT_MODES);
    let mut mode_count = modes.len() as i32;
    if cap_only && mode_count > 1 {
        mode_count = 1;
    }
    let mut xheight = 0.0f32;
    let mut ascrise = 0.0f32;
    let mut in_best_pile = false;
    let mut prev_size = -i32::MAX;
    let mut best_count = 0i32;
    let mut x = 0i32;
    while x < mode_count - 1 {
        if modes[x as usize] != prev_size + 1 {
            in_best_pile = false;
        }
        let modes_x_count =
            heights.pile_count(modes[x as usize]) - floating_heights.pile_count(modes[x as usize]);
        if (modes_x_count as f32 >= blob_count as f32 * TEXTORD_XHEIGHT_MODE_FRACTION)
            && (in_best_pile || modes_x_count > best_count)
        {
            let mut asc = x + 1;
            while asc < mode_count {
                let ratio = modes[asc as usize] as f32 / modes[x as usize] as f32;
                if TEXTORD_ASCX_RATIO_MIN < ratio
                    && ratio < TEXTORD_ASCX_RATIO_MAX
                    && (heights.pile_count(modes[asc as usize]) as f32
                        >= blob_count as f32 * TEXTORD_ASCHEIGHT_MODE_FRACTION)
                {
                    if modes_x_count > best_count {
                        in_best_pile = true;
                        best_count = modes_x_count;
                    }
                    prev_size = modes[x as usize];
                    xheight = modes[x as usize] as f32;
                    ascrise = (modes[asc as usize] - modes[x as usize]) as f32;
                }
                asc += 1;
            }
        }
        x += 1;
    }
    if xheight == 0.0 {
        // single mode: subtract floating counts, find mode, restore.
        if floating_heights.get_total() > 0 {
            let mut xi = min_height;
            while xi < max_height {
                heights.add(xi, -floating_heights.pile_count(xi));
                xi += 1;
            }
            let blob_index2 = heights.mode();
            let mut xj = min_height;
            while xj < max_height {
                heights.add(xj, floating_heights.pile_count(xj));
                xj += 1;
            }
            xheight = blob_index2 as f32;
            ascrise = 0.0;
            best_count = heights.pile_count(blob_index2);
        } else {
            xheight = blob_index as f32;
            ascrise = 0.0;
            best_count = heights.pile_count(blob_index);
        }
    }
    (best_count, xheight, ascrise)
}

/// `compute_row_descdrop` (`makerow.cpp:1567-1615`). `asc_heights` is the
/// row's height `Stats`; returns the (non-positive) descdrop.
fn compute_row_descdrop(
    row: &ToRow,
    gradient: f32,
    xheight_blob_count: i32,
    asc_heights: &Stats,
) -> i32 {
    let mut i_min = asc_heights.min_bucket();
    if (i_min as f32 / row.xheight) < TEXTORD_ASCX_RATIO_MIN {
        i_min = (row.xheight * TEXTORD_ASCX_RATIO_MIN + 0.5).floor() as i32;
    }
    let mut i_max = asc_heights.max_bucket();
    if (i_max as f32 / row.xheight) > TEXTORD_ASCX_RATIO_MAX {
        i_max = (row.xheight * TEXTORD_ASCX_RATIO_MAX).floor() as i32;
    }
    let mut num_potential_asc = 0i32;
    let mut i = i_min;
    while i <= i_max {
        num_potential_asc += asc_heights.pile_count(i);
        i += 1;
    }
    let min_height = (row.xheight * TEXTORD_DESCX_RATIO_MIN + 0.5).floor() as i32;
    let max_height = (row.xheight * TEXTORD_DESCX_RATIO_MAX).floor() as i32;
    let mut heights = Stats::new(min_height, max_height);
    for &(left, bottom, right, _top) in &row.blobs {
        // joined_to_prev always false on this carrier (documented PIN).
        let xcentre = (left + right) as f32 / 2.0;
        let height = gradient * xcentre + row.parallel_c() - bottom as f32;
        if height >= min_height as f32 && height <= max_height as f32 {
            heights.add((height + 0.5).floor() as i32, 1);
        }
    }
    let blob_index = heights.mode();
    let mut blob_count = heights.pile_count(blob_index);
    let total_fraction = TEXTORD_DESCHEIGHT_MODE_FRACTION + TEXTORD_ASCHEIGHT_MODE_FRACTION;
    if ((blob_count + num_potential_asc) as f32) < xheight_blob_count as f32 * total_fraction {
        blob_count = 0;
    }
    if blob_count > 0 {
        -blob_index
    } else {
        0
    }
}

/// `Textord::compute_row_xheight` (`makerow.cpp:1391-1413`).
pub fn compute_row_xheight(row: &mut ToRow, rotation_y: f32, gradient: f32, block_line_size: i32) {
    // mark_repeated_chars: on this plain-tuple carrier there are no BTFT_LEADER
    // flags and the synthetic fixtures carry no repeated-char runs, so it is a
    // documented no-op that only sets the marked flag (PARITY PIN).
    if !row.rep_chars_marked {
        row.rep_chars_marked = true;
    }
    let (min_height, max_height) = get_min_max_xheight(block_line_size);
    let mut heights = Stats::new(min_height, max_height);
    let mut floating_heights = Stats::new(min_height, max_height);
    fill_heights_into(
        row,
        gradient,
        min_height,
        max_height,
        &mut heights,
        &mut floating_heights,
    );
    row.ascrise = 0.0;
    row.xheight = 0.0;
    let cap_only = TEXTORD_SINGLE_HEIGHT_MODE && rotation_y == 0.0;
    let (evidence, xh, asc) = compute_xheight_from_modes(
        &mut heights,
        &floating_heights,
        cap_only,
        min_height,
        max_height,
    );
    row.xheight_evidence = evidence;
    row.xheight = xh;
    row.ascrise = asc;
    row.descdrop = 0.0;
    if row.xheight > 0.0 {
        row.descdrop = compute_row_descdrop(row, gradient, row.xheight_evidence, &heights) as f32;
    }
}

/// `correct_row_xheight` (`makerow.cpp:1621-1690`).
pub fn correct_row_xheight(row: &mut ToRow, xheight: f32, ascrise: f32, descdrop: f32) {
    let row_category = get_row_category(row);
    let normal_xheight = within_error_margin(row.xheight, xheight, TEXTORD_XHEIGHT_ERROR_MARGIN);
    let cap_xheight =
        within_error_margin(row.xheight, xheight + ascrise, TEXTORD_XHEIGHT_ERROR_MARGIN);
    if row_category == RowCategory::AscendersFound {
        if row.descdrop >= 0.0 {
            row.descdrop = row.xheight * (descdrop / xheight);
        }
    } else if row_category == RowCategory::Invalid
        || (row_category == RowCategory::DescendersFound && (normal_xheight || cap_xheight))
        || (row_category == RowCategory::Unknown && normal_xheight)
    {
        row.xheight = xheight;
        row.ascrise = ascrise;
        row.descdrop = descdrop;
    } else if row_category == RowCategory::DescendersFound {
        row.ascrise = row.xheight * (ascrise / xheight);
    } else if row_category == RowCategory::Unknown {
        row.all_caps = true;
        if cap_xheight {
            row.xheight = xheight;
            row.ascrise = ascrise;
            row.descdrop = descdrop;
        } else {
            row.ascrise = row.xheight * (ascrise / (xheight + ascrise));
            row.xheight -= row.ascrise;
            row.descdrop = row.xheight * (descdrop / xheight);
        }
    }
}

/// `Textord::compute_block_xheight` (`makerow.cpp:1279-1389`). `rotation_y`
/// is `block->block->classify_rotation().y()` (0.0 on the eng single-column
/// path). Mutates every row's xheight/ascrise/descdrop and the block xheight.
pub fn compute_block_xheight(block: &mut ToBlockCtx, gradient: f32, rotation_y: f32) {
    let asc_frac_xheight = (K_ASCENDER_FRACTION / K_XHEIGHT_FRACTION) as f32;
    let desc_frac_xheight = (K_DESCENDER_FRACTION / K_XHEIGHT_FRACTION) as f32;
    if block.rows.is_empty() {
        return;
    }
    let block_linesize = block.line_size as i32;
    let (min_height, max_height) = get_min_max_xheight(block_linesize);
    let mut row_asc_xheights = Stats::new(min_height, max_height);
    let mut row_asc_ascrise = Stats::new(
        (min_height as f32 * asc_frac_xheight) as i32,
        (max_height as f32 * asc_frac_xheight) as i32,
    );
    let min_desc_height = (min_height as f32 * desc_frac_xheight) as i32;
    let max_desc_height = (max_height as f32 * desc_frac_xheight) as i32;
    let mut row_asc_descdrop = Stats::new(min_desc_height, max_desc_height);
    let mut row_desc_xheights = Stats::new(min_height, max_height);
    let mut row_desc_descdrop = Stats::new(min_desc_height, max_desc_height);
    let mut row_cap_xheights = Stats::new(min_height, max_height);
    let mut row_cap_floating_xheights = Stats::new(min_height, max_height);

    for idx in 0..block.rows.len() {
        if block.rows[idx].xheight <= 0.0 {
            compute_row_xheight(&mut block.rows[idx], rotation_y, gradient, block_linesize);
        }
        let category = get_row_category(&block.rows[idx]);
        let row = &block.rows[idx];
        match category {
            RowCategory::AscendersFound => {
                row_asc_xheights.add(row.xheight as i32, row.xheight_evidence);
                row_asc_ascrise.add(row.ascrise as i32, row.xheight_evidence);
                row_asc_descdrop.add((-row.descdrop) as i32, row.xheight_evidence);
            }
            RowCategory::DescendersFound => {
                row_desc_xheights.add(row.xheight as i32, row.xheight_evidence);
                row_desc_descdrop.add((-row.descdrop) as i32, row.xheight_evidence);
            }
            RowCategory::Unknown => {
                fill_heights_into(
                    row,
                    gradient,
                    min_height,
                    max_height,
                    &mut row_cap_xheights,
                    &mut row_cap_floating_xheights,
                );
            }
            RowCategory::Invalid => {}
        }
    }

    let mut xheight;
    let mut ascrise = 0.0f32;
    let mut descdrop = 0.0f32;
    if row_asc_xheights.get_total() > 0 {
        xheight = row_asc_xheights.median() as f32;
        ascrise = row_asc_ascrise.median() as f32;
        descdrop = -(row_asc_descdrop.median() as f32);
    } else if row_desc_xheights.get_total() > 0 {
        xheight = row_desc_xheights.median() as f32;
        descdrop = -(row_desc_descdrop.median() as f32);
    } else if row_cap_xheights.get_total() > 0 {
        let cap_only = TEXTORD_SINGLE_HEIGHT_MODE && rotation_y == 0.0;
        let (_evidence, xh, asc) = compute_xheight_from_modes(
            &mut row_cap_xheights,
            &row_cap_floating_xheights,
            cap_only,
            min_height,
            max_height,
        );
        xheight = xh;
        ascrise = asc;
        if ascrise == 0.0 {
            xheight = row_cap_xheights.median() as f32 * K_XHEIGHT_CAP_RATIO as f32;
        }
    } else {
        xheight = block.line_size * K_XHEIGHT_FRACTION as f32;
    }
    let mut corrected_xheight = false;
    if xheight < TEXTORD_MIN_XHEIGHT {
        xheight = TEXTORD_MIN_XHEIGHT;
        corrected_xheight = true;
    }
    if corrected_xheight || ascrise <= 0.0 {
        ascrise = xheight * asc_frac_xheight;
    }
    if corrected_xheight || descdrop >= 0.0 {
        descdrop = -(xheight * desc_frac_xheight);
    }
    block.xheight = xheight;
    for row in &mut block.rows {
        correct_row_xheight(row, xheight, ascrise, descdrop);
    }
}

#[cfg(test)]
mod wave2_tests {
    use super::*;

    #[test]
    fn two_row_assignment_hand_computed() {
        // Two well-separated horizontal lines, single-column, single pass
        // (make_new_rows=true, reject_misses=true -- the pass-0 config).
        let mut block = ToBlockCtx {
            blobs: vec![
                // "line 1" around y=[100,120]
                (0, 100, 10, 120),
                (12, 100, 22, 120),
                (24, 100, 34, 120),
                // "line 2" around y=[0,20], far enough away to be a
                // distinct row (line_size=20, so a gap of 80 is way
                // outside max_blob_size).
                (0, 0, 10, 20),
                (12, 0, 22, 20),
            ],
            block_left: 0,
            line_spacing: 100.0,
            line_size: 20.0,
            max_blob_size: 40.0,
            ..Default::default()
        };
        assign_blobs_to_rows(&mut block, None, true, true);
        assert_eq!(
            block.rows.len(),
            2,
            "expected two distinct rows: {:?}",
            block.rows
        );
        // Rows are kept in descending-min_y order (topmost first).
        assert_eq!(block.rows[0].blobs.len(), 3);
        assert_eq!(block.rows[1].blobs.len(), 2);
        assert!(block.rows[0].min_y() > block.rows[1].min_y());
        assert!(
            block.blobs.is_empty(),
            "every blob should have been assigned"
        );
    }

    #[test]
    fn noise_blob_rejected_when_make_new_rows_false() {
        // One established row; a lone noise blob far above it. With
        // make_new_rows=false and reject_misses=true, the noise blob must
        // be REJECTed (left in the pool), not folded into the row or
        // given a row of its own.
        let mut block = ToBlockCtx {
            blobs: vec![],
            block_left: 0,
            line_spacing: 30.0,
            line_size: 20.0,
            max_blob_size: 40.0,
            rows: vec![ToRow::new((0, 100, 10, 120), 120.0, 100.0, 20.0)],
            ..Default::default()
        };
        block.blobs.push((50, 500, 60, 520)); // far above, isolated
        assign_blobs_to_rows(&mut block, Some(0.0), true, false);
        assert_eq!(block.rows.len(), 1, "no new row should have been created");
        assert_eq!(
            block.rows[0].blobs.len(),
            1,
            "the noise blob must not join the row"
        );
        assert_eq!(
            block.blobs.len(),
            1,
            "the noise blob stays in the pool (REJECT)"
        );
    }

    #[test]
    fn row_y_order_sorts_descending_by_parallel_c() {
        let mut rows = [
            ToRow::new((0, 0, 10, 10), 10.0, 0.0, 10.0),
            ToRow::new((0, 0, 10, 10), 10.0, 0.0, 10.0),
            ToRow::new((0, 0, 10, 10), 10.0, 0.0, 10.0),
        ];
        rows[0].set_parallel_line(0.0, 5.0, 0.0);
        rows[1].set_parallel_line(0.0, 50.0, 0.0);
        rows[2].set_parallel_line(0.0, 20.0, 0.0);
        rows.sort_by(row_y_order_cmp);
        assert!((rows[0].parallel_c() - 50.0).abs() < 1e-6);
        assert!((rows[1].parallel_c() - 20.0).abs() < 1e-6);
        assert!((rows[2].parallel_c() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn expand_rows_limit_arithmetic_hand_case() {
        // adjust_row_limits: size=(max-min)/1.0; ymax=size*0.75;
        // ymin=-size*0.25; set_limits(intercept+ymin, intercept+ymax).
        let mut block = ToBlockCtx {
            line_spacing: 40.0,
            line_size: 20.0,
            max_blob_size: 40.0,
            block_left: 0,
            rows: vec![ToRow::new((0, 90, 10, 110), 110.0, 90.0, 20.0)],
            ..Default::default()
        };
        block.rows[0].set_parallel_line(0.0, 100.0, 0.0); // intercept() == 100.0
        let before_min = block.rows[0].min_y();
        let before_max = block.rows[0].max_y();
        adjust_row_limits(&mut block);
        let size = before_max - before_min;
        let expect_max = 100.0 + size * 0.75;
        let expect_min = 100.0 - size * 0.25;
        assert!((block.rows[0].max_y() - expect_max).abs() < 1e-4);
        assert!((block.rows[0].min_y() - expect_min).abs() < 1e-4);
    }

    #[test]
    fn compute_page_skew_single_row_uses_desperate_fallback() {
        // A lone row with zero blobs and row_err forced to 1 still isn't
        // enough for the biased path to push any samples (blob_count=0
        // stays 0 after the double-divide), so the desperate fallback
        // must push it: page_m == that row's line_m().
        let mut row = ToRow::default();
        row.set_line(1.5, 0.0, 0.0);
        row.set_parallel_line(1.5, 0.0, 0.0);
        let rows = vec![row];
        let blocks: [&[ToRow]; 1] = [&rows];
        let (page_m, page_err) = compute_page_skew(&blocks);
        assert!((page_m - 1.5).abs() < 1e-6);
        assert_eq!(page_err, 0.0);
    }

    #[test]
    fn compute_page_skew_empty_returns_zero() {
        let blocks: [&[ToRow]; 0] = [];
        assert_eq!(compute_page_skew(&blocks), (0.0, 0.0));
    }

    #[test]
    fn make_rows_end_to_end_two_lines_single_block() {
        // Two clean horizontal lines of blobs; end-to-end make_rows should
        // converge on exactly two rows with all blobs assigned.
        let mut block = ToBlockCtx {
            blobs: vec![
                (0, 100, 10, 120),
                (12, 102, 22, 122),
                (24, 98, 34, 118),
                (36, 100, 46, 120),
                (0, 0, 10, 20),
                (12, 1, 22, 21),
                (24, 0, 34, 20),
            ],
            block_left: 0,
            line_spacing: 100.0,
            line_size: 20.0,
            max_blob_size: 40.0,
            ..Default::default()
        };
        let page_m = make_rows(std::slice::from_mut(&mut block));
        assert!(page_m.abs() < 1.0, "near-zero synthetic skew: {page_m}");
        assert_eq!(block.rows.len(), 2, "rows: {:?}", block.rows);
        let total_assigned: usize = block.rows.iter().map(|r| r.blobs.len()).sum();
        assert_eq!(total_assigned, 7, "every blob should end up in some row");
    }
}

#[cfg(test)]
mod wave3_tests {
    use super::*;

    /// `get_row_category` covers all four `ROW_CATEGORY` branches.
    #[test]
    fn row_category_branches() {
        let mut r = ToRow::default();
        assert_eq!(get_row_category(&r), RowCategory::Invalid); // xheight <= 0
        r.xheight = 12.0;
        assert_eq!(get_row_category(&r), RowCategory::Unknown); // ascrise 0, descdrop 0
        r.descdrop = -3.0;
        assert_eq!(get_row_category(&r), RowCategory::DescendersFound);
        r.ascrise = 4.0;
        assert_eq!(get_row_category(&r), RowCategory::AscendersFound);
    }

    /// `within_error_margin` matches the C++ `[num*(1-m), num*(1+m)]` band.
    #[test]
    fn error_margin_band() {
        assert!(within_error_margin(10.0, 10.0, 0.1));
        assert!(within_error_margin(10.9, 10.0, 0.1));
        assert!(!within_error_margin(11.1, 10.0, 0.1));
        assert!(within_error_margin(9.1, 10.0, 0.1));
        assert!(!within_error_margin(8.9, 10.0, 0.1));
    }

    /// `get_min_max_xheight`: floor(linesize*0.25) clamped to >=10, ceil(*3).
    #[test]
    fn min_max_xheight_range() {
        // linesize 20 -> min = floor(5) = 5 -> clamped to 10; max = ceil(60) = 60.
        assert_eq!(get_min_max_xheight(20), (10, 60));
        // linesize 100 -> min = floor(25) = 25; max = ceil(300) = 300.
        assert_eq!(get_min_max_xheight(100), (25, 300));
    }

    /// A single-mode row (all blobs one height) yields that height as the
    /// x-height with zero ascrise (the `compute_xheight_from_modes` single-mode
    /// fallback).
    #[test]
    fn single_mode_xheight() {
        let mut heights = Stats::new(10, 60);
        for _ in 0..8 {
            heights.add(20, 1);
        }
        let floating = Stats::new(10, 60);
        let (evidence, xh, asc) =
            compute_xheight_from_modes(&mut heights, &floating, false, 10, 60);
        assert_eq!(xh, 20.0);
        assert_eq!(asc, 0.0);
        assert_eq!(evidence, 8);
    }

    /// A bimodal row (x-height mode + an ascender mode at ~1.5x) recovers both.
    #[test]
    fn bimodal_xheight_ascrise() {
        let mut heights = Stats::new(10, 60);
        for _ in 0..10 {
            heights.add(20, 1); // x-height pile
        }
        for _ in 0..3 {
            heights.add(30, 1); // ascender pile (30/20 = 1.5, in [1.25,1.8])
        }
        let floating = Stats::new(10, 60);
        let (_ev, xh, asc) = compute_xheight_from_modes(&mut heights, &floating, false, 10, 60);
        assert_eq!(xh, 20.0);
        assert_eq!(asc, 10.0); // 30 - 20
    }
}
