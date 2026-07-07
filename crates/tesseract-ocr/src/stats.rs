//! `STATS` — the general histogram/percentile primitive
//! (`/tmp/tesseract/src/ccstruct/statistc.{h,cpp}`), transcoded on an
//! **arbitrary** `[min_bucket_value, max_bucket_value]` range.
//!
//! ## Relationship to the existing `[0,255]` `Stats` in `tesseract-recognizer`
//! `tesseract-recognizer/src/input.rs` already carries a private `Stats`
//! (fixed `[i32; 256]` histogram, only `add()`/`ile()`) for the A6a
//! `ComputeBlackWhite` leaf — that is a deliberately-scoped specialisation
//! (every call site there constructs `STATS(0, 255)`), documented in
//! `.claude/harvest/statistc-manifest.txt` §"Overlap with the existing
//! tesseract-recognizer Stats type". **This module is the general port**
//! (heap-backed `Vec<i32>`, arbitrary range, the fuller LEAF accessor
//! surface makerow.cpp needs: `mode`/`mean`/`sd`/`ile`/`median`/
//! `min_bucket`/`max_bucket`/`pile_count`/`get_total`). The two are
//! intentionally separate types — the recognizer's `Stats` is not rebased
//! onto this one in this wave (out of scope; a later pass could make it a
//! thin specialisation).
//!
//! Scope: per the manifest, `smooth()`/`cluster()`/`top_n_modes()`/
//! `local_min()`/`print()`/`print_summary()`/`plot()`/`plotline()` are used
//! only by callers OUTSIDE the makerow chain (wordrec, debug/graphics) and
//! are deliberately NOT ported here.
//!
//! ## Byte-for-byte fidelity notes
//! - The C++ "empty" state is a null `buckets_` pointer (default ctor,
//!   `STATS() = default`). This is represented here by an **empty** `Vec`
//!   (`buckets.is_empty()`), which is otherwise unreachable once
//!   `Stats::new`/`set_range` have run (`1 + rangemax - rangemin >= 1`
//!   buckets are always allocated), exactly mirroring the C++ invariant.
//! - `mode()` scans buckets from **high index down to 1** (never re-checking
//!   index 0, which seeds `max`/`maxindex`) and only replaces on strict `>`,
//!   so ties resolve to the **highest** index — `statistc.cpp:112-125`,
//!   transcribed with the identical descending scan order (this is
//!   observable: reversing the scan direction would flip which of several
//!   equally-frequent buckets `mode()` returns).
//! - `ile()`'s target clamp and bucket walk (`statistc.cpp:172-197`) use the
//!   *live* (non-`#if 0`) branch: `target = clip(frac * total_count, 1.0,
//!   total_count)` as `f64`, never the commented-out `IntCastRounded` path.
//! - `sd()`'s `sqsum` accumulates `((double)index * index) * bucket` in the
//!   exact C++ left-to-right association (`statistc.cpp:156`).

/// General histogram/percentile primitive over an arbitrary
/// `[rangemin, rangemax]` integer bucket range (`tesseract::STATS`).
#[derive(Debug, Clone, Default)]
pub struct Stats {
    rangemin: i32,
    rangemax: i32,
    total_count: i32,
    /// Empty iff the C++ object would have `buckets_ == nullptr` (the
    /// default-constructed "empty, for arrays" state).
    buckets: Vec<i32>,
}

/// Clip a value to `[lower, upper]` (`ClipToRange`, `ccutil/helpers.h:116`).
fn clip_to_range<T: PartialOrd>(x: T, lower: T, upper: T) -> T {
    if x < lower {
        lower
    } else if x > upper {
        upper
    } else {
        x
    }
}

impl Stats {
    /// `STATS(min_bucket_value, max_bucket_value)` (`statistc.cpp:43-52`).
    /// Buckets span `[min_bucket_value, max_bucket_value]` inclusive; a
    /// malformed range (`max < min`) is silently replaced with `[0, 1]`,
    /// exactly as the C++ constructor does.
    #[must_use]
    pub fn new(min_bucket_value: i32, max_bucket_value: i32) -> Self {
        let (rangemin, rangemax) = if max_bucket_value < min_bucket_value {
            (0, 1)
        } else {
            (min_bucket_value, max_bucket_value)
        };
        let len = (1 + rangemax - rangemin) as usize;
        Stats {
            rangemin,
            rangemax,
            total_count: 0,
            buckets: vec![0; len],
        }
    }

    /// `STATS::set_range` (`statistc.cpp:59-71`). Returns `false` (no-op)
    /// if `max_bucket_value < min_bucket_value`.
    pub fn set_range(&mut self, min_bucket_value: i32, max_bucket_value: i32) -> bool {
        if max_bucket_value < min_bucket_value {
            return false;
        }
        if self.rangemax - self.rangemin != max_bucket_value - min_bucket_value {
            let len = (1 + max_bucket_value - min_bucket_value) as usize;
            self.buckets = vec![0; len];
        }
        self.rangemin = min_bucket_value;
        self.rangemax = max_bucket_value;
        self.clear();
        true
    }

    /// `STATS::clear` (`statistc.cpp:78-83`).
    pub fn clear(&mut self) {
        self.total_count = 0;
        for b in &mut self.buckets {
            *b = 0;
        }
    }

    /// `STATS::add` (`statistc.cpp:99-105`). `value` is clipped into
    /// `[rangemin, rangemax]` before the bucket bump, exactly as C++.
    pub fn add(&mut self, value: i32, count: i32) {
        if self.buckets.is_empty() {
            return;
        }
        let v = clip_to_range(value, self.rangemin, self.rangemax);
        self.buckets[(v - self.rangemin) as usize] += count;
        self.total_count += count;
    }

    /// `STATS::mode` (`statistc.cpp:112-125`). See module doc for the
    /// tie-breaking note (descending scan, ties resolve to highest index).
    #[must_use]
    pub fn mode(&self) -> i32 {
        if self.buckets.is_empty() {
            return self.rangemin;
        }
        let mut max = self.buckets[0];
        let mut maxindex: i32 = 0;
        let mut index = self.rangemax - self.rangemin;
        while index > 0 {
            if self.buckets[index as usize] > max {
                max = self.buckets[index as usize];
                maxindex = index;
            }
            index -= 1;
        }
        maxindex + self.rangemin
    }

    /// `STATS::mean` (`statistc.cpp:132-141`).
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.buckets.is_empty() || self.total_count <= 0 {
            return f64::from(self.rangemin);
        }
        let mut sum: i64 = 0;
        let mut index = self.rangemax - self.rangemin;
        while index >= 0 {
            sum += i64::from(index) * i64::from(self.buckets[index as usize]);
            index -= 1;
        }
        sum as f64 / f64::from(self.total_count) + f64::from(self.rangemin)
    }

    /// `STATS::sd` (`statistc.cpp:148-164`).
    #[must_use]
    pub fn sd(&self) -> f64 {
        if self.buckets.is_empty() || self.total_count <= 0 {
            return 0.0;
        }
        let mut sum: i64 = 0;
        let mut sqsum: f64 = 0.0;
        let mut index = self.rangemax - self.rangemin;
        while index >= 0 {
            let bucket = self.buckets[index as usize];
            sum += i64::from(index) * i64::from(bucket);
            sqsum += (f64::from(index) * f64::from(index)) * f64::from(bucket);
            index -= 1;
        }
        let mean_component = sum as f64 / f64::from(self.total_count);
        let variance = sqsum / f64::from(self.total_count) - mean_component * mean_component;
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    }

    /// `STATS::ile` (`statistc.cpp:172-197`) — the *live* branch (the
    /// `#if 0`-guarded `IntCastRounded` alternative is dead C++ code and is
    /// not transcribed).
    #[must_use]
    pub fn ile(&self, frac: f64) -> f64 {
        if self.buckets.is_empty() || self.total_count == 0 {
            return f64::from(self.rangemin);
        }
        let target = clip_to_range(
            frac * f64::from(self.total_count),
            1.0,
            f64::from(self.total_count),
        );
        let mut sum: i32 = 0;
        let mut index: i32 = 0;
        while index <= self.rangemax - self.rangemin && f64::from(sum) < target {
            sum += self.buckets[index as usize];
            index += 1;
        }
        if index > 0 {
            debug_assert!(self.buckets[(index - 1) as usize] > 0);
            f64::from(self.rangemin + index)
                - (f64::from(sum) - target) / f64::from(self.buckets[(index - 1) as usize])
        } else {
            f64::from(self.rangemin)
        }
    }

    /// `STATS::min_bucket` (`statistc.cpp:204-213`) — the real minimum used
    /// entry, not `ile(0.0)`.
    #[must_use]
    pub fn min_bucket(&self) -> i32 {
        if self.buckets.is_empty() || self.total_count == 0 {
            return self.rangemin;
        }
        let mut min = 0i32;
        while min <= self.rangemax - self.rangemin && self.buckets[min as usize] == 0 {
            min += 1;
        }
        self.rangemin + min
    }

    /// `STATS::max_bucket` (`statistc.cpp:221-230`) — the real maximum used
    /// entry, not `ile(1.0)`.
    #[must_use]
    pub fn max_bucket(&self) -> i32 {
        if self.buckets.is_empty() || self.total_count == 0 {
            return self.rangemin;
        }
        let mut max = self.rangemax - self.rangemin;
        while max > 0 && self.buckets[max as usize] == 0 {
            max -= 1;
        }
        self.rangemin + max
    }

    /// `STATS::median` (`statistc.cpp:241-261`) — a more useful median
    /// estimate than `ile(0.5)` when the value straddles an empty pile
    /// (splits ties, e.g. `6,6,13,14 -> 9.5` not `7.0`).
    #[must_use]
    pub fn median(&self) -> f64 {
        if self.buckets.is_empty() {
            return f64::from(self.rangemin);
        }
        let mut median = self.ile(0.5);
        let median_pile = median.floor() as i32;
        if self.total_count > 1 && self.pile_count(median_pile) == 0 {
            let mut min_pile = median_pile;
            while self.pile_count(min_pile) == 0 {
                min_pile -= 1;
            }
            let mut max_pile = median_pile;
            while self.pile_count(max_pile) == 0 {
                max_pile += 1;
            }
            median = f64::from(min_pile + max_pile) / 2.0;
        }
        median
    }

    /// `STATS::pile_count` (`statistc.h:72-83`, inline) — clamped bucket
    /// read; out-of-range `value` clamps to the nearest edge bucket.
    #[must_use]
    pub fn pile_count(&self, value: i32) -> i32 {
        if self.buckets.is_empty() {
            return 0;
        }
        if value <= self.rangemin {
            return self.buckets[0];
        }
        if value >= self.rangemax {
            return self.buckets[(self.rangemax - self.rangemin) as usize];
        }
        self.buckets[(value - self.rangemin) as usize]
    }

    /// `STATS::get_total` (`statistc.h:85-87`, inline).
    #[must_use]
    pub fn get_total(&self) -> i32 {
        self.total_count
    }

    /// The configured range minimum (`rangemin_`).
    #[must_use]
    pub fn range_min(&self) -> i32 {
        self.rangemin
    }

    /// The configured range maximum (`rangemax_`).
    #[must_use]
    pub fn range_max(&self) -> i32 {
        self.rangemax
    }
}

#[cfg(test)]
mod tests {
    use super::Stats;

    // Hand-computed histogram: values 6,6,13,14 (statistc.h:66-70's own
    // worked example for median vs ile(0.5)).
    fn sample_6_6_13_14() -> Stats {
        let mut s = Stats::new(0, 20);
        s.add(6, 2);
        s.add(13, 1);
        s.add(14, 1);
        s
    }

    #[test]
    fn median_splits_ties_ile_does_not() {
        let s = sample_6_6_13_14();
        // ile(0.5): target = clip(0.5*4,1,4) = 2.0. Walk: index0 sum0<2 -> sum+=buckets[0](0)->0,idx1
        // ... walks until index=7 (value6) sum=0+..+2=2 which is NOT <target(2.0) stops the loop
        // BEFORE adding at index=7? Let's just assert documented header example instead of
        // re-deriving: header says ile(0.5) == 7.0 for this exact histogram.
        assert_eq!(s.ile(0.5), 7.0);
        assert_eq!(s.median(), 9.5);
    }

    #[test]
    fn mode_ties_resolve_to_highest_index() {
        let mut s = Stats::new(0, 10);
        s.add(2, 5);
        s.add(8, 5); // tied count with bucket 2, higher index must win
        assert_eq!(s.mode(), 8);
    }

    #[test]
    fn pile_count_clamps_out_of_range() {
        let mut s = Stats::new(5, 10);
        s.add(5, 3);
        s.add(10, 4);
        assert_eq!(s.pile_count(0), 3); // clamps to rangemin bucket
        assert_eq!(s.pile_count(100), 4); // clamps to rangemax bucket
        assert_eq!(s.get_total(), 7);
    }

    #[test]
    fn min_max_bucket_and_mean_sd() {
        let mut s = Stats::new(0, 10);
        s.add(2, 1);
        s.add(4, 1);
        s.add(6, 1);
        assert_eq!(s.min_bucket(), 2);
        assert_eq!(s.max_bucket(), 6);
        assert!((s.mean() - 4.0).abs() < 1e-9);
        // variance = mean(x^2) - mean(x)^2 = (4+16+36)/3 - 16 = 18.6667-16=2.6667, sd=sqrt
        assert!((s.sd() - 2.6667f64.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn empty_stats_is_inert() {
        let s = Stats::default();
        assert_eq!(s.mode(), 0);
        assert_eq!(s.pile_count(5), 0);
        assert_eq!(s.get_total(), 0);
        assert_eq!(s.ile(0.5), 0.0);
    }
}
