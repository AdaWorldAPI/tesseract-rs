//! Activation functions (recognizer Leaf 3) — the transcode of Tesseract's
//! `lstm/functions.h` non-linearities, in **f32** to match the FAST_FLOAT build.
//!
//! `Tanh` / `Logistic` are 4096-entry lookup tables with linear interpolation
//! (`kTableSize = 4096`, `kScaleFactor = 256`; functions.h:44-72). The tables
//! are the generated `TanhTable[i] = tanh(i/256)` / `LogisticTable[i] =
//! logistic(i/256)` (generate_lut.py) — computed here in f64 then stored as f32,
//! exactly as the generator does, and byte-parity-verified against
//! libtesseract's baked tables. `Relu` / `ClipF` / `ClipG` / `Identity` /
//! `Softmax` are direct (functions.h:85-207).

use std::sync::LazyLock;

/// `kTableSize` (functions.h:35).
const TABLE_SIZE: usize = 4096;
/// `kScaleFactor` (functions.h:37) — float arg → table index.
const SCALE_FACTOR: f32 = 256.0;

/// `TanhTable[i] = tanh(i / 256)` — generated in f64, stored f32 (generate_lut.py:20).
static TANH_TABLE: LazyLock<[f32; TABLE_SIZE]> = LazyLock::new(|| {
    let mut t = [0.0_f32; TABLE_SIZE];
    for (i, e) in t.iter_mut().enumerate() {
        *e = (i as f64 / 256.0).tanh() as f32;
    }
    t
});

/// `LogisticTable[i] = 1 / (1 + exp(-i / 256))` (generate_lut.py:25).
static LOGISTIC_TABLE: LazyLock<[f32; TABLE_SIZE]> = LazyLock::new(|| {
    let mut t = [0.0_f32; TABLE_SIZE];
    for (i, e) in t.iter_mut().enumerate() {
        *e = (1.0 / (1.0 + (-(i as f64) / 256.0).exp())) as f32;
    }
    t
});

/// Table-interpolated sigmoid shared by [`tanh`] and [`logistic`]: for `x >= 0`,
/// index `TABLE` at `x·256` and linearly interpolate. Returns `1.0` past the
/// table (functions.h:48-56 / 63-71).
#[inline]
fn lut_interp(table: &[f32; TABLE_SIZE], x: f32) -> f32 {
    let xs = x * SCALE_FACTOR;
    let index = xs as usize; // (unsigned) truncation toward zero (x >= 0 here)
    if index >= TABLE_SIZE - 1 {
        return 1.0;
    }
    let v0 = table[index];
    let v1 = table[index + 1];
    v0 + (v1 - v0) * (xs - index as f32)
}

/// `Tanh(x)` (functions.h:44-57) — odd: `Tanh(-x) = -Tanh(x)`.
#[must_use]
pub fn tanh(x: f32) -> f32 {
    if x < 0.0 {
        return -tanh(-x);
    }
    lut_interp(&TANH_TABLE, x)
}

/// `Logistic(x)` (functions.h:59-72) — `Logistic(-x) = 1 - Logistic(x)`.
#[must_use]
pub fn logistic(x: f32) -> f32 {
    if x < 0.0 {
        return 1.0 - logistic(-x);
    }
    lut_interp(&LOGISTIC_TABLE, x)
}

/// `Relu(x)` (functions.h:101-108): `0` for `x <= 0`, else `x`.
#[must_use]
pub fn relu(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        x
    }
}

/// `ClipFFunc(x)` (functions.h:85-95): clamp to `[0, 1]`.
#[must_use]
pub fn clip_f(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// `ClipGFunc(x)` (functions.h:124-134): clamp to `[-1, 1]`.
#[must_use]
pub fn clip_g(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// `IdentityFunc(x)` (functions.h:156-160).
#[must_use]
pub fn identity(x: f32) -> f32 {
    x
}

/// `SoftmaxInPlace` (functions.h:180-207): shift by the max, `exp` (clipped to
/// `[-86, 0]` to guarantee non-zero output), normalize. No-op on an empty slice.
pub fn softmax_in_place(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    const MAX_ACTIVATION: f32 = 86.0;
    let max_output = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;
    for x in v.iter_mut() {
        let p = (*x - max_output).clamp(-MAX_ACTIVATION, 0.0).exp();
        total += p;
        *x = p;
    }
    if total > 0.0 {
        for x in v.iter_mut() {
            *x /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_is_odd_and_bounded() {
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(-1.5), -tanh(1.5));
        assert!((tanh(1.0) - 1.0_f32.tanh()).abs() < 1e-3);
        assert_eq!(tanh(100.0), 1.0, "clamps past the table");
        assert_eq!(tanh(-100.0), -1.0);
    }

    #[test]
    fn logistic_reflects_and_bounds() {
        assert!((logistic(0.0) - 0.5).abs() < 1e-3);
        // Logistic(-x) = 1 - Logistic(x).
        assert!((logistic(-2.0) - (1.0 - logistic(2.0))).abs() < 1e-6);
        assert_eq!(logistic(100.0), 1.0);
        assert_eq!(logistic(-100.0), 0.0);
    }

    #[test]
    fn relu_clip_identity() {
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(2.5), 2.5);
        assert_eq!(clip_f(-1.0), 0.0);
        assert_eq!(clip_f(2.0), 1.0);
        assert_eq!(clip_f(0.3), 0.3);
        assert_eq!(clip_g(-2.0), -1.0);
        assert_eq!(clip_g(2.0), 1.0);
        assert_eq!(identity(0.7), 0.7);
    }

    #[test]
    fn softmax_normalizes() {
        let mut v = [1.0_f32, 2.0, 3.0];
        softmax_in_place(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        assert!(v[2] > v[1] && v[1] > v[0], "monotone in the input");
        let mut empty: [f32; 0] = [];
        softmax_in_place(&mut empty); // no panic
    }
}
