//! `TRand` — the transcode of Tesseract's deterministic RNG (`ccutil/helpers.h`
//! `class TRand`), which wraps C++ `std::minstd_rand` (the Lehmer/Park-Miller
//! LCG: `x_{n+1} = 48271·x_n mod (2^31 − 1)`, seed default `1`).
//!
//! The recognizer is NOT noise-free: `Convolve::Forward` and the image `Input`
//! path fill **out-of-image positions** with `randomizer_->SignedRand(...)`
//! (`convolve.cpp:68/75`, `networkio.cpp:416-429`), so byte-parity of the 2-D
//! front-end requires this exact RNG, exactly seeded. `LSTMRecognizer` seeds it
//! once with a fixed seed + one warm-up draw (`lstmrecognizer.h:290-291`).
//!
//! `std::minstd_rand` semantics transcribed (C++17 [rand.eng.lcong]):
//! - `seed(s)`: state = `s mod m`; if that is `0` (and `c == 0`), state = `1`.
//! - `operator()`: state = `a·state mod m`, then RETURNS the new state —
//!   values are in `[1, m−1]`, never 0.
//!
//! `SignedRand(range) = range·2·IntRand()/INT32_MAX − range` and
//! `UnsignedRand(range) = range·IntRand()/INT32_MAX` are computed in `f64`
//! (`double`), exactly as C++; callers narrow to `f32`/int themselves.

/// `minstd_rand` multiplier (C++ `std::minstd_rand`, a = 48271).
const A: u64 = 48271;
/// `minstd_rand` modulus (2^31 − 1, the Mersenne prime).
const M: u64 = 2_147_483_647;
/// `INT32_MAX` — the denominator in `SignedRand`/`UnsignedRand` (== `M` here,
/// but kept separate to mirror the C++ text).
const INT32_MAX_F64: f64 = 2_147_483_647.0;

/// Tesseract's `TRand` (`helpers.h`): `std::minstd_rand` + the two float
/// helpers. `Clone` is cheap; state is a single `u32`.
#[derive(Debug, Clone)]
pub struct TRand {
    /// Current LCG state, in `[1, M−1]`.
    state: u32,
}

impl Default for TRand {
    /// `std::minstd_rand` default-constructs with seed `1` (the standard's
    /// `default_seed`).
    fn default() -> Self {
        Self { state: 1 }
    }
}

impl TRand {
    /// `TRand::set_seed(uint64_t)` → `e.seed(seed)`: state = `seed mod m`,
    /// promoted to `1` when the residue is `0`.
    pub fn set_seed(&mut self, seed: u64) {
        let s = seed % M;
        self.state = if s == 0 { 1 } else { s as u32 };
    }

    /// `TRand::IntRand()` → `e()`: advance the LCG and return the new state
    /// (in `[1, 2^31−2]`).
    pub fn int_rand(&mut self) -> i32 {
        self.state = ((u64::from(self.state) * A) % M) as u32;
        self.state as i32
    }

    /// `SignedRand(range)`: uniform double in `[−range, range]`
    /// (`range * 2.0 * IntRand() / INT32_MAX - range`, evaluated left-to-right
    /// in f64 exactly as C++).
    pub fn signed_rand(&mut self, range: f64) -> f64 {
        range * 2.0 * f64::from(self.int_rand()) / INT32_MAX_F64 - range
    }

    /// `UnsignedRand(range)`: uniform double in `[0, range]`.
    pub fn unsigned_rand(&mut self, range: f64) -> f64 {
        range * f64::from(self.int_rand()) / INT32_MAX_F64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_minstd_reference_sequence() {
        // The canonical minstd_rand check: from seed 1, the 10000th draw is
        // 399268537 (the classic Park-Miller value for the 48271 variant is
        // pinned by the C++ standard's minstd_rand definition).
        let mut r = TRand::default();
        let mut last = 0;
        for _ in 0..10_000 {
            last = r.int_rand();
        }
        assert_eq!(
            last, 399_268_537,
            "std::minstd_rand 10000th output from seed 1"
        );
    }

    #[test]
    fn seed_zero_residue_promotes_to_one() {
        // seed(0) and seed(M) both leave the residue 0 -> state 1 == default.
        let mut a = TRand::default();
        let mut b = TRand::default();
        b.set_seed(0);
        let mut c = TRand::default();
        c.set_seed(M);
        let x = a.int_rand();
        assert_eq!(x, b.int_rand());
        assert_eq!(x, c.int_rand());
        // First draw from seed 1 is a itself: 48271.
        assert_eq!(x, 48_271);
    }

    #[test]
    fn signed_rand_spans_range() {
        let mut r = TRand::default();
        r.set_seed(42);
        for _ in 0..100 {
            let v = r.signed_rand(127.0);
            assert!((-127.0..=127.0).contains(&v));
        }
    }
}
