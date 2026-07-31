//! Reservoir side-by-side — the proven OCR LSTM kernel driven as an
//! echo-state recurrence, WITHOUT touching the OCR path.
//!
//! Baby-step probe (operator, 2026-07-31): "keep the existing LSTM in
//! parallel and do side by side." Concretely:
//!
//! - **Nothing in `src/` changes.** Both arms below go through the SAME
//!   proven surface — `Lstm::from_le_bytes` + `Lstm::forward` (byte-parity
//!   green vs libtesseract, `E-OCR-LSTM-1`). This example only *synthesizes
//!   wire bytes* for that loader, exactly as the unit tests already do. The
//!   OCR arm's guarantee is the crate's own test suite staying green.
//! - **No training, no learned weights.** The recurrent weights are FIXED,
//!   seeded, never updated — reservoir / echo-state framing. The kernel is a
//!   codebook here, not a model (codebook-not-matmul stays intact).
//! - The input shape is the 4x4 Morton tile: 16 lanes = `ni = 16`, state
//!   `ns = 16` (na = 32) — the domino/SoA board shape.
//!
//! Two arms, same weight PATTERN, different gains, same input sequences:
//!
//! - **RESERVOIR** (contractive): fading memory. Two sequences that differ
//!   only in their first 8 timesteps and share a 48-step suffix must
//!   CONVERGE on the suffix (the echo-state property). Can-it-fire.
//! - **LATCH** (forget-gate biased ~1, integrator cell in the linear zone):
//!   the same prefix difference must PERSIST. Can-it-stay-silent — a
//!   "convergence" test that any recurrence passes asserts nothing.
//!
//! Plus the step-2 preview: the per-step int8 quantization residual
//! (`h - q(h)/127`) — information the recurrence had but the stored state
//! dropped — printed as a surprise ("free energy") trace. That residual is
//! the honest byte-derived candidate for a MetaWord `free_e` source.
//!
//! Run: `cargo run -p tesseract-recognizer --example reservoir_side_by_side`
//! Self-validating: asserts its falsifiers, exits nonzero on failure.

use tesseract_recognizer::Lstm;

/// SplitMix64 — deterministic, dependency-free.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A weight byte in `[-amp, amp]`.
    fn i8_amp(&mut self, amp: i8) -> i8 {
        if amp == 0 {
            return 0;
        }
        let span = (amp as i64) * 2 + 1;
        ((self.next() % span as u64) as i64 - amp as i64) as i8
    }
}

const K_INT8_FLAG: u8 = 1;
const K_DOUBLE_FLAG: u8 = 128;

/// One gate's reservoir spec: weight amplitude, runtime scale, bias byte.
/// Wire scale = runtime scale x 127 (the loader divides by INT8_MAX).
#[derive(Clone, Copy)]
struct Gate {
    amp: i8,
    scale: f64,
    bias: i8,
}

/// A deterministic int-mode `WeightMatrix` in wire form (`ns x (na+1)`),
/// byte-compatible with `WeightMatrix::from_le_bytes_prefix` — the same
/// synthesis the crate's own unit tests use, with controlled gain.
fn wm_wire(ns: usize, na: usize, g: Gate, rng: &mut Rng) -> Vec<u8> {
    let dim2 = na + 1;
    let mut b = Vec::new();
    b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
    b.extend_from_slice(&(ns as u32).to_le_bytes());
    b.extend_from_slice(&(dim2 as u32).to_le_bytes());
    b.push(0); // empty_
    for _row in 0..ns {
        for col in 0..dim2 {
            // Last column is the bias input (implicit 1 in the source vector).
            let w = if col == na { g.bias } else { rng.i8_amp(g.amp) };
            b.push(w as u8);
        }
    }
    b.extend_from_slice(&(ns as u32).to_le_bytes());
    for _ in 0..ns {
        b.extend_from_slice(&(g.scale * 127.0).to_le_bytes());
    }
    b
}

/// A full LSTM payload — `i32 na_` + the four gates (CI, GI, GF1, GO) —
/// consumed by the PROVEN `Lstm::from_le_bytes`, never a parallel loader.
fn lstm_wire(ni: usize, ns: usize, gates: [Gate; 4], seed: u64) -> Vec<u8> {
    let na = ni + ns;
    let mut rng = Rng(seed);
    let mut b = Vec::new();
    b.extend_from_slice(&(na as i32).to_le_bytes());
    for g in gates {
        b.extend_from_slice(&wm_wire(ns, na, g, &mut rng));
    }
    b
}

/// A 16-lane int8 Morton-tile timestep.
fn tile(rng: &mut Rng) -> Vec<i8> {
    (0..16).map(|_| rng.i8_amp(50)).collect()
}

/// Two sequences: different 8-step prefixes, one shared 48-step suffix.
fn washout_pair() -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let mut ra = Rng(0xA11CE);
    let mut rb = Rng(0xB0B);
    let mut rs = Rng(0x5EED);
    let pa: Vec<Vec<i8>> = (0..8).map(|_| tile(&mut ra)).collect();
    let pb: Vec<Vec<i8>> = (0..8).map(|_| tile(&mut rb)).collect();
    let suffix: Vec<Vec<i8>> = (0..48).map(|_| tile(&mut rs)).collect();
    let a = pa.into_iter().chain(suffix.iter().cloned()).collect();
    let b = pb.into_iter().chain(suffix).collect();
    (a, b)
}

/// Max |h_A - h_B| per timestep over the shared suffix.
fn divergence(lstm: &Lstm, a: &[Vec<i8>], b: &[Vec<i8>]) -> Vec<f32> {
    let ar: Vec<&[i8]> = a.iter().map(Vec::as_slice).collect();
    let br: Vec<&[i8]> = b.iter().map(Vec::as_slice).collect();
    let oa = lstm.forward(&ar).expect("forward A");
    let ob = lstm.forward(&br).expect("forward B");
    oa.iter()
        .zip(&ob)
        .skip(8) // the shared suffix only
        .map(|(la, lb)| {
            la.iter()
                .zip(lb)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        })
        .collect()
}

/// Mean per-lane int8 quantization residual |h - q(h)/127| per timestep —
/// what the stored (recurrent) state DROPPED. The step-2 `free_e` candidate.
fn surprise(lstm: &Lstm, seq: &[Vec<i8>]) -> Vec<f32> {
    let r: Vec<&[i8]> = seq.iter().map(Vec::as_slice).collect();
    lstm.forward(&r)
        .expect("forward")
        .iter()
        .map(|line| {
            line.iter()
                .map(|&h| {
                    let q = (h * 127.0).round().clamp(-127.0, 127.0) / 127.0;
                    (h - q).abs()
                })
                .sum::<f32>()
                / line.len() as f32
        })
        .collect()
}

fn main() {
    let (ni, ns) = (16, 16);

    // RESERVOIR arm — decisively contractive. First cut used four uniform
    // gates at scale 5e-4 and the falsifier below CAUGHT it: the recurrent
    // Jacobian sat near spectral radius 1 (divergence decayed only ~50x over
    // 48 steps — measured, not estimated). The fix is the classic leaky-ESN
    // shape: the forget gate is bias-only and LOW — logistic(-1.6) ~ 0.17
    // constant leak, killing the cell channel in a few steps — while the
    // input/output gates stay small so the h-feedback coupling is weak. The
    // reservoir dynamics live in CI/GI, driven by the input.
    let g = |scale: f64| Gate {
        amp: 24,
        scale,
        bias: 0,
    };
    let reservoir_gates = [
        g(2e-4), // CI
        g(2e-4), // GI
        Gate {
            amp: 0,
            scale: 0.02,
            bias: -80,
        }, // GF1: logistic(-1.6) leak
        g(2e-4), // GO
    ];
    let reservoir_bytes = lstm_wire(ni, ns, reservoir_gates, 0xDEAD_BEEF);
    let (reservoir, used) = Lstm::from_le_bytes(&reservoir_bytes).expect("reservoir loads");
    assert_eq!(
        used,
        reservoir_bytes.len(),
        "loader consumes the whole payload"
    );
    assert_eq!((reservoir.num_inputs(), reservoir.state_size()), (ni, ns));

    // LATCH arm — same CI/GI/GO pattern at low gain, but GF1 is bias-only and
    // saturated high: forget ~ logistic(6) ~ 0.9975, cell = near-pure
    // integrator in the linear zone. Prefix differences must persist.
    let latch_gates = [
        Gate {
            amp: 24,
            scale: 2e-4,
            bias: 0,
        }, // CI
        Gate {
            amp: 24,
            scale: 2e-4,
            bias: 0,
        }, // GI
        Gate {
            amp: 0,
            scale: 0.10,
            bias: 60,
        }, // GF1: logistic(6.0) forget
        Gate {
            amp: 24,
            scale: 2e-4,
            bias: 0,
        }, // GO
    ];
    let latch_bytes = lstm_wire(ni, ns, latch_gates, 0xDEAD_BEEF);
    let (latch, _) = Lstm::from_le_bytes(&latch_bytes).expect("latch loads");

    let (a, b) = washout_pair();
    let dr = divergence(&reservoir, &a, &b);
    let dl = divergence(&latch, &a, &b);
    let (r0, rn) = (dr[0], *dr.last().unwrap());
    let (l0, ln) = (dl[0], *dl.last().unwrap());

    println!("suffix divergence  max|h_A - h_B|   (prefixes differ, suffix shared)");
    println!("  step        reservoir      latch");
    for t in [0, 1, 3, 7, 15, 31, 47] {
        println!("  t={:>3}    {:>12.3e} {:>10.3e}", t, dr[t], dl[t]);
    }

    // Anti-vacuity: the prefixes must actually have separated both arms.
    assert!(
        r0 > 1e-3 && l0 > 1e-3,
        "prefix difference visible at suffix start"
    );
    // Can it fire: the contractive arm forgets its prefix (echo-state).
    assert!(
        rn < 1e-4,
        "reservoir must converge on the shared suffix (got {rn:e})"
    );
    assert!(
        rn < r0 / 100.0,
        "reservoir must shrink >=100x (got {r0:e} -> {rn:e})"
    );
    // Can it stay silent: the latch keeps the difference the reservoir lost.
    assert!(ln > 0.02, "latch must NOT converge (got {ln:e})");

    let s = surprise(&reservoir, &a);
    let mean = s.iter().sum::<f32>() / s.len() as f32;
    println!("\nquantization-residual surprise trace (free_e candidate), reservoir arm:");
    println!(
        "  mean {:.5}  first {:.5}  last {:.5}",
        mean,
        s[0],
        s[s.len() - 1]
    );
    // The residual is real information loss: nonzero, and bounded by the
    // half-step of the int8 grid (0.5/127 per lane).
    assert!(mean > 0.0, "residual must be nonzero");
    assert!(
        s.iter().all(|&x| x <= 0.5 / 127.0 + 1e-6),
        "bounded by the int8 half-step"
    );

    println!("\nOK — echo-state fires, latch stays, OCR path untouched (same kernel, wire-only).");
}
