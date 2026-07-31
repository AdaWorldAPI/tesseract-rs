//! Domino vs reservoir — the two `CognitiveWork` candidate dynamics, head to
//! head, both through the PROVEN kernel. Step 2 of the baby-step ladder
//! (step 1: `reservoir_side_by_side.rs`).
//!
//! The substrate's kanban `CognitiveWork` phase currently runs a domino sweep:
//! an UNGATED linear map with int8 requant feedback (`symbiont/domino.rs`,
//! `C = A·W` on bf16 AMX tiles, C re-quantised back into the tiles). The
//! candidate replacement is a GATED recurrence — the byte-parity LSTM kernel
//! driven as a fixed-weight reservoir. This probe runs BOTH dynamics on the
//! same 16-board × 16-lane Morton-tile stream and measures the one property
//! that separates them:
//!
//!   what can each do with a PERTURBATION (a one-timestep input difference)?
//!
//! **Ungated linear + requant** (domino shape, `FcActivation::Linear` through
//! the proven `fully_connected_forward`, identity-diagonal recurrent block so
//! the gain is a clean scalar knob):
//!
//! ```text
//! gain 0.4  forgets: the perturbation decays below the int8 grid and dies
//!           (gain * 1 LSB < 0.5 LSB cannot re-round).
//! gain 0.8  LSB GHOST (measured, then understood): the perturbation locks at
//!           a few int8 LSB forever - gain * dq >= 0.5 LSB re-rounds to the
//!           same dq, so the requant feedback sustains a quantization-scale
//!           difference that never decays and never grows. Spurious memory at
//!           the grid scale, for any gain in (0.5, 1). The first cut asserted
//!           "0.8 forgets" and the falsifier caught the flatline at exactly
//!           0.8 * 4/127.
//! gain 1.0  knife-edge (measured and printed, not asserted - not a robust
//!           operating point; any drift falls off either side).
//! gain 1.25 RAILS: the f32 state escapes (-1, 1) and the +/-127 clamp does
//!           real, value-destroying clipping - the only high-gain "memory".
//! ```
//!
//! **Gated recurrence** (the proven `Lstm::forward`, fixed seeded weights):
//!
//! ```text
//! leak   forgets - step 1's echo-state result, reconfirmed at board scale.
//! latch  HOLDS the perturbation BOUNDED: every f32 state stays inside
//!        (-1, 1) by construction (h = tanh * logistic), so the int8 clamp
//!        NEVER engages - memory without the rail, the regime the ungated
//!        map does not have. (|q| may still print high: tanh compresses
//!        toward +/-1; that is not the clamp working - the assert is on the
//!        PRE-quant state.) And no LSB ghost: the gated loop gain at the LSB
//!        scale is far below the 0.5-LSB re-rounding threshold, and the cell
//!        channel decays in f32 with no rounding in the loop.
//! ```
//!
//! Plus the step-2 readout: the int8 quantization residual per board,
//! quantised to the 6-bit range MetaWord's `free_e` field carries (the
//! contract type is NOT re-implemented here — this computes the VALUE that
//! `MetaWord::new(.., free_e)` will receive at wiring time, in lance-graph,
//! which is the operator's channel). Falsifier: the residual must
//! DISCRIMINATE — boards receiving fresh input carry more surprise than
//! boards whose input froze. An awareness signal that reads the same for
//! active and quiet boards carries no information.
//!
//! Same rules as step 1: nothing in `src/` changes; both arms consume only
//! the public proven surface; weights are fixed/seeded/never trained; the
//! wire-synthesis helpers are duplicated from `reservoir_side_by_side.rs`
//! on purpose (an example must not grow the crate's API to share 60 lines).
//! Domino's REAL sweep runs on ndarray bf16 AMX tiles; this probe runs the
//! same-shape dynamics on the proven int8 path — the comparison is about
//! GATING, not throughput.
//!
//! Run: `cargo run -p tesseract-recognizer --example domino_vs_reservoir`

use tesseract_recognizer::{fully_connected_forward, FcActivation, Lstm, WeightMatrix};

/// One board's input stream: `STEPS` timesteps of 16-lane int8 tiles.
type BoardSeq = Vec<Vec<i8>>;
/// One board's state trace: `STEPS` lines of 16 f32 lanes.
type Trace = Vec<Vec<f32>>;

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
const LANES: usize = 16; // one 4x4 Morton tile
const BOARDS: usize = 16; // one AMX tile-batch worth
const STEPS: usize = 49; // t=0 is the perturbed step; 48 observed steps

/// `NetworkIO::WriteTimeStepPart` int-mode quant — same math as the kernel's
/// (pub(crate)) `quantize_i8`; restated here because an example sees only the
/// public surface.
fn q(x: f32) -> i8 {
    let s = x * 127.0;
    let r: i32 = if s >= 0.0 {
        (s + 0.5) as i32
    } else {
        -((-s + 0.5) as i32)
    };
    r.clamp(-127, 127) as i8
}

#[derive(Clone, Copy)]
struct Gate {
    amp: i8,
    scale: f64,
    bias: i8,
}

/// Int-mode `WeightMatrix` wire bytes (`ns x (na+1)`), random block.
fn wm_wire(ns: usize, na: usize, g: Gate, rng: &mut Rng) -> Vec<u8> {
    let dim2 = na + 1;
    let mut b = Vec::new();
    b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
    b.extend_from_slice(&(ns as u32).to_le_bytes());
    b.extend_from_slice(&(dim2 as u32).to_le_bytes());
    b.push(0);
    for _row in 0..ns {
        for col in 0..dim2 {
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

/// The DOMINO-arm weight matrix: input block random, recurrent block the
/// IDENTITY times `w0` — so the recurrent gain is the clean scalar
/// `gain = w0 · 127 · scale`, the knife-edge knob under test. (Domino's real
/// W mixes lanes; mixing has a spectral radius and the same knife-edge —
/// the diagonal only isolates the knob.)
fn domino_wm(gain: f64, rng: &mut Rng) -> WeightMatrix {
    let na = 2 * LANES;
    let w0: i8 = 100;
    let scale = gain / (f64::from(w0) * 127.0);
    let dim2 = na + 1;
    let mut b = Vec::new();
    b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
    b.extend_from_slice(&(LANES as u32).to_le_bytes());
    b.extend_from_slice(&(dim2 as u32).to_le_bytes());
    b.push(0);
    for row in 0..LANES {
        for col in 0..dim2 {
            let w: i8 = if col < LANES {
                rng.i8_amp(24) // input coupling
            } else if col == LANES + row {
                w0 // identity recurrence
            } else {
                0
            };
            b.push(w as u8);
        }
    }
    b.extend_from_slice(&(LANES as u32).to_le_bytes());
    for _ in 0..LANES {
        b.extend_from_slice(&(scale * 127.0).to_le_bytes());
    }
    WeightMatrix::from_le_bytes(&b).expect("domino wm loads")
}

/// The LSTM payload (i32 na_ + 4 gates), consumed by the proven loader.
fn lstm_wire(gates: [Gate; 4], seed: u64) -> Vec<u8> {
    let na = 2 * LANES;
    let mut rng = Rng(seed);
    let mut b = Vec::new();
    b.extend_from_slice(&(na as i32).to_le_bytes());
    for g in gates {
        b.extend_from_slice(&wm_wire(LANES, na, g, &mut rng));
    }
    b
}

/// Per-board input stream: boards 0..7 DRIVEN (fresh tile each step), boards
/// 8..15 FROZEN (zero input after t=0). `perturbed` swaps board inputs at
/// t=0 only — the perturbation whose fate both dynamics are judged on.
fn inputs(perturbed: bool) -> Vec<BoardSeq> {
    (0..BOARDS)
        .map(|b| {
            let mut r = Rng(0x1000 + b as u64);
            let mut rp = Rng(0x9000 + b as u64);
            (0..STEPS)
                .map(|t| {
                    if t == 0 && perturbed {
                        // Burn the base draw so `r`'s state — and therefore the
                        // ENTIRE shared suffix — is identical in both runs.
                        // (The first cut skipped this and the falsifier caught
                        // it immediately: every driven step after t=0 differed,
                        // so nothing could ever "forget".)
                        let _: Vec<i8> = (0..LANES).map(|_| r.i8_amp(50)).collect();
                        return (0..LANES).map(|_| rp.i8_amp(50)).collect();
                    }
                    if t == 0 || b < 8 {
                        (0..LANES).map(|_| r.i8_amp(50)).collect()
                    } else {
                        vec![0_i8; LANES] // frozen board, quiet after t=0
                    }
                })
                .collect()
        })
        .collect()
}

/// The domino dynamic: `state' = Linear(W · [x | q(state)])` — ungated linear
/// map + int8 requant feedback, everything through the proven forward.
fn run_domino(w: &WeightMatrix, seq: &[Vec<i8>]) -> Trace {
    let mut state = vec![0.0_f32; LANES];
    let mut src = vec![0_i8; 2 * LANES];
    let mut out = Vec::with_capacity(seq.len());
    for x in seq {
        src[..LANES].copy_from_slice(x);
        for i in 0..LANES {
            src[LANES + i] = q(state[i]);
        }
        state = fully_connected_forward(w, &src, FcActivation::Linear).expect("fc forward");
        out.push(state.clone());
    }
    out
}

/// Max |a - b| over boards and lanes at each observed step (t >= 1).
fn divergence(a: &[Trace], b: &[Trace]) -> Vec<f32> {
    (1..STEPS)
        .map(|t| {
            (0..BOARDS)
                .map(|bd| {
                    a[bd][t]
                        .iter()
                        .zip(&b[bd][t])
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0_f32, f32::max)
                })
                .fold(0.0_f32, f32::max)
        })
        .collect()
}

/// Largest |q(state)| seen anywhere — 127 means the int8 rail.
fn max_rail(states: &[Trace]) -> i32 {
    states
        .iter()
        .flatten()
        .flatten()
        .map(|&x| i32::from(q(x)).abs())
        .max()
        .unwrap_or(0)
}

/// Largest PRE-quant |state| — the honest rail metric. The int8 rail only
/// destroys value when the f32 state has escaped (-1, 1) and the ±127 clamp
/// is doing real clipping. A gated `h = tanh(c)·logistic(·)` can PRINT a high
/// |q| (tanh compresses toward ±1) while the clamp never engages — the first
/// cut asserted on |q| < 120 for the latch and the falsifier caught exactly
/// that misreading (latch |q| hit 120 with every f32 state still inside the
/// band).
fn max_abs(states: &[Trace]) -> f32 {
    states
        .iter()
        .flatten()
        .flatten()
        .fold(0.0_f32, |m, &x| m.max(x.abs()))
}

/// Mean 6-bit free_e per board over the run: the int8 quantization residual
/// |h - q(h)/127|, mapped onto [0, 63] against its own bound (the half-step
/// 0.5/127). This is the VALUE destined for `MetaWord::new(.., free_e)`.
fn free_e_6bit(states: &[Trace]) -> Vec<f32> {
    let half_step = 0.5 / 127.0;
    (0..BOARDS)
        .map(|b| {
            let per_step: Vec<f32> = states[b]
                .iter()
                .map(|line| {
                    let r = line
                        .iter()
                        .map(|&h| (h - f32::from(q(h)) / 127.0).abs())
                        .sum::<f32>()
                        / line.len() as f32;
                    ((r / half_step) * 63.0).min(63.0)
                })
                .collect();
            per_step.iter().sum::<f32>() / per_step.len() as f32
        })
        .collect()
}

fn main() {
    let base = inputs(false);
    let pert = inputs(true);
    let run_all = |f: &dyn Fn(&[Vec<i8>]) -> Trace, set: &[BoardSeq]| {
        set.iter().map(|s| f(s)).collect::<Vec<Trace>>()
    };

    // ── the ungated arm across the four gain regimes ──────────────────────
    println!("perturbation fate  max|state_A - state_B|  (one-timestep input difference)");
    println!("  arm                     t=1        t=8       t=24       t=48   rail  max|s|");
    let mut knife_edge = 0.0_f32;
    let lsb = 1.0_f32 / 127.0;
    for gain in [0.4_f64, 0.8, 1.0, 1.25] {
        let w = domino_wm(gain, &mut Rng(0xD0));
        let a = run_all(&|s| run_domino(&w, s), &base);
        let b = run_all(&|s| run_domino(&w, s), &pert);
        let d = divergence(&a, &b);
        let rail = max_rail(&a);
        let mx = max_abs(&a);
        println!(
            "  domino gain {gain:<4}   {:>9.3e} {:>9.3e} {:>9.3e} {:>9.3e}   {rail:>4}  {mx:>6.2}",
            d[0], d[7], d[23], d[47]
        );
        match gain {
            g if g < 0.5 => {
                // can it fire (forget): gain·1 LSB < 0.5 LSB cannot re-round,
                // so the difference falls off the grid and dies completely.
                assert!(d[47] < 1e-3, "gain 0.4 must forget (got {:e})", d[47]);
                assert!(rail < 120, "gain 0.4 must not rail");
            }
            g if g < 1.0 => {
                // the LSB ghost: locked at a few grid steps — neither decaying
                // (that would need gain·Δq to round DOWN) nor growing.
                assert!(
                    d[47] > 0.5 * lsb && d[47] < 8.0 * lsb,
                    "gain 0.8 must hold a quantization-scale ghost (got {:e})",
                    d[47]
                );
                assert!(rail < 120, "gain 0.8 must not rail");
            }
            g if g > 1.0 => {
                // the only high-gain "memory" is the rail: the f32 state has
                // escaped the representable band and the ±127 clamp is doing
                // real, value-destroying clipping.
                assert!(rail == 127, "gain 1.25 must pin the int8 rail (got {rail})");
                assert!(mx > 1.5, "gain 1.25 must explode past the band (got {mx})");
            }
            _ => knife_edge = d[47], // measured, printed, NOT asserted
        }
        assert!(
            d[0] > 1e-3,
            "anti-vacuity: the perturbation must register at t=1"
        );
    }

    // ── the gated arm, two configs, SAME kernel ───────────────────────────
    let g = |scale: f64| Gate {
        amp: 24,
        scale,
        bias: 0,
    };
    let leak = [
        g(2e-4),
        g(2e-4),
        Gate {
            amp: 0,
            scale: 0.02,
            bias: -80,
        },
        g(2e-4),
    ];
    let latch = [
        Gate {
            amp: 24,
            scale: 2e-4,
            bias: 0,
        },
        Gate {
            amp: 24,
            scale: 2e-4,
            bias: 0,
        },
        Gate {
            amp: 0,
            scale: 0.10,
            bias: 60,
        },
        Gate {
            amp: 24,
            scale: 2e-4,
            bias: 0,
        },
    ];
    let mut lstm_results = Vec::new();
    for (name, gates) in [("leak", leak), ("latch", latch)] {
        let (lstm, _) = Lstm::from_le_bytes(&lstm_wire(gates, 0xDEAD_BEEF)).expect("lstm loads");
        let fwd = |s: &[Vec<i8>]| {
            let r: Vec<&[i8]> = s.iter().map(Vec::as_slice).collect();
            lstm.forward(&r).expect("lstm forward")
        };
        let a = run_all(&fwd, &base);
        let b = run_all(&fwd, &pert);
        let d = divergence(&a, &b);
        let rail = max_rail(&a);
        let mx = max_abs(&a);
        println!(
            "  lstm  {name:<10}   {:>9.3e} {:>9.3e} {:>9.3e} {:>9.3e}   {rail:>4}  {mx:>6.2}",
            d[0], d[7], d[23], d[47]
        );
        lstm_results.push((name, d, mx, a));
    }
    let (_, d_leak, _, a_leak) = &lstm_results[0];
    let (_, d_latch, mx_latch, _) = &lstm_results[1];

    // Gated forget: same result as step 1, now at board scale.
    assert!(
        d_leak[47] < 1e-4,
        "lstm leak must forget (got {:e})",
        d_leak[47]
    );
    // THE differential: bounded persistence. The latch holds the perturbation
    // within an order of magnitude WITHOUT touching the rail — the regime the
    // ungated map does not possess at any gain.
    assert!(
        d_latch[47] > d_latch[0] / 10.0 && d_latch[47] > 0.02,
        "lstm latch must hold the perturbation (t1 {:e} -> t48 {:e})",
        d_latch[0],
        d_latch[47]
    );
    // Bounded BY CONSTRUCTION: h = tanh(c)·logistic(·) never leaves (-1, 1),
    // so the ±127 clamp never engages — memory without the clamp's help.
    assert!(
        *mx_latch < 1.0,
        "lstm latch must stay inside the band (got {mx_latch})"
    );

    // ── the free_e readout must DISCRIMINATE ──────────────────────────────
    let fe = free_e_6bit(a_leak);
    let driven = fe[..8].iter().sum::<f32>() / 8.0;
    let frozen = fe[8..].iter().sum::<f32>() / 8.0;
    println!(
        "\nfree_e (6-bit, MetaWord-bound): driven boards {driven:.2}  frozen boards {frozen:.2}"
    );
    println!(
        "domino knife-edge d[48] at gain 1.0 (measured, not an operating point): {knife_edge:.3e}"
    );
    assert!(
        driven > 2.0 * frozen,
        "free_e must discriminate driven from frozen boards"
    );
    assert!(
        driven > 1.0,
        "driven boards must register surprise on the 6-bit scale"
    );

    println!("\nOK — ungated: forget / LSB-ghost / knife-edge / rail; gated: clean forget or bounded hold. free_e discriminates.");
}
