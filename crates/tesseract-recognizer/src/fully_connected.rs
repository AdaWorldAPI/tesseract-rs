//! `FullyConnected::Forward` (int8 path) — recognizer Leaf 4, the first COMPLETE
//! network layer. Composes the two byte-parity-proven pieces: Leaf 2
//! ([`WeightMatrix::forward`], the int8 `MatrixDotVector`) and Leaf 3
//! ([`crate::activation`], the non-linearities).
//!
//! Tesseract's `FullyConnected::ForwardTimeStep(const int8_t*, …)`
//! (`fullyconnected.cpp:230-234`) is exactly two operations, in order and with NO
//! intermediate step:
//!
//! ```text
//! weights_.MatrixDotVector(i_input, output_line);   // Leaf 2: W·u → f32 line (num_out)
//! ForwardTimeStep(t, output_line);                  // Leaf 3: activation by type_, in place
//! ```
//!
//! where `ForwardTimeStep(t, …)` (`fullyconnected.cpp:203-219`) dispatches the
//! activation on the layer's `NetworkType`:
//!
//! | `NetworkType` | C++ | this crate |
//! |---|---|---|
//! | `NT_TANH` | `FuncInplace<GFunc>` | [`activation::tanh`] |
//! | `NT_LOGISTIC` | `FuncInplace<FFunc>` | [`activation::logistic`] |
//! | `NT_POSCLIP` | `FuncInplace<ClipFFunc>` | [`activation::clip_f`] (clamp `[0,1]`) |
//! | `NT_SYMCLIP` | `FuncInplace<ClipGFunc>` | [`activation::clip_g`] (clamp `[-1,1]`) |
//! | `NT_RELU` | `FuncInplace<Relu>` | [`activation::relu`] |
//! | `NT_SOFTMAX` / `NT_SOFTMAX_NO_CTC` | `SoftmaxInPlace` | [`activation::softmax_in_place`] |
//! | `NT_LINEAR` | (none) | identity (no-op) |
//!
//! Because the two halves are independently proven byte-parity (E-OCR-WEIGHTMATRIX-1,
//! E-OCR-ACTIVATION-1), Leaf 4 proves the **composition**: that the order is
//! `matmul → activation` with no scaling/quant between them. The oracle
//! (`/tmp/fc_oracle.cpp`) confirms it by running the REAL `WeightMatrix::MatrixDotVector`
//! then the REAL `FuncInplace<…>` — the exact two library calls `ForwardTimeStep`
//! makes — on the same bytes.
//!
//! # Structure vs compute
//!
//! Which activation a given `FullyConnected` layer applies is fixed by its
//! `NetworkType` — a *structure* fact the Core (`lance_graph_contract::network`)
//! owns. This crate is the *compute* tier (deps ndarray) and does not depend on
//! the Core; it names its activation modes locally in [`FcActivation`] and maps
//! from the stable `NetworkType` ordinal at the boundary
//! ([`FcActivation::from_network_type_ordinal`]). This is NOT a parallel network
//! model — it is the compute vocabulary of which non-linearity to apply.

use crate::activation;
use crate::{RecognizerError, WeightMatrix};

/// The activation a `FullyConnected` layer applies — the subset of `NetworkType`s
/// that are FullyConnected variants (`fullyconnected.cpp:203-219`). Named locally
/// so the compute crate needs no dependency on the Core's `NetworkType`; the map
/// from the Core's stable ordinal is [`from_network_type_ordinal`].
///
/// [`from_network_type_ordinal`]: FcActivation::from_network_type_ordinal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FcActivation {
    /// `NT_TANH` — `tanh` (`GFunc`).
    Tanh,
    /// `NT_LOGISTIC` — `logistic` (`FFunc`).
    Logistic,
    /// `NT_POSCLIP` — clamp `[0,1]` (`ClipFFunc`).
    PosClip,
    /// `NT_SYMCLIP` — clamp `[-1,1]` (`ClipGFunc`).
    SymClip,
    /// `NT_RELU` — rectifier.
    Relu,
    /// `NT_SOFTMAX` — softmax (with CTC downstream).
    Softmax,
    /// `NT_SOFTMAX_NO_CTC` — softmax (no CTC).
    SoftmaxNoCtc,
    /// `NT_LINEAR` — identity (no non-linearity).
    Linear,
}

impl FcActivation {
    /// Map a `lance_graph_contract::network::NetworkType` ordinal (the stable
    /// on-wire discriminant, `network.h:41-78`) to the activation a
    /// `FullyConnected` of that type applies. `None` for an ordinal that is NOT a
    /// FullyConnected variant (Series / LSTM / Convolve / … are not activations).
    ///
    /// The ordinals are the Core's `NetworkType as u8` — the boundary is the u8,
    /// not a type dependency: `NT_LOGISTIC=16`, `NT_POSCLIP=17`, `NT_SYMCLIP=18`,
    /// `NT_TANH=19`, `NT_RELU=20`, `NT_LINEAR=21`, `NT_SOFTMAX=22`,
    /// `NT_SOFTMAX_NO_CTC=23`.
    #[must_use]
    pub const fn from_network_type_ordinal(ordinal: u8) -> Option<FcActivation> {
        Some(match ordinal {
            16 => FcActivation::Logistic,
            17 => FcActivation::PosClip,
            18 => FcActivation::SymClip,
            19 => FcActivation::Tanh,
            20 => FcActivation::Relu,
            21 => FcActivation::Linear,
            22 => FcActivation::Softmax,
            23 => FcActivation::SoftmaxNoCtc,
            _ => return None,
        })
    }

    /// Apply this activation in place over the `num_out` line — the transcode of
    /// `FullyConnected::ForwardTimeStep(int t, …)` (`fullyconnected.cpp:203-219`).
    /// Element-wise for the point non-linearities; whole-line for softmax
    /// (normalizes over the outputs); no-op for `Linear`.
    pub fn apply(self, line: &mut [f32]) {
        match self {
            FcActivation::Tanh => {
                for x in line.iter_mut() {
                    *x = activation::tanh(*x);
                }
            }
            FcActivation::Logistic => {
                for x in line.iter_mut() {
                    *x = activation::logistic(*x);
                }
            }
            FcActivation::PosClip => {
                for x in line.iter_mut() {
                    *x = activation::clip_f(*x);
                }
            }
            FcActivation::SymClip => {
                for x in line.iter_mut() {
                    *x = activation::clip_g(*x);
                }
            }
            FcActivation::Relu => {
                for x in line.iter_mut() {
                    *x = activation::relu(*x);
                }
            }
            FcActivation::Softmax | FcActivation::SoftmaxNoCtc => {
                activation::softmax_in_place(line);
            }
            FcActivation::Linear => {}
        }
    }
}

/// One int8-input timestep of `FullyConnected::Forward` — `activation(W·u)`, the
/// transcode of `ForwardTimeStep(const int8_t*, …)` (`fullyconnected.cpp:230-234`).
/// Returns the `num_out` activated line. This is the recognizer's int8 hot path
/// (a layer whose input is the int8 output of the prior layer / the quantized
/// image); the float-input path is [`WeightMatrix::forward_f64`] + [`FcActivation::apply`].
///
/// # Errors
///
/// Propagates [`WeightMatrix::forward`]'s dimension / GEMM errors (e.g. `u.len()`
/// != the matrix's `num_inputs`).
pub fn fully_connected_forward(
    weights: &WeightMatrix,
    input: &[i8],
    activation: FcActivation,
) -> Result<Vec<f32>, RecognizerError> {
    let mut line = weights.forward(input)?;
    activation.apply(&mut line);
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_INT8_FLAG: u8 = 1;
    const K_DOUBLE_FLAG: u8 = 128;
    const INT8_MAX_F64: f64 = 127.0;

    // A deterministic int-mode WeightMatrix (the Serialize layout Leaf 2 loads).
    fn wm_bytes(num_out: usize, num_in: usize) -> Vec<u8> {
        let dim2 = num_in + 1;
        let mut b = Vec::new();
        b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
        b.extend_from_slice(&(num_out as u32).to_le_bytes());
        b.extend_from_slice(&(dim2 as u32).to_le_bytes());
        b.push(0); // empty_
        for i in 0..num_out {
            for j in 0..dim2 {
                b.push((((i as i64 * 7 + j as i64 * 3) % 251) - 125) as i8 as u8);
            }
        }
        b.extend_from_slice(&(num_out as u32).to_le_bytes());
        for i in 0..num_out {
            b.extend_from_slice(&((((i % 7) + 1) as f64 * 0.03) * INT8_MAX_F64).to_le_bytes());
        }
        b
    }

    fn input(num_in: usize) -> Vec<i8> {
        (0..num_in)
            .map(|j| (((j as i64 * 5 + 2) % 251) - 125) as i8)
            .collect()
    }

    #[test]
    fn ordinal_map_covers_the_fc_variants_and_rejects_others() {
        // The 8 FullyConnected NetworkType ordinals map; container/recurrent types
        // (Series=9, LSTM=14, Convolve=2, Input=1) do not.
        assert_eq!(
            FcActivation::from_network_type_ordinal(19),
            Some(FcActivation::Tanh)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(16),
            Some(FcActivation::Logistic)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(17),
            Some(FcActivation::PosClip)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(18),
            Some(FcActivation::SymClip)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(20),
            Some(FcActivation::Relu)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(21),
            Some(FcActivation::Linear)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(22),
            Some(FcActivation::Softmax)
        );
        assert_eq!(
            FcActivation::from_network_type_ordinal(23),
            Some(FcActivation::SoftmaxNoCtc)
        );
        for non_fc in [0u8, 1, 2, 9, 14, 15, 24, 26, 27, 255] {
            assert_eq!(FcActivation::from_network_type_ordinal(non_fc), None);
        }
    }

    #[test]
    fn forward_is_matmul_then_activation() {
        // The composition: fully_connected_forward == activation applied to the
        // raw WeightMatrix::forward, with NO intermediate step (the thing Leaf 4
        // proves on top of Leaf 2 + Leaf 3).
        let wm = WeightMatrix::from_le_bytes(&wm_bytes(8, 5)).expect("valid wm");
        let u = input(5);
        let raw = wm.forward(&u).expect("forward");

        for act in [
            FcActivation::Tanh,
            FcActivation::Logistic,
            FcActivation::PosClip,
            FcActivation::SymClip,
            FcActivation::Relu,
            FcActivation::Softmax,
            FcActivation::Linear,
        ] {
            let got = fully_connected_forward(&wm, &u, act).expect("fc forward");
            let mut expect = raw.clone();
            act.apply(&mut expect);
            assert_eq!(got, expect, "{act:?}: forward == activation∘matmul");
            assert_eq!(got.len(), wm.num_outputs());
        }
    }

    #[test]
    fn linear_is_the_raw_line_and_relu_floors_at_zero() {
        let wm = WeightMatrix::from_le_bytes(&wm_bytes(6, 4)).expect("valid wm");
        let u = input(4);
        let linear = fully_connected_forward(&wm, &u, FcActivation::Linear).expect("ok");
        assert_eq!(
            linear,
            wm.forward(&u).expect("forward"),
            "Linear = identity"
        );

        let relu = fully_connected_forward(&wm, &u, FcActivation::Relu).expect("ok");
        assert!(relu.iter().all(|&x| x >= 0.0), "Relu floors at 0");
        for (r, l) in relu.iter().zip(&linear) {
            assert_eq!(*r, l.max(0.0));
        }
    }

    #[test]
    fn softmax_line_is_a_distribution() {
        let wm = WeightMatrix::from_le_bytes(&wm_bytes(10, 7)).expect("valid wm");
        let out = fully_connected_forward(&wm, &input(7), FcActivation::Softmax).expect("ok");
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax normalizes to 1 (got {sum})"
        );
        assert!(out.iter().all(|&x| x > 0.0), "softmax outputs are positive");
    }

    #[test]
    fn dim_mismatch_propagates() {
        let wm = WeightMatrix::from_le_bytes(&wm_bytes(4, 5)).expect("valid wm");
        // Wrong input length → the underlying GEMM's DimMismatch surfaces.
        assert!(fully_connected_forward(&wm, &input(4), FcActivation::Tanh).is_err());
    }
}
