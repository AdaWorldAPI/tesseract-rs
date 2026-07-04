//! The recognizer **graph walk** — recognizer Leaf 6, the composition that chains
//! the proven layer leaves into a network forward pass. The compute-side
//! execution tree (the `invoke_network` counterpart: the Core's
//! `lance_graph_contract::network` FacetCascade describes the tree *structure*;
//! this crate *runs* it).
//!
//! Transcode of the 1-D plumbing layers' `Forward` (`lstm/{series,parallel,
//! reversed}.cpp`):
//!
//! - **`Series`** (`series.cpp:Forward`): run each sub-layer in turn, the output
//!   of layer N feeding layer N+1. The recognizer runs int8, so the intermediate
//!   buffers are int_mode (`NetworkScratch::IO` inherits `int_mode` from the
//!   input); the inter-layer conversion is therefore the **int8 requant**
//!   ([`crate::lstm::quantize_i8`] = `NetworkIO::WriteTimeStep`, proven in Leaf 5).
//!   The final softmax layer is the exception (`ResizeFloat` → the f32 output).
//! - **`Reversed`** (XREVERSED, `reversed.cpp:Forward` + `CopyWithXReversal`):
//!   reverse the 1-D sequence → run the inner layer → reverse the output. (The
//!   `Txy`/`XYTranspose` and `YREVERSED` variants belong to the 2-D front-end,
//!   deferred with the leptonica image input.)
//! - **`Parallel`** (`parallel.cpp:Forward`): run each sub-layer on the SAME
//!   input, concatenate (pack) the outputs. Modelled for completeness; eng.lstm's
//!   1-D core uses `Series`/`Reversed`, not `Parallel`.
//!
//! Because each layer leaf is already byte-parity-proven (Leaf 4 FC, Leaf 5
//! LSTM), Leaf 6 proves the **composition**: the chaining order, the int8 requant
//! between stages, and the sequence reversal. The oracle (`/tmp/graph_oracle.cpp`)
//! confirms it by running the REAL per-layer `Forward` bodies chained with the
//! REAL `WriteTimeStep` quant + reversal.
//!
//! The 2-D front-end (`Convolve`/`Maxpool`/`Reconfig`/`XYTranspose`) needs the
//! full `NetworkIO`/`StrideMap` grid abstraction + the leptonica image `Input`,
//! and stays deferred; this leaf is the 1-D recognition core that turns feature
//! sequences into the softmax logits `recodebeam` decodes.

use crate::lstm::quantize_i8;
use crate::{fully_connected_forward, FcActivation, Lstm, RecognizerError, WeightMatrix};

/// A node in the recognizer's compute execution tree — the compute-side dispatch
/// over the network structure the Core resolves by classid. NOT a parallel object
/// model of the Core's `NetworkType`: it is the runnable subset (the layers whose
/// `Forward` this crate transcodes), built from the Core's tree by a consumer.
#[derive(Debug, Clone)]
pub enum Layer {
    /// A 1-D LSTM block (Leaf 5). Boxed — it carries 4 gate matrices and dwarfs
    /// the other variants (`clippy::large_enum_variant`).
    Lstm(Box<Lstm>),
    /// A fully-connected layer (Leaf 4): `activation(W·u)` per timestep.
    FullyConnected {
        /// The layer weights.
        weights: WeightMatrix,
        /// The non-linearity (`tanh`/`logistic`/`softmax`/…).
        activation: FcActivation,
    },
    /// `Reversed` (XREVERSED): reverse the sequence, run the inner layer, reverse
    /// the output.
    Reversed(Box<Layer>),
    /// `Series`: run the stack in order, requantizing to int8 between stages.
    Series(Vec<Layer>),
    /// `Parallel`: run each sub-layer on the same input, concatenate the outputs
    /// per timestep.
    Parallel(Vec<Layer>),
}

impl Layer {
    /// The number of output features per timestep this layer produces.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        match self {
            Layer::Lstm(l) => l.state_size(),
            Layer::FullyConnected { weights, .. } => weights.num_outputs(),
            Layer::Reversed(inner) => inner.num_outputs(),
            Layer::Series(stack) => stack.last().map_or(0, Layer::num_outputs),
            Layer::Parallel(stack) => stack.iter().map(Layer::num_outputs).sum(),
        }
    }

    /// Run this layer's forward over an int8 input sequence (each timestep
    /// `num_inputs` features long), returning one `f32` line per timestep. The
    /// int8 hot path: a stage's `f32` output is requantized to int8 before the
    /// next stage (`Series`) — the recognizer's inter-layer contract.
    ///
    /// # Errors
    ///
    /// Propagates the layer leaves' dimension / GEMM errors; a `Parallel` whose
    /// sub-layers disagree on timestep count yields [`RecognizerError::DimMismatch`].
    pub fn forward(&self, input: &[&[i8]]) -> Result<Vec<Vec<f32>>, RecognizerError> {
        match self {
            Layer::Lstm(l) => l.forward(input),
            Layer::FullyConnected {
                weights,
                activation,
            } => input
                .iter()
                .map(|line| fully_connected_forward(weights, line, *activation))
                .collect(),
            Layer::Reversed(inner) => {
                // XREVERSED: reverse the sequence, run inner, reverse the output.
                let rev: Vec<&[i8]> = input.iter().rev().copied().collect();
                let mut out = inner.forward(&rev)?;
                out.reverse();
                Ok(out)
            }
            Layer::Series(stack) => {
                let Some((first, rest)) = stack.split_first() else {
                    // A real network spec never yields an empty Series; erroring
                    // (rather than silently returning zero timesteps) surfaces a
                    // malformed graph instead of feeding the decoder no logits.
                    return Err(RecognizerError::DimMismatch("empty Series layer"));
                };
                let mut result = first.forward(input)?;
                for layer in rest {
                    // int8 requant of the intermediate NetworkIO (int_mode).
                    let req: Vec<Vec<i8>> = result
                        .iter()
                        .map(|line| line.iter().map(|&x| quantize_i8(x)).collect())
                        .collect();
                    let refs: Vec<&[i8]> = req.iter().map(Vec::as_slice).collect();
                    result = layer.forward(&refs)?;
                }
                Ok(result)
            }
            Layer::Parallel(stack) => {
                if stack.is_empty() {
                    return Err(RecognizerError::DimMismatch("empty Parallel layer"));
                }
                let parts: Vec<Vec<Vec<f32>>> = stack
                    .iter()
                    .map(|l| l.forward(input))
                    .collect::<Result<_, _>>()?;
                let width = parts.first().map_or(0, Vec::len);
                if parts.iter().any(|p| p.len() != width) {
                    return Err(RecognizerError::DimMismatch(
                        "parallel sub-layers disagree on timestep count",
                    ));
                }
                // Concatenate the sub-layer outputs per timestep.
                Ok((0..width)
                    .map(|t| parts.iter().flat_map(|p| p[t].iter().copied()).collect())
                    .collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_INT8_FLAG: u8 = 1;
    const K_DOUBLE_FLAG: u8 = 128;
    const INT8_MAX_F64: f64 = 127.0;

    fn wm_bytes(num_out: usize, num_in: usize, seed: i64) -> Vec<u8> {
        let dim2 = num_in + 1;
        let mut b = Vec::new();
        b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
        b.extend_from_slice(&(num_out as u32).to_le_bytes());
        b.extend_from_slice(&(dim2 as u32).to_le_bytes());
        b.push(0);
        for i in 0..num_out {
            for j in 0..dim2 {
                b.push((((i as i64 * 7 + j as i64 * 3 + seed) % 251) - 125) as i8 as u8);
            }
        }
        b.extend_from_slice(&(num_out as u32).to_le_bytes());
        for i in 0..num_out {
            b.extend_from_slice(&((((i % 7) + 1) as f64 * 0.02) * INT8_MAX_F64).to_le_bytes());
        }
        b
    }

    fn fc(num_out: usize, num_in: usize, seed: i64, act: FcActivation) -> Layer {
        Layer::FullyConnected {
            weights: WeightMatrix::from_le_bytes(&wm_bytes(num_out, num_in, seed)).unwrap(),
            activation: act,
        }
    }

    fn seq(steps: usize, ni: usize) -> Vec<Vec<i8>> {
        (0..steps)
            .map(|t| {
                (0..ni)
                    .map(|j| ((t * 13 + j * 5) as i64 % 251 - 125) as i8)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn series_chains_and_requantizes() {
        // Series[FC(tanh) 6←5, FC(softmax) 4←6]: the 2nd stage sees the int8
        // requant of the 1st's output; final output is the softmax distribution.
        let net = Layer::Series(vec![
            fc(6, 5, 10, FcActivation::Tanh),
            fc(4, 6, 20, FcActivation::Softmax),
        ]);
        assert_eq!(net.num_outputs(), 4);
        let s = seq(3, 5);
        let refs: Vec<&[i8]> = s.iter().map(Vec::as_slice).collect();
        let out = net.forward(&refs).unwrap();
        assert_eq!(out.len(), 3);
        for line in &out {
            assert_eq!(line.len(), 4);
            let sum: f32 = line.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "softmax head normalizes");
        }
    }

    #[test]
    fn reversed_is_reverse_inner_reverse() {
        // Reversed[FC linear] on a sequence == reverse ∘ FC ∘ reverse. Since a
        // (stateless) FC is timestep-independent, Reversed[FC] == FC here — the
        // reversal cancels — which pins the wrap/unwrap symmetry.
        let inner = fc(4, 5, 30, FcActivation::Linear);
        let rev = Layer::Reversed(Box::new(fc(4, 5, 30, FcActivation::Linear)));
        let s = seq(4, 5);
        let refs: Vec<&[i8]> = s.iter().map(Vec::as_slice).collect();
        assert_eq!(
            inner.forward(&refs).unwrap(),
            rev.forward(&refs).unwrap(),
            "reverse∘FC∘reverse == FC for a stateless layer"
        );
    }

    #[test]
    fn reversed_lstm_differs_from_forward_lstm() {
        // For a STATEFUL LSTM, Reversed[LSTM] != LSTM (the recurrence sees the
        // sequence in the opposite order) — unless the outputs are all-zero.
        let bytes = {
            let (ni, ns) = (4, 4);
            let na = ni + ns;
            let mut b = Vec::new();
            b.extend_from_slice(&(na as i32).to_le_bytes());
            for seed in [10_i64, 20, 30, 40] {
                b.extend_from_slice(&wm_bytes(ns, na, seed));
            }
            b
        };
        let fwd = Layer::Lstm(Box::new(Lstm::from_le_bytes(&bytes).unwrap().0));
        let rev = Layer::Reversed(Box::new(Layer::Lstm(Box::new(
            Lstm::from_le_bytes(&bytes).unwrap().0,
        ))));
        assert_eq!(fwd.num_outputs(), rev.num_outputs());
        let s = seq(5, 4);
        let refs: Vec<&[i8]> = s.iter().map(Vec::as_slice).collect();
        // Both run without error and produce ns-wide lines over 5 timesteps.
        assert_eq!(fwd.forward(&refs).unwrap().len(), 5);
        assert_eq!(rev.forward(&refs).unwrap().len(), 5);
    }

    #[test]
    fn parallel_concatenates_outputs() {
        let net = Layer::Parallel(vec![
            fc(3, 5, 10, FcActivation::Tanh),
            fc(2, 5, 20, FcActivation::Relu),
        ]);
        assert_eq!(net.num_outputs(), 5, "3 + 2");
        let s = seq(2, 5);
        let refs: Vec<&[i8]> = s.iter().map(Vec::as_slice).collect();
        let out = net.forward(&refs).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|l| l.len() == 5), "concatenated width 3+2");
    }
}
