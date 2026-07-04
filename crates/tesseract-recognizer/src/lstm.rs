//! `LSTM::Forward` (1-D int8 path) — recognizer Leaf 5, the recurrent layer.
//! Reuses Leaf 4 ([`fully_connected_forward`]) for its four gates and adds the
//! new pieces: the cell/hidden **state recurrence** and the int8 quantization of
//! the recurrent feedback.
//!
//! Tesseract's `LSTM::Forward` (`lstm.cpp:291-503`) runs, per timestep `t`, over
//! a 1-D non-softmax block (the eng.lstm case — `nf_ = 0`, `is_2d_ = false`):
//!
//! ```text
//! source = [ input(ni) | int8_quantize(prev_output)(ns) ]      // lstm.cpp:363-369
//! CI  = tanh   (W_CI  · source)     // GateType::CI,  FuncInplace<GFunc>  (l.381-385)
//! GI  = logistic(W_GI · source)     //          GI,  FuncInplace<FFunc>  (l.390-394)
//! GF1 = logistic(W_GF1· source)     //          GF1, FuncInplace<FFunc>  (l.399-403)
//! GO  = logistic(W_GO · source)     //          GO,  FuncInplace<FFunc>  (l.418-422)
//! c  *= GF1                          // MultiplyVectorsInPlace  (forget)  (l.426)
//! c  += CI * GI                      // MultiplyAccumulate      (input)   (l.441)
//! c   = clip(c, -100, 100)           // ClipVector, kStateClip=100        (l.443)
//! h   = tanh(c) * GO                 // FuncMultiply<HFunc>, HFunc=Tanh   (l.454)
//! output[t] = h;  h feeds the NEXT timestep's source                     (l.476, 367)
//! ```
//!
//! Each gate is exactly `MatrixDotVector` + `FuncInplace` — i.e.
//! [`fully_connected_forward`] with the gate's activation (`FullyConnected` is
//! "reused inside LSTM", `fullyconnected.cpp:189`). What Leaf 5 NEWLY proves is
//! the recurrence: the gate order (`CI/GI/GF1/GO`), the `c = f·c + i·g` cell
//! update with the ±100 clip, the `h = tanh(c)·o` output, and — the crux of the
//! int8 path — that the recurrent `h` is **quantized back to int8** before the
//! next timestep's gate matmuls (`NetworkIO::WriteTimeStepPart`,
//! `networkio.cpp:662-666`: `clip(IntCastRounded(x·127), -127, 127)`).
//!
//! # Serialized form (`LSTM::DeSerialize`, `lstm.cpp:253-287`)
//!
//! The base `Network` header is read first by the factory
//! (`lance_graph_contract::network::NetworkHeader` — gives `ni`); the LSTM
//! payload that follows is `i32 na_` then the four gate `WeightMatrix`es
//! (`CI, GI, GF1, GO`; `GFS` is skipped for 1-D) serialized back-to-back. `ns`
//! (the state size) `= CI.num_outputs`; for a 1-D non-softmax LSTM
//! `na_ = ni + ns`, so `ni = na_ − ns`.

use crate::{activation, fully_connected_forward, FcActivation, RecognizerError, WeightMatrix};

/// `kStateClip` (`lstm.cpp:71`) — the cell-state is clipped to `[-100, 100]`.
const K_STATE_CLIP: f32 = 100.0;

/// A loaded 1-D `LSTM` block: the four gate weight matrices + the derived dims.
/// The transcode of the load + forward side of Tesseract's `LSTM`
/// (`lstm/lstm.{h,cpp}`), 1-D non-softmax path (`nf_ = 0`, `is_2d_ = false`).
#[derive(Debug, Clone)]
pub struct Lstm {
    /// Cell-input gate weights (`GateType::CI`) — `tanh` activation.
    ci: WeightMatrix,
    /// Input gate weights (`GateType::GI`) — `logistic`.
    gi: WeightMatrix,
    /// 1-D forget gate weights (`GateType::GF1`) — `logistic`.
    gf1: WeightMatrix,
    /// Output gate weights (`GateType::GO`) — `logistic`.
    go: WeightMatrix,
    /// Number of inputs per timestep (`ni_` = `na_ − ns`).
    ni: usize,
    /// State size (`ns_` = `CI.num_outputs`).
    ns: usize,
    /// Augmented input size (`na_` = `ni + ns`, the gate matmul width).
    na: usize,
}

impl Lstm {
    /// Parse the `LSTM` payload (the bytes AFTER the base `Network` header) —
    /// `i32 na_` then the four gate `WeightMatrix`es (`CI, GI, GF1, GO`), the
    /// transcode of `LSTM::DeSerialize` (`lstm.cpp:253-287`) for the 1-D
    /// non-softmax path. Returns the block and the number of bytes consumed.
    ///
    /// # Errors
    ///
    /// [`RecognizerError::UnexpectedEof`] on a truncated buffer;
    /// [`RecognizerError::DimMismatch`] for a negative `na_`, `na_ < ns`, or a
    /// gate whose shape is not `ns × (na_+1)`; propagates
    /// [`WeightMatrix::from_le_bytes_prefix`]'s format errors.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<(Self, usize), RecognizerError> {
        let na_bytes: [u8; 4] = bytes
            .get(0..4)
            .ok_or(RecognizerError::UnexpectedEof)?
            .try_into()
            .map_err(|_| RecognizerError::UnexpectedEof)?;
        let na = i32::from_le_bytes(na_bytes);
        if na < 0 {
            return Err(RecognizerError::DimMismatch("negative na_"));
        }
        let na = na as usize;
        let mut off = 4;
        let (ci, c) = WeightMatrix::from_le_bytes_prefix(&bytes[off..])?;
        off += c;
        let (gi, c) = WeightMatrix::from_le_bytes_prefix(&bytes[off..])?;
        off += c;
        let (gf1, c) = WeightMatrix::from_le_bytes_prefix(&bytes[off..])?;
        off += c;
        let (go, c) = WeightMatrix::from_le_bytes_prefix(&bytes[off..])?;
        off += c;

        let ns = ci.num_outputs();
        // na_ = ni + nf + ns with nf = 0 for a plain NT_LSTM (lstm.cpp:262).
        if ns == 0 || na < ns {
            return Err(RecognizerError::DimMismatch("na_ < ns"));
        }
        let ni = na - ns;
        for w in [&ci, &gi, &gf1, &go] {
            if w.num_outputs() != ns || w.num_inputs() != na {
                return Err(RecognizerError::DimMismatch("gate is not ns × (na_+1)"));
            }
        }
        Ok((
            Self {
                ci,
                gi,
                gf1,
                go,
                ni,
                ns,
                na,
            },
            off,
        ))
    }

    /// Number of inputs per timestep.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.ni
    }

    /// State size (`ns`) — the per-timestep output width.
    #[must_use]
    pub fn state_size(&self) -> usize {
        self.ns
    }

    /// Run the 1-D int8 forward recurrence over a sequence of int8 input
    /// timesteps (each `num_inputs()` long), returning one `state_size()`-long
    /// `f32` output line per timestep. Transcode of `LSTM::Forward`
    /// (`lstm.cpp:350-489`, 1-D non-softmax path); `curr_state`/`curr_output`
    /// start at zero (`l.315-317`).
    ///
    /// # Errors
    ///
    /// [`RecognizerError::DimMismatch`] if any input timestep is not
    /// `num_inputs()` long; propagates the gate GEMM errors.
    pub fn forward(&self, inputs: &[&[i8]]) -> Result<Vec<Vec<f32>>, RecognizerError> {
        let (ni, ns, na) = (self.ni, self.ns, self.na);
        let mut curr_state = vec![0.0_f32; ns];
        let mut curr_output = vec![0.0_f32; ns];
        let mut source = vec![0_i8; na];
        let mut outputs = Vec::with_capacity(inputs.len());

        for &input in inputs {
            if input.len() != ni {
                return Err(RecognizerError::DimMismatch("lstm input len != ni"));
            }
            // source = [ input(ni) | int8_quantize(prev_output)(ns) ]  (nf_ = 0).
            source[..ni].copy_from_slice(input);
            for i in 0..ns {
                source[ni + i] = quantize_i8(curr_output[i]);
            }
            // The four gates: activation(W·source) — Leaf 4 per gate.
            let ci = fully_connected_forward(&self.ci, &source, FcActivation::Tanh)?;
            let gi = fully_connected_forward(&self.gi, &source, FcActivation::Logistic)?;
            let gf1 = fully_connected_forward(&self.gf1, &source, FcActivation::Logistic)?;
            let go = fully_connected_forward(&self.go, &source, FcActivation::Logistic)?;

            // Cell: c = clip(GF1·c + CI·GI, ±100), in the C++ op order.
            for i in 0..ns {
                curr_state[i] *= gf1[i]; // MultiplyVectorsInPlace  (forget)
                curr_state[i] += ci[i] * gi[i]; // MultiplyAccumulate (input)
                curr_state[i] = curr_state[i].clamp(-K_STATE_CLIP, K_STATE_CLIP);
            }
            // Output: h = tanh(c) · GO   (FuncMultiply<HFunc>, HFunc = Tanh).
            for i in 0..ns {
                curr_output[i] = activation::tanh(curr_state[i]) * go[i];
            }
            outputs.push(curr_output.clone());
        }
        Ok(outputs)
    }
}

/// Quantize a recurrent-output `f32` (in `[-1, 1]`) to int8 for the next
/// timestep's gate matmuls — the transcode of `NetworkIO::WriteTimeStepPart`
/// int-mode (`networkio.cpp:662-666`): `clip(IntCastRounded(x·127), -127, 127)`.
/// `IntCastRounded` rounds **half away from zero** (`helpers.h:189`), and the
/// clip is `[-INT8_MAX, INT8_MAX]` — never `-128`.
#[inline]
pub(crate) fn quantize_i8(x: f32) -> i8 {
    const INT8_MAX: f32 = 127.0;
    let scaled = x * INT8_MAX;
    // (int)(x + 0.5) truncates toward zero == round half away from zero here.
    let rounded: i32 = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        -((-scaled + 0.5) as i32)
    };
    rounded.clamp(-127, 127) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_INT8_FLAG: u8 = 1;
    const K_DOUBLE_FLAG: u8 = 128;
    const INT8_MAX_F64: f64 = 127.0;

    // A deterministic int-mode WeightMatrix (num_out × (num_in+1)) in wire form.
    fn wm_bytes(num_out: usize, num_in: usize, seed: i64) -> Vec<u8> {
        let dim2 = num_in + 1;
        let mut b = Vec::new();
        b.push(K_INT8_FLAG | K_DOUBLE_FLAG);
        b.extend_from_slice(&(num_out as u32).to_le_bytes());
        b.extend_from_slice(&(dim2 as u32).to_le_bytes());
        b.push(0); // empty_
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

    // A full LSTM payload: i32 na_ then the 4 gate matrices (all ns × (na+1)).
    fn lstm_bytes(ni: usize, ns: usize) -> Vec<u8> {
        let na = ni + ns;
        let mut b = Vec::new();
        b.extend_from_slice(&(na as i32).to_le_bytes());
        for (g, seed) in [0, 1, 2, 3].into_iter().zip([10, 20, 30, 40]) {
            let _ = g;
            b.extend_from_slice(&wm_bytes(ns, na, seed));
        }
        b
    }

    #[test]
    fn deserialize_derives_dims_and_consumes_all() {
        let (ni, ns) = (6, 4);
        let bytes = lstm_bytes(ni, ns);
        let (lstm, consumed) = Lstm::from_le_bytes(&bytes).expect("valid lstm");
        assert_eq!(lstm.num_inputs(), ni, "ni = na - ns");
        assert_eq!(lstm.state_size(), ns, "ns = CI.num_outputs");
        assert_eq!(lstm.na, ni + ns);
        assert_eq!(
            consumed,
            bytes.len(),
            "the 4 gates consume the whole payload"
        );
    }

    #[test]
    fn quantize_matches_networkio_writetimesteppart() {
        // clip(round_half_away(x·127), -127, 127); never -128.
        assert_eq!(quantize_i8(0.0), 0);
        assert_eq!(quantize_i8(1.0), 127);
        assert_eq!(quantize_i8(-1.0), -127);
        assert_eq!(quantize_i8(2.0), 127, "clips at +127");
        assert_eq!(quantize_i8(-2.0), -127, "clips at -127, never -128");
        // round half away from zero: 0.5/127 · 127 = 0.5 → 1; -0.5/127 → -1.
        assert_eq!(quantize_i8(0.5 / 127.0), 1);
        assert_eq!(quantize_i8(-0.5 / 127.0), -1);
        assert_eq!(quantize_i8(0.4 / 127.0), 0, "0.4 rounds to 0");
    }

    #[test]
    fn forward_shapes_and_output_range() {
        let (ni, ns) = (5, 3);
        let (lstm, _) = Lstm::from_le_bytes(&lstm_bytes(ni, ns)).expect("valid");
        let t0: Vec<i8> = (0..ni).map(|j| (j as i64 * 5 - 10) as i8).collect();
        let t1: Vec<i8> = (0..ni).map(|j| (j as i64 * 3 + 7) as i8).collect();
        let seq = [t0.as_slice(), t1.as_slice()];
        let out = lstm.forward(&seq).expect("forward");
        assert_eq!(out.len(), 2, "one line per timestep");
        for line in &out {
            assert_eq!(line.len(), ns);
            // h = tanh(c)·sigmoid(·) ∈ (-1, 1).
            assert!(line.iter().all(|&x| x.abs() < 1.0), "outputs in (-1,1)");
        }
    }

    #[test]
    fn forward_is_deterministic() {
        // The forward is a pure function of (weights, sequence): two runs match
        // bit-for-bit. (The recurrence *math* — that t1 consumes t0's quantized
        // output — is proven by the byte-parity oracle, `/tmp/lstm_oracle.cpp`;
        // an all-zero first output on synthetic small-scale weights can't prove
        // statefulness here, but the feedback path IS wired: source[ni..] =
        // quantize_i8(curr_output).)
        let (ni, ns) = (4, 4);
        let (lstm, _) = Lstm::from_le_bytes(&lstm_bytes(ni, ns)).expect("valid");
        let t: Vec<i8> = (0..ni).map(|j| (j as i64 * 11 - 5) as i8).collect();
        let seq = [t.as_slice(), t.as_slice(), t.as_slice()];
        let a = lstm.forward(&seq).expect("forward");
        let b = lstm.forward(&seq).expect("forward");
        assert_eq!(a, b, "forward is deterministic");
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn rejects_wrong_input_len_and_truncation() {
        let (lstm, _) = Lstm::from_le_bytes(&lstm_bytes(5, 3)).expect("valid");
        let bad = [1_i8, 2, 3]; // len 3 != ni 5
        assert_eq!(
            lstm.forward(&[&bad]).unwrap_err(),
            RecognizerError::DimMismatch("lstm input len != ni")
        );
        assert_eq!(
            Lstm::from_le_bytes(&[0, 0]).unwrap_err(),
            RecognizerError::UnexpectedEof
        );
    }
}
