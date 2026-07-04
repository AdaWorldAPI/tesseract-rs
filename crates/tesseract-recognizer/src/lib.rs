//! # tesseract-recognizer — the compute tier of the pure-Rust Tesseract transcode
//!
//! The recoder / unicharset leaves were codec **tables** (the zero-dep content
//! tier in `lance-graph-contract`). The recognizer is **compute**: Tesseract's
//! LSTM forward pass is dense int8 GEMM. Per the two-foundations architecture
//! this crate consumes **ndarray** — the SIMD GEMM foundation — and never
//! re-transcodes SIMD (the `simd-savant` invariant: all SIMD comes from
//! `ndarray::simd`). See `lance-graph/.claude/board/EPIPHANIES.md`
//! `E-OCR-COMPUTE-NDARRAY-SEAM-1` and
//! `tesseract-rs/.claude/plans/recognizer-core-shape-v1.md`.
//!
//! ## Leaf 1 — `MatrixDotVector` (the hardware-acceleration leaf)
//!
//! [`matrix_dot_vector`] is the transcode of Tesseract's base
//! `IntSimdMatrix::MatrixDotVector` (`src/arch/intsimdmatrix.cpp:78-117`),
//! consuming ndarray's `matmul_i8_to_i32` (AMX `TDPBUSD` → `VPDPBUSD`-zmm →
//! -ymm → scalar). Because int8×int8→i32 accumulation is **exact and
//! order-independent**, the result is identical across every SIMD tier — the
//! recognizer's integer matmul is bit-reproducible, which is what makes
//! byte-parity against libtesseract clean.

use ndarray::{Array2, ArrayView2};

pub mod activation;
pub mod weight_matrix;
pub use weight_matrix::WeightMatrix;

/// `INT8_MAX` (127) — the value the recognizer's imaginary bias input `1.0`
/// quantizes to (tesseract `intsimdmatrix.cpp:101`: the input is int8-quantized
/// where `1.0` maps to `INT8_MAX`, **not** `1`).
const INT8_MAX_I8: i8 = i8::MAX;

/// A failure in a recognizer compute primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecognizerError {
    /// A matrix/vector dimension did not satisfy the operation's contract.
    DimMismatch(&'static str),
    /// The underlying ndarray int8 GEMM rejected the shapes (should not occur
    /// once [`matrix_dot_vector`]'s own dimension checks have passed).
    Gemm(String),
    /// A serialized buffer ended mid-field.
    UnexpectedEof,
    /// A serialized `WeightMatrix` used a format this loader does not handle
    /// (old float format, or float mode — this crate loads int mode only).
    UnsupportedFormat(&'static str),
}

impl std::fmt::Display for RecognizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimMismatch(what) => write!(f, "recognizer dimension mismatch: {what}"),
            Self::Gemm(msg) => write!(f, "int8 gemm rejected the shapes: {msg}"),
            Self::UnexpectedEof => write!(f, "weight buffer ended mid-field"),
            Self::UnsupportedFormat(what) => write!(f, "unsupported weight format: {what}"),
        }
    }
}

impl std::error::Error for RecognizerError {}

/// Compute `v = W·u` in int mode — the transcode of Tesseract's base
/// `IntSimdMatrix::MatrixDotVector` (`intsimdmatrix.cpp:78-117`).
///
/// - `w` is `num_out × (num_in + 1)` int8; the **last column is the bias**.
/// - `u` is `num_in` int8 (the int8-quantized layer input).
/// - `scales` is `num_out` `f64` — the per-output reproducing factor
///   (`max_abs_row / INT8_MAX` from `WeightMatrix::ConvertToInt`).
///
/// Returns `v[i] = (Σ_j w(i,j)·u[j] + w(i,num_in)·INT8_MAX) · scales[i]`. The
/// bias falls out of a single matmul by padding `u` with a trailing
/// [`INT8_MAX_I8`] — the int8 quantization of the imaginary `1.0` bias input.
/// The `Σ` + bias is an **exact i32** (ndarray's `matmul_i8_to_i32`, identical
/// across every SIMD tier); only the per-row scale is a float multiply.
///
/// # Errors
///
/// [`RecognizerError::DimMismatch`] if `w` has no columns, `u.len() != num_in`,
/// or `scales.len() != num_out`; [`RecognizerError::Gemm`] if the shaped int8
/// GEMM is rejected (unreachable once the dimension checks pass).
pub fn matrix_dot_vector(
    w: ArrayView2<'_, i8>,
    scales: &[f64],
    u: &[i8],
) -> Result<Vec<f64>, RecognizerError> {
    let combined = matrix_dot_vector_i32(w, u)?;
    if scales.len() != combined.len() {
        return Err(RecognizerError::DimMismatch("scales length != num_out"));
    }
    Ok(combined
        .iter()
        .zip(scales)
        .map(|(&c, &s)| f64::from(c) * s)
        .collect())
}

/// The exact **integer** combined value per output —
/// `Σ_j w(i,j)·u[j] + w(i,num_in)·INT8_MAX` — the `TFloat`-agnostic core of
/// [`matrix_dot_vector`]. The caller applies the per-output scale in whatever
/// precision matches its target: f64 in [`matrix_dot_vector`], **f32** in
/// [`WeightMatrix::forward`] to match Tesseract's FAST_FLOAT (`TFloat = float`).
/// It is this value the recognizer's int8 accumulate is byte-parity-proven on
/// (identical across every SIMD tier — see the Core board `E-OCR-MATDOTVEC-1`).
///
/// `w` is `num_out × (num_in + 1)` int8 (last column = bias); `u` is `num_in`.
///
/// # Errors
///
/// [`RecognizerError::DimMismatch`] if `w` has no columns or `u.len() != num_in`;
/// [`RecognizerError::Gemm`] if the shaped int8 GEMM is rejected.
pub fn matrix_dot_vector_i32(w: ArrayView2<'_, i8>, u: &[i8]) -> Result<Vec<i32>, RecognizerError> {
    let num_out = w.nrows();
    let num_in = w
        .ncols()
        .checked_sub(1)
        .ok_or(RecognizerError::DimMismatch("weight matrix has no columns"))?;
    if u.len() != num_in {
        return Err(RecognizerError::DimMismatch("u length != num_in"));
    }
    // Pad the input with a trailing INT8_MAX so the bias column w[:, num_in] is
    // picked up by the same matmul (the imaginary 1.0 bias input -> 127).
    let mut u_padded = Vec::with_capacity(num_in + 1);
    u_padded.extend_from_slice(u);
    u_padded.push(INT8_MAX_I8);
    let u_col = ArrayView2::from_shape((num_in + 1, 1), u_padded.as_slice())
        .map_err(|e| RecognizerError::Gemm(e.to_string()))?;
    let mut out = Array2::<i32>::zeros((num_out, 1));
    ndarray::simd_runtime::matmul_i8_to_i32(w, u_col, out.view_mut())
        .map_err(|e| RecognizerError::Gemm(format!("{e:?}")))?;
    Ok((0..num_out).map(|i| out[[i, 0]]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The exact base formula (intsimdmatrix.cpp:78-117), hand-computed:
    /// `v[i] = (Σ_j w(i,j)·u[j] + w(i,num_in)·127) · scales[i]`.
    #[test]
    fn matrix_dot_vector_matches_the_base_formula() {
        // num_out=2, num_in=2, last column is the bias.
        let w = array![[1_i8, 2, 3], [4, 5, 6]];
        let u = [10_i8, 20];
        let scales = [0.5_f64, 0.25];
        // row0: 1*10 + 2*20 = 50; bias 3*127 = 381; (50 + 381) * 0.5   = 215.5
        // row1: 4*10 + 5*20 = 140; bias 6*127 = 762; (140 + 762) * 0.25 = 225.5
        let v = matrix_dot_vector(w.view(), &scales, &u).expect("valid dims");
        assert_eq!(v, vec![215.5, 225.5]);
    }

    /// Signed weights + a zero bias column → pure scaled dot product.
    #[test]
    fn signed_weights_zero_bias_is_pure_dot() {
        let w = array![[2_i8, -3, 0]];
        let u = [7_i8, 5];
        let scales = [1.0_f64];
        // 2*7 + (-3)*5 = 14 - 15 = -1; bias 0; -1 * 1.0 = -1.0
        let v = matrix_dot_vector(w.view(), &scales, &u).expect("valid dims");
        assert_eq!(v, vec![-1.0]);
    }

    /// A wider case (num_in not a nice register multiple) exercises the GEMM's
    /// tail handling and the bias-as-padded-127 trick together.
    #[test]
    fn wider_case_with_bias() {
        // num_out=1, num_in=5, bias = 2.
        let w = array![[1_i8, -1, 2, -2, 3, 2]];
        let u = [1_i8, 2, 3, 4, 5];
        let scales = [0.1_f64];
        // dot = 1 - 2 + 6 - 8 + 15 = 12; bias 2*127 = 254; (12 + 254) * 0.1 = 26.6
        let v = matrix_dot_vector(w.view(), &scales, &u).expect("valid dims");
        assert!((v[0] - 26.6).abs() < 1e-9, "got {}", v[0]);
    }

    #[test]
    fn dim_mismatch_is_typed_error() {
        let w = array![[1_i8, 2, 3]]; // num_in = 2
        assert_eq!(
            matrix_dot_vector(w.view(), &[1.0], &[1_i8]),
            Err(RecognizerError::DimMismatch("u length != num_in"))
        );
        assert_eq!(
            matrix_dot_vector(w.view(), &[1.0, 2.0], &[1_i8, 2]),
            Err(RecognizerError::DimMismatch("scales length != num_out"))
        );
    }
}
