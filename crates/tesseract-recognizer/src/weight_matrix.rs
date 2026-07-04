//! `WeightMatrix` load side (recognizer Leaf 2) — the int-mode `DeSerialize` of
//! Tesseract's `WeightMatrix` (`lstm/weightmatrix.cpp:280-320`), consuming the
//! byte-parity-proven [`crate::matrix_dot_vector_i32`] for the forward step.
//!
//! ## Binary format (int mode; little-endian `TFile`)
//!
//! ```text
//! u8    mode                     // kInt8Flag(1) | kDoubleFlag(128) [| kAdamFlag(4)]
//! -- wi_ : GENERIC_2D_ARRAY<int8_t> (matrix.h Serialize) --
//! u32   dim1 (= num_out)
//! u32   dim2 (= num_in + 1)      // the last column is the bias
//! i8    empty_                   // the array's fill sentinel (0), serialized
//! (dim1·dim2) × i8   data        // row-major
//! -- scales_ --
//! u32   num_scales
//! num_scales × f64   scale_disk  // = scale_mem · INT8_MAX ; loaded as / INT8_MAX
//! ```
//!
//! Three details that only the source reveals: `mode` always carries
//! `kDoubleFlag` in this (post-2018) format — its absence is the old float
//! layout ([`crate::RecognizerError::UnsupportedFormat`]); the `empty_` fill
//! byte sits *between* the dims and the data; and **scales are doubles on disk
//! regardless of `FAST_FLOAT`** (`weightmatrix.cpp:257`). `Init` may pad
//! `scales_` past `num_out` for the SIMD layout — the extra entries are unused
//! by the base forward, so a `num_scales > num_out` buffer keeps only the first
//! `num_out`.
//!
//! [`WeightMatrix::forward`] scales in **f32** to match Tesseract's in-env
//! `TFloat = float` (FAST_FLOAT build); the integer accumulate underneath is the
//! byte-parity-proven [`crate::matrix_dot_vector_i32`].

use ndarray::Array2;

use crate::{matrix_dot_vector, matrix_dot_vector_i32, RecognizerError};

/// `kInt8Flag` (`weightmatrix.cpp:229`) — mode bit: the matrix stores int8 weights.
const K_INT8_FLAG: u8 = 1;
/// `kAdamFlag` (`weightmatrix.cpp:231`) — training metadata bit.
const K_ADAM_FLAG: u8 = 4;
/// `kDoubleFlag` (`weightmatrix.cpp:235`) — set in the current format; its
/// absence means the old float layout.
const K_DOUBLE_FLAG: u8 = 128;
/// `INT8_MAX` — the on-disk scale factor (scales are stored `· INT8_MAX`).
const INT8_MAX_F64: f64 = 127.0;
/// Guard on a declared weight-element count (mirrors the `serialis.h` cap).
const MAX_WEIGHTS: usize = 50_000_000;

/// A loaded int-mode `WeightMatrix` — int8 weights `wi_` (`num_out ×
/// (num_in+1)`, last column = bias) plus the per-output scales — the transcode
/// of the load side of Tesseract's `WeightMatrix` (`lstm/weightmatrix.{h,cpp}`).
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    /// int8 weights, `num_out × (num_in + 1)`; the last column is the bias.
    wi: Array2<i8>,
    /// per-output scale (`scale_disk / INT8_MAX`), length `num_out`.
    scales: Vec<f64>,
    /// Whether the adam flag was set (training metadata; inference ignores it).
    use_adam: bool,
}

impl WeightMatrix {
    /// Load an int-mode `WeightMatrix` from its little-endian serialized bytes —
    /// the transcode of `WeightMatrix::DeSerialize` (`weightmatrix.cpp:280-320`,
    /// int-mode arm). See the module docs for the layout.
    ///
    /// # Errors
    ///
    /// [`RecognizerError::UnsupportedFormat`] for the old float layout or a
    /// float-mode matrix; [`RecognizerError::UnexpectedEof`] on a truncated
    /// buffer; [`RecognizerError::DimMismatch`] if the weight dims overflow or
    /// the scale count is below `num_out`.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, RecognizerError> {
        let mut r = ByteReader::new(bytes);
        let mode = r.read_u8()?;
        if (mode & K_DOUBLE_FLAG) == 0 {
            return Err(RecognizerError::UnsupportedFormat(
                "old (pre-kDoubleFlag) weight format",
            ));
        }
        if (mode & K_INT8_FLAG) == 0 {
            return Err(RecognizerError::UnsupportedFormat(
                "float-mode weight matrix (this loader is int mode only)",
            ));
        }
        let use_adam = (mode & K_ADAM_FLAG) != 0;
        // wi_ = GENERIC_2D_ARRAY<int8_t>: u32 dim1, u32 dim2, i8 empty_, data.
        let dim1 = r.read_u32()? as usize; // num_out
        let dim2 = r.read_u32()? as usize; // num_in + 1
        let _empty = r.read_i8()?; // the array's empty sentinel (0), consumed
        let n = dim1.checked_mul(dim2).filter(|&n| n <= MAX_WEIGHTS).ok_or(
            RecognizerError::DimMismatch("weight element count out of range"),
        )?;
        let mut data = vec![0_i8; n];
        for d in &mut data {
            *d = r.read_i8()?;
        }
        let wi = Array2::from_shape_vec((dim1, dim2), data)
            .map_err(|_| RecognizerError::DimMismatch("weight data length != dim1*dim2"))?;
        // scales_: u32 count, then count doubles (= scale_mem · INT8_MAX).
        let num_scales = r.read_u32()? as usize;
        let mut scales = Vec::with_capacity(num_scales.min(MAX_WEIGHTS));
        for _ in 0..num_scales {
            scales.push(r.read_f64()? / INT8_MAX_F64);
        }
        // Init may pad scales_ past num_out for the SIMD layout; the base forward
        // uses only the first num_out (one per real output row of wi_).
        if scales.len() < dim1 {
            return Err(RecognizerError::DimMismatch("fewer scales than outputs"));
        }
        scales.truncate(dim1);
        Ok(Self {
            wi,
            scales,
            use_adam,
        })
    }

    /// The number of outputs (rows of `wi_`).
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.wi.nrows()
    }

    /// The number of inputs (`wi_` columns minus the bias column).
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.wi.ncols() - 1
    }

    /// Whether the adam training flag was set (inference ignores it).
    #[must_use]
    pub fn use_adam(&self) -> bool {
        self.use_adam
    }

    /// The per-output scales (`scale_disk / INT8_MAX`), one per output.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// The int8 forward step `v = W·u`, matching Tesseract's FAST_FLOAT build:
    /// `v[i] = f32(combined[i]) · f32(scale[i])`, where `combined` is the
    /// byte-parity-proven integer accumulate. The scale is applied in **f32** so
    /// the result is bit-identical to libtesseract's `MatrixDotVector` on the
    /// in-env (`TFloat = float`) library.
    ///
    /// # Errors
    ///
    /// Propagates [`crate::matrix_dot_vector_i32`]'s dimension / GEMM errors.
    pub fn forward(&self, u: &[i8]) -> Result<Vec<f32>, RecognizerError> {
        let combined = matrix_dot_vector_i32(self.wi.view(), u)?;
        Ok(combined
            .iter()
            .zip(&self.scales)
            .map(|(&c, &s)| (c as f32) * (s as f32))
            .collect())
    }

    /// The int8 forward step in f64 (higher precision than the FAST_FLOAT build)
    /// — `v[i] = f64(combined[i]) · scale[i]`, via [`matrix_dot_vector`].
    ///
    /// # Errors
    ///
    /// Propagates [`matrix_dot_vector`]'s dimension / GEMM errors.
    pub fn forward_f64(&self, u: &[i8]) -> Result<Vec<f64>, RecognizerError> {
        matrix_dot_vector(self.wi.view(), &self.scales, u)
    }
}

/// A little-endian byte cursor — the `TFile` reader surface Leaf 2 needs.
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], RecognizerError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(RecognizerError::UnexpectedEof)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(RecognizerError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }
    fn read_u8(&mut self) -> Result<u8, RecognizerError> {
        Ok(self.take(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8, RecognizerError> {
        Ok(self.take(1)?[0] as i8)
    }
    fn read_u32(&mut self) -> Result<u32, RecognizerError> {
        let a: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| RecognizerError::UnexpectedEof)?;
        Ok(u32::from_le_bytes(a))
    }
    fn read_f64(&mut self) -> Result<f64, RecognizerError> {
        let a: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| RecognizerError::UnexpectedEof)?;
        Ok(f64::from_le_bytes(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build an int-mode `WeightMatrix` serialization from `wi` (row-major,
    /// including the bias column) + per-output in-memory scales, in the exact
    /// little-endian wire layout `WeightMatrix::Serialize` writes.
    fn build(dim1: usize, dim2: usize, wi: &[i8], scales_mem: &[f64], adam: bool) -> Vec<u8> {
        assert_eq!(wi.len(), dim1 * dim2);
        assert_eq!(scales_mem.len(), dim1);
        let mut b = Vec::new();
        let mode: u8 = K_INT8_FLAG | K_DOUBLE_FLAG | if adam { K_ADAM_FLAG } else { 0 };
        b.push(mode);
        b.extend_from_slice(&(dim1 as u32).to_le_bytes());
        b.extend_from_slice(&(dim2 as u32).to_le_bytes());
        b.push(0); // empty_
        for &w in wi {
            b.push(w as u8);
        }
        b.extend_from_slice(&(scales_mem.len() as u32).to_le_bytes());
        for &s in scales_mem {
            // On disk the scale carries an extra INT8_MAX factor (removed on load).
            b.extend_from_slice(&(s * INT8_MAX_F64).to_le_bytes());
        }
        b
    }

    #[test]
    fn deserialize_reads_the_int_mode_format() {
        let bytes = build(2, 3, &[1, 2, 3, 4, 5, 6], &[0.5, 0.25], false);
        let wm = WeightMatrix::from_le_bytes(&bytes).expect("valid");
        assert_eq!(wm.num_outputs(), 2);
        assert_eq!(wm.num_inputs(), 2);
        assert!(!wm.use_adam());
        assert_eq!(wm.scales(), &[0.5, 0.25]);
    }

    #[test]
    fn forward_matches_the_matrix_dot_vector_formula() {
        let bytes = build(2, 3, &[1, 2, 3, 4, 5, 6], &[0.5, 0.25], false);
        let wm = WeightMatrix::from_le_bytes(&bytes).expect("valid");
        // row0: (1*10 + 2*20 + 3*127) * 0.5   = (50 + 381) * 0.5   = 215.5
        // row1: (4*10 + 5*20 + 6*127) * 0.25  = (140 + 762) * 0.25 = 225.5
        let v = wm.forward(&[10, 20]).expect("valid");
        assert_eq!(v, vec![215.5_f32, 225.5]);
        let v64 = wm.forward_f64(&[10, 20]).expect("valid");
        assert_eq!(v64, vec![215.5_f64, 225.5]);
    }

    #[test]
    fn scales_padded_past_num_out_are_truncated() {
        let mut bytes = build(2, 3, &[1, 2, 3, 4, 5, 6], &[0.5, 0.25], false);
        // Rewrite the scale count (u32 after the data) from 2 -> 3 and append a
        // third (Init-padding) scale of 0.0. num_scales offset:
        //   1(mode) + 4 + 4 + 1(dims + empty) + 6(data) = 16.
        let ofs = 1 + 4 + 4 + 1 + 6;
        bytes[ofs..ofs + 4].copy_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(0.0_f64 * INT8_MAX_F64).to_le_bytes());
        let wm = WeightMatrix::from_le_bytes(&bytes).expect("valid");
        assert_eq!(wm.scales().len(), 2, "padding scale truncated to num_out");
        assert_eq!(wm.scales(), &[0.5, 0.25]);
    }

    #[test]
    fn old_or_float_format_is_unsupported() {
        assert_eq!(
            WeightMatrix::from_le_bytes(&[K_INT8_FLAG]).unwrap_err(),
            RecognizerError::UnsupportedFormat("old (pre-kDoubleFlag) weight format")
        );
        assert_eq!(
            WeightMatrix::from_le_bytes(&[K_DOUBLE_FLAG]).unwrap_err(),
            RecognizerError::UnsupportedFormat(
                "float-mode weight matrix (this loader is int mode only)"
            )
        );
    }

    #[test]
    fn truncated_buffer_errors() {
        let bytes = build(2, 3, &[1, 2, 3, 4, 5, 6], &[0.5, 0.25], false);
        assert_eq!(
            WeightMatrix::from_le_bytes(&bytes[..bytes.len() - 1]).unwrap_err(),
            RecognizerError::UnexpectedEof
        );
    }
}
