//! `NetworkIO` — recognizer Leaf A1 (part 2): the transcode of Tesseract's
//! network input/output tensor (`lstm/networkio.{h,cpp}`), the multi-dim
//! int8/f32 SoA every 2-D layer reads and writes through a [`StrideMap`].
//!
//! Scope: the **forward-pass op subset** — resizes, zeroing, per-timestep
//! copies, the maxpool step, the x/y reversals + XY transpose, the timestep
//! read/write quantizers, and the out-of-image randomizer fill. Training-side
//! ops (backward deltas, combiner, Pix I/O) are out of scope; the image
//! `Input` path (A6) lands separately.
//!
//! Transcription notes (byte-parity relevant):
//! - C++ `ResizeNoInit(size1, size2, pad)` allocates `size1·size2 + pad` — the
//!   SIMD padding is a single **global tail** (`matrix.h`), `dim2_` stays
//!   `size2`, and the row stride is `dim2_`. The Rust grid therefore uses a
//!   plain `width × num_features` `Vec` (no padding: we never over-read).
//! - C++ leaves resized cells **uninitialized**; `ResizeToMap` then zeroes the
//!   invalid (padding) cells via `ZeroInvalidElements`. This grid allocates
//!   zeroed instead, which is observably identical: every valid cell is
//!   written by the producer, every invalid cell is zeroed on both sides.
//! - `MaxpoolTimeStep` uses **strict** `<` (`networkio.cpp:684/694`): ties
//!   keep the earlier source timestep in `max_line`.
//! - `Randomize` (`networkio.cpp:416-429`): int mode stores
//!   `IntCastRounded(SignedRand(127))` (f64 → round-half-away → i8); float
//!   mode stores `SignedRand(1.0)` narrowed f64 → f32. Byte-parity of
//!   `Convolve`'s out-of-image fill rides on [`TRand`] being exact.
//! - `WriteTimeStepPart` int mode is the Leaf-5-proven quantizer:
//!   `clip(IntCastRounded(x·127), ±127)` with `IntCastRounded(float)`
//!   (`helpers.h:189-192`, half-away-from-zero), never −128.

use crate::stridemap::{FlexDim, StrideIndex, StrideMap};
use crate::trand::TRand;

/// `INT8_MAX` as f32/f64 — the quantization scale.
const INT8_MAX_F32: f32 = 127.0;
const INT8_MAX_F64: f64 = 127.0;

/// `IntCastRounded(float)` (`helpers.h:189`): round half away from zero.
#[must_use]
fn int_cast_rounded_f32(x: f32) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        -((-x + 0.5) as i32)
    }
}

/// `IntCastRounded(double)` (`helpers.h:181`).
#[must_use]
fn int_cast_rounded_f64(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        -((-x + 0.5) as i32)
    }
}

/// A dense `dim1 × dim2` row-major grid — the value-carrying part of C++
/// `GENERIC_2D_ARRAY<T>` (row stride = `dim2`, no SIMD tail padding).
#[derive(Debug, Clone, Default)]
struct Grid<T> {
    data: Vec<T>,
    dim1: usize,
    dim2: usize,
}

impl<T: Copy + Default> Grid<T> {
    /// (Re)size to `dim1 × dim2`, zero-initialised (see module notes on why
    /// zeroed-vs-uninitialized is observably identical here).
    fn resize(&mut self, dim1: usize, dim2: usize) {
        self.dim1 = dim1;
        self.dim2 = dim2;
        self.data.clear();
        self.data.resize(dim1 * dim2, T::default());
    }

    fn row(&self, t: usize) -> &[T] {
        &self.data[t * self.dim2..(t + 1) * self.dim2]
    }

    fn row_mut(&mut self, t: usize) -> &mut [T] {
        &mut self.data[t * self.dim2..(t + 1) * self.dim2]
    }

    /// Zero a contiguous span of `n` values starting at flat position
    /// `(t, offset)` — the C++ `ZeroVector(fill_size, array[t] + offset)`
    /// pattern, which may run across row boundaries (rows are contiguous).
    fn zero_span(&mut self, t: usize, offset: usize, n: usize) {
        let start = t * self.dim2 + offset;
        self.data[start..start + n].fill(T::default());
    }
}

/// The network I/O tensor (`NetworkIO`, `networkio.h:38`): an int8 OR f32
/// `width × num_features` grid plus the [`StrideMap`] that names each row's
/// `[batch][y][x]` position.
#[derive(Debug, Clone, Default)]
pub struct NetworkIo {
    f: Grid<f32>,
    i: Grid<i8>,
    int_mode: bool,
    stride_map: StrideMap,
}

impl NetworkIo {
    /// `Resize2d` (`networkio.cpp:35-43`): a plain 2-D temp buffer — empty
    /// stride map, no batches, no y-dim.
    #[must_use]
    pub fn new_2d(int_mode: bool, width: usize, num_features: usize) -> Self {
        let mut io = Self {
            int_mode,
            ..Self::default()
        };
        if int_mode {
            io.i.resize(width, num_features);
        } else {
            io.f.resize(width, num_features);
        }
        io
    }

    /// `ResizeToMap` (`networkio.cpp:46-58`): adopt the stride map, size the
    /// grid to its packed width, and zero the invalid (ragged-padding) cells.
    pub fn resize_to_map(&mut self, int_mode: bool, stride_map: &StrideMap, num_features: usize) {
        self.stride_map = stride_map.clone();
        self.int_mode = int_mode;
        let width = stride_map.width().max(0) as usize;
        if int_mode {
            self.i.resize(width, num_features);
            self.f = Grid::default();
        } else {
            self.f.resize(width, num_features);
            self.i = Grid::default();
        }
        self.zero_invalid_elements();
    }

    /// `Resize` (`networkio.h:44`): same stride and mode as `src`, given
    /// feature count.
    pub fn resize_like(&mut self, src: &NetworkIo, num_features: usize) {
        self.resize_to_map(src.int_mode, &src.stride_map, num_features);
    }

    /// `ResizeFloat` (`networkio.h:51`): as `resize_like` but forcing floats.
    pub fn resize_float(&mut self, src: &NetworkIo, num_features: usize) {
        self.resize_to_map(false, &src.stride_map, num_features);
    }

    /// `ResizeScaled` (`networkio.cpp:61-65`): shrink x/y by integer factors
    /// (the `Maxpool`/`Reconfig` output shape).
    pub fn resize_scaled(
        &mut self,
        src: &NetworkIo,
        x_scale: i32,
        y_scale: i32,
        num_features: usize,
    ) {
        let mut stride_map = src.stride_map.clone();
        stride_map.scale_xy(x_scale, y_scale);
        self.resize_to_map(src.int_mode, &stride_map, num_features);
    }

    /// `ResizeXTo1` (`networkio.cpp:68-72`).
    pub fn resize_x_to_1(&mut self, src: &NetworkIo, num_features: usize) {
        let mut stride_map = src.stride_map.clone();
        stride_map.reduce_width_to_1();
        self.resize_to_map(src.int_mode, &stride_map, num_features);
    }

    /// `Width()`.
    #[must_use]
    pub fn width(&self) -> usize {
        if self.int_mode {
            self.i.dim1
        } else {
            self.f.dim1
        }
    }

    /// `NumFeatures()`.
    #[must_use]
    pub fn num_features(&self) -> usize {
        if self.int_mode {
            self.i.dim2
        } else {
            self.f.dim2
        }
    }

    /// `int_mode()`.
    #[must_use]
    pub fn int_mode(&self) -> bool {
        self.int_mode
    }

    /// `stride_map()`.
    #[must_use]
    pub fn stride_map(&self) -> &StrideMap {
        &self.stride_map
    }

    /// `f(t)` — a float timestep row. Panics in int mode (the C++ `ASSERT_HOST`).
    #[must_use]
    pub fn f(&self, t: usize) -> &[f32] {
        assert!(!self.int_mode, "f() on an int-mode NetworkIo");
        self.f.row(t)
    }

    /// Mutable float row.
    pub fn f_mut(&mut self, t: usize) -> &mut [f32] {
        assert!(!self.int_mode, "f_mut() on an int-mode NetworkIo");
        self.f.row_mut(t)
    }

    /// `i(t)` — an int8 timestep row. Panics in float mode.
    #[must_use]
    pub fn i(&self, t: usize) -> &[i8] {
        assert!(self.int_mode, "i() on a float-mode NetworkIo");
        self.i.row(t)
    }

    /// `Zero()`.
    pub fn zero(&mut self) {
        for t in 0..self.width() {
            self.zero_time_step(t);
        }
    }

    /// `ZeroTimeStep(t)` (`networkio.h:147`).
    pub fn zero_time_step(&mut self, t: usize) {
        if self.int_mode {
            self.i.row_mut(t).fill(0);
        } else {
            self.f.row_mut(t).fill(0.0);
        }
    }

    /// `ZeroInvalidElements` (`networkio.cpp:86-120`): zero every cell beyond
    /// each image's true width/height. The C++ ignores the validity of the
    /// probe offsets and uses their computed `t` anyway — mirrored here (our
    /// `add_offset` also recomputes `t` unconditionally).
    pub fn zero_invalid_elements(&mut self) {
        let num_features = self.num_features();
        let map = self.stride_map.clone(); // walk a snapshot; mutate the grid
        let full_width = map.size(FlexDim::Width);
        let full_height = map.size(FlexDim::Height);
        let mut b_index = map.index_first();
        loop {
            let end_x = b_index.max_index_of_dim(FlexDim::Width) + 1;
            if end_x < full_width {
                // The width is small: fill for every valid y.
                let fill = num_features * (full_width - end_x) as usize;
                let mut y_index = b_index.clone();
                loop {
                    let mut z_index = y_index.clone();
                    z_index.add_offset(end_x, FlexDim::Width);
                    let t = z_index.t() as usize;
                    if self.int_mode {
                        self.i.zero_span(t, 0, fill);
                    } else {
                        self.f.zero_span(t, 0, fill);
                    }
                    if !y_index.add_offset(1, FlexDim::Height) {
                        break;
                    }
                }
            }
            let end_y = b_index.max_index_of_dim(FlexDim::Height) + 1;
            if end_y < full_height {
                // The height is small: fill the remaining rows in one go.
                let mut y_index = b_index.clone();
                y_index.add_offset(end_y, FlexDim::Height);
                let t = y_index.t() as usize;
                let fill = num_features * (full_width as usize) * (full_height - end_y) as usize;
                if self.int_mode {
                    self.i.zero_span(t, 0, fill);
                } else {
                    self.f.zero_span(t, 0, fill);
                }
            }
            if !b_index.add_offset(1, FlexDim::Batch) {
                break;
            }
        }
    }

    /// `CopyTimeStepFrom` (`networkio.cpp:395-402`): whole-row copy; modes must
    /// match.
    pub fn copy_time_step_from(&mut self, dest_t: usize, src: &NetworkIo, src_t: usize) {
        assert_eq!(self.int_mode, src.int_mode, "mode mismatch");
        if self.int_mode {
            let n = self.i.dim2;
            self.i.row_mut(dest_t)[..n].copy_from_slice(&src.i.row(src_t)[..n]);
        } else {
            let n = self.f.dim2;
            self.f.row_mut(dest_t)[..n].copy_from_slice(&src.f.row(src_t)[..n]);
        }
    }

    /// `CopyTimeStepGeneral` (`networkio.cpp:405-413`): partial-row copy at
    /// offsets — the `Convolve`/`Reconfig` stacking primitive.
    pub fn copy_time_step_general(
        &mut self,
        dest_t: usize,
        dest_offset: usize,
        num_features: usize,
        src: &NetworkIo,
        src_t: usize,
        src_offset: usize,
    ) {
        assert_eq!(self.int_mode, src.int_mode, "mode mismatch");
        if self.int_mode {
            self.i.row_mut(dest_t)[dest_offset..dest_offset + num_features]
                .copy_from_slice(&src.i.row(src_t)[src_offset..src_offset + num_features]);
        } else {
            self.f.row_mut(dest_t)[dest_offset..dest_offset + num_features]
                .copy_from_slice(&src.f.row(src_t)[src_offset..src_offset + num_features]);
        }
    }

    /// `Randomize` (`networkio.cpp:416-429`): fill `[offset, offset+n)` of row
    /// `t` from the randomizer — the out-of-image noise fill.
    pub fn randomize(
        &mut self,
        t: usize,
        offset: usize,
        num_features: usize,
        randomizer: &mut TRand,
    ) {
        if self.int_mode {
            let line = &mut self.i.row_mut(t)[offset..offset + num_features];
            for v in line {
                *v = int_cast_rounded_f64(randomizer.signed_rand(INT8_MAX_F64)) as i8;
            }
        } else {
            let line = &mut self.f.row_mut(t)[offset..offset + num_features];
            for v in line {
                *v = randomizer.signed_rand(1.0) as f32;
            }
        }
    }

    /// `MaxpoolTimeStep` (`networkio.cpp:677-700`): elementwise running max of
    /// `src[src_t]` into `self[dest_t]`, recording the winning source timestep
    /// per feature in `max_line`. STRICT `<`: ties keep the earlier winner.
    pub fn maxpool_time_step(
        &mut self,
        dest_t: usize,
        src: &NetworkIo,
        src_t: usize,
        max_line: &mut [i32],
    ) {
        assert_eq!(self.int_mode, src.int_mode, "mode mismatch");
        if self.int_mode {
            let dim = self.i.dim2;
            let dest = self.i.row_mut(dest_t);
            let s = src.i.row(src_t);
            for k in 0..dim {
                if dest[k] < s[k] {
                    dest[k] = s[k];
                    max_line[k] = src_t as i32;
                }
            }
        } else {
            let dim = self.f.dim2;
            let dest = self.f.row_mut(dest_t);
            let s = src.f.row(src_t);
            for k in 0..dim {
                if dest[k] < s[k] {
                    dest[k] = s[k];
                    max_line[k] = src_t as i32;
                }
            }
        }
    }

    /// `ReadTimeStep` (`networkio.cpp:610-622`): row → f32s; int rows dequantize
    /// by `/127`.
    pub fn read_time_step(&self, t: usize, output: &mut [f32]) {
        if self.int_mode {
            for (o, &v) in output.iter_mut().zip(self.i.row(t)) {
                *o = f32::from(v) / INT8_MAX_F32;
            }
        } else {
            output[..self.f.dim2].copy_from_slice(self.f.row(t));
        }
    }

    /// `WriteTimeStep` (`networkio.cpp:656-658`).
    pub fn write_time_step(&mut self, t: usize, input: &[f32]) {
        self.write_time_step_part(t, 0, self.num_features(), input);
    }

    /// `WriteTimeStepPart` (`networkio.cpp:662-674`): the Leaf-5-proven int8
    /// quantizer (`clip(IntCastRounded(x·127), ±127)`) or a plain f32 store.
    pub fn write_time_step_part(
        &mut self,
        t: usize,
        offset: usize,
        num_features: usize,
        input: &[f32],
    ) {
        if self.int_mode {
            let line = &mut self.i.row_mut(t)[offset..offset + num_features];
            for (o, &x) in line.iter_mut().zip(input) {
                *o = int_cast_rounded_f32(x * INT8_MAX_F32).clamp(-127, 127) as i8;
            }
        } else {
            self.f.row_mut(t)[offset..offset + num_features]
                .copy_from_slice(&input[..num_features]);
        }
    }

    /// `SetPixel` (`networkio.cpp:290-297`): store one image pixel into the
    /// grid, stretching `[black, black + 2·contrast]` to the dynamic range.
    /// `float_pixel = (pixel − black)/contrast − 1`; int mode quantizes with
    /// **×(INT8_MAX+1) = ×128** (NOT the ×127 of `write_time_step` — a distinct
    /// rounding constant), `clip(round(128·float_pixel), ±127)`.
    pub fn set_pixel(&mut self, t: usize, f: usize, pixel: i32, black: f32, contrast: f32) {
        let float_pixel = (pixel as f32 - black) / contrast - 1.0;
        if self.int_mode {
            self.i.row_mut(t)[f] =
                int_cast_rounded_f32(float_pixel * (INT8_MAX_F32 + 1.0)).clamp(-127, 127) as i8;
        } else {
            self.f.row_mut(t)[f] = float_pixel;
        }
    }

    /// `CopyWithYReversal` (`networkio.cpp:868-885`): per image, swap row
    /// bands top-to-bottom.
    pub fn copy_with_y_reversal(&mut self, src: &NetworkIo) {
        let num_features = src.num_features();
        self.resize_like(src, num_features);
        let map = src.stride_map.clone();
        let mut b_index = map.index_first();
        loop {
            let width = b_index.max_index_of_dim(FlexDim::Width) + 1;
            let mut fwd_index = b_index.clone();
            let mut rev_index = b_index.clone();
            let last_y = rev_index.max_index_of_dim(FlexDim::Height);
            rev_index.add_offset(last_y, FlexDim::Height);
            loop {
                let fwd_t = fwd_index.t() as usize;
                let rev_t = rev_index.t() as usize;
                for k in 0..width as usize {
                    self.copy_time_step_from(rev_t + k, src, fwd_t + k);
                }
                if !(fwd_index.add_offset(1, FlexDim::Height)
                    && rev_index.add_offset(-1, FlexDim::Height))
                {
                    break;
                }
            }
            if !b_index.add_offset(1, FlexDim::Batch) {
                break;
            }
        }
    }

    /// `CopyWithXReversal` (`networkio.cpp:888-903`): per image row, swap
    /// columns left-to-right.
    pub fn copy_with_x_reversal(&mut self, src: &NetworkIo) {
        let num_features = src.num_features();
        self.resize_like(src, num_features);
        let map = src.stride_map.clone();
        let mut b_index = map.index_first();
        loop {
            let mut y_index = b_index.clone();
            loop {
                let mut fwd_index = y_index.clone();
                let mut rev_index = y_index.clone();
                let last_x = rev_index.max_index_of_dim(FlexDim::Width);
                rev_index.add_offset(last_x, FlexDim::Width);
                loop {
                    self.copy_time_step_from(rev_index.t() as usize, src, fwd_index.t() as usize);
                    if !(fwd_index.add_offset(1, FlexDim::Width)
                        && rev_index.add_offset(-1, FlexDim::Width))
                    {
                        break;
                    }
                }
                if !y_index.add_offset(1, FlexDim::Height) {
                    break;
                }
            }
            if !b_index.add_offset(1, FlexDim::Batch) {
                break;
            }
        }
    }

    /// `CopyWithXYTranspose` (`networkio.cpp:906-924`): the `Txy` grid
    /// transpose — dest adopts the transposed stride map, then per image the
    /// (y,x) walk of src pairs with the (x,y) walk of dest.
    pub fn copy_with_xy_transpose(&mut self, src: &NetworkIo) {
        let num_features = src.num_features();
        let mut map = src.stride_map.clone();
        map.transpose_xy();
        self.resize_to_map(src.int_mode, &map, num_features);
        let dest_map = map; // snapshot for walking while mutating self
        let src_map = src.stride_map.clone();
        let mut src_b: StrideIndex<'_> = src_map.index_first();
        let mut dest_b = dest_map.index_first();
        loop {
            let mut src_y = src_b.clone();
            let mut dest_x = dest_b.clone();
            loop {
                let mut src_x = src_y.clone();
                let mut dest_y = dest_x.clone();
                loop {
                    self.copy_time_step_from(dest_y.t() as usize, src, src_x.t() as usize);
                    if !(src_x.add_offset(1, FlexDim::Width)
                        && dest_y.add_offset(1, FlexDim::Height))
                    {
                        break;
                    }
                }
                if !(src_y.add_offset(1, FlexDim::Height) && dest_x.add_offset(1, FlexDim::Width)) {
                    break;
                }
            }
            if !(src_b.add_offset(1, FlexDim::Batch) && dest_b.add_offset(1, FlexDim::Batch)) {
                break;
            }
        }
    }

    /// `CopyPacking` (`networkio.cpp:929-951`): stack `src`'s features into
    /// `self` at `feature_offset` (the `Parallel` packer); rows of `self`
    /// beyond `src`'s width get that feature band zeroed. Returns the next
    /// feature offset.
    pub fn copy_packing(&mut self, src: &NetworkIo, feature_offset: usize) -> usize {
        assert_eq!(self.int_mode, src.int_mode, "mode mismatch");
        let width = src.width();
        assert!(width <= self.width());
        let num_features = src.num_features();
        assert!(num_features + feature_offset <= self.num_features());
        if self.int_mode {
            for t in 0..width {
                self.i.row_mut(t)[feature_offset..feature_offset + num_features]
                    .copy_from_slice(src.i.row(t));
            }
            for t in width..self.i.dim1 {
                self.i.row_mut(t)[..num_features].fill(0);
            }
        } else {
            for t in 0..width {
                self.f.row_mut(t)[feature_offset..feature_offset + num_features]
                    .copy_from_slice(src.f.row(t));
            }
            for t in width..self.f.dim1 {
                self.f.row_mut(t)[..num_features].fill(0.0);
            }
        }
        num_features + feature_offset
    }

    /// `CopyUnpacking` (`networkio.cpp:955-968`): extract a feature band.
    pub fn copy_unpacking(&mut self, src: &NetworkIo, feature_offset: usize, num_features: usize) {
        self.resize_like(src, num_features);
        assert!(num_features + feature_offset <= src.num_features());
        for t in 0..src.width() {
            if self.int_mode {
                let band = &src.i.row(t)[feature_offset..feature_offset + num_features];
                self.i.row_mut(t).copy_from_slice(band);
            } else {
                let band = &src.f.row(t)[feature_offset..feature_offset + num_features];
                self.f.row_mut(t).copy_from_slice(band);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared ragged batch: (h,w) = (5,7),(3,4),(4,6); nf = 3.
    fn ragged_map() -> StrideMap {
        let mut m = StrideMap::default();
        m.set_stride(&[(5, 7), (3, 4), (4, 6)]);
        m
    }

    /// Deterministic fill of every VALID cell via the stride walk; invalid
    /// cells stay at the resize-time zero.
    fn filled(int_mode: bool) -> NetworkIo {
        let map = ragged_map();
        let mut io = NetworkIo::default();
        io.resize_to_map(int_mode, &map, 3);
        let walk_map = map.clone();
        let mut idx = walk_map.index_first();
        loop {
            let t = idx.t() as usize;
            let vals: Vec<f32> = (0..3)
                .map(|f| ((t as i32 * 31 + f * 17) % 200 - 100) as f32 / 100.0)
                .collect();
            io.write_time_step(t, &vals);
            if !idx.increment() {
                break;
            }
        }
        io
    }

    #[test]
    fn invalid_cells_are_zero_valid_cells_survive() {
        let io = filled(true);
        assert_eq!(io.width(), 105);
        // Image 1 (b=1, 3x4): row t=35 valid; x=4..6 of that row are invalid:
        // t = 35 + 4 = 39 is a padding cell -> all-zero.
        assert!(io.i(39).iter().all(|&v| v == 0));
        // Rows y=3,4 of image 1 are invalid: t = 35 + 3*7 = 56 -> zero.
        assert!(io.i(56).iter().all(|&v| v == 0));
        // A valid cell is non-zero as filled (t=35: 35*31 % 200 ... != 0).
        assert!(io.i(35).iter().any(|&v| v != 0));
    }

    #[test]
    fn write_read_round_trip_int8() {
        let mut io = NetworkIo::new_2d(true, 2, 4);
        io.write_time_step(0, &[1.0, -1.0, 0.5, -0.004]);
        assert_eq!(io.i(0), &[127, -127, 64, -1]); // 0.5*127=63.5 -> 64 (half away)
        let mut out = [0.0f32; 4];
        io.read_time_step(0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6 && (out[2] - 64.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn maxpool_strict_less_keeps_earlier_tie() {
        let mut src = NetworkIo::new_2d(false, 3, 2);
        src.f_mut(0).copy_from_slice(&[1.0, 5.0]);
        src.f_mut(1).copy_from_slice(&[1.0, 7.0]); // f0 ties, f1 wins
        let mut dest = NetworkIo::new_2d(false, 1, 2);
        let mut max_line = [0_i32; 2];
        dest.copy_time_step_from(0, &src, 0);
        dest.maxpool_time_step(0, &src, 1, &mut max_line);
        assert_eq!(dest.f(0), &[1.0, 7.0]);
        assert_eq!(max_line, [0, 1], "tie keeps the earlier timestep");
    }

    #[test]
    fn xy_transpose_round_trips() {
        let src = filled(false);
        let mut once = NetworkIo::default();
        once.copy_with_xy_transpose(&src);
        assert_eq!(once.stride_map().size(FlexDim::Height), 7);
        assert_eq!(once.stride_map().size(FlexDim::Width), 5);
        let mut twice = NetworkIo::default();
        twice.copy_with_xy_transpose(&once);
        // Transposing twice restores every valid cell (padding stays zero on
        // both sides, so whole-store equality holds).
        assert_eq!(twice.width(), src.width());
        for t in 0..src.width() {
            assert_eq!(twice.f(t), src.f(t), "t={t}");
        }
    }

    #[test]
    fn x_and_y_reversal_are_involutions() {
        let src = filled(false);
        for rev in ["x", "y"] {
            let mut once = NetworkIo::default();
            let mut twice = NetworkIo::default();
            if rev == "x" {
                once.copy_with_x_reversal(&src);
                twice.copy_with_x_reversal(&once);
            } else {
                once.copy_with_y_reversal(&src);
                twice.copy_with_y_reversal(&once);
            }
            for t in 0..src.width() {
                assert_eq!(twice.f(t), src.f(t), "{rev} t={t}");
            }
        }
    }

    #[test]
    fn packing_unpacking_round_trip() {
        let a = filled(true);
        let mut wide = NetworkIo::default();
        wide.resize_to_map(true, a.stride_map(), 7);
        let next = wide.copy_packing(&a, 2);
        assert_eq!(next, 5);
        let mut back = NetworkIo::default();
        back.copy_unpacking(&wide, 2, 3);
        for t in 0..a.width() {
            assert_eq!(back.i(t), a.i(t), "t={t}");
        }
    }

    #[test]
    fn randomize_matches_trand_semantics() {
        let mut io = NetworkIo::new_2d(true, 1, 4);
        let mut r = TRand::default();
        r.set_seed(42);
        io.randomize(0, 0, 4, &mut r);
        // Reference: replay the LCG by hand.
        let mut r2 = TRand::default();
        r2.set_seed(42);
        let expect: Vec<i8> = (0..4)
            .map(|_| {
                let v = r2.signed_rand(127.0);
                int_cast_rounded_f64(v) as i8
            })
            .collect();
        assert_eq!(io.i(0), expect.as_slice());
    }
}
