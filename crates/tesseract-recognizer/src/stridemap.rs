//! `StrideMap` — recognizer Leaf A1 (part 1): the transcode of Tesseract's
//! 4-D-tensor-in-2-D-array index map (`lstm/stridemap.{h,cpp}`).
//!
//! A `NetworkIO` is a 4-D tensor `[batch][y][x][depth]` stored as a 2-D array
//! whose first dimension is the packed `t = batch·H·W + y·W + x` and whose
//! second is depth. `StrideMap` holds the three flexible (non-depth) dimension
//! sizes plus the per-image real heights/widths of a (possibly ragged) batch,
//! and [`StrideIndex`] walks `t` while honouring each image's true bounds —
//! the neighbour access (`AddOffset(x, Width)`) every 2-D layer (`Convolve`/
//! `Maxpool`/`Reconfig`) is built on.
//!
//! Transcription notes (all byte-parity relevant):
//! - `shape_[d]` is the MAX over the batch; `heights_[b]`/`widths_[b]` are the
//!   per-image true sizes. `Index::MaxIndexOfDim` clamps HEIGHT/WIDTH to the
//!   current batch's true size (`stridemap.cpp:46-63`).
//! - `Increment`/`Decrement` iterate raggedly: `Increment` steps `t` by the
//!   precomputed per-dimension increments and carries; `Decrement` re-inits on
//!   a batch borrow because the upper limits change per batch
//!   (`stridemap.cpp:75-110`).
//! - `ScaleXY` divides sizes with C++ integer division (`5/2 == 2`), matching
//!   `Maxpool`/`Reconfig` shrink semantics (`stridemap.cpp:153-163`).

/// `FlexDimensions` (`stridemap.h:32`): the three non-depth dimensions, in
/// packing order (batch outermost, width innermost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlexDim {
    /// `FD_BATCH` — index of multiple images.
    Batch = 0,
    /// `FD_HEIGHT` — y-coordinate in the image.
    Height = 1,
    /// `FD_WIDTH` — x-coordinate in the image.
    Width = 2,
}

/// `FD_DIMSIZE`.
const DIMSIZE: usize = 3;
/// The dimensions in index order, for loops that mirror `for (d = 0; d < FD_DIMSIZE)`.
const DIMS: [FlexDim; DIMSIZE] = [FlexDim::Batch, FlexDim::Height, FlexDim::Width];

/// The mapping from `[batch][y][x]` to the first index of the underlying 2-D
/// array (`StrideMap`, `stridemap.h:41`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StrideMap {
    /// `shape_` — size of each non-depth dimension (max over the batch).
    shape: [i32; DIMSIZE],
    /// `t_increments_` — the stride of each dimension in the packed array.
    t_increments: [i32; DIMSIZE],
    /// `heights_[b]` — the true height of each image in the batch.
    heights: Vec<i32>,
    /// `widths_[b]` — the true width of each image in the batch.
    widths: Vec<i32>,
}

impl StrideMap {
    /// `SetStride(h_w_pairs)` (`stridemap.cpp:131-150`): record per-image
    /// `(height, width)`, set the shape to the maxima, compute increments.
    pub fn set_stride(&mut self, h_w_pairs: &[(i32, i32)]) {
        let mut max_height = 0;
        let mut max_width = 0;
        for &(height, width) in h_w_pairs {
            self.heights.push(height);
            self.widths.push(width);
            max_height = max_height.max(height);
            max_width = max_width.max(width);
        }
        self.shape[FlexDim::Batch as usize] = self.heights.len() as i32;
        self.shape[FlexDim::Height as usize] = max_height;
        self.shape[FlexDim::Width as usize] = max_width;
        self.compute_t_increments();
    }

    /// `ScaleXY(x_factor, y_factor)` (`stridemap.cpp:153-163`): C++ integer
    /// division on every height/width and on the shape.
    pub fn scale_xy(&mut self, x_factor: i32, y_factor: i32) {
        for h in &mut self.heights {
            *h /= y_factor;
        }
        for w in &mut self.widths {
            *w /= x_factor;
        }
        self.shape[FlexDim::Height as usize] /= y_factor;
        self.shape[FlexDim::Width as usize] /= x_factor;
        self.compute_t_increments();
    }

    /// `ReduceWidthTo1` (`stridemap.cpp:166-170`).
    pub fn reduce_width_to_1(&mut self) {
        for w in &mut self.widths {
            *w = 1;
        }
        self.shape[FlexDim::Width as usize] = 1;
        self.compute_t_increments();
    }

    /// `TransposeXY` (`stridemap.cpp:173-177`): swap the height/width shape
    /// AND the per-image vectors.
    pub fn transpose_xy(&mut self) {
        self.shape
            .swap(FlexDim::Height as usize, FlexDim::Width as usize);
        std::mem::swap(&mut self.heights, &mut self.widths);
        self.compute_t_increments();
    }

    /// `Size(dimension)`.
    #[must_use]
    pub fn size(&self, dim: FlexDim) -> i32 {
        self.shape[dim as usize]
    }

    /// `Width()` — the total packed width `t_increments[BATCH] · shape[BATCH]`.
    #[must_use]
    pub fn width(&self) -> i32 {
        self.t_increments[FlexDim::Batch as usize] * self.shape[FlexDim::Batch as usize]
    }

    /// `ComputeTIncrements` (`stridemap.cpp:180-185`): innermost stride 1,
    /// each outer stride = inner stride × inner shape.
    fn compute_t_increments(&mut self) {
        self.t_increments[DIMSIZE - 1] = 1;
        for d in (0..DIMSIZE - 1).rev() {
            self.t_increments[d] = self.t_increments[d + 1] * self.shape[d + 1];
        }
    }

    /// An index positioned at the first valid location (`Index(stride_map)`).
    #[must_use]
    pub fn index_first(&self) -> StrideIndex<'_> {
        StrideIndex {
            map: self,
            t: 0,
            indices: [0; DIMSIZE],
        }
    }

    /// An index positioned at `[batch][y][x]` (`Index(stride_map, b, y, x)`).
    /// The position need not be valid — mirror C++, where validity is the
    /// caller's question via [`StrideIndex::is_valid`].
    #[must_use]
    pub fn index_at(&self, batch: i32, y: i32, x: i32) -> StrideIndex<'_> {
        let mut idx = StrideIndex {
            map: self,
            t: 0,
            indices: [batch, y, x],
        };
        idx.set_t_from_indices();
        idx
    }

    /// An index positioned at the last valid location (`InitToLast`).
    #[must_use]
    pub fn index_last(&self) -> StrideIndex<'_> {
        let mut idx = self.index_first();
        idx.init_to_last_of_batch(idx.max_index_of_dim(FlexDim::Batch));
        idx
    }
}

/// `StrideMap::Index` (`stridemap.h:44`): the non-depth indices + the packed
/// `t`, walking a (ragged) batch. Borrows the map — the C++ borrowed pointer,
/// safe by lifetime.
#[derive(Debug, Clone)]
pub struct StrideIndex<'a> {
    map: &'a StrideMap,
    /// `t_` — index into the first dimension of the underlying array.
    t: i32,
    /// `indices_` — the `[batch, y, x]` position.
    indices: [i32; DIMSIZE],
}

impl StrideIndex<'_> {
    /// `t()`.
    #[must_use]
    pub fn t(&self) -> i32 {
        self.t
    }

    /// `index(dimension)`.
    #[must_use]
    pub fn index(&self, dim: FlexDim) -> i32 {
        self.indices[dim as usize]
    }

    /// `IsValid()` (`stridemap.cpp:24-37`).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.indices.iter().any(|&i| i < 0) {
            return false;
        }
        DIMS.iter()
            .all(|&d| self.indices[d as usize] <= self.max_index_of_dim(d))
    }

    /// `IsLast(dimension)` (`stridemap.cpp:40-42`).
    #[must_use]
    pub fn is_last(&self, dim: FlexDim) -> bool {
        self.max_index_of_dim(dim) == self.indices[dim as usize]
    }

    /// `MaxIndexOfDim(dim)` (`stridemap.cpp:46-63`): shape max for BATCH; the
    /// current batch's true height/width − 1 for the others (falling back to
    /// the shape max when the batch index is out of the vectors' range or the
    /// per-image size exceeds the shape).
    #[must_use]
    pub fn max_index_of_dim(&self, dim: FlexDim) -> i32 {
        let max_index = self.map.shape[dim as usize] - 1;
        if dim == FlexDim::Batch {
            return max_index;
        }
        debug_assert!(self.indices[FlexDim::Batch as usize] >= 0);
        let batch = self.indices[FlexDim::Batch as usize] as usize;
        if dim == FlexDim::Height {
            if batch >= self.map.heights.len() || self.map.heights[batch] > max_index {
                return max_index;
            }
            return self.map.heights[batch] - 1;
        }
        if batch >= self.map.widths.len() || self.map.widths[batch] > max_index {
            return max_index;
        }
        self.map.widths[batch] - 1
    }

    /// `AddOffset(offset, dimension)` (`stridemap.cpp:67-71`): move, recompute
    /// `t`, report validity. NOTE: mirrors C++ in mutating even when the result
    /// is invalid — callers use a cloned index for probing.
    pub fn add_offset(&mut self, offset: i32, dim: FlexDim) -> bool {
        self.indices[dim as usize] += offset;
        self.set_t_from_indices();
        self.is_valid()
    }

    /// `Increment()` (`stridemap.cpp:75-87`): ragged row-major advance with
    /// carry; returns `false` when iteration is complete.
    pub fn increment(&mut self) -> bool {
        for d in (0..DIMSIZE).rev() {
            let dim = DIMS[d];
            if !self.is_last(dim) {
                self.t += self.map.t_increments[d];
                self.indices[d] += 1;
                return true;
            }
            self.t -= self.map.t_increments[d] * self.indices[d];
            self.indices[d] = 0;
            // carry to the next dimension.
        }
        false
    }

    /// `Decrement()` (`stridemap.cpp:92-110`): the reverse walk; a batch
    /// borrow re-initialises to the last of the new batch (per-image bounds
    /// change with the batch).
    pub fn decrement(&mut self) -> bool {
        for d in (0..DIMSIZE).rev() {
            let dim = DIMS[d];
            if self.indices[d] > 0 {
                self.indices[d] -= 1;
                if dim == FlexDim::Batch {
                    self.init_to_last_of_batch(self.indices[d]);
                } else {
                    self.t -= self.map.t_increments[d];
                }
                return true;
            }
            self.indices[d] = self.max_index_of_dim(dim);
            self.t += self.map.t_increments[d] * self.indices[d];
            // borrow from the next dimension.
        }
        false
    }

    /// `InitToLastOfBatch(batch)` (`stridemap.cpp:114-120`).
    fn init_to_last_of_batch(&mut self, batch: i32) {
        self.indices[FlexDim::Batch as usize] = batch;
        for (d, &dim) in DIMS.iter().enumerate().skip(FlexDim::Batch as usize + 1) {
            self.indices[d] = self.max_index_of_dim(dim);
        }
        self.set_t_from_indices();
    }

    /// `SetTFromIndices` (`stridemap.cpp:123-128`).
    fn set_t_from_indices(&mut self) {
        self.t = (0..DIMSIZE)
            .map(|d| self.map.t_increments[d] * self.indices[d])
            .sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ragged 3-image batch used across the A1 tests: (h,w) = (5,7),(3,4),(4,6)
    /// → shape B=3, H=5, W=7, Width() = 105.
    fn ragged() -> StrideMap {
        let mut m = StrideMap::default();
        m.set_stride(&[(5, 7), (3, 4), (4, 6)]);
        m
    }

    #[test]
    fn shape_and_width() {
        let m = ragged();
        assert_eq!(m.size(FlexDim::Batch), 3);
        assert_eq!(m.size(FlexDim::Height), 5);
        assert_eq!(m.size(FlexDim::Width), 7);
        // t_increments = [35, 7, 1]; Width = 35*3.
        assert_eq!(m.width(), 105);
    }

    #[test]
    fn increment_visits_only_valid_ragged_cells() {
        // Valid cells: 5*7 + 3*4 + 4*6 = 35 + 12 + 24 = 71 (not 105 — the walk
        // honours the per-image true sizes).
        let m = ragged();
        let mut idx = m.index_first();
        let mut count = 1;
        let mut ts = vec![idx.t()];
        while idx.increment() {
            count += 1;
            ts.push(idx.t());
        }
        assert_eq!(count, 71);
        // Image 1 (b=1) starts at t = 35 and is 4 wide: 35,36,37,38 then next
        // row jumps by the full stride 7 to 42.
        let pos35 = ts.iter().position(|&t| t == 35).unwrap();
        assert_eq!(&ts[pos35..pos35 + 5], &[35, 36, 37, 38, 42]);
    }

    #[test]
    fn decrement_reverses_increment() {
        let m = ragged();
        let mut fwd = vec![];
        let mut idx = m.index_first();
        fwd.push(idx.t());
        while idx.increment() {
            fwd.push(idx.t());
        }
        let mut rev = vec![];
        let mut idx = m.index_last();
        rev.push(idx.t());
        while idx.decrement() {
            rev.push(idx.t());
        }
        rev.reverse();
        assert_eq!(fwd, rev, "Decrement from last visits the same cells");
    }

    #[test]
    fn add_offset_bounds_are_per_image() {
        let m = ragged();
        // In image 1 (4 wide), x=3 is the last valid column: +1 is invalid
        // even though the shape width is 7.
        let mut idx = m.index_at(1, 0, 3);
        assert!(idx.is_valid());
        assert!(!idx.add_offset(1, FlexDim::Width));
        // In image 0 (7 wide) the same move from x=3 is fine.
        let mut idx = m.index_at(0, 0, 3);
        assert!(idx.add_offset(1, FlexDim::Width));
        assert_eq!(idx.t(), 4);
    }

    #[test]
    fn scale_and_transpose_match_cpp_integer_semantics() {
        let mut m = ragged();
        m.scale_xy(2, 2);
        // heights 5,3,4 -> 2,1,2 ; widths 7,4,6 -> 3,2,3 ; shape H=2, W=3.
        assert_eq!(m.size(FlexDim::Height), 2);
        assert_eq!(m.size(FlexDim::Width), 3);
        assert_eq!(m.width(), 3 * 2 * 3);
        let mut t = ragged();
        t.transpose_xy();
        assert_eq!(t.size(FlexDim::Height), 7);
        assert_eq!(t.size(FlexDim::Width), 5);
        // Per-image sizes swapped too: image 1 is now 4 wide -> height... i.e.
        // heights==old widths: max_index checks confirm via index_at probing.
        let idx = t.index_at(1, 3, 2);
        assert!(idx.is_valid(), "(y,x)=(3,2) valid: h=4(old w), w=3(old h)");
        let idx2 = t.index_at(1, 0, 3);
        assert!(!idx2.is_valid(), "x=3 exceeds image 1's new width 3");
    }
}
