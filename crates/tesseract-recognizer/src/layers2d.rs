//! The 2-D front-end layer forwards — recognizer Leaves A2-A5: `Convolve`,
//! `Maxpool`, `Reconfig` (and the `Txy` transpose, which is
//! [`NetworkIo::copy_with_xy_transpose`] directly). Transcodes of
//! `lstm/{convolve,maxpool,reconfig}.cpp::Forward`, the layers that turn the
//! image grid into the feature sequence the 1-D LSTM core consumes
//! (eng.lstm spec `[1,36,0,1[C3,3Ft16]Mp3,3Txy...]`).
//!
//! All three are pure [`StrideMap`](crate::stridemap::StrideMap) + row-copy
//! choreography over [`NetworkIo`] — no arithmetic beyond `Maxpool`'s
//! elementwise max — so they work in int8 and float modes alike, exactly as
//! C++ (the mode rides on the input).
//!
//! Byte-parity notes:
//! - `Convolve` fills **out-of-image** window cells with
//!   `randomizer->SignedRand(...)` noise (`convolve.cpp:68/75`) — the caller
//!   passes the [`TRand`] whose seed/warm-up matches the C++ recognizer's.
//!   The draw ORDER is the dest-index walk order and is part of parity.
//! - `Maxpool` seeds each output cell by copying the window's origin timestep
//!   (`maxpool.cpp:53`), then running-maxes the (x,y)-offset window cells in
//!   x-major order (`maxpool.cpp:57-64`) with strict `<` ties-keep-first.
//! - `Reconfig` stacks `x_scale × y_scale` window cells depth-wise at offset
//!   `(x·y_scale + y)·ni` (`reconfig.cpp:80-87`); out-of-image window cells
//!   are simply skipped (the dest was zeroed by the resize — C++ relies on
//!   `ResizeToMap`'s `ZeroInvalidElements` + uninitialized-but-never-read
//!   cells; on ragged batches the skipped cells stay zero on both sides).

use crate::networkio::NetworkIo;
use crate::stridemap::FlexDim;
use crate::trand::TRand;

/// `Convolve::Forward` (`convolve.cpp:55-88`): stack the
/// `(2·half_x+1) × (2·half_y+1) × ni` window around every position into the
/// output depth; out-of-image columns/rows are filled with randomizer noise.
/// `ni` must be the input's feature count; the output has
/// `ni·(2·half_x+1)·(2·half_y+1)` features on the SAME stride map.
#[must_use]
pub fn convolve_forward(
    input: &NetworkIo,
    half_x: i32,
    half_y: i32,
    randomizer: &mut TRand,
) -> NetworkIo {
    let ni = input.num_features();
    let y_scale = (2 * half_y + 1) as usize;
    let no = ni * (2 * half_x + 1) as usize * y_scale;
    let mut output = NetworkIo::default();
    output.resize_like(input, no);
    let dest_map = output.stride_map().clone();
    let mut dest_index = dest_map.index_first();
    loop {
        let t = dest_index.t() as usize;
        let mut out_ix = 0_usize;
        for x in -half_x..=half_x {
            let mut x_index = dest_index.clone();
            if !x_index.add_offset(x, FlexDim::Width) {
                // This x is outside the image: noise-fill the whole column band.
                output.randomize(t, out_ix, y_scale * ni, randomizer);
            } else {
                let mut out_iy = out_ix;
                for y in -half_y..=half_y {
                    let mut y_index = x_index.clone();
                    if !y_index.add_offset(y, FlexDim::Height) {
                        // This y is outside the image: noise-fill one cell band.
                        output.randomize(t, out_iy, ni, randomizer);
                    } else {
                        output.copy_time_step_general(
                            t,
                            out_iy,
                            ni,
                            input,
                            y_index.t() as usize,
                            0,
                        );
                    }
                    out_iy += ni;
                }
            }
            out_ix += y_scale * ni;
        }
        if !dest_index.increment() {
            break;
        }
    }
    output
}

/// `Maxpool::Forward` (`maxpool.cpp:37-66`): downscale by `x_scale × y_scale`,
/// keeping the elementwise max per feature over each window (strict `<`:
/// ties keep the window-origin/earlier cell). Output has the input's feature
/// count on the scaled stride map. The per-cell argmax record (`maxes_`) is
/// training-side (backward) state and is not kept.
#[must_use]
pub fn maxpool_forward(input: &NetworkIo, x_scale: i32, y_scale: i32) -> NetworkIo {
    let ni = input.num_features();
    let mut output = NetworkIo::default();
    output.resize_scaled(input, x_scale, y_scale, ni);
    let dest_map = output.stride_map().clone();
    let in_map = input.stride_map().clone();
    let mut max_line = vec![0_i32; ni];
    let mut dest_index = dest_map.index_first();
    loop {
        let out_t = dest_index.t() as usize;
        let src_index = in_map.index_at(
            dest_index.index(FlexDim::Batch),
            dest_index.index(FlexDim::Height) * y_scale,
            dest_index.index(FlexDim::Width) * x_scale,
        );
        let in_t = src_index.t() as usize;
        output.copy_time_step_from(out_t, input, in_t);
        max_line.fill(in_t as i32);
        for x in 0..x_scale {
            for y in 0..y_scale {
                let mut src_xy = src_index.clone();
                if src_xy.add_offset(x, FlexDim::Width) && src_xy.add_offset(y, FlexDim::Height) {
                    output.maxpool_time_step(out_t, input, src_xy.t() as usize, &mut max_line);
                }
            }
        }
        if !dest_index.increment() {
            break;
        }
    }
    output
}

/// `Reconfig::Forward` (`reconfig.cpp:69-89`): the `Ft` scale-and-deepen —
/// shrink x/y by the scale factors, stacking each `x_scale × y_scale` window
/// depth-wise (WITHOUT maxing) at offset `(x·y_scale + y)·ni`. Output has
/// `ni·x_scale·y_scale` features on the scaled stride map; window cells that
/// fall outside a ragged image stay zero.
#[must_use]
pub fn reconfig_forward(input: &NetworkIo, x_scale: i32, y_scale: i32) -> NetworkIo {
    let ni = input.num_features();
    let no = ni * (x_scale * y_scale) as usize;
    let mut output = NetworkIo::default();
    output.resize_scaled(input, x_scale, y_scale, no);
    let dest_map = output.stride_map().clone();
    let in_map = input.stride_map().clone();
    let mut dest_index = dest_map.index_first();
    loop {
        let out_t = dest_index.t() as usize;
        let src_index = in_map.index_at(
            dest_index.index(FlexDim::Batch),
            dest_index.index(FlexDim::Height) * y_scale,
            dest_index.index(FlexDim::Width) * x_scale,
        );
        for x in 0..x_scale {
            for y in 0..y_scale {
                let mut src_xy = src_index.clone();
                if src_xy.add_offset(x, FlexDim::Width) && src_xy.add_offset(y, FlexDim::Height) {
                    output.copy_time_step_general(
                        out_t,
                        ((x * y_scale + y) * ni as i32) as usize,
                        ni,
                        input,
                        src_xy.t() as usize,
                        0,
                    );
                }
            }
        }
        if !dest_index.increment() {
            break;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stridemap::StrideMap;

    /// A single 4x6 image, nf=2, int8, cells filled t-deterministically.
    fn image_4x6() -> NetworkIo {
        let mut map = StrideMap::default();
        map.set_stride(&[(4, 6)]);
        let mut io = NetworkIo::default();
        io.resize_to_map(true, &map, 2);
        let walk = map.clone();
        let mut idx = walk.index_first();
        loop {
            let t = idx.t() as usize;
            io.write_time_step(
                t,
                &[
                    (t as i32 % 100) as f32 / 100.0,
                    -((t as i32 % 50) as f32) / 100.0,
                ],
            );
            if !idx.increment() {
                break;
            }
        }
        io
    }

    #[test]
    fn convolve_center_window_copies_input() {
        // half_x=half_y=1 -> no = 2*9 = 18; the CENTER band (x=0,y=0) of every
        // interior cell equals the input row. Center offset: x-index 1 of 3 ->
        // out_ix = 1*3*2 = 6, y-index 1 -> +2 => 8.
        let input = image_4x6();
        let mut rng = TRand::default();
        rng.set_seed(7);
        let out = convolve_forward(&input, 1, 1, &mut rng);
        assert_eq!(out.num_features(), 18);
        assert_eq!(out.width(), input.width());
        // Interior cell (y=1,x=1) -> t = 1*6+1 = 7.
        let t = 7_usize;
        assert_eq!(&out.i(t)[8..10], input.i(t));
        // Its left neighbour band (x=-1 -> out_ix 0, y=0 -> +2): input t-1.
        assert_eq!(&out.i(t)[2..4], input.i(t - 1));
        // Its up neighbour (x=0 band start 6, y=-1 -> offset 6): input t-6.
        assert_eq!(&out.i(t)[6..8], input.i(t - 6));
    }

    #[test]
    fn convolve_edge_uses_randomizer_noise() {
        // At (0,0), the x=-1 column band and y=-1 cells are noise-filled; the
        // draws must match a hand-replayed TRand in walk order.
        let input = image_4x6();
        let mut rng = TRand::default();
        rng.set_seed(7);
        let out = convolve_forward(&input, 1, 1, &mut rng);
        let mut replay = TRand::default();
        replay.set_seed(7);
        // First dest cell t=0: x=-1 invalid -> 6 noise draws fill [0..6).
        let expect: Vec<i8> = (0..6)
            .map(|_| {
                let v = replay.signed_rand(127.0);
                (if v >= 0.0 {
                    (v + 0.5) as i32
                } else {
                    -((-v + 0.5) as i32)
                }) as i8
            })
            .collect();
        assert_eq!(&out.i(0)[0..6], expect.as_slice());
    }

    #[test]
    fn maxpool_takes_window_max() {
        let input = image_4x6();
        let out = maxpool_forward(&input, 2, 2);
        // 4x6 / 2x2 -> 2x3, nf unchanged.
        assert_eq!(out.width(), 6);
        assert_eq!(out.num_features(), 2);
        // Window for out (0,0): input cells t=0,1,6,7. Feature 0 grows with t
        // -> max is t=7's value; feature 1 shrinks (negative) -> max is t=0's.
        let f0_max = input.i(7)[0];
        let f1_max = input.i(0)[1];
        assert_eq!(out.i(0), &[f0_max, f1_max]);
    }

    #[test]
    fn reconfig_stacks_windows_depthwise() {
        let input = image_4x6();
        let out = reconfig_forward(&input, 2, 2);
        // no = 2*4 = 8; window (x,y) lands at (x*2+y)*2.
        assert_eq!(out.num_features(), 8);
        assert_eq!(out.width(), 6);
        // out (0,0): [in(0,0), in(1,0), in(0,1), in(1,1)] = t 0, 6, 1, 7.
        let row = out.i(0);
        assert_eq!(&row[0..2], input.i(0)); // x=0,y=0
        assert_eq!(&row[2..4], input.i(6)); // x=0,y=1
        assert_eq!(&row[4..6], input.i(1)); // x=1,y=0
        assert_eq!(&row[6..8], input.i(7)); // x=1,y=1
    }
}
