//! Recognizer **A6a**: `NetworkIO::FromPix` (`networkio.cpp:127-251`) — the
//! image-pixel → int8 grid step, with the `ComputeBlackWhite` contrast
//! normalization. This is the pixel-facing half of the leptonica front-end;
//! the image DECODE + `pixScale`-to-target-height (A6b) is a separate,
//! commodity concern deliberately kept out of the byte-parity surface (its
//! byte-exactness is leptonica's `pixScale`, not a Tesseract algorithm).
//!
//! **Scope:** the 8-bit **grey**, 2-D path (`shape.height() > 1`, `depth == 1`)
//! — the `eng.lstm` case (`[1,36,0,1…]`). The caller supplies an 8-bit grey
//! image ALREADY at the target height (so C++ `PreparePixInput` does no scale /
//! no depth-convert, and `FromPix` runs directly). The 1-D vertical-strip path
//! (`shape.height() == 1`) and the colour path (`depth == 3`) are not part of
//! this leaf.
//!
//! ## The transcoded algorithm (all from `networkio.cpp`)
//! - `ComputeBlackWhite` (127): the MIDDLE row's local minima/maxima →
//!   two `STATS(0,255)` histograms; `black = mins.ile(0.25)`,
//!   `white = maxes.ile(0.75)`.
//! - `contrast = (white − black) / 2`, clamped `>= 1`.
//! - `Copy2DImage` (216): walk `y ∈ [0,target_height)`, `x ∈ [0,width)`
//!   row-major, `set_pixel` each grey byte; pad `x ∈ [width,target_width)`
//!   with `randomize`.
//! - `SetPixel` (290): `clip(round(128·((pixel−black)/contrast − 1)), ±127)`.

use crate::networkio::NetworkIo;
use crate::stridemap::StrideMap;
use crate::trand::TRand;

/// A `STATS(0, 255)` histogram (`statistc.{h,cpp}`) — 256 integer buckets over
/// `[0, 255]`, the exact shape `ComputeBlackWhite` uses.
struct Stats {
    buckets: [i32; 256],
    total: i32,
}

impl Stats {
    fn new() -> Self {
        Self {
            buckets: [0; 256],
            total: 0,
        }
    }

    /// `STATS::add(value, 1)` — bump the bucket for `value ∈ [0, 255]`.
    fn add(&mut self, value: i32) {
        let idx = value.clamp(0, 255) as usize;
        self.buckets[idx] += 1;
        self.total += 1;
    }

    /// `STATS::ile(frac)` (`statistc.cpp:172-197`): the fractile value such that
    /// `frac` of the samples are below it — a bucket walk + linear interpolation
    /// within the crossing bucket. `rangemin_ = 0`.
    fn ile(&self, frac: f64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut target = frac * f64::from(self.total);
        target = target.clamp(1.0, f64::from(self.total));
        // for (index=0; index <= 255 && sum < target; sum += buckets[index++]) ;
        let mut sum = 0_i32;
        let mut index = 0_usize;
        while index <= 255 && f64::from(sum) < target {
            sum += self.buckets[index];
            index += 1;
        }
        if index > 0 {
            // rangemin(0) + index - (sum - target)/buckets[index-1]
            index as f64 - (f64::from(sum) - target) / f64::from(self.buckets[index - 1])
        } else {
            0.0
        }
    }
}

/// `ComputeBlackWhite` (`networkio.cpp:127-158`): the per-image contrast levels
/// from the middle row's local extrema. Returns `(black, white)` as `f32` (the
/// C++ narrows `ile`'s double to `float *black`/`white`).
fn compute_black_white(grey: &[u8], width: usize, height: usize) -> (f32, f32) {
    let mut mins = Stats::new();
    let mut maxes = Stats::new();
    if width >= 3 {
        let y = height / 2;
        let row = &grey[y * width..y * width + width];
        let mut prev = i32::from(row[0]);
        let mut curr = i32::from(row[1]);
        // for (int x = 1; x + 1 < width; ++x)
        for x in 1..width - 1 {
            let next = i32::from(row[x + 1]);
            if (curr < prev && curr <= next) || (curr <= prev && curr < next) {
                mins.add(curr); // local minimum
            }
            if (curr > prev && curr >= next) || (curr >= prev && curr > next) {
                maxes.add(curr); // local maximum
            }
            prev = curr;
            curr = next;
        }
    }
    if mins.total == 0 {
        mins.add(0);
    }
    if maxes.total == 0 {
        maxes.add(255);
    }
    (mins.ile(0.25) as f32, maxes.ile(0.75) as f32)
}

/// `NetworkIO::FromPix` for an 8-bit grey image on the 2-D path
/// (`FromPixes` + `Copy2DImage`, `networkio.cpp:171-251`): build the int8 grid
/// from `grey` (`height × width`, row-major, one byte per pixel).
///
/// `target_height` / `target_width` are the network `StaticShape`'s dims (0 =
/// "use the image's"). eng: `target_height = 36`, `target_width = 0`. `rng`
/// feeds the width-padding noise (`randomize`) when `target_width > width`.
///
/// # Panics
///
/// Panics if `target_height == 1` (the 1-D vertical-strip path is out of scope)
/// or if `grey.len() < width·height`.
#[must_use]
pub fn from_grey_pix(
    grey: &[u8],
    width: usize,
    height: usize,
    target_height: i32,
    target_width: i32,
    rng: &mut TRand,
) -> NetworkIo {
    assert!(
        target_height != 1,
        "1-D vertical-strip path is out of A6a scope"
    );
    assert!(grey.len() >= width * height, "grey buffer too small");

    // FromPixes: stride dims are the target dims, or the image's when 0.
    let grid_h = if target_height != 0 {
        target_height
    } else {
        height as i32
    };
    let grid_w = if target_width != 0 {
        target_width
    } else {
        width as i32
    };
    let mut map = StrideMap::default();
    map.set_stride(&[(grid_h, grid_w)]);
    let mut out = NetworkIo::default();
    out.resize_to_map(true, &map, 1); // grey => depth/num_features = 1

    let (black, white) = compute_black_white(grey, width, height);
    let mut contrast = (white - black) / 2.0;
    if contrast <= 0.0 {
        contrast = 1.0;
    }

    // Copy2DImage: y-outer, x-inner, t incrementing (== the stride map's
    // row-major (y,x) order for a single image, so plain t++ tracks index.t()).
    let target_h = map.size(crate::stridemap::FlexDim::Height);
    let target_w = map.size(crate::stridemap::FlexDim::Width);
    let copy_w = (width as i32).min(target_w); // if width > target_width: width = target_width
    let mut t = 0_usize;
    for y in 0..target_h {
        let mut x = 0_i32;
        if y < height as i32 {
            while x < copy_w {
                let pixel = i32::from(grey[y as usize * width + x as usize]);
                out.set_pixel(t, 0, pixel, black, contrast);
                t += 1;
                x += 1;
            }
        }
        while x < target_w {
            out.randomize(t, 0, 1, rng); // num_features = 1 (grey)
            t += 1;
            x += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_ile_matches_cpp_bucket_walk() {
        // 4 samples at values 10,20,20,30. total=4. ile(0.25): target=1.0.
        // walk: buckets[10]=1 -> sum=1 >= 1 at index=11. return 11 - (1-1)/1 = 11.
        let mut s = Stats::new();
        for v in [10, 20, 20, 30] {
            s.add(v);
        }
        assert!((s.ile(0.25) - 11.0).abs() < 1e-12);
        // ile(0.75): target=3.0. sum after bucket 10 =1, after 20 =3 >=3 at
        // index=21. return 21 - (3-3)/2 = 21.
        assert!((s.ile(0.75) - 21.0).abs() < 1e-12);
        // empty stats -> rangemin (0).
        assert_eq!(Stats::new().ile(0.5), 0.0);
    }

    #[test]
    fn compute_black_white_middle_row_extrema() {
        // 5-wide, 3-tall; middle row (y=1) = [50, 10, 90, 10, 50]. Walk x=1..3:
        // x=1 curr=10 (prev50,next90): local min -> mins{10}. x=2 curr=90
        // (prev10,next10): local max -> maxes{90}. x=3 curr=10 (prev90,next50):
        // local min -> mins{10,10}. black=mins.ile(.25), white=maxes.ile(.75).
        let mut grey = vec![0_u8; 15];
        grey[5..10].copy_from_slice(&[50, 10, 90, 10, 50]);
        let (black, white) = compute_black_white(&grey, 5, 3);
        // mins has two 10s: ile(0.25) target=clamp(0.5,1,2)=1 -> bucket10 sum2>=1
        // at index11 -> 11 - (2-1)/2 = 10.5. white: maxes one 90, ile(.75)
        // target=clamp(.75,1,1)=1 -> 91 - (1-1)/1 = 91.
        assert!((black - 10.5).abs() < 1e-4, "black={black}");
        assert!((white - 91.0).abs() < 1e-4, "white={white}");
    }

    #[test]
    fn from_grey_pix_shapes_the_grid() {
        // 4-wide x 3-tall grey, target_height=3 (== image, no row pad),
        // target_width=0 (== image width, no col pad). Grid = 3x4x1 int8.
        let grey: Vec<u8> = (0..12).map(|i| (i * 20) as u8).collect();
        let mut rng = TRand::default();
        rng.set_seed(1);
        let io = from_grey_pix(&grey, 4, 3, 3, 0, &mut rng);
        assert_eq!(io.num_features(), 1);
        assert_eq!(io.width(), 12, "3*4 timesteps");
        assert!(io.int_mode());
    }
}
