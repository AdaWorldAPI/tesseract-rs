//! Otsu-threshold chain: `OtsuThreshold → {HistogramRect, OtsuStats}`
//! (`/tmp/tesseract/src/ccstruct/otsuthr.{h,cpp}`), plus the consumer
//! `ImageThresholder::ThresholdRectToPix` (`ccmain/thresholder.cpp:394-421`)
//! that turns the per-channel `(threshold, hi_value)` pairs into a bitonal
//! image.
//!
//! Per the `ruff_cpp_spo` intra-TU dispatch manifest banked at
//! `.claude/harvest/leptonica-scale-callgraph.txt` (otsuthr section):
//! `OtsuThreshold` dispatches to `HistogramRect` and `OtsuStats`; both
//! callees are LEAVES — `HistogramRect` has zero in-TU calls (only
//! leptonica pixel-access macros), `OtsuStats` has zero calls at all
//! (pure math). This module transcodes all three, scoped to the 8-bit
//! **grey** (1-channel) path used by the LSTM line-recognition pipeline;
//! the loop structure is written generically over `num_channels` so a
//! multi-channel (e.g. RGB) caller is a thin wrapper, matching the C++
//! shape where `OtsuThreshold`/`HistogramRect` are already channel-generic.
//!
//! ## Byte-for-byte fidelity notes
//! - `OtsuStats` accumulates `mu_T`/`mu_t` in `f64` (C++ `double`) while
//!   `H`/`omega_0`/`omega_1` stay `i32` (C++ `int`) — mixed-precision
//!   exactly as the source, per the `f64`-precision-audit lesson from
//!   `E-OCR-PIXSCALE-COMPLETE-1`.
//! - `OtsuThreshold`'s fraction tests (`H * 0.5`, `H * 0.75`, `H * 0.25`)
//!   promote the `i32 H`/`best_omega_0` to `f64` for the comparison,
//!   exactly as C++'s usual arithmetic conversions do for `int OP double`.
//! - `ThresholdRectToPix::pixel > threshold` uses **strict** `>`, and the
//!   per-pixel decision is `(pixel > threshold) == (hi_value == 0)` —
//!   transcoded verbatim, including the `hi_value < 0` "this channel has
//!   no opinion" skip.
//!
//! ## Output convention (this crate's choice, not a C++ literal)
//! Leptonica's 1bpp output pix uses `SET_DATA_BIT` (bit = 1) for
//! `!white_result` (foreground/text) and `CLEAR_DATA_BIT` (bit = 0) for
//! `white_result` (background). This crate represents a bitonal image as
//! `&[u8]` with **0 = foreground/black, 255 = background/white**, matching
//! the grey-image convention used throughout `tesseract-recognizer`
//! (`from_grey_pix`, `pix_scale_grey`). `white_result == true` therefore
//! maps to `255`, `false` to `0` — the inverse of the leptonica bit but the
//! same *semantic* image.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "pixel bytes are 0..=255; histogram indices and counts stay in i32/usize range for any real image"
)]

/// Size of a pixel-value histogram (`kHistogramSize` in `otsuthr.h`).
pub const HISTOGRAM_SIZE: usize = 256;

/// The result of [`otsu_stats`]: `OtsuStats`'s three outputs
/// (`otsuthr.cpp:118-160`) — the return value `best_t`, and the two
/// out-parameters `H` and `omega0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OtsuStatsResult {
    /// `best_t` — the return value: the threshold index that maximizes the
    /// between-class variance, or `-1` if the histogram is degenerate
    /// (never actually reached with a non-empty histogram, since the loop
    /// always finds at least one `t` with `omega_0 != 0 && omega_1 != 0`
    /// once `H > histogram[0]` and `H > histogram[last]`... but transcoded
    /// verbatim rather than assumed).
    pub best_t: i32,
    /// `*H_out` — total count in the histogram.
    pub h: i32,
    /// `*omega0_out` — count of histogram entries at or below `best_t`.
    pub best_omega_0: i32,
}

/// One channel's Otsu decision, as produced by [`otsu_threshold_gray`] /
/// [`otsu_threshold_channels`] — mirrors `thresholds[ch]` / `hi_values[ch]`
/// from `OtsuThreshold` (`otsuthr.cpp:34-84`).
///
/// `threshold == -1` means the channel was empty (`best_omega_0 == 0` or
/// `== H`) and carries no thresholding information. `hi_value == -1` means
/// "no opinion" for the final bitonal decision (see the module docs for the
/// `hi_value` semantics); `threshold_rect_to_binary` treats a `-1` `hi_value`
/// as "this channel never overrides the default `white_result`".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OtsuChannel {
    /// The Otsu threshold for this channel, or `-1` if the channel is empty.
    pub threshold: i32,
    /// `0` = pixel value `> threshold` is foreground, `1` = pixel value
    /// `> threshold` is background, `-1` = no opinion.
    pub hi_value: i32,
}

/// The multi-channel result of [`otsu_threshold_channels`] — mirrors the
/// `thresholds`/`hi_values` out-vectors of `OtsuThreshold`.
#[derive(Debug, Clone)]
pub struct OtsuResult {
    /// Per-channel thresholds (`-1` = empty channel).
    pub thresholds: Vec<i32>,
    /// Per-channel hi-value decisions (`-1` = no opinion).
    pub hi_values: Vec<i32>,
}

/// `HistogramRect` (`otsuthr.cpp:88-105`), generalized over `num_channels`
/// exactly as the C++ original (`GET_DATA_BYTE(linedata, (x + left) *
/// num_channels + channel)`): counts occurrences of each of the 256
/// possible byte values for one `channel` of an interleaved `num_channels`
/// byte-per-pixel image, over the rectangle
/// `[left, left+width) x [top, top+height)`.
///
/// `img_w` is the FULL image width (the row stride in pixels, i.e. what
/// `src_wpl`/`num_channels` corresponds to for a byte-packed buffer) — NOT
/// the rectangle width. `channel` is clipped to `[0, num_channels - 1]`
/// exactly as `ClipToRange(channel, 0, num_channels - 1)`.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "faithful transcode of HistogramRect's rectangle + channel parameterization"
)]
pub fn histogram_rect_multi(
    buf: &[u8],
    img_w: usize,
    num_channels: usize,
    channel: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
) -> [i32; HISTOGRAM_SIZE] {
    let channel = channel.min(num_channels.saturating_sub(1));
    let mut histogram = [0i32; HISTOGRAM_SIZE];
    for y in top..top + height {
        let row_base = y * img_w;
        for x in 0..width {
            let idx = (row_base + x + left) * num_channels + channel;
            let pixel = buf[idx];
            histogram[pixel as usize] += 1;
        }
    }
    histogram
}

/// `HistogramRect` scoped to an 8-bit grey (`num_channels == 1`) buffer —
/// the leaf used by the LSTM line-recognition front end. `channel` is
/// always `0` (clipped from `num_channels - 1 == 0`).
#[must_use]
pub fn histogram_rect_gray(
    grey: &[u8],
    img_w: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
) -> [i32; HISTOGRAM_SIZE] {
    histogram_rect_multi(grey, img_w, 1, 0, left, top, width, height)
}

/// `HistogramRect` scoped to an interleaved 3-channel (RGB) buffer, one
/// byte per channel per pixel (`num_channels == 3`).
#[must_use]
pub fn histogram_rect_rgb(
    rgb: &[u8],
    img_w: usize,
    channel: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
) -> [i32; HISTOGRAM_SIZE] {
    histogram_rect_multi(rgb, img_w, 3, channel, left, top, width, height)
}

/// `OtsuStats` (`otsuthr.cpp:118-160`) — the pure between-class-variance
/// argmax over a single-channel histogram. Zero external calls (a LEAF per
/// the harvest manifest); transcoded statement-for-statement including the
/// `i32`/`f64` mixed precision.
#[must_use]
pub fn otsu_stats(histogram: &[i32; HISTOGRAM_SIZE]) -> OtsuStatsResult {
    let mut h: i32 = 0;
    let mut mu_t_total: f64 = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        h += count;
        mu_t_total += i as f64 * f64::from(count);
    }

    // Now maximize sig_sq_B over t.
    let mut best_t: i32 = -1;
    let mut best_omega_0: i32 = 0;
    let mut best_sig_sq_b: f64 = 0.0;
    let mut omega_0: i32 = 0;
    let mut mu_t: f64 = 0.0;
    for (t, &count) in histogram.iter().enumerate().take(HISTOGRAM_SIZE - 1) {
        omega_0 += count;
        mu_t += t as f64 * f64::from(count);
        if omega_0 == 0 {
            continue;
        }
        let omega_1 = h - omega_0;
        if omega_1 == 0 {
            break;
        }
        let mu_0 = mu_t / f64::from(omega_0);
        let mu_1 = (mu_t_total - mu_t) / f64::from(omega_1);
        let mut sig_sq_b = mu_1 - mu_0;
        sig_sq_b = sig_sq_b * sig_sq_b * f64::from(omega_0) * f64::from(omega_1);
        if best_t < 0 || sig_sq_b > best_sig_sq_b {
            best_sig_sq_b = sig_sq_b;
            best_t = t as i32;
            best_omega_0 = omega_0;
        }
    }

    OtsuStatsResult {
        best_t,
        h,
        best_omega_0,
    }
}

/// `OtsuThreshold`'s post-histogram decision logic (`otsuthr.cpp:34-84`),
/// generalized over a slice of already-computed per-channel histograms.
/// Transcoded verbatim, including the "keep the best of the bad lot" cross-
/// channel bookkeeping (`best_hi_value`/`best_hi_index`/`best_hi_dist`) that
/// only matters when every channel is inconclusive.
#[must_use]
pub fn otsu_threshold_channels(histograms: &[[i32; HISTOGRAM_SIZE]]) -> OtsuResult {
    let num_channels = histograms.len();
    let mut thresholds = vec![-1i32; num_channels];
    let mut hi_values = vec![-1i32; num_channels];
    let mut best_hi_value: i32 = 1;
    let mut best_hi_index: usize = 0;
    let mut any_good_hivalue = false;
    let mut best_hi_dist: f64 = 0.0;

    for (ch, histogram) in histograms.iter().enumerate() {
        let stats = otsu_stats(histogram);
        if stats.best_omega_0 == 0 || stats.best_omega_0 == stats.h {
            // This channel is empty.
            continue;
        }
        // To be a convincing foreground we must have a small fraction of H
        // or to be a convincing background we must have a large fraction
        // of H. In between we assume this channel contains no
        // thresholding information.
        let hi_value_flag = f64::from(stats.best_omega_0) < f64::from(stats.h) * 0.5;
        thresholds[ch] = stats.best_t;
        if f64::from(stats.best_omega_0) > f64::from(stats.h) * 0.75 {
            any_good_hivalue = true;
            hi_values[ch] = 0;
        } else if f64::from(stats.best_omega_0) < f64::from(stats.h) * 0.25 {
            any_good_hivalue = true;
            hi_values[ch] = 1;
        } else {
            // In case all channels are like this, keep the best of the bad lot.
            let hi_dist = if hi_value_flag {
                f64::from(stats.h - stats.best_omega_0)
            } else {
                f64::from(stats.best_omega_0)
            };
            if hi_dist > best_hi_dist {
                best_hi_dist = hi_dist;
                best_hi_value = i32::from(hi_value_flag);
                best_hi_index = ch;
            }
        }
    }

    if !any_good_hivalue {
        // Use the best of the ones that were not good enough.
        hi_values[best_hi_index] = best_hi_value;
    }

    OtsuResult {
        thresholds,
        hi_values,
    }
}

/// `OtsuThreshold` scoped to a single 8-bit grey channel: computes the
/// histogram via [`histogram_rect_gray`] then runs
/// [`otsu_threshold_channels`] over the one-element histogram list.
#[must_use]
pub fn otsu_threshold_gray(
    grey: &[u8],
    img_w: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
) -> OtsuChannel {
    let histogram = histogram_rect_gray(grey, img_w, left, top, width, height);
    let result = otsu_threshold_channels(std::slice::from_ref(&histogram));
    OtsuChannel {
        threshold: result.thresholds[0],
        hi_value: result.hi_values[0],
    }
}

/// `ImageThresholder::ThresholdRectToPix` (`ccmain/thresholder.cpp:394-421`),
/// generalized over `num_channels`, transcoded per-pixel exactly:
/// `white_result` starts `true`; for each channel with `hi_values[ch] >= 0`,
/// if `(pixel > thresholds[ch]) == (hi_values[ch] == 0)` the pixel is
/// foreground in that channel and `white_result` becomes `false` (any
/// channel voting foreground wins, first-match `break`).
///
/// Output convention (see module docs): `white_result == true` → `255`
/// (background/white), `false` → `0` (foreground/black) — the same
/// semantic image as leptonica's 1bpp `SET_DATA_BIT`/`CLEAR_DATA_BIT`, with
/// bit polarity inverted to match this crate's grey-image convention.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "faithful transcode of ThresholdRectToPix's rectangle + per-channel decision arrays"
)]
pub fn threshold_rect_to_binary_multi(
    buf: &[u8],
    img_w: usize,
    num_channels: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    thresholds: &[i32],
    hi_values: &[i32],
) -> Vec<u8> {
    assert_eq!(thresholds.len(), num_channels);
    assert_eq!(hi_values.len(), num_channels);
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        let src_row = (y + top) * img_w;
        for x in 0..width {
            let mut white_result = true;
            for ch in 0..num_channels {
                let idx = (src_row + x + left) * num_channels + ch;
                let pixel = i32::from(buf[idx]);
                if hi_values[ch] >= 0 && (pixel > thresholds[ch]) == (hi_values[ch] == 0) {
                    white_result = false;
                    break;
                }
            }
            out[y * width + x] = if white_result { 255 } else { 0 };
        }
    }
    out
}

/// `ThresholdRectToPix` scoped to a single 8-bit grey channel, taking the
/// channel's [`OtsuChannel`] decision directly.
#[must_use]
pub fn threshold_rect_to_binary(
    grey: &[u8],
    img_w: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    otsu: OtsuChannel,
) -> Vec<u8> {
    threshold_rect_to_binary_multi(
        grey,
        img_w,
        1,
        left,
        top,
        width,
        height,
        &[otsu.threshold],
        &[otsu.hi_value],
    )
}

#[cfg(test)]
mod tests {
    use super::{
        histogram_rect_gray, otsu_stats, otsu_threshold_gray, threshold_rect_to_binary,
        HISTOGRAM_SIZE,
    };

    #[test]
    fn otsu_stats_bimodal_finds_threshold_between_spikes() {
        // Two spikes: 100 pixels at value 10, 100 pixels at value 200.
        // The optimal threshold must fall strictly between the two spikes.
        let mut histogram = [0i32; HISTOGRAM_SIZE];
        histogram[10] = 100;
        histogram[200] = 100;
        let stats = otsu_stats(&histogram);
        assert_eq!(stats.h, 200);
        assert_eq!(stats.best_omega_0, 100);
        assert!(
            stats.best_t >= 10 && stats.best_t < 200,
            "best_t={} should split the two spikes",
            stats.best_t
        );
    }

    #[test]
    fn otsu_stats_uniform_histogram_runs_without_panicking() {
        // A perfectly uniform histogram: every bin has the same count. The
        // between-class variance is symmetric; there is no "correct" answer
        // to hardcode, but the routine must terminate with a valid t and H
        // must equal the sum of all bins (no channel is degenerate: the
        // best_omega_0 found along the way is never 0 or H because the loop
        // breaks the moment omega_1 hits 0, i.e. at t == 255 which is
        // outside the loop range 0..254).
        let histogram = [4i32; HISTOGRAM_SIZE];
        let stats = otsu_stats(&histogram);
        assert_eq!(stats.h, 4 * HISTOGRAM_SIZE as i32);
        assert!((0..HISTOGRAM_SIZE as i32).contains(&stats.best_t));
        assert!(stats.best_omega_0 > 0 && stats.best_omega_0 < stats.h);
    }

    #[test]
    fn otsu_stats_empty_histogram_never_finds_a_threshold() {
        let histogram = [0i32; HISTOGRAM_SIZE];
        let stats = otsu_stats(&histogram);
        assert_eq!(stats.h, 0);
        assert_eq!(stats.best_t, -1);
        assert_eq!(stats.best_omega_0, 0);
    }

    #[test]
    fn histogram_rect_counts_known_4x4_tile() {
        // A 4x4 grey image (img_w == 4, no cropping):
        //   row0: 0 0 1 1
        //   row1: 0 0 1 1
        //   row2: 2 2 3 3
        //   row3: 2 2 3 3
        // Full-rect histogram: value0->4, value1->4, value2->4, value3->4.
        #[rustfmt::skip]
        let grey: [u8; 16] = [
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3,
        ];
        let histogram = histogram_rect_gray(&grey, 4, 0, 0, 4, 4);
        assert_eq!(histogram[0], 4);
        assert_eq!(histogram[1], 4);
        assert_eq!(histogram[2], 4);
        assert_eq!(histogram[3], 4);
        assert_eq!(histogram.iter().sum::<i32>(), 16);

        // Now a 2x2 sub-rect at (left=2, top=2): should be all value 3.
        let sub = histogram_rect_gray(&grey, 4, 2, 2, 2, 2);
        assert_eq!(sub[3], 4);
        assert_eq!(sub.iter().sum::<i32>(), 4);
    }

    #[test]
    fn threshold_apply_hi_value_minus_one_passes_through_white() {
        // hi_value == -1 means "no opinion": white_result must stay true
        // for every pixel regardless of value/threshold, i.e. the output is
        // uniformly 255 (background/white in this crate's convention).
        let grey: [u8; 4] = [0, 50, 200, 255];
        let out = threshold_rect_to_binary(
            &grey,
            4,
            0,
            0,
            4,
            1,
            super::OtsuChannel {
                threshold: 100,
                hi_value: -1,
            },
        );
        assert_eq!(out, vec![255, 255, 255, 255]);
    }

    #[test]
    fn threshold_apply_hi_value_zero_marks_above_threshold_as_foreground() {
        // hi_value == 0: pixel > threshold => foreground (0), else background (255).
        let grey: [u8; 4] = [0, 100, 101, 255];
        let out = threshold_rect_to_binary(
            &grey,
            4,
            0,
            0,
            4,
            1,
            super::OtsuChannel {
                threshold: 100,
                hi_value: 0,
            },
        );
        assert_eq!(out, vec![255, 255, 0, 0]);
    }

    #[test]
    fn threshold_apply_hi_value_one_marks_at_or_below_threshold_as_foreground() {
        // hi_value == 1: pixel > threshold => background (255), else foreground (0).
        let grey: [u8; 4] = [0, 100, 101, 255];
        let out = threshold_rect_to_binary(
            &grey,
            4,
            0,
            0,
            4,
            1,
            super::OtsuChannel {
                threshold: 100,
                hi_value: 1,
            },
        );
        assert_eq!(out, vec![0, 0, 255, 255]);
    }

    #[test]
    fn otsu_threshold_gray_bimodal_image_finds_threshold_and_hi_value() {
        // A synthetic bimodal grey rectangle: left half dark (30), right
        // half light (220). Otsu should split cleanly around the middle
        // value range, and since roughly half the pixels are "high" (light)
        // and it's an even 50/50 split, the channel should NOT reach the
        // 0.75/0.25 "convincing" cutoffs -- it falls into the "best of the
        // bad lot" branch, which for a single channel always wins (there is
        // nothing else to compare against), so hi_value must not be -1.
        let w = 8;
        let h = 4;
        let mut grey = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                grey[y * w + x] = if x < w / 2 { 30 } else { 220 };
            }
        }
        let otsu = otsu_threshold_gray(&grey, w, 0, 0, w, h);
        assert!(otsu.threshold >= 30 && otsu.threshold < 220);
        assert_ne!(otsu.hi_value, -1);
    }
}
