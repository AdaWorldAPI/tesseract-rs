// Batch 3F2 leaf 2 oracle: Textord::filter_noise_blobs + Textord::filter_blobs
// (tordmain.cpp:238-360). Same route as the Batch 3E wave-1/wave-2 oracles:
// STATS is COMPILED DIRECTLY from /tmp/tesseract source (statistc.cpp) --
// NOT linked against the installed -ltesseract (ABI-skew rule: in-env lib is
// 5.3.4, source headers are 5.5.0).
//
// filter_noise_blobs/filter_blobs operate on BLOBNBOX_LIST/BLOBNBOX_IT (the
// full ELIST doubly-linked-list machinery) which is out of reach for a
// standalone oracle -- same situation Batch 3E wave-1/2 hit for
// compute_line_occupation/assign_blobs_to_rows. This oracle instead
// hand-transcribes filter_noise_blobs + filter_blobs verbatim (checked
// line-by-line against /tmp/tesseract/src/textord/tordmain.cpp) onto a
// minimal OracleBlob shell carrying exactly the fields this leaf reads:
// bounding_box().height()/width() and enclosed_area() (BLOBNBOX::area,
// blobbox.h:150/262 -- upstream of this leaf, Batch 3F2 leaf 1's
// conn_comp_areas pixel_count is its port-side analogue).
//
// PARITY-PIN doctrine (board E-OCR-MAKEROW-2): filter_noise_blobs contains
// NEITHER a sort NOR an nth_element -- every classification is a per-element
// threshold test against scalars fixed before the pass runs, so it is
// provably invariant to traversal order. No comparator needs pinning. The
// real BLOBNBOX_IT::add_after_then_move splice order (inserts after the
// iterator's CURRENT cursor, not at the list tail -- ccutil/elst.h:333-366)
// is consequently NOT reproduced here; this oracle uses plain
// append-preserving-first-seen-order partitioning, matching the tesseract-ocr
// blob_filter.rs port's own documented choice. The dump below therefore
// reports each output list's MEMBERSHIP (sorted original-fixture indices),
// not raw traversal-order sequences -- see blob_filter.rs's module doc for
// the full justification (order has zero effect on SET membership or any
// scalar computed here, and assign_blobs_to_rows immediately re-sorts by x
// downstream anyway).
//
// Build:
//   g++ -std=c++17 -DGRAPHICS_DISABLED \
//     -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
//     -I/tmp/tesseract/src/viewer -I/tmp/tesseract/src/arch \
//     -I/tmp/tesseract/include -I/usr/include/leptonica \
//     /tmp/tesseract/src/ccstruct/statistc.cpp \
//     /tmp/blob_filter_oracle.cpp \
//     -o /tmp/blob_filter_oracle \
//     $(pkg-config --libs tesseract) $(pkg-config --libs lept)

#include "statistc.h"
#include "tesserrstream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tesseract {
FILE *get_debugfp() {
  return stderr;
}
TessErrStream tesserr;
} // namespace tesseract

using tesseract::STATS;

// tordmain.cpp:61
#define MAX_NEAREST_DIST 600

// textord.cpp:146,148,150,152 (INT_MEMBER/double_MEMBER defaults)
static const int32_t TEXTORD_MAX_NOISE_SIZE = 7;
static const double TEXTORD_NOISE_AREA_RATIO = 0.7;
static const double TEXTORD_INITIALX_ILE = 0.75;
static const double TEXTORD_INITIALASC_ILE = 0.90;
// makerow.cpp:75,80,81 (double_VAR defaults)
static const double TEXTORD_WIDTH_LIMIT = 8.0;
static const double TEXTORD_MIN_LINESIZE = 1.25;
static const double TEXTORD_EXCESS_BLOBSIZE = 1.3;

// ccstruct.cpp:25-29
static const double K_DESCENDER_FRACTION = 0.25;
static const double K_XHEIGHT_FRACTION = 0.5;
static const double K_ASCENDER_FRACTION = 0.25;
static const double K_XHEIGHT_CAP_RATIO =
    K_XHEIGHT_FRACTION / (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION);

// Minimal BLOBNBOX shell: the (left,bottom,right,top) box (top>bottom) +
// enclosed_area() + an original-fixture-index tag for canonical dumping.
struct OracleBlob {
  int32_t orig_index;
  int32_t left, bottom, right, top;
  int32_t pixel_count;
  int16_t height() const { return static_cast<int16_t>(top - bottom); }
  int16_t width() const { return static_cast<int16_t>(right - left); }
};

// Textord::filter_noise_blobs (tordmain.cpp:291-360), transcribed verbatim
// onto OracleBlob. `src` is mutated in place to become the final surviving
// pool (mirrors src_list's final state); noise/small/large are appended to.
// Returns the `initial_x` (== the caller's `block->line_size` pre-scale).
static float filter_noise_blobs(std::vector<OracleBlob> &src, std::vector<OracleBlob> &noise,
                                 std::vector<OracleBlob> &small, std::vector<OracleBlob> &large) {
  STATS size_stats(0, MAX_NEAREST_DIST - 1);
  float min_y, max_y, max_x, max_height;

  // ---- pass 1 (tordmain.cpp:310-319) ----
  std::vector<OracleBlob> pool;
  for (auto &b : src) {
    if (b.height() < TEXTORD_MAX_NOISE_SIZE) {
      noise.push_back(b);
    } else if (static_cast<double>(b.pixel_count) >=
               static_cast<double>(b.height()) * static_cast<double>(b.width()) *
                   TEXTORD_NOISE_AREA_RATIO) {
      small.push_back(b);
    } else {
      pool.push_back(b);
    }
  }

  // ---- size_stats over the pool (tordmain.cpp:320-322) ----
  for (auto &b : pool) {
    size_stats.add(b.height(), 1);
  }

  // ---- initial_x / max_y / min_y / max_x (tordmain.cpp:323-329) ----
  float initial_x = static_cast<float>(size_stats.ile(TEXTORD_INITIALX_ILE));
  max_y = static_cast<float>(
      std::ceil(static_cast<double>(initial_x) *
                (K_DESCENDER_FRACTION + K_XHEIGHT_FRACTION + 2 * K_ASCENDER_FRACTION) /
                K_XHEIGHT_FRACTION));
  min_y = std::floor(initial_x / 2.0f);
  max_x = static_cast<float>(std::ceil(static_cast<double>(initial_x) * TEXTORD_WIDTH_LIMIT));

  // ---- small-list rescue pass (tordmain.cpp:330-338) ----
  std::vector<OracleBlob> still_small;
  for (auto &b : small) {
    float height = static_cast<float>(b.height());
    if (height > max_y) {
      large.push_back(b);
    } else if (height >= min_y) {
      pool.push_back(b);
    } else {
      still_small.push_back(b);
    }
  }
  small = still_small;

  // ---- re-partition pass (tordmain.cpp:340-350) ----
  size_stats.clear();
  std::vector<OracleBlob> final_pool;
  for (auto &b : pool) {
    float height = static_cast<float>(b.height());
    float width = static_cast<float>(b.width());
    if (height < min_y) {
      small.push_back(b);
    } else if (height > max_y || width > max_x) {
      large.push_back(b);
    } else {
      size_stats.add(b.height(), 1);
      final_pool.push_back(b);
    }
  }
  src = final_pool;

  // ---- max_height / initial_x finalization (tordmain.cpp:351-359) ----
  max_height = static_cast<float>(size_stats.ile(TEXTORD_INITIALASC_ILE));
  max_height = static_cast<float>(static_cast<double>(max_height) * K_XHEIGHT_CAP_RATIO);
  if (max_height > initial_x) {
    initial_x = max_height;
  }
  return initial_x;
}

static uint32_t f32_bits(float v) {
  uint32_t bits;
  memcpy(&bits, &v, 4);
  return bits;
}

static void dump_indices(const char *tag, std::vector<OracleBlob> &v) {
  std::vector<int32_t> idx;
  idx.reserve(v.size());
  for (auto &b : v) {
    idx.push_back(b.orig_index);
  }
  std::sort(idx.begin(), idx.end());
  printf("%s\t%zu", tag, idx.size());
  for (auto i : idx) {
    printf("\t%d", i);
  }
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]);
    return 1;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  uint32_t n = 0;
  if (fread(&n, 4, 1, f) != 1) {
    abort();
  }
  std::vector<OracleBlob> blobs(n);
  for (uint32_t i = 0; i < n; i++) {
    int32_t l = 0, b = 0, r = 0, t = 0, pc = 0;
    if (fread(&l, 4, 1, f) != 1 || fread(&b, 4, 1, f) != 1 || fread(&r, 4, 1, f) != 1 ||
        fread(&t, 4, 1, f) != 1 || fread(&pc, 4, 1, f) != 1) {
      abort();
    }
    blobs[i] = {static_cast<int32_t>(i), l, b, r, t, pc};
  }
  fclose(f);

  printf("FIXTURE n_blobs=%u\n", n);

  std::vector<OracleBlob> noise, small, large;
  float initial_x = filter_noise_blobs(blobs, noise, small, large);

  // ---- Textord::filter_blobs's line-size setup (tordmain.cpp:254-263) ----
  float line_size = initial_x;
  if (line_size == 0.0f) {
    line_size = 1.0f;
  }
  float line_spacing = static_cast<float>(
      static_cast<double>(line_size) *
      (K_DESCENDER_FRACTION + K_XHEIGHT_FRACTION + 2 * K_ASCENDER_FRACTION) /
      K_XHEIGHT_FRACTION);
  line_size = static_cast<float>(static_cast<double>(line_size) * TEXTORD_MIN_LINESIZE);
  float max_blob_size =
      static_cast<float>(static_cast<double>(line_size) * TEXTORD_EXCESS_BLOBSIZE);

  dump_indices("BLOBS", blobs);
  dump_indices("NOISE", noise);
  dump_indices("SMALL", small);
  dump_indices("LARGE", large);
  printf("SCALARS line_size_hex=%08x line_spacing_hex=%08x max_blob_size_hex=%08x\n",
         f32_bits(line_size), f32_bits(line_spacing), f32_bits(max_blob_size));
  return 0;
}
