// Batch 3E wave-1 oracle: STATS + makerow pure-math leaves + DetLineFit.
//
// Route: STATS/DetLineFit are COMPILED DIRECTLY from /tmp/tesseract source
// (statistc.cpp, detlinefit.cpp) alongside this TU -- NOT linked against the
// installed -ltesseract (ABI-skew rule: in-env lib is 5.3.4, source headers
// are 5.5.0). Only two small infra symbols (tesseract::get_debugfp,
// tesseract::tesserr) are stubbed here since ConstrainedFit's debug-print
// branch references the global `tesserr` stream even though this oracle
// always passes debug=false (the branch body is unreachable, but the
// TessErrStream symbol must still resolve at link time).
//
// compute_line_occupation / compute_occupation_threshold /
// compute_dropout_distances / compute_height_modes / fill_heights are FREE
// FUNCTIONS in makerow.cpp that take heavy TO_BLOCK*/TO_ROW*/BLOBNBOX*
// object graphs (compute_line_occupation, fill_heights) that are out of
// reach for a standalone oracle. This oracle instead hand-transcribes the
// identical loop bodies (verified line-by-line against
// /tmp/tesseract/src/textord/makerow.cpp) driving the REAL ICOORD/FCOORD
// classes from points.h for every piece of arithmetic (rotate/cross/dot/
// sqlength) -- i.e. the loop *shells* are copied, the *arithmetic primitives*
// are the genuine compiled-from-source classes, matching this repo's
// established oracle-construction pattern for functions with un-instantiable
// object dependencies.
//
// Build:
//   g++ -std=c++17 -DGRAPHICS_DISABLED \
//     -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
//     -I/tmp/tesseract/src/viewer -I/tmp/tesseract/src/arch \
//     -I/tmp/tesseract/include -I/usr/include/leptonica \
//     /tmp/tesseract/src/ccstruct/statistc.cpp \
//     /tmp/tesseract/src/ccstruct/detlinefit.cpp \
//     /tmp/textline_math_oracle.cpp \
//     -o /tmp/textline_math_oracle \
//     $(pkg-config --libs tesseract) $(pkg-config --libs lept)

#include "statistc.h"
#include "detlinefit.h"
#include "points.h"
#include "tesserrstream.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <cmath>

namespace tesseract {
FILE *get_debugfp() { return stderr; }
TessErrStream tesserr;
}

using tesseract::STATS;
using tesseract::DetLineFit;
using tesseract::ICOORD;
using tesseract::FCOORD;

// ---------------- fixture reader ----------------
struct Reader {
  FILE *f;
  explicit Reader(const char *path) { f = fopen(path, "rb"); }
  ~Reader() { if (f) fclose(f); }
  int32_t i32() { int32_t v; fread(&v, 4, 1, f); return v; }
  uint32_t u32() { uint32_t v; fread(&v, 4, 1, f); return v; }
  uint8_t u8() { uint8_t v; fread(&v, 1, 1, f); return v; }
  float f32() { float v; fread(&v, 4, 1, f); return v; }
  double f64() { double v; fread(&v, 8, 1, f); return v; }
};

static void dump_f64_hex(FILE *out, double v) {
  uint64_t bits;
  memcpy(&bits, &v, 8);
  fprintf(out, "%016llx", (unsigned long long)bits);
}
static void dump_f32_hex(FILE *out, float v) {
  uint32_t bits;
  memcpy(&bits, &v, 4);
  fprintf(out, "%08x", bits);
}

// ---------------- hand-transcribed loop shells (see file header) ----------------

// makerow.cpp:799-845 (ASSERT_HOST replaced with a plain crash-on-violation,
// since a well-formed fixture never violates it).
static void compute_line_occupation(const std::vector<std::array<int32_t,4>> &blobs,
                                     float gradient, int32_t min_y, int32_t max_y,
                                     std::vector<int32_t> &occupation,
                                     std::vector<int32_t> &deltas) {
  int32_t line_count = max_y - min_y + 1;
  float length = std::sqrt(gradient * gradient + 1);
  FCOORD rotation(1 / length, -gradient / length);
  deltas.assign(line_count, 0);
  for (auto &b : blobs) {
    ICOORD bot_left(b[0], b[1]);
    ICOORD top_right(b[2], b[3]);
    bot_left.rotate(rotation);
    top_right.rotate(rotation);
    int32_t width = top_right.x() - bot_left.x();
    int32_t index = bot_left.y() - min_y;
    if (!(index >= 0 && index < line_count)) { fprintf(stderr, "OOB bottom\n"); abort(); }
    deltas[index] += width;
    index = top_right.y() - min_y;
    if (!(index >= 0 && index < line_count)) { fprintf(stderr, "OOB top\n"); abort(); }
    deltas[index] -= width;
  }
  occupation.assign(line_count, 0);
  occupation[0] = deltas[0];
  for (int32_t line_index = 1; line_index < line_count; line_index++) {
    occupation[line_index] = occupation[line_index - 1] + deltas[line_index];
  }
}

// makerow.cpp:852-926, verbatim variable-for-variable.
static void compute_occupation_threshold(int32_t low_window, int32_t high_window,
                                          int32_t line_count, int32_t *occupation,
                                          double textord_occupancy_threshold,
                                          int32_t *thresholds) {
  int32_t line_index, low_index, high_index, sum, divisor, min_index, min_occ, test_index;
  divisor = static_cast<int32_t>(ceil((low_window + high_window) / textord_occupancy_threshold));
  if (low_window + high_window < line_count) {
    for (sum = 0, high_index = 0; high_index < low_window; high_index++) {
      sum += occupation[high_index];
    }
    for (low_index = 0; low_index < high_window; low_index++, high_index++) {
      sum += occupation[high_index];
    }
    min_occ = occupation[0];
    min_index = 0;
    for (test_index = 1; test_index < high_index; test_index++) {
      if (occupation[test_index] <= min_occ) {
        min_occ = occupation[test_index];
        min_index = test_index;
      }
    }
    for (line_index = 0; line_index < low_window; line_index++) {
      thresholds[line_index] = (sum - min_occ) / divisor + min_occ;
    }
    for (low_index = 0; high_index < line_count; low_index++, high_index++) {
      sum -= occupation[low_index];
      sum += occupation[high_index];
      if (occupation[high_index] <= min_occ) {
        min_occ = occupation[high_index];
        min_index = high_index;
      }
      if (min_index <= low_index) {
        min_occ = occupation[low_index + 1];
        min_index = low_index + 1;
        for (test_index = low_index + 2; test_index <= high_index; test_index++) {
          if (occupation[test_index] <= min_occ) {
            min_occ = occupation[test_index];
            min_index = test_index;
          }
        }
      }
      thresholds[line_index++] = (sum - min_occ) / divisor + min_occ;
    }
  } else {
    min_occ = occupation[0];
    min_index = 0;
    for (sum = 0, low_index = 0; low_index < line_count; low_index++) {
      if (occupation[low_index] < min_occ) {
        min_occ = occupation[low_index];
        min_index = low_index;
      }
      sum += occupation[low_index];
    }
    line_index = 0;
  }
  (void)min_index;
  for (; line_index < line_count; line_index++) {
    thresholds[line_index] = (sum - min_occ) / divisor + min_occ;
  }
}

// makerow.cpp:933-967, verbatim.
static void compute_dropout_distances(int32_t *occupation, int32_t *thresholds, int32_t line_count) {
  int32_t line_index, distance, next_dist, back_index, prev_threshold;
  distance = -line_count;
  line_index = 0;
  do {
    do {
      distance--;
      prev_threshold = thresholds[line_index];
      thresholds[line_index] = distance;
      line_index++;
    } while (line_index < line_count && (occupation[line_index] < thresholds[line_index] ||
                                         occupation[line_index - 1] >= prev_threshold));
    if (line_index < line_count) {
      back_index = line_index - 1;
      next_dist = 1;
      while (next_dist < -distance && back_index >= 0) {
        thresholds[back_index] = next_dist;
        back_index--;
        next_dist++;
        distance++;
      }
      distance = 1;
    }
  } while (line_index < line_count);
}

// makerow.cpp:1629-1682, verbatim.
static int32_t compute_height_modes(STATS *heights, int32_t min_height, int32_t max_height,
                                     int32_t *modes, int32_t maxmodes) {
  int32_t pile_count, src_count, src_index, least_count, least_index, dest_count;
  src_count = max_height + 1 - min_height;
  dest_count = 0;
  least_count = INT32_MAX;
  least_index = -1;
  for (src_index = 0; src_index < src_count; src_index++) {
    pile_count = heights->pile_count(min_height + src_index);
    if (pile_count > 0) {
      if (dest_count < maxmodes) {
        if (pile_count < least_count) {
          least_count = pile_count;
          least_index = dest_count;
        }
        modes[dest_count++] = min_height + src_index;
      } else if (pile_count >= least_count) {
        while (least_index < maxmodes - 1) {
          modes[least_index] = modes[least_index + 1];
          least_index++;
        }
        modes[maxmodes - 1] = min_height + src_index;
        if (pile_count == least_count) {
          least_index = maxmodes - 1;
        } else {
          least_count = heights->pile_count(modes[0]);
          least_index = 0;
          for (dest_count = 1; dest_count < maxmodes; dest_count++) {
            pile_count = heights->pile_count(modes[dest_count]);
            if (pile_count < least_count) {
              least_count = pile_count;
              least_index = dest_count;
            }
          }
        }
      }
    }
  }
  return dest_count;
}

// makerow.cpp:1418-1462, the non-baseline-spline (textord_fix_xheight_bug ==
// false) branch, on plain box tuples (see tesseract-ocr textline.rs module
// doc for the equivalent Rust-side scoping note).
static void fill_heights(const std::vector<std::array<int32_t,4>> &boxes, float gradient,
                          float parallel_c, int min_height, int max_height,
                          float min_blob_height_fraction,
                          STATS *heights, STATS *floating_heights) {
  for (auto &b : boxes) {
    float xcentre = (b[0] + b[2]) / 2.0f;
    float top = static_cast<float>(b[3]);
    float height = static_cast<float>(b[3] - b[1]);
    top -= gradient * xcentre + parallel_c;
    if (top >= min_height && top <= max_height) {
      heights->add(static_cast<int32_t>(floor(top + 0.5)), 1);
      if (height / top < min_blob_height_fraction) {
        floating_heights->add(static_cast<int32_t>(floor(top + 0.5)), 1);
      }
    }
  }
}

int main() {
  Reader r("/tmp/textline_math_input.bin");
  if (!r.f) { fprintf(stderr, "cannot open fixture\n"); return 1; }

  // ---- Section 1: STATS ----
  {
    int32_t rmin = r.i32(), rmax = r.i32();
    STATS s(rmin, rmax);
    uint32_t n = r.u32();
    for (uint32_t i = 0; i < n; i++) {
      int32_t v = r.i32(), c = r.i32();
      s.add(v, c);
    }
    printf("SECTION1 total=%d mode=%d\n", s.get_total(), s.mode());
    printf("SECTION1 mean_hex="); dump_f64_hex(stdout, s.mean()); printf("\n");
    printf("SECTION1 sd_hex="); dump_f64_hex(stdout, s.sd()); printf("\n");
    printf("SECTION1 median_hex="); dump_f64_hex(stdout, s.median()); printf("\n");
    printf("SECTION1 min_bucket=%d max_bucket=%d\n", s.min_bucket(), s.max_bucket());
    uint32_t nf = r.u32();
    for (uint32_t i = 0; i < nf; i++) {
      double frac = r.f64();
      printf("SECTION1 ile[%.4f]_hex=", frac); dump_f64_hex(stdout, s.ile(frac)); printf("\n");
    }
  }

  // ---- Section 2: occupation/threshold/dropout ----
  {
    uint32_t line_count = r.u32();
    std::vector<int32_t> occupation(line_count);
    for (uint32_t i = 0; i < line_count; i++) occupation[i] = r.i32();
    int32_t low_window = r.i32(), high_window = r.i32();
    double occ_thresh = r.f64();
    std::vector<int32_t> thresholds(line_count);
    compute_occupation_threshold(low_window, high_window, line_count, occupation.data(),
                                  occ_thresh, thresholds.data());
    printf("SECTION2 thresholds=");
    for (uint32_t i = 0; i < line_count; i++) printf("%d,", thresholds[i]);
    printf("\n");
    std::vector<int32_t> dropout = thresholds; // compute_dropout_distances mutates in place
    compute_dropout_distances(occupation.data(), dropout.data(), line_count);
    printf("SECTION2 dropout=");
    for (uint32_t i = 0; i < line_count; i++) printf("%d,", dropout[i]);
    printf("\n");
  }

  // ---- Section 3: height modes ----
  {
    int32_t rmin = r.i32(), rmax = r.i32();
    STATS heights(rmin, rmax);
    uint32_t n = r.u32();
    for (uint32_t i = 0; i < n; i++) {
      int32_t v = r.i32(), c = r.i32();
      heights.add(v, c);
    }
    int32_t min_height = r.i32(), max_height = r.i32(), maxmodes = r.i32();
    std::vector<int32_t> modes(maxmodes, 0);
    int32_t count = compute_height_modes(&heights, min_height, max_height, modes.data(), maxmodes);
    printf("SECTION3 count=%d modes=", count);
    for (int32_t i = 0; i < count; i++) printf("%d,", modes[i]);
    printf("\n");
  }

  // ---- Section 4: fill_heights ----
  {
    uint32_t n = r.u32();
    std::vector<std::array<int32_t,4>> boxes(n);
    for (uint32_t i = 0; i < n; i++) {
      boxes[i] = {r.i32(), r.i32(), r.i32(), r.i32()};
    }
    float gradient = r.f32();
    float parallel_c = r.f32();
    int32_t min_height = r.i32(), max_height = r.i32();
    float min_blob_height_fraction = r.f32();
    STATS heights(min_height, max_height);
    STATS floating(min_height, max_height);
    fill_heights(boxes, gradient, parallel_c, min_height, max_height, min_blob_height_fraction,
                 &heights, &floating);
    printf("SECTION4 heights_total=%d floating_total=%d heights_mode=%d\n",
           heights.get_total(), floating.get_total(), heights.mode());
    printf("SECTION4 heights_mean_hex="); dump_f64_hex(stdout, heights.mean()); printf("\n");
  }

  // ---- Section 5: compute_line_occupation ----
  {
    uint32_t n = r.u32();
    std::vector<std::array<int32_t,4>> blobs(n);
    for (uint32_t i = 0; i < n; i++) {
      blobs[i] = {r.i32(), r.i32(), r.i32(), r.i32()};
    }
    float gradient = r.f32();
    int32_t min_y = r.i32(), max_y = r.i32();
    std::vector<int32_t> occupation, deltas;
    compute_line_occupation(blobs, gradient, min_y, max_y, occupation, deltas);
    printf("SECTION5 occupation=");
    for (auto v : occupation) printf("%d,", v);
    printf("\n");
    printf("SECTION5 deltas=");
    for (auto v : deltas) printf("%d,", v);
    printf("\n");
  }

  // ---- Section 6: DetLineFit ----
  {
    uint32_t n_configs = r.u32();
    for (uint32_t ci = 0; ci < n_configs; ci++) {
      uint8_t kind = r.u8();
      uint32_t n_pts = r.u32();
      DetLineFit lms;
      for (uint32_t i = 0; i < n_pts; i++) {
        int32_t x = r.i32(), y = r.i32(), hw = r.i32();
        if (hw != 0) {
          lms.Add(ICOORD(x, y), hw);
        } else {
          lms.Add(ICOORD(x, y));
        }
      }
      if (kind == 0) {
        float m, c;
        double error = lms.Fit(&m, &c);
        printf("SECTION6[%u] kind=fit m_hex=", ci); dump_f32_hex(stdout, m);
        printf(" c_hex="); dump_f32_hex(stdout, c);
        printf(" error_hex="); dump_f64_hex(stdout, error); printf("\n");
      } else if (kind == 1) {
        float dx = r.f32(), dy = r.f32();
        double mind = r.f64(), maxd = r.f64();
        ICOORD line_pt;
        double error = lms.ConstrainedFit(FCOORD(dx, dy), mind, maxd, false, &line_pt);
        printf("SECTION6[%u] kind=constrained_dir pt=(%d,%d) error_hex=", ci, line_pt.x(), line_pt.y());
        dump_f64_hex(stdout, error); printf("\n");
      } else if (kind == 2) {
        double m = r.f64();
        float c;
        double error = lms.ConstrainedFit(m, &c);
        printf("SECTION6[%u] kind=constrained_m c_hex=", ci); dump_f32_hex(stdout, c);
        printf(" error_hex="); dump_f64_hex(stdout, error); printf("\n");
      }
    }
  }

  return 0;
}
