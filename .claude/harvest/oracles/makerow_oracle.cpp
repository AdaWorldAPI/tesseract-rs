// Batch 3E wave-2 oracle: the row-assignment + cleanup chain (makerow.cpp,
// single-column/single-block case): assign_blobs_to_rows,
// most_overlapping_row, fit_parallel_rows, fit_parallel_lms,
// delete_non_dropout_rows, deskew_block_coords, find_best_dropout_row,
// expand_rows, adjust_row_limits, compute_row_stats, make_initial_textrows,
// compute_page_skew, and the cleanup_rows_making reconsolidation.
//
// Route: SAME as wave 1's /tmp/textline_math_oracle.cpp -- STATS/DetLineFit
// are COMPILED DIRECTLY from /tmp/tesseract source (statistc.cpp,
// detlinefit.cpp) alongside this TU (ABI-skew rule: in-env lib is 5.3.4,
// source headers are 5.5.0, so linking -ltesseract for these is unsafe).
//
// Every makerow-level function below operates on TO_ROW/TO_BLOCK/BLOBNBOX,
// which drag in the full ELIST2 doubly-linked-list machinery, C_BLOB
// outlines, WERD_LIST, ICOORDELT_LIST, QSPLINE, and ScrollView debug draws
// -- out of reach for a standalone oracle (same situation wave 1 hit for
// compute_line_occupation/fill_heights). This oracle instead hand-
// transcribes every one of them verbatim (checked line-by-line against
// /tmp/tesseract/src/textord/makerow.cpp and
// /tmp/tesseract/src/ccstruct/blobbox.{h,cpp}) onto a minimal local
// OracleRow/OracleBlock struct mirror -- a std::vector<OracleRow> playing
// the role of TO_ROW_LIST, indexed exactly the way ELIST2_ITERATOR's
// forward()/backward()/data_relative()/extract() semantics behave (verified
// against ccutil/elst2.h's actual iterator contract, in particular
// ELIST_ITERATOR::extract() at elst.h:610-640: "removing current ... NOT
// updating the iterator ... so that any calling loop can do this" -- i.e. a
// subsequent forward() lands on what was already `next`, exactly matching
// a Vec::remove(i) with the loop NOT advancing past i). Every piece of
// arithmetic that has a real compiled equivalent (DetLineFit::Add/Fit/
// ConstrainedFit, ICOORD/FCOORD rotate/cross/dot) uses the genuine
// compiled-from-source classes; only the TO_ROW/TO_BLOCK/BLOBNBOX shells and
// their list-walking drivers are hand-transcribed. Each function below
// cites the exact makerow.cpp / blobbox.{h,cpp} / rect.{h,cpp} line range
// it was transcribed from.
//
// compute_line_occupation / compute_occupation_threshold /
// compute_dropout_distances are the SAME wave-1 hand-transcriptions
// (byte-parity proven there against the real makerow.cpp already), copied
// in verbatim rather than re-derived, since delete_non_dropout_rows
// (this wave) calls them exactly as wave 1's oracle already validated.
//
// Build:
//   g++ -std=c++17 -DGRAPHICS_DISABLED \
//     -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
//     -I/tmp/tesseract/src/viewer -I/tmp/tesseract/src/arch \
//     -I/tmp/tesseract/include -I/usr/include/leptonica \
//     /tmp/tesseract/src/ccstruct/statistc.cpp \
//     /tmp/tesseract/src/ccstruct/detlinefit.cpp \
//     /tmp/makerow_oracle.cpp \
//     -o /tmp/makerow_oracle \
//     $(pkg-config --libs tesseract) $(pkg-config --libs lept)

#include "statistc.h"
#include "detlinefit.h"
#include "points.h"
#include "tesserrstream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace tesseract {
FILE *get_debugfp() { return stderr; }
TessErrStream tesserr;
} // namespace tesseract

using tesseract::DetLineFit;
using tesseract::FCOORD;
using tesseract::ICOORD;
using tesseract::STATS;

using Box = std::array<int32_t, 4>; // left,bottom,right,top

// ============================================================
// pinned textord_* params + CCStruct fractions (see
// .claude/harvest/makerow-callgraph.txt and the tesseract-ocr
// textline.rs module doc for the float/double promotion audit this
// oracle mirrors exactly)
// ============================================================
static const int32_t TEXTORD_MIN_BLOBS_IN_ROW = 4;
static const double TEXTORD_SKEWSMOOTH_OFFSET = 4.0;
static const double TEXTORD_SKEWSMOOTH_OFFSET2 = 1.0;
static const double TEXTORD_SKEW_LAG = 0.02;
static const double TEXTORD_SKEW_ILE = 0.5;
static const bool TEXTORD_BIASED_SKEWCALC = true;
static const bool TEXTORD_INTERPOLATING_SKEW = true;
static const double TEXTORD_OVERLAP_X = 0.375;
static const bool TEXTORD_FIX_MAKEROW_BUG = true;
static const double TEXTORD_EXPANSION_FACTOR = 1.0;
static const double TEXTORD_LINESPACE_IQRLIMIT = 0.2;
static const float TEXTORD_MIN_XHEIGHT = 10.0f;
static const double TEXTORD_EXCESS_BLOBSIZE = 1.3;
static const double TEXTORD_OCCUPANCY_THRESHOLD = 0.4;

static const double K_DESCENDER_FRACTION = 0.25;
static const double K_XHEIGHT_FRACTION = 0.5;
static const double K_ASCENDER_FRACTION = 0.25;

static const float TO_ROW_K_ERROR_WEIGHT = 3.0f;

enum OverlapState { OS_ASSIGN, OS_REJECT, OS_NEW_ROW };

// ============================================================
// OracleRow / OracleBlock: minimal TO_ROW / TO_BLOCK mirror
// (blobbox.h:555-806, ctor/add_blob/set_parallel_line bodies from
// blobbox.cpp:690-765).
// ============================================================
struct OracleRow {
  std::vector<Box> blobs;
  float y_min = 0, y_max = 0, initial_y_min = 0;
  float m = 0, c = 0, error = 0;
  float para_c = 0, para_error = 0;
  float y_origin = 0, credibility = 0;
  float spacing = 0;
  bool merged = false;
  float xheight = 0, ascrise = 0, descdrop = 0;
  int xheight_evidence = 0;
  bool all_caps = false, rep_chars_marked = false;

  static OracleRow FromBlob(const Box &blob, float top, float bottom, float row_size) {
    OracleRow row;
    row.y_min = bottom;
    row.y_max = top;
    row.initial_y_min = bottom;
    row.blobs.push_back(blob);
    float diff = top - bottom - row_size;
    if (diff > 0) {
      row.y_max -= diff / 2;
      row.y_min += diff / 2;
    } else if ((top - bottom) * 3 < row_size) {
      diff = row_size / 3 + bottom - top;
      row.y_max += diff / 2;
      row.y_min -= diff / 2;
    }
    return row;
  }

  float max_y() const { return y_max; }
  float min_y() const { return y_min; }
  float initial_min_y() const { return initial_y_min; }
  float line_m() const { return m; }
  float line_c() const { return c; }
  float line_error() const { return error; }
  float parallel_c() const { return para_c; }
  float believability() const { return credibility; }
  float intercept() const { return y_origin; }

  void add_blob(const Box &blob, float top, float bottom, float row_size) {
    blobs.push_back(blob);
    float allowed = row_size + y_min - y_max;
    if (allowed > 0) {
      float available = top > y_max ? top - y_max : 0;
      if (bottom < y_min) {
        available += y_min - bottom;
      }
      if (available > 0) {
        available += available;
        if (available < allowed) {
          available = allowed;
        }
        if (bottom < y_min) {
          y_min -= (y_min - bottom) * allowed / available;
        }
        if (top > y_max) {
          y_max += (top - y_max) * allowed / available;
        }
      }
    }
  }
  void set_line(float new_m, float new_c, double new_error) {
    m = new_m;
    c = new_c;
    error = static_cast<float>(new_error);
  }
  void set_parallel_line(float gradient, float new_c, double new_error) {
    para_c = new_c;
    para_error = static_cast<float>(new_error);
    credibility = static_cast<float>(blobs.size()) - TO_ROW_K_ERROR_WEIGHT * static_cast<float>(new_error);
    y_origin = new_c / std::sqrt(1.0f + gradient * gradient);
  }
  void set_limits(float new_min, float new_max) {
    y_min = new_min;
    y_max = new_max;
  }
};

struct OracleBlock {
  std::vector<Box> blobs; // unassigned pool
  std::vector<OracleRow> rows;
  int32_t block_left = 0;
  float line_spacing = 0, line_size = 0, max_blob_size = 0, baseline_offset = 0;
  float xheight = 0;
};

// ELIST2_ITERATOR::data_relative circular-index helper.
static size_t data_relative(size_t len, size_t idx, long offset) {
  long n = static_cast<long>(len);
  long v = (static_cast<long>(idx) + offset) % n;
  if (v < 0) {
    v += n;
  }
  return static_cast<size_t>(v);
}

// TBOX::major_x_overlap (rect.h:419-428).
static bool major_x_overlap(const Box &lhs, const Box &rhs) {
  int32_t lhs_width = lhs[2] - lhs[0];
  int32_t rhs_width = rhs[2] - rhs[0];
  int32_t overlap = rhs_width;
  if (lhs[0] > rhs[0]) {
    overlap -= lhs[0] - rhs[0];
  }
  if (lhs[2] < rhs[2]) {
    overlap -= rhs[2] - lhs[2];
  }
  return overlap >= rhs_width / 2 || overlap >= lhs_width / 2;
}

// PARITY PIN: real Tesseract sorts blobs with blob_x_order (left only) via
// qsort-style ELIST sort -- tie order is UNSPECIFIED. Both probe shells pin
// the same TOTAL order (full box tuple) so equal-left blobs arrive in the
// same sequence; add_blob expansion is order-dependent.
static bool blob_x_less(const Box &a, const Box &b) { return a < b; }

// ---------------- wave-1 occupation leaves (verbatim, byte-parity proven
// already; copied from /tmp/textline_math_oracle.cpp) ----------------
static void compute_line_occupation(const std::vector<Box> &blobs, float gradient, int32_t min_y,
                                     int32_t max_y, std::vector<int32_t> &occupation,
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
    if (!(index >= 0 && index < line_count)) {
      fprintf(stderr, "OOB bottom\n");
      abort();
    }
    deltas[index] += width;
    index = top_right.y() - min_y;
    if (!(index >= 0 && index < line_count)) {
      fprintf(stderr, "OOB top\n");
      abort();
    }
    deltas[index] -= width;
  }
  occupation.assign(line_count, 0);
  occupation[0] = deltas[0];
  for (int32_t line_index = 1; line_index < line_count; line_index++) {
    occupation[line_index] = occupation[line_index - 1] + deltas[line_index];
  }
}

static void compute_occupation_threshold(int32_t low_window, int32_t high_window, int32_t line_count,
                                          int32_t *occupation, double textord_occupancy_threshold,
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
    } while (line_index < line_count &&
             (occupation[line_index] < thresholds[line_index] || occupation[line_index - 1] >= prev_threshold));
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

// ---------------- fit_lms_line (makerow.cpp:296-307) ----------------
static void fit_lms_line(const std::vector<Box> &boxes, float *m, float *c, double *error) {
  DetLineFit lms;
  for (auto &b : boxes) {
    lms.Add(ICOORD((b[0] + b[2]) / 2, b[1]));
  }
  *error = lms.Fit(m, c);
}

// ---------------- deskew_block_coords (makerow.cpp:765-791) + TBOX::rotate
// (rect.h:210-214) + TBOX(ICOORD,ICOORD) 2-corner ctor (rect.cpp:35-56) +
// TBOX::operator+= (rect.cpp:214-234) ----------------
static void deskew_block_coords(const OracleBlock &block, float gradient, int32_t *out_left,
                                 int32_t *out_bottom, int32_t *out_right, int32_t *out_top) {
  float length = std::sqrt(gradient * gradient + 1);
  FCOORD rotation(1 / length, -gradient / length);
  const int32_t NULL_MAX = 32767; // INT16_MAX
  int32_t rl = NULL_MAX, rb = NULL_MAX, rr = -NULL_MAX, rt = -NULL_MAX;
  for (auto &row : block.rows) {
    for (auto &b : row.blobs) {
      ICOORD bot_left(b[0], b[1]);
      ICOORD top_right(b[2], b[3]);
      bot_left.rotate(rotation);
      top_right.rotate(rotation);
      int32_t bl_x = bot_left.x(), bl_y = bot_left.y();
      int32_t tr_x = top_right.x(), tr_y = top_right.y();
      int32_t box_l = std::min(bl_x, tr_x);
      int32_t box_b = std::min(bl_y, tr_y);
      int32_t box_r = std::max(bl_x, tr_x);
      int32_t box_t = std::max(bl_y, tr_y);
      rl = std::min(rl, box_l);
      rb = std::min(rb, box_b);
      rr = std::max(rr, box_r);
      rt = std::max(rt, box_t);
    }
  }
  *out_left = rl;
  *out_bottom = rb;
  *out_right = rr;
  *out_top = rt;
}

// ---------------- most_overlapping_row (makerow.cpp:2451-2535) ----------------
static OverlapState most_overlapping_row(std::vector<OracleRow> &rows, size_t start_idx, float top,
                                          float bottom, float rowsize, size_t *out_row_idx) {
  OverlapState result = OS_ASSIGN;
  size_t row_idx = start_idx;
  float bestover = top - bottom;
  if (top > rows[row_idx].max_y()) {
    bestover -= top - rows[row_idx].max_y();
  }
  if (bottom < rows[row_idx].min_y()) {
    bestover -= rows[row_idx].min_y() - bottom;
  }

  size_t cursor = start_idx;
  for (;;) {
    if (cursor + 1 >= rows.size()) {
      break;
    }
    cursor += 1;
    if (rows[cursor].min_y() <= top && rows[cursor].max_y() >= bottom) {
      float merge_top = std::max(rows[cursor].max_y(), rows[row_idx].max_y());
      float merge_bottom = std::min(rows[cursor].min_y(), rows[row_idx].min_y());
      if (merge_top - merge_bottom <= rowsize) {
        rows[cursor].set_limits(merge_bottom, merge_top);
        std::vector<Box> moved = std::move(rows[row_idx].blobs);
        rows[row_idx].blobs.clear();
        for (auto &bx : moved) {
          rows[cursor].blobs.push_back(bx);
        }
        std::sort(rows[cursor].blobs.begin(), rows[cursor].blobs.end(), blob_x_less);
        size_t victim = cursor - 1;
        rows.erase(rows.begin() + static_cast<long>(victim));
        cursor -= 1;
        bestover = -1.0f;
      }
      float overlap = top - bottom;
      if (top > rows[cursor].max_y()) {
        overlap -= top - rows[cursor].max_y();
      }
      if (bottom < rows[cursor].min_y()) {
        overlap -= rows[cursor].min_y() - bottom;
      }
      if (bestover >= rowsize - 1 && overlap >= rowsize - 1) {
        result = OS_REJECT;
      }
      if (overlap > bestover) {
        bestover = overlap;
        row_idx = cursor;
      }
    } else {
      break;
    }
  }
  while (cursor != row_idx) {
    cursor -= 1;
  }
  if (static_cast<double>(top - bottom - bestover) > static_cast<double>(rowsize) * TEXTORD_OVERLAP_X &&
      (!TEXTORD_FIX_MAKEROW_BUG ||
       static_cast<double>(bestover) < static_cast<double>(rowsize) * TEXTORD_OVERLAP_X) &&
      result == OS_ASSIGN) {
    result = OS_NEW_ROW;
  }
  *out_row_idx = row_idx;
  return result;
}

// ---------------- assign_blobs_to_rows (makerow.cpp:2272-2444) ----------------
// `pass`/`drawing_skew` dropped (debug/ScrollView-only, zero effect on
// state -- see the tesseract-ocr textline.rs doc for the same call).
static void assign_blobs_to_rows(OracleBlock &block, const float *gradient, bool reject_misses,
                                  bool make_new_rows) {
  float g_length = 1.0f;
  if (gradient != nullptr) {
    g_length = std::sqrt(1 + (*gradient) * (*gradient));
  }

  std::sort(block.blobs.begin(), block.blobs.end(), blob_x_less);

  float smooth_factor = 1.0f;
  float block_skew = 0.0f;
  int32_t row_count = static_cast<int32_t>(block.rows.size());

  int32_t left_x = block.blobs.empty() ? block.block_left : block.blobs.front()[0];
  int32_t last_x = left_x;

  size_t i = 0;
  while (i < block.blobs.size()) {
    Box blob = block.blobs[i];
    int32_t left = blob[0], bottom_raw = blob[1], right = blob[2], top_raw = blob[3];
    if (gradient != nullptr) {
      block_skew = (1 - 1 / g_length) * static_cast<float>(bottom_raw) +
                   (*gradient) / g_length * static_cast<float>(left);
    } else if (static_cast<float>(left - last_x) > block.line_size / 2 &&
               static_cast<float>(last_x - left_x) > block.line_size * 2 && TEXTORD_INTERPOLATING_SKEW) {
      block_skew *= static_cast<float>(left - left_x) / static_cast<float>(last_x - left_x);
    }
    last_x = left;
    float top = static_cast<float>(top_raw) - block_skew;
    float bottom = static_cast<float>(bottom_raw) - block_skew;

    OverlapState overlap_result;
    size_t dest_idx = 0;

    if (!block.rows.empty()) {
      size_t r = 0;
      while (r + 1 < block.rows.size() && block.rows[r].min_y() > top) {
        r += 1;
      }
      size_t cursor_pos = r;
      if (block.rows[r].min_y() <= top && block.rows[r].max_y() >= bottom) {
        size_t winner;
        overlap_result = most_overlapping_row(block.rows, r, top, bottom, block.line_size, &winner);
        cursor_pos = winner;
        dest_idx = winner;
        if (overlap_result == OS_NEW_ROW && !reject_misses) {
          overlap_result = OS_ASSIGN;
        }
      } else {
        overlap_result = OS_NEW_ROW;
        dest_idx = r;
        if (!make_new_rows) {
          size_t prev_idx = data_relative(block.rows.size(), r, -1);
          float near_dist = block.rows[prev_idx].min_y() - top;
          if (bottom < block.rows[r].min_y()) {
            if (static_cast<double>(block.rows[r].min_y() - bottom) <=
                static_cast<double>(block.line_spacing - block.line_size) * K_DESCENDER_FRACTION) {
              overlap_result = OS_ASSIGN;
              dest_idx = r;
            }
          } else if (near_dist > 0 && near_dist < bottom - block.rows[r].max_y()) {
            dest_idx = prev_idx;
            if (static_cast<double>(block.rows[dest_idx].min_y() - bottom) <=
                static_cast<double>(block.line_spacing - block.line_size) * K_DESCENDER_FRACTION) {
              overlap_result = OS_ASSIGN;
            }
          } else if (static_cast<double>(top - block.rows[r].max_y()) <=
                     static_cast<double>(block.line_spacing - block.line_size) *
                         (TEXTORD_OVERLAP_X + K_ASCENDER_FRACTION)) {
            overlap_result = OS_ASSIGN;
            dest_idx = r;
          }
        }
      }

      if (overlap_result == OS_ASSIGN) {
        block.blobs.erase(block.blobs.begin() + static_cast<long>(i));
        block.rows[dest_idx].add_blob(blob, top, bottom, block.line_size);
      } else if (overlap_result == OS_NEW_ROW) {
        if (make_new_rows && top - bottom < block.max_blob_size) {
          block.blobs.erase(block.blobs.begin() + static_cast<long>(i));
          row_count++;
          OracleRow new_row = OracleRow::FromBlob(blob, top, bottom, block.line_size);
          size_t insert_at;
          if (bottom > block.rows[cursor_pos].min_y()) {
            insert_at = cursor_pos;
          } else {
            insert_at = cursor_pos + 1;
          }
          block.rows.insert(block.rows.begin() + static_cast<long>(insert_at), new_row);
          dest_idx = insert_at;
          smooth_factor = static_cast<float>(
              1.0 / (static_cast<double>(row_count) * TEXTORD_SKEW_LAG + TEXTORD_SKEWSMOOTH_OFFSET));
        } else {
          overlap_result = OS_REJECT;
        }
      }
    } else if (make_new_rows && top - bottom < block.max_blob_size) {
      overlap_result = OS_NEW_ROW;
      block.blobs.erase(block.blobs.begin() + static_cast<long>(i));
      row_count++;
      block.rows.push_back(OracleRow::FromBlob(blob, top, bottom, block.line_size));
      dest_idx = block.rows.size() - 1;
      smooth_factor = static_cast<float>(
          1.0 / (static_cast<double>(row_count) * TEXTORD_SKEW_LAG + TEXTORD_SKEWSMOOTH_OFFSET2));
    } else {
      overlap_result = OS_REJECT;
    }

    if (overlap_result != OS_REJECT) {
      size_t d = dest_idx;
      while (d > 0 && block.rows[d].min_y() > block.rows[d - 1].min_y()) {
        std::swap(block.rows[d], block.rows[d - 1]);
        d -= 1;
      }
      while (d + 1 < block.rows.size() && block.rows[d].min_y() < block.rows[d + 1].min_y()) {
        std::swap(block.rows[d], block.rows[d + 1]);
        d += 1;
      }
      bool singleton = block.rows[d].blobs.size() == 1;
      bool should_update;
      if (singleton) {
        should_update = true;
      } else {
        size_t n = block.rows[d].blobs.size();
        Box prev_box = block.rows[d].blobs[n - 2];
        Box this_blob = {left, bottom_raw, right, top_raw};
        should_update = !major_x_overlap(prev_box, this_blob);
      }
      if (should_update) {
        block_skew = (1 - smooth_factor) * block_skew +
                     smooth_factor * (static_cast<float>(bottom_raw) - block.rows[d].initial_min_y());
      }
    } else {
      i += 1;
    }
  }

  block.rows.erase(std::remove_if(block.rows.begin(), block.rows.end(),
                                   [](const OracleRow &r) { return r.blobs.empty(); }),
                    block.rows.end());
}

// ---------------- row_y_order (makerow.cpp:107-122) ----------------
static bool row_y_order_less(const OracleRow &a, const OracleRow &b) {
  // qsort: -1 (a first) if a.parallel_c() > b.parallel_c() -- descending.
  return a.parallel_c() > b.parallel_c();
}

// ---------------- fit_parallel_lms (makerow.cpp:1970-1991) ----------------
// textord_straight_baselines pinned false -- the unconstrained re-fit
// branch it would gate is not ported (see tesseract-ocr textline.rs doc).
static void fit_parallel_lms(float gradient, OracleRow &row) {
  DetLineFit lms;
  for (auto &b : row.blobs) {
    lms.Add(ICOORD((b[0] + b[2]) / 2, b[1]));
  }
  float c;
  double error = lms.ConstrainedFit(static_cast<double>(gradient), &c);
  row.set_parallel_line(gradient, c, error);
  row.set_line(gradient, c, error);
}

// ---------------- fit_parallel_rows (makerow.cpp:1928-1961) ----------------
static void fit_parallel_rows(OracleBlock &block, float gradient) {
  size_t i = 0;
  while (i < block.rows.size()) {
    if (block.rows[i].blobs.empty()) {
      block.rows.erase(block.rows.begin() + static_cast<long>(i));
    } else {
      fit_parallel_lms(gradient, block.rows[i]);
      i += 1;
    }
  }
  std::sort(block.rows.begin(), block.rows.end(), row_y_order_less);
}

// ---------------- find_best_dropout_row (makerow.cpp:696-757) ----------------
static bool find_best_dropout_row(const std::vector<OracleRow> &rows, size_t row_idx, int32_t distance,
                                   float dist_limit, int32_t line_index) {
  size_t n = rows.size();
  int32_t row_inc, abs_dist;
  if (distance < 0) {
    row_inc = 1;
    abs_dist = -distance;
  } else {
    row_inc = -1;
    abs_dist = distance;
  }
  if (static_cast<float>(abs_dist) > dist_limit) {
    return true;
  }
  bool at_last = row_idx == n - 1;
  bool at_first = row_idx == 0;
  if ((distance < 0 && !at_last) || (distance >= 0 && !at_first)) {
    int32_t row_offset = row_inc;
    for (;;) {
      size_t next_idx = data_relative(n, row_idx, row_offset);
      int32_t next_index = static_cast<int32_t>(std::floor(rows[next_idx].intercept()));
      bool nearer_neighbour =
          (distance < 0 && next_index < line_index && next_index > line_index + distance + distance) ||
          (distance >= 0 && next_index > line_index && next_index < line_index + distance + distance);
      bool tied_but_more_believable = (next_index == line_index || next_index == line_index + distance + distance) &&
                                       rows[row_idx].believability() <= rows[next_idx].believability();
      if (nearer_neighbour || tied_but_more_believable) {
        return true;
      }
      row_offset += row_inc;
      bool cont = (next_index == line_index || next_index == line_index + distance + distance) &&
                  row_offset < static_cast<int32_t>(n);
      if (!cont) {
        break;
      }
    }
  }
  return false;
}

// ---------------- delete_non_dropout_rows (makerow.cpp:612-688) ----------------
static void delete_non_dropout_rows(OracleBlock &block, float gradient) {
  if (block.rows.empty()) {
    return;
  }
  int32_t block_left, block_bottom, block_right, block_top;
  deskew_block_coords(block, gradient, &block_left, &block_bottom, &block_right, &block_top);
  int32_t min_y = block_bottom - 1;
  int32_t max_y = block_top + 1;
  for (auto &row : block.rows) {
    int32_t line_index = static_cast<int32_t>(std::floor(row.intercept()));
    if (line_index <= min_y) {
      min_y = line_index - 1;
    }
    if (line_index >= max_y) {
      max_y = line_index + 1;
    }
  }
  int32_t line_count = max_y - min_y + 1;
  if (line_count <= 0) {
    return;
  }

  std::vector<Box> all_blobs;
  for (auto &row : block.rows) {
    for (auto &b : row.blobs) {
      all_blobs.push_back(b);
    }
  }
  std::vector<int32_t> occupation, deltas;
  compute_line_occupation(all_blobs, gradient, min_y, max_y, occupation, deltas);
  int32_t low_window =
      static_cast<int32_t>(ceil(static_cast<double>(block.line_spacing) * (K_DESCENDER_FRACTION + K_ASCENDER_FRACTION)));
  int32_t high_window =
      static_cast<int32_t>(ceil(static_cast<double>(block.line_spacing) * (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION)));
  std::vector<int32_t> thresholds(line_count);
  compute_occupation_threshold(low_window, high_window, line_count, occupation.data(),
                                TEXTORD_OCCUPANCY_THRESHOLD, thresholds.data());
  compute_dropout_distances(occupation.data(), thresholds.data(), line_count);
  // 'thresholds' now holds the final dropout distances (matches the real
  // C++ reusing 'deltas' in place for both calls).

  size_t i = 0;
  while (i < block.rows.size()) {
    int32_t line_index = static_cast<int32_t>(std::floor(block.rows[i].intercept()));
    int32_t distance = thresholds[static_cast<size_t>(line_index - min_y)];
    if (find_best_dropout_row(block.rows, i, distance, block.line_spacing / 2, line_index)) {
      for (auto &b : block.rows[i].blobs) {
        block.blobs.push_back(b);
      }
      block.rows.erase(block.rows.begin() + static_cast<long>(i));
    } else {
      i += 1;
    }
  }
  for (auto &row : block.rows) {
    for (auto &b : row.blobs) {
      block.blobs.push_back(b);
    }
    row.blobs.clear();
  }
}

// ---------------- adjust_row_limits (makerow.cpp:1129-1156) ----------------
static void adjust_row_limits(OracleBlock &block) {
  for (auto &row : block.rows) {
    float size0 = row.max_y() - row.min_y();
    float size = static_cast<float>(static_cast<double>(size0) /
                                     (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION + K_DESCENDER_FRACTION));
    float ymax = static_cast<float>(static_cast<double>(size) * (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION));
    float ymin = static_cast<float>(-static_cast<double>(size) * K_DESCENDER_FRACTION);
    float intercept = row.intercept();
    row.set_limits(intercept + ymin, intercept + ymax);
    row.merged = false;
  }
}

// ---------------- compute_row_stats (makerow.cpp:1163-1247) ----------------
// textord_new_initial_xheight pinned true (see doc).
static void compute_row_stats(OracleBlock &block) {
  size_t n = block.rows.size();
  if (n == 0) {
    return;
  }
  std::vector<size_t> spacing_rows;
  bool have_prev = false;
  size_t prev_idx = 0;
  for (size_t k = 0; k < n; k++) {
    size_t idx = n - 1 - k; // n-1, n-2, ..., 0
    if (have_prev) {
      spacing_rows.push_back(prev_idx);
      float spacing = block.rows[idx].intercept() - block.rows[prev_idx].intercept();
      block.rows[prev_idx].spacing = (spacing < 0.1f && spacing > -0.1f) ? 0.0f : spacing;
    }
    prev_idx = idx;
    have_prev = true;
  }
  size_t final_row = prev_idx;
  block.baseline_offset = std::fmod(block.rows[final_row].parallel_c(), block.line_spacing);

  if (!spacing_rows.empty()) {
    size_t rowcount = spacing_rows.size();
    // PARITY PIN: row_spacing_order compares spacing only; std::nth_element's
    // tie handling is unspecified (as is the Rust side's). Both shells pin
    // the same TOTAL order: spacing, then intercept, then min_y.
    auto cmp = [&](size_t a, size_t b) {
      if (block.rows[a].spacing != block.rows[b].spacing) {
        return block.rows[a].spacing < block.rows[b].spacing;
      }
      if (block.rows[a].intercept() != block.rows[b].intercept()) {
        return block.rows[a].intercept() < block.rows[b].intercept();
      }
      return block.rows[a].min_y() < block.rows[b].min_y();
    };
    size_t ri = rowcount * 3 / 4;
    std::nth_element(spacing_rows.begin(), spacing_rows.begin() + static_cast<long>(ri), spacing_rows.end(), cmp);
    float iqr = block.rows[spacing_rows[ri]].spacing;
    size_t ri2 = rowcount / 4;
    std::nth_element(spacing_rows.begin(), spacing_rows.begin() + static_cast<long>(ri2), spacing_rows.end(), cmp);
    iqr -= block.rows[spacing_rows[ri2]].spacing;
    size_t ri3 = rowcount / 2;
    std::nth_element(spacing_rows.begin(), spacing_rows.begin() + static_cast<long>(ri3), spacing_rows.end(), cmp);
    size_t key_row_idx = spacing_rows[ri3];
    float key_spacing = block.rows[key_row_idx].spacing;
    if (rowcount > 2 &&
        static_cast<double>(iqr) < static_cast<double>(key_spacing) * TEXTORD_LINESPACE_IQRLIMIT) {
      if (key_spacing < block.line_spacing) {
        block.line_size = key_spacing;
      } else {
        block.line_size = block.line_spacing;
      }
      if (block.line_size < TEXTORD_MIN_XHEIGHT) {
        block.line_size = TEXTORD_MIN_XHEIGHT;
      }
      block.line_spacing = key_spacing;
      block.max_blob_size = static_cast<float>(static_cast<double>(block.line_spacing) * TEXTORD_EXCESS_BLOBSIZE);
    }
    block.baseline_offset = std::fmod(block.rows[key_row_idx].intercept(), block.line_spacing);
  }
}

// ---------------- expand_rows (makerow.cpp:976-1122) ----------------
// textord_new_initial_xheight pinned true (see doc).
static void expand_rows(OracleBlock &block, float gradient) {
  adjust_row_limits(block);
  if (block.rows.empty()) {
    return;
  }
  compute_row_stats(block);
  assign_blobs_to_rows(block, &gradient, true, false); // pass 4
  if (block.rows.empty()) {
    return;
  }
  fit_parallel_rows(block, gradient);
  if (block.rows.empty()) {
    return;
  }

  size_t row_idx = block.rows.size() - 1;
  for (;;) {
    float y_max0 = block.rows[row_idx].max_y();
    float y_min0 = block.rows[row_idx].min_y();
    float intercept = block.rows[row_idx].intercept();
    float y_bottom = static_cast<float>(static_cast<double>(intercept) -
                                         static_cast<double>(block.line_size) * TEXTORD_EXPANSION_FACTOR *
                                             K_DESCENDER_FRACTION);
    float y_top = static_cast<float>(static_cast<double>(intercept) +
                                      static_cast<double>(block.line_size) * TEXTORD_EXPANSION_FACTOR *
                                          (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION));
    float y_min = y_min0;
    float y_max = y_max0;

    if (y_min0 > y_bottom) {
      bool swallowed = true;
      while (swallowed && row_idx + 1 < block.rows.size()) {
        swallowed = false;
        size_t test_idx = row_idx + 1;
        if (block.rows[test_idx].max_y() > y_bottom) {
          if (block.rows[test_idx].min_y() > y_bottom) {
            for (auto &b : block.rows[test_idx].blobs) {
              block.rows[row_idx].blobs.push_back(b);
            }
            block.rows.erase(block.rows.begin() + static_cast<long>(test_idx));
            swallowed = true;
          } else if (block.rows[test_idx].max_y() < y_min) {
            y_bottom = block.rows[test_idx].max_y();
          } else {
            y_bottom = y_min;
          }
        }
      }
      y_min = y_bottom;
    }
    if (y_max0 < y_top) {
      bool swallowed = true;
      while (swallowed && row_idx > 0) {
        swallowed = false;
        size_t test_idx = row_idx - 1;
        if (block.rows[test_idx].min_y() < y_top) {
          if (block.rows[test_idx].max_y() < y_top) {
            for (auto &b : block.rows[test_idx].blobs) {
              block.rows[row_idx].blobs.push_back(b);
            }
            block.rows.erase(block.rows.begin() + static_cast<long>(test_idx));
            row_idx -= 1;
            swallowed = true;
          } else if (block.rows[test_idx].min_y() < y_max) {
            y_top = block.rows[test_idx].min_y();
          } else {
            y_top = y_max;
          }
        }
      }
      y_max = y_top;
    }
    block.rows[row_idx].set_limits(y_min, y_max);

    if (row_idx == 0) {
      break;
    }
    row_idx -= 1;
  }
}

// ---------------- make_initial_textrows (makerow.cpp:254-289) ----------------
static void make_initial_textrows(OracleBlock &block) {
  assign_blobs_to_rows(block, nullptr, true, true);
  for (auto &row : block.rows) {
    float m, c;
    double error;
    fit_lms_line(row.blobs, &m, &c, &error);
    row.set_line(m, c, error);
  }
}

// ---------------- compute_page_skew (makerow.cpp:315-405) ----------------
static void compute_page_skew(const std::vector<OracleRow> &rows, float *page_m, float *page_err) {
  size_t row_count_total = rows.size();
  if (row_count_total == 0) {
    *page_m = 0.0f;
    *page_err = 0.0f;
    return;
  }
  std::vector<float> gradients, errors;
  for (auto &row : rows) {
    int32_t blob_count = static_cast<int32_t>(row.blobs.size());
    int32_t row_err = static_cast<int32_t>(std::ceil(row.line_error()));
    if (row_err <= 0) {
      row_err = 1;
    }
    if (TEXTORD_BIASED_SKEWCALC) {
      blob_count /= row_err;
      for (blob_count /= row_err; blob_count > 0; blob_count--) {
        gradients.push_back(row.line_m());
        errors.push_back(row.line_error());
      }
    } else if (blob_count >= TEXTORD_MIN_BLOBS_IN_ROW) {
      gradients.push_back(row.line_m());
      errors.push_back(row.line_error());
    }
  }
  if (gradients.empty()) {
    for (auto &row : rows) {
      gradients.push_back(row.line_m());
      errors.push_back(row.line_error());
    }
  }
  size_t row_count = gradients.size();
  size_t row_index = static_cast<size_t>(static_cast<double>(row_count) * TEXTORD_SKEW_ILE);
  std::nth_element(gradients.begin(), gradients.begin() + static_cast<long>(row_index), gradients.end());
  *page_m = gradients[row_index];
  std::nth_element(errors.begin(), errors.begin() + static_cast<long>(row_index), errors.end());
  *page_err = errors[row_index];
}

// ============================================================
// fixture reader + main
// ============================================================
struct Reader {
  FILE *f;
  explicit Reader(const char *path) { f = fopen(path, "rb"); }
  ~Reader() {
    if (f) {
      fclose(f);
    }
  }
  int32_t i32() {
    int32_t v;
    if (fread(&v, 4, 1, f) != 1) {
      abort();
    }
    return v;
  }
  uint32_t u32() {
    uint32_t v;
    if (fread(&v, 4, 1, f) != 1) {
      abort();
    }
    return v;
  }
  float f32() {
    float v;
    if (fread(&v, 4, 1, f) != 1) {
      abort();
    }
    return v;
  }
};

static void dump_f32_hex(float v) {
  uint32_t bits;
  memcpy(&bits, &v, 4);
  printf("%08x", bits);
}

static uint32_t f32_bits(float v) {
  uint32_t bits;
  memcpy(&bits, &v, 4);
  return bits;
}

static std::string dump_row_blobs(const OracleRow &row) {
  std::string s;
  for (size_t i = 0; i < row.blobs.size(); i++) {
    if (i) {
      s += ";";
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%d,%d,%d,%d", row.blobs[i][0], row.blobs[i][1], row.blobs[i][2], row.blobs[i][3]);
    s += buf;
  }
  return s;
}


// ============================================================
// Wave 3: x-height chain (makerow.cpp:1276-1690 + makerow.h inlines),
// transcribed verbatim against real (compiled) STATS. This mirrors the
// tesseract-ocr textline.rs wave-3 port; parity = the two transcriptions
// agree AND this side is a faithful copy of makerow.cpp.
// ============================================================
static const int XH_MAX_HEIGHT_MODES = 12;
static const double TEXTORD_MINXH = 0.25;
static const float TEXTORD_XHEIGHT_MODE_FRACTION = 0.4f;
static const float TEXTORD_ASCHEIGHT_MODE_FRACTION = 0.08f;
static const float TEXTORD_DESCHEIGHT_MODE_FRACTION = 0.08f;
static const float TEXTORD_ASCX_RATIO_MIN = 1.25f;
static const float TEXTORD_ASCX_RATIO_MAX = 1.8f;
static const float TEXTORD_DESCX_RATIO_MIN = 0.25f;
static const float TEXTORD_DESCX_RATIO_MAX = 0.6f;
static const float TEXTORD_XHEIGHT_ERROR_MARGIN = 0.1f;
static const float TEXTORD_MIN_BLOB_HEIGHT_FRACTION = 0.75f;
static const bool TEXTORD_SINGLE_HEIGHT_MODE = false;
static const double K_XHEIGHT_CAP_RATIO =
    K_XHEIGHT_FRACTION / (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION);

enum XhRowCategory { XH_ASC, XH_DESC, XH_UNKNOWN, XH_INVALID };

static void get_min_max_xheight(int block_linesize, int *min_height, int *max_height) {
  *min_height = static_cast<int32_t>(std::floor(block_linesize * TEXTORD_MINXH));
  if (*min_height < static_cast<int>(TEXTORD_MIN_XHEIGHT)) {
    *min_height = static_cast<int>(TEXTORD_MIN_XHEIGHT);
  }
  *max_height = static_cast<int32_t>(std::ceil(block_linesize * 3.0));
}

static XhRowCategory get_row_category(const OracleRow &row) {
  if (row.xheight <= 0) {
    return XH_INVALID;
  }
  return (row.ascrise > 0) ? XH_ASC : (row.descdrop != 0) ? XH_DESC : XH_UNKNOWN;
}

static bool within_error_margin(float test, float num, float margin) {
  return (test >= num * (1 - margin) && test <= num * (1 + margin));
}

static void fill_heights_ora(const OracleRow &row, float gradient, int min_height, int max_height,
                             STATS *heights, STATS *floating_heights) {
  for (const Box &b : row.blobs) {
    float xcentre = (b[0] + b[2]) / 2.0f;
    float height = static_cast<float>(b[3] - b[1]);
    float top_adj = static_cast<float>(b[3]) - (gradient * xcentre + row.parallel_c());
    if (top_adj >= min_height && top_adj <= max_height) {
      int bucket = static_cast<int>(std::floor(top_adj + 0.5f));
      heights->add(bucket, 1);
      if (height / top_adj < TEXTORD_MIN_BLOB_HEIGHT_FRACTION) {
        floating_heights->add(bucket, 1);
      }
    }
  }
}

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

static int compute_xheight_from_modes(STATS *heights, STATS *floating_heights, bool cap_only,
                                      int min_height, int max_height, float *xheight,
                                      float *ascrise) {
  int blob_index = heights->mode();
  int blob_count = heights->pile_count(blob_index);
  if (blob_count == 0) {
    return 0;
  }
  int modes[XH_MAX_HEIGHT_MODES];
  bool in_best_pile = false;
  int prev_size = -INT32_MAX;
  int best_count = 0;
  int mode_count = compute_height_modes(heights, min_height, max_height, modes, XH_MAX_HEIGHT_MODES);
  if (cap_only && mode_count > 1) {
    mode_count = 1;
  }
  int x;
  for (x = 0; x < mode_count - 1; x++) {
    if (modes[x] != prev_size + 1) {
      in_best_pile = false;
    }
    int modes_x_count = heights->pile_count(modes[x]) - floating_heights->pile_count(modes[x]);
    if ((modes_x_count >= blob_count * TEXTORD_XHEIGHT_MODE_FRACTION) &&
        (in_best_pile || modes_x_count > best_count)) {
      for (int asc = x + 1; asc < mode_count; asc++) {
        float ratio = static_cast<float>(modes[asc]) / static_cast<float>(modes[x]);
        if (TEXTORD_ASCX_RATIO_MIN < ratio && ratio < TEXTORD_ASCX_RATIO_MAX &&
            (heights->pile_count(modes[asc]) >= blob_count * TEXTORD_ASCHEIGHT_MODE_FRACTION)) {
          if (modes_x_count > best_count) {
            in_best_pile = true;
            best_count = modes_x_count;
          }
          prev_size = modes[x];
          *xheight = static_cast<float>(modes[x]);
          *ascrise = static_cast<float>(modes[asc] - modes[x]);
        }
      }
    }
  }
  if (*xheight == 0) {
    if (floating_heights->get_total() > 0) {
      for (x = min_height; x < max_height; ++x) {
        heights->add(x, -(floating_heights->pile_count(x)));
      }
      blob_index = heights->mode();
      for (x = min_height; x < max_height; ++x) {
        heights->add(x, floating_heights->pile_count(x));
      }
    }
    *xheight = static_cast<float>(blob_index);
    *ascrise = 0.0f;
    best_count = heights->pile_count(blob_index);
  }
  return best_count;
}

static int32_t compute_row_descdrop(OracleRow &row, float gradient, int xheight_blob_count,
                                    STATS *asc_heights) {
  int i_min = asc_heights->min_bucket();
  if ((i_min / row.xheight) < TEXTORD_ASCX_RATIO_MIN) {
    i_min = static_cast<int>(std::floor(row.xheight * TEXTORD_ASCX_RATIO_MIN + 0.5));
  }
  int i_max = asc_heights->max_bucket();
  if ((i_max / row.xheight) > TEXTORD_ASCX_RATIO_MAX) {
    i_max = static_cast<int>(std::floor(row.xheight * TEXTORD_ASCX_RATIO_MAX));
  }
  int num_potential_asc = 0;
  for (int i = i_min; i <= i_max; ++i) {
    num_potential_asc += asc_heights->pile_count(i);
  }
  int32_t min_height = static_cast<int32_t>(std::floor(row.xheight * TEXTORD_DESCX_RATIO_MIN + 0.5));
  int32_t max_height = static_cast<int32_t>(std::floor(row.xheight * TEXTORD_DESCX_RATIO_MAX));
  STATS heights(min_height, max_height);
  for (const Box &b : row.blobs) {
    float xcentre = (b[0] + b[2]) / 2.0f;
    float height = (gradient * xcentre + row.parallel_c() - b[1]);
    if (height >= min_height && height <= max_height) {
      heights.add(static_cast<int>(std::floor(height + 0.5)), 1);
    }
  }
  int blob_index = heights.mode();
  int blob_count = heights.pile_count(blob_index);
  float total_fraction = (TEXTORD_DESCHEIGHT_MODE_FRACTION + TEXTORD_ASCHEIGHT_MODE_FRACTION);
  if (static_cast<float>(blob_count + num_potential_asc) < xheight_blob_count * total_fraction) {
    blob_count = 0;
  }
  return blob_count > 0 ? -blob_index : 0;
}

static void compute_row_xheight(OracleRow &row, float rotation_y, float gradient,
                                int block_line_size) {
  if (!row.rep_chars_marked) {
    row.rep_chars_marked = true;
  }
  int min_height, max_height;
  get_min_max_xheight(block_line_size, &min_height, &max_height);
  STATS heights(min_height, max_height);
  STATS floating_heights(min_height, max_height);
  fill_heights_ora(row, gradient, min_height, max_height, &heights, &floating_heights);
  row.ascrise = 0.0f;
  row.xheight = 0.0f;
  row.xheight_evidence = compute_xheight_from_modes(
      &heights, &floating_heights, TEXTORD_SINGLE_HEIGHT_MODE && rotation_y == 0.0, min_height,
      max_height, &(row.xheight), &(row.ascrise));
  row.descdrop = 0.0f;
  if (row.xheight > 0) {
    row.descdrop = static_cast<float>(compute_row_descdrop(row, gradient, row.xheight_evidence, &heights));
  }
}

static void correct_row_xheight(OracleRow &row, float xheight, float ascrise, float descdrop) {
  XhRowCategory row_category = get_row_category(row);
  bool normal_xheight = within_error_margin(row.xheight, xheight, TEXTORD_XHEIGHT_ERROR_MARGIN);
  bool cap_xheight =
      within_error_margin(row.xheight, xheight + ascrise, TEXTORD_XHEIGHT_ERROR_MARGIN);
  if (row_category == XH_ASC) {
    if (row.descdrop >= 0) {
      row.descdrop = row.xheight * (descdrop / xheight);
    }
  } else if (row_category == XH_INVALID ||
             (row_category == XH_DESC && (normal_xheight || cap_xheight)) ||
             (row_category == XH_UNKNOWN && normal_xheight)) {
    row.xheight = xheight;
    row.ascrise = ascrise;
    row.descdrop = descdrop;
  } else if (row_category == XH_DESC) {
    row.ascrise = row.xheight * (ascrise / xheight);
  } else if (row_category == XH_UNKNOWN) {
    row.all_caps = true;
    if (cap_xheight) {
      row.xheight = xheight;
      row.ascrise = ascrise;
      row.descdrop = descdrop;
    } else {
      row.ascrise = row.xheight * (ascrise / (xheight + ascrise));
      row.xheight -= row.ascrise;
      row.descdrop = row.xheight * (descdrop / xheight);
    }
  }
}

static void compute_block_xheight(OracleBlock &block, float gradient, float rotation_y) {
  float asc_frac_xheight = K_ASCENDER_FRACTION / K_XHEIGHT_FRACTION;
  float desc_frac_xheight = K_DESCENDER_FRACTION / K_XHEIGHT_FRACTION;
  if (block.rows.empty()) {
    return;
  }
  int block_linesize = static_cast<int>(block.line_size);
  int min_height, max_height;
  get_min_max_xheight(block_linesize, &min_height, &max_height);
  STATS row_asc_xheights(min_height, max_height);
  STATS row_asc_ascrise(static_cast<int>(min_height * asc_frac_xheight),
                        static_cast<int>(max_height * asc_frac_xheight));
  int min_desc_height = static_cast<int>(min_height * desc_frac_xheight);
  int max_desc_height = static_cast<int>(max_height * desc_frac_xheight);
  STATS row_asc_descdrop(min_desc_height, max_desc_height);
  STATS row_desc_xheights(min_height, max_height);
  STATS row_desc_descdrop(min_desc_height, max_desc_height);
  STATS row_cap_xheights(min_height, max_height);
  STATS row_cap_floating_xheights(min_height, max_height);
  for (auto &row : block.rows) {
    if (row.xheight <= 0) {
      compute_row_xheight(row, rotation_y, gradient, block_linesize);
    }
    XhRowCategory row_category = get_row_category(row);
    if (row_category == XH_ASC) {
      row_asc_xheights.add(static_cast<int32_t>(row.xheight), row.xheight_evidence);
      row_asc_ascrise.add(static_cast<int32_t>(row.ascrise), row.xheight_evidence);
      row_asc_descdrop.add(static_cast<int32_t>(-row.descdrop), row.xheight_evidence);
    } else if (row_category == XH_DESC) {
      row_desc_xheights.add(static_cast<int32_t>(row.xheight), row.xheight_evidence);
      row_desc_descdrop.add(static_cast<int32_t>(-row.descdrop), row.xheight_evidence);
    } else if (row_category == XH_UNKNOWN) {
      fill_heights_ora(row, gradient, min_height, max_height, &row_cap_xheights,
                       &row_cap_floating_xheights);
    }
  }
  float xheight = 0.0, ascrise = 0.0, descdrop = 0.0;
  if (row_asc_xheights.get_total() > 0) {
    xheight = row_asc_xheights.median();
    ascrise = row_asc_ascrise.median();
    descdrop = -row_asc_descdrop.median();
  } else if (row_desc_xheights.get_total() > 0) {
    xheight = row_desc_xheights.median();
    descdrop = -row_desc_descdrop.median();
  } else if (row_cap_xheights.get_total() > 0) {
    compute_xheight_from_modes(&row_cap_xheights, &row_cap_floating_xheights,
                               TEXTORD_SINGLE_HEIGHT_MODE && rotation_y == 0.0, min_height,
                               max_height, &(xheight), &(ascrise));
    if (ascrise == 0) {
      xheight = row_cap_xheights.median() * K_XHEIGHT_CAP_RATIO;
    }
  } else {
    xheight = block.line_size * K_XHEIGHT_FRACTION;
  }
  bool corrected_xheight = false;
  if (xheight < TEXTORD_MIN_XHEIGHT) {
    xheight = static_cast<float>(TEXTORD_MIN_XHEIGHT);
    corrected_xheight = true;
  }
  if (corrected_xheight || ascrise <= 0) {
    ascrise = xheight * asc_frac_xheight;
  }
  if (corrected_xheight || descdrop >= 0) {
    descdrop = -(xheight * desc_frac_xheight);
  }
  block.xheight = xheight;
  for (auto &row : block.rows) {
    correct_row_xheight(row, xheight, ascrise, descdrop);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]);
    return 1;
  }
  Reader r(argv[1]);
  if (!r.f) {
    fprintf(stderr, "cannot open fixture %s\n", argv[1]);
    return 1;
  }

  uint32_t n_blobs = r.u32();
  OracleBlock block;
  block.blobs.resize(n_blobs);
  for (uint32_t i = 0; i < n_blobs; i++) {
    block.blobs[i] = {r.i32(), r.i32(), r.i32(), r.i32()};
  }
  block.line_spacing = r.f32();
  block.line_size = r.f32();
  block.max_blob_size = r.f32();
  block.block_left = r.i32();

  printf("FIXTURE n_blobs=%u\n", n_blobs);

  // ---- Stage 1 ----
  make_initial_textrows(block);
  printf("STAGE1_ROWS %zu\n", block.rows.size());
  for (size_t i = 0; i < block.rows.size(); i++) {
    const OracleRow &row = block.rows[i];
    printf("STAGE1_ROW[%zu] min_hex=", i);
    dump_f32_hex(row.min_y());
    printf(" max_hex=");
    dump_f32_hex(row.max_y());
    printf(" m_hex=");
    dump_f32_hex(row.line_m());
    printf(" c_hex=");
    dump_f32_hex(row.line_c());
    printf(" err_hex=");
    dump_f32_hex(row.line_error());
    printf(" blobs=%s\n", dump_row_blobs(row).c_str());
  }

  // ---- Stage 2 ----
  float page_m, page_err;
  compute_page_skew(block.rows, &page_m, &page_err);
  printf("STAGE2_SKEW page_m_hex=");
  dump_f32_hex(page_m);
  printf(" page_err_hex=");
  dump_f32_hex(page_err);
  printf("\n");

  // ---- Stage 3 ----
  fit_parallel_rows(block, page_m);
  printf("STAGE3_ROWS %zu\n", block.rows.size());
  for (size_t i = 0; i < block.rows.size(); i++) {
    const OracleRow &row = block.rows[i];
    printf("STAGE3_ROW[%zu] parc_hex=", i);
    dump_f32_hex(row.parallel_c());
    printf(" intercept_hex=");
    dump_f32_hex(row.intercept());
    printf(" believ_hex=");
    dump_f32_hex(row.believability());
    printf(" nblobs=%zu\n", row.blobs.size());
  }

  // ---- Stage 4 ----
  delete_non_dropout_rows(block, page_m);
  printf("STAGE4_ROWS %zu pool=%zu\n", block.rows.size(), block.blobs.size());
  for (size_t i = 0; i < block.rows.size(); i++) {
    const OracleRow &row = block.rows[i];
    printf("STAGE4_ROW[%zu] min_hex=", i);
    dump_f32_hex(row.min_y());
    printf(" max_hex=");
    dump_f32_hex(row.max_y());
    printf(" intercept_hex=");
    dump_f32_hex(row.intercept());
    printf("\n");
  }

  // ---- Stage 5 ----
  expand_rows(block, page_m);
  printf("STAGE5_ROWS %zu line_spacing_hex=", block.rows.size());
  dump_f32_hex(block.line_spacing);
  printf(" line_size_hex=");
  dump_f32_hex(block.line_size);
  printf(" max_blob_size_hex=");
  dump_f32_hex(block.max_blob_size);
  printf(" baseline_offset_hex=");
  dump_f32_hex(block.baseline_offset);
  printf("\n");
  for (size_t i = 0; i < block.rows.size(); i++) {
    const OracleRow &row = block.rows[i];
    printf("STAGE5_ROW[%zu] min_hex=", i);
    dump_f32_hex(row.min_y());
    printf(" max_hex=");
    dump_f32_hex(row.max_y());
    printf("\n");
  }

  // ---- Stage 6: reconsolidate + the three assign_blobs_to_rows passes ----
  for (auto &row : block.rows) {
    for (auto &b : row.blobs) {
      block.blobs.push_back(b);
    }
    row.blobs.clear();
  }
  assign_blobs_to_rows(block, &page_m, false, false); // pass 1
  assign_blobs_to_rows(block, &page_m, true, true);   // pass 2
  assign_blobs_to_rows(block, &page_m, false, false); // pass 3

  size_t total_assigned = 0;
  for (auto &row : block.rows) {
    total_assigned += row.blobs.size();
  }
  printf("STAGE6_ROWS %zu pool=%zu total_assigned=%zu\n", block.rows.size(), block.blobs.size(), total_assigned);
  for (size_t i = 0; i < block.rows.size(); i++) {
    printf("STAGE6_ROW[%zu] blobs=%s\n", i, dump_row_blobs(block.rows[i]).c_str());
  }


  // ---- Stage 7: compute_block_xheight (wave 3) ----
  compute_block_xheight(block, page_m, 0.0f);
  printf("STAGE7_BLOCK xheight_hex=%08x\n", f32_bits(block.xheight));
  for (size_t i = 0; i < block.rows.size(); ++i) {
    const OracleRow &row = block.rows[i];
    XhRowCategory cat = get_row_category(row);
    int catn = (cat == XH_ASC) ? 0 : (cat == XH_DESC) ? 1 : (cat == XH_UNKNOWN) ? 2 : 3;
    printf("STAGE7_ROW[%zu] xheight_hex=%08x ascrise_hex=%08x descdrop_hex=%08x evidence=%d category=%d all_caps=%d\n",
           i, f32_bits(row.xheight), f32_bits(row.ascrise), f32_bits(row.descdrop),
           row.xheight_evidence, catn, row.all_caps ? 1 : 0);
  }
  return 0;
}
