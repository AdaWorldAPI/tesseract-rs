// Byte-parity oracle for the Otsu binarization chain: the REAL
// `tesseract::OtsuThreshold` (otsuthr.cpp) followed by a replication of
// `ImageThresholder::ThresholdRectToPix`'s per-pixel decision
// (thresholder.cpp:394-422). Validates a Rust transcode of the grey-image
// -> Otsu-threshold -> binary-image chain that feeds the recognizer's
// binarization path for non-pre-binarized input.
//
// NOTE on ThresholdRectToPix: that method is a private/protected member of
// `tesseract::ImageThresholder` (bound to instance state rect_left_/
// rect_top_/rect_width_/rect_height_ via the class, not freely callable).
// It is not reachable through a public API without constructing a full
// ImageThresholder + TessBaseAPI. Per the task's fallback clause, this
// oracle REPLICATES its documented per-pixel decision directly
// (thresholder.cpp:406-419), specialized to num_channels==1 (single-channel
// grey source, which is what this leaf always sees):
//
//   pixel = src grey value at (x, y)
//   white_result = true
//   if hi_values[0] >= 0 && (pixel > thresholds[0]) == (hi_values[0] == 0):
//       white_result = false
//   out = white_result ? 0 (CLEAR_DATA_BIT) : 255 (SET_DATA_BIT, foreground)
//
// This is a straight unpacking of the bit-level CLEAR_DATA_BIT/SET_DATA_BIT
// writes into 0/255 bytes for the dump; the *decision* per pixel is
// byte-for-byte the same predicate as the real method, just read via
// pixGetPixel instead of the packed 1bpp pixdata/wpl indexing (which is an
// unrelated storage-format detail, not part of the algorithm being proven).
//
// OtsuThreshold itself IS the real public tesseract:: function, called
// exactly as OtsuThresholdRectToPix calls it (full-image rect: left=0,
// top=0, width=w, height=h).
//
// Reads a synthetic 8-bit grey image from a shared .bin: i32 width, i32
// height, then width*height grey bytes, row-major. If absent, self-generates
// with the session pattern ((x*37+y*11)^(x*y)) % 256.
//
//   ./otsu_oracle [bin]
#include "otsuthr.h"
#include "image.h"

#include <allheaders.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tesseract;

static bool generate_input(const char *path, int32_t w, int32_t h) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  fwrite(&w, sizeof(int32_t), 1, f);
  fwrite(&h, sizeof(int32_t), 1, f);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      uint8_t v = static_cast<uint8_t>(((x * 37 + y * 11) ^ (x * y)) % 256);
      fputc(v, f);
    }
  }
  fclose(f);
  return true;
}

int main(int argc, char **argv) {
  const char *bin_path = argc > 1 ? argv[1] : "/tmp/otsu_input.bin";

  FILE *bf = fopen(bin_path, "rb");
  if (!bf) {
    fprintf(stderr, "note: %s absent, self-generating 24x36\n", bin_path);
    if (!generate_input(bin_path, 24, 36)) {
      fprintf(stderr, "failed to generate %s\n", bin_path);
      return 1;
    }
    bf = fopen(bin_path, "rb");
    if (!bf) {
      fprintf(stderr, "open %s failed after generate\n", bin_path);
      return 1;
    }
  }

  auto read_i32 = [&](int32_t *v) {
    if (fread(v, sizeof(int32_t), 1, bf) != 1) {
      fprintf(stderr, "bin truncated (header)\n");
      exit(1);
    }
  };
  int32_t width = 0, height = 0;
  read_i32(&width);
  read_i32(&height);
  if (width <= 0 || height <= 0) {
    fprintf(stderr, "bad dims %d x %d\n", width, height);
    return 1;
  }
  const size_t npix = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<uint8_t> pixels(npix);
  if (fread(pixels.data(), 1, pixels.size(), bf) != pixels.size()) {
    fprintf(stderr, "bin truncated (pixels)\n");
    fclose(bf);
    return 1;
  }
  fclose(bf);

  // ---- Build a REAL 8bpp leptonica Pix. ----
  PIX *pix = pixCreate(width, height, 8);
  if (!pix) {
    fprintf(stderr, "pixCreate failed\n");
    return 1;
  }
  for (int32_t y = 0; y < height; ++y) {
    for (int32_t x = 0; x < width; ++x) {
      pixSetPixel(pix, x, y, pixels[static_cast<size_t>(y) * width + x]);
    }
  }

  // ---- The REAL public tesseract::OtsuThreshold, full-image rect (matches
  // OtsuThresholdRectToPix's call with rect_left_=rect_top_=0,
  // rect_width_=w, rect_height_=h). ----
  std::vector<int> thresholds;
  std::vector<int> hi_values;
  int num_channels = OtsuThreshold(Image(pix), 0, 0, width, height, thresholds, hi_values);

  printf("otsu\t%d\t%d\n", thresholds.empty() ? -1 : thresholds[0],
         hi_values.empty() ? -1 : hi_values[0]);

  // ---- Replicated ThresholdRectToPix decision (num_channels==1 case; see
  // header comment for why this is a faithful unpacking, not a
  // reimplementation of the algorithm). ----
  int threshold0 = thresholds.empty() ? 0 : thresholds[0];
  int hi0 = hi_values.empty() ? -1 : hi_values[0];
  (void)num_channels;
  for (int32_t y = 0; y < height; ++y) {
    printf("b\t%d", y);
    for (int32_t x = 0; x < width; ++x) {
      l_uint32 val = 0;
      pixGetPixel(pix, x, y, &val);
      int pixel = static_cast<int>(val);
      bool white_result = true;
      if (hi0 >= 0 && (pixel > threshold0) == (hi0 == 0)) {
        white_result = false;
      }
      printf("\t%d", white_result ? 255 : 0);
    }
    printf("\n");
  }

  pixDestroy(&pix);
  return 0;
}
