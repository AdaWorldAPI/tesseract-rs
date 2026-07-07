// Byte-parity oracle for the colour -> grey leaf: leptonica's
// `pixConvertRGBToGray` (and its default-weight luminance config, which is
// what `pixConvertRGBToLuminance` calls: pixConvertRGBToGray(pixs,0,0,0) —
// see pixconv.c line 744). Validates a Rust transcode of the RGB->grey step
// that feeds the recognizer's ComputeBlackWhite/FromPix pipeline for colour
// source images.
//
// Reads a synthetic 24bpp RGB image from a shared .bin: i32 width, i32
// height, then width*height*3 RGB bytes, row-major (row 0 first, R,G,B per
// pixel). Builds a REAL leptonica 32bpp Pix via composeRGBPixel +
// pixSetPixel, calls the REAL public `pixConvertRGBToGray(pixs, rwt, gwt,
// bwt)`, and dumps the resulting 8bpp grey pixel grid for a byte-identical
// diff against the Rust side.
//
// If the shared .bin is absent, self-generates it with the session pattern:
//   r = (x*37 + y*11) % 256
//   g = (x*7  + y*13) % 256
//   b = ((x*3) ^ (y*5)) % 256
//
//   ./pixconv_oracle [bin] [rwt gwt bwt]
//   default weights 0 0 0 -> the luminance config (pixConvertRGBToGray's
//   internal default when all three weights are 0.0, per pixconv.c).
#include <allheaders.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool generate_input(const char *path, int32_t w, int32_t h) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  fwrite(&w, sizeof(int32_t), 1, f);
  fwrite(&h, sizeof(int32_t), 1, f);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      uint8_t r = static_cast<uint8_t>((x * 37 + y * 11) % 256);
      uint8_t g = static_cast<uint8_t>((x * 7 + y * 13) % 256);
      uint8_t b = static_cast<uint8_t>(((x * 3) ^ (y * 5)) % 256);
      fputc(r, f);
      fputc(g, f);
      fputc(b, f);
    }
  }
  fclose(f);
  return true;
}

int main(int argc, char **argv) {
  const char *bin_path = argc > 1 ? argv[1] : "/tmp/pixconv_input.bin";
  float rwt = argc > 2 ? static_cast<float>(atof(argv[2])) : 0.0f;
  float gwt = argc > 3 ? static_cast<float>(atof(argv[3])) : 0.0f;
  float bwt = argc > 4 ? static_cast<float>(atof(argv[4])) : 0.0f;

  FILE *bf = fopen(bin_path, "rb");
  if (!bf) {
    // Self-generate a default 24x36 input if the shared file is absent.
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
  std::vector<uint8_t> rgb(npix * 3);
  if (fread(rgb.data(), 1, rgb.size(), bf) != rgb.size()) {
    fprintf(stderr, "bin truncated (pixels)\n");
    fclose(bf);
    return 1;
  }
  fclose(bf);

  // ---- Build a REAL 32bpp leptonica Pix from the RGB buffer. ----
  PIX *pix = pixCreate(width, height, 32);
  if (!pix) {
    fprintf(stderr, "pixCreate failed\n");
    return 1;
  }
  for (int32_t y = 0; y < height; ++y) {
    for (int32_t x = 0; x < width; ++x) {
      const size_t idx = (static_cast<size_t>(y) * width + x) * 3;
      l_uint32 val = 0;
      composeRGBPixel(rgb[idx], rgb[idx + 1], rgb[idx + 2], &val);
      pixSetPixel(pix, x, y, val);
    }
  }

  // ---- The REAL public pixConvertRGBToGray. rwt=gwt=bwt=0 is the
  // luminance config (pixConvertRGBToLuminance calls this exact form). ----
  PIX *pixg = pixConvertRGBToGray(pix, rwt, gwt, bwt);
  if (!pixg) {
    fprintf(stderr, "pixConvertRGBToGray failed\n");
    return 1;
  }
  int wd = pixGetWidth(pixg), hd = pixGetHeight(pixg);
  printf("dim\t%d\t%d\n", wd, hd);
  for (int y = 0; y < hd; ++y) {
    printf("r\t%d", y);
    for (int x = 0; x < wd; ++x) {
      l_uint32 val = 0;
      pixGetPixel(pixg, x, y, &val);
      printf("\t%u", val);
    }
    printf("\n");
  }

  pixDestroy(&pixg);
  pixDestroy(&pix);
  return 0;
}
