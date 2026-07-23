// sauvola_oracle.cpp — byte-parity oracle for tesseract_ocr::binarize::sauvola_binarize.
//
// Calls leptonica pixSauvolaBinarize(pixs, whsize, factor, addborder=1, ...) and
// dumps, per pixel index, "<idx>\t<threshold>\t<binary 0|1>" — the exact shape the
// Rust example `sauvola_dump` prints. `addborder=1` is the document path (mirror
// border of whsize+1). Leptonica here is 1.82.0 (installed via apt); the transcode
// is from the AdaWorldAPI/leptonica fork src/{binarize.c,convolve.c,pix2.c}.
//
// Build:
//   g++ -std=c++17 sauvola_oracle.cpp -I/usr/include/leptonica -lleptonica -o /tmp/sauvola_oracle
// Run:
//   /tmp/sauvola_oracle <grey.pgm> <whsize> <factor>

#include <leptonica/allheaders.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <grey.pgm> <whsize> <factor>\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  l_int32 whsize = atoi(argv[2]);
  l_float32 factor = static_cast<l_float32>(atof(argv[3]));

  PIX* pixs = pixRead(path);
  if (!pixs) {
    fprintf(stderr, "pixRead failed: %s\n", path);
    return 1;
  }
  PIX* pixg = pixConvertTo8(pixs, 0);  // no-op copy when already 8bpp grey

  PIX* pixth = nullptr;
  PIX* pixd = nullptr;
  if (pixSauvolaBinarize(pixg, whsize, factor, 1, nullptr, nullptr, &pixth, &pixd)) {
    fprintf(stderr, "pixSauvolaBinarize failed\n");
    return 1;
  }

  l_int32 w, h;
  pixGetDimensions(pixth, &w, &h, nullptr);
  for (l_int32 i = 0; i < h; i++) {
    for (l_int32 j = 0; j < w; j++) {
      l_uint32 tv = 0, bv = 0;
      pixGetPixel(pixth, j, i, &tv);
      pixGetPixel(pixd, j, i, &bv);
      printf("%d\t%u\t%u\n", i * w + j, tv, bv);
    }
  }

  pixDestroy(&pixs);
  pixDestroy(&pixg);
  pixDestroy(&pixth);
  pixDestroy(&pixd);
  return 0;
}
