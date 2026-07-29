// skew_oracle.cpp — byte-parity oracle for the DESKEW WAVE:
// leptonica skew DETECTION (pixFindSkew family) + arbitrary-angle ROTATION
// (pixRotate family), plus their composition (pixDeskew).
//
// Public-API-only, deliberately: every arm calls only exported `pix*` entry
// points, never a `*Low` internal or a struct field. Same style as
// network_forward_oracle.cpp / image_text_oracle_ctc.cpp. Here there is no ABI
// skew to dodge (installed liblept is 1.82.0 and /tmp/leptonica-src is tag
// 1.82.0 — an exact match, verified before this file was written), but
// public-API-only keeps the oracle honest about what it is actually proving:
// the *observable* contract, not a guess at internal layout.
//
// Build:
//   g++ -std=c++17 skew_oracle.cpp -I/usr/include/leptonica -lleptonica -o /tmp/skew_oracle
//
// Arms (argv[1]):
//   findskew  <pgm> <thresh>
//   sweep     <pgm> <thresh> <redsweep> <sweeprange> <sweepdelta>
//   dss       <pgm> <thresh> <angle_deg>
//   rotamgray <pgm> <angle_deg> <grayval>
//   deskew    <pgm> <redsearch>
//
// ── CONVENTIONS (get these wrong and every diff is noise) ────────────────────
//
// * 1bpp polarity. leptonica 1bpp is **1 = ON = black ink**; this crate's
//   binary buffers are **0 = ON**. The `bin` arm below dumps leptonica's
//   polarity verbatim — the Rust side must invert (or compare inverted). The
//   pageseg oracles already established this; do not "fix" it here.
//
// * Binarization is an INPUT, not part of what is being proven. pixFindSkew
//   requires 1bpp, so an explicit `thresh` is taken on the command line and
//   applied with pixThresholdToBinary (ON where value < thresh). Passing the
//   SAME threshold on the Rust side removes binarization as a free variable —
//   otherwise an Otsu difference of one grey level shows up as a skew-angle
//   difference and the diff blames the wrong leaf.
//
// * Floats are dumped as BOTH the raw IEEE-754 bit pattern (hex) and a decimal
//   rendering. The bits are what the parity diff compares; the decimal is for
//   a human reading the failure. Never compare the decimal.
//
// * Angle sign/units. pixFindSkew returns DEGREES via *pangle. Do not assume a
//   sign convention — the `dss` arm exists precisely so the Rust side can pin
//   the convention empirically (score a known angle, compare) instead of
//   reasoning about it.

#include <leptonica/allheaders.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Dump an l_float32 as "<hexbits>\t<decimal>" — bits are the parity subject.
static void print_f32(const char* label, l_float32 v) {
  l_uint32 bits;
  memcpy(&bits, &v, sizeof(bits));
  printf("%s\t0x%08x\t%.9g\n", label, bits, static_cast<double>(v));
}

// Read a PGM (or anything pixRead handles) as 8bpp grey.
static PIX* read_grey(const char* path) {
  PIX* pixs = pixRead(path);
  if (!pixs) {
    fprintf(stderr, "pixRead failed: %s\n", path);
    return nullptr;
  }
  PIX* pixg = pixConvertTo8(pixs, 0);  // no-op copy when already 8bpp
  pixDestroy(&pixs);
  return pixg;
}

// Grey -> 1bpp at an explicit threshold (ON == 1 == value < thresh).
static PIX* to_binary(PIX* pixg, l_int32 thresh) {
  return pixThresholdToBinary(pixg, thresh);
}

// Dump every pixel of a PIX as "<idx>\t<value>", preceded by its dimensions.
// Dimensions are per-arm because leptonica's reduce/expand/rotate steps floor
// independently — two masks in one run need not share w/h (the lesson the
// pageseg_regions oracle banked).
static void dump_pix(const char* tag, PIX* pix) {
  l_int32 w, h, d;
  pixGetDimensions(pix, &w, &h, &d);
  printf("%s_w\t%d\n%s_h\t%d\n%s_d\t%d\n", tag, w, tag, h, tag, d);
  for (l_int32 y = 0; y < h; y++) {
    for (l_int32 x = 0; x < w; x++) {
      l_uint32 v = 0;
      pixGetPixel(pix, x, y, &v);
      printf("%s\t%d\t%u\n", tag, y * w + x, v);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage:\n"
            "  %s findskew  <pgm> <thresh>\n"
            "  %s sweep     <pgm> <thresh> <redsweep> <sweeprange> <sweepdelta>\n"
            "  %s dss       <pgm> <thresh> <angle_deg>\n"
            "  %s rotamgray <pgm> <angle_deg> <grayval>\n"
            "  %s deskew    <pgm> <redsearch>\n",
            argv[0], argv[0], argv[0], argv[0], argv[0]);
    return 2;
  }
  const char* arm = argv[1];
  const char* path = argv[2];

  // ── ARM 1: pixFindSkew — the all-defaults detector entry point. ───────────
  // Proves the Rust find_skew's angle AND confidence against the default
  // sweep+search parameters, whatever they are internally.
  if (!strcmp(arm, "findskew")) {
    if (argc < 4) { fprintf(stderr, "findskew needs <thresh>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixb = to_binary(pixg, atoi(argv[3]));
    l_float32 angle = 0.0f, conf = 0.0f;
    l_int32 rc = pixFindSkew(pixb, &angle, &conf);
    printf("rc\t%d\n", rc);
    print_f32("angle", angle);
    print_f32("conf", conf);
    pixDestroy(&pixg);
    pixDestroy(&pixb);
    return 0;
  }

  // ── ARM 2: pixFindSkewSweep — the sweep half alone, explicit params. ──────
  // Isolates the coarse sweep from the refinement search, so a Rust
  // discrepancy can be localized to one of the two halves instead of only
  // being visible in the composed answer.
  if (!strcmp(arm, "sweep")) {
    if (argc < 7) { fprintf(stderr, "sweep needs <thresh> <redsweep> <range> <delta>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixb = to_binary(pixg, atoi(argv[3]));
    l_float32 angle = 0.0f;
    l_int32 rc = pixFindSkewSweep(pixb, &angle, atoi(argv[4]),
                                  static_cast<l_float32>(atof(argv[5])),
                                  static_cast<l_float32>(atof(argv[6])));
    printf("rc\t%d\n", rc);
    print_f32("angle", angle);
    pixDestroy(&pixg);
    pixDestroy(&pixb);
    return 0;
  }

  // ── ARM 3: the SCORING LEAF at a caller-chosen angle. ─────────────────────
  // pixFindDifferentialSquareSum over a page rotated to `angle_deg` is the
  // objective the sweep maximizes. Scoring one KNOWN angle is the cheapest
  // falsifier of the whole detector: if the score function matches pointwise,
  // any remaining angle discrepancy is in the search strategy, not the metric.
  // It also pins the angle SIGN convention empirically.
  if (!strcmp(arm, "dss")) {
    if (argc < 5) { fprintf(stderr, "dss needs <thresh> <angle_deg>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixb = to_binary(pixg, atoi(argv[3]));
    l_float32 deg = static_cast<l_float32>(atof(argv[4]));
    l_float32 rad = deg * 3.14159265358979323846f / 180.0f;
    // Shear-rotate about the center, matching what the sweep does internally
    // to score a candidate angle on 1bpp.
    PIX* pixr = pixRotateShearCenter(pixb, rad, L_BRING_IN_WHITE);
    l_float32 sum = 0.0f;
    l_int32 rc = pixFindDifferentialSquareSum(pixr, &sum);
    printf("rc\t%d\n", rc);
    print_f32("deg", deg);
    print_f32("sum", sum);
    pixDestroy(&pixg);
    pixDestroy(&pixb);
    pixDestroy(&pixr);
    return 0;
  }

  // ── ARM 4: pixRotateAMGray — the GREY area-map rotation, pixel-exact. ─────
  // This is the rotation the recognizer actually wants: the page is deskewed
  // in GREY (rotating the binary would throw away the antialiasing the LSTM
  // input step depends on). Center pivot, `grayval` fills the corners.
  if (!strcmp(arm, "rotamgray")) {
    if (argc < 5) { fprintf(stderr, "rotamgray needs <angle_deg> <grayval>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    l_float32 deg = static_cast<l_float32>(atof(argv[3]));
    l_float32 rad = deg * 3.14159265358979323846f / 180.0f;
    PIX* pixr = pixRotateAMGray(pixg, rad, static_cast<l_uint8>(atoi(argv[4])));
    if (!pixr) { fprintf(stderr, "pixRotateAMGray failed\n"); return 1; }
    print_f32("deg", deg);
    dump_pix("rot", pixr);
    pixDestroy(&pixg);
    pixDestroy(&pixr);
    return 0;
  }

  // ── ARM 5: pixDeskew — the composition (detect then rotate). ──────────────
  // The end-to-end gate. Runs on the GREY page: pixDeskew binarizes
  // internally for detection and rotates the input at its own depth, which is
  // exactly the composition the Rust `deskew_grey` must reproduce.
  if (!strcmp(arm, "deskew")) {
    if (argc < 4) { fprintf(stderr, "deskew needs <redsearch>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixd = pixDeskew(pixg, atoi(argv[3]));
    if (!pixd) { fprintf(stderr, "pixDeskew failed (returns NULL on failure)\n"); return 1; }
    dump_pix("deskew", pixd);
    pixDestroy(&pixg);
    pixDestroy(&pixd);
    return 0;
  }

  fprintf(stderr, "unknown arm: %s\n", arm);
  return 2;
}
