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
//   sweep     <pgm> <thresh> <redsweep> <redsearch> <sweepcenter> <sweeprange>
//             <sweepdelta> <minbsdelta> <pivot 1=corner|2=center>
//   dss       <pgm> <thresh> <angle_deg> <pivot 1=corner|2=center>
//   rot90     <pgm> <direction 1=cw|-1=ccw>
//   rotamgray <pgm> <angle_deg> <grayval>
//   rotamgraycorner <pgm> <angle_deg> <grayval>
//   deskew    <pgm> <redsearch>
//   deskewboth <pgm> <redsearch>
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
            "  %s findskew        <pgm> <thresh>\n"
            "  %s sweep           <pgm> <thresh> <redsweep> <redsearch> <sweepcenter>"
            " <sweeprange> <sweepdelta> <minbsdelta> <pivot 1=corner|2=center>\n"
            "  %s dss             <pgm> <thresh> <angle_deg> <pivot 1=corner|2=center>\n"
            "  %s rot90           <pgm> <direction 1=cw|-1=ccw>\n"
            "  %s rotamgray       <pgm> <angle_deg> <grayval>\n"
            "  %s rotamgraycorner <pgm> <angle_deg> <grayval>\n"
            "  %s deskew          <pgm> <redsearch>\n"
            "  %s deskewboth      <pgm> <redsearch>\n",
            argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
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

  // ── ARM 2: the REAL sweep-and-search, with explicit parameters. ──────────
  //
  // ⚠ CORRECTED (codex P1 on PR #58). This arm previously called
  // `pixFindSkewSweep` — the standalone sweep-only API. That was WRONG as a
  // D4 oracle for two compounding reasons:
  //   (a) it is NOT on `pixFindSkew`'s call path at all (the manifest's own
  //       STEP 3 classifies it SKIP for exactly that reason), and
  //   (b) it refines its coarse maximum with `numaFitMax`, whereas the
  //       targeted `pixFindSkewSweepAndSearchScorePivot` takes the RAW coarse
  //       maximum and then does a binary-search refinement.
  // So its output could neither isolate nor validate the algorithm D4 is
  // transcoding: a real sweep/search discrepancy would be invisible here, or
  // worse, misdiagnosed as a scoring bug. Always exercise the entry point the
  // production path actually reaches.
  //
  // `pendscore` is dumped too — it is the raw objective at the returned
  // angle, which lets a failing angle diff be split into "the score function
  // differs" vs "the search walked differently over the same scores".
  if (!strcmp(arm, "sweep")) {
    if (argc < 11) {
      fprintf(stderr,
              "sweep needs <thresh> <redsweep> <redsearch> <sweepcenter> "
              "<sweeprange> <sweepdelta> <minbsdelta> <pivot 1=corner|2=center>\n");
      return 2;
    }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixb = to_binary(pixg, atoi(argv[3]));
    l_float32 angle = 0.0f, conf = 0.0f, endscore = 0.0f;
    l_int32 rc = pixFindSkewSweepAndSearchScorePivot(
        pixb, &angle, &conf, &endscore,
        atoi(argv[4]),                                   // redsweep
        atoi(argv[5]),                                   // redsearch
        static_cast<l_float32>(atof(argv[6])),           // sweepcenter
        static_cast<l_float32>(atof(argv[7])),           // sweeprange
        static_cast<l_float32>(atof(argv[8])),           // sweepdelta
        static_cast<l_float32>(atof(argv[9])),           // minbsdelta
        atoi(argv[10]));                                 // pivot
    printf("rc\t%d\n", rc);
    print_f32("angle", angle);
    print_f32("conf", conf);
    print_f32("endscore", endscore);
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
  //
  // ⚠ CORRECTED (codex P1 on PR #58). This arm previously prepared `pixr`
  // with `pixRotateShearCenter` — a COMPOSED two/three-shear *rotation*. The
  // sweep does not do that: it applies a SINGLE vertical shear
  // (`pixVShearCorner` / `pixVShearCenter`, selected by its `pivot`
  // argument). Those produce different pixels, hence different scores, so the
  // old arm was actively harmful as a D3 gate — a CORRECT single-vertical-
  // shear transcode would have failed it, while an implementation of the
  // wrong operation could have passed. The rule this cost: an oracle must
  // reproduce the operation under test, not merely something in its
  // neighbourhood that "also rotates".
  if (!strcmp(arm, "dss")) {
    if (argc < 6) {
      fprintf(stderr, "dss needs <thresh> <angle_deg> <pivot 1=corner|2=center>\n");
      return 2;
    }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixb = to_binary(pixg, atoi(argv[3]));
    l_float32 deg = static_cast<l_float32>(atof(argv[4]));
    l_int32 pivot = atoi(argv[5]);
    // skew.c's own degree->radian constant: an f64 division narrowed to f32
    // on assignment (audit §1). Reproduced exactly rather than via a shared
    // PI constant, which would be MORE precise than the C literal and can
    // differ in the last bit.
    l_float32 deg2rad = static_cast<l_float32>(3.1415926535 / 180.);
    l_float32 rad = deg * deg2rad;
    // The SINGLE vertical shear the sweep actually scores, pivot-selected.
    PIX* pixr = (pivot == L_SHEAR_ABOUT_CORNER)
                    ? pixVShearCorner(nullptr, pixb, rad, L_BRING_IN_WHITE)
                    : pixVShearCenter(nullptr, pixb, rad, L_BRING_IN_WHITE);
    if (!pixr) { fprintf(stderr, "vshear failed\n"); return 1; }
    l_float32 sum = 0.0f;
    l_int32 rc = pixFindDifferentialSquareSum(pixr, &sum);
    // NOTE: the pivot is deliberately NOT printed. The Rust `dss` arm has no
    // pivot concept (it scores WITHOUT shearing — D3 is not built yet), so
    // the two sides are only diff-compatible at angle 0, where the shear is a
    // verified no-op (both pivots return the identical sum). Printing the
    // pivot would add a line the Rust side cannot produce and break that
    // diff for no information gain — the harness already records which pivot
    // it passed, in the test label.
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

  // ── ARM 4b: pixRotateAMGrayCorner — the CORNER-pivot variant. ────────────
  // Same f64-then-narrow-once precision trap as ARM 4 (audit §3) but a
  // different per-pixel formula (no xcen/ycen offset), so it needs its own
  // parity coverage — a center-pivot-only sweep would leave the corner
  // kernel entirely unexercised.
  if (!strcmp(arm, "rotamgraycorner")) {
    if (argc < 5) { fprintf(stderr, "rotamgraycorner needs <angle_deg> <grayval>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    l_float32 deg = static_cast<l_float32>(atof(argv[3]));
    l_float32 rad = deg * 3.14159265358979323846f / 180.0f;
    PIX* pixr = pixRotateAMGrayCorner(pixg, rad, static_cast<l_uint8>(atoi(argv[4])));
    if (!pixr) { fprintf(stderr, "pixRotateAMGrayCorner failed\n"); return 1; }
    print_f32("deg", deg);
    dump_pix("rot", pixr);
    pixDestroy(&pixg);
    pixDestroy(&pixr);
    return 0;
  }

  // ── ARM 1b: pixRotate90 — lossless orthogonal rotation (D1). ─────────────
  // A pure index remap: no interpolation, no float math at all, exact for
  // any depth. Needed by pixDeskewBoth's 90-degree round trip (D7). Cheapest
  // leaf in the wave and the one that proves the harness itself works before
  // any precision-sensitive kernel is diffed.
  if (!strcmp(arm, "rot90")) {
    if (argc < 4) { fprintf(stderr, "rot90 needs <direction 1=cw|-1=ccw>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixr = pixRotate90(pixg, atoi(argv[3]));
    if (!pixr) { fprintf(stderr, "pixRotate90 failed\n"); return 1; }
    dump_pix("rot90", pixr);
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

  // ── ARM 6: pixDeskewBoth — the two-pass orthogonal composition (D7). ─────
  // pixDeskew -> pixRotate90(+1) -> pixDeskew -> pixRotate90(-1). The 90°
  // round trip exists because the differential-square-sum detector is
  // DIRECTIONAL: one pass cannot see skew present in both axes at once.
  //
  // Added at the orchestrator level rather than by the transcode agent: the
  // agent correctly refused to invent a dump format with no oracle to diff
  // against (the `vshear` arm's own documented lesson). The right fix is to
  // supply the missing oracle side, not to lower the evidence bar — D7 gets
  // real byte-parity instead of unit tests alone.
  if (!strcmp(arm, "deskewboth")) {
    if (argc < 4) { fprintf(stderr, "deskewboth needs <redsearch>\n"); return 2; }
    PIX* pixg = read_grey(path);
    if (!pixg) return 1;
    PIX* pixd = pixDeskewBoth(pixg, atoi(argv[3]));
    if (!pixd) { fprintf(stderr, "pixDeskewBoth failed (returns NULL on failure)\n"); return 1; }
    dump_pix("deskewboth", pixd);
    pixDestroy(&pixg);
    pixDestroy(&pixd);
    return 0;
  }

  fprintf(stderr, "unknown arm: %s\n", arm);
  return 2;
}
