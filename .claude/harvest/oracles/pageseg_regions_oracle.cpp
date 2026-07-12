// Oracle for pixGetRegionsBinary (leptonica pageseg.c:113, v1.82.0) — the
// region-classifier composition. Builds a deterministic 320×280 1bpp fixture
// (a solid image block + two columns of horizontal text stripes), runs the
// REAL pixGetRegionsBinary, and dumps pixs + the three region masks
// (halftone / textline / textblock) as "name w h" + rows of '1'/'0'.
//
//   Build:  g++ -std=c++17 pageseg_regions_oracle.cpp -llept -o /tmp/preg
//   Run:    /tmp/preg > pageseg_regions_oracle_out.txt
//
// The fixture formula is duplicated byte-for-byte in the Rust test
// (crates/tesseract-ocr/src/pageseg.rs::regions_fixture) which asserts
// regions_src == this dump before diffing the masks. liblept 1.82.0 links
// directly (no ABI skew), so the diff is ground truth.
#include <leptonica/allheaders.h>
#include <cstdio>

// Ink predicate — IDENTICAL to the Rust regions_fixture().
static int ink_at(int x, int y) {
    // Solid image block: survives rank-4 cascade + 5×5 open → halftone region.
    if (x >= 30 && x < 130 && y >= 30 && y < 110) return 1;
    // Two text columns of horizontal char-stripes (5px lines every 12px,
    // 18-on/6-off char cells) → textline + textblock, never halftone.
    int cols[2] = {160, 250};
    for (int ci = 0; ci < 2; ci++) {
        int c0 = cols[ci];
        if (x >= c0 && x < c0 + 60) {
            for (int yb = 20; yb + 5 <= 260; yb += 12) {
                if (y >= yb && y < yb + 5 && ((x - c0) % 24) < 18) return 1;
            }
        }
    }
    return 0;
}

static void dump(const char *name, PIX *pix) {
    if (!pix) { printf("%s 0 0\n", name); return; }
    l_int32 w, h;
    pixGetDimensions(pix, &w, &h, NULL);
    printf("%s %d %d\n", name, w, h);
    for (l_int32 y = 0; y < h; y++) {
        for (l_int32 x = 0; x < w; x++) {
            l_uint32 v;
            pixGetPixel(pix, x, y, &v);
            putchar(v ? '1' : '0');
        }
        putchar('\n');
    }
}

int main(void) {
    l_int32 w = 320, h = 280;
    PIX *pixs = pixCreate(w, h, 1);
    for (l_int32 y = 0; y < h; y++)
        for (l_int32 x = 0; x < w; x++)
            if (ink_at(x, y)) pixSetPixel(pixs, x, y, 1);

    PIX *pixhm = NULL, *pixtm = NULL, *pixtb = NULL;
    pixGetRegionsBinary(pixs, &pixhm, &pixtm, &pixtb, NULL);

    dump("regions_src", pixs);
    dump("regions_hm", pixhm);
    dump("regions_tm", pixtm);
    dump("regions_tb", pixtb);

    pixDestroy(&pixs);
    pixDestroy(&pixhm);
    pixDestroy(&pixtm);
    pixDestroy(&pixtb);
    return 0;
}
