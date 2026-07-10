// Oracle: pixReduceRankBinary2 (levels 1-4), pixReduceRankBinaryCascade, and
// pixExpandBinaryPower2 (factors 2/4/8/16) on deterministic fixtures — the
// parity source for tesseract-ocr/src/binreduce.rs.
//
// Fixtures (must match the Rust tests exactly):
//   reduce:  w=97, h=61 (both odd), ON iff (x*7 + y*13) % 251 < 128
//   expand:  w=9,  h=5,             ON iff (x*3 + y*5)  %  17 < 8
//
// Dump format per result: "name w h" then h rows of '1'/'0'.
// Build: g++ -std=c++17 binreduce_oracle.cpp -llept -o binreduce_oracle
#include <leptonica/allheaders.h>
#include <cstdio>

static PIX* make_fixture(int w, int h, int mx, int my, int mod, int lim) {
    PIX* p = pixCreate(w, h, 1);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if ((x * mx + y * my) % mod < lim) pixSetPixel(p, x, y, 1);
    return p;
}

static void dump(const char* name, PIX* p) {
    l_int32 w = pixGetWidth(p), h = pixGetHeight(p);
    printf("%s %d %d\n", name, w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            l_uint32 v;
            pixGetPixel(p, x, y, &v);
            putchar(v ? '1' : '0');
        }
        putchar('\n');
    }
}

int main() {
    PIX* src = make_fixture(97, 61, 7, 13, 251, 128);
    dump("src", src);

    for (int level = 1; level <= 4; level++) {
        PIX* r = pixReduceRankBinary2(src, level, nullptr);
        char name[32];
        snprintf(name, sizeof name, "reduce_l%d", level);
        dump(name, r);
        pixDestroy(&r);
    }

    PIX* c1 = pixReduceRankBinaryCascade(src, 1, 2, 0, 0);
    dump("cascade_1_2", c1);
    pixDestroy(&c1);
    PIX* c2 = pixReduceRankBinaryCascade(src, 4, 4, 3, 0);
    dump("cascade_4_4_3", c2);
    pixDestroy(&c2);

    PIX* esrc = make_fixture(9, 5, 3, 5, 17, 8);
    dump("esrc", esrc);
    for (int f = 2; f <= 16; f *= 2) {
        PIX* e = pixExpandBinaryPower2(esrc, f);
        char name[32];
        snprintf(name, sizeof name, "expand_f%d", f);
        dump(name, e);
        pixDestroy(&e);
    }

    pixDestroy(&esrc);
    pixDestroy(&src);
    return 0;
}
