// Oracle: pixCloseSafeBrick, pixSeedfillBinary (4/8-conn + size-mismatch),
// pixExpandReplicate, and the composed pixGenerateHalftoneMask — the parity
// source for tesseract-ocr {morph::close_safe_brick, seedfill.rs, binreduce::
// expand_replicate, pageseg.rs}.
//
// Fixtures (must match the Rust tests exactly):
//   rf   97x61:  ON iff (7x+13y) % 251 < 128            (close_safe input)
//   mask 61x47:  ON iff ((x/9)+(y/7)) % 2 == 0          (9x7 tile checker —
//                tiles touch DIAGONALLY, so conn-4 stays in one tile while
//                conn-8 floods across; a live connectivity discriminator)
//   seed dots:   (4,3), (40,30) on ON tiles; (20,10) on an OFF tile (dead)
//   ht  130x117: halftone rect x[8,70) y[10,60): ON iff (31x+17y)%7 < 5;
//                text rows y {70,78,86,94}+3px, x[75,122), ON iff x%5 != 0
//                (130/117 are NOT multiples of 4 — pins the size-mismatch
//                semantics of the cascade->expand->seedfill->subtract chain)
//
// Dump format: "name w h" + rows of '1'/'0'; flags as "name_flag v".
// Build: g++ -std=c++17 pageseg_oracle.cpp -llept -o pageseg_oracle
#include <leptonica/allheaders.h>
#include <cstdio>
#include <initializer_list>

static void dump(const char* name, PIX* p) {
    l_int32 w = pixGetWidth(p), h = pixGetHeight(p);
    printf("%s %d %d\n", name, w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            l_uint32 v; pixGetPixel(p, x, y, &v); putchar(v ? '1' : '0');
        }
        putchar('\n');
    }
}

int main() {
    // rf fixture (same as binreduce oracle's src)
    PIX* rf = pixCreate(97, 61, 1);
    for (int y = 0; y < 61; y++)
        for (int x = 0; x < 97; x++)
            if ((x * 7 + y * 13) % 251 < 128) pixSetPixel(rf, x, y, 1);

    int cs[3][2] = {{4, 4}, {1, 7}, {6, 1}};
    for (auto& s : cs) {
        PIX* r = pixCloseSafeBrick(NULL, rf, s[0], s[1]);
        char name[32]; snprintf(name, sizeof name, "closesafe_%d_%d", s[0], s[1]);
        dump(name, r); pixDestroy(&r);
    }

    // seedfill fixtures
    PIX* mask = pixCreate(61, 47, 1);
    for (int y = 0; y < 47; y++)
        for (int x = 0; x < 61; x++)
            if (((x / 9) + (y / 7)) % 2 == 0) pixSetPixel(mask, x, y, 1);
    PIX* seed = pixCreate(61, 47, 1);
    pixSetPixel(seed, 4, 3, 1); pixSetPixel(seed, 40, 30, 1); pixSetPixel(seed, 20, 10, 1);
    dump("sf_mask", mask); dump("sf_seed", seed);

    PIX* f4 = pixSeedfillBinary(NULL, seed, mask, 4);
    dump("seedfill_c4", f4); pixDestroy(&f4);
    PIX* f8 = pixSeedfillBinary(NULL, seed, mask, 8);
    dump("seedfill_c8", f8); pixDestroy(&f8);

    PIX* seedsm = pixCreate(56, 44, 1);
    pixSetPixel(seedsm, 4, 3, 1); pixSetPixel(seedsm, 40, 30, 1); pixSetPixel(seedsm, 20, 10, 1);
    PIX* fm = pixSeedfillBinary(NULL, seedsm, mask, 4);
    dump("seedfill_mismatch", fm);
    pixDestroy(&fm); pixDestroy(&seedsm); pixDestroy(&seed); pixDestroy(&mask);

    // expand_replicate (the actual pageseg callee, expand.c) on the 9x5 esrc
    PIX* esrc = pixCreate(9, 5, 1);
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 9; x++)
            if ((x * 3 + y * 5) % 17 < 8) pixSetPixel(esrc, x, y, 1);
    for (int f : {3, 4}) {
        PIX* e = pixExpandReplicate(esrc, f);
        char name[32]; snprintf(name, sizeof name, "exprep_f%d", f);
        dump(name, e); pixDestroy(&e);
    }
    pixDestroy(&esrc);

    // composed halftone-mask fixture
    PIX* ht = pixCreate(130, 117, 1);
    for (int y = 10; y < 60; y++)
        for (int x = 8; x < 70; x++)
            if ((31 * x + 17 * y) % 7 < 5) pixSetPixel(ht, x, y, 1);
    for (int yb : {70, 78, 86, 94})
        for (int y = yb; y < yb + 3; y++)
            for (int x = 75; x < 122; x++)
                if (x % 5 != 0) pixSetPixel(ht, x, y, 1);
    dump("ht_src", ht);

    PIX* pixtext = NULL; l_int32 htfound = 0;
    PIX* m = pixGenerateHalftoneMask(ht, &pixtext, &htfound, NULL);
    printf("ht_found_flag %d\n", htfound);
    dump("ht_mask", m);
    dump("ht_text", pixtext);
    pixDestroy(&m); pixDestroy(&pixtext); pixDestroy(&ht);

    // second composed fixture: DENSE halftone block (solid) so the seed
    // survives rank4+rank4+open5x5 -> the found=1 arm is exercised too
    PIX* ht2 = pixCreate(130, 117, 1);
    for (int y = 10; y < 60; y++)
        for (int x = 8; x < 70; x++)
            pixSetPixel(ht2, x, y, 1);  /* SOLID block: rank4^2+open5x5 needs a hole-free 20x20 core, which real 150ppi halftones have (clustered dots merge) */
    for (int yb : {70, 78, 86, 94})
        for (int y = yb; y < yb + 3; y++)
            for (int x = 75; x < 122; x++)
                if (x % 5 != 0) pixSetPixel(ht2, x, y, 1);
    dump("ht2_src", ht2);
    PIX* pixtext2 = NULL; l_int32 htfound2 = 0;
    PIX* m2 = pixGenerateHalftoneMask(ht2, &pixtext2, &htfound2, NULL);
    printf("ht2_found_flag %d\n", htfound2);
    dump("ht2_mask", m2);
    dump("ht2_text", pixtext2);
    pixDestroy(&m2); pixDestroy(&pixtext2); pixDestroy(&ht2);

    pixDestroy(&rf);
    return 0;
}
