// Oracle: pixMorphSequence/-CompSequence (the exact pageseg sequences),
// pixMorphSequenceByComponent, pixSelectBySize, and the composed
// pixGenTextlineMask + pixGenTextblockMask — parity source for the
// tesseract-ocr textline/textblock leaves.
//
// Fixture (must match the Rust tests exactly): 260x220 two-column text page.
//   Columns: A x[15,115), B x[155,245)  (40px gutter > the c30.1 bridge)
//   Lines: y = 20, 32, ..., 188 (step 12), bar height 5
//   Words: within a column, 18px segments with 6px gaps (c30.1 bridges 6px)
//   Speck: 3x3 at (250,10) (o4.1 / selectBySize fodder)
//
// Dump format: "name w h" + rows of '1'/'0'; flags as "name_flag v".
// Build: g++ -std=c++17 pageseg2_oracle.cpp -llept -o pageseg2_oracle
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

static PIX* fixture() {
    PIX* p = pixCreate(260, 220, 1);
    int cols[2][2] = {{15, 115}, {155, 245}};
    for (auto& c : cols)
        for (int yb = 20; yb <= 188; yb += 12)
            for (int y = yb; y < yb + 5; y++)
                for (int x = c[0]; x < c[1]; x++)
                    if (((x - c[0]) % 24) < 18) pixSetPixel(p, x, y, 1);
    for (int y = 10; y < 13; y++)
        for (int x = 250; x < 253; x++) pixSetPixel(p, x, y, 1);
    return p;
}

int main() {
    PIX* src = fixture();
    dump("tl_src", src);

    // Isolated sequence pins (each on the RAW fixture).
    PIX* s1 = pixMorphCompSequence(src, "o80.60", 0);
    dump("seqcomp_o80_60", s1); pixDestroy(&s1);
    PIX* s2 = pixMorphCompSequence(src, "o5.1 + o1.200", 0);
    dump("seqcomp_o5_1_o1_200", s2); pixDestroy(&s2);
    PIX* s3 = pixMorphSequence(src, "c30.1", 0);
    dump("seq_c30_1", s3); pixDestroy(&s3);
    PIX* s4 = pixMorphSequence(src, "c1.10 + o4.1", 0);
    dump("seq_c1_10_o4_1", s4); pixDestroy(&s4);
    PIX* s5 = pixMorphSequenceByComponent(src, "c30.30 + d3.3", 8, 0, 0, NULL);
    dump("bycomp_c30_30_d3_3", s5); pixDestroy(&s5);
    PIX* s6 = pixSelectBySize(src, 25, 5, 8, L_SELECT_IF_BOTH, L_SELECT_IF_GTE, NULL);
    dump("selsize_25_5_both_gte", s6); pixDestroy(&s6);

    // Composed: textline mask (+ vws) then textblock mask.
    PIX* vws = NULL; l_int32 tlfound = 0;
    PIX* tl = pixGenTextlineMask(src, &vws, &tlfound, NULL);
    printf("tl_found_flag %d\n", tlfound);
    dump("tl_vws", vws);
    dump("tl_mask", tl);

    PIX* tb = pixGenTextblockMask(tl, vws, NULL);
    printf("tb_null_flag %d\n", tb == NULL ? 1 : 0);
    if (tb) { dump("tb_mask", tb); pixDestroy(&tb); }

    pixDestroy(&tl); pixDestroy(&vws); pixDestroy(&src);
    return 0;
}
