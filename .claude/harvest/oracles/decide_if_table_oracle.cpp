// Oracle for the DECISION CORE of pixDecideIfTable (leptonica pageseg.c, v1.82.0)
// — steps 5-9: the horizontal-line / vertical-line / vertical-whitespace count
// + 4-condition table score. The deskew/prepare1bpp FRONT-END (steps 1-4) is
// the separate deskew wave and is factored out: both this oracle and the Rust
// core take the SAME upright 1bpp region (= the C's `pix1` right after
// pixDeskewBoth), so the diff isolates the decision logic.
//
//   Build:  g++ -std=c++17 decide_if_table_oracle.cpp -llept -o /tmp/dit
//   Run:    /tmp/dit > decide_if_table_oracle_out.txt
//
// Two deterministic fixtures (240×280), duplicated byte-for-byte in the Rust
// test: a table grid (4 h-lines + 4 v-lines) and a text paragraph (char
// stripes). The oracle runs the REAL pixMorphSequence / pixSeedfillBinary /
// pixCountConnComp / pixSelectBySize exactly as pixDecideIfTable does.
#include <leptonica/allheaders.h>
#include <cstdio>

static int table_ink(int x, int y) {
    int hrows[4] = {20, 90, 160, 230};
    int vcols[4] = {20, 90, 160, 220};
    for (int i = 0; i < 4; i++)
        if (y >= hrows[i] && y < hrows[i] + 2 && x >= 20 && x < 220) return 1;
    for (int i = 0; i < 4; i++)
        if (x >= vcols[i] && x < vcols[i] + 2 && y >= 20 && y < 232) return 1;
    return 0;
}

static int text_ink(int x, int y) {
    // Paragraph: 5px-tall lines every 14px, 18-on/6-off char cells, full width.
    for (int yb = 20; yb + 5 <= 260; yb += 14)
        if (y >= yb && y < yb + 5 && x >= 20 && x < 220 && ((x - 20) % 24) < 18)
            return 1;
    return 0;
}

static void dump(const char *name, PIX *pix) {
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

// Steps 5-9 of pixDecideIfTable, verbatim, on an upright 1bpp `pix1`.
static void score_fixture(const char *tag, PIX *pix1, int dump_masks) {
    l_int32 nhb, nvb, nvw, score;
    PIX *pix2 = pixMorphSequence(pix1, (char *)"o100.1 + c1.4", 0);
    PIX *pix3 = pixSeedfillBinary(NULL, pix2, pix1, 8);
    PIX *pix4 = pixMorphSequence(pix1, (char *)"o1.100 + c4.1", 0);
    PIX *pix5 = pixSeedfillBinary(NULL, pix4, pix1, 8);
    PIX *pix6 = pixOr(NULL, pix3, pix5);
    pixCountConnComp(pix2, 8, &nhb);
    pixCountConnComp(pix4, 8, &nvb);

    PIX *work = pixCopy(NULL, pix1);
    pixSubtract(work, work, pix6);
    PIX *pix7 = pixMorphSequence(work, (char *)"c4.1 + o8.1", 0);
    pixInvert(pix7, pix7);
    PIX *pix8 = pixMorphSequence(pix7, (char *)"r1 + o1.100", 0);
    PIX *pix9 = pixSelectBySize(pix8, 5, 0, 8, L_SELECT_WIDTH, L_SELECT_IF_GTE, NULL);
    pixCountConnComp(pix9, 8, &nvw);

    score = 0;
    if (nhb > 1) score++;
    if (nvb > 2) score++;
    if (nvw > 3) score++;
    if (nvw > 6) score++;

    printf("%s_nhb_flag %d\n", tag, nhb);
    printf("%s_nvb_flag %d\n", tag, nvb);
    printf("%s_nvw_flag %d\n", tag, nvw);
    printf("%s_score_flag %d\n", tag, score);
    if (dump_masks) {
        char nm[64];
        snprintf(nm, sizeof nm, "%s_hlines", tag); dump(nm, pix2);
        snprintf(nm, sizeof nm, "%s_vlines", tag); dump(nm, pix4);
        snprintf(nm, sizeof nm, "%s_vwhite", tag); dump(nm, pix9);
    }

    pixDestroy(&pix2); pixDestroy(&pix3); pixDestroy(&pix4);
    pixDestroy(&pix5); pixDestroy(&pix6); pixDestroy(&work);
    pixDestroy(&pix7); pixDestroy(&pix8); pixDestroy(&pix9);
}

static PIX *build(int (*ink)(int, int), int w, int h) {
    PIX *p = pixCreate(w, h, 1);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (ink(x, y)) pixSetPixel(p, x, y, 1);
    return p;
}

int main(void) {
    int w = 240, h = 280;
    PIX *tab = build(table_ink, w, h);
    PIX *txt = build(text_ink, w, h);

    dump("tab_src", tab);
    dump("txt_src", txt);
    score_fixture("tab", tab, 1);  // full mask pins on the positive case
    score_fixture("txt", txt, 0);  // scalar pins on the negative case

    pixDestroy(&tab);
    pixDestroy(&txt);
    return 0;
}
