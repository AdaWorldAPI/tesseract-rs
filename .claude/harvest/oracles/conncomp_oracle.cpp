// Batch 3B oracle (pixConnCompBB byte-parity) + Batch 3F2 leaf-1 extension
// (--areas mode: per-component ink pixel count via pixConnCompPixa +
// pixCountPixels). Pure leptonica, no ABI-skew concern (unlike the
// tesseract-linked oracles): the installed libleptonica matches what's used
// here, so this links against -llept directly (no source-compile route
// needed).
//
// Build:
//   g++ -std=c++17 /tmp/conncomp_oracle.cpp -o /tmp/conncomp_oracle \
//     $(pkg-config --cflags --libs lept)
//
// Usage:
//   /tmp/conncomp_oracle [--areas] [fixture.bin]
//   (default fixture path: /tmp/conncomp_input.bin)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <leptonica/allheaders.h>

int main(int argc, char **argv) {
    const char *path = "/tmp/conncomp_input.bin";
    bool areas_mode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--areas") == 0) {
            areas_mode = true;
        } else {
            path = argv[i];
        }
    }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    int32_t w, h, connectivity;
    if (fread(&w, sizeof(int32_t), 1, f) != 1) return 1;
    if (fread(&h, sizeof(int32_t), 1, f) != 1) return 1;
    if (fread(&connectivity, sizeof(int32_t), 1, f) != 1) return 1;
    size_t n = (size_t)w * (size_t)h;
    uint8_t *buf = (uint8_t *)malloc(n);
    if (fread(buf, 1, n, f) != n) { fprintf(stderr, "truncated input\n"); return 1; }
    fclose(f);

    PIX *pix = pixCreate(w, h, 1);
    for (int32_t y = 0; y < h; y++) {
        for (int32_t x = 0; x < w; x++) {
            uint8_t byte = buf[(size_t)y * w + x];
            if (byte == 0) {
                pixSetPixel(pix, x, y, 1);
            }
        }
    }

    if (!areas_mode) {
        BOXA *boxa = pixConnCompBB(pix, connectivity);
        int32_t n_boxes = boxa ? boxaGetCount(boxa) : 0;
        printf("n\t%d\n", n_boxes);
        for (int32_t i = 0; i < n_boxes; i++) {
            int32_t bx, by, bw, bh;
            boxaGetBoxGeometry(boxa, i, &bx, &by, &bw, &bh);
            printf("bb\t%d\t%d\t%d\t%d\t%d\n", i, bx, by, bw, bh);
        }
        boxaDestroy(&boxa);
    } else {
        // pixConnCompPixa (conncomp.c) is a slight variation on
        // pixConnCompBB that additionally saves each component's own
        // sub-Pix (1 == this component's ink, elsewhere 0, within the
        // bbox); pixCountPixels on that sub-Pix gives the exact ink pixel
        // count -- the BLOBNBOX::enclosed_area() analogue.
        PIXA *pixa = nullptr;
        BOXA *boxa = pixConnCompPixa(pix, &pixa, connectivity);
        int32_t n_boxes = boxa ? boxaGetCount(boxa) : 0;
        printf("n\t%d\n", n_boxes);
        for (int32_t i = 0; i < n_boxes; i++) {
            int32_t bx, by, bw, bh;
            boxaGetBoxGeometry(boxa, i, &bx, &by, &bw, &bh);
            PIX *comp = pixaGetPix(pixa, i, L_CLONE);
            int32_t count = 0;
            pixCountPixels(comp, &count, nullptr);
            pixDestroy(&comp);
            printf("cc\t%d\t%d\t%d\t%d\t%d\t%d\n", i, bx, by, bw, bh, count);
        }
        pixaDestroy(&pixa);
        boxaDestroy(&boxa);
    }

    pixDestroy(&pix);
    free(buf);
    return 0;
}
