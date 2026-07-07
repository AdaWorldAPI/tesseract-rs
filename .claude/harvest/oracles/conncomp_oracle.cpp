#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <leptonica/allheaders.h>

int main(int argc, char **argv) {
    const char *path = "/tmp/conncomp_input.bin";
    if (argc > 1) path = argv[1];
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

    BOXA *boxa = pixConnCompBB(pix, connectivity);
    int32_t n_boxes = boxa ? boxaGetCount(boxa) : 0;
    printf("n\t%d\n", n_boxes);
    for (int32_t i = 0; i < n_boxes; i++) {
        int32_t bx, by, bw, bh;
        boxaGetBoxGeometry(boxa, i, &bx, &by, &bw, &bh);
        printf("bb\t%d\t%d\t%d\t%d\t%d\n", i, bx, by, bw, bh);
    }

    boxaDestroy(&boxa);
    pixDestroy(&pix);
    free(buf);
    return 0;
}
