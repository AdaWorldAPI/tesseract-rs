#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <leptonica/allheaders.h>

int main(int argc, char **argv) {
    const char *path = "/tmp/morph_input.bin";
    if (argc > 1) path = argv[1];
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    int32_t w, h, op, hsize, vsize;
    if (fread(&w, sizeof(int32_t), 1, f) != 1) return 1;
    if (fread(&h, sizeof(int32_t), 1, f) != 1) return 1;
    if (fread(&op, sizeof(int32_t), 1, f) != 1) return 1;
    if (fread(&hsize, sizeof(int32_t), 1, f) != 1) return 1;
    if (fread(&vsize, sizeof(int32_t), 1, f) != 1) return 1;
    size_t n = (size_t)w * (size_t)h;
    uint8_t *buf = (uint8_t *)malloc(n);
    if (fread(buf, 1, n, f) != n) { fprintf(stderr, "truncated input\n"); return 1; }
    fclose(f);

    PIX *pixs = pixCreate(w, h, 1);
    for (int32_t y = 0; y < h; y++) {
        for (int32_t x = 0; x < w; x++) {
            uint8_t byte = buf[(size_t)y * w + x];
            if (byte == 0) {
                pixSetPixel(pixs, x, y, 1);
            }
        }
    }

    PIX *pixd = nullptr;
    switch (op) {
        case 0: pixd = pixDilateBrick(NULL, pixs, hsize, vsize); break;
        case 1: pixd = pixErodeBrick(NULL, pixs, hsize, vsize); break;
        case 2: pixd = pixOpenBrick(NULL, pixs, hsize, vsize); break;
        case 3: pixd = pixCloseBrick(NULL, pixs, hsize, vsize); break;
        default:
            fprintf(stderr, "bad op %d\n", op);
            return 1;
    }
    if (!pixd) { fprintf(stderr, "op failed\n"); return 1; }

    for (int32_t y = 0; y < h; y++) {
        printf("m\t%d", y);
        for (int32_t x = 0; x < w; x++) {
            uint32_t val;
            pixGetPixel(pixd, x, y, &val);
            printf("\t%d", val ? 0 : 255);
        }
        printf("\n");
    }

    pixDestroy(&pixs);
    pixDestroy(&pixd);
    free(buf);
    return 0;
}
