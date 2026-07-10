// Oracle: pixCountPixelsByRow / pixCountPixelsByColumn on a deterministic
// synthetic image — pins the profile-count convention for the Rust xy_cut
// projection profiles (our 8bpp binary foreground=0 ↔ leptonica 1bpp ON=1).
//
// Fixture formula (must match the Rust test exactly):
//   w=97, h=61; grey(x,y) = (x*7 + y*13) % 251; ink iff grey < 128.
// Threshold done via pixThresholdToBinary(pixs, 128): pixel < 128 -> ON(1).
//
// Build: g++ -std=c++17 counts_oracle.cpp -llept -o counts_oracle
#include <leptonica/allheaders.h>
#include <cstdio>

int main() {
    const int w = 97, h = 61;
    PIX* pixs = pixCreate(w, h, 8);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            pixSetPixel(pixs, x, y, (l_uint32)((x * 7 + y * 13) % 251));
    PIX* pixb = pixThresholdToBinary(pixs, 128); // <128 -> 1 (ON/black)

    NUMA* rows = pixCountPixelsByRow(pixb, nullptr);
    NUMA* cols = pixCountPixelsByColumn(pixb);
    printf("w %d h %d\n", w, h);
    printf("rows");
    for (int i = 0; i < numaGetCount(rows); i++) {
        l_int32 v; numaGetIValue(rows, i, &v); printf(" %d", v);
    }
    printf("\ncols");
    for (int i = 0; i < numaGetCount(cols); i++) {
        l_float32 f; numaGetFValue(cols, i, &f); printf(" %d", (int)f);
    }
    printf("\n");
    return 0;
}
