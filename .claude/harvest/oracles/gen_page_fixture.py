#!/usr/bin/env python3
"""Deterministic stacked-page fixture generator for the 3F₂/line-feeding E2E
anchor (`makerow_page_tests::stacked_page_finds_two_deterministic_rows` +
`recognize_page_makerow_dump`).

The line TILE is the A6b synthetic `img_24.pgm` payload (24×36 grey, the
"qLLiy,," line), embedded verbatim below — the original arc generated it
inline and never banked the formula, so the bytes themselves are the durable
source of truth. Both fixtures stack that tile twice:

* `/tmp/page_test.pgm`  — ROOMY layout (32 px top/bottom margin, 64 px gap,
  24 px x-margins; 72×200). The typographic band of
  `Tesseract::LSTMRecognizeWord` (`linerec.cpp:239-246`: ink box extended to
  [baseline+descenders, baseline+xheight+ascenders]) plus `GetRectImage`'s
  `kImagePadding = 4` fits WITHOUT touching the image edges for either row,
  so the two rows' crops are pixel-identical modulo position →
  `recognize_page_makerow` must produce IDENTICAL text for both lines.
  This is the position-invariance falsifier for the line feeding.

* `/tmp/page_tight.pgm` — LEGACY tight layout (4 px margins, 8 px gap; 24×88,
  the original 3F₂ fixture). Here the padded typographic band clips at the
  image top for row A and at the image bottom for row B (faithful
  `GetRectImage` clipping), so band heights differ (57 vs 45 on this data)
  and the two lines recognize differently. Kept as the edge-clip regression
  material; not asserted by tests.

Usage: python3 gen_page_fixture.py
"""

import base64

TILE_W, TILE_H = 24, 36
TILE_B64 = """
ACVKb5S53gMoTXKXvOEGK1B1mr/kCS5TCzFXeZvB7wk7UXepy+EfOUuRt9n7AS9JFjlkg6LF+Bcu
cZy7yu0AX0aplPPSNWhHIUVtmbnV7TFRdY2Z+SUNYUGljdk5FQ0xLFV+l9DxEjN0Xbbv2DkKazzl
jqdAYQIjN1mLqd/pCxl3qZv5z1l7Kdf5i6l/KQv5QmGAo87lBG9auYibtm0cN/LRsHNeNfQfTXWZ
qf0lAWFNpfmpXXUxEe21mYm9xeEBWHWy38w5BmPA7Zq3dFEu+yhFYo+cyTYTY4G/ydMxd1nD4Y+Z
czEXCSNBX4nz0Vd5bpmswyoVcDfG4YRrUs34Dx5JvJOapUAneZXVySEFFTH5pYVpsdXlAXlV1emB
ZVUxhKXW1ygBKtPMvY6f0PkiGxQ1xqeYcbrDj7nD2RcJI8nfqYOp19kjeQ/Zw7l3iaPJmrH4IxYV
LP+ymYCr/s1kFwrhyFNmhfyvpcXxOQUV2cG1dYGZ1UV5MQXlMXmllZmhsMXaLwQ57sNYbYL3rEFW
KwA1al/U6b5zu/EnGQsh/8lrkYfpu0EvGRsRZznLoZ+Jxvk0AxIlyLd+sYybWk0w/zYJJNPCtXiH
0eU9eSnV3VFhtf25aUXd4REFPfnJVV2R3BUOdyDxwlOknYavaLn6K2wFHudQQbLj5xkbaS/Ju3mn
iYuZX9nLKWc5+ylvSdvZ8gEQI97FVG+q+bh7ps3MVyIRABNuNcS//TVpKc3FUaGdhYlprfVBcT31
CQkt5dFBCDViP/y5dqPwjYqX5NFeW9jlEg8M6SZzEyFvycOxZ7nzgX+547FHSfMhbwkDEWdZHllc
w9pVYJeWgZSr4q1Iz84pbDMKBXAnKVVFydFlpZGJhZXpoVVVwclVJekRBQURNEVG97hhuvO8fb7/
oHmy+0R1NucocQoDP3mz+adps4mPiZPpp1nT2V9Z09lnKRMJSnGo46aVnJ+CmZCrbq30t3pB+DN2
JewvVWWhmZWViYGFtfG5ZaXpUVXFwVlV1SlhYKXqr3S5/oOIjZKXnOGma7D1un/EyU5Ta7H3uXuB
j4mbkZdpq+G/eavRV1nbwU9JdrmEg4KFmJeO8bx7qu2gX+apdLPytUjHgYWNmZmVjXGx9a2ZmaVt
oeGlbdnZVa3x
"""


def build(tile: bytes, x_margin: int, top_margin: int, gap: int,
          bottom_margin: int) -> tuple[int, int, bytes]:
    w = TILE_W + 2 * x_margin
    h = top_margin + TILE_H + gap + TILE_H + bottom_margin
    page = bytearray(b"\xff" * (w * h))
    for band_top in (top_margin, top_margin + TILE_H + gap):
        for y in range(TILE_H):
            row = (band_top + y) * w + x_margin
            page[row:row + TILE_W] = tile[y * TILE_W:(y + 1) * TILE_W]
    return w, h, bytes(page)


def write_pgm(path: str, w: int, h: int, pix: bytes) -> None:
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(pix)
    print(f"{path}: {w}x{h}, {len(pix)} px")


def main() -> None:
    tile = base64.b64decode("".join(TILE_B64.split()))
    assert len(tile) == TILE_W * TILE_H, len(tile)
    # Roomy: margins clear the typographic band + kImagePadding by a wide
    # margin (worst-case extension beyond ink ≈ desc 0.6·xh + 4 ≈ 26 px at
    # xh=36; 32 > 26, and 64-px gap > down-ext + up-ext ≈ 41).
    w, h, pix = build(tile, x_margin=24, top_margin=32, gap=64, bottom_margin=32)
    write_pgm("/tmp/page_test.pgm", w, h, pix)
    # Legacy tight layout (the original 3F₂ fixture, 24×88).
    w, h, pix = build(tile, x_margin=0, top_margin=4, gap=8, bottom_margin=4)
    write_pgm("/tmp/page_tight.pgm", w, h, pix)


if __name__ == "__main__":
    main()
