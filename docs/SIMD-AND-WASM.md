# SIMD integration & the WASM question

> Do we need to wire ndarray's SIMD polyfill into the recognizer, and is the
> WASM arm useful for the web demo? Short answer: **the integration already
> exists**; the interesting frontier is a browser-side (wasm) build, which
> works today but is **not** yet SIMD-accelerated on the int8 hot path.

## 1. Verdict — the SIMD integration is already done

The recognizer's entire int8 hot path (every LSTM gate and fully-connected
layer is an `int8 × int8 → i32` matrix-vector product) is a **single call** to
ndarray's SIMD polyfill:

```rust
// crates/tesseract-recognizer/src/lib.rs:146
ndarray::simd::matmul_i8_to_i32(w, u_col, out.view_mut())
```

This is the `simd-savant` invariant ("all SIMD comes from `ndarray::simd`")
already satisfied — the recognizer never writes an intrinsic itself, it consumes
the one polyfilled kernel. **There is nothing to integrate.** Any speed we get
(or don't) is decided inside ndarray, transparently to tesseract-rs.

## 2. How the polyfill dispatches

`ndarray/src/simd.rs` is the dispatch entry over per-arch backends
(`simd_{amx,avx512,avx2,neon,neon_baseline,neon_bf16,neon_dotprod,scalar,wasm}.rs`),
with `simd_caps.rs` as a `LazyLock<SimdCaps>` — detect the CPU **once**, dispatch
forever.

`matmul_i8_to_i32` runtime-dispatches:

```
AMX TDPBUSD  →  AVX-512 VNNI (VPDPBUSD-zmm)  →  AVX2 VNNI (VPDPBUSD-ymm)  →  scalar
```

| Target | Path the int8 GEMM takes |
|---|---|
| x86-64 Sapphire Rapids+ (AMX) | AMX tile `TDPBUSD` |
| x86-64 with AVX-512 VNNI | `VPDPBUSD` on zmm |
| x86-64 with AVX2 VNNI | `VPDPBUSD` on ymm |
| x86-64 baseline / aarch64 non-dotprod | scalar (or NEON dotprod where present) |
| **wasm32** | **scalar** (see §4) |

x86/aarch64 use **runtime** detection (`is_x86_feature_detected!` etc.); the
result is cached in the `LazyLock`.

## 3. The server demo (Railway, x86-64) — already fast, nothing to do

The web demo binary runs on Railway's x86-64. The int8 GEMM is **already**
runtime-dispatched to the best available VNNI/AMX path — no build flag or code
change is required, because detection is at runtime. A host with AVX-512 VNNI
gets `VPDPBUSD-zmm` automatically; a baseline host falls to scalar but still
runs. **Recommendation: leave it — do not pin an exotic `target-cpu`** that
would trade portability for a path the runtime dispatch already selects.
(If you ever want to *force* a floor, `-C target-cpu=x86-64-v3` enables AVX2 as
the compiled baseline while runtime detection still lifts to AVX-512/AMX.)

## 4. The WASM arm — real, but the int8 GEMM isn't wired to it (yet)

ndarray **does** have a WASM backend — `simd_wasm.rs` plus full `wasm32` lane
constants in `simd.rs` (`f32x4`, `f64x2`, `i16x8`, … the v128/SIMD128 shapes),
selected at **compile time** (`target_feature = "simd128"`; WASM has no runtime
feature detection, so it's a build-time gate, not the `simd_caps` runtime tier).

**But `matmul_i8_to_i32` has no wasm-v128 arm.** Its dispatch ladder is
AMX → AVX-512 VNNI → AVX2 VNNI → **scalar**; on `wasm32` none of the first three
apply, so the recognizer's hot path runs **scalar** in the browser. The v128
backend accelerates the *generic* lane primitives, not this specific int8 GEMM.

Consequences for a browser-side OCR demo:

- **It works today.** Compiling the recognizer to `wasm32-unknown-unknown`
  yields a correct, pure-Rust OCR that runs entirely client-side (offline,
  private, no server round-trip) — but the int8 GEMM is **scalar**, so it's
  slower than the server.
- **To make it fast**, ndarray needs a `matmul_i8_to_i32` **wasm-v128 arm**.
  WASM SIMD128 has no VNNI (`VPDPBUSD`) equivalent, so the kernel would widen
  int8 → i16 and use `i32x4.dot_i16x8_s` (the `i16x8`-dot into `i32x4`) —
  standard for int8 GEMM under SIMD128. That is an **ndarray** change (per the
  `simd-savant` rule: add the arm in `ndarray/src/simd_*`, the recognizer
  consumes it unchanged), not a tesseract-rs change.
- **The browser demo is a different crate than `tesseract-ocr-web`.** That crate
  is an axum/tokio/reqwest **server** — none of which target browser-wasm. A
  client-side demo is a new thin `wasm-bindgen` crate exposing
  `recognize_image(bytes) -> doc.v1` over the recognizer, built
  `wasm32-unknown-unknown` with `-C target-feature=+simd128`.

## 5. Recommendation

| Target | Action |
|---|---|
| Server demo (x86 Railway) | **Nothing** — already SIMD-accelerated via runtime dispatch. |
| Browser (wasm) demo | Feasible now (scalar); worth a thin `wasm-bindgen` crate. For speed, add the **`matmul_i8_to_i32` wasm-v128 arm in ndarray** first. |

**Open probe (measure, don't assume):** once a wasm-v128 int8 GEMM exists,
benchmark scalar-wasm vs v128-wasm vs the x86 server on the same page to
quantify the browser speedup and confirm it clears the "usable in-browser"
bar. Until then, the browser path is correctness-first, not speed-first.
