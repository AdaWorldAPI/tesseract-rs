# SIMD-Integration & die WASM-Frage

> Müssen wir ndarrays SIMD-Polyfill in den Recognizer einbinden, und ist der
> WASM-Zweig für die Web-Demo nützlich? Kurz: **Die Integration existiert
> bereits.** Das interessante Neuland ist ein Browser-seitiger (WASM-)Build —
> der heute läuft, aber auf dem int8-Hotpfad **noch nicht** SIMD-beschleunigt ist.

## 1. Fazit — die SIMD-Integration ist bereits erledigt

Der gesamte int8-Hotpfad des Recognizers (jedes LSTM-Gate und jede
Fully-Connected-Schicht ist ein `int8 × int8 → i32`-Matrix-Vektor-Produkt) ist
ein **einziger Aufruf** von ndarrays SIMD-Polyfill:

```rust
// crates/tesseract-recognizer/src/lib.rs:146
ndarray::simd::matmul_i8_to_i32(w, u_col, out.view_mut())
```

Damit ist die `simd-savant`-Invariante („alle SIMD kommen aus `ndarray::simd`")
bereits erfüllt — der Recognizer schreibt nie selbst ein Intrinsic, er konsumiert
den einen polyfill-Kernel. **Es gibt nichts zu integrieren.** Welche
Geschwindigkeit wir bekommen (oder nicht), entscheidet sich innerhalb von
ndarray, transparent für tesseract-rs.

## 2. Wie das Polyfill dispatcht

`ndarray/src/simd.rs` ist der Dispatch-Einstieg über die architektur-spezifischen
Backends (`simd_{amx,avx512,avx2,neon,neon_baseline,neon_bf16,neon_dotprod,scalar,wasm}.rs`),
mit `simd_caps.rs` als `LazyLock<SimdCaps>` — die CPU **einmal** erkennen, dann für
immer dispatchen.

`matmul_i8_to_i32` dispatcht zur Laufzeit:

```
AMX TDPBUSD  →  AVX-512 VNNI (VPDPBUSD-zmm)  →  AVX2 VNNI (VPDPBUSD-ymm)  →  Skalar
```

| Ziel | Pfad des int8-GEMM |
|---|---|
| x86-64 Sapphire Rapids+ (AMX) | AMX-Kachel `TDPBUSD` |
| x86-64 mit AVX-512 VNNI | `VPDPBUSD` auf zmm |
| x86-64 mit AVX2 VNNI | `VPDPBUSD` auf ymm |
| x86-64 Baseline / aarch64 ohne dotprod | Skalar (bzw. NEON-dotprod, wo vorhanden) |
| **wasm32** | **Skalar** (siehe §4) |

x86/aarch64 nutzen **Laufzeit**-Erkennung (`is_x86_feature_detected!` usw.); das
Ergebnis wird im `LazyLock` zwischengespeichert.

## 3. Die Server-Demo (Railway, x86-64) — bereits schnell, nichts zu tun

Der Web-Demo-Binary läuft auf Railways x86-64. Der int8-GEMM wird **bereits** zur
Laufzeit auf den besten verfügbaren VNNI/AMX-Pfad dispatcht — kein Build-Flag,
keine Code-Änderung nötig, weil die Erkennung zur Laufzeit passiert. Ein Host mit
AVX-512 VNNI bekommt automatisch `VPDPBUSD-zmm`; ein Baseline-Host fällt auf
Skalar, läuft aber. **Empfehlung: so lassen — kein exotisches `target-cpu`
festverdrahten**, das Portabilität gegen einen Pfad tauschen würde, den die
Laufzeit-Dispatch ohnehin wählt. (Wer einen Boden erzwingen will:
`-C target-cpu=x86-64-v3` aktiviert AVX2 als kompilierten Baseline, während die
Laufzeit-Erkennung weiterhin auf AVX-512/AMX anhebt.)

## 4. Der WASM-Zweig — real, aber der int8-GEMM ist (noch) nicht daran gebunden

ndarray **hat** ein WASM-Backend — `simd_wasm.rs` plus vollständige
`wasm32`-Lane-Konstanten in `simd.rs` (`f32x4`, `f64x2`, `i16x8`, … die
v128/SIMD128-Formen), zur **Kompilierzeit** ausgewählt
(`target_feature = "simd128"`; WASM hat keine Laufzeit-Feature-Erkennung, es ist
also ein Build-Gate, nicht die `simd_caps`-Laufzeit-Stufe).

**Aber `matmul_i8_to_i32` hat keinen wasm-v128-Zweig.** Seine Dispatch-Leiter ist
AMX → AVX-512 VNNI → AVX2 VNNI → **Skalar**; auf `wasm32` trifft keiner der ersten
drei zu, der Hotpfad des Recognizers läuft im Browser also **skalar**. Das
v128-Backend beschleunigt die *generischen* Lane-Primitive, nicht diesen
spezifischen int8-GEMM.

Folgen für eine Browser-seitige OCR-Demo:

- **Sie läuft heute.** Ein Kompilat des Recognizers nach
  `wasm32-unknown-unknown` ergibt eine korrekte, reine Rust-OCR, die komplett
  Client-seitig läuft (offline, privat, ohne Server-Roundtrip) — aber der
  int8-GEMM ist **skalar**, also langsamer als der Server.
- **Um es schnell zu machen**, braucht ndarray einen `matmul_i8_to_i32`
  **wasm-v128-Zweig**. WASM SIMD128 hat kein VNNI-(`VPDPBUSD`-)Äquivalent, der
  Kernel würde int8 → i16 verbreitern und `i32x4.dot_i16x8_s` nutzen (den
  `i16x8`-Dot in `i32x4`) — Standard für int8-GEMM unter SIMD128. Das ist eine
  **ndarray**-Änderung (gemäß `simd-savant`-Regel: den Zweig in
  `ndarray/src/simd_*` hinzufügen, der Recognizer konsumiert ihn unverändert),
  keine tesseract-rs-Änderung.
- **Die Browser-Demo ist ein anderer Crate als `tesseract-ocr-web`.** Dieser
  Crate ist ein axum/tokio/reqwest-**Server** — nichts davon zielt auf
  Browser-WASM. Eine Client-seitige Demo ist ein neuer dünner
  `wasm-bindgen`-Crate, der `recognize_image(bytes) -> doc.v1` über den
  Recognizer freigibt, gebaut als `wasm32-unknown-unknown` mit
  `-C target-feature=+simd128`.

## 5. Empfehlung

| Ziel | Aktion |
|---|---|
| Server-Demo (x86 Railway) | **Nichts** — bereits SIMD-beschleunigt via Laufzeit-Dispatch. |
| Browser-(WASM-)Demo | Heute machbar (skalar); ein dünner `wasm-bindgen`-Crate lohnt sich. Für Tempo zuerst den **`matmul_i8_to_i32`-wasm-v128-Zweig in ndarray** ergänzen. |

**Offene Probe (messen, nicht annehmen):** Sobald ein wasm-v128-int8-GEMM
existiert, Skalar-WASM vs. v128-WASM vs. x86-Server auf derselben Seite
benchmarken, um den Browser-Speedup zu quantifizieren und zu bestätigen, dass er
die „im-Browser-benutzbar"-Schwelle nimmt. Bis dahin ist der Browser-Pfad
korrektheits-first, nicht tempo-first.
