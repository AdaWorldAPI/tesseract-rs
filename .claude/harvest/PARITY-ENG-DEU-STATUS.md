# eng + deu byte-parity across all features (+ Sauvola) — status tracker

Goal (operator, 2026-07-23): prove the tesseract-rs transcode is byte-parity
against libtesseract/leptonica for **both eng and deu** across every feature,
and add **Sauvola** adaptive binarization (leptonica leaf, transcoded from the
`AdaWorldAPI/leptonica` fork `src/binarize.c`).

## Method (the proven two-step)

1. **Install the oracle** — `apt-get install tesseract-ocr libtesseract-dev
   libleptonica-dev tesseract-ocr-{eng,deu}` (in-env lib: tesseract **5.3.4**,
   leptonica **1.82.0**). Source headers: `git clone --branch 5.3.4` →
   `/tmp/tesseract-src` (lib and headers both 5.3.4 → **zero ABI skew**, an
   improvement over the earlier 5.5.0-header/5.3.4-lib skew).
2. **Transcode + byte-parity** — dump the same field both sides, `diff`. The
   `bijection` half self-validates the object layout before any field half is
   trusted (E-CPP-PARITY-1).

deu components extracted with `combine_tessdata -u deu.traineddata corpus/model/deu.`
(deu.lstm 417 KB, unicharset 116 entries, recoder 116 codes = 4+116·9, word-dawg 1 MB).

## Feature status

| Feature | Leaf | eng | deu | Oracle |
|---|---|---|---|---|
| UNICHARSET bijection + 5 fields | E-CPP-PARITY-1..6 | ✅ 6/6 (112) | ✅ 6/6 (116, incl. Ä Ö Ü ä ö ü ß) | `unicharset_oracle.cpp` (rebuilt in-container) |
| UNICHAR utf8 codec | E-CPP-PARITY-2 | ⏳ | (model-indep) | reconstruct |
| recoder encode/decode/beam | E-CPP-PARITY-7 / -RECODER-BEAM-1 | ⏳ | ⏳ | reconstruct (unicharcompress.h) |
| network forward (softmax) | E-OCR-NETWORK-FORWARD-1 | ⏳ | ⏳ | Network::CreateFromFile |
| beam decode (CTC) | E-OCR-RECODEBEAM-1 | ⏳ | ⏳ | reconstruct |
| image → text | E-OCR-IMAGE-TEXT-1 | ⏳ | ⏳ | image_text_oracle_ctc.cpp (banked) |
| dict / dawg walk | (C1) | ⏳ | ⏳ | dict_walk oracle |
| **Sauvola adaptive binarize** | NEW | ⏳ | (model-indep) | liblept 1.82.0 (`pixSauvolaBinarizeTiled`) |

Leptonica image leaves (scale, morph, otsu, pageseg, decide_if_table, conncomp,
binreduce) are **script-independent** — already proven; a German page is still
just pixels, so they need no per-language re-run (re-confirm as regression only).

## In-container reproduction (2026-07-23)

- eng unicharset **6/6 byte-parity** — the container reproduces the known-green
  E-CPP-PARITY-1..6 with the 5.3.4/5.3.4 oracle.
- deu unicharset **6/6 byte-parity** — FIRST non-eng model proven; the UNICHARSET
  transcode is genuinely **model-agnostic**, not eng-overfit (multibyte umlaut/ß
  ids all byte-identical).

Harness: `run_unicharset_parity.sh <unicharset> <label>`.

## IN-CONTAINER RESULTS — 2026-07-23 (all green so far)

| Feature | eng | deu | Falsifier that proves it is real |
|---|---|---|---|
| UNICHARSET (bijection + 5 fields) | ✅ 6/6 (112) | ✅ 6/6 (116) | multibyte Ä Ö Ü ä ö ü ß ids byte-identical |
| UNICHAR utf8 codec | ✅ 268 | ✅ 268 (model-indep) | overlong-NUL→0, 4 illegal leads→ILLEGAL |
| recoder encode/decode/beam | ✅ 112/113/114 | ✅ 116/117/118 | code_range 111 vs 115; shared-code id 1→2; id-ordered beam final list |
| network forward (softmax) | ✅ 8/8 | ✅ 8/8 | deu nw=400979 vs eng 385807 (different arch), both agree |
| image → text (end-to-end capstone) | ✅ 6/6 | ✅ 6/6 | deu null_char=114 vs eng 110, sample_iteration differ; both agree |
| **Sauvola adaptive binarize** (NEW) | ✅ 5/5 configs | ✅ (model-indep) | 368640-px real page + usetab=1 LUT + whsize 4–15 all byte-identical |
| beam decode (standalone) | ✅ 2/2 modes | ✅ 2/2 modes | shared probs.bin, exact f32 bits; deu code_range 116/null 114 |
| dict / dawg walk | ✅ 14/14 | ✅ 14/14 | German trie: über/schön/ß, numbers, back_to_punc dead-ends |

**Sauvola** transcoded from `AdaWorldAPI/leptonica` `src/{binarize.c,convolve.c,pix2.c}`
into `crates/tesseract-ocr/src/binarize.rs`; byte-parity vs liblept 1.82.0
(`sauvola_oracle.cpp`). The full `pixSauvolaBinarize` chain (mirror border →
u32/f64 integral windowed mean+mean-square → threshold `m(1-k(1-s/128))` → apply)
is byte-identical; 3 unit tests, clippy-clean.

Headline: **the transcode is model-agnostic.** Every core leaf proven on eng is
byte-identical on deu too — the German model self-derives different constants
(charset 116, code_range 115, null_char 114, nw 400979) and the Rust reproduces
all of them. Nothing was eng-overfit.
