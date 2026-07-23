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
