# corpus/model — LSTM model components

The files in this directory are the individual components of the English
`eng` LSTM model, extracted from the upstream combined `eng.traineddata`
file via:

```sh
combine_tessdata -u eng.traineddata eng
```

`combine_tessdata -u` performs a lossless split of the combined container
into its named parts — no modification is made to any trained weight or
dictionary entry. These are the exact same component files
`tesseract-ocr`'s `LstmRecognizer::from_components` /
`LstmRecognizer::DeSerialize` load individually (the "split-traineddata"
path — see the top-level `CLAUDE.md`, "B2 is DONE").

## Components

| File | Format | Role |
|---|---|---|
| `eng.lstm` | binary (custom little-endian) | The trained LSTM network: a tree of typed layers (Convolve/Maxpool/LSTM/FullyConnected/Series/Reversed/...), int8-quantized weights. Loaded by `Network::from_le_bytes` (`tesseract-ocr`). |
| `eng.lstm-unicharset` | text | The UNICHARSET: the id<->unichar bijection plus per-character properties (alpha/lower/upper/digit/punctuation/ngram), the script table, case pairs, and direction/mirror flags. Loaded by `CharSet` (`tesseract-core`). |
| `eng.lstm-recoder` | binary (`TFile` little-endian) | The UNICHARCOMPRESS recoder: maps the network's compressed output codes back to unicharset ids (`code_range`, `EncodeUnichar`/`DecodeUnichar`), plus the beam-search maps (`is_valid_start_`/`final_codes_`/`next_codes_`). Loaded by `Recoder` (`tesseract-core`). |
| `eng.lstm-word-dawg` | binary (`SquishedDawg`) | Word dictionary DAWG (directed acyclic word graph) consumed by the dictionary beam to bias recognition toward in-vocabulary words. |
| `eng.lstm-punc-dawg` | binary (`SquishedDawg`) | Punctuation-pattern DAWG (leading/trailing punctuation shapes allowed around a dictionary word). |
| `eng.lstm-number-dawg` | binary (`SquishedDawg`) | Numeric-pattern DAWG (digit-string shapes: dates, amounts, etc). |

The three DAWGs are consumed together by `tesseract-core`'s DAWG walker
(`DictLite`) for the dictionary-beam path (plan Phase 1, Batch 1A / "C1").

## License

These files are derived from the `eng.traineddata` file distributed by the
[`tesseract-ocr/tessdata`](https://github.com/tesseract-ocr/tessdata)
repository, licensed under the
[Apache License 2.0](https://github.com/tesseract-ocr/tessdata/blob/main/LICENSE).
Redistribution here (as the split components produced by `combine_tessdata
-u`, byte-identical to their content inside the original combined file) is
under the same Apache License 2.0 terms; see the upstream `LICENSE` file
for the full text and the upstream repository for attribution.

## SHA256

Filled in by the orchestrator at commit time, against the exact bytes
committed to this directory:

| File | SHA256 |
|---|---|
| `eng.lstm` | `78637462a335f887f7acc052f34fc5bf60c8015908352587e638a69ea4ca2756` |
| `eng.lstm-unicharset` | `3a18fb4e5d2df0ffa66092609a4b07434c23160c90c2b9a315e3992e389a95fa` |
| `eng.lstm-recoder` | `a481e4cb27c2b832269a0578a1438c243a13228a70f9556162b7f06131d2e664` |
| `eng.lstm-word-dawg` | `a5dabb1725487e85b364a49b095b5a9af5cc2720ef29c962189e4cf5294fc81c` |
| `eng.lstm-punc-dawg` | `c3e90e22c6bfc25365e5f5cdf09397e9e3fd58e07903b6d1f76a4450893601bf` |
| `eng.lstm-number-dawg` | `7104fc60ebd9093f2ebfefd5bd27347a68fe9b6ce03be3135c8cbdabcdd99994` |

## Why these files are here

Committing the split components makes the golden suite (`corpus/golden/`,
see `../README.md`) **hermetic**: the full pipeline can be exercised in CI
with no network access, no C++ `tesseract`/`leptonica` build, and no
`combine_tessdata` step at test time — the golden tests read these files
directly off disk.
