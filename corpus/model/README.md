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

## Falsifier fixtures (not part of the `eng` or `deu` model)

Two components from *other* traineddata files live here for one reason: they
carry **varied** data where `eng`/`deu` carry uniform data, so a byte-parity
diff over them can actually fail. Each unblocks a leaf that was deferred not
for difficulty but because the available data could not falsify it.

| File | Source | Why it exists |
|---|---|---|
| `chi_sim.lstm-recoder` | `tessdata_fast/chi_sim.traineddata` | The recoder's `next_codes_` trie is **empty** for `eng`/`deu` — every code is length 1, so the multi-code beam paths are structurally unreachable. This one has 4022 entries with code lengths spanning **1-5** (histogram `{1:128, 2:278, 3:2077, 4:1515, 5:24}`), i.e. 3894 entries longer than 1. Unblocks "C3". Note this also corrects the plan's standing claim that Han codes are length-3: 3 is only the mode, not the range. |
| `eng.unicharset` | `tessdata/eng.traineddata` (the **legacy**, non-`fast`/`best` build) | The LSTM unicharset's bbox+stats CSV is identical on all 111 non-NULL lines (`0,255,0,255,0,0,0,0,0,0`), so `get_top_bottom` and the 6 float stats cannot be falsified against it. The legacy unicharset has **112/112 distinct** CSV blobs (e.g. `0,69,188,255,456,1188,0,30,486,1188`). Unblocks the deferred bbox/stats sub-leaf. |

Both were verified before being committed — the recoder histogram by parsing
the real wire format (`u32 count`, then per entry `i8 self_normalized · i32
length · length×i32 code`) with the parser cross-validated against
`eng`/`deu` first (both parse to 0 trailing bytes and match the naive
`4 + 9N` size formula exactly; `chi_sim` does **not** match it, which is the
positive evidence that it is genuinely multi-code).

**Do not confuse `eng.unicharset` with `eng.lstm-unicharset`.** They are
different components of different builds and the bare name is the real one
`combine_tessdata -u` emits for the legacy component — the LSTM path loads
only the `lstm-` prefixed file.

## License

These files are derived from traineddata distributed by the
[`tesseract-ocr/tessdata`](https://github.com/tesseract-ocr/tessdata)
and [`tessdata_fast`](https://github.com/tesseract-ocr/tessdata_fast)
repositories (`eng`/`deu` and the two falsifier fixtures above), licensed
under the
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
| `chi_sim.lstm-recoder` | `7ee2c195d397aa4fccd5efc5ab5e71d21d8e94425151d5f978cd74b546c4bb12` |
| `eng.unicharset` (legacy) | `c55602aa6fcff8461491ef52362a176bacc9950ad9b6eb4ccf8b78e46f504179` |

## Why these files are here

Committing the split components makes the golden suite (`corpus/golden/`,
see `../README.md`) **hermetic**: the full pipeline can be exercised in CI
with no network access, no C++ `tesseract`/`leptonica` build, and no
`combine_tessdata` step at test time — the golden tests read these files
directly off disk.
