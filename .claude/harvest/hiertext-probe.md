# HierText first-contact probe (2026-07-08)

**What:** 12 validation images of [HierText](https://github.com/google-research-datasets/hiertext)
(CC-BY-SA 4.0; images CVDF/Open Images) run through `recognize_page_makerow`
(release, non-dict) vs the line-level ground truth, side-by-side with the C++
`tesseract` CLI at `--psm 3` (full layout analysis) on the SAME PGMs.

**Fetch method (bounded, no full download):** GT `gt/validation.jsonl.gz`
(12 MB, raw.githubusercontent). Images: the 577 MB `validation.tgz` on the
public S3 bucket (`https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz`)
STREAMED via python `tarfile r|gz` over the socket, stopped after the first
12 image members (~4 MB transferred). JPG → grey PGM via PIL (probe-side
conversion; the in-repo JPEG path is the documented zune-jpeg approx arm).

**Metric:** per GT line, best-match CER against the output lines (two-row
Levenshtein); "recall" = GT lines with best CER ≤ 0.3.

## Results (line recall @CER≤0.3, mean best-CER)

| image | GT lines | ours | CLI psm3 | ours mCER | CLI mCER |
|---|---|---|---|---|---|
| 31205fe72d27f86d | 27 | 16 | 18 | 0.379 | 0.307 |
| 3a2da4708752effc | 62 | 1 | 1 | 0.958 | 0.958 |
| 5a7c5bfa8f9c3e75 | 22 | 0 | 0 | 0.894 | 0.961 |
| 6ce679754fa8bc1c | 25 | 0 | 2 | 0.898 | 0.822 |
| 757baf85b84deee3 | 36 | 1 | 21 | 0.821 | 0.306 |
| 7716494f93ad3fd7 | 95 | 0 | 4 | 0.816 | 0.725 |
| a37aa76a03fef256 | 35 | 0 | 2 | 0.892 | 0.854 |
| c14a3fe5dd2b12ad | 40 | 1 | 0 | 0.881 | 0.989 |
| d948faf8ce7aacd3 | 26 | 0 | 5 | 0.813 | 0.739 |
| ed7a3c39b2675a27 | 55 | 0 | 2 | 0.940 | 0.932 |
| f88965b98647b9bc | 25 | 0 | 0 | 0.921 | 1.000 |
| fe38725c7c9e9b65 | 20 | 0 | 0 | 0.842 | 1.000 |
| **total** | **468** | **19 (4.1%)** | **55 (11.8%)** | | |

## Reading (honest, per the never-blur rule)

1. **Scene text defeats BOTH engines.** The CLI with its full layout analysis
   reaches only 11.8% — HierText's scene/poster/magazine images are outside
   classic Tesseract's competence (Google built HierText to motivate
   detection-model pipelines). Our 4.1% is a SCOPE gap, not a transcode gap.
2. **The ours-vs-CLI delta is concentrated where multi-block LAYOUT wins:**
   `757baf85` (1 vs 21) and the small edges on `6ce`/`7716`/`d948` are
   ColumnFinder/tospace territory — exactly the documented Phase-7 items.
   On 4 of 12 images we equal or beat the CLI (incl. two where the CLI
   scores zero).
3. **The probe produced a product fix:** 5/12 images CRASHED the page path
   (Maxpool grid OOB on degenerate scene bands, scaled width 1-2 px). Root
   cause: the `Input::PrepareLSTMInputs` min-size guard
   (`width < XScaleFactor → "Image too small to scale!!"`, `input.cpp:92-96`)
   was untranscoded. Fixed faithfully: `Network::x_scale_factor()`
   (`network.h:214` + Series/Reconfig/Plumbing overrides; eng = 3 from
   `Mp3,3`) + the pre-scale dimension check in `recognize_grey_line`
   (too-small lines return empty, as the real chain skips them). After the
   fix: 12/12 run clean. Third instance of the noise-fixture blind-spot
   class: degenerate REAL-WORLD shapes never occur in synthetic fixtures.

**Repro:** fetch as above; `cargo run --release -p tesseract-ocr --example
recognize_page_makerow_dump -- corpus/model/eng.lstm{,-unicharset,-recoder}
<img.pgm>`; scoring script inline in the session (two-row Levenshtein,
best-match per GT line).
