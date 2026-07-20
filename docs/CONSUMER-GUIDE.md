# Consuming tesseract-rs OCR — the OGAR path

> How an OGAR-v3 / lance-graph consumer (medcare-rs, woa-rs, smb-office-rs,
> odoo-rs, …) wires OCR in with the least possible implementation debt.
> **One rule:** you talk to OGAR, not to the recognizer.

## The boundary (read this first)

Three layers, one seam. Know which is yours.

| Layer | Concern | You touch it? |
|---|---|---|
| **tesseract-rs** | Be the faithful best-Tesseract. **Recognize** — words, per-word confidence, region split (text / table / figure / header / footer), table cell grids, typed field harvest → a rich **`doc.v1` JSON**. | No — never call `LstmRecognizer`, `structured`, or the renderers directly. |
| **OGAR** | The **connective tissue**. `ogar-vocab` declares the capabilities as classids; `tesseract-ogar` is the in-binary executor that runs them. | **Yes** — this is your whole contact surface. |
| **Consumer (you)** | Decide the document **store**, the **graph** ingestion, and *whether* to seed `lance-graph-arm-discovery` / DeepNSM from the JSON. Build the **PDF-from-your-data**. | Yes — this is your policy, not tesseract-rs's. |

**The `doc.v1` JSON is the seam.** tesseract-rs produces it; everything after —
storage, graph, comprehension, PDF-from-data — is your call. tesseract-rs knows
nothing about any of it.

**`doc.v1` is the OBSERVATION retina, and the DocIr composition layer does not
change that.** Under the DocIr composition grounding (OGAR
`docs/DOCIR-COMPOSITION-LAYER.md` #217 / `docs/DOCIR-COMPOSITION-GROUNDING.md`
#218), *DocIr* also names a **composition layer** over the OGAR object graph —
but that layer is **OGAR-side**, and tesseract-rs stays a pure **producer**. A
composed document does not paste your regions; it **references a `doc.v1`
awareness subtree through an `ObjectSlot`** (a typed projection portal:
`ObjectRef` + `ClassView` + `WideFieldMask` + `ResolutionMode`). The
`@sha256:` resolution arm resolves against the content hash of the **original
input bytes** — the `ogar-doc-ir::DocIr.content_sha256` convergence key,
computed **OGAR-side at ingestion** over the bytes you fed in. **This crate's
`doc.v1` does NOT stamp a hash:** `structured.rs` emits `schema` / page metadata
/ `quality` / `regions` / `fields` only, so `@sha256:` is an OGAR-side concern,
not a field you read out of `doc_json`. With that hash in hand, an imported scan
appears in a report as a live, revision-pinnable portal, not a decaying copy. So
the boundary is unchanged: you emit `doc.v1`; OGAR composes it. Nothing new is
required of tesseract-rs.

## The path — classid → executor → `doc.v1`

```rust
// 1. The authoritative capability table lives in OGAR (not tesseract-rs).
use ogar_vocab::ocr_actions::{OCR_ACTION_NAMES, OCR_SUBJECT_CLASSIDS};

// 2. The executor — the one type you drive.
use tesseract_ogar::{OcrExecutor, OcrRequest, OcrResponse};

// Load once (the eng.lstm model + unicharset + recoder; optional dict dawgs).
let exec = OcrExecutor::from_data_paths(
    lstm_path, unicharset_path, recoder_path,
    None, None, None,          // optional word / punc / number dawgs (dict beam)
)?;

// 3. Drive a typed request → typed response, keyed by capability.
let resp = exec.execute(OcrRequest::RecognizeDocument {
    grey: &pixels,          // 8-bit grey, row-major (see "Input" below)
    width,
    height,
    with_dict: false,       // true → run the language-model dict beam
    harvest_profile: None,  // e.g. Some("german_invoice") → typed field harvest
})?;

let OcrResponse::DocumentOut { doc_json, fields } = resp else { /* … */ };
// doc_json is the doc.v1 string; store it, or seed it downstream — your call.
```

`classid` is the **join key**: `ogar-vocab` *declares* the classids
(`OCR_SUBJECT_CLASSIDS`), `tesseract-ogar` *covers* them, and a `const _` fuse
asserts the two lists are the same length — drift is a compile error, not a
runtime surprise. Pull the classid from `ogar-vocab`; **never** construct a
`*Bridge` or copy the codebook (OGAR consumer rule: classid is pure address,
the magic is at the resolution target).

## The 14 capabilities

`ogar_vocab::ocr_actions::OCR_ACTION_NAMES`, each an `OcrRequest` variant:

| Action | In → Out |
|---|---|
| `recognize_line` / `recognize_page` | grey → text (`OcrResponse::{Recognized,PageText}`) |
| `recognize_page_words` | grey → words + boxes + confidence |
| `recognize_document` | grey → **`doc.v1`** (regions + tables + fields + quality) |
| `render_text` / `render_tsv` / `render_hocr` | words → the classic Tesseract text / TSV / hOCR outputs |
| `render_searchable_pdf` | page(s) → PDF bytes (original raster + invisible text layer) |
| `harvest_fields` | page → typed fields (invoice amounts, IBAN, …) |
| `segment_page` | grey → layout regions |
| `detect_halftone_regions` | grey → image-region masks/boxes |
| `detect_page_furniture` | line boxes → header/footer/page-number |
| `extract_text_layer` / `extract_page_image` | PDF bytes → existing text / page raster |

## The `doc.v1` seed shape

`recognize_document` → a JSON string. This is what you store / seed:

```json
{ "schema": "tesseract-rs/doc.v1",
  "pages": [{
    "page": 1, "width": 2480, "height": 3508,
    "quality": { "mean_conf": 96.1, "low_confidence": false },
    "regions": [
      { "type": "text",   "bbox": [l,t,r,b], "lines": [ /* words + conf */ ] },
      { "type": "table",  "bbox": [l,t,r,b], "rows": 7, "cols": 4,
        "cells": [ {"row":0,"col":0,"bbox":[…],"text":"Pos","header":true},
                   {"row":1,"col":3,"bbox":[…],"text":"1.250,00","header":false} ] },
      { "type": "figure", "bbox": [l,t,r,b], "lines": [] }
    ],
    "fields": [ {"key":"netto","value":"1.250,00","value_cents":125000,"bbox":[…]} ]
  }]
}
```

- `type` values are **additive** — `text` / `table` / `figure` / `header` /
  `footer` / the plain `paragraph` default. Ignore unknown ones.
- **`table` regions carry a cell grid** (`rows`/`cols`/`cells`) — rows are the
  recognized lines, columns are the whitespace-separated bands, each cell has
  its bbox + text + a header flag. This is the delicate feature that makes the
  JSON a good structured seed.
- **`figure` regions** are image/picture bboxes (logo, signature, stamp,
  photo) — you crop the raster from your original image using the bbox; storing
  it is your concern.
- `quality.low_confidence` is the honesty flag: `true` warns the input is
  likely handwriting / low-res / not printed text.

## Optionally seeding the graph (your call, not ours)

The `doc.v1` JSON is an **optional** seed. If you want paper → graph:

- **Tables** → feed the `cells` rows to `lance-graph-arm-discovery` (tabular
  rows → association rules → NARS-revised → ratified SPO line-items) → your
  graph.
- **Text** → feed to DeepNSM comprehension when you want free-text → SPO
  (deferred by default — table + picture structure is the near-term value).
- **Raw** → store the `doc_json` (and/or the original raster) in your KV.

None of this is tesseract-rs's concern — it hands you the seed and stops.

## Dependency story — BBB-clean

`tesseract-ogar` depends on `ogar-vocab` + `lance-graph-contract` +
`tesseract-{ocr,core,pdf,recognizer}`. **No lance-graph engine / planner**
("brain" crates) enters your binary. It pulls `ndarray` (pure compute) and
nothing that violates the customer-binary firewall. You can drive OCR from a
lean customer binary.

## Input — encoded images, one pure-Rust call

The executor takes **8-bit grey** pixels (row-major). Get them one of three
pure-Rust ways, all through the tesseract crates — you never wire `image`
yourself:

- **Encoded containers** (PNG / JPEG / WebP / TIFF / GIF / BMP / PNM) —
  `tesseract_ogar::decode_image(bytes)` → `(grey, width, height)`, behind the
  `image-decode` feature (bounded against decode bombs: dimension / pixel /
  alloc caps). Enable it: `tesseract-ogar = { …, features = ["image-decode"] }`.
- **P5 PGM** — `tesseract_ocr::image_input::parse_pgm` (lossless, no extra deps).
- **Raw RGB** — `tesseract_ocr::image_input::rgb_to_gray`.

So a consumer's ingest is two pure-Rust calls — decode, execute — touching only
the tesseract crates:

```rust
let (grey, width, height) = tesseract_ogar::decode_image(png_or_jpeg_bytes)?;
let resp = exec.execute(OcrRequest::RecognizeDocument {
    grey: &grey, width, height, with_dict: false, harvest_profile: None,
})?;
```

## Runnable reference

`cargo run -p tesseract-ogar --example ocr_demo` prints the OGAR capability
table + the `14 == 14` fuse, then runs `recognize_document` on a bundled page
and shows the `doc.v1` output. That example is the copy-paste starting point.
