# tesseract-rs (Python SDK)

PyO3 + maturin wheel over [`tesseract-ogar::OcrExecutor`](../tesseract-ogar) --
pure-Rust, byte-parity Tesseract OCR (`eng` + `deu`). No `libtesseract`, no
`leptonica`, no C at runtime: `pip install tesseract-rs` ships one native
extension module, and you point it at wherever you ship the ~4 MB model
files (`corpus/model/{eng,deu}.lstm*`).

This crate's only non-`pyo3` dependency is `tesseract-ogar` (see its own
`Cargo.toml`) -- BBB-clean, matching
[`docs/SDK-PYTHON-AND-POWER-PLATFORM.md`](../../docs/SDK-PYTHON-AND-POWER-PLATFORM.md)
Section 1, the design this crate implements. See
[`docs/CONSUMER-GUIDE.md`](../../docs/CONSUMER-GUIDE.md) for the underlying
`OcrExecutor`/`doc.v1` seam this SDK is a thin adapter over.

## Build

```sh
cd crates/tesseract-ocr-python
maturin build --release      # -> a wheel under target/wheels/
# or, for local development against your active venv:
maturin develop --release
```

The compiled wheel is a single native extension module (`tesseract_rs`).
Model files are **not** bundled into the wheel -- point `from_model_dir` at
wherever you ship `corpus/model/` (or your own model directory with the same
`{lang}.lstm*` naming).

## Usage

```python
import tesseract_rs as ocr

engine = ocr.Engine.from_model_dir("corpus/model", lang="deu")   # or "eng"
doc = engine.recognize_document(image_bytes)     # -> dict (doc.v1: regions/tables/fields/quality)
txt = engine.recognize_text(image_bytes)         # -> str
pdf = engine.searchable_pdf(image_bytes)         # -> NotImplementedError today, see "Gaps" below
```

`image_bytes` is any pure-Rust-decodable container: PNG, JPEG, WebP, TIFF,
GIF, BMP, or PNM (via `tesseract_ogar::decode_image`, feature `image-decode`,
enabled unconditionally in this crate's `Cargo.toml` -- every `Engine`
method's Python signature is `(image_bytes)`, no width/height, so decode has
to happen inside this crate).

`recognize_document`'s return value is a genuine recursive JSON -> Python
conversion (`json_to_py` in `src/lib.rs`), not a string you `json.loads`
yourself -- `doc["pages"][0]["regions"]` is directly iterable, and table
regions carry `rows` / `cols` / `cells` per `docs/CONSUMER-GUIDE.md`'s
`doc.v1` seed shape.

There's also a standalone `tesseract_rs.decode_image(bytes) -> (grey_bytes,
width, height)` for callers who want the raw grey buffer without running OCR.

## Gaps

### `searchable_pdf` is not implemented -- a real `tesseract-ogar` gap, not missing glue

`OcrRequest::RenderSearchablePdf { pages: &[PageOcr], dpi }` needs a
freshly-built `tesseract_ocr_pdf::PageOcr` (a `GreyImage` + `Vec<PlacedWord>`,
one `PlacedWord` per recognized word: text + a top-down pixel box).
`tesseract-ogar` (`crates/tesseract-ogar/src/lib.rs`) re-exports only
`decode_image`/`ImageDecodeError` -- it does **not** re-export `PageOcr`,
`GreyImage`, or `PlacedWord`, and no `OcrResponse` variant ever hands one
back either (`OcrResponse::LineWordsOut(Vec<LineWords>)` is the closest, and
`LineWords`/`WordResult` carry bottom-up boxes, not the top-down
`PlacedWord::box_` shape `PageOcr` needs).

Since this crate depends on `tesseract-ogar` alone (BBB-clean, per the design
doc's dependency table), it has no way to *name* -- let alone construct --
the type `RenderSearchablePdf` requires (Rust's extern prelude only exposes
a crate's types to a direct dependent; `tesseract-ogar` never re-exports
these). `Engine.searchable_pdf` therefore raises `NotImplementedError` with
this same explanation rather than faking a result or reaching past the
declared dependency. Fixing it needs one of:

- `tesseract-ogar` re-exporting `PageOcr` / `GreyImage` / `PlacedWord` (plus
  the bottom-up -> top-down box conversion `tesseract_ocr::renderer::
  to_image_box` already does), or
- a new convenience method, e.g. `OcrExecutor::
  render_searchable_pdf_from_grey(grey, width, height, with_dict, dpi) ->
  Result<(Vec<u8>, RenderReport), OcrExecError>`, that builds `PageOcr`
  internally from a freshly recognized page and returns PDF bytes directly
  -- the shape a single-dependency SDK actually needs.

`structured_pdf` (the design doc's "B" variant) is out of scope for the same
reason and isn't stubbed here at all -- the design doc itself marks it
"sobald es landet" (not yet landed upstream either).

### Not exposed as parameters (by design, matching the spec exactly)

`recognize_document` / `recognize_text` always run with `with_dict=false`
and (for `recognize_document`) `harvest_profile=None`. The design doc's
Python signatures take only `image_bytes`, so this crate doesn't invent
extra kwargs beyond that spec.
