# SDK surfaces — Python & Power Platform (MS Graph)

> Two consumer SDKs tesseract-rs could grow, and the shape each should take.
> Both are **greenfield today** (no pyo3/maturin crate, no OpenAPI spec) — this
> doc is the design, not a description of existing code. The one rule both
> inherit: **you talk to the `OcrExecutor` / `doc.v1` seam, never the
> recognizer internals** (see `docs/CONSUMER-GUIDE.md`).

## The single surface both wrap

Everything downstream drives the one executor:

```rust
tesseract_ogar::OcrExecutor::from_data_paths(lstm, unicharset, recoder, …)
    .execute(OcrRequest::RecognizeDocument { grey, width, height, … })
    → OcrResponse::DocumentOut { doc_json /* doc.v1 */, fields }
```

Plus `tesseract_ogar::decode_image(bytes)` (PNG/JPEG/WebP/TIFF/GIF/BMP/PNM →
grey, bomb-bounded) so an SDK ingest is two pure-Rust calls. Both SDKs are thin
adapters over this — no new recognition logic, no lance-graph engine (BBB-clean).

## 1. Python SDK

**Why it's worth it:** the whole value proposition vs `pytesseract` is that ours
ships the **pure-Rust recognizer as a wheel** — no system `libtesseract`, no
`leptonica`, no `apt install`. `pip install tesseract-rs` and you have
byte-parity Tesseract (eng + deu) with zero C at runtime.

**Shape:** a PyO3 + maturin crate, e.g. `crates/tesseract-ocr-python`, exposing
a small module over `tesseract-ogar`:

```python
import tesseract_rs as ocr

engine = ocr.Engine.from_model_dir("corpus/model", lang="deu")   # or "eng"
doc = engine.recognize_document(image_bytes)     # -> dict (doc.v1: regions/tables/fields/quality)
txt = engine.recognize_text(image_bytes)         # -> str
pdf = engine.searchable_pdf(image_bytes)          # -> bytes (A: scan + invisible text)
# structured PDF (B) once it lands:
# pdf = engine.structured_pdf(image_bytes)
```

Design notes:
- `doc.v1` maps naturally to a Python `dict` (the JSON parses to native types);
  tables come through as `regions[].cells` — directly iterable.
- `decode_image` is exposed too, so callers pass raw PNG/JPEG bytes, not grey
  buffers.
- Feature-gate the `image-decode` extra so a lean build can stay PGM-only.
- Wheel build via `maturin build --release`; the ~4 MB `corpus/model` ships
  beside the wheel or is pointed at via `from_model_dir`.
- Mirrors the existing `lance-graph-python` (PyO3/maturin) convention in the
  workspace — same tooling, same packaging.

## 2. Power Platform integration SDK (custom connector)

**What it is:** a Power Platform **custom connector** is an OpenAPI 2.0 (Swagger)
spec over an HTTP endpoint; once imported, tesseract-rs becomes a set of
**Power Automate actions** ("Recognize Document", "Searchable PDF") usable in
flows and Power Apps.

**The endpoint already half-exists.** `tesseract-ocr-web` is an axum server; today
it serves `GET /` + `POST /ocr` (multipart, returns JSON or HTML). Multipart is
fine for a browser form but awkward for a connector. The connector-friendly
addition:

- `POST /api/v1/recognize` — body `application/octet-stream` (raw image/PDF
  bytes) **or** `{ "content_base64": "…", "lang": "deu" }` → `doc.v1` JSON.
- `POST /api/v1/pdf?mode=searchable|structured` — same input → `application/pdf`
  bytes.
- `GET /openapi.json` — the Swagger 2.0 spec Power Platform imports.

**MS Graph ergonomics (the reason binary-in matters).** A Power Automate flow
almost always sources the file from Microsoft 365 via Graph — *SharePoint /
OneDrive "Get file content"*, an *Outlook attachment*, a *Dataverse file
column*. Graph's "Get file content" returns the **raw bytes** (`$value`,
`application/octet-stream`). So the connector's primary action should accept
exactly that — a binary body — and return `doc.v1`. That makes the canonical
flow one hop:

```
When a file is created in SharePoint (Graph)
  → Get file content (Graph, binary)
  → [tesseract-rs] Recognize Document (binary → doc.v1)
  → Parse doc.v1 tables/fields
  → Create/update Dataverse rows  (the invoice-materialization / Klickwege use case)
  → [tesseract-rs] Searchable PDF → store back in SharePoint
```

Design notes:
- **Auth:** the current demo is open. A Power Platform deployment needs an
  identity — an **API-key header** (simplest) or **Azure AD / Entra OAuth**
  (natural in an MS-Graph tenant; the connector and the flow share the tenant).
  Add the security scheme to the OpenAPI spec.
- **Keep the SSRF guard.** The web crate already blocks non-public IPs / metadata
  endpoints on its URL-fetch arm; a URL-input action must keep that guard.
- **Body limits:** the connector spec should declare max payload; the server
  already has `DefaultBodyLimit`.
- **Statelessness:** each call is independent (load model once at boot, keep the
  `OcrExecutor` in `AppState`); no session — connectors are per-action.
- The connector is **just the OpenAPI-over-the-server** — no new recognition
  code, the same `OcrExecutor`/`doc.v1` seam. It ships as a small
  `apiDefinition.swagger.json` + `apiProperties.json` pair under, e.g.,
  `integrations/power-platform/`.

## Where each lives (dependency firewall)

| SDK | Crate / artifact | Depends on |
|---|---|---|
| Python | `crates/tesseract-ocr-python` (PyO3/maturin wheel) | `tesseract-ogar` only (BBB-clean) |
| Power Platform | OpenAPI spec + `/api/v1/*` routes on `tesseract-ocr-web` | the existing axum server + `tesseract-ogar` |

Both stay on the standalone side of the line: `tesseract-ogar` + the recognizer,
no lance-graph engine, no OGAR brain crates. The Python wheel and the connector
are the two "least-friction" ways to consume the pure-Rust OCR from outside Rust.
