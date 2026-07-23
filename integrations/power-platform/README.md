# tesseract-rs — Power Platform custom connector

A Power Platform (Power Automate / Power Apps) **custom connector** fronting
the `tesseract-ocr-web` server: three actions (recognize a document image,
export a searchable PDF, export a structured PDF) plus the OpenAPI 2.0
(Swagger) document the connector import reads. See
`docs/SDK-PYTHON-AND-POWER-PLATFORM.md` §2 for the design this implements.

No new recognition logic lives here — every action is a thin wrapper over
the same `doc.v1` seam the rest of this repo uses (`docs/CONSUMER-GUIDE.md`);
the connector is strictly the OpenAPI-over-the-existing-server layer.

## Files

| File | What it is |
|---|---|
| `apiDefinition.swagger.json` | The OpenAPI 2.0 spec: 3 operations (`RecognizeDocument`, `SearchablePdf`, `StructuredPdf`) + the `doc.v1` response schema. Served live at `GET /openapi.json` on the running server — byte-identical to this file (compiled in via `include_str!`, so the two can never drift). |
| `apiProperties.json` | Connector metadata: the `api_key` connection parameter (securestring) bound to the swagger's `api_key` security definition. |
| `README.md` | This file. |

## Prerequisites

A running `tesseract-ocr-web` instance reachable over **HTTPS** (the swagger
declares `"schemes": ["https"]`; Power Platform will not call a plain
`http://` backend outside of a local test connector). This repo's
`Dockerfile` + `railway.toml` deploy the server as a single binary on
Railway; any HTTPS-terminated host works the same way — see the crate's own
`README.md` for the deploy details (`MODEL_DIR`, `PORT`, etc.).

## 1. Point the spec at your deployment

`apiDefinition.swagger.json`'s `"host"` field is a placeholder
(`"CHANGE-ME.up.railway.app"`). Before importing, either:

- edit `host` in the file to your deployed hostname, **or**
- import as-is and change it afterwards in the Maker Portal: your connector
  → **Edit** → **General** tab → **Host**.

`basePath` is `"/"` — the three actions live at `/api/v1/recognize`,
`/api/v1/pdf`, and `/api/v1/pdf/structured` off that host.

## 2. Import

**Option A — Power Platform Maker Portal** (make.powerapps.com):

1. **Data** → **Custom connectors** → **New custom connector** →
   **Import an OpenAPI file** → select `apiDefinition.swagger.json` (or
   **Import an OpenAPI file from URL** →
   `https://<your-host>/openapi.json`, once the server is deployed — that
   route serves this same file byte-for-byte).
2. On the **Security** tab, confirm **API Key** is selected with header name
   `x-api-key` (prefilled from the swagger's `securityDefinitions`).
3. Review the three actions on the **Definition** tab, then **Create connector**.
4. Create a **connection** — this is where you paste your API key value, if
   your deployment enforces one (see §3 below).

**Option B — `paconn` CLI:**

```sh
pip install paconn
paconn login
paconn create \
  --api-prop apiProperties.json \
  --api-def apiDefinition.swagger.json \
  --secret <api-key-if-your-deployment-enforces-one>
```

Use `paconn update` for subsequent changes to either file.

## 3. Auth — the `x-api-key` header

The swagger declares an `apiKey` security scheme (`x-api-key` header), so the
connector will always prompt for a value when a connection is created.
**Server-side enforcement is opt-in and OFF by default**, matching the
existing open HTML demo:

- **`TESSERACT_API_KEY` unset** on the server (the default) → every request
  is accepted regardless of the `x-api-key` header's value. This is the
  demo/dev posture — fine for a private test deployment, not for a public one.
- **`TESSERACT_API_KEY=<secret>` set** on the server → `/api/v1/*`
  (`RecognizeDocument`, `SearchablePdf`, `StructuredPdf`) now requires a
  matching `x-api-key` header; a missing or wrong value gets HTTP `401` with
  a `{"error": "..."}` body. Give the same `<secret>` as the connection's API
  Key when you create the Power Platform connection.

`GET /openapi.json` is never gated — the discovery document must stay
fetchable without a key (Power Platform's importer, and anyone verifying the
connector, needs to read it directly).

**What this does NOT implement:** anything beyond a single shared-secret
header compare — no per-caller keys, no rotation, no rate limiting. Put this
behind Power Platform's own connection-level access control and your
platform's network/HTTPS boundary; this header is a coarse gate, not a full
auth system. If you need finer-grained auth (Azure AD / Entra OAuth, per-user
identity — see the design doc's alternative), that is a separate, unbuilt
security scheme; this pass ships only the API-key gate.

## 4. The three actions

| operationId | Method + path | Input | Output |
|---|---|---|---|
| `RecognizeDocument` | `POST /api/v1/recognize` | binary image body (`application/octet-stream`) **or** `application/json` `{"content_base64": "...", "lang": "eng"}` | `tesseract-rs/doc.v1` JSON |
| `SearchablePdf` | `POST /api/v1/pdf` (optional `?mode=searchable\|structured`, default `searchable`) | same as above | `application/pdf` |
| `StructuredPdf` | `POST /api/v1/pdf/structured` | same as above | `application/pdf` (always the structured reconstruction) |

Notes:

- **`lang` is accepted, not enforced.** This server loads exactly one model
  at startup (`MODEL_DIR` — see the crate's own `README.md`/`CLAUDE.md`); the
  field exists so a caller that sends it (mirroring the Python SDK's
  `lang=` constructor argument) doesn't get a parse error. It currently has
  no effect on which model recognizes the document — a per-request language
  switch would need a multi-model server, which is out of scope for this
  connector pass. A mismatch is logged server-side (stderr), not rejected.
- **Upload size** is capped at 12 MB, shared with the HTML upload form's
  `DefaultBodyLimit`/`RequestBodyLimitLayer` — a larger body is rejected by
  the framework before any handler runs.
- **No URL-fetch action.** The HTML demo's "paste an image URL" arm
  (SSRF-guarded in `src/fetch.rs`) is intentionally NOT exposed as a
  connector action — a Power Automate flow supplies file bytes directly
  (Graph/SharePoint/OneDrive/Dataverse), so there is no URL-fetch surface
  here that would need guarding in the first place.
- **Errors** are always `{"error": "<message>"}` JSON with a `4xx`/`5xx`
  status — never the HTML error page the browser-facing routes render.

## 5. Example flow — SharePoint invoice → Dataverse → searchable PDF

The scenario `docs/SDK-PYTHON-AND-POWER-PLATFORM.md` §2 is designed around
(Microsoft Graph's binary "Get file content" is why the primary input is raw
bytes, not base64):

1. **Trigger:** *When a file is created in a folder* (SharePoint) — or the
   equivalent OneDrive / Outlook-attachment / Dataverse-file-column trigger.
2. **Get file content** (SharePoint/Graph) — binary output.
3. **RecognizeDocument** (this connector) — body = the previous step's file
   content, unchanged. Output: `doc.v1` JSON.
4. **Parse JSON** (built-in Power Automate action) over the `doc.v1` output
   using the `DocV1` schema from `apiDefinition.swagger.json` — or reference
   `body('RecognizeDocument')?['pages'][0]['fields']` /
   `?['regions']` dynamically without a separate parse step.
5. Loop the parsed `fields` / table `cells` → **Add a new row** (Dataverse)
   or your line-of-business system of choice.
6. **SearchablePdf** (this connector) — same file content as step 2 — to get
   back a scan-plus-invisible-text-layer PDF.
7. **Create file** (SharePoint/Graph) — store the PDF from step 6 back next
   to (or in place of) the original.

## What this connector deliberately does NOT do

Per `docs/SDK-PYTHON-AND-POWER-PLATFORM.md`'s dependency firewall: no new
recognition code, no OGAR, no lance-graph engine. This is strictly the
OpenAPI-over-the-existing-server layer. Storage, graph ingestion, and "what
do I do with the `doc.v1` JSON" are policy decisions for your flow — the same
boundary `docs/CONSUMER-GUIDE.md` draws for any Rust consumer.
