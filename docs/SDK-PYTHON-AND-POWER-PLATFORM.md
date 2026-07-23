# SDK-Oberflächen — Python & Power Platform (MS Graph)

> Zwei Konsumenten-SDKs, die tesseract-rs bekommen könnte, und welche Form jedes
> annehmen sollte. Beide sind **heute Neuland** (kein pyo3/maturin-Crate, keine
> OpenAPI-Spezifikation) — dieses Dokument ist der Entwurf, nicht die
> Beschreibung von bestehendem Code. Die eine Regel, die beide erben: **Du
> sprichst mit der `OcrExecutor`/`doc.v1`-Naht, nie mit den Interna des
> Recognizers** (siehe `docs/CONSUMER-GUIDE.md`).

## Die eine Oberfläche, die beide umhüllen

Alles nachgelagerte steuert den einen Executor:

```rust
tesseract_ogar::OcrExecutor::from_data_paths(lstm, unicharset, recoder, …)
    .execute(OcrRequest::RecognizeDocument { grey, width, height, … })
    → OcrResponse::DocumentOut { doc_json /* doc.v1 */, fields }
```

Dazu `tesseract_ogar::decode_image(bytes)` (PNG/JPEG/WebP/TIFF/GIF/BMP/PNM → grau,
bomben-begrenzt), damit ein SDK-Ingest zwei reine Rust-Aufrufe sind. Beide SDKs
sind dünne Adapter darüber — keine neue Erkennungslogik, keine lance-graph-Engine
(BBB-sauber).

## 1. Python-SDK

**Warum es sich lohnt:** Das ganze Wertversprechen gegenüber `pytesseract` ist,
dass wir den **reinen Rust-Recognizer als Wheel** ausliefern — kein
System-`libtesseract`, kein `leptonica`, kein `apt install`. `pip install
tesseract-rs` und du hast byte-genaues Tesseract (eng + deu) mit null C zur
Laufzeit.

**Form:** ein PyO3- + maturin-Crate, z. B. `crates/tesseract-ocr-python`, das ein
kleines Modul über `tesseract-ogar` freigibt:

```python
import tesseract_rs as ocr

engine = ocr.Engine.from_model_dir("corpus/model", lang="deu")   # oder "eng"
doc = engine.recognize_document(image_bytes)     # -> dict (doc.v1: Regionen/Tabellen/Felder/Qualität)
txt = engine.recognize_text(image_bytes)         # -> str
pdf = engine.searchable_pdf(image_bytes)          # -> bytes (A: Scan + unsichtbarer Text)
# strukturiertes PDF (B), sobald es landet:
# pdf = engine.structured_pdf(image_bytes)
```

Entwurfsnotizen:
- `doc.v1` bildet sich natürlich auf ein Python-`dict` ab (das JSON parst zu
  nativen Typen); Tabellen kommen als `regions[].cells` durch — direkt iterierbar.
- `decode_image` wird ebenfalls freigegeben, damit Aufrufer rohe PNG/JPEG-Bytes
  übergeben, nicht Graustufen-Puffer.
- Das `image-decode`-Extra per Feature-Flag, damit ein schlanker Build nur-PGM
  bleiben kann.
- Wheel-Build via `maturin build --release`; das ~4 MB `corpus/model` liegt neben
  dem Wheel oder wird via `from_model_dir` referenziert.
- Spiegelt die bestehende `lance-graph-python`-Konvention (PyO3/maturin) im
  Workspace — gleiches Tooling, gleiches Packaging.

## 2. Power-Platform-Integrations-SDK (Custom Connector)

**Was es ist:** ein Power-Platform-**Custom-Connector** ist eine
OpenAPI-2.0-(Swagger-)Spezifikation über einem HTTP-Endpunkt; einmal importiert,
wird tesseract-rs zu einer Reihe von **Power-Automate-Aktionen** („Dokument
erkennen", „Durchsuchbares PDF"), nutzbar in Flows und Power Apps.

**Der Endpunkt existiert halb schon.** `tesseract-ocr-web` ist ein axum-Server;
heute bedient er `GET /` + `POST /ocr` (multipart, liefert JSON oder HTML).
Multipart ist für ein Browser-Formular in Ordnung, aber für einen Connector
umständlich. Die connector-freundliche Ergänzung:

- `POST /api/v1/recognize` — Body `application/octet-stream` (rohe Bild-/PDF-Bytes)
  **oder** `{ "content_base64": "…", "lang": "deu" }` → `doc.v1`-JSON.
- `POST /api/v1/pdf?mode=searchable|structured` — gleicher Input →
  `application/pdf`-Bytes.
- `GET /openapi.json` — die Swagger-2.0-Spezifikation, die Power Platform
  importiert.

**MS-Graph-Ergonomie (der Grund, warum Binär-Eingabe zählt).** Ein
Power-Automate-Flow bezieht die Datei fast immer aus Microsoft 365 via Graph —
*SharePoint / OneDrive „Get file content"*, ein *Outlook-Anhang*, eine
*Dataverse-Dateispalte*. Graphs „Get file content" liefert die **rohen Bytes**
(`$value`, `application/octet-stream`). Die primäre Connector-Aktion sollte also
genau das akzeptieren — einen binären Body — und `doc.v1` zurückgeben. Damit wird
der kanonische Flow ein Sprung:

```
Wenn eine Datei in SharePoint erstellt wird (Graph)
  → Dateiinhalt abrufen (Graph, binär)
  → [tesseract-rs] Dokument erkennen (binär → doc.v1)
  → doc.v1-Tabellen/Felder parsen
  → Dataverse-Zeilen erstellen/aktualisieren  (der Rechnungs-Materialisierungs-/Klickwege-Fall)
  → [tesseract-rs] Durchsuchbares PDF → zurück in SharePoint ablegen
```

Entwurfsnotizen:
- **Auth:** Die aktuelle Demo ist offen. Ein Power-Platform-Deployment braucht
  eine Identität — einen **API-Key-Header** (am einfachsten) oder **Azure AD /
  Entra OAuth** (natürlich in einem MS-Graph-Tenant; Connector und Flow teilen
  den Tenant). Das Security-Schema in die OpenAPI-Spezifikation aufnehmen.
- **Den SSRF-Schutz behalten.** Der Web-Crate blockiert auf seinem
  URL-Fetch-Zweig bereits nicht-öffentliche IPs / Metadaten-Endpunkte; eine
  URL-Eingabe-Aktion muss diesen Schutz behalten.
- **Body-Limits:** Die Connector-Spezifikation sollte die maximale Nutzlast
  deklarieren; der Server hat bereits `DefaultBodyLimit`.
- **Zustandslosigkeit:** Jeder Aufruf ist unabhängig (Modell einmal beim Start
  laden, den `OcrExecutor` in `AppState` halten); keine Session — Connectoren
  sind pro Aktion.
- Der Connector ist **nur die OpenAPI-über-dem-Server** — kein neuer
  Erkennungscode, dieselbe `OcrExecutor`/`doc.v1`-Naht. Er liefert als kleines
  Paar `apiDefinition.swagger.json` + `apiProperties.json` unter z. B.
  `integrations/power-platform/`.

## Wo jedes lebt (Abhängigkeits-Firewall)

| SDK | Crate / Artefakt | Hängt ab von |
|---|---|---|
| Python | `crates/tesseract-ocr-python` (PyO3/maturin-Wheel) | nur `tesseract-ogar` (BBB-sauber) |
| Power Platform | OpenAPI-Spezifikation + `/api/v1/*`-Routen auf `tesseract-ocr-web` | der bestehende axum-Server + `tesseract-ogar` |

Beide bleiben auf der standalone-Seite der Linie: `tesseract-ogar` + der
Recognizer, keine lance-graph-Engine, keine OGAR-Brain-Crates. Das Python-Wheel
und der Connector sind die beiden reibungsärmsten Wege, die reine Rust-OCR von
außerhalb von Rust zu konsumieren.
