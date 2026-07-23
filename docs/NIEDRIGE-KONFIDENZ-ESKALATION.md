# Eskalation bei niedriger Konfidenz — Handschrift & schwache Scans

> tesseract-rs ist schnell und byte-genau für **gedruckten** Text. Handschrift,
> vergilbte oder stark degradierte Scans sind der **Rand** (< 25 %), an dem das
> byte-genaue LSTM prinzipbedingt schwächelt. Dieses Dokument entwirft die
> Eskalation für genau diesen Rand: den bereits vorhandenen
> `quality.low_confidence`-Trigger nutzen, um die betroffene Region an ein
> starkes Transformer-/VLM-OCR-Modell — **chandra-ocr-2** — weiterzureichen,
> ausgeführt in Rust via **candle / burn / ort** (oder langfristig über den
> **bgz-tensor**-Codec). Entwurf, kein bestehender Code.

## 1. Der Trigger existiert schon

`doc.v1` trägt bereits das Ehrlichkeits-Signal:

```json
"quality": { "mean_conf": 96.1, "low_confidence": false }
```

`low_confidence = true` warnt, dass die Eingabe wahrscheinlich Handschrift /
niedrig aufgelöst / kein gedruckter Text ist. Zusätzlich liegt pro Wort/Region
eine Konfidenz vor. Das ist die **Routing-Entscheidung**: Stufe 1 (tesseract-rs)
läuft immer zuerst; nur wo die Konfidenz unter eine Schwelle fällt, wird diese
Seite/Region an Stufe 2 eskaliert. Der schnelle, reine Rust-Pfad bleibt der
Default; das schwere Modell zahlt nur der Rand.

## 2. Das Eskalationsziel — chandra-ocr-2

`datalab-to/chandra-ocr-2` (Hugging Face), Stand des Modell-Steckbriefs:

- **Art:** Vision-Language-Modell (VLM) auf Transformer-Basis (`AutoModelForMultimodalLM`).
- **Größe:** **5 Mrd. Parameter**, BF16. Gewichte als **safetensors**.
- **Eingabe/Ausgabe:** Bilder + PDFs → Markdown / HTML / JSON **mit** Layout-Info.
- **Stärken (laut Steckbrief):** „exzellente Handschrift-Unterstützung", Tabellen,
  Mathematik, komplexe Layouts, 90+ Sprachen, Formular-/Checkbox-Rekonstruktion,
  Bild-/Diagramm-Extraktion mit Beschriftung; betont starke Leistung auf
  schlechten Scans / alten Dokumenten (85,8 % olmOCR-Bench).
- **Lizenz:** modifizierte **OpenRAIL-M** — „frei für Forschung, private Nutzung
  und Startups < 2 Mio. $ Umsatz/Förderung", darf nicht mit ihrer kommerziellen
  API konkurrieren. **Vor kommerziellem Einsatz die Lizenz prüfen.**

Genau das Profil für den Rand: dort, wo das byte-genaue LSTM aufgibt (Handschrift,
Degradation), ist ein VLM deutlich überlegen.

## 3. Rust-Laufzeiten — die Optionen und ihr Preis

| Laufzeit | Charakter | Passung zur „reines Rust / kein C"-Invariante | Notiz |
|---|---|---|---|
| **candle** | HuggingFaces Rust-ML, safetensors-nativ | am nächsten am reinen Rust (candle-core); GPU via cuda/metal-Feature | kann VLMs laufen lassen; 5B in BF16 ≈ ~10 GB Gewichte → GPU empfohlen |
| **burn** | reines Rust, Multi-Backend (wgpu/candle/ndarray/tch) | rein-Rust-treu | VLM-Reife je nach Backend unterschiedlich; wgpu-Backend browsertauglich |
| **ort** | ONNX-Runtime-Rust-Bindings | **bricht** die Invariante (ONNX Runtime ist C++) | am schnellsten, wenn chandra nach ONNX exportiert wird; pragmatisch, nicht rein |
| **bgz-tensor** | Workspace-Codec „Attention als Tabellen-Lookup" | rein Rust, aber Modell muss erst gebacken werden | kein Nah-Ziel: chandra in das Base17/Palette-Format zu backen ist ein Forschungsaufwand (vgl. die Qwen-bgz7-Shards) |

Praktische Reihenfolge: **candle** (rein-Rust-treu, safetensors direkt) für einen
sauberen ersten Wurf; **ort** wenn ONNX-Tempo auf der GPU zählt und der C-Dep
akzeptiert wird; **bgz-tensor** ist der *Horizont* („später für O(1)-Inferenz
backen"), nicht der erste Schritt — die Philosophie des Workspaces
(Transformer-Inferenz → Lookup), aber für chandra noch nicht gebahnt.

## 4. Wo die Eskalation architektonisch sitzt — optional & getrennt

**Nicht** im Kern-Recognizer. Der bleibt schlank, rein Rust, standalone, ~4 MB
Binary + Modell. Die chandra-Stufe ist eine **optionale, feature-gegatete**
Fähigkeit oder ein **separater Dienst**:

- **Feature-gegated im selben Prozess:** `--features chandra-escalation` zieht
  candle/burn + die Gewichte nach; ohne das Feature ist die Pipeline unverändert
  schlank.
- **Separater Dienst (empfohlen bei GPU):** ein eigener Inferenz-Dienst
  (candle/ort auf GPU), den der Consumer nur für Regionen mit `low_confidence`
  aufruft. Der Kern-Web-Demo-Binary bleibt CPU-only und leicht.

Das Routing ist damit sauber: `recognize_document` (Stufe 1) → wo
`low_confidence` → die Region an chandra (Stufe 2) → dessen Markdown/JSON in
`doc.v1`-Regionen zurückfalten (die Naht bleibt `doc.v1`).

## 5. Ehrliche Abgrenzung

- **Zwei Philosophien, bewusst getrennt.** tesseract-rs ist **byte-genau
  beweisbar** (jedes Blatt vs. libtesseract). chandra ist ein **Black-Box-VLM** —
  kein Byte-Parity-Beweis möglich, kein deterministisches Orakel. Es ist die
  **Notausfahrt für den Rand**, nicht der Kern. Beide nicht vermischen.
- **Gewicht & Compute.** 5B Parameter ≈ ~10 GB BF16; GPU praktisch nötig für
  brauchbare Latenz. Das bricht „Single-4-MB-Binary/standalone" — deshalb
  optional/getrennt.
- **Latenz.** Ein VLM ist um Größenordnungen langsamer als das LSTM — deshalb
  nur der `low_confidence`-Rand, nie der Default-Pfad.
- **C-Abhängigkeit bei ort.** ONNX Runtime ist C++; wer die „kein C zur
  Laufzeit"-Invariante halten will, nimmt candle/burn.
- **Lizenz.** OpenRAIL-M mit Umsatz-/Wettbewerbsklausel — vor jedem kommerziellen
  Einsatz prüfen; nicht ohne Rechtsklärung in ein Kundenprodukt geben.

## 6. Fazit

Die Eskalation ist ein **überzeugendes „Zwei-Stufen-OCR"**: schneller,
byte-genauer, reiner Rust-Default für gedruckten Text; ein starkes VLM
(chandra-ocr-2 via candle) für den handschriftlichen / degradierten Rand,
ausgelöst durch das schon vorhandene `low_confidence`-Signal — **optional,
getrennt, den Kern nicht verunreinigend.** Erster konkreter Schritt (wenn
gewünscht): eine kleine `chandra-escalation`-Feature-Krate mit candle, die eine
Region → Markdown/JSON macht, plus die Rückfaltung in `doc.v1`; danach messen,
ob sich der Rand-Gewinn gegen Compute/Latenz lohnt.
