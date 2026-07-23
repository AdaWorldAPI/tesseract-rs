# tesseract-rs — Unterschiede zum Upstream

> Deutschsprachige Dokumentation. Was `AdaWorldAPI/tesseract-rs` gegenüber dem
> Upstream-`tesseract-rs` (dem libtesseract-FFI-Binding) leistet — und was in
> der jüngsten Iteration hinzugekommen ist.

## 1. Überblick — Transcode statt Binding

`AdaWorldAPI/tesseract-rs` ist **kein FFI-Wrapper** um Tesseracts C++-Bibliothek,
sondern ein **reiner Rust-Transcode** von Tesseract OCR. Das ist der grundlegende
Unterschied zum Upstream-`tesseract-rs` (dem antimatter15-Wrapper
`tesseract-sys` / `tesseract-plumbing`), der libtesseract nur über eine dünne
FFI-Schicht aufruft.

Bei uns wird die OCR **Blatt für Blatt** in Rust nachgebaut, und **jedes Blatt
wird byte-genau gegen das C++-Original bewiesen, bevor es landet**. Zur Laufzeit
läuft **kein C++**: libtesseract und leptonica werden ausschließlich als
Vergleichs-Orakel beim Beweis gelinkt, niemals im Rust-Pfad.

## 2. Der fundamentale Unterschied

| Aspekt | Upstream `tesseract-rs` (Binding) | AdaWorldAPI/tesseract-rs (Transcode) |
|---|---|---|
| Art | FFI-Wrapper um libtesseract (C++) | reiner Rust-Transcode |
| Laufzeit-Abhängigkeiten | libtesseract + leptonica (System-Bibliotheken) | keine C-Bibliotheken — nur der Rust-Binary |
| Speichersicherheit | C++ unter der Haube (`unsafe` FFI) | sicheres Rust, byte-parity-geprüft |
| Prüfbarkeit | Blackbox | jedes Blatt byte-genau gegen C++ bewiesen |
| Deployment | braucht die C-Bibliotheken im Image | Single-Binary (~4 MB Modell + glibc-Binary) |
| Modelle | was libtesseract eben lädt | **model-agnostisch, bewiesen für `eng` UND `deu`** |
| Ausgabe | Text / hOCR / TSV / PDF (von libtesseract) | dieselben Formate, plus `doc.v1` (Regionen/Tabellen/Felder) |

## 3. Byte-Parität — die bewährte Methode

Zweistufig (die „Zwei-Schritt-Methode"):

1. **Das Orakel installieren.** libtesseract **5.3.4** + leptonica **1.82.0**
   (per `apt`), dazu die passenden **5.3.4-Quellen** für die internen Header →
   **null ABI-Skew** (Bibliothek und Header derselben Version).
2. **Transcodieren + byte-genau vergleichen.** Dasselbe Feld auf beiden Seiten
   ausgeben, `diff`. Die Bijektions-Hälfte (id↔unichar) validiert das
   Objekt-Layout selbst, bevor einer Feld-Hälfte vertraut wird.

Kein Blatt landet, bevor sein `diff` gegen libtesseract/leptonica **exakt 0** ist.

## 4. Neuerungen dieser Iteration

### 4.1 eng + deu Byte-Parität — der Transcode ist model-agnostisch

Jedes bisher auf `eng` bewiesene Blatt ist jetzt auch auf dem **deutschen**
Modell (`deu.lstm`) byte-identisch:

| Feature | eng | deu | Falsifikator (warum das Grün echt ist) |
|---|---|---|---|
| UNICHARSET (Bijektion + 5 Felder) | 112 | 116 | Mehrbyte-Zeichen Ä Ö Ü ä ö ü ß byte-identisch |
| UNICHAR UTF-8-Codec | 268 | modell-unabhängig | Overlong-NUL, illegale Lead-Bytes |
| Recoder (encode/decode/beam) | ✅ | ✅ | code_range 111 vs 115; geteilter Code id1→2 |
| Netzwerk-Forward (Softmax) | 8/8 | 8/8 | deu nw=400979 vs eng 385807 (andere Architektur) |
| **Bild → Text (End-to-End)** | 6/6 | 6/6 | deu null_char=114 vs eng 110 |
| Wörterbuch/DAWG-Walk | 14/14 | 14/14 | deutscher Trie: über / schön / ß, Zahlen |
| Beam-Decode | 2/2 | 2/2 | exakte IEEE-754-f32-Bits |
| **Sauvola (neu)** | 5/5 | modell-unabhängig | echte 512×720-Seite + LUT-Pfad |

**Der Falsifikator:** Das deutsche Modell leitet **andere** kritische Konstanten
aus sich selbst ab — Zeichensatz 116, code_range 115, null_char 114, 400979
Netzwerk-Gewichte — und der reine Rust-Code reproduziert **jede einzelne
byte-genau**. Eine Pipeline, die `eng`-Werte fest verdrahtet hätte, wäre auf
`deu` abgewichen. Sie tat es nicht. Damit ist bewiesen, dass die Loader die
**Datei lesen**, nicht eine gelernte `eng`-Form erinnern.

### 4.2 Sauvola — adaptive Binarisierung (neues leptonica-Blatt)

Die komplette `pixSauvolaBinarize`-Kette, transcodiert aus der
`AdaWorldAPI/leptonica`-Quelle (`src/{binarize.c,convolve.c,pix2.c}`) nach
`crates/tesseract-ocr/src/binarize.rs`:

```
pixAddMirroredBorder(whsize+1)
  → pixWindowedMean       (u32-Integralbild, wrapping)
  → pixWindowedMeanSquare (f64-Integralbild)
  → Schwelle  t = m·(1 − k·(1 − s/128)),  s = √(ms − m²)
  → grau < t  ⇒ Vordergrund (schwarz)
```

Byte-identisch zu leptonica 1.82.0 über Fenstergrößen 4–15, k = 0.2–0.5, den
LUT-Pfad (`w·h > 100000`) und eine echte 512×720-Seite (368 640 Pixel).

**Warum es zählt:** Die bisherige Layout-Binarisierung ist globales Otsu, das
ungleichmäßig beleuchtete oder vergilbte Scans zerstört (die „ImproveQuality"-
Lehre). Sauvola ist die adaptive Alternative — eine Schwelle **pro Pixel** aus
lokalem Mittelwert und Standardabweichung, sodass eine schattige Ecke ihre
eigene Schwelle behält, statt komplett schwarz zu werden.

## 5. Die vollständige OCR-Kette (was transcodiert und byte-parity-bewiesen ist)

- **Zeichensatz-Schicht:** UNICHARSET (id↔unichar, Eigenschaften, Skript,
  other_case, Richtung, Spiegel), UNICHAR-UTF-8-Codec, Recoder
  (`UNICHARCOMPRESS`).
- **Rechenkern (Compute-Tier, deps `ndarray`):** `matrix_dot_vector` (int8-SIMD
  aus `ndarray::simd_runtime`), `WeightMatrix`, Aktivierungen (tanh/logistic/
  relu/clip/softmax), `FullyConnected::Forward`, `LSTM::Forward` (1-D int8,
  quantisierte Rekurrenz), Graph-Walk (Series/Reversed/Parallel).
- **Decodierung:** `RecodeBeamSearch::Decode` (nicht-Wörterbuch-CTC-Beam),
  Recoder-Beam-Maps (`SetupDecoder`).
- **Netzwerk-Loader:** `Network::from_le_bytes` (lädt das echte `eng.lstm`/
  `deu.lstm` in einen lauffähigen Knotenbaum), `LstmRecognizer::from_components`.
- **Bild-Frontend:** `from_grey_pix` (Pixel → int8-Gitter), `recognize_grid`
  (Gitter → Text), `recognize_image_file` (PGM → Text), allgemeine
  `pixScale`-Höhenskalierung.
- **Layout:** `pixGetRegionsBinary` (Regions-Klassifikator: Text/Bild/Tabelle),
  `pixDecideIfTable` (Tabellen-Erkennung), XY-Cut-Segmentierung,
  `structured.rs` → **`doc.v1`** (Regionen + Tabellen-Zellgitter + Felder +
  Qualität).
- **Binarisierung:** globales Otsu (Layout) **+ neu: Sauvola** (adaptiv).

## 6. Wahrheitsgetreue PDF-Ausgabe

### Variante A — durchsuchbares PDF (vorhanden, `render_searchable_pdf`)

Zeichnet den **Original-Scan** als Vollseiten-Bild und legt eine **unsichtbare,
pixelgenaue OCR-Textschicht** darüber (PDF-Rendermodus 3). Reines Rust
(`lopdf`), keine C-Bibliothek. „Bild UND Text wahrheitsgetreu" im stärksten
Sinn: das Bild **ist** der exakte Scan, der Text ist die exakte OCR — durchsuchbar
und kopierbar.

### Variante B — strukturiertes PDF (in Arbeit)

Die rekonstruierende Variante: echte, auswählbare Textblöcke + zugeschnittene
Abbildungen + gezeichnete Tabellen aus `doc.v1`. A und B rendern aus **einem
gemeinsamen Layout-Modell**, aus dem auch eine HTML-Vorschau erzeugt wird — so
sind **Vorschau ≡ PDF per Konstruktion** (Klickwege-Parität). B ist das
Fundament, mit dem Konsumenten Graph → Vorlage → PDF materialisieren (z. B.
Rechnungserstellung).

## 7. Ehrliche Grenzen (noch nicht transcodiert / bewusst zurückgestellt)

- **Sprach-Erkennung (OSD)** ist noch nicht transcodiert — geladen wird das
  gewählte Modell. Die ehrlichen Signale sind Modell + `mean_conf` +
  `low_confidence`, keine erfundene Sprach-Konfidenz.
- **Deskew-Welle** (`pixFindSkew` + Rotation), **Wörterbuch-Beam-
  Genauigkeitsschicht (C1)**, **CJK-Trie (C3)**, **2-D-LSTM / Softmax-LSTM-
  Pfade** — zurückgestellt (`eng.lstm`/`deu.lstm` sind 1-D nicht-Softmax).
- **bbox/stats-Unterfeld** — gated auf ein Legacy-`eng.unicharset` mit echten
  bbox/stats-Werten (auf dem LSTM-Unicharset sind sie uniform und daher kein
  Falsifikator).

## 8. Deployment — kein C zur Laufzeit

Weil der Erkennungspfad reines Rust ist (Bild-Dekodierung und TLS ebenfalls),
ist das Laufzeit-Image nur der glibc-Binary + ~4 MB `corpus/model`. Keine
libtesseract, keine leptonica zur Laufzeit — diese sind ausschließlich
Link-Abhängigkeiten des **Orakels** beim Paritätsbeweis, nie im Rust-Pfad.

---

*Diese Datei ist die deutschsprachige Zusammenfassung. Der maßgebliche,
fortlaufend gepflegte Stand steht in `CLAUDE.md` (Abschnitt „What's shipped")
und im Paritäts-Tracker `.claude/harvest/PARITY-ENG-DEU-STATUS.md`.*
