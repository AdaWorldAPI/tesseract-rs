# Plan — Sink the Tesseract `Network` layer graph via ruff→OGAR onto V3 SoA

**Status:** Leaf 1 (base header) EXECUTED + byte-parity GREEN. Follow-ups queued.
**Branch:** `claude/happy-hamilton-0azlw4` (lance-graph + ruff + tesseract-rs).
**Operator directive:** *"use new V3 substrate AR rail shaped (6x8:8) or 4x3 or
3x4 (16 bytes tenant, classid +12 bytes) use C#/cpp ruff>OGAR transpiler sink-in
substrate."*

## The shape (why this, not a hand-rolled enum)

Tesseract's recognizer is a polymorphic tree of `Network` subclasses. The wrong
move (rejected earlier this arc) was a hand-rolled `enum NetworkKind` with its own
dispatch — a **parallel object model**, the exact Core-First anti-pattern. The
right move, executed here:

1. **Harvest** the class graph via the `ruff→OGAR` C++ transpiler (Pipeline 1:
   `ruff_cpp_spo` libclang walker → SPO manifest), NOT hand-transcribe it.
2. **Sink** each node onto the existing V3 SoA primitive
   `lance_graph_contract::facet::FacetCascade` (16 B = `classid(4) | 6×(8:8)`),
   read under `CascadeShape::G6D2` (the operator's "6x8:8").
3. **Dispatch** by `classid` (`network_layer` canon + `NetworkType` ordinal in the
   custom-low half) — the `invoke_network` keystone, the `invoke_unicharset` analog.

Identity = classid; state = SoA tenant; structure ≠ compute (the `Forward`/weight
math is `tesseract-recognizer`, deps ndarray; the layer GRAPH is the Core).

## Executed

### The harvest (ruff → OGAR sink-in)
- `ruff/crates/ruff_cpp_spo/examples/harvest_network.rs` — walks the 11 network
  layer headers via libclang, emits the `has_function`/`inherits_from`/
  `virtually_overrides` SPO manifest. On real Tesseract 5.5.0 src: **62 classes,
  5060 triples** → `/tmp/network_manifest.ndjson`. The `Forward` override set
  (FullyConnected/LSTM/Series/Parallel/Convolve/Maxpool/Reversed/Reconfig/Input) =
  the compute-leaf list; the `DeSerialize` override set (FullyConnected/LSTM/
  Plumbing/Convolve/Maxpool/Reconfig/Input) = the binary-leaf list. This IS the
  `classid → ClassView` method-resolution manifest. Committed to ruff.

### The base-header leaf (Core-side, byte-parity proven)
- `lance-graph-contract/src/network.rs` — `NetworkType` (27 types, ordinal ==
  discriminant, `kTypeNames` on-wire strings) + `NetworkHeader::from_le_bytes`
  (the shared base header EVERY layer serializes, `network.cpp:214-248`) +
  `to_facet()` (the G6D2 sink) + `NetworkType::classid()` (the dispatch seed).
- `ogar_codebook`: `network_layer` = `0x0804` (ONE container mint in the 0x08 OCR
  domain; the 27 subclasses live in the classid custom-low, not 27 slots).
- **Byte-parity GREEN** on real `/tmp/eng.lstm`: Rust `NetworkHeader::from_le_bytes`
  == libtesseract `Network::CreateFromFile` for the outer node —
  `Series ni=36 no=111 num_weights=385807 name=Series` — with the oracle's
  `spec()` == the model spec string `[1,36,0,1[C3,3Ft16]Mp3,3TxyLfys48Lfx96Rx
  Lrx96Lfx192Fc111]` (the known-answer self-check guarding the 5.5.0-hdr/5.3.4-lib
  ABI skew). Example: `network_dump.rs`; oracle: `/tmp/network_spec_oracle.cpp`
  (built `-DFAST_FLOAT`). The facet `0x08040009` decodes losslessly:
  ni=36, no=111, flags=192, num_weights=385807 (tiers 3-4), lifecycle=0.

## The G6D2 sink (network_layer ClassView projection)

| tier | 8:8 u16 | field |
|---|---|---|
| 0 | ni | inputs |
| 1 | no | outputs |
| 2 | network_flags & 0xFFFF | behaviour flags |
| 3 | num_weights lo16 | cumulative weight count (lo) |
| 4 | num_weights hi16 | cumulative weight count (hi) |
| 5 | training : needs_backprop | lifecycle (lo:hi) |

`facet_classid = compose_classid(NETWORK_LAYER, ntype)`. The **name** and the
**weight blob** are out-of-line (`I-VSA-IDENTITIES`: the facet is identity + typed
dims; content is a keyed store / Lance column). `num_weights` COUNT rides tiers 3-4;
the blob does not.

## Deferred (the follow-up leaves)

1. **Per-subclass payload parse + tree recursion.** The base header proves the
   shared prefix; each subclass's `DeSerialize` then reads its payload (Plumbing =
   child `Network*` vector → `EdgeBlock`; FullyConnected/LSTM = `WeightMatrix`
   blobs → out-of-line Lance column; Convolve/Maxpool = kernel dims → tier5). The
   full tree walk reproduces the whole `[…]` spec. Falsifier: per-node
   `(type, ni, no)` diff vs an oracle that walks `Plumbing::children()`.
2. **`invoke_network` keystone** (classid → ClassView → subclass DeSerialize/
   Forward dispatch) — designed, unblocked (the generic dispatch is proven by
   E-CPP-KEYSTONE-1 / the recoder's `invoke_recoder`), not yet built.
3. **Recognizer compute leaves** (`tesseract-recognizer`, deps ndarray): Leaf 4
   `FullyConnected::Forward` (Leaf 2 WeightMatrix × Leaf 3 activation), Leaf 5
   `LSTM::Forward` (gates + cell/hidden state), then `Series`/`Parallel` graph
   walk → `recodebeam` (CTC) → the code lattice `recoded_to_text` eats. The
   compute-side binary reader (`crates/tesseract-recognizer/src/io.rs`,
   `ByteReader`) is written, awaiting the recognizer-side Network loader.
4. **recoder beam maps** (`GetNextCodes`/`GetFinalCodes`/`is_valid_start_`) needed
   by recodebeam — deferred in the recoder leaf.

## Iron rules honored
- Consumed the Core (`FacetCascade`, `compose_classid`, `0x08` OCR domain); no
  parallel object model.
- `ruff_cpp_spo` corpus stays UPSTREAM (never vendored).
- Structure (Core) vs compute (`tesseract-recognizer`, ndarray) split intact.
- Board hygiene: EPIPHANIES `E-OCR-NETWORK-SINK-1` + LATEST_STATE (lance-graph).
