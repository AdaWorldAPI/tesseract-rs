# tesseract-rs — AST-DLL Codegen v1 (the adapter-body half)

> **Type:** plan (the codegen blueprint — the "second half" of the Core-First
>   Transcode Doctrine; the SPO harvest is the first half, shipped as
>   `ruff_cpp_spo` PR #17).
> **Status:** CONJECTURE, floored by `PROBE-OGAR-ADAPTER-UNICHARSET`. Nothing
>   here is a FINDING until one unicharset leaf adapter hits byte-parity through
>   a `classid → ClassView` (doctrine §"The falsifier"). Do NOT scale across
>   modules before that probe is green.
> **Reads-into contract:** `tesseract-rs-ast-dll-codegen-v1` emits into the
>   receive-contract pinned by `tesseract-rs-receive-contract-v1.md` (corpus pin
>   `tesseract-ocr/tesseract@5.5.0`; `src/generated/`; three-SHA provenance
>   header; `transcode` feature; deterministic committed output; zero vendored
>   C++). That doc is the *target*; this doc is the *generator*.
> **Governing spec:** `lance-graph/.claude/knowledge/core-first-transcode-doctrine.md`
>   (Core-First inversion; OGAR assume-contract; v1→v2→v3 ladder; leaf-vs-hand-port;
>   iron guard; falsifier). This is v1 of that ladder.
> **Grounded against (read, not guessed):**
>   - `ruff/crates/ruff_spo_triplet/src/triple.rs` — `Triple{s,p,o,f,c}`, closed
>     53-predicate vocab (17 C++), `Predicate::from_str` (closed-vocab gate).
>   - `ruff/crates/ruff_spo_triplet/src/ndjson.rs` — wire = one `{s,p,o,f,c}`
>     JSON object per line; `from_ndjson` fails loud on unknown predicate.
>   - `ruff/crates/ruff_spo_triplet/src/expand.rs` — `cpp_method`/`cpp_field`/
>     `cpp_base`/`cpp_template`/`cpp_friend` exact triple shapes.
>   - `ruff/crates/ruff_cpp_spo/src/{lib.rs,clang_walker.rs}` — walker capture +
>     iron rule (C++ corpus stays UPSTREAM, never vendored).
>   - `lance-graph/crates/lance-graph-contract/src/canonical_node.rs` +
>     `class_view.rs` — the OGAR Core the adapters TARGET.

---

## §1 — Input: the harvest ndjson, grouped per class

### 1.1 What the harvester hands us

The generator consumes a **single ndjson file** (the deterministic
`to_ndjson(expand(extract_tree(corpus)))` artifact). Each line is one triple:

```json
{"s":"cpp:Tesseract::Recognizer.Recognize(int)","p":"returns_type","o":"int","f":0.95,"c":0.82}
```

Load it through `ruff_spo_triplet::from_ndjson` (NOT a hand-rolled parser) so the
generator inherits the **closed-vocab gate for free**: any `p` outside the 53
predicates is a hard error at line N, exactly as the harvester intends. `f`/`c`
(NARS truth) are **ignored by codegen** — they gate downstream SPO queries, not
adapter shape. The generator keys only on `(s, p, o)`.

### 1.2 Reassembling a class surface from flat triples

The triple stream is sorted by `(s,p,o)` and de-duplicated (guaranteed by
`expand`). Reassembly is a deterministic two-pass group-by, NO graph library:

**Pass A — partition by class IRI.** Every C++ subject is either a class IRI
(`cpp:NS::Class`) or a method/field IRI (`cpp:NS::Class.member(...)` /
`cpp:NS::Class.field`). The class IRI is the **longest `cpp:`-prefixed prefix of
the subject that ends before the first `.` following the namespace-qualified
class name.** Practically: the class node is identified by its own
`(cpp:NS::Class, rdf:type, ogit:ObjectType)` triple; every other subject whose
string starts with `cpp:NS::Class.` belongs to that class. Bucket the triples
into `BTreeMap<ClassIri, Vec<Triple>>` (BTreeMap → deterministic class order,
matching the harvester's own cross-TU dedup).

> **Parsing caveat (feed back to harvester — see §6 GAP-1):** the method IRI
> `cpp:NS::Class.method(params)` mixes two `::`-and-`.` namespaces in one string.
> A class `cpp:tesseract::Foo` with method `bar(int)` is
> `cpp:tesseract::Foo.bar(int)`. Splitting "class vs member" requires matching
> the `(rdf:type, ogit:ObjectType)` class anchors FIRST, then attributing
> members by string-prefix against the known class set — never by naive
> rightmost-`.` split (a param type like `const std::pair<A,B>&` contains no `.`
> but a default-arg or nested type could; and the class name itself contains
> `::` not `.`). The anchor-first rule is robust; document it as the contract.

**Pass B — within a class bucket, group by member IRI.** For the class's own
node, collect class-scoped predicates (`inherits_from`, `has_field`,
`template_specialises`, `template_instantiates`, `is_friend_of`,
`static_asserts`, `has_function`). For each distinct member IRI
`cpp:NS::Class.member(params)`, collect its method-scoped predicates
(`rdf:type`, `returns_type`, `has_param_type`, `is_const`, `is_static`,
`is_noexcept`, `is_pure_virtual`, `is_constexpr`, `virtually_overrides`,
`defines_operator`, `requires_concept`). The result is an in-memory
`ClassSurface { iri, bases, fields, methods, templates, friends, static_asserts }`
— a *reconstruction* of the harvester's `CppClass`, rebuilt from triples, never
a re-parse of C++.

This reconstruction is the generator's only IR. It carries no state the triples
don't carry; it is a grouping, not a model.

---

## §2 — Per-method adapter shape (signature reconstruction)

### 2.1 The triple set → a Rust adapter signature

For one method IRI, the signature is rebuilt from its predicate set:

| Triple | Drives |
|---|---|
| `has_param_type` `"<i>:<type>"` (one per param) | the ordered parameter list — **sort by the leading integer `i`**, then map `<type>` to a Rust param type |
| `returns_type` `"<type>"` (≤1, absent ⇒ void/ctor/dtor) | the Rust return type (`-> T`); absent ⇒ `()` |
| `is_const` `"true"` | read-accessor classification (→ `&self` receiver shape, never `&mut`) |
| `is_static` `"true"` | class-level (no receiver; free associated fn) |
| `is_noexcept` `"true"` | annotation only — Rust has no `noexcept`; record in a doc-comment, no `Result`-elision decision here |
| `is_pure_virtual` `"true"` | **NOT a body** — pure-virtual = no adapter body to generate; see §2.4 |
| `is_constexpr` `"constexpr"\|"consteval"` | candidate for a `const fn` adapter (leaf-only; if the body is hand-port, drop the marker) |
| `defines_operator` `"operator=="` | the method is an operator; map to the Rust trait impl shape (`PartialEq` etc.) where the leaf rule allows, else a named adapter `op_eq` |
| `requires_concept` `"<clause>"` | doc-comment only at v1 (C++20 concept → Rust trait bound is a hand-judgement, not mechanical) |

**Worked example (the locked-shape fixture, `ruff_cpp_spo` lib.rs):**

```
cpp:Tesseract::Recognizer.Recognize(int)  has_param_type   0:int
cpp:Tesseract::Recognizer.Recognize(int)  returns_type     int        (in the walker variant; void-returning in the locked fixture)
cpp:Tesseract::Recognizer.Recognize(int)  is_noexcept      true
cpp:Tesseract::Recognizer.Recognize(int)  virtually_overrides  cpp:Tesseract::Classify.Recognize(int)
```

→ generated adapter (shape, not final body — body routing is §4):

```rust
/// @generated adapter for cpp:Tesseract::Recognizer.Recognize(int)
/// noexcept; overrides Tesseract::Classify::Recognize(int)
pub fn recognize(/* &self via ClassView receiver */ , p0: i32) -> i32 { /* §4 body */ }
```

The C++→Rust **type mapping** (`int`→`i32`, `const Image &`→`&Image`,
`std::unique_ptr<LSTMRecognizer>`→`Box<LSTMRecognizer>`, …) is a generator-owned
lookup table seeded for the unicharset/recoder/dawg leaf surface and extended
deliberately. Unmapped types are a **hard generator error**, never a silent
`todo!()` — an unmapped type means either a leaf that isn't actually a leaf
(route to hand-port, §4) or a missing table entry (extend the table). This
mirrors the doctrine's iron guard at the type-mapping layer.

### 2.2 `(params)` IRI suffix keeps overloads distinct — load-bearing

The harvester appends `(<comma-joined-param-types>)` to every method IRI
precisely so `int process(int)` and `int process(double)` land on **separate
nodes** (`...process(int)` vs `...process(double)`), each with its own
`returns_type`/`has_param_type` set. The generator MUST preserve this: each
distinct member IRI → one adapter. Two overloads → two adapters with
disambiguated Rust names (`process_int` / `process_double`, derived
deterministically from the param-type suffix). **Never collapse overloads** —
that is the exact failure the harvester's per-overload IRI (codex P2 #17)
exists to prevent, and collapsing would silently drop one overload's body.

### 2.3 Ordered params from an unordered triple set

`has_param_type` objects carry a leading 0-based index (`0:int`, `1:const Image &`)
so a *set* of triples preserves arity + order. The generator parses the integer
prefix, sorts ascending, and rejects (hard error) any gap or duplicate index for
one method — a malformed signature is a harvester bug, surfaced loud, not
patched over.

### 2.4 Pure-virtual / abstract methods produce no body

`is_pure_virtual = "true"` means the C++ method has `= 0` — no implementation to
transcode. The generator emits no adapter body for it; instead it is recorded in
the class's ClassView manifest as an **interface slot the derived class's adapter
must fill** (this is where `virtually_overrides` does its work — §3). A pure
virtual with no overrider in the corpus is a `// no concrete adapter` note, not
an error.

---

## §3 — Composition: `classid → ClassView` from `has_function`; MRO from `virtually_overrides`

### 3.1 No parallel object model, no new `ValueSchema` variant

The doctrine's hardest rule: composition/inheritance is **`classid → ClassView`**,
which already exists (`class_view.rs`, PR #498). The generator does NOT emit a
struct hierarchy, vtables, or an MRO engine. It emits:

1. **One `classid` per harvested class** — minted into the OGAR key space
   (`NodeGuid`; the bootstrap default-basin `identity`-only address suffices for
   v1, `is_bootstrap_address()`). The classid is the adapter set's identity; the
   adapter does NOT carry type identity (doctrine assume-contract row 1).
2. **A `ClassView` impl** whose `fields(class)` returns the class's ordered
   `&[FieldRef]` (bit basis), built from the class's `has_field` + signature
   triples. `FieldRef{predicate_iri, label}` — `predicate_iri` is the harvested
   field/method IRI, `label` the bare member name. This is the
   *method-resolution manifest* the doctrine names: `has_function` lists *which*
   adapters the classid's ClassView composes.
3. **No `value_schema` override at v1** — every generated class inherits the POC
   blanket default `ValueSchema::Full` (`class_view.rs:value_schema` /
   `ReadMode::DEFAULT`). A leaf adapter's state maps onto existing
   `ValueTenant`s (§3.3); if it fits, no new column, no override. Specialisation
   to a smaller preset is a later, opt-IN memory optimization — never required
   for correctness.

### 3.2 `virtually_overrides` drives the resolution order

`virtually_overrides` objects are **fully-qualified base-method IRIs with the
`(params)` suffix** (`cpp:Tesseract::Classify.Recognize(int)`), so they join the
*exact* base overload, not just any same-named method. The generator reads them
to build the method-resolution manifest:

- For derived class `D` with method `m(params)` carrying
  `virtually_overrides cpp:NS::B.m(params)`, the ClassView for `D`'s classid
  records that `D::m` **shadows** `B::m` in resolution order. `inherits_from`
  edges give the base-class chain; `virtually_overrides` gives the per-method
  shadowing within that chain. Together they are the MRO — **expressed as
  ClassView composition, not a generated vtable**.
- A non-overriding inherited method (in `B`, not shadowed in `D`) resolves to
  `B`'s adapter via the `inherits_from` chain — `FieldMask::inherit` (the
  `subClassOf` union, `class_view.rs`) is the substrate for "child carries
  parent's fields plus its delta."

### 3.3 Where a method's state lives (assume-contract, concretely)

A leaf adapter's inputs/outputs map onto the OGAR Core's movable parts — the
adapter assumes them, does not re-implement them:

- **identity** → `classid` (the `NodeGuid` key). Adapter carries none.
- **state** → SoA `ValueTenant` columns in `NodeRow::value` (480 B slab):
  `Fingerprint` (32 B), `EntityType` (u16), `Meta`/`Qualia`, codec residues,
  etc. A unicharset id↔utf8 table, a recoder codebook, a dawg node array — these
  are *read from* value-tenant columns, never owned by the adapter struct.
- **relations** → `EdgeBlock` (12 in-family + 4 out-of-family). Adapter does not
  build a pointer web.
- **invocation** → `UnifiedStep` via `OrchestrationBridge`
  (`lance-graph-contract::orchestration`). The adapter is a fn the ClassView
  dispatches to, not a method bolted onto a god-object.

If a leaf method needs state that does NOT fit an existing `ValueTenant`, that is
a **Core gap → EXTEND-CORE** (file it, route to `core-gap-auditor`), NEVER an
adapter that grows its own field. This is the iron guard and the line v1 must not
cross.

---

## §4 — Leaf-vs-hand-port split (the routing decision)

The doctrine's scope boundary (§"Scope boundary") draws the line; the generator
enforces it per class/method. Routing is decided at codegen time and recorded in
the generated `mod.rs` so the split is auditable.

### 4.1 Codegen (thin DO-in/out adapters) — the leaf surface

Mechanical, data-shaped, value-in/value-out methods on Tesseract's utility
plane:

- **unicharset** (`src/ccutil/unicharset.{h,cpp}`) — `unichar_to_id` /
  `id_to_unichar` / size / properties. Table lookups over value-tenant columns.
  **This is the falsifier's subject (§5).**
- **recoder** (`src/ccutil/unicharcompress.*`) — encode/decode over a fixed
  codebook. Pure table transforms.
- **dawg** (`src/dict/dawg.*`) — membership / edge-walk over a node array.
  Read-only graph lookup expressible as value-tenant reads + EdgeBlock walks.

These satisfy all four assume-contract rows cleanly: identity=classid,
state=value tenants, relations=EdgeBlock, invocation=UnifiedStep. They are what
`PROBE-OGAR-ADAPTER-UNICHARSET` measures.

### 4.2 Hand-port (raw-pointer, NOT adapters) — the intrusive core

The doctrine forbids forcing these into the DO-adapter mold — that is the
**Frankenstein flattening** (`frankenstein-checklist.md`):

- **LSTM / recodebeam** (`src/lstm/*`, `src/ccstruct/recodebeam.*`) — BiLSTM
  numeric kernels, stateful beam search. Numeric hot loops, not table lookups.
- **leptonica** boundary — image buffers, C pointer ownership. Stays behind FFI
  (the existing `tesseract-sys` wrapper, receive-contract §3).
- **ELIST / CLIST** (`src/ccutil/elst*.h`, `clst*.h`) — intrusive doubly-linked
  lists with raw-pointer mutation. The canonical "do not flatten" case.

For these the generator emits **nothing** (or, at most, a `// HAND-PORT:` stub
naming the upstream symbol). It must NOT emit a DO-adapter that carries the
list's pointers or the LSTM's weight state as adapter fields — the moment an
adapter carries its own mutable state, the elegance is gone and the diff-gate
will fail it anyway. The Frankenstein guard is a generator **refusal**, not a
warning.

### 4.3 The routing signal

The generator's leaf-detector keys on the reconstructed `ClassSurface`:

- **Leaf candidate** iff: every method's params/return map cleanly to the type
  table (§2.1) AND the class has no `is_friend_of` web implying intrusive access
  AND its fields map onto value tenants. (CONJECTURE — the precise predicate is
  what the probe calibrates; until then the leaf set is the hand-curated
  unicharset/recoder/dawg list above, NOT an auto-classifier.)
- **Hand-port** otherwise. When in doubt → hand-port (the safe default; a
  wrongly-codegen'd intrusive method is a Frankenstein, a wrongly-hand-ported
  leaf is merely more work).

---

## §5 — Verification: byte-parity vs libtesseract (honestly gated)

### 5.1 The oracle and the gate

The truth oracle is **libtesseract via the existing FFI wrapper** (receive-
contract §3, D-OCR-42 diff-gate). For a transcoded leaf method, the generated
adapter's output is compared **byte-for-byte** against the FFI call on the same
input over a fixed corpus. This is `PROBE-OGAR-ADAPTER-UNICHARSET` end-to-end:

```
PROBE-OGAR-ADAPTER-UNICHARSET (P0 — the v1 falsifier, from the doctrine)
  1. Codegen unichar_to_id / id_to_unichar as classid-keyed DO-in/out adapters.
  2. Mint an OGAR classid whose ClassView composes them (has_function manifest).
  3. Invoke through the ClassView.
  Pass:  byte-parity with libtesseract (FFI oracle) on a fixed unicharset corpus.
  Fail / leak: the adapter needs state the value tenants can't carry, or a
     dispatch the ClassView can't express → a Core gap, found cheaply, BEFORE
     scaling the adapter approach.
```

Determinism is verified separately and **without** libtesseract: re-run codegen
on the same `(corpus SHA, IR-snapshot SHA, plan SHA)` and assert byte-identical
`src/generated/` output (receive-contract §4 / D-OCR-41; this is `CPP-AST-RT`
extended to the emit side).

### 5.2 HONEST environment gate (this checkout)

**tesseract-rs does not build in this checkout — there is no system leptonica.**
Therefore the verification layers split by what they need:

| Layer | Needs | Runs here? |
|---|---|---|
| **Generator** (ndjson → Rust source) | only the ndjson + the type table; pure Rust, no leptonica, no libtesseract | **YES — buildable + testable standalone.** Unit-test the generator on a fixture ndjson; golden-file the emitted `.rs`; assert determinism (run twice, diff). |
| **Compile the generated crate** | system leptonica (the FFI wrapper links it) | **NO** — gated on a build env with leptonica. |
| **Output byte-parity** (the FFI oracle / D-OCR-42) | libtesseract + leptonica at runtime | **NO** — gated on the same build env. **This is where `PROBE-OGAR-ADAPTER-UNICHARSET` actually passes/fails; until that env exists the probe is DEFINED but UNRUN.** |

Consequence: the generator can be built, tested, and its determinism proven
**now**; the doctrine's CONJECTURE→FINDING promotion **cannot happen in this
checkout** because the FFI oracle is unavailable. Do not claim the probe is green
from generator tests alone — generator-correctness ≠ output-parity. The
generator golden test proves "ndjson → expected Rust"; only the FFI oracle
proves "expected Rust → same bytes as libtesseract."

### 5.3 What "standalone-testable" buys at v1

A meaningful gate lands without leptonica:

1. **Reassembly test** — feed the `ruff_cpp_spo` locked-recognizer ndjson
   (or the `Tesseract::Recognizer` fixture from `expand.rs` tests) through the
   group-by; assert the reconstructed `ClassSurface` has the right
   bases/fields/methods/overload-split.
2. **Signature test** — assert `Recognize(int)` reconstructs to `(p0: i32) -> i32`
   with the noexcept doc + override note; assert `process(int)`/`process(double)`
   produce two distinct adapters.
3. **Golden + determinism** — emit `src/generated/<class>.rs`, diff against a
   committed golden, run twice and assert byte-identical (D-OCR-41).
4. **Provenance-header test** — every emitted file starts with `//! @generated`
   and three non-placeholder SHAs (receive-contract §5).
5. **Frankenstein-refusal test** — feed an ELIST-shaped fixture; assert the
   generator routes it to hand-port (emits no DO-adapter), not a stateful adapter.

---

## §6 — Status + gaps to feed back to the harvester

**Status: CONJECTURE, floored by `PROBE-OGAR-ADAPTER-UNICHARSET`.** The codegen
shape is coherent and grounded in the actual triple vocabulary and the locked
OGAR Core, but it is not a FINDING until one unicharset leaf adapter hits
byte-parity through a ClassView — and that requires a leptonica build env this
checkout lacks. Generator-side tests (§5.3) can land now; the parity promotion
is blocked on the build env, not on this plan.

### Harvest-surface gaps (feed back to `ruff_cpp_spo`)

These are places the 17-predicate C++ surface is *insufficient for clean codegen*
— each is a candidate harvester follow-up, not a generator hack:

- **GAP-1 (parse robustness, not data loss): class-vs-member IRI split is
  string-fragile.** The method IRI packs `::`-qualified class + `.`-joined member
  + `(params)` into one string. Reassembly works via anchor-first attribution
  (§1.2), but a machine-readable **`member_of` edge** (`cpp:NS::Class.m(params)
  → cpp:NS::Class`) would make grouping a triple lookup instead of string
  surgery. Cheap to emit; removes the only brittle parse in the generator.
- **GAP-2: no parameter *names*, only types.** `has_param_type` is `<i>:<type>`;
  the generator must synthesize `p0,p1,…`. Fine for behavior, poor for readable
  generated signatures. A `has_param_name` sibling (Inferred-tier) would let the
  generated Rust use real argument names.
- **GAP-3: access specifier (pub/protected/private) is dropped.** `cpp_base`
  drops `base.access`; methods carry no visibility predicate. The generator
  cannot tell a public leaf API from a private helper, so it must treat all
  generated adapters as `pub`. A `has_visibility` predicate would let codegen
  scope adapters correctly (and refine the §4.3 leaf-detector — private intrusive
  helpers are a hand-port signal).
- **GAP-4: no const-correctness on the receiver beyond `is_const`.** `is_const`
  marks a const member fn, but ref/ptr/value parameter mutability lives inside
  the raw `<type>` string (`const Image &` vs `Image &`). The generator parses it
  out of the type text; a normalized mutability flag would be cleaner and
  removes another string-parse.
- **GAP-5: `is_pure_virtual` + `virtually_overrides` give MRO, but there is no
  edge for a *non-overriding* inherited method's resolution.** MRO is
  reconstructable from `inherits_from` + `virtually_overrides`, but only for
  methods that exist as triples. A method inherited unchanged (no override, no
  redecl) has NO triple on the derived class — correct, but the generator must
  walk `inherits_from` to find it. Acceptable; noted so the ClassView builder
  knows to chase the base chain rather than expecting a derived-class triple.
- **GAP-6 (POC-default coupling, not a harvester gap): every generated class
  inherits `ValueSchema::Full`.** That is a deliberate POC default in
  `class_view.rs` (reverts to `Bootstrap` before merge, two-site flip). The
  generator must not bake `Full` in; it must read the ClassView default so the
  revert is one upstream change. Flagged so the codegen does not hard-code the
  temporary.
