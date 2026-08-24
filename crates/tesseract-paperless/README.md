# tesseract-paperless

A bounded PROBE of the tokenization seam — not a production carrier. It asks
whether a single versioned BPE tokenization of one source span can drive
lexical retrieval (Tantivy), lexical/grammar projection (DeepNSM-v2), and a
forward-prediction input surface at the same time, without re-tokenizing,
without a DataFrame, and without a second cognitive population. The law it
tests: **ONE TOKENIZATION RECEIPT, MANY BORROWED CONSUMERS.**

## Run it

```sh
cargo run --release -p tesseract-paperless --features token --example probe_token_seam
```

`--release` is not optional in spirit: the encode cost is reported as
merge-table probe COUNTS, and a debug run would still be correct but slow —
Alice trains a 255-cap merge table over 75 KB of paragraph text.

## Fixture provenance

| file | bytes | sha256 | source | licence/derivation |
|---|---|---|---|---|
| `corpus/alice.txt` | 174,693 | `15124d40c182677c2d90fba80310173d63428e0591ce0df3e9bdc01a789a89c6` | carried from `/home/user/tantivy/benches/alice.txt` | Project Gutenberg, *Alice's Adventures in Wonderland*, public domain. CRLF line terminators, UTF-8 with BOM — the probe strips both before splitting into paragraphs. |
| `corpus/coca_academic_20k.tsv` | 226,651 | `4ae20ce39dd3018346700e0f88df2b59e1a7df4e4a06e0f285fd44065166e0f0` | derived from `/home/user/lance-graph/crates/deepnsm/word_frequency/academic_20k.csv` (sha256 `1dfd5edaa5a6ac9b8ac5abbf87894abaf7de8a449ad7c09ca1f6324226396e2d`, 20,845 data rows + 1 header) | `awk -F, 'NR>1 {print $4"\t"$5}' academic_20k.csv > coca_academic_20k.tsv` — columns 4 and 5 of that CSV are `word` and `Pos`. |

### Why the fixtures are committed and not fetched

A test fixture that lives outside the repo is a time bomb with a skip-guard
for a fuse — on a fresh container the skip makes it pass instead of failing
honestly. This repo's own `CLAUDE.md` states the rule plainly: "Fixtures live
in the repo." Both files above are committed; nothing in this crate reaches
outside the checkout at build or run time.

### The one mirror, and its drift check

`corpus/coca_academic_20k.tsv` is a DERIVED copy of data that lives
authoritatively in `lance-graph`. It is the exact `word,Pos` projection
DeepNSM-v2's own `genre_shapes` example loads, carried here so this crate
stays hermetic. To confirm it has not drifted from its source, given a
`lance-graph` checkout at `<lance-graph>`:

```sh
awk -F, 'NR>1 {print $4"\t"$5}' <lance-graph>/crates/deepnsm/word_frequency/academic_20k.csv \
  | diff - corpus/coca_academic_20k.tsv && echo IN-SYNC
```

## What is ABSENT

- **The whole-KJV corpus.** Only the Genesis 2-3 scene (`SCENE` in the probe)
  is carried, verbatim from `PROBE-TOKEN-BPE-GEOMETRY-1`, so the two probes
  measure the same bytes. No larger KJV text is fetched, synthesized, or
  simulated.
- **`bible_vocab.txt`.** Not present on this machine; not reconstructed.
- **The trained `cam96_codebook.bin` / `cam96_codes.bin`.** Their absence
  means DeepNSM's semantic DISTANCE half cannot be exercised by this probe —
  only the lexical/grammar half (tagging, PoS, SPO extraction) runs.
- **Any MecCog corpus.** It does not exist anywhere on this machine in any
  form. The only trace of it anywhere in this workspace is one rhetorical
  mention in a MedCare-rs board file — not a corpus, not a fixture, not code.

Each of these is reported as absent where the probe's output would otherwise
imply it, and none is stood in for with a simulated or synthetic substitute.

## Layering

| module | owns | never |
|---|---|---|
| `contract` | the codebook + its identity | reads the lane |
| `lane` | resident particles + framing | owns text |
| `lexical` | the DeepNSM projection | reads the source |
| `seam_tantivy` | the index seam | owns offsets |
| `forward` | the prediction input surface | owns the sequence |

The probe's own gate list (`PROBE-TOKEN-SEAM-1`, run via the command above)
exists to PROVE each "never" — e.g. `T-CONTRACT-GATE` checks that a view
refuses to open under the wrong contract — rather than merely asserting the
boundary in prose.
