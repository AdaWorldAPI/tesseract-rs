# C3 CJK-trie falsifier — recoder beam maps, eng vs chi_sim

**TL;DR: the falsifier works.** `chi_sim.lstm-recoder`'s `next_codes_` trie is
genuinely populated (482 of 1788 distinct prefixes carry non-final
continuations, at prefix depths 0-3), in sharp contrast to `eng`/`deu` where
it is structurally empty (every entry is length 1). **No STOP condition was
hit.** This unblocks "C3" (the CJK multi-code trie item, tracked as deferred
in `tesseract-rs/CLAUDE.md` and `lance-graph` `EPIPHANIES.md` line 3995)
against real, falsifiable data.

## What I did, and what I could not do

Per this task's hard constraints, I did not run `cargo` at all and did not
touch any `.rs` file. Everything below is a **C++-only** oracle build/run
plus direct byte/line analysis of its output with `awk`/`diff`/`sha256sum`.
I did **not** run the Rust side myself — the exact command the orchestrator
should run is given below, and I flag explicitly (§ "Rust-side gap analysis")
what I verified about it by reading source vs. what remains to be confirmed
by actually running it.

## Deliverable 1 — the oracle

`.claude/harvest/oracles/recoder_beam_chisim_oracle.cpp` — loads a
`.lstm-recoder` via `TFile` + `UnicharCompress::DeSerialize` (which runs
`ComputeCodeRange` + `SetupDecoder` internally, exactly as the existing
`recoder_oracle.cpp` does), then dumps **only** the beam-search maps
(`is_valid_start_` / `final_codes_` / `next_codes_`) to **stdout** in the
exact shape `lance_graph_contract::unicharcompress::UnicharCompress::dump_beam()`
emits:

```
is_valid_start\t<code_range>
<code>\t<0|1>                                  // for code in 0..code_range
final\t<prefix csv>\t<GetFinalCodes csv | ->    // each distinct prefix, once
next\t<prefix csv>\t<GetNextCodes  csv | ->
```

A second, independent pass computes a human-readable summary (entry count,
code-length histogram, code_range, prefix/next/final occupancy) and prints it
to **stderr only** — it never touches stdout, so stdout stays the pure
byte-parity surface.

### Build (verified — ran successfully in this container)

```sh
g++ -std=c++17 .claude/harvest/oracles/recoder_beam_chisim_oracle.cpp \
    -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/include \
    -I/usr/include/leptonica \
    $(pkg-config --cflags --libs tesseract) $(pkg-config --libs lept) \
    -o /tmp/recoder_beam_chisim_oracle
```

Installed lib is tesseract **5.3.4**; `/tmp/tesseract-src` is checked out at
matching **5.3.4** source headers — zero ABI skew (confirmed: build produced
no warnings, and see the cross-validation below), so no bijection self-check
was needed (unlike the older 5.5.0-header method this repo used earlier).
`pkg-config --modversion tesseract` → `5.3.4`, `lept` → `1.82.0`.

### Run — both models, exactly as asked

```sh
/tmp/recoder_beam_chisim_oracle corpus/model/eng.lstm-recoder \
    > /tmp/eng_beam_out.txt 2> /tmp/eng_beam_summary.txt
/tmp/recoder_beam_chisim_oracle corpus/model/chi_sim.lstm-recoder \
    > /tmp/chi_sim_beam_out.txt 2> /tmp/chi_sim_beam_summary.txt
```

Both exited `0`. (I also ran `deu.lstm-recoder` as an unrequested third data
point — see below — same shape as `eng`.)

### Where the `*_out.txt` outputs actually live — a constraint conflict, flagged

The task's Deliverable-1 text says to "bank both outputs next to the oracle
as `*_out.txt`" (a real, established convention in this directory — e.g.
`decide_if_table_oracle_out.txt`, `pageseg_regions_oracle_out.txt`,
`counts_oracle_out.txt` all sit next to their `.cpp`). **But the HARD
CONSTRAINTS section for this task enumerates exactly two files I may write,
and neither is an `_out.txt`.** I resolved this in favor of the explicit hard
constraint: I did **not** commit `*_out.txt` files into
`.claude/harvest/oracles/`. Instead, the run outputs (and every intermediate
artifact used for the analysis below) are on disk at:

```
/tmp/claude-0/-home-user/fc66dc28-6793-51f4-8d29-e2e12f4b465c/scratchpad/
  recoder_beam_chisim_oracle          # the compiled binary
  eng_beam_out.txt                     # stdout, eng   (924 B,  114 lines)
  chi_sim_beam_out.txt                 # stdout, chi_sim (84,791 B, 3801 lines)
  deu_beam_out.txt                     # stdout, deu (bonus)  (118 lines)
  eng_beam_summary.txt                 # stderr summary, eng
  chi_sim_beam_summary.txt             # stderr summary, chi_sim
  deu_beam_summary.txt                 # stderr summary, deu (bonus)
  eng_beam_baseline.txt                # cross-check: pre-existing recoder_oracle.cpp, beam mode, eng
  chi_sim_beam_baseline.txt            # cross-check: same, chi_sim
  eng_encode_baseline.tsv              # cross-check: recoder_oracle.cpp encode mode, eng (histogram source)
  chi_sim_encode_baseline.tsv          # cross-check: same, chi_sim
```

If the orchestrator wants `eng_beam_out.txt`/`chi_sim_beam_out.txt` committed
into `.claude/harvest/oracles/` as `*_out.txt` to match the directory's own
convention, the two-command "Run" block above reproduces them byte-for-byte
(the binary and both `.lstm-recoder` inputs are unchanged), or they can be
copied straight from the scratchpad paths listed.

### Cross-validation performed (no Rust available to me, so I cross-checked in C++)

1. **SHA256 of the input files**, confirming I ran against the exact bytes
   committed to the repo (matches `corpus/model/README.md`'s table exactly):

   ```
   7ee2c195d397aa4fccd5efc5ab5e71d21d8e94425151d5f978cd74b546c4bb12  corpus/model/chi_sim.lstm-recoder
   a481e4cb27c2b832269a0578a1438c243a13228a70f9556162b7f06131d2e664  corpus/model/eng.lstm-recoder
   ```

2. **Independent re-implementation cross-check.** The pre-existing
   `.claude/harvest/oracles/recoder_oracle.cpp` already has a `beam` mode
   that walks the identical `SetupDecoder`/`IsValidFirstCode`/`GetFinalCodes`/
   `GetNextCodes` surface (I read it in full before writing the new file, per
   the task's read-first instruction). I built *that* oracle too
   (`recoder_oracle_baseline`) and ran it in `beam` mode on both models, then
   `diff`'d its stdout against my new dedicated oracle's stdout:

   ```
   diff eng_beam_out.txt eng_beam_baseline.txt      # exit 0 — byte-identical
   diff chi_sim_beam_out.txt chi_sim_beam_baseline.txt  # exit 0 — byte-identical
   ```

   Two independently-compiled translation units (different `main`, same
   logic, matching the C++ header/impl exactly) produce byte-identical output
   on both the trivial (eng) and non-trivial (chi_sim) case. This is not a
   substitute for the real Rust-side diff, but it is strong evidence the new
   oracle's C++ is correct before that diff is ever run.

3. **Histogram cross-check against the corpus README's claim.** I also ran
   `recoder_oracle_baseline` in `encode` mode (dumps `<id>\t<length>\t<codes>`
   per entry) and computed the length histogram directly with `awk`:

   ```
   chi_sim: 4022 total, {1:128, 2:278, 3:2077, 4:1515, 5:24}
   eng:      112 total, {1:112}
   ```

   This matches `corpus/model/README.md`'s stated histogram **exactly**.

## The numbers — eng vs deu vs chi_sim

| | `eng.lstm-recoder` | `deu.lstm-recoder` (bonus) | `chi_sim.lstm-recoder` |
|---|---:|---:|---:|
| entries (`count`) | 112 | 116 | **4022** |
| `code_range` | 111 | 115 | **224** |
| entries with length > 1 | 0 | 0 | **3894** |
| code length histogram | `{1:112}` | `{1:116}` | `{1:128, 2:278, 3:2077, 4:1515, 5:24}` |
| distinct prefixes walked | 1 | 1 | **1788** |
| prefixes w/ `final_codes_` populated | 1 | 1 | **1765** |
| prefixes w/ `next_codes_` populated | **0** | **0** | **482** |
| max `next_codes_` prefix depth | — | — | **3** (i.e. from ≥ length-4 entries) |
| max `final_codes_` prefix depth | 0 | 0 | **4** (i.e. from the 24 length-5 entries) |

The `next_codes_` depth distribution for chi_sim (populated prefixes only):
`depth 0: 1, depth 1: 25, depth 2: 432, depth 3: 24` — the trie has real,
multi-level structure, not just a single shallow branch. The `final_codes_`
depth distribution: `depth 0: 1, depth 1: 25, depth 2: 551, depth 3: 1164,
depth 4: 24` (sums to 1765, matching the occupancy count above; the sum of
final-code *values* across all rows is 4021, one short of the 4022 entry
count — an expected, harmless dedup effect: `SetupDecoder`'s
`if (!contains(*final_it->second, code(len))) push` skips a value that a
prior entry already registered under the same prefix).

For eng/deu the single distinct prefix walked is the empty prefix (depth 0);
`final_codes_[""]` holds every entry's sole code (i.e. plain pass-through),
and `next_codes_` never gets an entry at all because no encoder entry has
length > 1 to trigger the `while (--len >= 0)` walk in `SetupDecoder`
(`unicharcompress.cpp:412-427`). This is exactly the "structurally
unreachable, not merely untested" gap the task description named.

**No STOP condition applies.** chi_sim's `next_codes_` is unambiguously
non-empty (482 populated prefixes, spanning 4 depth levels), so the fixture
does falsify what it is meant to falsify, and C3 is unblocked to proceed.

## Rust-side gap analysis

**No gap found — I did not need to add anything, and I recommend the
orchestrator run it as-is.** I read
`/home/user/lance-graph/crates/lance-graph-contract/examples/recoder_dump.rs`
in full: it already has a `"beam" => print!("{}", recoder.dump_beam())` arm,
it takes the recoder path as `argv[1]` with **no hardcoded model name**, and
`UnicharCompress::dump_beam()` (in
`crates/lance-graph-contract/src/unicharcompress.rs`) is likewise
model-agnostic — it walks whatever `encoder_`/`code_range`/`final_codes`/
`next_codes` the loaded file produced. So the identical command that already
works for `eng` should work for `chi_sim` unmodified:

```sh
cargo run -q -p lance-graph-contract --example recoder_dump -- \
    corpus/model/eng.lstm-recoder beam > /tmp/rust_eng_beam.tsv
cargo run -q -p lance-graph-contract --example recoder_dump -- \
    corpus/model/chi_sim.lstm-recoder beam > /tmp/rust_chi_sim_beam.tsv

diff /tmp/eng_beam_out.txt /tmp/rust_eng_beam.tsv        # expect: byte-identical
diff /tmp/chi_sim_beam_out.txt /tmp/rust_chi_sim_beam.tsv  # expect: byte-identical (the real falsifier)
```

(Substitute the scratchpad paths above, or re-run the two C++ commands in
§"Run", for `/tmp/eng_beam_out.txt` / `/tmp/chi_sim_beam_out.txt`.)

**Caveat — this is inferred from source reading, not verified by execution.**
I did not run `cargo` per this task's hard constraint, so I have not
personally confirmed `dump_beam()`'s `RecodedCharId` truncation/hash/insert
walk produces byte-identical output to the C++ `SetupDecoder` walk on the
*specific* multi-code, multi-depth chi_sim data — only that (a) the Rust
source, read in full, implements the same algorithm shape as the C++ (see
`unicharcompress.rs` lines 389-441, which mirror `unicharcompress.cpp`
lines 395-434 near-line-for-line, including the `while len >= 0 { len -= 1;
... }` prefix-climb and the dedup-then-break on an already-seeded
`next_codes` entry), and (b) `RecodedCharId::MAX_CODE_LEN = 9` comfortably
covers chi_sim's observed max length of 5, so no truncation/`BadCodeLength`
edge case should fire on this data. The two `diff` commands above are the
actual falsifier; until the orchestrator runs them, "byte-parity" for C3 is
not yet a green result, just a well-supported expectation from independent
C++ cross-validation plus a source-level read of the Rust side.

## Files touched in this repo

- `.claude/harvest/oracles/recoder_beam_chisim_oracle.cpp` (new)
- `.claude/harvest/c3-cjk-trie-report.md` (new, this file)

No other file in `tesseract-rs` or `lance-graph` was modified. `git status`
in `tesseract-rs` shows only the `.cpp` oracle as untracked before this
report was added (the report itself completes the two-file allowance).
