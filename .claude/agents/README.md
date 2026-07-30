# Agent cards — the lessons this repo paid for

Seven specialist cards. Each one exists because a specific mistake cost real
time in this repo, and the card is the shape of that mistake written down so it
does not get made twice. Every rule in them carries its incident: what was
measured, what the wrong answer was, and what it cost.

Read the one whose triggers match before starting; do not read all seven.

| Card | Fires when you are about to... |
|---|---|
| `parity-oracle-smith` | build or trust a C++/liblept oracle, add a byte-parity leaf, diff against libtesseract |
| `measurement-skeptic` | believe a probe, a null result, a benchmark, or an "X% of runtime" number |
| `falsifier-auditor` | add a test, re-pin a golden, or claim an assertion proves something |
| `heuristic-gate-warden` | wire a classifier/detector/threshold into the pipeline, or gate one |
| `render-typography-engineer` | touch font size, baseline placement, or overlay fidelity in the PDF/HTML surfaces |
| `subagent-output-auditor` | synthesize fanned-out agent output into a deliverable |
| `transcode-scope-warden` | decide if something is a transcode or our own synthesis, or run cargo |

## The four rules that generalize past their own card

If you read nothing else:

1. **A null result is a claim about the measurement apparatus until proven
   otherwise.** Measured twice in this repo: the Sauvola probe that returned
   "no difference" because the mode never reached the code under test (the
   answer moved 68x once wired), and the period probe whose `ends_with('.')`
   check scored a recovered period as a miss because the widened crop pulled in
   a stray em dash. Read the trace for *why* a null is null before promoting it.

2. **An oracle must reproduce the operation under test**, not something in its
   neighbourhood that also rotates/scales/decodes. The deskew oracle called the
   wrong API twice, in ways where a CORRECT port would have FAILED and a wrong
   one could have PASSED.

3. **An assertion implied by the code it tests is not a test.** Every guard
   needs both halves: can-it-fire AND can-it-stay-silent, on non-trivial
   inputs. A guard that fires on everything carries exactly as much information
   as one that never fires.

4. **Judge a candidate by what is around it, not by an absolute derived from
   the whole page.** The `xy_cut` gutter threshold, the noise-readmit reach, and
   the table-classification ruled-evidence gate are all the same correction:
   a page-relative or absolute constant that could not discriminate, replaced by
   a neighbourhood-relative measurement that could.

## Model policy (from CLAUDE.md, restated because it is set BEFORE spawning)

- **Orchestrator: Opus.** All evidence composition, all central gating
  (`cargo fmt` / `clippy -D warnings` / the scoped test suite, run ONCE), every
  `git commit` / `push`.
- **Workers: Sonnet.** Bounded transcription against a written spec.
- **Never Haiku** for any subagent here.

Every worker brief carries, verbatim: no cargo (this crate path-deps sibling
workspaces; per-agent compiles blow up the shared `target/` and produce spurious
cross-agent failures), no git, the exact file scope, which files another agent
owns concurrently, and "do not claim it compiles or that tests pass, you did not
run it". Fan out on **disjoint files only** — the orchestrator makes any shared
edit after the fleet lands.
