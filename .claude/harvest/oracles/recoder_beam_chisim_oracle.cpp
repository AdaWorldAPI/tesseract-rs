// recoder_beam_chisim_oracle.cpp — byte-parity oracle for the "C3" CJK-trie
// falsifier: dumps ONLY the SetupDecoder beam-search maps (is_valid_start_ /
// final_codes_ / next_codes_) from a `.lstm-recoder` component, in the exact
// shape `lance_graph_contract::unicharcompress::UnicharCompress::dump_beam()`
// emits (see `crates/lance-graph-contract/src/unicharcompress.rs` and its
// `examples/recoder_dump.rs -- <path> beam`), so the diff is byte-for-byte:
//
//   is_valid_start\t<code_range>
//   <code>\t<0|1>                                  // for code in 0..code_range
//   final\t<prefix csv>\t<GetFinalCodes csv | ->    // each distinct prefix, once
//   next\t<prefix csv>\t<GetNextCodes  csv | ->
//
// Distinct prefixes are enumerated by walking every entry of `encoder_` in id
// order and, within each, truncation lengths 0..length ascending, emitting
// each prefix the FIRST time it is seen (an `unordered_set<RecodedCharID>`
// keyed by the shared `RecodedCharIDHash`) — identical to the Rust
// `dump_beam()` walk. The underlying calls
// (`SetupDecoder`/`IsValidFirstCode`/`GetFinalCodes`/`GetNextCodes`) are the
// same ones `unicharcompress.h:182-196` and `.cpp:395-434` declare/define, and
// the same ones the pre-existing `recoder_oracle.cpp`'s "beam" mode already
// exercises for `eng`/`deu` (E-OCR-RECODER-BEAM-1).
//
// WHY A SEPARATE FILE: this oracle is purpose-built for the two-model
// contrast the "C3" item cares about — `eng` (a structurally near-empty
// `next_codes_` trie: every encoder entry is length 1, so the multi-code beam
// paths are unreachable) vs `chi_sim` (the multi-code CJK falsifier fixture,
// `corpus/model/chi_sim.lstm-recoder`, code lengths spanning 1-5 — see
// `corpus/model/README.md` § "Falsifier fixtures"). `recoder_oracle.cpp`
// remains the general-purpose encode/decode/beam oracle for the
// model-agnostic eng/deu parity sweep (E-OCR-DEU-PARITY-MODEL-AGNOSTIC-1);
// this file is scoped to the beam maps only, plus a **stderr-only** summary
// (entry count, code-length histogram, code_range, prefix/next/final
// occupancy) that must NEVER touch stdout — stdout is the byte-parity
// surface diffed against the Rust `dump_beam()` output, and mixing a summary
// into it would silently break every future diff.
//
// ABI note: the installed lib is tesseract 5.3.4 and `/tmp/tesseract-src` is
// checked out at matching 5.3.4 headers — zero ABI skew, so (unlike the
// older 5.5.0-header method) no bijection self-check is required here.
//
// Build:
//   g++ -std=c++17 recoder_beam_chisim_oracle.cpp \
//       -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/include \
//       -I/usr/include/leptonica \
//       $(pkg-config --cflags --libs tesseract) $(pkg-config --libs lept) \
//       -o /tmp/recoder_beam_chisim_oracle
//
// Run (per model; stdout = diffable dump, stderr = human summary):
//   /tmp/recoder_beam_chisim_oracle corpus/model/eng.lstm-recoder \
//       > /tmp/eng_beam_out.txt 2> /tmp/eng_beam_summary.txt
//   /tmp/recoder_beam_chisim_oracle corpus/model/chi_sim.lstm-recoder \
//       > /tmp/chi_sim_beam_out.txt 2> /tmp/chi_sim_beam_summary.txt
//
// Rust side (the counterpart each stdout file diffs against — see the report
// for the exact orchestrator-run command):
//   cargo run -q -p lance-graph-contract --example recoder_dump -- \
//       corpus/model/eng.lstm-recoder beam > /tmp/rust_eng_beam.tsv
//   cargo run -q -p lance-graph-contract --example recoder_dump -- \
//       corpus/model/chi_sim.lstm-recoder beam > /tmp/rust_chi_sim_beam.tsv
//   diff /tmp/eng_beam_out.txt /tmp/rust_eng_beam.tsv
//   diff /tmp/chi_sim_beam_out.txt /tmp/rust_chi_sim_beam.tsv
//   # both byte-identical => the C3 CJK-trie beam maps are byte-parity green

#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "serialis.h"
#include "unicharcompress.h"

using tesseract::RecodedCharID;
using tesseract::TFile;
using tesseract::UnicharCompress;

// Print code(0..length-1) comma-joined (no trailing newline). Mirrors
// recoder_oracle.cpp's csv_codes exactly (kept byte-identical on purpose).
static void csv_codes(const RecodedCharID &code) {
  for (int i = 0; i < code.length(); ++i) {
    if (i > 0) {
      printf(",");
    }
    printf("%d", code(i));
  }
}

// Print an int-list, or "-" for a null (absent) list. Mirrors
// recoder_oracle.cpp's csv_list_or_dash exactly.
static void csv_list_or_dash(const std::vector<int> *list) {
  if (list == nullptr) {
    printf("-");
    return;
  }
  for (size_t i = 0; i < list->size(); ++i) {
    if (i > 0) {
      printf(",");
    }
    printf("%d", (*list)[i]);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <path/to/X.lstm-recoder>\n", argv[0]);
    fprintf(stderr,
            "  stdout: is_valid_start/final/next beam maps, dump_beam() shape\n");
    fprintf(stderr, "  stderr: human-readable summary only (never mixed into stdout)\n");
    return 2;
  }
  const char *path = argv[1];

  TFile fp;
  if (!fp.Open(path, nullptr)) {
    fprintf(stderr, "TFile::Open failed: %s\n", path);
    return 1;
  }
  UnicharCompress uc;
  if (!uc.DeSerialize(&fp)) {
    fprintf(stderr, "UnicharCompress::DeSerialize failed: %s\n", path);
    return 1;
  }

  // Recover encoder_.size() via the public API (private member, no size()):
  // EncodeUnichar returns 0 exactly when id >= size; every trained entry is
  // length >= 1, so this stops precisely at the count. Identical technique to
  // recoder_oracle.cpp.
  unsigned count = 0;
  {
    RecodedCharID tmp;
    while (uc.EncodeUnichar(count, &tmp) > 0) {
      ++count;
    }
  }

  // ---- stdout: the byte-parity surface (dump_beam() shape) ----
  printf("is_valid_start\t%d\n", uc.code_range());
  for (int c = 0; c < uc.code_range(); ++c) {
    printf("%d\t%d\n", c, uc.IsValidFirstCode(c) ? 1 : 0);
  }
  std::unordered_set<RecodedCharID, RecodedCharID::RecodedCharIDHash> seen;
  for (unsigned id = 0; id < count; ++id) {
    RecodedCharID full;
    uc.EncodeUnichar(id, &full);
    for (int l = 0; l < full.length(); ++l) {
      RecodedCharID prefix = full;
      prefix.Truncate(l); // length_ = l, code_ untouched -> identity = code[0..l]
      if (!seen.insert(prefix).second) {
        continue;
      }
      printf("final\t");
      csv_codes(prefix);
      printf("\t");
      csv_list_or_dash(uc.GetFinalCodes(prefix));
      printf("\n");
      printf("next\t");
      csv_codes(prefix);
      printf("\t");
      csv_list_or_dash(uc.GetNextCodes(prefix));
      printf("\n");
    }
  }

  // ---- stderr: a human-readable summary for the report, NEVER stdout ----
  // Independent second pass over the public API (not a re-derivation of the
  // stdout loop's bookkeeping), so a bug in one pass cannot self-confirm in
  // the other.
  int len_hist[RecodedCharID::kMaxCodeLen + 1] = {0};
  int longer_than_1 = 0;
  for (unsigned id = 0; id < count; ++id) {
    RecodedCharID full;
    uc.EncodeUnichar(id, &full);
    int len = full.length();
    if (len >= 0 && len <= RecodedCharID::kMaxCodeLen) {
      len_hist[len]++;
    }
    if (len > 1) {
      ++longer_than_1;
    }
  }
  int distinct_prefixes = 0;
  int next_populated = 0;
  int final_populated = 0;
  {
    std::unordered_set<RecodedCharID, RecodedCharID::RecodedCharIDHash> seen2;
    for (unsigned id = 0; id < count; ++id) {
      RecodedCharID full;
      uc.EncodeUnichar(id, &full);
      for (int l = 0; l < full.length(); ++l) {
        RecodedCharID prefix = full;
        prefix.Truncate(l);
        if (!seen2.insert(prefix).second) {
          continue;
        }
        ++distinct_prefixes;
        if (uc.GetNextCodes(prefix) != nullptr) {
          ++next_populated;
        }
        if (uc.GetFinalCodes(prefix) != nullptr) {
          ++final_populated;
        }
      }
    }
  }
  fprintf(stderr, "==== summary for %s ====\n", path);
  fprintf(stderr, "entries (count)          : %u\n", count);
  fprintf(stderr, "code_range               : %d\n", uc.code_range());
  fprintf(stderr, "entries with length > 1  : %d\n", longer_than_1);
  fprintf(stderr, "code length histogram    :");
  for (int l = 0; l <= RecodedCharID::kMaxCodeLen; ++l) {
    if (len_hist[l] > 0) {
      fprintf(stderr, " {%d:%d}", l, len_hist[l]);
    }
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "distinct prefixes seen   : %d\n", distinct_prefixes);
  fprintf(stderr, "prefixes w/ final_codes  : %d\n", final_populated);
  fprintf(stderr, "prefixes w/ next_codes   : %d (0 here would mean the multi-code\n",
          next_populated);
  fprintf(stderr, "                            trie paths are unreached -- STOP if so)\n");
  return 0;
}
