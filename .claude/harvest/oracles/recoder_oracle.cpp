// recoder_oracle.cpp — byte-parity oracle for
// lance_graph_contract::unicharcompress::UnicharCompress (the recoder load side).
// MODEL-DEPENDENT: run per language (eng, deu). Three modes match the Rust
// `recoder_dump` example (examples/recoder_dump.rs + src/unicharcompress.rs
// dump_encode / dump_decode / dump_beam) byte-for-byte:
//
//   encode : "<id>\t<length>\t<c0,c1,...>"                         per id
//   decode : "code_range\t<N>"  then  "<id>\t<DecodeUnichar(EncodeUnichar(id))>"
//   beam   : "is_valid_start\t<code_range>", then "<code>\t<0|1>" for
//            code in 0..code_range; then, walking every id in order and within
//            each id truncation lengths 0..length ascending (first-seen dedup):
//              "final\t<prefix csv>\t<GetFinalCodes csv | ->"
//              "next\t<prefix csv>\t<GetNextCodes  csv | ->"
//
// Loads the SAME .lstm-recoder component the Rust reads, via tesseract::TFile +
// UnicharCompress::DeSerialize (unicharcompress.cpp:323 -> ComputeCodeRange +
// SetupDecoder). encoder_ is private with no size(); EncodeUnichar returns 0 for
// id >= size and length>=1 for every trained entry, so the id domain is found by
// iterating until the first 0-return (== encoder_.size() on trained data).
//
// The Encode->Decode round-trip (decode mode) is the self-validation: a shared
// code decodes to the last-writer id on BOTH sides, and code_range is a computed
// cross-check, guarding the object layout for this binary leaf.
//
// Build (source headers 5.3.4 + installed lib 5.3.4 → zero ABI skew):
//   g++ -std=c++17 recoder_oracle.cpp \
//       -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/src/ccstruct \
//       -I/tmp/tesseract-src/include \
//       -ltesseract -lleptonica -o /tmp/recoder_oracle
// Run (per language, per mode):
//   /tmp/recoder_oracle corpus/model/eng.lstm-recoder encode > /tmp/o_eng_encode.tsv
//   cargo run -q -p lance-graph-contract --example recoder_dump -- \
//       corpus/model/eng.lstm-recoder encode > /tmp/r_eng_encode.tsv
//   diff /tmp/o_eng_encode.tsv /tmp/r_eng_encode.tsv   # byte-identical => parity

#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "unicharcompress.h"
#include "serialis.h"

using tesseract::RecodedCharID;
using tesseract::TFile;
using tesseract::UnicharCompress;

// Print code(0..length-1) comma-joined (no trailing newline).
static void csv_codes(const RecodedCharID &code) {
  for (int i = 0; i < code.length(); ++i) {
    if (i > 0) {
      printf(",");
    }
    printf("%d", code(i));
  }
}

// Print an int-list, or "-" for a null (absent) list.
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
  if (argc < 3) {
    fprintf(stderr, "usage: %s <path/to/X.lstm-recoder> [encode|decode|beam]\n", argv[0]);
    return 2;
  }
  const char *path = argv[1];
  const char *mode = argv[2];

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
  // length >= 1, so this stops precisely at the count.
  unsigned count = 0;
  {
    RecodedCharID tmp;
    while (uc.EncodeUnichar(count, &tmp) > 0) {
      ++count;
    }
  }

  if (!strcmp(mode, "decode")) {
    printf("code_range\t%d\n", uc.code_range());
    for (unsigned id = 0; id < count; ++id) {
      RecodedCharID code;
      uc.EncodeUnichar(id, &code);
      printf("%d\t%d\n", id, uc.DecodeUnichar(code));
    }
  } else if (!strcmp(mode, "beam")) {
    printf("is_valid_start\t%d\n", uc.code_range());
    for (int c = 0; c < uc.code_range(); ++c) {
      printf("%d\t%d\n", c, uc.IsValidFirstCode(c) ? 1 : 0);
    }
    // Distinct prefixes: id-order outer, truncation-length 0..length inner,
    // emit each the first time seen. The seen-set is membership only; emission
    // order is the deterministic walk (identical to the Rust HashSet-guarded
    // walk regardless of hash iteration order).
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
  } else { // encode (default)
    for (unsigned id = 0; id < count; ++id) {
      RecodedCharID code;
      uc.EncodeUnichar(id, &code);
      printf("%d\t%d\t", id, code.length());
      csv_codes(code);
      printf("\n");
    }
  }
  return 0;
}
