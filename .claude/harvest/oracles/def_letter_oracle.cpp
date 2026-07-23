// def_letter_oracle.cpp — byte-parity oracle for tesseract_core::dict_walker
// (`DictLite::default_dawgs` + `DictLite::def_letter_is_okay`) against the REAL
// `tesseract::Dict::{default_dawgs,def_letter_is_okay}` in libtesseract 5.3.4.
//
// The Rust example (`crates/tesseract-core/examples/dict_walk_dump.rs`) reads
// the split dawg components (`/tmp/<lang>.lstm-{word,punc,number}-dawg`, which
// this run harness copies to the hardcoded `/tmp/eng.lstm-*` paths) and the
// unicharset (`/tmp/eng.lstm-unicharset`). This oracle loads the SAME dawg
// bytes through `Dict::LoadLSTM` from the combined traineddata — verified
// `cmp`-identical to `corpus/model/<lang>.lstm-*-dawg` (combine_tessdata -u is
// a lossless split), so the walk rides the exact same edge arrays. Using the
// real `Dict::def_letter_is_okay` (not a re-implementation) makes this a true
// oracle rather than a circular check.
//
// Output shape MUST match `dict_walk_dump.rs` byte-for-byte:
//   step\t<i>\t<unichar_id>\t<word_end 0|1>\tperm=<permuter ordinal>\t
//        valid_end=<0|1>\tupdated={<count>}
//   p\t<dawg_index>\t<dawg_ref>\t<punc_index>\t<punc_ref>\t<back_to_punc 0|1>
// The `p` lines are sorted by (dawg_index, dawg_ref, punc_index, punc_ref,
// back_to_punc) exactly as the Rust `dump_positions` sorts, so the dump is
// insertion-order-independent on both sides.
//
// Build (source headers 5.3.4 + installed lib 5.3.4 -> no ABI skew):
//   g++ -std=c++17 def_letter_oracle.cpp \
//     -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/src/ccstruct \
//     -I/tmp/tesseract-src/src/dict -I/tmp/tesseract-src/src/lstm \
//     -I/tmp/tesseract-src/include -ltesseract -lleptonica -o /tmp/def_letter_oracle
// Run:
//   /tmp/def_letter_oracle <lang> <traineddata> <unicharset> <id0> [id1 ...]
//   e.g. /tmp/def_letter_oracle eng \
//        /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
//        /tmp/eng.lstm-unicharset 91 97 92

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ccutil.h"
#include "dict.h"
#include "tessdatamanager.h"
#include "unicharset.h"

using tesseract::CCUtil;
using tesseract::DawgArgs;
using tesseract::DawgPosition;
using tesseract::DawgPositionVector;
using tesseract::Dict;
using tesseract::TessdataManager;
using tesseract::UNICHARSET;

static void dump_positions(const DawgPositionVector &positions) {
  std::vector<DawgPosition> sorted(positions.begin(), positions.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const DawgPosition &a, const DawgPosition &b) {
              if (a.dawg_index != b.dawg_index)
                return a.dawg_index < b.dawg_index;
              if (a.dawg_ref != b.dawg_ref)
                return a.dawg_ref < b.dawg_ref;
              if (a.punc_index != b.punc_index)
                return a.punc_index < b.punc_index;
              if (a.punc_ref != b.punc_ref)
                return a.punc_ref < b.punc_ref;
              return (int)a.back_to_punc < (int)b.back_to_punc;
            });
  for (const auto &p : sorted) {
    printf("p\t%d\t%lld\t%d\t%lld\t%d\n", (int)p.dawg_index,
           (long long)p.dawg_ref, (int)p.punc_index, (long long)p.punc_ref,
           p.back_to_punc ? 1 : 0);
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <lang> <traineddata> <unicharset> <id0> [id1 ...]\n",
            argv[0]);
    return 2;
  }
  std::string lang = argv[1];
  const char *traineddata = argv[2];
  const char *unicharset_path = argv[3];
  std::vector<int> ids;
  for (int i = 4; i < argc; ++i)
    ids.push_back(atoi(argv[i]));

  UNICHARSET unicharset;
  if (!unicharset.load_from_file(unicharset_path)) {
    fprintf(stderr, "load unicharset failed: %s\n", unicharset_path);
    return 1;
  }

  TessdataManager mgr;
  if (!mgr.Init(traineddata)) {
    fprintf(stderr, "tessdata init failed: %s\n", traineddata);
    return 1;
  }

  CCUtil ccutil;
  Dict dict(&ccutil);
  dict.SetupForLoad(nullptr);
  dict.LoadLSTM(lang, &mgr);
  dict.FinishLoad();

  DawgPositionVector active;
  dict.default_dawgs(&active, false);

  int n = (int)ids.size();
  for (int i = 0; i < n; ++i) {
    bool word_end = (i + 1 == n);
    DawgPositionVector updated;
    DawgArgs args(&active, &updated, tesseract::NO_PERM);
    dict.def_letter_is_okay(&args, unicharset, ids[i], word_end);
    printf("step\t%d\t%d\t%d\tperm=%d\tvalid_end=%d\tupdated={%d}\n", i, ids[i],
           word_end ? 1 : 0, (int)args.permuter, args.valid_end ? 1 : 0,
           (int)updated.size());
    dump_positions(updated);
    active = updated;
  }
  return 0;
}
