// unicharset_oracle.cpp — byte-parity oracle for lance_graph_contract::unicharset::UniCharSet.
//
// Reconstructed from the ephemeral recipe in
//   lance-graph/crates/lance-graph-contract/examples/unicharset_dump.rs
// (E-CPP-PARITY-1..6). Prints one of six modes so ONE binary covers every
// UNICHARSET leaf for ANY model (eng, deu, …). The `bijection` half is the
// proven self-validating reference: if its diff is 0 the object layout is sound
// and the field half is trustworthy even across a header/lib version skew
// (here there is none — source and lib are both 5.3.4).
//
// Output shapes (must match the Rust dump_* byte-for-byte):
//   bijection   : "<id>\t<unichar>\n"
//   properties  : "<id>\t<isalpha> <islower> <isupper> <isdigit> <ispunctuation>\n"
//   script      : "<id>\t<script_id>\n"
//   other_case  : "<id>\t<other_case_id>\n"
//   direction   : "<id>\t<direction>\n"
//   mirror      : "<id>\t<mirror_id>\n"
//
// Build (source headers 5.3.4 + installed lib 5.3.4):
//   g++ -std=c++17 unicharset_oracle.cpp \
//       -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/include \
//       -ltesseract -lleptonica -o /tmp/unicharset_oracle
// Run:
//   /tmp/unicharset_oracle <path/to/X.lstm-unicharset> [bijection|properties|script|other_case|direction|mirror]

#include <cstdio>
#include <cstring>

#include "unicharset.h"

using tesseract::UNICHARSET;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s <unicharset> "
            "[bijection|properties|script|other_case|direction|mirror]\n",
            argv[0]);
    return 2;
  }
  const char* path = argv[1];
  const char* mode = (argc >= 3) ? argv[2] : "bijection";

  UNICHARSET u;
  if (!u.load_from_file(path)) {
    fprintf(stderr, "load_from_file failed: %s\n", path);
    return 1;
  }

  const int n = static_cast<int>(u.size());
  for (int id = 0; id < n; ++id) {
    if (!strcmp(mode, "properties")) {
      printf("%d\t%d %d %d %d %d\n", id, u.get_isalpha(id) ? 1 : 0,
             u.get_islower(id) ? 1 : 0, u.get_isupper(id) ? 1 : 0,
             u.get_isdigit(id) ? 1 : 0, u.get_ispunctuation(id) ? 1 : 0);
    } else if (!strcmp(mode, "script")) {
      printf("%d\t%d\n", id, u.get_script(id));
    } else if (!strcmp(mode, "other_case")) {
      printf("%d\t%d\n", id, u.get_other_case(id));
    } else if (!strcmp(mode, "direction")) {
      printf("%d\t%d\n", id, static_cast<int>(u.get_direction(id)));
    } else if (!strcmp(mode, "mirror")) {
      printf("%d\t%d\n", id, u.get_mirror(id));
    } else {  // bijection
      printf("%d\t%s\n", id, u.id_to_unichar(id));
    }
  }
  return 0;
}
