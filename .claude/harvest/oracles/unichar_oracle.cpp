// unichar_oracle.cpp — byte-parity oracle for lance_graph_contract::unichar
// (utf8_step + utf8_to_utf32). MODEL-INDEPENDENT: one run covers every language
// (the corpus is fixed hex, the step table is the 256-entry lead-byte table).
//
// Reproduces the exact TSV the Rust `unichar_dump` example prints
// (examples/unichar_dump.rs + src/unichar.rs):
//   Section 1: "STEP\t<byte 0..255>\t<utf8_step(byte)>"  for all 256 leads
//   Section 2: "U32\t<hex>\t<decoded>"  over the SAME 12-case corpus, where
//              decoded = comma-joined decimal codepoints, or "ILLEGAL" (empty
//              result vector), or "0" (the overlong-NUL c080 quirk).
//
// tesseract::UNICHAR::utf8_step is a PURE 256-entry table lookup (unichar.cpp:143
// — no continuation-byte validation), so passing a single lead byte reproduces
// the Rust `const fn utf8_step` value-for-value. UTF8ToUTF32 (unichar.cpp:220)
// clears the vector on an illegal LEAD byte; the corpus has no embedded NUL so
// the const-char* interface is exact.
//
// Build (source headers 5.3.4 + installed lib 5.3.4 → zero ABI skew):
//   g++ -std=c++17 unichar_oracle.cpp \
//       -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/include \
//       -ltesseract -lleptonica -o /tmp/unichar_oracle
// Run:
//   /tmp/unichar_oracle > /tmp/o_unichar.tsv
//   cargo run -q -p lance-graph-contract --example unichar_dump > /tmp/r_unichar.tsv
//   diff /tmp/o_unichar.tsv /tmp/r_unichar.tsv   # byte-identical => parity holds

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tesseract/unichar.h"

using tesseract::UNICHAR;
using tesseract::char32;

// The identical 12-case corpus hard-coded in unichar_dump.rs, in the SAME order.
static const char *CASES[] = {
    "41",           // A
    "c3a9",         // é  U+00E9
    "e4b8ad",       // 中 U+4E2D
    "f09f9880",     // 😀 U+1F600
    "414243",       // ABC
    "48c3a9",       // Hé
    "e4b8ade69687", // 中文
    "80",           // lone continuation -> ILLEGAL
    "bf",           // lone continuation -> ILLEGAL
    "f8",           // 5-byte form       -> ILLEGAL
    "ff",           // 0xFF              -> ILLEGAL
    "c080",         // overlong NUL      -> 0
};

static std::vector<unsigned char> hex_to_bytes(const char *hex) {
  std::vector<unsigned char> out;
  size_t n = strlen(hex) / 2;
  for (size_t i = 0; i < n; ++i) {
    unsigned int v = 0;
    sscanf(hex + 2 * i, "%2x", &v);
    out.push_back(static_cast<unsigned char>(v));
  }
  return out;
}

int main() {
  // Section 1: exhaustive utf8_step over all 256 lead bytes.
  for (int b = 0; b < 256; ++b) {
    char buf[2] = {static_cast<char>(b), 0};
    printf("STEP\t%d\t%d\n", b, UNICHAR::utf8_step(buf));
  }
  // Section 2: UTF8ToUTF32 over the corpus (no case has an embedded NUL, so the
  // NUL-terminated const-char* interface preserves every byte exactly).
  for (const char *hex : CASES) {
    std::vector<unsigned char> bytes = hex_to_bytes(hex);
    std::string s(bytes.begin(), bytes.end());
    std::vector<char32> cps = UNICHAR::UTF8ToUTF32(s.c_str());
    printf("U32\t%s\t", hex);
    if (cps.empty()) {
      printf("ILLEGAL");
    } else {
      for (size_t i = 0; i < cps.size(); ++i) {
        if (i > 0) {
          printf(",");
        }
        printf("%d", static_cast<int>(cps[i]));
      }
    }
    printf("\n");
  }
  return 0;
}
