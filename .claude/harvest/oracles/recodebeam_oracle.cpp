// recodebeam_oracle.cpp — byte-parity oracle for tesseract_core::recodebeam
// (`RecodeBeamSearch::Decode` + `ExtractBestPathAsLabels` +
// `ExtractBestPathAsUnicharIds`), recognizer Leaf 7b, against libtesseract
// 5.3.4.
//
// The Rust example (`crates/tesseract-core/examples/beam_dump.rs`) GENERATES a
// deterministic synthetic softmax matrix, decodes it with the Rust beam, and
// writes the SAME matrix to a `.bin` (i32 T, i32 N, then T*N f32 LE). This
// oracle reads that identical `.bin`, runs the REAL `RecodeBeamSearch`, and
// dumps the same labels/xcoords/unichar-ids — so the INPUT is byte-identical
// on both sides. Public-API-only (`Decode(GENERIC_2D_ARRAY<float>, ...)`),
// so no private-member access -> no 5.5.0/5.3.4 ABI-skew hazard (here both are
// 5.3.4 anyway).
//
// Output shape MUST match `beam_dump.rs` byte-for-byte:
//   label\t<int>          (ExtractBestPathAsLabels labels)
//   xcoord\t<int>         (ExtractBestPathAsLabels xcoords)
//   uid\t<int>            (ExtractBestPathAsUnicharIds unichar_ids)
//   uc\t<8-hex>           (certs[i] IEEE-754 f32 bit pattern, lowercase)
//   ur\t<8-hex>           (ratings[i] f32 bits)
//   ux\t<int>             (xcoords)
//
// Build:
//   g++ -std=c++17 recodebeam_oracle.cpp \
//     -I/tmp/tesseract-src/src/ccutil -I/tmp/tesseract-src/src/ccstruct \
//     -I/tmp/tesseract-src/src/dict -I/tmp/tesseract-src/src/lstm \
//     -I/tmp/tesseract-src/include -ltesseract -lleptonica -o /tmp/recodebeam_oracle
// Run:
//   /tmp/recodebeam_oracle <unicharset> <recoder> <probs.bin> <null_char> <simple 0|1>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "matrix.h"          // GENERIC_2D_ARRAY
#include "recodebeam.h"      // RecodeBeamSearch
#include "serialis.h"        // TFile
#include "unicharcompress.h" // UnicharCompress
#include "unicharset.h"      // UNICHARSET

using tesseract::GENERIC_2D_ARRAY;
using tesseract::RecodeBeamSearch;
using tesseract::TFile;
using tesseract::UnicharCompress;
using tesseract::UNICHARSET;

static std::vector<char> read_file(const char *path) {
  std::vector<char> data;
  FILE *f = fopen(path, "rb");
  if (!f)
    return data;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  data.resize(sz);
  if (sz > 0 && fread(data.data(), 1, sz, f) != (size_t)sz)
    data.clear();
  fclose(f);
  return data;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr,
            "usage: %s <unicharset> <recoder> <probs.bin> <null_char> <simple 0|1>\n",
            argv[0]);
    return 2;
  }
  const char *unicharset_path = argv[1];
  const char *recoder_path = argv[2];
  const char *bin_path = argv[3];
  int null_char = atoi(argv[4]);
  bool simple = (atoi(argv[5]) != 0);

  UNICHARSET unicharset;
  if (!unicharset.load_from_file(unicharset_path)) {
    fprintf(stderr, "load unicharset failed: %s\n", unicharset_path);
    return 1;
  }

  // Load the recoder (UnicharCompress) via TFile DeSerialize from the raw
  // extracted `.lstm-recoder` component (same bytes the Rust reads).
  std::vector<char> recoder_bytes = read_file(recoder_path);
  if (recoder_bytes.empty()) {
    fprintf(stderr, "read recoder failed: %s\n", recoder_path);
    return 1;
  }
  TFile rfp;
  rfp.Open(recoder_bytes.data(), recoder_bytes.size());
  UnicharCompress recoder;
  if (!recoder.DeSerialize(&rfp)) {
    fprintf(stderr, "recoder DeSerialize failed\n");
    return 1;
  }

  // Read the shared probs.bin: i32 T, i32 N, then T*N f32 LE.
  std::vector<char> raw = read_file(bin_path);
  if (raw.size() < 8) {
    fprintf(stderr, "read probs.bin failed: %s\n", bin_path);
    return 1;
  }
  int32_t T, N;
  memcpy(&T, raw.data(), 4);
  memcpy(&N, raw.data() + 4, 4);
  if ((size_t)raw.size() < (size_t)8 + (size_t)T * N * 4) {
    fprintf(stderr, "probs.bin too short: T=%d N=%d\n", T, N);
    return 1;
  }
  const float *flat = reinterpret_cast<const float *>(raw.data() + 8);

  GENERIC_2D_ARRAY<float> output(T, N, 0.0f);
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < N; ++c)
      output(t, c) = flat[t * N + c];

  // Real beam decode (non-dict: dict = nullptr), dict_ratio=1.0, cert_offset=0.0,
  // worst_dict_cert=0.0, charset=nullptr — exactly the Rust `decode(_, 1.0, 0.0)`.
  RecodeBeamSearch beam(recoder, null_char, simple, nullptr);
  beam.Decode(output, 1.0, 0.0, 0.0, nullptr);

  std::vector<int> labels, xcoords;
  beam.ExtractBestPathAsLabels(&labels, &xcoords);
  for (int label : labels)
    printf("label\t%d\n", label);
  for (int x : xcoords)
    printf("xcoord\t%d\n", x);

  std::vector<int> uids, uxcoords;
  std::vector<float> certs, ratings;
  beam.ExtractBestPathAsUnicharIds(false, &unicharset, &uids, &certs, &ratings,
                                   &uxcoords);
  for (int uid : uids)
    printf("uid\t%d\n", uid);
  for (float c : certs) {
    uint32_t bits;
    memcpy(&bits, &c, 4);
    printf("uc\t%08x\n", bits);
  }
  for (float r : ratings) {
    uint32_t bits;
    memcpy(&bits, &r, 4);
    printf("ur\t%08x\n", bits);
  }
  for (int x : uxcoords)
    printf("ux\t%d\n", x);
  return 0;
}
