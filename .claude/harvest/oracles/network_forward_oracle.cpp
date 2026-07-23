// Byte-parity oracle for the FULL network-tree forward pass vs the Rust
// transcode's `Network::forward` (tesseract-ocr/examples/network_dump.rs).
//
// Loads the real X.lstm via the REAL public `Network::CreateFromFile`,
// builds the same synthetic int8 input grid (read from the shared
// net_input.bin, written in exact StrideMap walk order), seeds the REAL
// `TRand` identically (seed 1, no warm-up draw), and calls the REAL
// polymorphic `Network::Forward` — which dispatches through Series/
// Convolve/Maxpool/Txy(Reversed)/LSTM/FullyConnected exactly as the C++
// library always has. Dumps the same `oshape` / `o` lines as the Rust side
// for a byte-identical diff. Public-API only → dodges the 5.3.4-lib /
// header ABI skew. Model-agnostic (eng, deu, …) — nothing eng-specific.
//
// Banked (2026-07-23) from recognizer-image-to-text-v2.md §B1 "Oracle source".
//
//   ./network_forward_oracle [X.lstm] [net_input.bin]
#include "network.h"
#include "networkio.h"
#include "networkscratch.h"
#include "stridemap.h"
#include "helpers.h"
#include "serialis.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace tesseract;

int main(int argc, char **argv) {
  const char *lstm_path = argc > 1 ? argv[1] : "/tmp/eng.lstm";
  const char *bin_path = argc > 2 ? argv[2] : "/tmp/net_input.bin";

  // ---- Load the network (the REAL recursive Network::CreateFromFile) ----
  std::vector<char> data;
  if (!LoadDataFromFile(lstm_path, &data)) {
    fprintf(stderr, "load fail: %s\n", lstm_path);
    return 1;
  }
  TFile fp;
  if (!fp.Open(data.data(), data.size())) {
    fprintf(stderr, "TFile::Open failed\n");
    return 1;
  }
  Network *net = Network::CreateFromFile(&fp);
  if (!net) {
    fprintf(stderr, "CreateFromFile failed\n");
    return 1;
  }
  printf("nw\t%d\n", net->num_weights());
  printf("ni\t%d\tno\t%d\n", net->NumInputs(), net->NumOutputs());

  // ---- Load the shared input .bin: i32 batch,height,width,depth then the
  // f32s in the exact StrideMap walk order (written by the Rust side). ----
  FILE *bf = fopen(bin_path, "rb");
  if (!bf) {
    fprintf(stderr, "open %s failed\n", bin_path);
    return 1;
  }
  auto read_i32 = [&](int32_t *v) {
    if (fread(v, sizeof(int32_t), 1, bf) != 1) {
      fprintf(stderr, "bin truncated (header)\n");
      exit(1);
    }
  };
  int32_t batch = 0, height = 0, width = 0, depth = 0;
  read_i32(&batch);
  read_i32(&height);
  read_i32(&width);
  read_i32(&depth);

  StrideMap map;
  std::vector<std::pair<int, int>> hw = {{static_cast<int>(height), static_cast<int>(width)}};
  map.SetStride(hw);

  NetworkIO input;
  input.ResizeToMap(true, map, depth);

  {
    StrideMap::Index idx(map);
    std::vector<float> vals(static_cast<size_t>(depth));
    do {
      int t = idx.t();
      if (depth > 0 &&
          fread(vals.data(), sizeof(float), static_cast<size_t>(depth), bf) !=
              static_cast<size_t>(depth)) {
        fprintf(stderr, "bin truncated at t=%d\n", t);
        exit(1);
      }
      input.WriteTimeStep(t, vals.data());
    } while (idx.Increment());
  }
  fclose(bf);

  // ---- Randomizer: seed 1, NO warm-up draw — matches the Rust side exactly.
  // Set BEFORE Forward: Convolve pulls out-of-image noise from it. Plumbing
  // ::SetRandomizer recurses into every child (Series/Reversed/...), so one
  // call at the root reaches the Convolve node wherever it sits in the tree.
  TRand trand;
  trand.set_seed(1);
  net->SetRandomizer(&trand);

  // ---- Forward: the REAL polymorphic dispatch through the whole tree. ----
  NetworkScratch scratch;
  NetworkIO output;
  net->Forward(false, input, nullptr, &scratch, &output);

  // ---- Dump: byte-identical format to the Rust side's oshape/o lines. ----
  printf("oshape\t%d\t%d\t%d\t%d\t%d\n", output.stride_map().Size(FD_BATCH),
         output.stride_map().Size(FD_HEIGHT), output.stride_map().Size(FD_WIDTH),
         output.Width(), output.NumFeatures());
  for (int t = 0; t < output.Width(); ++t) {
    printf("o\t%d", t);
    if (output.int_mode()) {
      const int8_t *row = output.i(t);
      for (int f = 0; f < output.NumFeatures(); ++f) {
        printf("\t%d", static_cast<int>(row[f]));
      }
    } else {
      const float *row = output.f(t);
      for (int f = 0; f < output.NumFeatures(); ++f) {
        uint32_t u;
        float v = row[f];
        std::memcpy(&u, &v, 4);
        printf("\t%08x", u);
      }
    }
    printf("\n");
  }
  return 0;
}
