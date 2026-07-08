// Byte-parity oracle for the recognizer leaf A6b: IMAGE FILE ON DISK -> text
// (the image-file-input variant of `recognize_grid_oracle.cpp`'s grid -> text
// core), vs the Rust transcode's `from_grey_pix` + `recognize_grid`.
//
// This is a MECHANICAL MODIFICATION of recognize_grid_oracle.cpp (itself a
// proven composition of network_forward_oracle.cpp +
// recodebeam_oracle.cpp/recoder_oracle.cpp). Steps 1-3 (load network, load
// charset, load recoder) and steps 5-8 (randomizer+Forward, beam decode,
// extract best path, build+dump text) are VERBATIM. The ONLY change is step
// 4: instead of building a synthetic grid from a shared .bin (an i32
// batch,height,width,depth header then f32s in StrideMap walk order), we
// `pixRead` a real image file and run it through the REAL
// `Input::PreparePixInput` -- exactly the call
// `LSTMRecognizer::RecognizeLine` makes (lstmrecognizer.cpp:347:
// `Input::PreparePixInput(network_->InputShape(), pix, &randomizer_,
// inputs);`), skipping only `Input::PrepareLSTMInputs`'s line-finding/
// auto-invert heuristics (out of scope for this leaf; the oracle is proving
// PreparePixInput's depth-convert + scale-to-target-height + FromPix, not
// line-finding).
//
// `Input::PreparePixInput` (input.cpp:107-144): converts the pix to 8-bit
// grey (already 8-bit here, so a `clone()`, no `pixConvertTo8`), then scales
// to `shape.height()` if that differs from the pix height. For our
// height-36 8-bit grey PGM against eng.lstm's Input shape (height=36,
// confirmed by network_forward_oracle.cpp / network_spec_oracle.cpp:
// `target_height == height` -> NO `pixScale` call), this is an
// identity-scale, so the only real work is `FromPix` -- already proven
// byte-parity green in frompix_oracle.cpp (E-OCR leaf A6a). This oracle
// proves the seam one level up: image FILE -> PreparePixInput -> Forward ->
// beam -> text, end to end.
//
// null_char=110 / simple_text=true / dict=nullptr: identical convention to
// recognize_grid_oracle.cpp. The randomizer is seeded via the REAL
// `LSTMRecognizer::SetRandomSeed()` (`(int64_t)sample_iteration_ * 0x10000001`
// then one `IntRand()` warm-up; eng.lstm's sample_iteration_ = 6352704 from the
// B2 DeSerialize dump) -- NOT a bare set_seed(1). This is NOT inert: while
// `PreparePixInput`'s FromPix makes no draws here (`shape.width()==0`, no
// width-padding), `Forward`'s `Convolve` DOES draw out-of-image noise from this
// randomizer, and that noise reaches the recognized text -- switching from
// set_seed(1) to SetRandomSeed changed the output "aLLiii," -> "qLLiy,,". So
// this oracle proves the Rust transcode reproduces the ACTUAL RecognizeLine
// (its real seeding), not merely "correct for an arbitrary seed".
//
//   ./image_text_oracle [eng.lstm] [eng.lstm-unicharset] [eng.lstm-recoder] [image.pgm]
#include "network.h"
#include "networkio.h"
#include "networkscratch.h"
#include "stridemap.h"
#include "helpers.h"
#include "serialis.h"
#include "unicharset.h"
#include "unicharcompress.h"
#include "recodebeam.h"
#include "input.h"
#include <cstring>
#include <tesseract/baseapi.h>
#include "tesseractclass.h"
#include "dict.h"
#include "pageres.h"
#include "ratngs.h"
#include "static_shape.h"

#include <allheaders.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tesseract;

int main(int argc, char **argv) {
  const char *lstm_path = argc > 1 ? argv[1] : "/tmp/eng.lstm";
  const char *uni_path = argc > 2 ? argv[2] : "/tmp/eng.lstm-unicharset";
  const char *rec_path = argc > 3 ? argv[3] : "/tmp/eng.lstm-recoder";
  const char *img_path = argc > 4 ? argv[4] : "/tmp/line36.pgm";

  // ---- 1. Load the network: VERBATIM from network_forward_oracle.cpp /
  // recognize_grid_oracle.cpp. ----
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
  fprintf(stderr, "nw=%d ni=%d no=%d\n", net->num_weights(), net->NumInputs(),
          net->NumOutputs());

  // ---- 2. Load the charset: UNICHARSET::load_from_file (single-arg
  // overload, defaults skip_fragments=false -- same convention as
  // recoder_oracle.cpp's `u.load_from_file(argv[1])`). ----
  UNICHARSET unicharset;
  if (!unicharset.load_from_file(uni_path)) {
    fprintf(stderr, "unicharset load failed: %s\n", uni_path);
    return 1;
  }
  fprintf(stderr, "unicharset size=%d\n", static_cast<int>(unicharset.size()));

  // ---- 3. Load the recoder: VERBATIM pattern from recoder_oracle.cpp /
  // recodebeam_oracle.cpp (LoadDataFromFile + TFile::Open + DeSerialize). ----
  std::vector<char> rec_data;
  if (!LoadDataFromFile(rec_path, &rec_data)) {
    fprintf(stderr, "recoder load failed: %s\n", rec_path);
    return 1;
  }
  TFile rec_fp;
  if (!rec_fp.Open(rec_data.data(), rec_data.size())) {
    fprintf(stderr, "recoder TFile::Open failed\n");
    return 1;
  }
  UnicharCompress recoder;
  if (!recoder.DeSerialize(&rec_fp)) {
    fprintf(stderr, "recoder DeSerialize failed\n");
    return 1;
  }
  fprintf(stderr, "recoder code_range=%d\n", recoder.code_range());

  // ---- 4. NEW for A6b: read the image file + run it through the REAL
  // Input::PreparePixInput -- replaces recognize_grid_oracle.cpp's synthetic
  // .bin grid build. `pixRead` (leptonica, allheaders.h) handles PGM/PNG/
  // etc. transparently. One `TRand` is declared here (seed 1, no warm-up
  // draw) and reused for BOTH PreparePixInput's randomizer arg and (below)
  // net->SetRandomizer before Forward -- matching
  // LSTMRecognizer::RecognizeLine's single-randomizer-instance shape
  // (lstmrecognizer.cpp:321-347), even though the concrete seed convention
  // differs from RecognizeLine's SetRandomSeed() (see file header). ----
  Image pix = pixRead(img_path);
  if (pix == nullptr) {
    fprintf(stderr, "pixRead failed: %s\n", img_path);
    return 1;
  }
  fprintf(stderr, "pix w=%d h=%d d=%d\n", pixGetWidth(pix), pixGetHeight(pix),
          pixGetDepth(pix));

  // LSTMRecognizer::SetRandomSeed (lstmrecognizer.h:287-291): the exact seeding
  // RecognizeLine uses -- seed = (int64_t)sample_iteration_ * 0x10000001, then
  // one IntRand() warm-up. eng.lstm's sample_iteration_ = 6352704 (the B2
  // DeSerialize trailing-field dump, /tmp/oracle_lstmrec.tsv "siter 6352704").
  TRand trand;
  int64_t rseed = static_cast<int64_t>(6352704) * 0x10000001;
  trand.set_seed(rseed);
  trand.IntRand();

  // Must set int_mode BEFORE (RecognizeLine does
  // inputs->set_int_mode(IsIntMode()) -- eng is int mode).
  NetworkIO input;
  input.set_int_mode(true);
  // Build the input grid via the REAL Input::PreparePixInput (depth-convert
  // + scale-to-target-height + FromPix), using the REAL network's declared
  // input shape -- exactly network_->InputShape() as RecognizeLine calls it.
  const int pix_w_saved = pixGetWidth(pix);
  const int pix_h_saved = pixGetHeight(pix);
  Input::PreparePixInput(net->InputShape(), pix, &trand, &input);
  pix.destroy();
  fprintf(stderr, "input width=%d features=%d int_mode=%d\n", input.Width(),
          input.NumFeatures(), input.int_mode() ? 1 : 0);

  // ---- 5. Randomizer + Forward: SAME trand object as step 4 (PreparePixInput
  // may consult it for Copy2DImage's width-padding noise branch, but a
  // full-width image -- shape.width()==0, dynamic -- makes no draws, so the
  // randomizer entering Forward is still fresh seed-1, matching the
  // established network_forward_oracle.cpp / recognize_grid_oracle.cpp
  // convention). ----
  net->SetRandomizer(&trand);

  NetworkScratch scratch;
  NetworkIO outputs;
  net->Forward(false, input, nullptr, &scratch, &outputs);
  {
    double s = 0.0; int cnt = 0;
    for (int t = 0; t < outputs.Width(); ++t) {
      const float *row = outputs.f(t);
      for (int f = 0; f < outputs.NumFeatures(); ++f, ++cnt) s += row[f];
    }
    fprintf(stderr, "ARGMAX ");
    for (int t = 0; t < outputs.Width(); ++t) {
      const float *row = outputs.f(t); int best = 0;
      for (int f = 1; f < outputs.NumFeatures(); ++f) if (row[f] > row[best]) best = f;
      fprintf(stderr, "%d,", best);
    }
    fprintf(stderr, "\n");
    const float *r0 = outputs.f(0);
    fprintf(stderr, "SUM336=%.9g f0..f5= %.6g %.6g %.6g %.6g %.6g %.6g\n", s,
            r0[0], r0[1], r0[2], r0[3], r0[4], r0[5]);
  }
  fprintf(stderr, "oshape width=%d features=%d int_mode=%d\n",
          outputs.Width(), outputs.NumFeatures(), outputs.int_mode() ? 1 : 0);

  // ---- 6. Beam decode: the REAL non-dict CTC beam
  // (RecodeBeamSearch::Decode(const NetworkIO&, ...) overload, which reads
  // via output.f(t) -- requires int_mode()==false, satisfied since the
  // final FC layer's softmax activation always produces float). ----
  const int null_char = 110; // eng.lstm's real DeSerialize'd value
  const bool simple_text = false; // OutputLossType()==LT_SOFTMAX for eng.lstm
  // DICT VARIANT: argv[5] = tessdata dir ("nodict" => bare beam self-check).
  tesseract::Dict *dict = nullptr;
  tesseract::TessBaseAPI api;
  const char *tessdata = (argc > 5) ? argv[5] : "nodict";
  if (strcmp(tessdata, "nodict") != 0) {
    if (api.Init(tessdata, "eng", tesseract::OEM_LSTM_ONLY) != 0) {
      fprintf(stderr, "api.Init failed for %s\n", tessdata);
      return 1;
    }
    dict = &api.tesseract()->getDict();
    fprintf(stderr, "dict loaded: NumDawgs=%d\n", dict->NumDawgs());
  }
  RecodeBeamSearch beam(recoder, null_char, simple_text, dict);
  // Production params captured LIVE from the CLI via gdb on this box:
  // dict_ratio=2.25 cert_offset=-0.085 worst_dict_cert=-25/7=-3.5714286.
  if (dict != nullptr) {
    beam.Decode(outputs, 2.25, -0.085, -25.0 / 7.0, &unicharset, 0);
  } else {
    beam.Decode(outputs, 1.0, 0.0, 0.0, &unicharset, 0);
  }

  // ---- 7b. DICT VARIANT: ALSO extract as WORDS (the CLI's actual text
  // source -- dawg-committed word chains preferred). ----
  if (dict != nullptr) {
    PointerVector<WERD_RES> words;
    TBOX line_box(0, 0, pix_w_saved, pix_h_saved);
    beam.ExtractBestPathAsWords(line_box, 1.0f, false, &unicharset, &words);
    std::string wtext;
    for (unsigned wi = 0; wi < words.size(); ++wi) {
      WERD_RES *w = words[wi];
      if (w->best_choice != nullptr) {
        if (!wtext.empty()) wtext += " ";
        wtext += w->best_choice->unichar_string().c_str();
      }
    }
    printf("wordstext\t%s\n", wtext.c_str());
  }

  // ---- 7. Extract best path as unichar ids: the REAL public API. ----
  std::vector<int> unichar_ids, xcoords;
  std::vector<float> certs, ratings;
  beam.ExtractBestPathAsUnicharIds(false, &unicharset, &unichar_ids, &certs,
                                    &ratings, &xcoords);

  // ---- 8. Build text: id_to_unichar concatenation. ----
  std::string text;
  for (int uid : unichar_ids) {
    text += unicharset.id_to_unichar(uid);
  }

  // ---- Dump: EXACTLY tab-separated. ----
  printf("uids");
  for (int uid : unichar_ids) {
    printf("\t%d", uid);
  }
  printf("\n");
  printf("text\t%s\n", text.c_str());
  return 0;
}
