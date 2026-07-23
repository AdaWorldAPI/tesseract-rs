// Byte-parity oracle for the recognizer capstone: IMAGE FILE ON DISK -> text,
// vs the Rust transcode's `LstmRecognizer::recognize_image_file` (non-dict),
// dumped by tesseract-ocr/examples/recognize_image_dump.rs.
//
// MODEL-AGNOSTIC variant of image_text_oracle_ctc.cpp (banked 2026-07-08).
// The banked oracle HARDCODED eng.lstm's sample_iteration_ (6352704) and
// null_char (110); this one SELF-READS them from the model, so it is byte-
// parity correct for ANY split-traineddata model (eng, deu, ...) with ZERO
// per-model constants. That matches the Rust side, whose `from_components`
// parses those same trailing fields (B2 / E-OCR-RECOGNIZER-LOAD-1) and whose
// `recognize_image_file` seeds the randomizer from the model's own
// sample_iteration and derives simple_text from the loaded net.
//
// How the model-agnostic fields are recovered (identical to
// LSTMRecognizer::DeSerialize, lstmrecognizer.cpp): after the network,
// the raw X.lstm component's tail is
//   network_str_(std::string) + i32{training_flags, training_iteration,
//   sample_iteration, null_char} + f32{adam_beta, learning_rate, momentum}
// (split-traineddata path; the embedded charset is skipped because the
// unicharset+recoder are their own components). We read the first five and
// the three floats to leave fp consistent (only siter/null/flags are used).
//
//   - seed  = (int64_t)sample_iteration * 0x10000001, then one IntRand()
//             warm-up  (== LSTMRecognizer::SetRandomSeed, lstmrecognizer.h)
//   - null_char        (== the RecodeBeamSearch null; deu != eng)
//   - int_mode  = (training_flags & TF_INT_MODE)      (TF_INT_MODE=1)
//   - simple_text = net->OutputShape(...).loss_type() == LT_SOFTMAX
//             (== LSTMRecognizer::SimpleTextOutput; eng.lstm/deu.lstm are
//              NT_SOFTMAX => LT_CTC => simple_text=false, full CTC collapse)
//
// Steps 4-8 (pixRead + Input::PreparePixInput + Forward + non-dict CTC beam +
// ExtractBestPathAsUnicharIds + id->text) are VERBATIM from the banked oracle.
//
//   ./image_text_agnostic_oracle <X.lstm> <X.lstm-unicharset> <X.lstm-recoder> <image.pgm>
#include "network.h"
#include "networkio.h"
#include "networkscratch.h"
#include "stridemap.h"
#include "helpers.h"
#include "serialis.h"
#include "static_shape.h"
#include "unicharset.h"
#include "unicharcompress.h"
#include "recodebeam.h"
#include "input.h"

#include <allheaders.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tesseract;

// TrainingFlags::TF_INT_MODE (lstmrecognizer.h:45).
static const int32_t kTfIntMode = 1;

int main(int argc, char **argv) {
  const char *lstm_path = argc > 1 ? argv[1] : "/tmp/eng.lstm";
  const char *uni_path = argc > 2 ? argv[2] : "/tmp/eng.lstm-unicharset";
  const char *rec_path = argc > 3 ? argv[3] : "/tmp/eng.lstm-recoder";
  const char *img_path = argc > 4 ? argv[4] : "/tmp/line36.pgm";

  // ---- 1. Load the network (REAL recursive Network::CreateFromFile). ----
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

  // ---- 1b. NEW (model-agnostic): read the trailing scalar fields exactly as
  // LSTMRecognizer::DeSerialize does, straight off the same fp. ----
  std::string network_str;
  int32_t training_flags = 0, training_iteration = 0, sample_iteration = 0,
          null_char = 0;
  float adam_beta = 0.f, learning_rate = 0.f, momentum = 0.f;
  if (!fp.DeSerialize(network_str) || !fp.DeSerialize(&training_flags) ||
      !fp.DeSerialize(&training_iteration) || !fp.DeSerialize(&sample_iteration) ||
      !fp.DeSerialize(&null_char) || !fp.DeSerialize(&adam_beta) ||
      !fp.DeSerialize(&learning_rate) || !fp.DeSerialize(&momentum)) {
    fprintf(stderr, "trailing-field DeSerialize failed\n");
    return 1;
  }
  const bool int_mode = (training_flags & kTfIntMode) != 0;
  fprintf(stderr, "nw=%d ni=%d no=%d spec=%s\n", net->num_weights(),
          net->NumInputs(), net->NumOutputs(), net->spec().c_str());
  fprintf(stderr, "siter=%d null_char=%d tflags=%d int_mode=%d\n",
          sample_iteration, null_char, training_flags, int_mode);

  // ---- 2. Load the charset. ----
  UNICHARSET unicharset;
  if (!unicharset.load_from_file(uni_path)) {
    fprintf(stderr, "unicharset load failed: %s\n", uni_path);
    return 1;
  }

  // ---- 3. Load the recoder. ----
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

  // ---- 4. Read the image + REAL Input::PreparePixInput (VERBATIM). ----
  Image pix = pixRead(img_path);
  if (pix == nullptr) {
    fprintf(stderr, "pixRead failed: %s\n", img_path);
    return 1;
  }
  fprintf(stderr, "pix w=%d h=%d d=%d\n", pixGetWidth(pix), pixGetHeight(pix),
          pixGetDepth(pix));

  // LSTMRecognizer::SetRandomSeed: seed = (int64_t)sample_iteration_ *
  // 0x10000001, then one IntRand() warm-up. MODEL-DERIVED (not hardcoded).
  TRand trand;
  int64_t rseed = static_cast<int64_t>(sample_iteration) * 0x10000001;
  trand.set_seed(rseed);
  trand.IntRand();

  NetworkIO input;
  input.set_int_mode(int_mode);  // RecognizeLine: inputs->set_int_mode(IsIntMode())
  Input::PreparePixInput(net->InputShape(), pix, &trand, &input);
  pix.destroy();
  fprintf(stderr, "input width=%d features=%d int_mode=%d\n", input.Width(),
          input.NumFeatures(), input.int_mode() ? 1 : 0);

  // ---- 5. Randomizer + Forward (same trand instance). ----
  net->SetRandomizer(&trand);
  NetworkScratch scratch;
  NetworkIO outputs;
  net->Forward(false, input, nullptr, &scratch, &outputs);

  // ---- 6. Beam decode: non-dict CTC beam. simple_text MODEL-DERIVED via
  // net->OutputShape(...).loss_type() (== LSTMRecognizer::SimpleTextOutput). --
  StaticShape oshape;
  oshape = net->OutputShape(oshape);
  const bool simple_text = (oshape.loss_type() == LT_SOFTMAX);
  fprintf(stderr, "loss_type=%d simple_text=%d\n",
          static_cast<int>(oshape.loss_type()), simple_text);
  RecodeBeamSearch beam(recoder, null_char, simple_text, /*dict=*/nullptr);
  beam.Decode(outputs, 1.0, 0.0, 0.0, &unicharset, 0);

  // ---- 7. Extract best path as unichar ids (REAL public API). ----
  std::vector<int> unichar_ids, xcoords;
  std::vector<float> certs, ratings;
  beam.ExtractBestPathAsUnicharIds(false, &unicharset, &unichar_ids, &certs,
                                   &ratings, &xcoords);

  // ---- 8. Build text + dump (byte-identical to recognize_image_dump.rs). --
  std::string text;
  for (int uid : unichar_ids) {
    text += unicharset.id_to_unichar(uid);
  }
  printf("uids");
  for (int uid : unichar_ids) {
    printf("\t%d", uid);
  }
  printf("\n");
  printf("text\t%s\n", text.c_str());
  return 0;
}
