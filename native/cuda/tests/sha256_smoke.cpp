#include "cubbs_bbs_gpu_abi.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" int32_t cubbs_bbs_gpu_sha256_batch_for_test_v2(
    uint32_t abi_version, const uint8_t *messages, const size_t *offsets,
    const size_t *lengths, size_t item_count, uint8_t *digests);

static std::string hex_digest(const uint8_t *digest) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (int i = 0; i < 32; ++i) {
    out << std::setw(2) << static_cast<unsigned>(digest[i]);
  }
  return out.str();
}

int main() {
  const std::vector<std::string> messages = {
      "",
      "abc",
      "hello world",
      "The quick brown fox jumps over the lazy dog",
      "The quick brown fox jumps over the lazy dog."};
  const std::vector<std::string> expected = {
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
      "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
      "ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c"};

  std::vector<uint8_t> packed;
  std::vector<size_t> offsets;
  std::vector<size_t> lengths;
  for (const auto &message : messages) {
    offsets.push_back(packed.size());
    lengths.push_back(message.size());
    packed.insert(packed.end(), message.begin(), message.end());
  }

  std::vector<uint8_t> digests(messages.size() * 32u);
  const int32_t status = cubbs_bbs_gpu_sha256_batch_for_test_v2(
      CUBBS_BBS_GPU_ABI_VERSION, packed.data(), offsets.data(), lengths.data(),
      messages.size(), digests.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    std::cerr << "sha256 batch status=" << status << "\n";
    return 1;
  }

  for (size_t i = 0; i < messages.size(); ++i) {
    const std::string actual = hex_digest(digests.data() + (i * 32u));
    if (actual != expected[i]) {
      std::cerr << "case " << i << " mismatch\nexpected " << expected[i] << "\nactual   "
                << actual << "\n";
      return 1;
    }
  }

  std::cout << "cuda sha256 batch smoke ok: " << messages.size() << " vectors\n";
  return 0;
}
