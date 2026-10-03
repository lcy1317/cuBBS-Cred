#include "cubbs_bbs_gpu_abi.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" int32_t cubbs_bbs_gpu_expand_xmd_sha256_48_batch_for_test_v2(
    uint32_t abi_version, const uint8_t *messages, const size_t *offsets,
    const size_t *lengths, size_t item_count, const uint8_t *dst, size_t dst_len,
    uint8_t *uniform_bytes);

extern "C" int32_t cubbs_bbs_gpu_messages_to_scalar_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuMessagesToScalarItem *items, size_t item_count,
    CuBbsBbsGpuResult *results);

static std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
  std::vector<uint8_t> out;
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

static std::string hex_bytes(const uint8_t *bytes, size_t len) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < len; ++i) {
    out << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return out.str();
}

static int run_case_set(const std::string &name, const std::string &dst_hex,
                        const std::vector<std::string> &message_hex,
                        const std::vector<std::string> &expected) {
  const std::vector<uint8_t> dst = hex_to_bytes(dst_hex);

  std::vector<uint8_t> packed;
  std::vector<size_t> offsets;
  std::vector<size_t> lengths;
  for (const auto &hex : message_hex) {
    const auto bytes = hex_to_bytes(hex);
    offsets.push_back(packed.size());
    lengths.push_back(bytes.size());
    packed.insert(packed.end(), bytes.begin(), bytes.end());
  }

  std::vector<uint8_t> scalars(message_hex.size() * 32u);
  std::vector<CuBbsBbsGpuMessagesToScalarItem> items(message_hex.size());
  std::vector<CuBbsBbsGpuResult> results(message_hex.size());
  for (size_t i = 0; i < message_hex.size(); ++i) {
    items[i].message = CuBbsBbsGpuByteSlice{packed.data() + offsets[i], lengths[i]};
    items[i].dst = CuBbsBbsGpuByteSlice{dst.data(), dst.size()};
    items[i].scalar_out = CuBbsBbsGpuByteBuffer{scalars.data() + (i * 32u), 32u, 0};
  }
  const int32_t status = cubbs_bbs_gpu_messages_to_scalar_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, items.data(), items.size(), results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    std::cerr << "hash-to-scalar status=" << status << "\n";
    return 1;
  }

  for (size_t i = 0; i < message_hex.size(); ++i) {
    if (results[i].code != CUBBS_BBS_GPU_SUCCESS || results[i].bytes_written != 32u) {
      std::cerr << "case " << i << " result code=" << results[i].code << "\n";
      return 1;
    }
    const std::string actual = hex_bytes(scalars.data() + (i * 32u), 32);
    if (actual != expected[i]) {
      std::cerr << "case " << i << " mismatch\nexpected " << expected[i] << "\nactual   "
                << actual << "\n";
      return 1;
    }
  }
  std::cout << "cuda hash_to_scalar sha256 " << name << " ok: " << message_hex.size()
            << " vectors\n";
  return 0;
}

int main() {
  if (run_case_set(
          "h2s",
          "4242535f424c53313233383147315f584d443a5348412d3235365f535357555f524f5f4832475f484d32535f4832535f",
          {"9872ad089e452c7b6e283dfac2a80d58e8d0ff71cc4d5e310a1debdda4a45f02"},
          {"0f90cbee27beb214e6545becb8404640d3612da5d6758dffeccd77ed7169807c"}) != 0) {
    return 1;
  }

  if (run_case_set(
          "map-message-to-scalar",
          "4242535f424c53313233383147315f584d443a5348412d3235365f535357555f524f5f4832475f484d32535f4d41505f4d53475f544f5f5343414c41525f41535f484153485f",
          {"9872ad089e452c7b6e283dfac2a80d58e8d0ff71cc4d5e310a1debdda4a45f02",
           "c344136d9ab02da4dd5908bbba913ae6f58c2cc844b802a6f811f5fb075f9b80",
           "7372e9daa5ed31e6cd5c825eac1b855e84476a1d94932aa348e07b73",
           "77fe97eb97a1ebe2e81e4e3597a3ee740a66e9ef2412472c",
           "496694774c5604ab1b2544eababcf0f53278ff50",
           "515ae153e22aae04ad16f759e07237b4",
           "d183ddc6e2665aa4e2f088af",
           "ac55fb33a75909ed",
           "96012096",
           ""},
          {"1cb5bb86114b34dc438a911617655a1db595abafac92f47c5001799cf624b430",
           "154249d503c093ac2df516d4bb88b510d54fd97e8d7121aede420a25d9521952",
           "0c7c4c85cdab32e6fdb0de267b16fa3212733d4e3a3f0d0f751657578b26fe22",
           "4a196deafee5c23f630156ae13be3e46e53b7e39094d22877b8cba7f14640888",
           "34c5ea4f2ba49117015a02c711bb173c11b06b3f1571b88a2952b93d0ed4cf7e",
           "4045b39b83055cd57a4d0203e1660800fabe434004dbdc8730c21ce3f0048b08",
           "064621da4377b6b1d05ecc37cf3b9dfc94b9498d7013dc5c4a82bf3bb1750743",
           "34ac9196ace0a37e147e32319ea9b3d8cc7d21870d3c3ba071246859cca49b02",
           "57eb93f417c43200e9784fa5ea5a59168d3dbc38df707a13bb597c871b2a5f74",
           "08e3afeb2b4f2b5f907924ef42856616e6f2d5f1fb373736db1cca32707a7d16"}) != 0) {
    return 1;
  }

  return 0;
}
