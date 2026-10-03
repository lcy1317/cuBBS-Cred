#include "cubbs_bbs_gpu_abi.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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

static double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t index = std::min(values.size() - 1, static_cast<size_t>((values.size() - 1) * p));
  return values[index];
}

static void run_case(size_t batch_size, size_t message_len, size_t iterations,
                     const std::vector<uint8_t> &dst) {
  std::vector<uint8_t> packed(batch_size * message_len);
  std::vector<size_t> offsets(batch_size);
  std::vector<size_t> lengths(batch_size, message_len);
  std::vector<uint8_t> scalars(batch_size * 32u);
  std::vector<CuBbsBbsGpuMessagesToScalarItem> items(batch_size);
  std::vector<CuBbsBbsGpuResult> results(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    offsets[i] = i * message_len;
    for (size_t j = 0; j < message_len; ++j) {
      packed[offsets[i] + j] = static_cast<uint8_t>((i * 131u + j * 17u + 29u) & 0xffu);
    }
    items[i].message = CuBbsBbsGpuByteSlice{packed.data() + offsets[i], message_len};
    items[i].dst = CuBbsBbsGpuByteSlice{dst.data(), dst.size()};
    items[i].scalar_out = CuBbsBbsGpuByteBuffer{scalars.data() + (i * 32u), 32u, 0};
  }

  // Warm up CUDA context and allocator path.
  int32_t status = cubbs_bbs_gpu_messages_to_scalar_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, items.data(), items.size(), results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    std::cerr << "warmup failed status=" << status << "\n";
    std::exit(1);
  }

  std::vector<double> total_ms;
  total_ms.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    status = cubbs_bbs_gpu_messages_to_scalar_batch_v2(
        CUBBS_BBS_GPU_ABI_VERSION, items.data(), items.size(), results.data());
    const auto end = std::chrono::steady_clock::now();
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      std::cerr << "bench failed status=" << status << "\n";
      std::exit(1);
    }
    total_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  const double sum = std::accumulate(total_ms.begin(), total_ms.end(), 0.0);
  const double avg_ms = sum / static_cast<double>(total_ms.size());
  const double p50_ms = percentile(total_ms, 0.50);
  const double p95_ms = percentile(total_ms, 0.95);
  const double p99_ms = percentile(total_ms, 0.99);
  const double ops_per_sec = (static_cast<double>(batch_size) * 1000.0) / avg_ms;
  const double item_p50_us = (p50_ms * 1000.0) / static_cast<double>(batch_size);

  std::cout << std::setw(8) << batch_size << " | " << std::setw(7) << message_len << " | "
            << std::setw(10) << std::fixed << std::setprecision(3) << avg_ms << " | "
            << std::setw(10) << p50_ms << " | " << std::setw(10) << p95_ms << " | "
            << std::setw(10) << p99_ms << " | " << std::setw(11) << std::setprecision(2)
            << ops_per_sec << " | " << std::setw(12) << item_p50_us << "\n";
}

int main(int argc, char **argv) {
  size_t iterations = 64;
  if (argc > 1) {
    iterations = static_cast<size_t>(std::stoul(argv[1]));
  }

  const auto dst = hex_to_bytes(
      "4242535f424c53313233383147315f584d443a5348412d3235365f535357555f524f5f4832475f484d32535f4d41505f4d53475f544f5f5343414c41525f41535f484153485f");

  std::cout << "batch    | msg_len | avg_ms     | p50_ms     | p95_ms     | p99_ms     | scalars/s   | p50_us/item\n";
  std::cout << "---------|---------|------------|------------|------------|------------|-------------|------------\n";
  for (size_t message_len : {32u, 128u}) {
    for (size_t batch_size : {1u, 16u, 64u, 256u, 1024u, 4096u}) {
      run_case(batch_size, message_len, iterations, dst);
    }
  }
  return 0;
}
