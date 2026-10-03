#include "cubbs_bbs_gpu_abi.h"
#include "prepare_b_msm.cuh"
#include "sha256.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <thread>
#include <vector>

__global__ void cubbs_bbs_canary_kernel(uint32_t *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = 0x0BB52023u;
  }
}

static bool abi_ok(uint32_t abi_version) {
  return abi_version == CUBBS_BBS_GPU_ABI_VERSION;
}

static size_t env_size_or_default(const char *name, size_t default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

static size_t fused_group_worker_count(size_t group_count) {
  const char *configured_env = std::getenv("CUBBS_BBS_FUSED_GROUP_WORKERS");
  if (configured_env == nullptr || configured_env[0] == '\0' ||
      group_count < env_size_or_default("CUBBS_BBS_FUSED_GROUP_PARALLEL_MIN_GROUPS",
                                        2u)) {
    return 1u;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  const size_t default_workers =
      hardware == 0u ? 4u : std::min<size_t>(4u, static_cast<size_t>(hardware));
  const size_t configured =
      env_size_or_default("CUBBS_BBS_FUSED_GROUP_WORKERS", default_workers);
  return std::max<size_t>(1u, std::min(group_count, configured));
}

extern "C" int32_t cubbs_bbs_gpu_sha256_batch_for_test_v2(
    uint32_t abi_version, const uint8_t *messages, const size_t *offsets,
    const size_t *lengths, size_t item_count, uint8_t *digests) {
  if (!abi_ok(abi_version) || (item_count > 0 && messages == nullptr) ||
      (item_count > 0 && offsets == nullptr) || (item_count > 0 && lengths == nullptr) ||
      (item_count > 0 && digests == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  size_t total_len = 0;
  for (size_t i = 0; i < item_count; ++i) {
    const size_t end = offsets[i] + lengths[i];
    if (end < offsets[i]) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (end > total_len) {
      total_len = end;
    }
  }

  uint8_t *device_messages = nullptr;
  size_t *device_offsets = nullptr;
  size_t *device_lengths = nullptr;
  uint8_t *device_digests = nullptr;
  cudaError_t status = cudaMalloc(&device_messages, total_len);
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_offsets, item_count * sizeof(size_t));
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_lengths, item_count * sizeof(size_t));
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_digests, item_count * 32u);
  }
  if (status == cudaSuccess && total_len > 0) {
    status = cudaMemcpy(device_messages, messages, total_len, cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(device_offsets, offsets, item_count * sizeof(size_t),
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(device_lengths, lengths, item_count * sizeof(size_t),
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cubbs_bbs_sha256_batch_host(device_messages, device_offsets, device_lengths,
                                          item_count, device_digests);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(digests, device_digests, item_count * 32u, cudaMemcpyDeviceToHost);
  }

  cudaFree(device_digests);
  cudaFree(device_lengths);
  cudaFree(device_offsets);
  cudaFree(device_messages);

  return status == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS : CUBBS_BBS_GPU_ERR_INTERNAL;
}

extern "C" int32_t cubbs_bbs_gpu_expand_xmd_sha256_48_batch_for_test_v2(
    uint32_t abi_version, const uint8_t *messages, const size_t *offsets,
    const size_t *lengths, size_t item_count, const uint8_t *dst, size_t dst_len,
    uint8_t *uniform_bytes) {
  if (!abi_ok(abi_version) || dst_len > 255 || (dst_len > 0 && dst == nullptr) ||
      (item_count > 0 && messages == nullptr) || (item_count > 0 && offsets == nullptr) ||
      (item_count > 0 && lengths == nullptr) || (item_count > 0 && uniform_bytes == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  size_t total_len = 0;
  for (size_t i = 0; i < item_count; ++i) {
    const size_t end = offsets[i] + lengths[i];
    if (end < offsets[i]) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (end > total_len) {
      total_len = end;
    }
  }

  uint8_t *device_messages = nullptr;
  size_t *device_offsets = nullptr;
  size_t *device_lengths = nullptr;
  uint8_t *device_dst = nullptr;
  uint8_t *device_uniform = nullptr;
  cudaError_t status = cudaMalloc(&device_messages, total_len);
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_offsets, item_count * sizeof(size_t));
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_lengths, item_count * sizeof(size_t));
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_dst, dst_len == 0 ? 1 : dst_len);
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_uniform, item_count * 48u);
  }
  if (status == cudaSuccess && total_len > 0) {
    status = cudaMemcpy(device_messages, messages, total_len, cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(device_offsets, offsets, item_count * sizeof(size_t),
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(device_lengths, lengths, item_count * sizeof(size_t),
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess && dst_len > 0) {
    status = cudaMemcpy(device_dst, dst, dst_len, cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cubbs_bbs_expand_xmd_sha256_48_batch_host(
        device_messages, device_offsets, device_lengths, item_count, device_dst, dst_len,
        device_uniform);
  }
  if (status == cudaSuccess) {
    status =
        cudaMemcpy(uniform_bytes, device_uniform, item_count * 48u, cudaMemcpyDeviceToHost);
  }

  cudaFree(device_uniform);
  cudaFree(device_dst);
  cudaFree(device_lengths);
  cudaFree(device_offsets);
  cudaFree(device_messages);

  return status == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS : CUBBS_BBS_GPU_ERR_INTERNAL;
}

struct CuBbsHashToScalarWorkspace {
  uint8_t *messages{nullptr};
  size_t *offsets{nullptr};
  size_t *lengths{nullptr};
  uint8_t *dst{nullptr};
  uint8_t *scalars{nullptr};
  uint8_t *sign_contexts{nullptr};
  uint8_t *sign_e_scalars{nullptr};
  size_t message_capacity{0};
  size_t item_capacity{0};
  size_t dst_capacity{0};
  size_t sign_context_capacity{0};
  size_t sign_e_scalar_capacity{0};
  size_t last_scalar_count{0};
  cudaStream_t stream{nullptr};

  CuBbsHashToScalarWorkspace() {
    (void)cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }

  ~CuBbsHashToScalarWorkspace() {
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    cudaFree(sign_e_scalars);
    cudaFree(sign_contexts);
    cudaFree(scalars);
    cudaFree(dst);
    cudaFree(lengths);
    cudaFree(offsets);
    cudaFree(messages);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }
};

static CuBbsHashToScalarWorkspace &cubbs_hash_to_scalar_workspace() {
  static thread_local CuBbsHashToScalarWorkspace workspace;
  return workspace;
}

extern "C" int32_t cubbs_bbs_gpu_hash_to_scalar_sha256_batch_for_test_v2(
    uint32_t abi_version, const uint8_t *messages, const size_t *offsets,
    const size_t *lengths, size_t item_count, const uint8_t *dst, size_t dst_len,
    uint8_t *scalars) {
  if (!abi_ok(abi_version) || dst_len > 255 || (dst_len > 0 && dst == nullptr) ||
      (item_count > 0 && messages == nullptr) || (item_count > 0 && offsets == nullptr) ||
      (item_count > 0 && lengths == nullptr) || (item_count > 0 && scalars == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  auto &workspace = cubbs_hash_to_scalar_workspace();
  const char *nonblocking_env = std::getenv("CUBBS_BBS_HASH_NONBLOCKING");
  const bool nonblocking = nonblocking_env == nullptr ||
      (std::strcmp(nonblocking_env, "0") != 0 &&
       std::strcmp(nonblocking_env, "false") != 0 &&
       std::strcmp(nonblocking_env, "FALSE") != 0);

  size_t total_len = 0;
  for (size_t i = 0; i < item_count; ++i) {
    const size_t end = offsets[i] + lengths[i];
    if (end < offsets[i]) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (end > total_len) {
      total_len = end;
    }
  }

  cudaError_t status = cudaSuccess;
  const size_t required_messages = std::max<size_t>(1u, total_len);
  if (workspace.message_capacity < required_messages) {
    cudaFree(workspace.messages);
    workspace.messages = nullptr;
    workspace.message_capacity = 0;
    status = cudaMalloc(&workspace.messages, required_messages);
    if (status == cudaSuccess) workspace.message_capacity = required_messages;
  }
  if (status == cudaSuccess && workspace.item_capacity < item_count) {
    cudaFree(workspace.offsets);
    cudaFree(workspace.lengths);
    cudaFree(workspace.scalars);
    workspace.offsets = nullptr;
    workspace.lengths = nullptr;
    workspace.scalars = nullptr;
    workspace.item_capacity = 0;
    status = cudaMalloc(&workspace.offsets, item_count * sizeof(size_t));
    if (status == cudaSuccess)
      status = cudaMalloc(&workspace.lengths, item_count * sizeof(size_t));
    if (status == cudaSuccess)
      status = cudaMalloc(&workspace.scalars, item_count * 32u);
    if (status == cudaSuccess) workspace.item_capacity = item_count;
  }
  const size_t required_dst = std::max<size_t>(1u, dst_len);
  if (status == cudaSuccess && workspace.dst_capacity < required_dst) {
    cudaFree(workspace.dst);
    workspace.dst = nullptr;
    workspace.dst_capacity = 0;
    status = cudaMalloc(&workspace.dst, required_dst);
    if (status == cudaSuccess) workspace.dst_capacity = required_dst;
  }
  if (!nonblocking && status == cudaSuccess && total_len > 0) {
    status = cudaMemcpy(workspace.messages, messages, total_len,
                        cudaMemcpyHostToDevice);
  }
  if (!nonblocking && status == cudaSuccess) {
    status = cudaMemcpy(workspace.offsets, offsets,
                        item_count * sizeof(size_t), cudaMemcpyHostToDevice);
  }
  if (!nonblocking && status == cudaSuccess) {
    status = cudaMemcpy(workspace.lengths, lengths,
                        item_count * sizeof(size_t), cudaMemcpyHostToDevice);
  }
  if (!nonblocking && status == cudaSuccess && dst_len > 0) {
    status = cudaMemcpy(workspace.dst, dst, dst_len, cudaMemcpyHostToDevice);
  }
  if (!nonblocking && status == cudaSuccess) {
    status = cubbs_bbs_hash_to_scalar_sha256_batch_host(
        workspace.messages, workspace.offsets, workspace.lengths, item_count,
        workspace.dst, dst_len, workspace.scalars);
  }
  if (!nonblocking && status == cudaSuccess) {
    status = cudaMemcpy(scalars, workspace.scalars, item_count * 32u,
                        cudaMemcpyDeviceToHost);
  }
  if (nonblocking && status == cudaSuccess && total_len > 0) {
    status = cudaMemcpyAsync(workspace.messages, messages, total_len,
                             cudaMemcpyHostToDevice, workspace.stream);
  }
  if (nonblocking && status == cudaSuccess) {
    status = cudaMemcpyAsync(workspace.offsets, offsets,
                             item_count * sizeof(size_t), cudaMemcpyHostToDevice,
                             workspace.stream);
  }
  if (nonblocking && status == cudaSuccess) {
    status = cudaMemcpyAsync(workspace.lengths, lengths,
                             item_count * sizeof(size_t), cudaMemcpyHostToDevice,
                             workspace.stream);
  }
  if (nonblocking && status == cudaSuccess && dst_len > 0) {
    status = cudaMemcpyAsync(workspace.dst, dst, dst_len, cudaMemcpyHostToDevice,
                             workspace.stream);
  }
  if (nonblocking && status == cudaSuccess) {
    status = cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
        workspace.messages, workspace.offsets, workspace.lengths, item_count,
        workspace.dst, dst_len, workspace.scalars, workspace.stream);
  }
  if (nonblocking && status == cudaSuccess) {
    status = cudaMemcpyAsync(scalars, workspace.scalars, item_count * 32u,
                             cudaMemcpyDeviceToHost, workspace.stream);
  }
  if (nonblocking && status == cudaSuccess) {
    status = cudaStreamSynchronize(workspace.stream);
  }
  workspace.last_scalar_count =
      status == cudaSuccess ? item_count : 0u;

  return status == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS : CUBBS_BBS_GPU_ERR_INTERNAL;
}

extern "C" int32_t cubbs_bbs_gpu_messages_to_scalar_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuMessagesToScalarItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  size_t total_len = 0;
  for (size_t i = 0; i < item_count; ++i) {
    if (items[i].message.len > 0 && items[i].message.data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (items[i].dst.len > 255 || (items[i].dst.len > 0 && items[i].dst.data == nullptr) ||
        items[i].scalar_out.data == nullptr || items[i].scalar_out.capacity < 32u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    total_len += items[i].message.len;
  }

  struct HostPackWorkspace {
    std::vector<uint8_t> packed;
    std::vector<size_t> offsets;
    std::vector<size_t> lengths;
    std::vector<uint8_t> scalars;
  };
  static thread_local HostPackWorkspace host_workspace;
  host_workspace.packed.resize(total_len);
  host_workspace.offsets.resize(item_count);
  host_workspace.lengths.resize(item_count);
  host_workspace.scalars.resize(item_count * 32u);
  auto &packed = host_workspace.packed;
  auto &offsets = host_workspace.offsets;
  auto &lengths = host_workspace.lengths;
  size_t cursor = 0;
  for (size_t i = 0; i < item_count; ++i) {
    offsets[i] = cursor;
    lengths[i] = items[i].message.len;
    if (items[i].message.len > 0) {
      std::memcpy(packed.data() + cursor, items[i].message.data, items[i].message.len);
    }
    cursor += items[i].message.len;
  }

  // BBS SHA-256 uses one DST per batch in cuBBS-Cred. Mixed DSTs can be supported by
  // grouping at the caller, and are rejected here to keep the production ABI
  // deterministic and efficient.
  const CuBbsBbsGpuByteSlice first_dst = items[0].dst;
  for (size_t i = 1; i < item_count; ++i) {
    if (items[i].dst.len != first_dst.len ||
        (first_dst.len > 0 &&
         std::memcmp(items[i].dst.data, first_dst.data, first_dst.len) != 0)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }

  bool contiguous_output = items[0].scalar_out.data != nullptr;
  uint8_t *direct_output = items[0].scalar_out.data;
  for (size_t i = 1; i < item_count && contiguous_output; ++i) {
    contiguous_output =
        items[i].scalar_out.data == direct_output + (i * 32u);
  }
  auto &staged_scalars = host_workspace.scalars;
  uint8_t *scalar_output =
      contiguous_output ? direct_output : staged_scalars.data();
  const int32_t status = cubbs_bbs_gpu_hash_to_scalar_sha256_batch_for_test_v2(
      abi_version, packed.data(), offsets.data(), lengths.data(), item_count, first_dst.data,
      first_dst.len, scalar_output);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    return status;
  }

  for (size_t i = 0; i < item_count; ++i) {
    if (!contiguous_output) {
      std::memcpy(items[i].scalar_out.data,
                  staged_scalars.data() + (i * 32u), 32u);
    }
    items[i].scalar_out.len = 32u;
    results[i].code = CUBBS_BBS_GPU_SUCCESS;
    results[i].value = 0;
    results[i].bytes_written = 32u;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

struct CuBbsDeviceSignEContext {
  uint8_t secret_key[32];
  uint8_t domain[32];
  size_t scalar_offset;
  size_t message_count;
  size_t output_offset;
};

__global__ static void cubbs_bbs_pack_sign_e_inputs_kernel(
    const CuBbsDeviceSignEContext *contexts, size_t item_count,
    const uint8_t *message_scalars, uint8_t *packed_inputs,
    size_t *offsets, size_t *lengths) {
  const size_t item = blockIdx.x;
  if (item >= item_count) return;
  const auto context = contexts[item];
  const size_t input_len = (context.message_count + 2u) * 32u;
  if (threadIdx.x == 0u) {
    offsets[item] = context.output_offset;
    lengths[item] = input_len;
  }
  for (size_t byte = threadIdx.x; byte < input_len; byte += blockDim.x) {
    uint8_t value;
    if (byte < 32u) {
      value = context.secret_key[byte];
    } else if (byte < (context.message_count + 1u) * 32u) {
      const size_t scalar_byte = byte - 32u;
      value = message_scalars[
          context.scalar_offset * 32u + scalar_byte];
    } else {
      value = context.domain[byte - (context.message_count + 1u) * 32u];
    }
    packed_inputs[context.output_offset + byte] = value;
  }
}

extern "C" int32_t cubbs_bbs_gpu_sign_e_from_resident_scalars_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuSignEFromResidentScalarsItem *items,
    size_t item_count, CuBbsBbsGpuByteSlice message_dst,
    CuBbsBbsGpuByteSlice e_dst, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || message_dst.len > 255u || e_dst.len > 255u ||
      (message_dst.len > 0u && message_dst.data == nullptr) ||
      (e_dst.len > 0u && e_dst.data == nullptr) ||
      (item_count > 0u && (items == nullptr || results == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;

  auto &workspace = cubbs_hash_to_scalar_workspace();
  const bool raw_pipeline = items[0].messages != nullptr;
  std::vector<CuBbsDeviceSignEContext> contexts(item_count);
  size_t total_scalar_count = 0u;
  size_t total_e_input_bytes = 0u;
  size_t total_raw_message_bytes = 0u;
  for (size_t item = 0u; item < item_count; ++item) {
    const size_t count = items[item].message_count;
    const size_t scalar_end = items[item].message_scalar_offset + count;
    const size_t input_len = (count + 2u) * 32u;
    if (count == 0u || scalar_end < items[item].message_scalar_offset ||
        items[item].secret_key.data == nullptr ||
        items[item].secret_key.len != 32u ||
        items[item].domain.data == nullptr || items[item].domain.len != 32u ||
        items[item].scalar_out.data == nullptr ||
        items[item].scalar_out.capacity < 32u ||
        (items[item].messages != nullptr) != raw_pipeline ||
        total_e_input_bytes > SIZE_MAX - input_len) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (raw_pipeline) {
      if (items[item].message_scalar_offset != total_scalar_count ||
          items[item].message_scalars_out.data == nullptr ||
          items[item].message_scalars_out.capacity < count * 32u) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      for (size_t message = 0u; message < count; ++message) {
        const auto &slice = items[item].messages[message];
        if ((slice.len > 0u && slice.data == nullptr) ||
            total_raw_message_bytes > SIZE_MAX - slice.len) {
          return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
        }
        total_raw_message_bytes += slice.len;
      }
      total_scalar_count += count;
    } else if (scalar_end > workspace.last_scalar_count) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    auto &context = contexts[item];
    std::memcpy(context.secret_key, items[item].secret_key.data, 32u);
    std::memcpy(context.domain, items[item].domain.data, 32u);
    context.scalar_offset = items[item].message_scalar_offset;
    context.message_count = count;
    context.output_offset = total_e_input_bytes;
    total_e_input_bytes += input_len;
  }
  if (!raw_pipeline) total_scalar_count = workspace.last_scalar_count;

  cudaError_t status = cudaSuccess;
  const size_t required_message_bytes =
      std::max<size_t>(1u, std::max(total_raw_message_bytes,
                                   total_e_input_bytes));
  if (workspace.message_capacity < required_message_bytes) {
    cudaFree(workspace.messages);
    workspace.messages = nullptr;
    workspace.message_capacity = 0u;
    status = cudaMalloc(&workspace.messages, required_message_bytes);
    if (status == cudaSuccess) {
      workspace.message_capacity = required_message_bytes;
    }
  }
  if (status == cudaSuccess && raw_pipeline &&
      workspace.item_capacity < total_scalar_count) {
    cudaFree(workspace.offsets);
    cudaFree(workspace.lengths);
    cudaFree(workspace.scalars);
    workspace.offsets = nullptr;
    workspace.lengths = nullptr;
    workspace.scalars = nullptr;
    workspace.item_capacity = 0u;
    status = cudaMalloc(
        &workspace.offsets, total_scalar_count * sizeof(size_t));
    if (status == cudaSuccess) {
      status = cudaMalloc(
          &workspace.lengths, total_scalar_count * sizeof(size_t));
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(&workspace.scalars, total_scalar_count * 32u);
    }
    if (status == cudaSuccess) {
      workspace.item_capacity = total_scalar_count;
    }
  }
  const size_t context_bytes = contexts.size() * sizeof(contexts[0]);
  if (status == cudaSuccess &&
      workspace.sign_context_capacity < context_bytes) {
    cudaFree(workspace.sign_contexts);
    workspace.sign_contexts = nullptr;
    workspace.sign_context_capacity = 0u;
    status = cudaMalloc(&workspace.sign_contexts, context_bytes);
    if (status == cudaSuccess) {
      workspace.sign_context_capacity = context_bytes;
    }
  }
  if (status == cudaSuccess &&
      workspace.sign_e_scalar_capacity < item_count) {
    cudaFree(workspace.sign_e_scalars);
    workspace.sign_e_scalars = nullptr;
    workspace.sign_e_scalar_capacity = 0u;
    status = cudaMalloc(&workspace.sign_e_scalars, item_count * 32u);
    if (status == cudaSuccess) {
      workspace.sign_e_scalar_capacity = item_count;
    }
  }
  const size_t required_dst =
      std::max<size_t>(1u, std::max(message_dst.len, e_dst.len));
  if (status == cudaSuccess && workspace.dst_capacity < required_dst) {
    cudaFree(workspace.dst);
    workspace.dst = nullptr;
    workspace.dst_capacity = 0u;
    status = cudaMalloc(&workspace.dst, required_dst);
    if (status == cudaSuccess) workspace.dst_capacity = required_dst;
  }

  std::vector<uint8_t> raw_messages;
  std::vector<size_t> raw_offsets;
  std::vector<size_t> raw_lengths;
  if (status == cudaSuccess && raw_pipeline) {
    raw_messages.resize(total_raw_message_bytes);
    raw_offsets.resize(total_scalar_count);
    raw_lengths.resize(total_scalar_count);
    size_t message_index = 0u;
    size_t cursor = 0u;
    for (size_t item = 0u; item < item_count; ++item) {
      for (size_t message = 0u; message < items[item].message_count;
           ++message) {
        const auto &slice = items[item].messages[message];
        raw_offsets[message_index] = cursor;
        raw_lengths[message_index] = slice.len;
        if (slice.len > 0u) {
          std::memcpy(raw_messages.data() + cursor, slice.data, slice.len);
        }
        cursor += slice.len;
        ++message_index;
      }
    }
    if (total_raw_message_bytes > 0u) {
      status = cudaMemcpyAsync(
          workspace.messages, raw_messages.data(), total_raw_message_bytes,
          cudaMemcpyHostToDevice, workspace.stream);
    }
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(
          workspace.offsets, raw_offsets.data(),
          total_scalar_count * sizeof(size_t), cudaMemcpyHostToDevice,
          workspace.stream);
    }
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(
          workspace.lengths, raw_lengths.data(),
          total_scalar_count * sizeof(size_t), cudaMemcpyHostToDevice,
          workspace.stream);
    }
    if (status == cudaSuccess && message_dst.len > 0u) {
      status = cudaMemcpyAsync(
          workspace.dst, message_dst.data, message_dst.len,
          cudaMemcpyHostToDevice, workspace.stream);
    }
    if (status == cudaSuccess) {
      status = cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
          workspace.messages, workspace.offsets, workspace.lengths,
          total_scalar_count, workspace.dst, message_dst.len,
          workspace.scalars, workspace.stream);
    }
  }

  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(
        workspace.sign_contexts, contexts.data(), context_bytes,
        cudaMemcpyHostToDevice, workspace.stream);
  }
  if (status == cudaSuccess && e_dst.len > 0u) {
    status = cudaMemcpyAsync(
        workspace.dst, e_dst.data, e_dst.len, cudaMemcpyHostToDevice,
        workspace.stream);
  }
  if (status == cudaSuccess) {
    cubbs_bbs_pack_sign_e_inputs_kernel
        <<<static_cast<uint32_t>(item_count), 128u, 0u, workspace.stream>>>(
            reinterpret_cast<const CuBbsDeviceSignEContext *>(
                workspace.sign_contexts),
            item_count, workspace.scalars, workspace.messages,
            workspace.offsets, workspace.lengths);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    status = cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
        workspace.messages, workspace.offsets, workspace.lengths, item_count,
        workspace.dst, e_dst.len, workspace.sign_e_scalars,
        workspace.stream);
  }

  std::vector<uint8_t> staged_message_scalars(
      raw_pipeline ? total_scalar_count * 32u : 0u);
  std::vector<uint8_t> staged_e_scalars(item_count * 32u);
  if (status == cudaSuccess && raw_pipeline) {
    status = cudaMemcpyAsync(
        staged_message_scalars.data(), workspace.scalars,
        staged_message_scalars.size(), cudaMemcpyDeviceToHost,
        workspace.stream);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(
        staged_e_scalars.data(), workspace.sign_e_scalars,
        staged_e_scalars.size(), cudaMemcpyDeviceToHost, workspace.stream);
  }
  if (status == cudaSuccess) status = cudaStreamSynchronize(workspace.stream);
  if (status != cudaSuccess) {
    workspace.last_scalar_count = 0u;
    return CUBBS_BBS_GPU_ERR_INTERNAL;
  }
  workspace.last_scalar_count = total_scalar_count;

  for (size_t item = 0u; item < item_count; ++item) {
    if (raw_pipeline) {
      const size_t begin = items[item].message_scalar_offset * 32u;
      const size_t bytes = items[item].message_count * 32u;
      std::memcpy(items[item].message_scalars_out.data,
                  staged_message_scalars.data() + begin, bytes);
      items[item].message_scalars_out.len = bytes;
    }
    std::memcpy(items[item].scalar_out.data,
                staged_e_scalars.data() + item * 32u, 32u);
    items[item].scalar_out.len = 32u;
    results[item].code = CUBBS_BBS_GPU_SUCCESS;
    results[item].value = 0u;
    results[item].bytes_written = 32u;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

static bool cuda_canary_ok() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return false;
  }

  uint32_t *device_out = nullptr;
  uint32_t host_out = 0;
  if (cudaMalloc(&device_out, sizeof(uint32_t)) != cudaSuccess) {
    return false;
  }

  cubbs_bbs_canary_kernel<<<1, 1>>>(device_out);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaMemcpy(&host_out, device_out, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
  }
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  cudaFree(device_out);

  return status == cudaSuccess && host_out == 0x0BB52023u;
}

static void mark_unimplemented(CuBbsBbsGpuResult *results, size_t item_count) {
  if (results == nullptr) {
    return;
  }
  for (size_t i = 0; i < item_count; ++i) {
    results[i].code = CUBBS_BBS_GPU_ERR_INTERNAL;
    results[i].value = 0;
    results[i].bytes_written = 0;
  }
}

static int32_t validate_common_points_groups(
    CuBbsBbsGpuG1MsmCommonPointsBatchItem *items, size_t group_count,
    size_t *total_outputs) {
  if (group_count > 0 && items == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  size_t outputs = 0;
  for (size_t group = 0; group < group_count; ++group) {
    const auto &item = items[group];
    if (item.term_count > (SIZE_MAX / 32u) - 1u ||
        item.output_count > SIZE_MAX - outputs ||
        (item.term_count != 0u && item.output_count > SIZE_MAX / item.term_count) ||
        (item.scalar_row_stride != 0u && item.output_count > SIZE_MAX / item.scalar_row_stride)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    const bool has_packed_scalars = items[group].scalar_rows_be != nullptr;
    if ((items[group].term_count > 0 && items[group].points == nullptr) ||
        (items[group].output_count > 0 &&
         ((!has_packed_scalars && items[group].scalar_rows == nullptr) ||
          items[group].point_outs == nullptr)) ||
        (has_packed_scalars &&
         items[group].scalar_row_stride < items[group].term_count * 32u)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t term = 0; term < items[group].term_count; ++term) {
      if (items[group].points[term].len != 48u ||
          items[group].points[term].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
    }
    for (size_t row = 0; row < items[group].output_count; ++row) {
      if (items[group].point_outs[row].data == nullptr ||
          items[group].point_outs[row].capacity < 48u) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      if (!has_packed_scalars) {
        for (size_t term = 0; term < items[group].term_count; ++term) {
          const size_t scalar_index = (row * items[group].term_count) + term;
          if (items[group].scalar_rows[scalar_index].len != 32u ||
              items[group].scalar_rows[scalar_index].data == nullptr) {
            return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
          }
        }
      }
    }
    outputs += items[group].output_count;
  }
  if (total_outputs != nullptr) {
    *total_outputs = outputs;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t
cubbs_bbs_gpu_backend_info_v2(uint32_t abi_version,
                               CuBbsBbsGpuBackendInfo *info_out) {
  if (!abi_ok(abi_version) || info_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  const bool canary_ok = cuda_canary_ok();
  info_out->backend_kind = canary_ok ? CUBBS_BBS_GPU_BACKEND_KIND_CUDA
                                     : CUBBS_BBS_GPU_BACKEND_KIND_UNKNOWN;
  // This library proves CUDA runtime reachability. It intentionally does not
  // claim BBS hardware acceleration until sign/verify/proof kernels are
  // implemented and validated against zkryptium vectors.
  info_out->hardware_acceleration = 0;
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_sign_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuSignItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  mark_unimplemented(results, item_count);
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_prepare_b_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuPrepareBItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t i = 0; i < item_count; ++i) {
    if (items[i].public_key.len != CUBBS_BBS_PUBLIC_KEY_LEN ||
        items[i].public_key.data == nullptr ||
        items[i].header.len != CUBBS_BBS_2023_HEADER_LEN ||
        items[i].header.data == nullptr ||
        (items[i].message_count > 0 && items[i].message_scalars == nullptr) ||
        items[i].b_out.data == nullptr || items[i].b_out.capacity < 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t j = 0; j < items[i].message_count; ++j) {
      if (items[i].message_scalars[j].len != 32u ||
          items[i].message_scalars[j].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
    }
  }

  std::vector<CuBbsPrepareBInput> msm_inputs(item_count);
  for (size_t i = 0; i < item_count; ++i) {
    msm_inputs[i] = CuBbsPrepareBInput{
        items[i].public_key.data,
        items[i].header.data,
        items[i].message_scalars,
        items[i].message_count,
        items[i].b_out.data,
    };
  }

  return cubbs_bbs_cuda_prepare_b_msm_batch(msm_inputs.data(), item_count, results);
}

extern "C" int32_t cubbs_bbs_gpu_g1_msm_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  for (size_t i = 0; i < item_count; ++i) {
    if ((items[i].term_count > 0 &&
         (items[i].points == nullptr || items[i].scalars == nullptr)) ||
        items[i].point_out.data == nullptr || items[i].point_out.capacity < 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t j = 0; j < items[i].term_count; ++j) {
      if (items[i].points[j].len != 48u || items[i].points[j].data == nullptr ||
          items[i].scalars[j].len != 32u || items[i].scalars[j].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
    }
  }

  std::vector<CuBbsG1MsmInput> msm_inputs(item_count);
  for (size_t i = 0; i < item_count; ++i) {
    msm_inputs[i] = CuBbsG1MsmInput{
        items[i].points,
        items[i].scalars,
        items[i].term_count,
        items[i].point_out.data,
    };
  }

  const int32_t status =
      cubbs_bbs_cuda_g1_msm_batch(msm_inputs.data(), item_count, results);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    return status;
  }
  for (size_t i = 0; i < item_count; ++i) {
    if (results[i].code == CUBBS_BBS_GPU_SUCCESS) {
      items[i].point_out.len = results[i].bytes_written;
    }
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmCommonPointsBatchItem *items,
    size_t group_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (group_count > 0 && items == nullptr) ||
      (group_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  size_t total_outputs = 0;
  const int32_t validation_status =
      validate_common_points_groups(items, group_count, &total_outputs);
  if (validation_status != CUBBS_BBS_GPU_SUCCESS) {
    return validation_status;
  }

  size_t result_cursor = 0;
  for (size_t group = 0; group < group_count; ++group) {
    const int32_t status = cubbs_bbs_cuda_g1_msm_common_points_matrix(
        items[group].points, items[group].term_count, items[group].scalar_rows,
        items[group].output_count, items[group].scalar_rows_be,
        items[group].scalar_row_stride, items[group].point_outs,
        results + result_cursor);
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
    for (size_t row = 0; row < items[group].output_count; ++row) {
      if (results[result_cursor + row].code == CUBBS_BBS_GPU_SUCCESS) {
        items[group].point_outs[row].len = results[result_cursor + row].bytes_written;
      }
    }
    result_cursor += items[group].output_count;
  }

  (void)total_outputs;
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmCommonPointsBatchItem *items,
    size_t group_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (group_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  size_t total_outputs = 0;
  const int32_t validation_status =
      validate_common_points_groups(items, group_count, &total_outputs);
  if (validation_status != CUBBS_BBS_GPU_SUCCESS) {
    return validation_status;
  }

  std::vector<size_t> result_offsets(group_count + 1u, 0u);
  for (size_t group = 0; group < group_count; ++group) {
    result_offsets[group + 1u] = result_offsets[group] + items[group].output_count;
  }

  auto run_group = [&](size_t group) -> int32_t {
    const size_t result_cursor = result_offsets[group];
    const int32_t status = cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
        items[group].points, items[group].term_count, items[group].scalar_rows,
        items[group].output_count, items[group].scalar_rows_be,
        items[group].scalar_row_stride, items[group].point_outs,
        results + result_cursor);
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
    for (size_t row = 0; row < items[group].output_count; ++row) {
      if (results[result_cursor + row].code == CUBBS_BBS_GPU_SUCCESS) {
        items[group].point_outs[row].len = results[result_cursor + row].bytes_written;
      }
    }
    return CUBBS_BBS_GPU_SUCCESS;
  };

  const size_t worker_count = fused_group_worker_count(group_count);
  if (worker_count <= 1u) {
    for (size_t group = 0; group < group_count; ++group) {
      const int32_t status = run_group(group);
      if (status != CUBBS_BBS_GPU_SUCCESS) {
        return status;
      }
    }
  } else {
    std::atomic<size_t> next_group{0u};
    std::atomic<int32_t> first_status{CUBBS_BBS_GPU_SUCCESS};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        while (first_status.load() == CUBBS_BBS_GPU_SUCCESS) {
          const size_t group = next_group.fetch_add(1u);
          if (group >= group_count) {
            break;
          }
          const int32_t status = run_group(group);
          if (status != CUBBS_BBS_GPU_SUCCESS) {
            int32_t expected = CUBBS_BBS_GPU_SUCCESS;
            first_status.compare_exchange_strong(expected, status);
            break;
          }
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    const int32_t status = first_status.load();
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
  }

  (void)total_outputs;
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t
cubbs_bbs_gpu_g1_msm_common_points_plus_variable_batch_v2(
    uint32_t abi_version,
    CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem *items,
    size_t group_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (group_count > 0u && items == nullptr) ||
      (group_count > 0u && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  size_t result_cursor = 0u;
  for (size_t group = 0u; group < group_count; ++group) {
    auto &item = items[group];
    if ((item.term_count > 0u &&
         (item.points == nullptr || item.scalar_rows == nullptr)) ||
        (item.output_count > 0u &&
         (item.variable_points == nullptr || item.variable_scalars == nullptr ||
          item.point_outs == nullptr))) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (item.term_count > SIZE_MAX / 32u ||
        item.output_count > SIZE_MAX - result_cursor ||
        (item.term_count != 0u && item.output_count > SIZE_MAX / item.term_count)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t row = 0u; row < item.output_count; ++row) {
      if (item.point_outs[row].data == nullptr || item.point_outs[row].capacity < 48u)
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    const int32_t status =
        cubbs_bbs_cuda_g1_msm_fused_common_points_plus_variable_matrix(
            item.points, item.term_count, item.scalar_rows, item.output_count,
            item.variable_points, item.variable_scalars, item.point_outs,
            results + result_cursor);
    if (status != CUBBS_BBS_GPU_SUCCESS) return status;
    for (size_t row = 0u; row < item.output_count; ++row) {
      if (results[result_cursor + row].code == CUBBS_BBS_GPU_SUCCESS) {
        item.point_outs[row].len = results[result_cursor + row].bytes_written;
      }
    }
    result_cursor += item.output_count;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_proof_verify_msm_pair_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuProofVerifyMsmPairItem *items,
    size_t group_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (group_count > 0u && items == nullptr) ||
      (group_count > 0u && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  size_t result_cursor = 0u;
  for (size_t group = 0u; group < group_count; ++group) {
    auto &item = items[group];
    const int32_t status = cubbs_bbs_cuda_proof_verify_msm_pair_matrix(
        item.common_points, item.common_term_count, item.common_scalar_rows,
        item.proof_points, item.proof_scalars, item.output_count, item.t1_outs,
        item.t2_outs, results + result_cursor);
    if (status != CUBBS_BBS_GPU_SUCCESS) return status;
    for (size_t row = 0u; row < item.output_count; ++row) {
      if (results[result_cursor + row].code == CUBBS_BBS_GPU_SUCCESS) {
        item.t1_outs[row].len = 48u;
        item.t2_outs[row].len = 48u;
      }
    }
    result_cursor += item.output_count;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_g1_dual_msm_same_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1DualMsmSamePointsItem *items,
    size_t item_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0u && items == nullptr) ||
      (item_count > 0u && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t item_index = 0u; item_index < item_count; ++item_index) {
    auto &item = items[item_index];
    if (item.term_count == 0u || item.points == nullptr ||
        item.lhs_scalars == nullptr || item.rhs_scalars == nullptr ||
        item.lhs_out.data == nullptr || item.lhs_out.capacity < 48u ||
        item.rhs_out.data == nullptr || item.rhs_out.capacity < 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    const int32_t status = cubbs_bbs_cuda_g1_dual_msm_same_points(
        item.points, item.lhs_scalars, item.rhs_scalars, item.term_count,
        item.lhs_out.data, item.rhs_out.data);
    if (status != CUBBS_BBS_GPU_SUCCESS) return status;
    item.lhs_out.len = 48u;
    item.rhs_out.len = 48u;
    results[item_index].code = CUBBS_BBS_GPU_SUCCESS;
    results[item_index].value = 6u;
    results[item_index].bytes_written = 96u;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_g1_dual_msm_same_points_lanes_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1DualMsmSamePointsLanesItem *items,
    size_t item_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0u && items == nullptr) ||
      (item_count > 0u && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t item_index = 0u; item_index < item_count; ++item_index) {
    auto &item = items[item_index];
    if (item.lane_count > SIZE_MAX / 96u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    const size_t output_bytes = item.lane_count * 48u;
    if (item.term_count == 0u || item.lane_count == 0u ||
        item.lane_count > item.term_count || item.points == nullptr ||
        item.lhs_scalars == nullptr || item.rhs_scalars == nullptr ||
        item.lhs_outs.data == nullptr ||
        item.lhs_outs.capacity < output_bytes ||
        item.rhs_outs.data == nullptr ||
        item.rhs_outs.capacity < output_bytes) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    const int32_t status = cubbs_bbs_cuda_g1_dual_msm_same_points_lanes(
        item.points, item.lhs_scalars, item.rhs_scalars, item.term_count,
        item.lane_count, item.lhs_outs.data, item.rhs_outs.data);
    if (status != CUBBS_BBS_GPU_SUCCESS) return status;
    item.lhs_outs.len = output_bytes;
    item.rhs_outs.len = output_bytes;
    results[item_index].code = CUBBS_BBS_GPU_SUCCESS;
    results[item_index].value = 6u;
    results[item_index].bytes_written = output_bytes * 2u;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_verify_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuVerifyItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  mark_unimplemented(results, item_count);
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_proof_gen_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuProofGenItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  mark_unimplemented(results, item_count);
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_proof_verify_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuProofVerifyItem *items,
    size_t item_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  mark_unimplemented(results, item_count);
  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_proof_gen_t2_prepare_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuProofGenT2PrepareItem *items,
    size_t item_count, CuBbsBbsGpuResult *results) {
  if (!abi_ok(abi_version) || (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  // Validate fixed-length inputs up front; the native body re-checks per item.
  for (size_t i = 0; i < item_count; ++i) {
    bool ok = items[i].compressed_b.len == 48u && items[i].signature_a.len == 48u &&
              items[i].d_out.capacity >= 48u && items[i].abar_out.capacity >= 48u &&
              items[i].bbar_out.capacity >= 48u && items[i].t1_out.capacity >= 48u;
    for (size_t s = 0; s < 7u && ok; ++s) {
      ok = items[i].scalars_be[s].len == 32u && items[i].scalars_be[s].data != nullptr;
    }
    if (!ok) {
      results[i].code = CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      results[i].value = 0u;
      results[i].bytes_written = 0u;
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }
  return cubbs_bbs_cuda_proof_gen_t2_prepare(items, item_count, results);
}
