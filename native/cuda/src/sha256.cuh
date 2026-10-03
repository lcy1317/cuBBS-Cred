#ifndef CUBBS_BBS_CUDA_SHA256_CUH
#define CUBBS_BBS_CUDA_SHA256_CUH

#include <cuda_runtime.h>

#include <stdint.h>

static __device__ __forceinline__ uint32_t rotr32(uint32_t value, uint32_t bits) {
  return (value >> bits) | (value << (32u - bits));
}

static __device__ __forceinline__ uint32_t load_be32(const uint8_t *bytes) {
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         static_cast<uint32_t>(bytes[3]);
}

static __device__ __forceinline__ void store_be32(uint8_t *bytes, uint32_t value) {
  bytes[0] = static_cast<uint8_t>(value >> 24);
  bytes[1] = static_cast<uint8_t>(value >> 16);
  bytes[2] = static_cast<uint8_t>(value >> 8);
  bytes[3] = static_cast<uint8_t>(value);
}

static __device__ void sha256_transform(const uint8_t block[64], uint32_t state[8]) {
  static constexpr uint32_t K[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
      0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
      0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
      0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

  // A 16-word rolling schedule avoids the 256-byte per-thread w[64] array,
  // which otherwise spills to local memory and limits hash-kernel occupancy.
  uint32_t w[16];
  for (int i = 0; i < 16; ++i) {
    w[i] = load_be32(block + (i * 4));
  }

  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];
  uint32_t f = state[5];
  uint32_t g = state[6];
  uint32_t h = state[7];

  for (int i = 0; i < 64; ++i) {
    uint32_t schedule_word;
    if (i < 16) {
      schedule_word = w[i];
    } else {
      const uint32_t w15 = w[(i + 1) & 15];
      const uint32_t w2 = w[(i + 14) & 15];
      const uint32_t s0 = rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >> 3);
      const uint32_t s1 = rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >> 10);
      schedule_word = w[i & 15] + s0 + w[(i + 9) & 15] + s1;
      w[i & 15] = schedule_word;
    }
    const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + s1 + ch + K[i] + schedule_word;
    const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

static __device__ void sha256_digest_device(const uint8_t *message, size_t len, uint8_t out[32]) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  uint8_t block[64];
  size_t offset = 0;
  while (len - offset >= 64) {
    for (int i = 0; i < 64; ++i) {
      block[i] = message[offset + i];
    }
    sha256_transform(block, state);
    offset += 64;
  }

  const size_t remaining = len - offset;
  for (int i = 0; i < 64; ++i) {
    block[i] = 0;
  }
  for (size_t i = 0; i < remaining; ++i) {
    block[i] = message[offset + i];
  }
  block[remaining] = 0x80u;

  if (remaining >= 56) {
    sha256_transform(block, state);
    for (int i = 0; i < 64; ++i) {
      block[i] = 0;
    }
  }

  const uint64_t bit_len = static_cast<uint64_t>(len) * 8u;
  for (int i = 0; i < 8; ++i) {
    block[63 - i] = static_cast<uint8_t>(bit_len >> (i * 8));
  }
  sha256_transform(block, state);

  for (int i = 0; i < 8; ++i) {
    store_be32(out + (i * 4), state[i]);
  }
}

static __device__ void sha256_digest_2_device(const uint8_t *a, size_t a_len, const uint8_t *b,
                                              size_t b_len, uint8_t out[32]) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  uint8_t block[64];
  size_t block_len = 0;
  size_t total_len = 0;

  for (size_t source = 0; source < 2; ++source) {
    const uint8_t *data = source == 0 ? a : b;
    const size_t len = source == 0 ? a_len : b_len;
    for (size_t i = 0; i < len; ++i) {
      block[block_len++] = data[i];
      total_len += 1;
      if (block_len == 64) {
        sha256_transform(block, state);
        block_len = 0;
      }
    }
  }

  block[block_len++] = 0x80u;
  if (block_len > 56) {
    while (block_len < 64) {
      block[block_len++] = 0;
    }
    sha256_transform(block, state);
    block_len = 0;
  }
  while (block_len < 56) {
    block[block_len++] = 0;
  }
  const uint64_t bit_len = static_cast<uint64_t>(total_len) * 8u;
  for (int i = 7; i >= 0; --i) {
    block[block_len++] = static_cast<uint8_t>(bit_len >> (i * 8));
  }
  sha256_transform(block, state);

  for (int i = 0; i < 8; ++i) {
    store_be32(out + (i * 4), state[i]);
  }
}

static __device__ void sha256_digest_6_device(
    const uint8_t *a, size_t a_len, const uint8_t *b, size_t b_len,
    const uint8_t *c, size_t c_len, const uint8_t *d, size_t d_len,
    const uint8_t *e, size_t e_len, const uint8_t *f, size_t f_len,
    uint8_t out[32]) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  uint8_t block[64];
  size_t block_len = 0;
  size_t total_len = 0;
  const uint8_t *chunks[6] = {a, b, c, d, e, f};
  const size_t lens[6] = {a_len, b_len, c_len, d_len, e_len, f_len};

  for (int source = 0; source < 6; ++source) {
    for (size_t i = 0; i < lens[source]; ++i) {
      block[block_len++] = chunks[source][i];
      total_len += 1;
      if (block_len == 64) {
        sha256_transform(block, state);
        block_len = 0;
      }
    }
  }

  block[block_len++] = 0x80u;
  if (block_len > 56) {
    while (block_len < 64) {
      block[block_len++] = 0;
    }
    sha256_transform(block, state);
    block_len = 0;
  }
  while (block_len < 56) {
    block[block_len++] = 0;
  }
  const uint64_t bit_len = static_cast<uint64_t>(total_len) * 8u;
  for (int i = 7; i >= 0; --i) {
    block[block_len++] = static_cast<uint8_t>(bit_len >> (i * 8));
  }
  sha256_transform(block, state);

  for (int i = 0; i < 8; ++i) {
    store_be32(out + (i * 4), state[i]);
  }
}

static __device__ void sha256_digest_5_device(
    const uint8_t *a, size_t a_len, const uint8_t *b, size_t b_len,
    const uint8_t *c, size_t c_len, const uint8_t *d, size_t d_len,
    const uint8_t *e, size_t e_len, uint8_t out[32]) {
  const uint8_t empty = 0;
  sha256_digest_6_device(a, a_len, b, b_len, c, c_len, d, d_len,
                         e, e_len, &empty, 0, out);
}

static __device__ void sha256_digest_3_device(const uint8_t *a, size_t a_len, const uint8_t *b,
                                              size_t b_len, const uint8_t *c, size_t c_len,
                                              uint8_t out[32]) {
  const uint8_t empty = 0;
  sha256_digest_5_device(a, a_len, b, b_len, c, c_len, &empty, 0, &empty, 0, out);
}

static __device__ void expand_message_xmd_sha256_48_device(const uint8_t *message, size_t len,
                                                           const uint8_t *dst, size_t dst_len,
                                                           uint8_t out[48]) {
  uint8_t z_pad[64] = {0};
  uint8_t len_bytes[2] = {0, 48};
  uint8_t zero = 0;
  uint8_t one = 1;
  uint8_t two = 2;
  uint8_t dst_len_byte = static_cast<uint8_t>(dst_len);
  uint8_t empty = 0;

  uint8_t b0[32];
  uint8_t b1[32];
  uint8_t b2[32];
  uint8_t b0_xor_b1[32];
  sha256_digest_6_device(z_pad, 64, message, len, len_bytes, 2, &zero, 1,
                         dst, dst_len, &dst_len_byte, 1, b0);
  sha256_digest_5_device(b0, 32, &one, 1, dst, dst_len, &dst_len_byte, 1,
                         &empty, 0, b1);
  for (int i = 0; i < 32; ++i) {
    b0_xor_b1[i] = b0[i] ^ b1[i];
  }
  sha256_digest_5_device(b0_xor_b1, 32, &two, 1, dst, dst_len,
                         &dst_len_byte, 1, &empty, 0, b2);
  for (int i = 0; i < 32; ++i) {
    out[i] = b1[i];
  }
  for (int i = 0; i < 16; ++i) {
    out[32 + i] = b2[i];
  }
}

static __device__ int scalar_ge_order(const uint8_t value[32]) {
  static constexpr uint8_t ORDER[32] = {
      0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48, 0x33, 0x39, 0xd8,
      0x08, 0x09, 0xa1, 0xd8, 0x05, 0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe,
      0x5b, 0xfe, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01};
  for (int i = 0; i < 32; ++i) {
    if (value[i] > ORDER[i]) {
      return 1;
    }
    if (value[i] < ORDER[i]) {
      return 0;
    }
  }
  return 1;
}

static __device__ void scalar_sub_order(uint8_t value[32]) {
  static constexpr uint8_t ORDER[32] = {
      0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48, 0x33, 0x39, 0xd8,
      0x08, 0x09, 0xa1, 0xd8, 0x05, 0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe,
      0x5b, 0xfe, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01};
  int borrow = 0;
  for (int i = 31; i >= 0; --i) {
    int diff = static_cast<int>(value[i]) - static_cast<int>(ORDER[i]) - borrow;
    if (diff < 0) {
      diff += 256;
      borrow = 1;
    } else {
      borrow = 0;
    }
    value[i] = static_cast<uint8_t>(diff);
  }
}

static __device__ void scalar_shift_add_bit(uint8_t value[32], uint8_t bit) {
  uint16_t carry = bit;
  for (int i = 31; i >= 0; --i) {
    const uint16_t shifted = static_cast<uint16_t>(value[i]) * 2u + carry;
    value[i] = static_cast<uint8_t>(shifted & 0xffu);
    carry = shifted >> 8;
  }
}

static __device__ void bls12_381_scalar_from_okm_48_device(const uint8_t uniform[48],
                                                           uint8_t scalar[32]) {
  // Fr is 255 bits, so a reduced value and its one-bit shift fit in four
  // 64-bit limbs. The previous byte-wise reducer touched 32 bytes for each of
  // 384 input bits; limb arithmetic cuts that serial work by roughly 8x.
  static constexpr uint64_t ORDER_LE[4] = {
      0xffffffff00000001ull, 0x53bda402fffe5bfeull,
      0x3339d80809a1d805ull, 0x73eda753299d7d48ull};
  uint64_t value[4] = {0ull, 0ull, 0ull, 0ull};
  for (int byte_index = 0; byte_index < 48; ++byte_index) {
    const uint8_t byte = uniform[byte_index];
    for (int bit_index = 7; bit_index >= 0; --bit_index) {
      uint64_t carry = static_cast<uint64_t>((byte >> bit_index) & 1u);
#pragma unroll
      for (int limb = 0; limb < 4; ++limb) {
        const uint64_t next = value[limb] >> 63u;
        value[limb] = (value[limb] << 1u) | carry;
        carry = next;
      }
      bool ge = true;
#pragma unroll
      for (int limb = 3; limb >= 0; --limb) {
        if (value[limb] != ORDER_LE[limb]) {
          ge = value[limb] > ORDER_LE[limb];
          break;
        }
      }
      if (ge) {
        uint64_t borrow = 0ull;
#pragma unroll
        for (int limb = 0; limb < 4; ++limb) {
          const uint64_t current = value[limb];
          const uint64_t modulus = ORDER_LE[limb];
          const uint64_t next_borrow =
              (current < modulus) || (borrow != 0ull && current == modulus);
          value[limb] = current - modulus - borrow;
          borrow = next_borrow;
        }
      }
    }
  }
#pragma unroll
  for (int byte_index = 0; byte_index < 32; ++byte_index) {
    const int little_byte = 31 - byte_index;
    scalar[byte_index] = static_cast<uint8_t>(
        value[little_byte / 8] >> ((little_byte % 8) * 8));
  }
}

__global__ void cubbs_bbs_expand_xmd_sha256_48_batch_kernel(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *uniform_bytes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= item_count) {
    return;
  }
  expand_message_xmd_sha256_48_device(messages + offsets[index], lengths[index], dst, dst_len,
                                      uniform_bytes + (index * 48));
}

__global__ void cubbs_bbs_hash_to_scalar_sha256_batch_kernel(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *scalars) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= item_count) {
    return;
  }
  uint8_t uniform[48];
  expand_message_xmd_sha256_48_device(messages + offsets[index], lengths[index], dst, dst_len,
                                      uniform);
  bls12_381_scalar_from_okm_48_device(uniform, scalars + (index * 32));
}

__global__ void cubbs_bbs_sha256_batch_kernel(const uint8_t *messages,
                                               const size_t *offsets,
                                               const size_t *lengths,
                                               size_t item_count,
                                               uint8_t *digests) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= item_count) {
    return;
  }
  sha256_digest_device(messages + offsets[index], lengths[index], digests + (index * 32));
}

static inline cudaError_t cubbs_bbs_sha256_batch_host(const uint8_t *messages,
                                                       const size_t *offsets,
                                                       const size_t *lengths,
                                                       size_t item_count,
                                                       uint8_t *digests) {
  if (item_count == 0) {
    return cudaSuccess;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((item_count + threads - 1) / threads);
  cubbs_bbs_sha256_batch_kernel<<<blocks, threads>>>(messages, offsets, lengths, item_count,
                                                      digests);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceSynchronize();
}

static inline cudaError_t cubbs_bbs_expand_xmd_sha256_48_batch_host(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *uniform_bytes) {
  if (item_count == 0) {
    return cudaSuccess;
  }
  if (dst_len > 255) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((item_count + threads - 1) / threads);
  cubbs_bbs_expand_xmd_sha256_48_batch_kernel<<<blocks, threads>>>(
      messages, offsets, lengths, item_count, dst, dst_len, uniform_bytes);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceSynchronize();
}

static inline cudaError_t cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *scalars, cudaStream_t stream) {
  if (item_count == 0) {
    return cudaSuccess;
  }
  if (dst_len > 255) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((item_count + threads - 1) / threads);
  cubbs_bbs_hash_to_scalar_sha256_batch_kernel<<<blocks, threads, 0, stream>>>(
      messages, offsets, lengths, item_count, dst, dst_len, scalars);
  return cudaGetLastError();
}

static inline cudaError_t cubbs_bbs_hash_to_scalar_sha256_batch_host(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *scalars) {
  cudaError_t status = cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
      messages, offsets, lengths, item_count, dst, dst_len, scalars, nullptr);
  return status == cudaSuccess ? cudaDeviceSynchronize() : status;
}

#endif
