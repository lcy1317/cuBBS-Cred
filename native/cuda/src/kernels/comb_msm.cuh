#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

template <uint32_t kThreadsPerRow>
__global__ static void cubbs_bbs_g1_msm_fixed_comb4_kernel(
    const cubbs_sppark_bucket_t *comb_table, const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);

  const size_t row = blockIdx.x;
  if (row >= output_count) {
    return;
  }

  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + (row * term_count * 32u);
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    cubbs_sppark_bucket_t term_acc;
    term_acc.inf();
    const uint8_t *scalar = row_scalars + (term * 32u);
    const cubbs_sppark_bucket_t *term_table =
        comb_table + (term * 64u * 16u);
#pragma unroll
    for (uint32_t window = 0u; window < 64u; ++window) {
      const uint8_t digit = cubbs_bbs_scalar_le_window4(scalar, window);
      if (digit != 0u) {
        term_acc.add(term_table[(window * 16u) + digit]);
      }
    }
    acc.add(term_acc);
  }
  partials[tid] = acc;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partials[tid].add(partials[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) {
    outputs[row] = partials[0];
  }
}

template <uint32_t kThreadsPerRow>
__global__ static void cubbs_bbs_g1_msm_fixed_signed_comb5_kernel(
    const cubbs_sppark_bucket_t *comb_table, const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) {
    return;
  }

  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + (row * term_count * 32u);
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    cubbs_sppark_bucket_t term_acc;
    term_acc.inf();
    const uint8_t *scalar = row_scalars + (term * 32u);
    const cubbs_sppark_bucket_t *term_table =
        comb_table + (term * 52u * 17u);
    uint32_t carry = 0u;
#pragma unroll
    for (uint32_t window = 0u; window < 52u; ++window) {
      const uint32_t value =
          static_cast<uint32_t>(cubbs_bbs_scalar_le_window5(scalar, window)) +
          carry;
      const bool negative = value > 16u;
      const uint32_t digit = negative ? 32u - value : value;
      carry = negative ? 1u : 0u;
      if (digit != 0u) {
        auto point = term_table[(window * 17u) + digit];
        point.cneg(negative);
        term_acc.add(point);
      }
    }
    acc.add(term_acc);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partials[tid].add(partials[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) {
    outputs[row] = partials[0];
  }
}

template <uint32_t kThreadsPerRow>
__global__ static void cubbs_bbs_g1_msm_fixed_signed_comb6_kernel(
    const cubbs_sppark_bucket_t *comb_table, const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) return;

  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + (row * term_count * 32u);
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    cubbs_sppark_bucket_t term_acc;
    term_acc.inf();
    const uint8_t *scalar = row_scalars + (term * 32u);
    const cubbs_sppark_bucket_t *term_table =
        comb_table + (term * 43u * 33u);
    uint32_t carry = 0u;
#pragma unroll
    for (uint32_t window = 0u; window < 43u; ++window) {
      const uint32_t value =
          static_cast<uint32_t>(cubbs_bbs_scalar_le_window6(scalar, window)) +
          carry;
      const bool negative = value > 32u;
      const uint32_t digit = negative ? 64u - value : value;
      carry = negative ? 1u : 0u;
      if (digit != 0u) {
        auto point = term_table[(window * 33u) + digit];
        point.cneg(negative);
        term_acc.add(point);
      }
    }
    acc.add(term_acc);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

template <uint32_t kThreadsPerRow>
__global__ static void cubbs_bbs_g1_msm_fixed_signed_comb8_kernel(
    const cubbs_sppark_bucket_t *comb_table, const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + (row * term_count * 32u);
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    const uint8_t *scalar = row_scalars + (term * 32u);
    const cubbs_sppark_bucket_t *term_table =
        comb_table + (term * 32u * 129u);
    uint32_t carry = 0u;
#pragma unroll
    for (uint32_t window = 0u; window < 32u; ++window) {
      const uint32_t value =
          static_cast<uint32_t>(cubbs_bbs_scalar_le_window8(scalar, window)) +
          carry;
      const bool negative = value > 128u;
      const uint32_t digit = negative ? 256u - value : value;
      carry = negative ? 1u : 0u;
      if (digit != 0u) {
        auto point = term_table[(window * 129u) + digit];
        point.cneg(negative);
        acc.add(point);
      }
    }
  }
  partials[tid] = acc;
  __syncthreads();
  const uint32_t active_threads =
      static_cast<uint32_t>(term_count < blockDim.x ? term_count : blockDim.x);
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride && tid + stride < active_threads) {
      partials[tid].add(partials[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

template <uint32_t kThreadsPerRow>
__global__ static void cubbs_bbs_g1_msm_fixed_signed_comb10_kernel(
    const cubbs_sppark_bucket_t *comb_table, const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    const uint8_t *scalar = row_scalars + term * 32u;
    const auto *term_table = comb_table + term * 26u * 513u;
    uint32_t carry = 0u;
#pragma unroll
    for (uint32_t window = 0u; window < 26u; ++window) {
      const uint32_t value =
          static_cast<uint32_t>(cubbs_bbs_scalar_le_window10(scalar, window)) +
          carry;
      const bool negative = value > 512u;
      const uint32_t digit = negative ? 1024u - value : value;
      carry = negative ? 1u : 0u;
      if (digit != 0u) {
        auto point = term_table[window * 513u + digit];
        point.cneg(negative);
        acc.add(point);
      }
    }
  }
  partials[tid] = acc;
  __syncthreads();
  const uint32_t active_threads =
      static_cast<uint32_t>(term_count < blockDim.x ? term_count : blockDim.x);
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride && tid + stride < active_threads) {
      partials[tid].add(partials[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

template <uint32_t kThreadsPerRow>
__global__ static void cubbs_bbs_g1_msm_fixed_signed_comb9_kernel(
    const cubbs_sppark_bucket_t *comb_table, const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    const uint8_t *scalar = row_scalars + term * 32u;
    const auto *term_table = comb_table + term * 29u * 257u;
    uint32_t carry = 0u;
#pragma unroll
    for (uint32_t window = 0u; window < 29u; ++window) {
      const uint32_t value =
          static_cast<uint32_t>(cubbs_bbs_scalar_le_window9(scalar, window)) +
          carry;
      const bool negative = value > 256u;
      const uint32_t digit = negative ? 512u - value : value;
      carry = negative ? 1u : 0u;
      if (digit != 0u) {
        auto point = term_table[window * 257u + digit];
        point.cneg(negative);
        acc.add(point);
      }
    }
  }
  partials[tid] = acc;
  __syncthreads();
  const uint32_t active_threads =
      static_cast<uint32_t>(term_count < blockDim.x ? term_count : blockDim.x);
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride && tid + stride < active_threads) {
      partials[tid].add(partials[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

template <uint32_t kThreadsPerRow, uint32_t kBits>
__global__ static void cubbs_bbs_g1_msm_fixed_signed_affine_comb_kernel(
    const cubbs_sppark_plain_affine_t *comb_table,
    const uint8_t *scalars_le, size_t term_count, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  constexpr uint32_t windows = kBits == 8u ? 32u : (kBits == 9u ? 29u : 26u);
  constexpr uint32_t half = 1u << (kBits - 1u);
  constexpr uint32_t modulus = 1u << kBits;
  constexpr uint32_t entries = half + 1u;
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    const uint8_t *scalar = row_scalars + term * 32u;
    const auto *term_table = comb_table + term * windows * entries;
    uint32_t carry = 0u;
#pragma unroll
    for (uint32_t window = 0u; window < windows; ++window) {
      uint32_t raw;
      if constexpr (kBits == 8u) {
        raw = cubbs_bbs_scalar_le_window8(scalar, window);
      } else if constexpr (kBits == 9u) {
        raw = cubbs_bbs_scalar_le_window9(scalar, window);
      } else {
        raw = cubbs_bbs_scalar_le_window10(scalar, window);
      }
      const uint32_t value = raw + carry;
      const bool negative = value > half;
      const uint32_t digit = negative ? modulus - value : value;
      carry = negative ? 1u : 0u;
      if (digit != 0u) {
        acc.add(term_table[window * entries + digit], negative);
      }
    }
  }
  partials[tid] = acc;
  __syncthreads();
  const uint32_t active_threads =
      static_cast<uint32_t>(term_count < blockDim.x ? term_count : blockDim.x);
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride && tid + stride < active_threads) {
      partials[tid].add(partials[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

template <uint32_t kBits>
static cudaError_t cubbs_bbs_launch_affine_comb(
    const cubbs_sppark_plain_affine_t *table, const uint8_t *scalars,
    size_t term_count, size_t output_count, uint32_t threads,
    cudaStream_t stream, cubbs_sppark_bucket_t *outputs) {
  const uint32_t blocks = static_cast<uint32_t>(output_count);
  if (threads == 32u) {
    cubbs_bbs_g1_msm_fixed_signed_affine_comb_kernel<32u, kBits>
        <<<blocks, 32u, 0, stream>>>(table, scalars, term_count,
                                    output_count, outputs);
  } else if (threads == 64u) {
    cubbs_bbs_g1_msm_fixed_signed_affine_comb_kernel<64u, kBits>
        <<<blocks, 64u, 0, stream>>>(table, scalars, term_count,
                                    output_count, outputs);
  } else {
    cubbs_bbs_g1_msm_fixed_signed_affine_comb_kernel<128u, kBits>
        <<<blocks, 128u, 0, stream>>>(table, scalars, term_count,
                                     output_count, outputs);
  }
  return cudaGetLastError();
}

// Normalize 256 cached XYZZ comb entries with one Montgomery batch inversion.
// Digit-zero entries are infinity and are represented as affine (0, 0); the
// production kernels never load digit zero.
__global__ static void cubbs_bbs_g1_comb_xyzz_to_affine_kernel(
    const cubbs_sppark_bucket_t *input,
    cubbs_sppark_plain_affine_t *output, size_t count) {
  constexpr uint32_t kThreads = 256u;
  __shared__ __align__(16) uint8_t
      scan_storage[2u * kThreads * sizeof(fp_t)];
  __shared__ __align__(16) uint8_t total_storage[sizeof(fp_t)];
  auto *scan_a = reinterpret_cast<fp_t *>(scan_storage);
  auto *scan_b = scan_a + kThreads;
  auto *total_product = reinterpret_cast<fp_t *>(total_storage);
  const uint32_t tid = threadIdx.x;
  const size_t index = static_cast<size_t>(blockIdx.x) * kThreads + tid;
  const bool valid = index < count;
  cubbs_sppark_bucket_t point;
  if (valid) point = input[index];
  const bool finite = valid && !point.is_inf();
  const fp_t *coordinates = reinterpret_cast<const fp_t *>(&point);
  const fp_t one = fp_t::one();
  const fp_t z = finite ? coordinates[2] : one;

  scan_b[tid] = z;
  __syncthreads();
  if (tid == 0u) {
    fp_t product = one;
    for (uint32_t item = 0u; item < kThreads; ++item) {
      scan_a[item] = product;
      product *= scan_b[item];
    }
    *total_product = product;
  }
  __syncthreads();
  if (tid == 0u) {
    static constexpr uint64_t modulus_minus_two[6] = {
        0xb9feffffffffaaa9ull, 0x1eabfffeb153ffffull,
        0x6730d2a0f6b0f624ull, 0x64774b84f38512bfull,
        0x4b1ba7b6434bacd7ull, 0x1a0111ea397fe69aull};
    fp_t inverse = one;
    const fp_t base = *total_product;
    for (int bit = 380; bit >= 0; --bit) {
      inverse *= inverse;
      if (((modulus_minus_two[bit / 64] >> (bit % 64)) & 1ull) != 0ull) {
        inverse *= base;
      }
    }
    for (int item = static_cast<int>(kThreads) - 1; item >= 0; --item) {
      const fp_t inverse_item = inverse * scan_a[item];
      inverse *= scan_b[item];
      scan_a[item] = inverse_item;
    }
  }
  __syncthreads();

  if (valid) {
    if (!finite) {
      fp_t zero;
      zero.zero();
      output[index] = cubbs_sppark_plain_affine_t{zero, zero};
    } else {
      const fp_t inverse_zzz = scan_a[tid];
      fp_t affine_y = inverse_zzz * coordinates[1];
      fp_t inverse_z = inverse_zzz * coordinates[3];
      fp_t affine_x = inverse_z * inverse_z;
      affine_x *= coordinates[0];
      output[index] =
          cubbs_sppark_plain_affine_t{affine_x, affine_y};
    }
  }
}


#endif
