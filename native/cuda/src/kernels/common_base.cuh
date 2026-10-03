#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

__device__ __forceinline__ static uint8_t cubbs_bbs_scalar_le_window4(
    const uint8_t *scalar_le, uint32_t window) {
  const uint32_t bit = window << 2u;
  const uint32_t byte_index = bit >> 3u;
  const uint32_t shift = bit & 7u;
  return static_cast<uint8_t>((scalar_le[byte_index] >> shift) & 0x0fu);
}

__device__ __forceinline__ static uint8_t cubbs_bbs_scalar_le_window5(
    const uint8_t *scalar_le, uint32_t window) {
  const uint32_t bit = window * 5u;
  const uint32_t byte_index = bit >> 3u;
  const uint32_t shift = bit & 7u;
  uint16_t word = byte_index < 32u ? scalar_le[byte_index] : 0u;
  if (byte_index + 1u < 32u) {
    word |= static_cast<uint16_t>(scalar_le[byte_index + 1u]) << 8u;
  }
  return static_cast<uint8_t>((word >> shift) & 0x1fu);
}

__device__ __forceinline__ static uint8_t cubbs_bbs_scalar_le_window6(
    const uint8_t *scalar_le, uint32_t window) {
  const uint32_t bit = window * 6u;
  const uint32_t byte_index = bit >> 3u;
  const uint32_t shift = bit & 7u;
  uint16_t word = byte_index < 32u ? scalar_le[byte_index] : 0u;
  if (byte_index + 1u < 32u) {
    word |= static_cast<uint16_t>(scalar_le[byte_index + 1u]) << 8u;
  }
  return static_cast<uint8_t>((word >> shift) & 0x3fu);
}

__device__ __forceinline__ static uint8_t cubbs_bbs_scalar_le_window8(
    const uint8_t *scalar_le, uint32_t window) {
  return scalar_le[window];
}

__device__ __forceinline__ static uint16_t cubbs_bbs_scalar_le_window10(
    const uint8_t *scalar_le, uint32_t window) {
  const uint32_t bit = window * 10u;
  const uint32_t byte_index = bit >> 3u;
  const uint32_t shift = bit & 7u;
  uint32_t word = byte_index < 32u ? scalar_le[byte_index] : 0u;
  if (byte_index + 1u < 32u)
    word |= static_cast<uint32_t>(scalar_le[byte_index + 1u]) << 8u;
  if (byte_index + 2u < 32u)
    word |= static_cast<uint32_t>(scalar_le[byte_index + 2u]) << 16u;
  return static_cast<uint16_t>((word >> shift) & 0x03ffu);
}

__device__ __forceinline__ static uint16_t cubbs_bbs_scalar_le_window9(
    const uint8_t *scalar_le, uint32_t window) {
  const uint32_t bit = window * 9u;
  const uint32_t byte_index = bit >> 3u;
  const uint32_t shift = bit & 7u;
  uint32_t word = byte_index < 32u ? scalar_le[byte_index] : 0u;
  if (byte_index + 1u < 32u)
    word |= static_cast<uint32_t>(scalar_le[byte_index + 1u]) << 8u;
  if (byte_index + 2u < 32u)
    word |= static_cast<uint32_t>(scalar_le[byte_index + 2u]) << 16u;
  return static_cast<uint16_t>((word >> shift) & 0x01ffu);
}

__global__ static void cubbs_bbs_g1_msm_common_points_window4_kernel(
    const cubbs_sppark_plain_affine_t *points,
    const uint8_t *scalars_le, size_t term_count,
    size_t output_count, cubbs_sppark_bucket_t *outputs) {
  constexpr uint32_t kThreadsPerRow = 128u;
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
    const cubbs_sppark_plain_affine_t point = points[term];
    const uint8_t *scalar = row_scalars + (term * 32u);

    cubbs_sppark_bucket_t multiples[16];
    multiples[0].inf();
    multiples[1] = point;
    for (uint32_t digit = 2u; digit < 16u; ++digit) {
      multiples[digit] = multiples[digit - 1u];
      multiples[digit].add(point);
    }

    for (int window = 63; window >= 0; --window) {
      if (!term_acc.is_inf()) {
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
          term_acc.add(term_acc);
        }
      }
      const uint8_t digit =
          cubbs_bbs_scalar_le_window4(scalar, static_cast<uint32_t>(window));
      if (digit != 0u) {
        term_acc.add(multiples[digit]);
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

__global__ static void
cubbs_bbs_g1_msm_common_points_window4_precomputed_kernel(
    const cubbs_sppark_bucket_t *point_multiples,
    const uint8_t *scalars_le, size_t term_count,
    size_t output_count, cubbs_sppark_bucket_t *outputs) {
  constexpr uint32_t kThreadsPerRow = 128u;
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
    const cubbs_sppark_bucket_t *multiples =
        point_multiples + (term * 16u);

    for (int window = 63; window >= 0; --window) {
      if (!term_acc.is_inf()) {
#pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
          term_acc.add(term_acc);
        }
      }
      const uint8_t digit =
          cubbs_bbs_scalar_le_window4(scalar, static_cast<uint32_t>(window));
      if (digit != 0u) {
        term_acc.add(multiples[digit]);
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


#endif
