#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

__global__ static void cubbs_bbs_g1_window4_multiples_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *point_multiples) {
  const size_t term = blockIdx.x;
  const uint32_t digit = threadIdx.x;
  if (term >= term_count || digit >= 16u) return;
  auto *multiples = point_multiples + term * 16u;
  if (digit == 0u) {
    multiples[0].inf();
    return;
  }
  cubbs_sppark_bucket_t multiple = points[term];
  for (uint32_t value = 1u; value < digit; ++value) {
    multiple.add(points[term]);
  }
  multiples[digit] = multiple;
}

__global__ static void cubbs_bbs_g1_window8_multiples_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *point_multiples) {
  const size_t term = blockIdx.x * blockDim.x + threadIdx.x;
  if (term >= term_count) return;
  auto *multiples = point_multiples + term * 256u;
  multiples[0].inf();
  multiples[1] = points[term];
  for (uint32_t digit = 2u; digit < 256u; ++digit) {
    multiples[digit] = multiples[digit - 1u];
    multiples[digit].add(points[term]);
  }
}

// The beta^2 Montgomery limbs and z^2 GLV split used below are derived from
// BLST src/e1.c (Copyright Supranational LLC, Apache-2.0). See NOTICE and
// THIRD_PARTY.md; the surrounding scheduling and kernels are cuBBS-Cred code.
__global__ static void cubbs_bbs_g1_glv_window4_multiples_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *point_multiples) {
  const size_t term = blockIdx.x;
  const uint32_t digit = threadIdx.x;
  if (term >= term_count || digit >= 16u) return;
  auto *multiples = point_multiples + term * 32u;
  if (digit == 0u) {
    multiples[0].inf();
    multiples[16u].inf();
    return;
  }
  cubbs_sppark_bucket_t multiple = points[term];
  for (uint32_t value = 1u; value < digit; ++value) multiple.add(points[term]);
  multiples[digit] = multiple;

  fp_t *coordinates = reinterpret_cast<fp_t *>(&multiple);
  fp_t beta_sq;
  const uint64_t beta_sq_limbs[6] = {
      0x30f1361b798a64e8, 0xf3b8ddab7ece5a2a, 0x16a8ca3ac61577f7,
      0xc26a2ff874fd029b, 0x3636b76660701c6e, 0x051ba4ab241b6160};
  memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
  coordinates[0] = coordinates[0] * beta_sq;
  multiple.cneg(true);
  multiples[16u + digit] = multiple;
}

__global__ static void cubbs_bbs_g1_glv_wnaf8_multiples_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *point_multiples) {
  const size_t term = blockIdx.x;
  const uint32_t slot = threadIdx.x;
  if (term >= term_count || slot >= 64u) return;
  const uint32_t odd = slot * 2u + 1u;
  cubbs_sppark_bucket_t multiple = points[term];
  for (uint32_t value = 1u; value < odd; ++value) multiple.add(points[term]);
  auto *multiples = point_multiples + term * 128u;
  multiples[slot] = multiple;
  fp_t *coordinates = reinterpret_cast<fp_t *>(&multiple);
  fp_t beta_sq;
  const uint64_t beta_sq_limbs[6] = {
      0x30f1361b798a64e8, 0xf3b8ddab7ece5a2a, 0x16a8ca3ac61577f7,
      0xc26a2ff874fd029b, 0x3636b76660701c6e, 0x051ba4ab241b6160};
  memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
  coordinates[0] = coordinates[0] * beta_sq;
  multiple.cneg(true);
  multiples[64u + slot] = multiple;
}

template <uint32_t kThreadsPerTile>
__global__ static void cubbs_bbs_g1_msm_large_window4_tiled_kernel(
    const cubbs_sppark_bucket_t *point_multiples,
    const uint8_t *scalars_le, size_t term_count, size_t tile_terms,
    size_t tile_count, size_t output_count,
    cubbs_sppark_bucket_t *tile_partials) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerTile * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  const size_t tile = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  if (row >= output_count || tile >= tile_count) return;

  const size_t begin = tile * tile_terms;
  const size_t end = min(term_count, begin + tile_terms);
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t term = begin + tid; term < end; term += kThreadsPerTile) {
    cubbs_sppark_bucket_t term_acc;
    term_acc.inf();
    const uint8_t *scalar = row_scalars + term * 32u;
    const auto *multiples = point_multiples + term * 16u;
    for (int window = 63; window >= 0; --window) {
      if (!term_acc.is_inf()) {
#pragma unroll
        for (uint32_t doubling = 0u; doubling < 4u; ++doubling) {
          term_acc.add(term_acc);
        }
      }
      const uint8_t digit = cubbs_bbs_scalar_le_window4(
          scalar, static_cast<uint32_t>(window));
      if (digit != 0u) term_acc.add(multiples[digit]);
    }
    acc.add(term_acc);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreadsPerTile >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) tile_partials[row * tile_count + tile] = partials[0];
}

__device__ __forceinline__ static void cubbs_bbs_scalar_div_by_z2_device(
    const uint8_t scalar_le[32], uint8_t remainder_le[16],
    uint8_t quotient_le[16]);

template <uint32_t kThreadsPerTile>
__global__ static void cubbs_bbs_g1_msm_large_glv_window4_tiled_kernel(
    const cubbs_sppark_bucket_t *point_multiples,
    const uint8_t *scalars_le, size_t term_count, size_t tile_terms,
    size_t tile_count, size_t output_count,
    cubbs_sppark_bucket_t *tile_partials) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerTile * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  const size_t tile = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  if (row >= output_count || tile >= tile_count) return;
  const size_t begin = tile * tile_terms;
  const size_t tile_end = begin + tile_terms;
  const size_t end = tile_end < term_count ? tile_end : term_count;
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t term = begin + tid; term < end; term += kThreadsPerTile) {
    cubbs_sppark_bucket_t term_acc;
    term_acc.inf();
    bool started = false;
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    cubbs_bbs_scalar_div_by_z2_device(
        row_scalars + term * 32u, remainder_le, quotient_le);
    const auto *multiples = point_multiples + term * 32u;
    for (int window = 31; window >= 0; --window) {
      if (started) {
#pragma unroll
        for (uint32_t doubling = 0u; doubling < 4u; ++doubling)
          term_acc.add(term_acc);
      }
      const uint8_t d1 = cubbs_bbs_scalar_le_window4(
          remainder_le, static_cast<uint32_t>(window));
      const uint8_t d2 = cubbs_bbs_scalar_le_window4(
          quotient_le, static_cast<uint32_t>(window));
      if (d1 != 0u) {
        if (!started) { term_acc = multiples[d1]; started = true; }
        else term_acc.add(multiples[d1]);
      }
      if (d2 != 0u) {
        if (!started) { term_acc = multiples[16u + d2]; started = true; }
        else term_acc.add(multiples[16u + d2]);
      }
    }
    acc.add(term_acc);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreadsPerTile >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) tile_partials[row * tile_count + tile] = partials[0];
}

__device__ __forceinline__ static void cubbs_bbs_wnaf8_recode_128(
    const uint8_t scalar_le[16], int8_t digits[130]) {
  uint64_t value[3] = {0u, 0u, 0u};
  memcpy(value, scalar_le, 16u);
  for (uint32_t bit = 0u; bit < 130u; ++bit) {
    int32_t digit = 0;
    if ((value[0] & 1u) != 0u) {
      digit = static_cast<int32_t>(value[0] & 0xffu);
      if (digit > 127) digit -= 256;
      if (digit > 0) {
        const uint64_t subtract = static_cast<uint64_t>(digit);
        const uint64_t old = value[0];
        value[0] -= subtract;
        uint64_t borrow = old < subtract;
        for (uint32_t limb = 1u; limb < 3u && borrow != 0u; ++limb) {
          const uint64_t prior = value[limb];
          value[limb] -= borrow;
          borrow = prior == 0u;
        }
      } else {
        const uint64_t add = static_cast<uint64_t>(-digit);
        const uint64_t old = value[0];
        value[0] += add;
        uint64_t carry = value[0] < old;
        for (uint32_t limb = 1u; limb < 3u && carry != 0u; ++limb) {
          const uint64_t prior = value[limb];
          value[limb] += carry;
          carry = value[limb] < prior;
        }
      }
    }
    digits[bit] = static_cast<int8_t>(digit);
    value[0] = (value[0] >> 1u) | (value[1] << 63u);
    value[1] = (value[1] >> 1u) | (value[2] << 63u);
    value[2] >>= 1u;
  }
}

__global__ static void cubbs_bbs_glv_wnaf8_recode_kernel(
    const uint8_t *scalars_le, size_t scalar_count, int8_t *digits) {
  const size_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar >= scalar_count) return;
  uint8_t remainder_le[16];
  uint8_t quotient_le[16];
  cubbs_bbs_scalar_div_by_z2_device(
      scalars_le + scalar * 32u, remainder_le, quotient_le);
  int8_t *scalar_digits = digits + scalar * 260u;
  cubbs_bbs_wnaf8_recode_128(remainder_le, scalar_digits);
  cubbs_bbs_wnaf8_recode_128(quotient_le, scalar_digits + 130u);
}

template <uint32_t kThreadsPerTile>
__global__ static void cubbs_bbs_g1_msm_large_glv_wnaf8_tiled_kernel(
    const cubbs_sppark_bucket_t *point_multiples,
    const int8_t *scalar_digits, size_t term_count, size_t tile_terms,
    size_t tile_count, size_t output_count,
    cubbs_sppark_bucket_t *tile_partials) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerTile * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  const size_t tile = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  if (row >= output_count || tile >= tile_count) return;
  const size_t begin = tile * tile_terms;
  const size_t tile_end = begin + tile_terms;
  const size_t end = tile_end < term_count ? tile_end : term_count;
  const int8_t *row_digits = scalar_digits + row * term_count * 260u;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t term = begin + tid; term < end; term += kThreadsPerTile) {
    const int8_t *term_digits = row_digits + term * 260u;
    const auto *multiples = point_multiples + term * 128u;
    cubbs_sppark_bucket_t term_acc;
    term_acc.inf();
    bool started = false;
    for (int bit = 129; bit >= 0; --bit) {
      if (started) term_acc.add(term_acc);
      const int32_t digits[2] = {term_digits[bit], term_digits[130 + bit]};
#pragma unroll
      for (uint32_t component = 0u; component < 2u; ++component) {
        const int32_t digit = digits[component];
        if (digit == 0) continue;
        const uint32_t magnitude =
            static_cast<uint32_t>(digit < 0 ? -digit : digit);
        auto addend = multiples[component * 64u + ((magnitude - 1u) >> 1u)];
        if (digit < 0) addend.cneg(true);
        if (!started) { term_acc = addend; started = true; }
        else term_acc.add(addend);
      }
    }
    acc.add(term_acc);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreadsPerTile >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) tile_partials[row * tile_count + tile] = partials[0];
}

template <uint32_t kThreads>
__global__ static void cubbs_bbs_g1_msm_large_reduce_kernel(
    const cubbs_sppark_bucket_t *tile_partials, size_t tile_count,
    size_t output_count, cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreads * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (row >= output_count) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t tile = tid; tile < tile_count; tile += kThreads) {
    acc.add(tile_partials[row * tile_count + tile]);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreads >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

template <uint32_t kThreads>
__global__ static void cubbs_bbs_g1_msm_large_pippenger_locked_buckets_kernel(
    const cubbs_sppark_plain_affine_t *points, const uint8_t *scalars_le,
    size_t term_count, size_t tile_terms, size_t tile_count,
    size_t output_count, cubbs_sppark_bucket_t *bucket_partials) {
  __shared__ __align__(16) uint8_t
      bucket_storage[15u * sizeof(cubbs_sppark_bucket_t)];
  __shared__ uint32_t bucket_locks[15u];
  auto *buckets = reinterpret_cast<cubbs_sppark_bucket_t *>(bucket_storage);
  const size_t row = blockIdx.x;
  const uint32_t window = blockIdx.y;
  const size_t tile = blockIdx.z;
  const uint32_t tid = threadIdx.x;
  if (row >= output_count || window >= 64u || tile >= tile_count) return;
  if (tid < 15u) {
    buckets[tid].inf();
    bucket_locks[tid] = 0u;
  }
  __syncthreads();

  const size_t begin = tile * tile_terms;
  const size_t tile_end = begin + tile_terms;
  const size_t end = tile_end < term_count ? tile_end : term_count;
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  for (size_t term = begin + tid; term < end; term += kThreads) {
    const uint8_t digit = cubbs_bbs_scalar_le_window4(
        row_scalars + term * 32u, window);
    if (digit == 0u) continue;
    const uint32_t slot = static_cast<uint32_t>(digit - 1u);
    while (atomicCAS(bucket_locks + slot, 0u, 1u) != 0u) {
    }
    buckets[slot].add(points[term]);
    __threadfence_block();
    atomicExch(bucket_locks + slot, 0u);
  }
  __syncthreads();
  if (tid < 15u) {
    const size_t output_base =
        (((row * 64u + window) * tile_count + tile) * 15u);
    bucket_partials[output_base + tid] = buckets[tid];
  }
}

template <uint32_t kThreads>
__global__ static void cubbs_bbs_g1_msm_large_pippenger_windows_kernel(
    const cubbs_sppark_bucket_t *bucket_partials, size_t tile_count,
    size_t output_count, cubbs_sppark_bucket_t *window_partials) {
  __shared__ __align__(16) uint8_t
      bucket_storage[kThreads * sizeof(cubbs_sppark_bucket_t)];
  auto *buckets = reinterpret_cast<cubbs_sppark_bucket_t *>(bucket_storage);
  const size_t row = blockIdx.x;
  const uint32_t window = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  if (row >= output_count || window >= 64u) return;

  cubbs_sppark_bucket_t bucket_sum;
  bucket_sum.inf();
  if (tid < 15u) {
    for (size_t tile = 0u; tile < tile_count; ++tile) {
      const size_t index =
          (((row * 64u + window) * tile_count + tile) * 15u) + tid;
      bucket_sum.add(bucket_partials[index]);
    }
  }
  buckets[tid] = bucket_sum;
  __syncthreads();
  if (tid == 0u) {
    cubbs_sppark_bucket_t running;
    cubbs_sppark_bucket_t weighted;
    running.inf();
    weighted.inf();
    for (int bucket = 14; bucket >= 0; --bucket) {
      running.add(buckets[bucket]);
      weighted.add(running);
    }
    window_partials[row * 64u + window] = weighted;
  }
}

__global__ static void cubbs_bbs_g1_msm_large_pippenger_rows_kernel(
    const cubbs_sppark_bucket_t *window_partials, size_t output_count,
    cubbs_sppark_bucket_t *outputs) {
  const size_t row = blockIdx.x;
  if (row >= output_count || threadIdx.x != 0u) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (int window = 63; window >= 0; --window) {
    if (!acc.is_inf()) {
#pragma unroll
      for (uint32_t doubling = 0u; doubling < 4u; ++doubling) acc.add(acc);
    }
    acc.add(window_partials[row * 64u + static_cast<size_t>(window)]);
  }
  outputs[row] = acc;
}

template <uint32_t kThreadsPerRow>
__global__ static void
cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel(
    const cubbs_sppark_bucket_t *point_multiples,
    const uint8_t *scalars_le, size_t term_count,
    size_t output_count, cubbs_sppark_bucket_t *outputs) {
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
        point_multiples + (term * 256u);

    for (int window = 31; window >= 0; --window) {
      if (!term_acc.is_inf()) {
#pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
          term_acc.add(term_acc);
        }
      }
      const uint8_t digit =
          cubbs_bbs_scalar_le_window8(scalar, static_cast<uint32_t>(window));
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
