#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_LIMLEE_MSM_CUH
#define CUBBS_BBS_LIMLEE_MSM_CUH

// ============================================================================
// GLV-Split Signed Lim-Lee Multi-Generator Fixed-Base Comb MSM
// ============================================================================
// Mathematical identity (exact for all BLS12-381 scalars s_j in [0, r-1]):
//   1. Split s_j = k1_j + k2_j * z^2 (mod r) with 0 <= k1_j, k2_j < 2^128.
//   2. Let Q_j = 2^-1 * P_j (mod r). For any 128-bit integer k with bits b_i:
//        sum_{i=0}^{127} (2*b_i - 1) * 2^i * Q_j = k * P_j - ((2^128 - 1)/2) * P_j.
//   3. Group generators into groups of g in [11..18]. Every sign pattern
//      (s_0, ..., s_{gl-1}) in {-1, +1}^gl has s_0 = +1 or -1; by symmetry
//      only 2^(gl-1) affine points are stored per subtable!
//   4. Vertically split 128 bits into v subtables (v in {1, 2, 4}).
//   5. Result = acc1 + phi_{z^2}(acc2) + C_total, where
//        C_total = ((2^128 - 1)/2) * (1 + z^2) * sum_j P_j
//      is a single precomputed constant point!
//
// L2-Aware Tuning on RTX 5060 Ti (32 MB L2, 36 SMs):
//   - M=64  (66 terms): g=17, v=1 -> 4 groups (19.5 MB in L2!), 1,024 mixed adds
//   - M=256 (258 terms): g=15, v=1 -> 18 groups (25.5 MB in L2!), 4,608 mixed adds
//   - M=2048 (2050 terms): g=15, v=1 -> 137 groups (204 MB), 35,072 mixed adds

static constexpr uint32_t CUBBS_LIMLEE_MAX_GROUP_SIZE = 18u;

// Single-multiplier state-machine XYZZ doubling for BLS12-381 (a = 0).
// Costs 6M + 3S (vs 10M + 3S for generic xyzz_t::add(acc)) with 1 mul unit.
__device__ static void cubbs_bbs_xyzz_dbl_u(cubbs_sppark_bucket_t &pt) {
  if (pt.is_inf()) return;
  fp_t *c = reinterpret_cast<fp_t *>(&pt);
  fp_t X = c[0];
  fp_t Y = c[1];
  fp_t ZZZ = c[2];
  fp_t ZZ = c[3];

  fp_t U = Y + Y;
  fp_t A = U;
  fp_t B = U;
  fp_t V, W, S, M;
  int pc = -1;
  #pragma unroll 1
  do {
    A = A * B;
    switch (++pc) {
    case 0: // V = U^2
      V = A;
      B = U;
      break;
    case 1: // W = V * U
      W = A;
      A = X;
      B = V;
      break;
    case 2: // S = X * V
      S = A;
      A = X;
      B = X;
      break;
    case 3: // X^2 -> M = 3 * X^2
      M = A + A + A;
      A = M;
      B = M;
      break;
    case 4: // M^2 -> X3 = M^2 - 2*S
      X = A - S - S;
      A = S - X;
      B = M;
      break;
    case 5: // M * (S - X3)
      M = A;
      A = W;
      B = Y;
      break;
    case 6: // W * Y -> Y3 = M*(S-X3) - W*Y
      Y = M - A;
      A = V;
      B = ZZ;
      break;
    case 7: // ZZ3 = V * ZZ
      ZZ = A;
      A = W;
      B = ZZZ;
      break;
    case 8: // ZZZ3 = W * ZZZ
      ZZZ = A;
      break;
    }
  } while (pc < 8);

  c[0] = X;
  c[1] = Y;
  c[2] = ZZZ;
  c[3] = ZZ;
}

__device__ __forceinline__ static void cubbs_bbs_limlee_xyzz_add_noinline(
    cubbs_sppark_bucket_t &dst, const cubbs_sppark_bucket_t &src) {
  dst.uadd(src);
}

__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_limlee_scalar_mul_255_device(const cubbs_sppark_bucket_t &base,
                                       const uint8_t scalar_le[32]) {
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (int bit = 254; bit >= 0; --bit) {
    if (!acc.is_inf()) cubbs_bbs_xyzz_dbl_u(acc);
    if (((scalar_le[bit >> 3] >> (bit & 7)) & 1u) != 0u) {
      cubbs_bbs_limlee_xyzz_add_noinline(acc, base);
    }
  }
  return acc;
}

__global__ static void cubbs_bbs_limlee_prepare_q_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count, uint32_t v_sub,
    cubbs_sppark_bucket_t *q_out) {
  // inv2 = (r + 1) / 2 in little-endian 32 bytes.
  static constexpr uint8_t INV2_LE[32] = {
      0x01u, 0x00u, 0x00u, 0x80u, 0xffu, 0xffu, 0xffu, 0x7fu,
      0xffu, 0x2du, 0xffu, 0x7fu, 0x01u, 0xd2u, 0xdeu, 0xa9u,
      0x02u, 0xecu, 0xd0u, 0x04u, 0x04u, 0xecu, 0x9cu, 0x19u,
      0xa4u, 0xbeu, 0xceu, 0x94u, 0xa9u, 0xd3u, 0xf6u, 0x39u};
  const size_t term = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (term >= term_count) return;
  cubbs_sppark_bucket_t p;
  p = points[term];
  cubbs_sppark_bucket_t q = cubbs_bbs_limlee_scalar_mul_255_device(p, INV2_LE);
  const uint32_t num_bits = 128u / v_sub;
  q_out[term * v_sub] = q;
  for (uint32_t sv = 1u; sv < v_sub; ++sv) {
    for (uint32_t d = 0u; d < num_bits; ++d) cubbs_bbs_xyzz_dbl_u(q);
    q_out[term * v_sub + sv] = q;
  }
}

__global__ static void cubbs_bbs_limlee_prepare_c_total_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *c_total_out) {
  // C_TOTAL_COEFF = ((2^128 - 1) / 2) * (1 + z^2) mod r in little-endian 32 bytes.
  static constexpr uint8_t C_TOTAL_LE[32] = {
      0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0x7fu,
      0xffu, 0xffu, 0xffu, 0xffu, 0xfdu, 0x5bu, 0xfeu, 0x7fu,
      0xfdu, 0x13u, 0x2fu, 0x7bu, 0xfcu, 0x13u, 0x63u, 0xe6u,
      0x5cu, 0x13u, 0x32u, 0xebu, 0x56u, 0xfeu, 0x2bu, 0x1cu};
  __shared__ __align__(16) uint8_t smem[128u * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(smem);
  const uint32_t tid = threadIdx.x;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t term = tid; term < term_count; term += 128u) {
    acc.uadd(points[term]);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = 64u; stride > 0u; stride >>= 1u) {
    if (tid < stride) cubbs_bbs_limlee_xyzz_add_noinline(partials[tid], partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) {
    *c_total_out = cubbs_bbs_limlee_scalar_mul_255_device(partials[0], C_TOTAL_LE);
  }
}

// Builds all 2^(gl-1) signed subset sums for each subtable in binary-tree BFS.
__global__ static void cubbs_bbs_limlee_build_subtables_kernel(
    const cubbs_sppark_bucket_t *q_points, size_t term_count,
    uint32_t group_size, uint32_t v_sub,
    const size_t *subtable_offsets, cubbs_sppark_bucket_t *subtables_xyzz) {
  const uint32_t s = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t g = s / v_sub;
  const uint32_t sv = s - (g * v_sub);
  const size_t start = static_cast<size_t>(g) * group_size;
  if (start >= term_count) return;
  const uint32_t gl = static_cast<uint32_t>(
      min(static_cast<size_t>(group_size), term_count - start));
  cubbs_sppark_bucket_t *table = subtables_xyzz + subtable_offsets[s];

  if (tid == 0u) {
    table[0] = q_points[start * v_sub + sv];
  }
  __syncthreads();

  for (uint32_t k = 1u; k < gl; ++k) {
    const uint32_t half = 1u << (k - 1u);
    const cubbs_sppark_bucket_t qk = q_points[(start + k) * v_sub + sv];
    cubbs_sppark_bucket_t neg_qk = qk;
    neg_qk.cneg(true);
    for (uint32_t idx = tid; idx < half; idx += blockDim.x) {
      const cubbs_sppark_bucket_t base = table[idx];
      cubbs_sppark_bucket_t plus = base;
      cubbs_bbs_limlee_xyzz_add_noinline(plus, qk);
      cubbs_sppark_bucket_t minus = base;
      cubbs_bbs_limlee_xyzz_add_noinline(minus, neg_qk);
      table[idx + half] = plus;
      table[idx] = minus;
    }
    __syncthreads();
  }
}

// Kernel A: GLV split + coalesced bit-transpose into compact uint16_t lookup masks
// (when group_size <= 16: bit 15 = sign, bits 0..14 = idx in [0..32767], halving L2/DRAM traffic!).
template <typename MaskT>
__global__ static void cubbs_bbs_limlee_bit_transpose_kernel(
    const uint8_t *scalars_le, size_t term_count, size_t output_count,
    uint32_t num_groups, uint32_t group_size, uint32_t v_sub,
    MaskT *masks_out) {
  __shared__ uint64_t s_k[4u][CUBBS_LIMLEE_MAX_GROUP_SIZE][32u];
  const uint32_t tid = threadIdx.x;
  const size_t task = static_cast<size_t>(blockIdx.x) * 32u + tid;
  const size_t total_tasks = output_count * num_groups;
  if (task >= total_tasks) return;

  const uint32_t g = static_cast<uint32_t>(task / output_count);
  const size_t row = task - (static_cast<size_t>(g) * output_count);
  const size_t start = static_cast<size_t>(g) * group_size;
  const uint32_t gl = static_cast<uint32_t>(
      min(static_cast<size_t>(group_size), term_count - start));
  const uint8_t *row_scalars = scalars_le + (row * term_count * 32u);

  for (uint32_t k = 0u; k < gl; ++k) {
    uint8_t rem_le[16];
    uint8_t quot_le[16];
    cubbs_bbs_scalar_div_by_z2_device(
        row_scalars + ((start + k) * 32u), rem_le, quot_le);
    uint64_t w0, w1, w2, w3;
    memcpy(&w0, rem_le, 8u);
    memcpy(&w1, rem_le + 8u, 8u);
    memcpy(&w2, quot_le, 8u);
    memcpy(&w3, quot_le + 8u, 8u);
    s_k[0u][k][tid] = w0;
    s_k[1u][k][tid] = w1;
    s_k[2u][k][tid] = w2;
    s_k[3u][k][tid] = w3;
  }

  const uint32_t num_bits = 128u / v_sub;
  const size_t stride = output_count * 2u;
  const size_t col0 = row * 2u;
  const size_t col1 = col0 + 1u;
  const uint32_t flip_mask = (1u << (gl - 1u)) - 1u;
  constexpr uint32_t sign_bit = (sizeof(MaskT) == 2u) ? 0x8000u : 0x80000000u;

  for (uint32_t sv = 0u; sv < v_sub; ++sv) {
    const uint32_t s = g * v_sub + sv;
    const uint32_t bit_base = sv * num_bits;
    MaskT *sub_base = masks_out + (static_cast<size_t>(s) * num_bits * stride);
    for (uint32_t lb = 0u; lb < num_bits; ++lb) {
      const uint32_t bit = bit_base + lb;
      const uint32_t shift = bit & 63u;
      const uint32_t w_k1 = (bit >= 64u) ? 1u : 0u;
      const uint32_t w_k2 = w_k1 + 2u;

      const uint32_t b0_1 = static_cast<uint32_t>((s_k[w_k1][0u][tid] >> shift) & 1ull);
      uint32_t idx1 = 0u;
      for (uint32_t k = 1u; k < gl; ++k) {
        const uint32_t bk = static_cast<uint32_t>((s_k[w_k1][k][tid] >> shift) & 1ull);
        idx1 |= (bk << (k - 1u));
      }
      const uint32_t out1 = (b0_1 == 0u) ? (sign_bit | (idx1 ^ flip_mask)) : idx1;
      sub_base[static_cast<size_t>(lb) * stride + col0] = static_cast<MaskT>(out1);

      const uint32_t b0_2 = static_cast<uint32_t>((s_k[w_k2][0u][tid] >> shift) & 1ull);
      uint32_t idx2 = 0u;
      for (uint32_t k = 1u; k < gl; ++k) {
        const uint32_t bk = static_cast<uint32_t>((s_k[w_k2][k][tid] >> shift) & 1ull);
        idx2 |= (bk << (k - 1u));
      }
      const uint32_t out2 = (b0_2 == 0u) ? (sign_bit | (idx2 ^ flip_mask)) : idx2;
      sub_base[static_cast<size_t>(lb) * stride + col1] = static_cast<MaskT>(out2);
    }
  }
}

__device__ __forceinline__ static void cubbs_bbs_apply_glv_endomorphism(
    cubbs_sppark_bucket_t &pt) {
  if (!pt.is_inf()) {
    fp_t *coords = reinterpret_cast<fp_t *>(&pt);
    fp_t beta_sq;
    const uint64_t beta_sq_limbs[6] = {
        0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
        0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
    memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
    coords[0] = coords[0] * beta_sq;
    pt.cneg(true);
  }
}

// Kernel B (kSplits == 1): 2 threads per row, 32 rows per 64-thread block, in-block SMEM combine
// (no global-memory partials round-trip!) + compact uint16_t masks when group_size <= 16.
template <typename MaskT>
__global__ static void __launch_bounds__(64, 4)
cubbs_bbs_g1_msm_glv_limlee_direct_kernel(
    const cubbs_sppark_plain_affine_t *affines,
    const size_t *subtable_offsets,
    const cubbs_sppark_bucket_t *c_total,
    const MaskT *masks,
    size_t output_count, uint32_t total_subtables, uint32_t num_bits,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t smem[32u * sizeof(cubbs_sppark_bucket_t)];
  auto *k2_partials = reinterpret_cast<cubbs_sppark_bucket_t *>(smem);

  const uint32_t tid = threadIdx.x;
  const size_t col = static_cast<size_t>(blockIdx.x) * 64u + tid;
  const size_t row = col >> 1u;
  const uint32_t half = tid & 1u;
  const size_t stride = output_count * 2u;
  constexpr uint32_t sign_bit = (sizeof(MaskT) == 2u) ? 0x8000u : 0x80000000u;
  constexpr uint32_t idx_mask = sign_bit - 1u;

  cubbs_sppark_bucket_t acc;
  acc.inf();
  if (row < output_count) {
    for (int bit = static_cast<int>(num_bits) - 1; bit >= 0; --bit) {
      if (bit != static_cast<int>(num_bits) - 1) {
        cubbs_bbs_xyzz_dbl_u(acc);
      }
      for (uint32_t s = 0u; s < total_subtables; ++s) {
        const uint32_t entry = static_cast<uint32_t>(
            masks[(static_cast<size_t>(s) * num_bits + static_cast<size_t>(bit)) * stride + col]);
        const bool neg = (entry & sign_bit) != 0u;
        const uint32_t idx = entry & idx_mask;
        acc.uadd(affines[subtable_offsets[s] + idx], neg);
      }
    }
    if (half == 1u) {
      cubbs_bbs_apply_glv_endomorphism(acc);
      k2_partials[tid >> 1u] = acc;
    }
  }
  __syncthreads();
  if (row < output_count && half == 0u) {
    cubbs_bbs_limlee_xyzz_add_noinline(acc, k2_partials[tid >> 1u]);
    cubbs_bbs_limlee_xyzz_add_noinline(acc, *c_total);
    outputs[row] = acc;
  }
}

// Kernel B (kSplits > 1): in-block SMEM tree reduction across splits (no global partials round-trip!).
template <typename MaskT>
__global__ static void __launch_bounds__(64, 4)
cubbs_bbs_g1_msm_glv_limlee_split_kernel(
    const cubbs_sppark_plain_affine_t *affines,
    const size_t *subtable_offsets,
    const cubbs_sppark_bucket_t *c_total,
    const MaskT *masks,
    size_t output_count, uint32_t total_subtables, uint32_t num_bits,
    uint32_t splits,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t smem[64u * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(smem);

  const uint32_t threads_per_row = 2u * splits;
  const uint32_t rows_per_block = 64u / threads_per_row;
  const uint32_t tid = threadIdx.x;
  const uint32_t local_row = tid / threads_per_row;
  const uint32_t in_row = tid - (local_row * threads_per_row);
  const uint32_t split_id = in_row >> 1u;
  const uint32_t half = in_row & 1u;
  const size_t row = static_cast<size_t>(blockIdx.x) * rows_per_block + local_row;
  constexpr uint32_t sign_bit = (sizeof(MaskT) == 2u) ? 0x8000u : 0x80000000u;
  constexpr uint32_t idx_mask = sign_bit - 1u;

  cubbs_sppark_bucket_t acc;
  acc.inf();
  if (row < output_count) {
    const size_t stride = output_count * 2u;
    const size_t col = row * 2u + half;
    const uint32_t per_split = (total_subtables + splits - 1u) / splits;
    const uint32_t s_begin = split_id * per_split;
    const uint32_t s_end = min(total_subtables, s_begin + per_split);
    if (s_begin < s_end) {
      for (int bit = static_cast<int>(num_bits) - 1; bit >= 0; --bit) {
        if (bit != static_cast<int>(num_bits) - 1) {
          cubbs_bbs_xyzz_dbl_u(acc);
        }
        for (uint32_t s = s_begin; s < s_end; ++s) {
          const uint32_t entry = static_cast<uint32_t>(
              masks[(static_cast<size_t>(s) * num_bits + static_cast<size_t>(bit)) * stride + col]);
          const bool neg = (entry & sign_bit) != 0u;
          const uint32_t idx = entry & idx_mask;
          acc.uadd(affines[subtable_offsets[s] + idx], neg);
        }
      }
      if (half == 1u) {
        cubbs_bbs_apply_glv_endomorphism(acc);
      }
    }
  }
  partials[tid] = acc;
  __syncthreads();

  for (uint32_t stride_r = threads_per_row >> 1u; stride_r > 0u; stride_r >>= 1u) {
    if (in_row < stride_r) {
      cubbs_bbs_limlee_xyzz_add_noinline(partials[tid], partials[tid + stride_r]);
    }
    __syncthreads();
  }
  if (row < output_count && in_row == 0u) {
    cubbs_sppark_bucket_t sum = partials[tid];
    cubbs_bbs_limlee_xyzz_add_noinline(sum, *c_total);
    outputs[row] = sum;
  }
}

static uint32_t cubbs_bbs_limlee_target_threads(uint32_t total_subtables) {
  static const uint32_t env_target = []() -> uint32_t {
    const char *env = std::getenv("CUBBS_BBS_LIMLEE_TARGET_THREADS");
    if (env != nullptr && env[0] != '\0') {
      const unsigned long parsed = std::strtoul(env, nullptr, 10);
      if (parsed >= 64ul && parsed <= 65536ul) return static_cast<uint32_t>(parsed);
    }
    return 0u;
  }();
  if (env_target != 0u) return env_target;
  return (total_subtables >= 17u && total_subtables <= 32u) ? 2048u : 4096u;
}

template <typename MaskT>
static cudaError_t cubbs_bbs_launch_glv_limlee_typed(
    const cubbs_sppark_plain_affine_t *affines,
    const size_t *subtable_offsets,
    const cubbs_sppark_bucket_t *c_total,
    const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    uint32_t num_groups, uint32_t group_size, uint32_t v_sub,
    cudaStream_t stream, cubbs_sppark_bucket_t *outputs) {
  const uint32_t total_subtables = num_groups * v_sub;
  const uint32_t num_bits = 128u / v_sub;
  const size_t mask_count =
      static_cast<size_t>(total_subtables) * num_bits * output_count * 2u;

  auto &msm_arena = cubbs_bbs_thread_msm_arena();
  cudaError_t err = msm_arena.reserve_and_reset(
      CuBbsDeviceArena::align_up(mask_count * sizeof(MaskT)));
  if (err != cudaSuccess) return err;
  MaskT *device_masks = msm_arena.alloc<MaskT>(mask_count);

  {
    constexpr uint32_t tr_threads = 32u;
    const size_t total_tasks = output_count * num_groups;
    const uint32_t tr_blocks =
        static_cast<uint32_t>((total_tasks + tr_threads - 1u) / tr_threads);
    cubbs_bbs_limlee_bit_transpose_kernel<MaskT><<<tr_blocks, tr_threads, 0, stream>>>(
        scalars_le, term_count, output_count, num_groups, group_size, v_sub, device_masks);
    err = cudaGetLastError();
  }

  if (err == cudaSuccess) {
    const uint32_t target_threads = cubbs_bbs_limlee_target_threads(total_subtables);
    uint32_t splits = 1u;
    while (splits < 32u && splits * 2u <= total_subtables &&
           (output_count * 2u * splits) < target_threads) {
      splits <<= 1u;
    }
    if (splits == 1u) {
      const uint32_t blocks = static_cast<uint32_t>((output_count * 2u + 63u) / 64u);
      cubbs_bbs_g1_msm_glv_limlee_direct_kernel<MaskT><<<blocks, 64u, 0, stream>>>(
          affines, subtable_offsets, c_total, device_masks,
          output_count, total_subtables, num_bits, outputs);
    } else {
      const uint32_t rows_per_block = 64u / (2u * splits);
      const uint32_t blocks = static_cast<uint32_t>(
          (output_count + rows_per_block - 1u) / rows_per_block);
      cubbs_bbs_g1_msm_glv_limlee_split_kernel<MaskT><<<blocks, 64u, 0, stream>>>(
          affines, subtable_offsets, c_total, device_masks,
          output_count, total_subtables, num_bits, splits, outputs);
    }
    err = cudaGetLastError();
  }
  return err;
}

static cudaError_t cubbs_bbs_launch_glv_limlee(
    const cubbs_sppark_plain_affine_t *affines,
    const size_t *subtable_offsets,
    const cubbs_sppark_bucket_t *c_total,
    const uint8_t *scalars_le,
    size_t term_count, size_t output_count,
    uint32_t num_groups, uint32_t group_size, uint32_t v_sub,
    cudaStream_t stream, cubbs_sppark_bucket_t *outputs) {
  if (output_count == 0u || term_count == 0u) return cudaSuccess;
  if (group_size <= 16u) {
    return cubbs_bbs_launch_glv_limlee_typed<uint16_t>(
        affines, subtable_offsets, c_total, scalars_le, term_count, output_count,
        num_groups, group_size, v_sub, stream, outputs);
  }
  return cubbs_bbs_launch_glv_limlee_typed<uint32_t>(
      affines, subtable_offsets, c_total, scalars_le, term_count, output_count,
      num_groups, group_size, v_sub, stream, outputs);
}

#endif // CUBBS_BBS_LIMLEE_MSM_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
