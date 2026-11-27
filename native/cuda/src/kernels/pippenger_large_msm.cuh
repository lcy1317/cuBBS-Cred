#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_PIPPENGER_LARGE_MSM_CUH
#define CUBBS_BBS_PIPPENGER_LARGE_MSM_CUH

// ============================================================================
// cuBBS-Cred GPU Adaptive-W (10-bit / 13-bit) Signed GLV Bucket-Histogram
// Zero-Spill Pippenger MSM (M > 2100)
// ============================================================================
// - Adaptive window width:
//     K <= 12288: W = 10 (13 windows x  512 buckets =  6,656 buckets/row)
//     K >  12288: W = 13 (10 windows x 4096 buckets = 40,960 buckets/row, -23% bucket adds!)
// - Zero-spill, zero-atomic Pass 4 (1 thread per (row, win, bucket), __launch_bounds__(64, 4))
//   keeps all 36 SMs saturated even at B = 1 while cutting average bucket depth from
//   256 -> 32 at M=65,536 (dramatically reducing Poisson tail skew!).

// Recode a 128-bit LE integer into kWindows signed W-bit digits in [-2^(W-1), +2^(W-1)].
template <uint32_t kW, uint32_t kWindows>
__device__ __forceinline__ static void cubbs_large_pip_recode_128(
    const uint8_t bytes16[16], uint16_t packed_out[kWindows]) {
  constexpr uint32_t kBuckets = 1u << (kW - 1u);
  constexpr uint32_t kMask = (1u << kW) - 1u;
  constexpr uint32_t kValidBit = 1u << (kW - 1u);
  constexpr uint32_t kNegBit = 1u << kW;

  uint64_t lo = 0ull, hi = 0ull;
  memcpy(&lo, bytes16, 8u);
  memcpy(&hi, bytes16 + 8u, 8u);

  int32_t carry = 0;
  #pragma unroll
  for (uint32_t w = 0u; w < kWindows; ++w) {
    const uint32_t bit = w * kW;
    uint32_t raw = 0u;
    if (bit + kW <= 64u) {
      raw = static_cast<uint32_t>((lo >> bit) & static_cast<uint64_t>(kMask));
    } else if (bit >= 64u) {
      const uint32_t sh = bit - 64u;
      raw = static_cast<uint32_t>((hi >> sh) & static_cast<uint64_t>(kMask));
    } else {
      const uint32_t lo_bits = 64u - bit;
      raw = static_cast<uint32_t>((lo >> bit) | ((hi << lo_bits) & static_cast<uint64_t>(kMask)));
    }
    const int32_t val = static_cast<int32_t>(raw) + carry;
    int32_t digit = val;
    carry = 0;
    if (w + 1u < kWindows && val > static_cast<int32_t>(kBuckets)) {
      digit = val - static_cast<int32_t>(1u << kW);
      carry = 1;
    }
    if (digit == 0) {
      packed_out[w] = 0u;
    } else {
      const bool neg = digit < 0;
      const uint32_t mag = static_cast<uint32_t>(neg ? -digit : digit);
      packed_out[w] = static_cast<uint16_t>((mag - 1u) | kValidBit | (neg ? kNegBit : 0u));
    }
  }
}

// Pass 1: GLV split + signed W-bit recode + global L2 histogram count per (row, win, bucket).
template <uint32_t kW, uint32_t kWindows>
__global__ static void cubbs_large_pip_recode_count_kernel(
    const uint8_t *scalars_le,
    size_t term_count,
    size_t chunk_rows,
    uint16_t *digits_out,      // [chunk_rows * term_count * (2 * kWindows)]
    uint32_t *bucket_counts) { // [chunk_rows * kWindows * kBuckets]
  constexpr uint32_t kBuckets = 1u << (kW - 1u);
  constexpr uint32_t kBucketMask = kBuckets - 1u;
  constexpr uint32_t kValidBit = kBuckets;

  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = chunk_rows * term_count;
  if (idx >= total) return;
  const size_t row = idx / term_count;

  uint8_t k1[16], k2[16];
  cubbs_bbs_scalar_div_by_z2_device(scalars_le + (idx * 32u), k1, k2);

  uint16_t d1[kWindows];
  uint16_t d2[kWindows];
  cubbs_large_pip_recode_128<kW, kWindows>(k1, d1);
  cubbs_large_pip_recode_128<kW, kWindows>(k2, d2);

  uint16_t *dst = digits_out + (idx * (2u * kWindows));
  uint32_t *row_counts = bucket_counts + (row * kWindows * kBuckets);
  #pragma unroll
  for (uint32_t w = 0u; w < kWindows; ++w) {
    const uint16_t p1 = d1[w];
    const uint16_t p2 = d2[w];
    dst[w] = p1;
    dst[kWindows + w] = p2;
    if ((p1 & kValidBit) != 0u) {
      atomicAdd(&row_counts[w * kBuckets + (p1 & kBucketMask)], 1u);
    }
    if ((p2 & kValidBit) != 0u) {
      atomicAdd(&row_counts[w * kBuckets + (p2 & kBucketMask)], 1u);
    }
  }
}

// Pass 2: Exclusive prefix sum over kBuckets (512 or 4096) per (row, win) using 512 threads.
template <uint32_t kBuckets>
__global__ static void cubbs_large_pip_prefix_sum_kernel(
    const uint32_t *bucket_counts,
    uint32_t *bucket_offsets,
    uint32_t *bucket_cursors) {
  constexpr uint32_t kThreads = 512u;
  constexpr uint32_t kPerThread = kBuckets / kThreads;
  __shared__ uint32_t s_buf[kThreads];

  const uint32_t tid = threadIdx.x;
  const size_t base = static_cast<size_t>(blockIdx.x) * kBuckets;
  const uint32_t start = tid * kPerThread;

  uint32_t local[kPerThread];
  uint32_t sum = 0u;
  #pragma unroll
  for (uint32_t k = 0u; k < kPerThread; ++k) {
    const uint32_t c = bucket_counts[base + start + k];
    local[k] = sum;
    sum += c;
  }
  s_buf[tid] = sum;
  __syncthreads();

  #pragma unroll
  for (uint32_t stride = 1u; stride < kThreads; stride <<= 1u) {
    const uint32_t idx = ((tid + 1u) * (stride << 1u)) - 1u;
    if (idx < kThreads) {
      s_buf[idx] += s_buf[idx - stride];
    }
    __syncthreads();
  }
  if (tid == 0u) s_buf[kThreads - 1u] = 0u;
  __syncthreads();
  #pragma unroll
  for (uint32_t stride = kThreads >> 1u; stride >= 1u; stride >>= 1u) {
    const uint32_t idx = ((tid + 1u) * (stride << 1u)) - 1u;
    if (idx < kThreads) {
      const uint32_t t = s_buf[idx - stride];
      s_buf[idx - stride] = s_buf[idx];
      s_buf[idx] += t;
    }
    __syncthreads();
  }

  const uint32_t base_off = s_buf[tid];
  #pragma unroll
  for (uint32_t k = 0u; k < kPerThread; ++k) {
    const uint32_t off = base_off + local[k];
    bucket_offsets[base + start + k] = off;
    bucket_cursors[base + start + k] = off;
  }
}

// One-time cache fill: computes points_glv[term_count + j] = phi(P_j) = (beta^2*x_j, -y_j)
// so Pass 4 eliminates 100% of is_k2 branches, 655k Fp muls/row, and 12 beta_sq regs!
__global__ static void cubbs_large_pip_fill_glv_points_kernel(
    cubbs_sppark_plain_affine_t *points_glv, size_t term_count) {
  const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= term_count) return;
  cubbs_sppark_plain_affine_t p = points_glv[j];
  if (!p.is_inf()) {
    fp_t beta_sq;
    const uint64_t beta_sq_limbs[6] = {
        0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
        0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
    memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
    fp_t *pc = reinterpret_cast<fp_t *>(&p);
    pc[0] = pc[0] * beta_sq;
    pc[1].cneg(true);
  }
  points_glv[term_count + j] = p;
}

// Pass 3: Scatter packed uint32_t entries = (glv_point_idx << 1) | neg
// where glv_point_idx = term (for k1) or term + term_count (for k2 -> precomputed phi(P)!).
template <uint32_t kW, uint32_t kWindows>
__global__ static void cubbs_large_pip_scatter_kernel(
    const uint16_t *digits_in,
    size_t term_count,
    size_t chunk_rows,
    uint32_t *bucket_cursors,
    uint32_t *sorted_entries) {
  constexpr uint32_t kBuckets = 1u << (kW - 1u);
  constexpr uint32_t kBucketMask = kBuckets - 1u;
  constexpr uint32_t kValidBit = kBuckets;

  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = chunk_rows * term_count;
  if (idx >= total) return;
  const size_t row = idx / term_count;
  const uint32_t term = static_cast<uint32_t>(idx - (row * term_count));
  const uint32_t term_k2 = term + static_cast<uint32_t>(term_count);

  const uint16_t *src = digits_in + (idx * (2u * kWindows));
  const size_t win_stride = 2u * term_count;
  uint32_t *row_cursors = bucket_cursors + (row * kWindows * kBuckets);
  uint32_t *row_sorted = sorted_entries + (row * kWindows * win_stride);

  #pragma unroll
  for (uint32_t w = 0u; w < kWindows; ++w) {
    const uint16_t p1 = src[w];
    if ((p1 & kValidBit) != 0u) {
      const uint32_t b = p1 & kBucketMask;
      const uint32_t neg = (p1 >> kW) & 1u;
      const uint32_t pos = atomicAdd(&row_cursors[w * kBuckets + b], 1u);
      row_sorted[w * win_stride + pos] = (term << 1u) | neg;
    }
    const uint16_t p2 = src[kWindows + w];
    if ((p2 & kValidBit) != 0u) {
      const uint32_t b = p2 & kBucketMask;
      const uint32_t neg = (p2 >> kW) & 1u;
      const uint32_t pos = atomicAdd(&row_cursors[w * kBuckets + b], 1u);
      row_sorted[w * win_stride + pos] = (term_k2 << 1u) | neg;
    }
  }
}

// Pass 4: Zero-spill, zero-branch, zero-atomic bucket accumulation (1 thread/bucket).
// Precomputed phi(P) eliminates is_k2 divergence + 655k Fp muls/row + 12 beta_sq regs,
// enabling __launch_bounds__(64, 5) = 10 warps/SM (+25% occupancy)!
template <uint32_t kBucketLog2>
__global__ static void __launch_bounds__(64, 5)
cubbs_large_pip_accumulate_buckets_kernel(
    const cubbs_sppark_plain_affine_t *points_glv,
    size_t term_count,
    size_t total_buckets,
    const uint32_t *bucket_offsets,
    const uint32_t *bucket_counts,
    const uint32_t *sorted_entries,
    cubbs_sppark_bucket_t *buckets_out) {
  const size_t gid = static_cast<size_t>(blockIdx.x) * 64u + threadIdx.x;
  if (gid >= total_buckets) return;

  const uint32_t count = bucket_counts[gid];
  cubbs_sppark_bucket_t acc;
  acc.inf();
  if (count == 0u) {
    buckets_out[gid] = acc;
    return;
  }

  const size_t rw = gid >> kBucketLog2;
  const uint32_t start = bucket_offsets[gid];
  const uint32_t *entries = sorted_entries + (rw * 2u * term_count) + start;

  #pragma unroll 1
  for (uint32_t k = 0u; k < count; ++k) {
    const uint32_t e = entries[k];
    cubbs_bbs_xyzz_madd_lazy_8m2s(acc, points_glv[e >> 1u], (e & 1u) != 0u);
  }
  cubbs_bbs_xyzz_canonicalize(acc);
  buckets_out[gid] = acc;
}

// Pass 5a: 2-level parallel reduction of kBuckets buckets per (row, win).
// Eliminates the O(kThreads) serial bottleneck at tid==0:
//   Level 1: kThreads (64 or 128) threads each reduce seg_len buckets in parallel.
//   Level 2: 8 threads each reduce (kThreads/8) segments in parallel.
//   Level 3: tid==0 combines only 8 super-segments (21 XYZZ adds instead of 381!).
template <uint32_t kBuckets, uint32_t kThreads, uint32_t kSegLog2>
__global__ static void cubbs_large_pip_reduce_buckets_kernel(
    const cubbs_sppark_bucket_t *buckets,
    cubbs_sppark_bucket_t *window_sums) {
  constexpr uint32_t kSegLen = 1u << kSegLog2;
  constexpr uint32_t kGroups = 8u;
  constexpr uint32_t kPerGroup = kThreads / kGroups;
  constexpr uint32_t kPerGroupLog2 = (kPerGroup == 16u) ? 4u : ((kPerGroup == 8u) ? 3u : 2u);
  static_assert(kThreads * kSegLen == kBuckets);
  static_assert(kPerGroup * kGroups == kThreads);

  __shared__ __align__(16) uint8_t smem[2u * kThreads * sizeof(cubbs_sppark_bucket_t)];
  __shared__ __align__(16) uint8_t smem_g[2u * kGroups * sizeof(cubbs_sppark_bucket_t)];
  auto *s_wsum = reinterpret_cast<cubbs_sppark_bucket_t *>(smem);
  auto *s_run = s_wsum + kThreads;
  auto *s_gw = reinterpret_cast<cubbs_sppark_bucket_t *>(smem_g);
  auto *s_gr = s_gw + kGroups;

  const uint32_t tid = threadIdx.x;
  const cubbs_sppark_bucket_t *win_b =
      buckets + (static_cast<size_t>(blockIdx.x) * kBuckets);

  // Level 1: each thread reduces kSegLen contiguous buckets.
  cubbs_sppark_bucket_t run, wsum;
  run.inf();
  wsum.inf();
  const uint32_t seg_base = tid * kSegLen;
  #pragma unroll 1
  for (int k = static_cast<int>(kSegLen) - 1; k >= 0; --k) {
    cubbs_bbs_limlee_xyzz_add_noinline(run, win_b[seg_base + static_cast<uint32_t>(k)]);
    cubbs_bbs_limlee_xyzz_add_noinline(wsum, run);
  }
  s_wsum[tid] = wsum;
  s_run[tid] = run;
  __syncthreads();

  // Level 2: 8 threads reduce kPerGroup segments each in parallel!
  if (tid < kGroups) {
    const uint32_t base = tid * kPerGroup;
    cubbs_sppark_bucket_t gw = s_wsum[base];
    for (uint32_t i = 1u; i < kPerGroup; ++i) {
      cubbs_bbs_limlee_xyzz_add_noinline(gw, s_wsum[base + i]);
    }
    cubbs_sppark_bucket_t gr = s_run[base];
    cubbs_sppark_bucket_t suf, tot_suf;
    suf.inf();
    tot_suf.inf();
    for (int i = static_cast<int>(kPerGroup) - 1; i >= 1; --i) {
      cubbs_bbs_limlee_xyzz_add_noinline(suf, s_run[base + static_cast<uint32_t>(i)]);
      cubbs_bbs_limlee_xyzz_add_noinline(tot_suf, suf);
    }
    cubbs_bbs_limlee_xyzz_add_noinline(gr, suf);
    #pragma unroll 1
    for (uint32_t d = 0u; d < kSegLog2; ++d) {
      cubbs_bbs_xyzz_dbl_u(tot_suf);
    }
    cubbs_bbs_limlee_xyzz_add_noinline(gw, tot_suf);
    s_gw[tid] = gw;
    s_gr[tid] = gr;
  }
  __syncthreads();

  // Level 3: tid==0 combines only 8 super-groups (shift = kSegLog2 + kPerGroupLog2).
  if (tid == 0u) {
    cubbs_sppark_bucket_t total_w = s_gw[0];
    for (uint32_t g = 1u; g < kGroups; ++g) {
      cubbs_bbs_limlee_xyzz_add_noinline(total_w, s_gw[g]);
    }
    cubbs_sppark_bucket_t suf, total_suf;
    suf.inf();
    total_suf.inf();
    for (int g = static_cast<int>(kGroups) - 1; g >= 1; --g) {
      cubbs_bbs_limlee_xyzz_add_noinline(suf, s_gr[g]);
      cubbs_bbs_limlee_xyzz_add_noinline(total_suf, suf);
    }
    #pragma unroll 1
    for (uint32_t d = 0u; d < kSegLog2 + kPerGroupLog2; ++d) {
      cubbs_bbs_xyzz_dbl_u(total_suf);
    }
    cubbs_bbs_limlee_xyzz_add_noinline(total_w, total_suf);
    window_sums[blockIdx.x] = total_w;
  }
}

// Pass 5b: Combine kWindows windows per row via Horner's rule (kW doublings/window).
template <uint32_t kW, uint32_t kWindows>
__global__ static void cubbs_large_pip_combine_windows_kernel(
    const cubbs_sppark_bucket_t *window_sums,
    size_t chunk_rows,
    cubbs_sppark_bucket_t *outputs) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= chunk_rows) return;
  const cubbs_sppark_bucket_t *ws = window_sums + (row * kWindows);
  cubbs_sppark_bucket_t acc = ws[kWindows - 1u];
  #pragma unroll 1
  for (int w = static_cast<int>(kWindows) - 2; w >= 0; --w) {
    #pragma unroll 1
    for (uint32_t d = 0u; d < kW; ++d) {
      cubbs_bbs_xyzz_dbl_u(acc);
    }
    cubbs_bbs_limlee_xyzz_add_noinline(acc, ws[w]);
  }
  outputs[row] = acc;
}

template <uint32_t kW, uint32_t kWindows, uint32_t kRedThreads, uint32_t kSegLog2>
static cudaError_t cubbs_bbs_launch_pippenger_large_glv_msm_impl(
    const cubbs_sppark_plain_affine_t *d_points,
    const uint8_t *d_scalars_le,
    size_t term_count,
    size_t output_count,
    cubbs_sppark_bucket_t *d_outputs,
    cudaStream_t stream) {
  constexpr uint32_t kBucketLog2 = kW - 1u;
  constexpr uint32_t kBuckets = 1u << kBucketLog2;

  // Keep per-stream working set L2-friendly across concurrent streams:
  // for W=13 (40,960 buckets/row = 7.86 MB/row), 1-2 rows already saturates all 36 SMs!
  const size_t target_buckets = 49152u;
  const size_t by_buckets = std::max<size_t>(1u, target_buckets / (kWindows * kBuckets));
  const size_t by_terms = std::max<size_t>(1u, 262144u / std::max<size_t>(1u, term_count));
  const size_t max_chunk_rows = std::min(by_buckets, by_terms);
  const size_t first_chunk = std::min(output_count, max_chunk_rows);

  const size_t digits_count = first_chunk * term_count * (2u * kWindows);
  const size_t buckets_total = first_chunk * kWindows * kBuckets;
  const size_t sorted_count = first_chunk * kWindows * 2u * term_count;
  const size_t win_sums_count = first_chunk * kWindows;

  auto &msm_arena = cubbs_bbs_thread_msm_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(digits_count * sizeof(uint16_t)) +
      CuBbsDeviceArena::align_up(buckets_total * sizeof(uint32_t)) * 3u +
      CuBbsDeviceArena::align_up(sorted_count * sizeof(uint32_t)) +
      CuBbsDeviceArena::align_up(buckets_total * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(win_sums_count * sizeof(cubbs_sppark_bucket_t));
  cudaError_t err = msm_arena.reserve_and_reset(arena_bytes);
  if (err != cudaSuccess) return err;

  uint16_t *d_digits = msm_arena.alloc<uint16_t>(digits_count);
  uint32_t *d_counts = msm_arena.alloc<uint32_t>(buckets_total);
  uint32_t *d_offsets = msm_arena.alloc<uint32_t>(buckets_total);
  uint32_t *d_cursors = msm_arena.alloc<uint32_t>(buckets_total);
  uint32_t *d_sorted = msm_arena.alloc<uint32_t>(sorted_count);
  cubbs_sppark_bucket_t *d_buckets = msm_arena.alloc<cubbs_sppark_bucket_t>(buckets_total);
  cubbs_sppark_bucket_t *d_win_sums = msm_arena.alloc<cubbs_sppark_bucket_t>(win_sums_count);

  for (size_t row_off = 0u; row_off < output_count; row_off += first_chunk) {
    const size_t chunk = std::min(first_chunk, output_count - row_off);
    const size_t chunk_terms = chunk * term_count;
    const size_t chunk_wins = chunk * kWindows;
    const size_t chunk_buckets = chunk_wins * kBuckets;

    err = cudaMemsetAsync(d_counts, 0, chunk_buckets * sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    constexpr uint32_t t256 = 256u;
    const uint32_t term_blocks = static_cast<uint32_t>((chunk_terms + t256 - 1u) / t256);
    cubbs_large_pip_recode_count_kernel<kW, kWindows><<<term_blocks, t256, 0, stream>>>(
        d_scalars_le + (row_off * term_count * 32u),
        term_count, chunk, d_digits, d_counts);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    cubbs_large_pip_prefix_sum_kernel<kBuckets><<<static_cast<uint32_t>(chunk_wins), 512u, 0, stream>>>(
        d_counts, d_offsets, d_cursors);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    cubbs_large_pip_scatter_kernel<kW, kWindows><<<term_blocks, t256, 0, stream>>>(
        d_digits, term_count, chunk, d_cursors, d_sorted);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const uint32_t acc_blocks = static_cast<uint32_t>((chunk_buckets + 63u) / 64u);
    cubbs_large_pip_accumulate_buckets_kernel<kBucketLog2><<<acc_blocks, 64u, 0, stream>>>(
        d_points, term_count, chunk_buckets, d_offsets, d_counts, d_sorted, d_buckets);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    cubbs_large_pip_reduce_buckets_kernel<kBuckets, kRedThreads, kSegLog2>
        <<<static_cast<uint32_t>(chunk_wins), kRedThreads, 0, stream>>>(
            d_buckets, d_win_sums);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    constexpr uint32_t c_threads = 32u;
    const uint32_t c_blocks = static_cast<uint32_t>((chunk + c_threads - 1u) / c_threads);
    cubbs_large_pip_combine_windows_kernel<kW, kWindows><<<c_blocks, c_threads, 0, stream>>>(
        d_win_sums, chunk, d_outputs + row_off);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return cudaSuccess;
}

static cudaError_t cubbs_bbs_launch_pippenger_large_glv_msm(
    const cubbs_sppark_plain_affine_t *d_points,
    const uint8_t *d_scalars_le,
    size_t term_count,
    size_t output_count,
    cubbs_sppark_bucket_t *d_outputs,
    cudaStream_t stream) {
  if (output_count == 0u || term_count == 0u) return cudaSuccess;

  // Adaptive window selection:
  //   K > 48000 (M=65,536):            W=13 (10 windows x 4096 buckets, -23% bucket adds!)
  //   K <= 48000 (U=32,768, M=4k..8k): W=10 (13 windows x  512 buckets)
  if (term_count > 48000u) {
    return cubbs_bbs_launch_pippenger_large_glv_msm_impl<13u, 10u, 64u, 6u>(
        d_points, d_scalars_le, term_count, output_count, d_outputs, stream);
  }
  return cubbs_bbs_launch_pippenger_large_glv_msm_impl<10u, 13u, 32u, 4u>(
      d_points, d_scalars_le, term_count, output_count, d_outputs, stream);
}

#endif // CUBBS_BBS_PIPPENGER_LARGE_MSM_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
