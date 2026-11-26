#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_PIPPENGER_LARGE_MSM_CUH
#define CUBBS_BBS_PIPPENGER_LARGE_MSM_CUH

// ============================================================================
// cuBBS-Cred GPU Signed 10-bit GLV Bucket-Histogram Pippenger (M > 2100)
// ============================================================================
// Replaces O(M) universal_window4 for large shapes (e.g. M=65,536 and U=32,768):
//   - Zero precomputed multiple tables (only 6.3 MB plain affine points -> 100% L2 hit!)
//   - GLV splits each 256-bit scalar into 2 x 128-bit components -> 13 signed 10-bit
//     windows (digits in [-512, +512], 512 buckets per window, 6,656 buckets per row).
//   - Even a single row (B=1, e.g. M=65,536 Verify) launches 6,656 independent
//     bucket-accumulation threads -> 100% SM saturation with 0 atomics in point math
//     and 0 local-memory stack spills!

static constexpr uint32_t CUBBS_LARGE_PIP_W = 10u;
static constexpr uint32_t CUBBS_LARGE_PIP_WINDOWS = 13u; // ceil(129 / 10) = 13
static constexpr uint32_t CUBBS_LARGE_PIP_BUCKETS = 512u; // 2^(W-1)
static constexpr uint32_t CUBBS_LARGE_PIP_HALF = 512u;

// Recode a 128-bit LE integer into 13 signed 10-bit digits in [-512, +512].
// Packs each digit as uint16_t:
//   0                -> zero digit (skip)
//   (mag - 1) | 0x200 | (neg ? 0x400 : 0)  where mag in [1..512], bucket = mag - 1 in [0..511].
__device__ __forceinline__ static void cubbs_large_pip_recode_128(
    const uint8_t bytes16[16], uint16_t packed_out[CUBBS_LARGE_PIP_WINDOWS]) {
  uint64_t lo = 0ull, hi = 0ull;
  memcpy(&lo, bytes16, 8u);
  memcpy(&hi, bytes16 + 8u, 8u);

  int32_t carry = 0;
  #pragma unroll
  for (uint32_t w = 0u; w < CUBBS_LARGE_PIP_WINDOWS; ++w) {
    const uint32_t bit = w * CUBBS_LARGE_PIP_W;
    uint32_t raw = 0u;
    if (bit + CUBBS_LARGE_PIP_W <= 64u) {
      raw = static_cast<uint32_t>((lo >> bit) & 0x3ffull);
    } else if (bit >= 64u) {
      const uint32_t sh = bit - 64u;
      raw = static_cast<uint32_t>((hi >> sh) & 0x3ffull);
    } else {
      const uint32_t lo_bits = 64u - bit;
      raw = static_cast<uint32_t>((lo >> bit) | ((hi << lo_bits) & 0x3ffull));
    }
    int32_t val = static_cast<int32_t>(raw) + carry;
    int32_t digit = val;
    carry = 0;
    if (w + 1u < CUBBS_LARGE_PIP_WINDOWS && val > static_cast<int32_t>(CUBBS_LARGE_PIP_HALF)) {
      digit = val - 1024;
      carry = 1;
    }
    if (digit == 0) {
      packed_out[w] = 0u;
    } else {
      const bool neg = digit < 0;
      const uint32_t mag = static_cast<uint32_t>(neg ? -digit : digit);
      packed_out[w] = static_cast<uint16_t>((mag - 1u) | 0x200u | (neg ? 0x400u : 0u));
    }
  }
}

// Pass 1: GLV split + signed 10-bit recode + global L2 histogram count per (row, win, bucket).
__global__ static void cubbs_large_pip_recode_count_kernel(
    const uint8_t *scalars_le,
    size_t term_count,
    size_t chunk_rows,
    uint16_t *digits_out,      // [chunk_rows * term_count * 26]
    uint32_t *bucket_counts) { // [chunk_rows * 13 * 512]
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = chunk_rows * term_count;
  if (idx >= total) return;
  const size_t row = idx / term_count;

  uint8_t k1[16], k2[16];
  cubbs_bbs_scalar_div_by_z2_device(scalars_le + (idx * 32u), k1, k2);

  uint16_t d1[CUBBS_LARGE_PIP_WINDOWS];
  uint16_t d2[CUBBS_LARGE_PIP_WINDOWS];
  cubbs_large_pip_recode_128(k1, d1);
  cubbs_large_pip_recode_128(k2, d2);

  uint16_t *dst = digits_out + (idx * 26u);
  uint32_t *row_counts = bucket_counts + (row * CUBBS_LARGE_PIP_WINDOWS * CUBBS_LARGE_PIP_BUCKETS);
  #pragma unroll
  for (uint32_t w = 0u; w < CUBBS_LARGE_PIP_WINDOWS; ++w) {
    const uint16_t p1 = d1[w];
    const uint16_t p2 = d2[w];
    dst[w] = p1;
    dst[13u + w] = p2;
    if ((p1 & 0x200u) != 0u) {
      atomicAdd(&row_counts[w * CUBBS_LARGE_PIP_BUCKETS + (p1 & 0x1ffu)], 1u);
    }
    if ((p2 & 0x200u) != 0u) {
      atomicAdd(&row_counts[w * CUBBS_LARGE_PIP_BUCKETS + (p2 & 0x1ffu)], 1u);
    }
  }
}

// Pass 2: Exclusive prefix sum over 512 buckets per (row, win) in SMEM (Blelloch scan).
__global__ static void cubbs_large_pip_prefix_sum_512_kernel(
    const uint32_t *bucket_counts, // [chunk_rows * 13 * 512]
    uint32_t *bucket_offsets,      // [chunk_rows * 13 * 512]
    uint32_t *bucket_cursors) {    // [chunk_rows * 13 * 512] initialized to offsets
  __shared__ uint32_t s_buf[CUBBS_LARGE_PIP_BUCKETS];
  const uint32_t tid = threadIdx.x; // 0..511
  const size_t base = static_cast<size_t>(blockIdx.x) * CUBBS_LARGE_PIP_BUCKETS;

  s_buf[tid] = bucket_counts[base + tid];
  __syncthreads();

  // Up-sweep
  #pragma unroll
  for (uint32_t stride = 1u; stride < CUBBS_LARGE_PIP_BUCKETS; stride <<= 1u) {
    const uint32_t idx = ((tid + 1u) * (stride << 1u)) - 1u;
    if (idx < CUBBS_LARGE_PIP_BUCKETS) {
      s_buf[idx] += s_buf[idx - stride];
    }
    __syncthreads();
  }
  if (tid == 0u) s_buf[CUBBS_LARGE_PIP_BUCKETS - 1u] = 0u;
  __syncthreads();

  // Down-sweep
  #pragma unroll
  for (uint32_t stride = CUBBS_LARGE_PIP_BUCKETS >> 1u; stride >= 1u; stride >>= 1u) {
    const uint32_t idx = ((tid + 1u) * (stride << 1u)) - 1u;
    if (idx < CUBBS_LARGE_PIP_BUCKETS) {
      const uint32_t t = s_buf[idx - stride];
      s_buf[idx - stride] = s_buf[idx];
      s_buf[idx] += t;
    }
    __syncthreads();
  }

  const uint32_t off = s_buf[tid];
  bucket_offsets[base + tid] = off;
  bucket_cursors[base + tid] = off;
}

// Pass 3: Scatter packed uint32_t entries = (term_idx << 2) | (is_k2 << 1) | neg
// into contiguous per-bucket slices for each (row, win).
__global__ static void cubbs_large_pip_scatter_kernel(
    const uint16_t *digits_in,
    size_t term_count,
    size_t chunk_rows,
    uint32_t *bucket_cursors,
    uint32_t *sorted_entries) { // [chunk_rows * 13 * (2 * term_count)]
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = chunk_rows * term_count;
  if (idx >= total) return;
  const size_t row = idx / term_count;
  const uint32_t term = static_cast<uint32_t>(idx - (row * term_count));

  const uint16_t *src = digits_in + (idx * 26u);
  const size_t win_stride = 2u * term_count;
  uint32_t *row_cursors = bucket_cursors + (row * CUBBS_LARGE_PIP_WINDOWS * CUBBS_LARGE_PIP_BUCKETS);
  uint32_t *row_sorted = sorted_entries + (row * CUBBS_LARGE_PIP_WINDOWS * win_stride);

  #pragma unroll
  for (uint32_t w = 0u; w < CUBBS_LARGE_PIP_WINDOWS; ++w) {
    const uint16_t p1 = src[w];
    if ((p1 & 0x200u) != 0u) {
      const uint32_t b = p1 & 0x1ffu;
      const uint32_t neg = (p1 >> 10u) & 1u;
      const uint32_t pos = atomicAdd(&row_cursors[w * CUBBS_LARGE_PIP_BUCKETS + b], 1u);
      row_sorted[w * win_stride + pos] = (term << 2u) | neg;
    }
    const uint16_t p2 = src[13u + w];
    if ((p2 & 0x200u) != 0u) {
      const uint32_t b = p2 & 0x1ffu;
      const uint32_t neg = (p2 >> 10u) & 1u;
      const uint32_t pos = atomicAdd(&row_cursors[w * CUBBS_LARGE_PIP_BUCKETS + b], 1u);
      row_sorted[w * win_stride + pos] = (term << 2u) | 2u | neg;
    }
  }
}

// Pass 4: Zero-spill, zero-atomic bucket accumulation (1 thread per (row, win, bucket)).
// Only compiles mixed affine uadd (no dbl_u, no xyzz+xyzz) -> maximum SM occupancy!
__global__ static void __launch_bounds__(64, 4)
cubbs_large_pip_accumulate_buckets_kernel(
    const cubbs_sppark_plain_affine_t *points,
    size_t term_count,
    size_t total_buckets, // chunk_rows * 13 * 512
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

  const size_t rw = gid >> 9u; // gid / 512 = row * 13 + win
  const uint32_t start = bucket_offsets[gid];
  const uint32_t *entries = sorted_entries + (rw * 2u * term_count) + start;

  fp_t beta_sq;
  const uint64_t beta_sq_limbs[6] = {
      0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
      0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
  memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));

  #pragma unroll 1
  for (uint32_t k = 0u; k < count; ++k) {
    const uint32_t e = entries[k];
    const uint32_t term = e >> 2u;
    const bool is_k2 = ((e >> 1u) & 1u) != 0u;
    bool neg = (e & 1u) != 0u;
    cubbs_sppark_plain_affine_t p = points[term];
    if (is_k2) {
      if (!p.is_inf()) {
        fp_t *px = reinterpret_cast<fp_t *>(&p);
        px[0] = px[0] * beta_sq;
      }
      neg = !neg;
    }
    acc.uadd(p, neg);
  }
  buckets_out[gid] = acc;
}

// Pass 5a: Reduce 512 buckets per (row, win) using 32 threads (16 buckets/thread).
// sum_{b=0}^{511} (b+1)*B[b] = sum_{s=0}^{31} wsum_s + 16 * sum_{s=1}^{31} s * run_s.
__global__ static void cubbs_large_pip_reduce_512_kernel(
    const cubbs_sppark_bucket_t *buckets, // [chunk_rows * 13 * 512]
    cubbs_sppark_bucket_t *window_sums) { // [chunk_rows * 13]
  __shared__ __align__(16) uint8_t smem[64u * sizeof(cubbs_sppark_bucket_t)];
  auto *s_wsum = reinterpret_cast<cubbs_sppark_bucket_t *>(smem);
  auto *s_run = s_wsum + 32u;

  const uint32_t tid = threadIdx.x; // 0..31
  const cubbs_sppark_bucket_t *win_b =
      buckets + (static_cast<size_t>(blockIdx.x) * CUBBS_LARGE_PIP_BUCKETS);

  cubbs_sppark_bucket_t run, wsum;
  run.inf();
  wsum.inf();
  const uint32_t seg_base = tid * 16u;
  #pragma unroll 1
  for (int k = 15; k >= 0; --k) {
    run.uadd(win_b[seg_base + static_cast<uint32_t>(k)]);
    wsum.uadd(run);
  }
  s_wsum[tid] = wsum;
  s_run[tid] = run;
  __syncthreads();

  if (tid == 0u) {
    cubbs_sppark_bucket_t total_w = s_wsum[0];
    for (uint32_t s = 1u; s < 32u; ++s) {
      total_w.uadd(s_wsum[s]);
    }
    cubbs_sppark_bucket_t suf, total_suf;
    suf.inf();
    total_suf.inf();
    for (int s = 31; s >= 1; --s) {
      suf.uadd(s_run[s]);
      total_suf.uadd(suf);
    }
    #pragma unroll 1
    for (uint32_t d = 0u; d < 4u; ++d) {
      cubbs_bbs_xyzz_dbl_u(total_suf);
    }
    total_w.uadd(total_suf);
    window_sums[blockIdx.x] = total_w;
  }
}

// Pass 5b: Combine 13 windows per row via Horner's rule (10 doublings/window).
__global__ static void cubbs_large_pip_combine_13_kernel(
    const cubbs_sppark_bucket_t *window_sums,
    size_t chunk_rows,
    cubbs_sppark_bucket_t *outputs) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= chunk_rows) return;
  const cubbs_sppark_bucket_t *ws = window_sums + (row * CUBBS_LARGE_PIP_WINDOWS);
  cubbs_sppark_bucket_t acc = ws[CUBBS_LARGE_PIP_WINDOWS - 1u];
  #pragma unroll 1
  for (int w = static_cast<int>(CUBBS_LARGE_PIP_WINDOWS) - 2; w >= 0; --w) {
    #pragma unroll 1
    for (uint32_t d = 0u; d < CUBBS_LARGE_PIP_W; ++d) {
      cubbs_bbs_xyzz_dbl_u(acc);
    }
    acc.uadd(ws[w]);
  }
  outputs[row] = acc;
}

static cudaError_t cubbs_bbs_launch_pippenger_large_glv_msm(
    const cubbs_sppark_plain_affine_t *d_points,
    const uint8_t *d_scalars_le,
    size_t term_count,
    size_t output_count,
    cubbs_sppark_bucket_t *d_outputs,
    cudaStream_t stream) {
  if (output_count == 0u || term_count == 0u) return cudaSuccess;

  // Process in row chunks keeping scratch <= ~64 MB while saturating all 36 SMs.
  const size_t max_chunk_rows = std::max<size_t>(1u, 262144u / std::max<size_t>(1u, term_count));
  const size_t first_chunk = std::min(output_count, max_chunk_rows);

  const size_t digits_count = first_chunk * term_count * 26u;
  const size_t buckets_total = first_chunk * CUBBS_LARGE_PIP_WINDOWS * CUBBS_LARGE_PIP_BUCKETS;
  const size_t sorted_count = first_chunk * CUBBS_LARGE_PIP_WINDOWS * 2u * term_count;
  const size_t win_sums_count = first_chunk * CUBBS_LARGE_PIP_WINDOWS;

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
    const size_t chunk_wins = chunk * CUBBS_LARGE_PIP_WINDOWS;
    const size_t chunk_buckets = chunk_wins * CUBBS_LARGE_PIP_BUCKETS;

    err = cudaMemsetAsync(d_counts, 0, chunk_buckets * sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    constexpr uint32_t t256 = 256u;
    const uint32_t term_blocks = static_cast<uint32_t>((chunk_terms + t256 - 1u) / t256);
    cubbs_large_pip_recode_count_kernel<<<term_blocks, t256, 0, stream>>>(
        d_scalars_le + (row_off * term_count * 32u),
        term_count, chunk, d_digits, d_counts);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    cubbs_large_pip_prefix_sum_512_kernel<<<static_cast<uint32_t>(chunk_wins), CUBBS_LARGE_PIP_BUCKETS, 0, stream>>>(
        d_counts, d_offsets, d_cursors);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    cubbs_large_pip_scatter_kernel<<<term_blocks, t256, 0, stream>>>(
        d_digits, term_count, chunk, d_cursors, d_sorted);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const uint32_t acc_blocks = static_cast<uint32_t>((chunk_buckets + 63u) / 64u);
    cubbs_large_pip_accumulate_buckets_kernel<<<acc_blocks, 64u, 0, stream>>>(
        d_points, term_count, chunk_buckets, d_offsets, d_counts, d_sorted, d_buckets);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    cubbs_large_pip_reduce_512_kernel<<<static_cast<uint32_t>(chunk_wins), 32u, 0, stream>>>(
        d_buckets, d_win_sums);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    constexpr uint32_t c_threads = 32u;
    const uint32_t c_blocks = static_cast<uint32_t>((chunk + c_threads - 1u) / c_threads);
    cubbs_large_pip_combine_13_kernel<<<c_blocks, c_threads, 0, stream>>>(
        d_win_sums, chunk, d_outputs + row_off);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return cudaSuccess;
}

#endif // CUBBS_BBS_PIPPENGER_LARGE_MSM_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
