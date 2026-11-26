#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_FUSED_SIGN_VERIFY_CUH
#define CUBBS_BBS_FUSED_SIGN_VERIFY_CUH

// ============================================================================
// GPU Montgomery Batch Inversion + BLS12-381 G1 Compression (xyzz -> 48B BE)
// ============================================================================

__device__ __forceinline__ static void cubbs_bbs_compress_affine_fp_to_be48(
    fp_t affine_x, fp_t affine_y, bool finite, uint8_t *out48) {
  if (!finite) {
    out48[0] = 0xc0u;
#pragma unroll
    for (int i = 1; i < 48; ++i) out48[i] = 0u;
    return;
  }
  affine_x.from();
  affine_y.from();

  uint64_t x_limbs[6];
  uint64_t y_limbs[6];
  memcpy(x_limbs, &affine_x, 48u);
  memcpy(y_limbs, &affine_y, 48u);

  // (p - 1) / 2 in little-endian 64-bit limbs for BLS12-381 base field Fp.
  static constexpr uint64_t FP_HALF_LE[6] = {
      0xdcff7fffffffd555ull, 0x0f55ffff58a9ffffull,
      0xb39869507b587b12ull, 0xb23ba5c279c2895full,
      0x258dd3db21a5d66bull, 0x0d0088f51cbff34dull};
  bool y_gt_half = false;
#pragma unroll
  for (int limb = 5; limb >= 0; --limb) {
    if (y_limbs[limb] != FP_HALF_LE[limb]) {
      y_gt_half = y_limbs[limb] > FP_HALF_LE[limb];
      break;
    }
  }

#pragma unroll
  for (int limb = 0; limb < 6; ++limb) {
    const uint64_t v = x_limbs[5 - limb];
    const int base = limb * 8;
    out48[base + 0] = static_cast<uint8_t>(v >> 56);
    out48[base + 1] = static_cast<uint8_t>(v >> 48);
    out48[base + 2] = static_cast<uint8_t>(v >> 40);
    out48[base + 3] = static_cast<uint8_t>(v >> 32);
    out48[base + 4] = static_cast<uint8_t>(v >> 24);
    out48[base + 5] = static_cast<uint8_t>(v >> 16);
    out48[base + 6] = static_cast<uint8_t>(v >> 8);
    out48[base + 7] = static_cast<uint8_t>(v);
  }
  out48[0] |= static_cast<uint8_t>(0x80u | (y_gt_half ? 0x20u : 0x00u));
}

// Converts xyzz_t points directly to 48-byte compressed BLS12-381 G1 bytes on GPU
// using 256-way Montgomery batch inversion with SPPARK's warp-cooperative Euclidean
// reciprocal(). Eliminates host BLST compression and shrinks D2H by 4x.
__global__ static void cubbs_bbs_g1_xyzz_to_compressed_stride_kernel(
    const cubbs_sppark_bucket_t *input, size_t count, uint8_t *out,
    size_t out_stride, size_t out_offset) {
  constexpr uint32_t kThreads = 64u;
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
  // xyzz_t layout: [0]=X, [1]=Y, [2]=ZZZ, [3]=ZZ.
  const fp_t zzz = finite ? coordinates[2] : one;

  scan_b[tid] = zzz;
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
  // All 32 lanes of warp 0 participate in fp_t::reciprocal() so shfl_xor(mask=0xffffffff)
  // is strictly uniform across the warp.
  if (tid < 32u) {
    const fp_t val = (tid == 0u) ? *total_product : one;
    fp_t inverse = val.reciprocal();
    if (tid == 0u) {
      for (int item = static_cast<int>(kThreads) - 1; item >= 0; --item) {
        const fp_t inverse_item = inverse * scan_a[item];
        inverse *= scan_b[item];
        scan_a[item] = inverse_item;
      }
    }
  }
  __syncthreads();

  if (valid) {
    fp_t affine_x;
    fp_t affine_y;
    if (finite) {
      const fp_t inverse_zzz = scan_a[tid];
      affine_y = inverse_zzz * coordinates[1];
      const fp_t inverse_z = inverse_zzz * coordinates[3];
      affine_x = inverse_z * inverse_z;
      affine_x *= coordinates[0];
    }
    cubbs_bbs_compress_affine_fp_to_be48(
        affine_x, affine_y, finite, out + (index * out_stride) + out_offset);
  }
}

static cudaError_t cubbs_bbs_launch_xyzz_to_compressed(
    const cubbs_sppark_bucket_t *device_outputs, size_t count,
    uint8_t *device_compressed_out, size_t out_stride, size_t out_offset,
    cudaStream_t stream) {
  if (count == 0u) return cudaSuccess;
  constexpr uint32_t kThreads = 64u;
  const uint32_t blocks = static_cast<uint32_t>((count + kThreads - 1u) / kThreads);
  cubbs_bbs_g1_xyzz_to_compressed_stride_kernel<<<blocks, kThreads, 0, stream>>>(
      device_outputs, count, device_compressed_out, out_stride, out_offset);
  return cudaGetLastError();
}

// ============================================================================
// Shared Cached Common-Points MSM Kernel Launcher
// ============================================================================

struct CuBbsCommonMsmTransient {
  int8_t *large_wnaf_digits{nullptr};
  cubbs_sppark_bucket_t *large_partials{nullptr};
  cubbs_sppark_bucket_t *large_bucket_partials{nullptr};
  cubbs_sppark_bucket_t *large_window_partials{nullptr};
  size_t large_tile_terms{0u};
  size_t large_tile_count{0u};
  bool large_glv{false};
  bool large_glv_wnaf{false};
  bool large_pippenger{false};
  bool large_window8{false};
  bool large_window4{false};

  cudaError_t alloc(const CuBbsFusedCommonPointsCache &cached,
                    size_t term_count, size_t output_count, bool is_large,
                    cudaStream_t stream) {
    if (cached.device_limlee_affines != nullptr || cached.use_pippenger_large) {
      large_window4 = false;
      large_glv = false;
      large_glv_wnaf = false;
      large_pippenger = false;
      large_window8 = false;
      return cudaSuccess;
    }
    large_window4 = is_large;
    large_glv = is_large && cached.device_point_glv_multiples != nullptr &&
                !bbs_large_shape_pippenger_enabled() &&
                !bbs_large_shape_window8_enabled();
    large_glv_wnaf = large_glv && cached.device_point_glv_wnaf_multiples != nullptr;
    large_tile_terms = bbs_large_shape_tile_terms();
    large_tile_count = is_large
        ? (term_count + large_tile_terms - 1u) / large_tile_terms
        : 0u;
    large_pippenger = is_large && bbs_large_shape_pippenger_enabled();
    large_window8 = is_large && !large_pippenger &&
                    cached.device_point_multiples8 != nullptr;

    cudaError_t err = cudaSuccess;
    if (large_glv_wnaf) {
      err = cudaMallocAsync(&large_wnaf_digits, output_count * term_count * 260u, stream);
    }
    if (err == cudaSuccess && is_large && !large_pippenger && !large_window8) {
      err = cudaMallocAsync(
          &large_partials,
          output_count * large_tile_count * sizeof(*large_partials), stream);
    }
    if (err == cudaSuccess && large_pippenger) {
      err = cudaMallocAsync(
          &large_bucket_partials,
          output_count * 64u * large_tile_count * 15u *
              sizeof(*large_bucket_partials),
          stream);
    }
    if (err == cudaSuccess && large_pippenger) {
      err = cudaMallocAsync(
          &large_window_partials,
          output_count * 64u * sizeof(*large_window_partials), stream);
    }
    return err;
  }

  void free_async(cudaStream_t stream) {
    if (large_wnaf_digits != nullptr) (void)cudaFreeAsync(large_wnaf_digits, stream);
    if (large_partials != nullptr) (void)cudaFreeAsync(large_partials, stream);
    if (large_bucket_partials != nullptr) (void)cudaFreeAsync(large_bucket_partials, stream);
    if (large_window_partials != nullptr) (void)cudaFreeAsync(large_window_partials, stream);
    large_wnaf_digits = nullptr;
    large_partials = nullptr;
    large_bucket_partials = nullptr;
    large_window_partials = nullptr;
  }
};

static cudaError_t cubbs_bbs_launch_cached_common_msm_range(
    const CuBbsFusedCommonPointsCache &cached,
    CuBbsCommonMsmTransient &transient,
    const uint8_t *device_scalars_le,
    size_t term_count,
    size_t output_count,
    cubbs_sppark_bucket_t *device_outputs,
    cudaStream_t stream) {
  if (output_count == 0u) return cudaSuccess;
  if (transient.large_glv_wnaf) {
    constexpr uint32_t recode_threads = 128u;
    const size_t total_scalars = output_count * term_count;
    const uint32_t recode_blocks = static_cast<uint32_t>(
        (total_scalars + recode_threads - 1u) / recode_threads);
    cubbs_bbs_glv_wnaf8_recode_kernel<<<recode_blocks, recode_threads, 0, stream>>>(
        device_scalars_le, total_scalars, transient.large_wnaf_digits);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  const uint32_t threads = fused_common_points_adaptive_threads(term_count, output_count);
  const uint32_t blocks = static_cast<uint32_t>(output_count);
  if (cached.use_pippenger_large) {
    return cubbs_bbs_launch_pippenger_large_glv_msm(
        cached.device_points, device_scalars_le, term_count, output_count,
        device_outputs, stream);
  }
  if (cached.device_limlee_affines != nullptr) {
    return cubbs_bbs_launch_glv_limlee(
        cached.device_limlee_affines, cached.device_limlee_offsets,
        cached.device_limlee_c_total, device_scalars_le, term_count,
        output_count, cached.limlee_num_groups, cached.limlee_group_size,
        cached.limlee_v_sub, stream, device_outputs);
  }
  if (term_count >= 18u && cached.device_fixed_glv_affine_comb9 != nullptr) {
    return cubbs_bbs_launch_glv_affine_comb<9u>(
        cached.device_fixed_glv_affine_comb9, device_scalars_le, term_count,
        output_count, stream, device_outputs);
  }
  if (term_count >= 18u && cached.device_fixed_glv_affine_comb10 != nullptr) {
    return cubbs_bbs_launch_glv_affine_comb<10u>(
        cached.device_fixed_glv_affine_comb10, device_scalars_le, term_count,
        output_count, stream, device_outputs);
  }
  if (term_count >= 18u && cached.device_fixed_glv_affine_comb8 != nullptr) {
    return cubbs_bbs_launch_glv_affine_comb<8u>(
        cached.device_fixed_glv_affine_comb8, device_scalars_le, term_count,
        output_count, stream, device_outputs);
  }
  if (term_count >= 18u && cached.device_fixed_affine_comb9 != nullptr) {
    return cubbs_bbs_launch_affine_comb<9u>(
        cached.device_fixed_affine_comb9, device_scalars_le, term_count,
        output_count, threads, stream, device_outputs);
  }
  if (term_count >= 18u && cached.device_fixed_affine_comb10 != nullptr) {
    return cubbs_bbs_launch_affine_comb<10u>(
        cached.device_fixed_affine_comb10, device_scalars_le, term_count,
        output_count, threads, stream, device_outputs);
  }
  if (term_count >= 18u && cached.device_fixed_affine_comb8 != nullptr) {
    return cubbs_bbs_launch_affine_comb<8u>(
        cached.device_fixed_affine_comb8, device_scalars_le, term_count,
        output_count, threads, stream, device_outputs);
  }
  if (transient.large_glv) {
    constexpr uint32_t glv_threads = 128u;
    const dim3 tiled_grid(blocks, static_cast<uint32_t>(transient.large_tile_count));
    if (transient.large_glv_wnaf) {
      cubbs_bbs_g1_msm_large_glv_wnaf8_tiled_kernel<glv_threads>
          <<<tiled_grid, glv_threads, 0, stream>>>(
              cached.device_point_glv_wnaf_multiples,
              transient.large_wnaf_digits, term_count,
              transient.large_tile_terms, transient.large_tile_count,
              output_count, transient.large_partials);
    } else {
      cubbs_bbs_g1_msm_large_glv_window4_tiled_kernel<glv_threads>
          <<<tiled_grid, glv_threads, 0, stream>>>(
              cached.device_point_glv_multiples, device_scalars_le, term_count,
              transient.large_tile_terms, transient.large_tile_count,
              output_count, transient.large_partials);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    constexpr uint32_t reduce_threads = 32u;
    cubbs_bbs_g1_msm_large_reduce_kernel<reduce_threads>
        <<<blocks, reduce_threads, 0, stream>>>(
            transient.large_partials, transient.large_tile_count,
            output_count, device_outputs);
    return cudaGetLastError();
  }
  if (transient.large_window8) {
    constexpr uint32_t large_threads = 256u;
    cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<large_threads>
        <<<blocks, large_threads, 0, stream>>>(
            cached.device_point_multiples8, device_scalars_le, term_count,
            output_count, device_outputs);
    return cudaGetLastError();
  }
  if (transient.large_pippenger) {
    constexpr uint32_t bucket_threads = 128u;
    const dim3 bucket_grid(blocks, 64u, static_cast<uint32_t>(transient.large_tile_count));
    cubbs_bbs_g1_msm_large_pippenger_locked_buckets_kernel<bucket_threads>
        <<<bucket_grid, bucket_threads, 0, stream>>>(
            cached.device_points, device_scalars_le, term_count,
            transient.large_tile_terms, transient.large_tile_count,
            output_count, transient.large_bucket_partials);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    constexpr uint32_t window_threads = 32u;
    const dim3 window_grid(blocks, 64u);
    cubbs_bbs_g1_msm_large_pippenger_windows_kernel<window_threads>
        <<<window_grid, window_threads, 0, stream>>>(
            transient.large_bucket_partials, transient.large_tile_count,
            output_count, transient.large_window_partials);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    cubbs_bbs_g1_msm_large_pippenger_rows_kernel<<<blocks, 1u, 0, stream>>>(
        transient.large_window_partials, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (transient.large_window4) {
    constexpr uint32_t tile_threads = 128u;
    const dim3 tiled_grid(blocks, static_cast<uint32_t>(transient.large_tile_count));
    cubbs_bbs_g1_msm_large_window4_tiled_kernel<tile_threads>
        <<<tiled_grid, tile_threads, 0, stream>>>(
            cached.device_point_multiples, device_scalars_le, term_count,
            transient.large_tile_terms, transient.large_tile_count,
            output_count, transient.large_partials);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    constexpr uint32_t reduce_threads = 32u;
    cubbs_bbs_g1_msm_large_reduce_kernel<reduce_threads>
        <<<blocks, reduce_threads, 0, stream>>>(
            transient.large_partials, transient.large_tile_count,
            output_count, device_outputs);
    return cudaGetLastError();
  }
  if (cached.device_fixed_comb9 != nullptr) {
    if (threads == 32u)
      cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<32u><<<blocks, 32u, 0, stream>>>(
          cached.device_fixed_comb9, device_scalars_le, term_count, output_count, device_outputs);
    else if (threads == 64u)
      cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<64u><<<blocks, 64u, 0, stream>>>(
          cached.device_fixed_comb9, device_scalars_le, term_count, output_count, device_outputs);
    else
      cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<128u><<<blocks, 128u, 0, stream>>>(
          cached.device_fixed_comb9, device_scalars_le, term_count, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (cached.device_fixed_comb10 != nullptr) {
    if (threads == 32u)
      cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<32u><<<blocks, 32u, 0, stream>>>(
          cached.device_fixed_comb10, device_scalars_le, term_count, output_count, device_outputs);
    else if (threads == 64u)
      cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<64u><<<blocks, 64u, 0, stream>>>(
          cached.device_fixed_comb10, device_scalars_le, term_count, output_count, device_outputs);
    else
      cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<128u><<<blocks, 128u, 0, stream>>>(
          cached.device_fixed_comb10, device_scalars_le, term_count, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (cached.device_fixed_comb8 != nullptr) {
    if (threads == 32u)
      cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<32u><<<blocks, 32u, 0, stream>>>(
          cached.device_fixed_comb8, device_scalars_le, term_count, output_count, device_outputs);
    else if (threads == 64u)
      cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<64u><<<blocks, 64u, 0, stream>>>(
          cached.device_fixed_comb8, device_scalars_le, term_count, output_count, device_outputs);
    else
      cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<128u><<<blocks, 128u, 0, stream>>>(
          cached.device_fixed_comb8, device_scalars_le, term_count, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (cached.device_fixed_comb6 != nullptr) {
    if (threads == 32u)
      cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<32u><<<blocks, 32u, 0, stream>>>(
          cached.device_fixed_comb6, device_scalars_le, term_count, output_count, device_outputs);
    else if (threads == 64u)
      cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<64u><<<blocks, 64u, 0, stream>>>(
          cached.device_fixed_comb6, device_scalars_le, term_count, output_count, device_outputs);
    else
      cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<128u><<<blocks, 128u, 0, stream>>>(
          cached.device_fixed_comb6, device_scalars_le, term_count, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (cached.device_fixed_comb5 != nullptr) {
    if (threads == 32u)
      cubbs_bbs_g1_msm_fixed_signed_comb5_kernel<32u><<<blocks, 32u, 0, stream>>>(
          cached.device_fixed_comb5, device_scalars_le, term_count, output_count, device_outputs);
    else if (threads == 64u)
      cubbs_bbs_g1_msm_fixed_signed_comb5_kernel<64u><<<blocks, 64u, 0, stream>>>(
          cached.device_fixed_comb5, device_scalars_le, term_count, output_count, device_outputs);
    else
      cubbs_bbs_g1_msm_fixed_signed_comb5_kernel<128u><<<blocks, 128u, 0, stream>>>(
          cached.device_fixed_comb5, device_scalars_le, term_count, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (cached.device_fixed_comb4 != nullptr) {
    if (threads == 32u)
      cubbs_bbs_g1_msm_fixed_comb4_kernel<32u><<<blocks, 32u, 0, stream>>>(
          cached.device_fixed_comb4, device_scalars_le, term_count, output_count, device_outputs);
    else if (threads == 64u)
      cubbs_bbs_g1_msm_fixed_comb4_kernel<64u><<<blocks, 64u, 0, stream>>>(
          cached.device_fixed_comb4, device_scalars_le, term_count, output_count, device_outputs);
    else
      cubbs_bbs_g1_msm_fixed_comb4_kernel<128u><<<blocks, 128u, 0, stream>>>(
          cached.device_fixed_comb4, device_scalars_le, term_count, output_count, device_outputs);
    return cudaGetLastError();
  }
  if (threads == 32u)
    cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<32u><<<blocks, 32u, 0, stream>>>(
        cached.device_point_multiples8, device_scalars_le, term_count, output_count, device_outputs);
  else if (threads == 64u)
    cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<64u><<<blocks, 64u, 0, stream>>>(
        cached.device_point_multiples8, device_scalars_le, term_count, output_count, device_outputs);
  else if (threads == 256u)
    cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<256u><<<blocks, 256u, 0, stream>>>(
        cached.device_point_multiples8, device_scalars_le, term_count, output_count, device_outputs);
  else
    cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<128u><<<blocks, 128u, 0, stream>>>(
        cached.device_point_multiples8, device_scalars_le, term_count, output_count, device_outputs);
  return cudaGetLastError();
}

// ============================================================================
// Thread-local Pinned Staging Workspace for Fused Sign & Verify
// ============================================================================

struct CuBbsFusedPipelinePinnedWorkspace {
  uint8_t *messages{nullptr};
  size_t messages_cap{0u};
  uint64_t *offsets{nullptr};
  uint64_t *lengths{nullptr};
  size_t msg_meta_cap{0u};
  uint8_t *aux_in{nullptr};
  size_t aux_in_cap{0u};
  uint8_t *aux_out{nullptr};
  size_t aux_out_cap{0u};
  uint32_t *invalid_flag{nullptr};

  // Cached G2 lines for Verify pairing (avoids re-copying 19.5KB per batch).
  uint8_t pk_key[96]{};
  uint8_t g2_key[96]{};
  bool pk_valid{false};
  bool g2_valid{false};
  uint8_t pk_lines[68u * 6u * sizeof(fp_t)]{};
  uint8_t g2_lines[68u * 6u * sizeof(fp_t)]{};

  cudaError_t ensure_messages(size_t bytes) {
    const size_t req = std::max<size_t>(1u, bytes);
    if (messages_cap >= req) return cudaSuccess;
    if (messages != nullptr) (void)cudaFreeHost(messages);
    messages = nullptr;
    messages_cap = 0u;
    const size_t alloc = std::max<size_t>(req, 4096u);
    cudaError_t err = cudaMallocHost(&messages, alloc);
    if (err == cudaSuccess) messages_cap = alloc;
    return err;
  }

  cudaError_t ensure_msg_meta(size_t count) {
    const size_t req = std::max<size_t>(1u, count);
    if (msg_meta_cap >= req) return cudaSuccess;
    if (offsets != nullptr) (void)cudaFreeHost(offsets);
    if (lengths != nullptr) (void)cudaFreeHost(lengths);
    offsets = nullptr;
    lengths = nullptr;
    msg_meta_cap = 0u;
    cudaError_t err = cudaMallocHost(&offsets, req * sizeof(uint64_t));
    if (err == cudaSuccess)
      err = cudaMallocHost(&lengths, req * sizeof(uint64_t));
    if (err == cudaSuccess) msg_meta_cap = req;
    return err;
  }

  cudaError_t ensure_aux(size_t in_bytes, size_t out_bytes) {
    cudaError_t err = cudaSuccess;
    const size_t req_in = std::max<size_t>(1u, in_bytes);
    const size_t req_out = std::max<size_t>(1u, out_bytes);
    if (aux_in_cap < req_in) {
      if (aux_in != nullptr) (void)cudaFreeHost(aux_in);
      aux_in = nullptr;
      aux_in_cap = 0u;
      err = cudaMallocHost(&aux_in, req_in);
      if (err == cudaSuccess) aux_in_cap = req_in;
    }
    if (err == cudaSuccess && aux_out_cap < req_out) {
      if (aux_out != nullptr) (void)cudaFreeHost(aux_out);
      aux_out = nullptr;
      aux_out_cap = 0u;
      err = cudaMallocHost(&aux_out, req_out);
      if (err == cudaSuccess) aux_out_cap = req_out;
    }
    if (err == cudaSuccess && invalid_flag == nullptr) {
      err = cudaMallocHost(&invalid_flag, sizeof(uint32_t));
    }
    return err;
  }

  ~CuBbsFusedPipelinePinnedWorkspace() {
    if (messages != nullptr) (void)cudaFreeHost(messages);
    if (offsets != nullptr) (void)cudaFreeHost(offsets);
    if (lengths != nullptr) (void)cudaFreeHost(lengths);
    if (aux_in != nullptr) (void)cudaFreeHost(aux_in);
    if (aux_out != nullptr) (void)cudaFreeHost(aux_out);
    if (invalid_flag != nullptr) (void)cudaFreeHost(invalid_flag);
  }
};

static CuBbsFusedPipelinePinnedWorkspace &cubbs_fused_pipeline_pinned_workspace() {
  static thread_local CuBbsFusedPipelinePinnedWorkspace ws;
  return ws;
}

int32_t cubbs_bbs_cuda_pipeline_pinned_scratch(
    size_t msg_bytes, size_t msg_count, uint8_t **messages_out,
    uint64_t **offsets_out, uint64_t **lengths_out) {
  if (messages_out == nullptr || offsets_out == nullptr || lengths_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  cubbs_bbs_ensure_cuda_device_schedule();
  auto &ws = cubbs_fused_pipeline_pinned_workspace();
  cudaError_t err = ws.ensure_messages(msg_bytes);
  if (err == cudaSuccess && msg_count > 0u) {
    err = ws.ensure_msg_meta(msg_count);
  }
  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
  *messages_out = ws.messages;
  *offsets_out = ws.offsets;
  *lengths_out = ws.lengths;
  return CUBBS_BBS_GPU_SUCCESS;
}

// ============================================================================
// GPU-Resident Fused BBS Sign Kernels
// ============================================================================

// Hash every message directly into LE MSM scalar layout at slot (2 + msg_idx) of each row.
// Supports 3 metadata modes:
//   - offsets == nullptr && lengths == nullptr: uniform message length `uniform_len` (0B metadata!)
//   - offsets != nullptr && lengths == nullptr: compact uint32_t prefix offsets[total + 1] (4B/msg)
//   - offsets != nullptr && lengths != nullptr: legacy uint64_t offsets[total] + lengths[total]
__global__ static void cubbs_bbs_sign_h2s_messages_to_le_kernel(
    const uint8_t *messages, const uint64_t *offsets, const uint64_t *lengths,
    size_t uniform_len, size_t item_count, size_t message_count, size_t term_count,
    const uint8_t *message_dst, size_t message_dst_len, uint8_t *scalars_le) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = item_count * message_count;
  if (idx >= total) return;

  size_t msg_off = 0u;
  size_t msg_len = uniform_len;
  if (lengths != nullptr) {
    msg_off = static_cast<size_t>(offsets[idx]);
    msg_len = static_cast<size_t>(lengths[idx]);
  } else if (offsets != nullptr) {
    const auto *offs32 = reinterpret_cast<const uint32_t *>(offsets);
    msg_off = static_cast<size_t>(offs32[idx]);
    msg_len = static_cast<size_t>(offs32[idx + 1u] - offs32[idx]);
  } else {
    msg_off = idx * uniform_len;
  }

  const size_t row = idx / message_count;
  const size_t msg = idx - (row * message_count);
  uint32_t uniform_words[12];
  expand_message_xmd_sha256_48_words_device(
      messages + msg_off, msg_len,
      message_dst, message_dst_len, uniform_words);
  uint64_t le[4];
  bls12_381_fr_reduce_words12_to_le64(uniform_words, le);
  bls12_381_fr_store_le(le, scalars_le + ((row * term_count + 2u + msg) * 32u));
}

// Computes e = H2S(sk || m_0_be || ... || m_{M-1}_be || domain_be, e_dst) reading
// message scalars directly from scalars_le (zero intermediate e-input buffer!),
// stores e_be into signatures_out[row*80 + 48..80], populates slot 0 (1) and slot 1 (domain_le),
// and computes row_scales_le[row] = (sk + e)^-1 using warp-cooperative fr_t::reciprocal().
__global__ static void cubbs_bbs_sign_e_and_inv_kernel(
    const uint8_t *secret_keys_be, size_t secret_key_stride,
    const uint8_t *domains_be, size_t domain_stride,
    size_t item_count, size_t message_count, size_t term_count,
    const uint8_t *e_dst, size_t e_dst_len,
    uint8_t *scalars_le, uint8_t *row_scales_le,
    uint8_t *signatures_out, uint32_t *invalid_out) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = row < item_count;

  cubbs_sppark_scalar_t denom = cubbs_sppark_scalar_t::one();
  bool row_valid = active;

  if (active) {
    const uint8_t *sk_be = secret_keys_be + (row * secret_key_stride);
    const uint8_t *dom_be = domains_be + (row * domain_stride);

    // Validate secret_key is canonical and nonzero, and domain is canonical.
    bool sk_nonzero = false;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      if (sk_be[i] != 0u) sk_nonzero = true;
    }
    const bool sk_ok = sk_nonzero && cubbs_bbs_scalar_be_is_canonical(sk_be);
    const bool dom_ok = cubbs_bbs_scalar_be_is_canonical(dom_be);
    if (!sk_ok || !dom_ok) {
      row_valid = false;
      atomicExch(invalid_out, 1u);
    }

    uint8_t *row_scalars = scalars_le + (row * term_count * 32u);

    // Compute e = H2S(sk_be || msg_0_be || ... || msg_{M-1}_be || dom_be, e_dst).
    CuBbsSha256Ctx ctx;
    uint32_t b0[8], b1[8], b2[8];
    ctx.init_after_zpad();
    ctx.update(sk_be, 32u);
    ctx.update_scalars_le_as_be_bulk(row_scalars + 64u, message_count);
    ctx.update(dom_be, 32u);
    ctx.push_byte(0u);
    ctx.push_byte(48u);
    ctx.push_byte(0u);
    ctx.update(e_dst, e_dst_len);
    ctx.push_byte(static_cast<uint8_t>(e_dst_len));
    ctx.finalize_words(b0);

    ctx.init();
    ctx.update_words_be(b0, 8u);
    ctx.push_byte(1u);
    ctx.update(e_dst, e_dst_len);
    ctx.push_byte(static_cast<uint8_t>(e_dst_len));
    ctx.finalize_words(b1);

    uint32_t b0_xor_b1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) b0_xor_b1[i] = b0[i] ^ b1[i];
    ctx.init();
    ctx.update_words_be(b0_xor_b1, 8u);
    ctx.push_byte(2u);
    ctx.update(e_dst, e_dst_len);
    ctx.push_byte(static_cast<uint8_t>(e_dst_len));
    ctx.finalize_words(b2);

    uint32_t uniform_words[12];
#pragma unroll
    for (int i = 0; i < 8; ++i) uniform_words[i] = b1[i];
#pragma unroll
    for (int i = 0; i < 4; ++i) uniform_words[8 + i] = b2[i];

    uint64_t e_le[4];
    bls12_381_fr_reduce_words12_to_le64(uniform_words, e_le);
    bls12_381_fr_store_be(e_le, signatures_out + (row * 80u) + 48u);

    // Populate slot 0 = 1 (LE) and slot 1 = domain (LE).
    row_scalars[0] = 1u;
#pragma unroll
    for (int i = 1; i < 32; ++i) row_scalars[i] = 0u;
    uint8_t *dom_slot = row_scalars + 32u;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      dom_slot[i] = dom_ok ? dom_be[31 - i] : 0u;
    }

    if (row_valid) {
      uint8_t sk_le[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) sk_le[i] = sk_be[31 - i];
      cubbs_sppark_scalar_t fr_sk;
      cubbs_sppark_scalar_t fr_e;
      memcpy(&fr_sk, sk_le, 32u);
      memcpy(&fr_e, e_le, 32u);
      fr_sk.to();
      fr_e.to();
      denom = fr_sk + fr_e;
      if (denom.is_zero()) {
        row_valid = false;
        denom = cubbs_sppark_scalar_t::one();
        atomicExch(invalid_out, 1u);
      }
    }
  }

  // All threads in the warp execute reciprocal() together (required by shfl_xor).
  cubbs_sppark_scalar_t inv = denom.reciprocal();
  if (active) {
    inv.from();
    if (!row_valid) inv.zero();
    memcpy(row_scales_le + (row * 32u), &inv, 32u);
  }
}

int32_t cubbs_bbs_cuda_sign_fused_common_batch(
    const CuBbsBbsGpuSignFusedCommonBatchItem *item,
    CuBbsBbsGpuResult *results) {
  if (item == nullptr) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  const size_t item_count = item->item_count;
  const size_t term_count = item->common_term_count;
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  if (term_count < 2u || item->common_points == nullptr ||
      item->secret_keys_be == nullptr || item->domains_be == nullptr ||
      (item->secret_key_stride != 0u && item->secret_key_stride != 32u) ||
      (item->domain_stride != 0u && item->domain_stride != 32u) ||
      item->message_dst.len > 255u || item->e_dst.len > 255u ||
      (item->message_dst.len > 0u && item->message_dst.data == nullptr) ||
      (item->e_dst.len > 0u && item->e_dst.data == nullptr) ||
      item->signatures_out == nullptr || results == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t message_count = term_count - 2u;
  const size_t total_messages = item_count * message_count;
  const bool meta_uniform =
      total_messages > 0u && item->message_offsets == nullptr && item->message_lengths == nullptr;
  const bool meta_u32 =
      total_messages > 0u && item->message_offsets != nullptr && item->message_lengths == nullptr;
  const bool meta_u64 =
      total_messages > 0u && item->message_offsets != nullptr && item->message_lengths != nullptr;
  if (total_messages > 0u &&
      ((!meta_uniform && !meta_u32 && !meta_u64) ||
       (meta_uniform && (item->packed_messages_len % total_messages) != 0u) ||
       (meta_u32 && item->packed_messages_len > 0xFFFFFFFFull) ||
       (item->packed_messages_len > 0u && item->packed_messages == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t uniform_msg_len = meta_uniform ? (item->packed_messages_len / total_messages) : 0u;

  cubbs_bbs_ensure_cuda_device_schedule();

  // Select the exact same cached common-points route as msm_dispatch.cuh.
  bool large_window4 = false;
  bool universal_window4 = false;
  if (bbs_homogeneous_kernel_enabled() &&
      bbs_specialized_common_points_supported(term_count) &&
      item_count >= bbs_homogeneous_kernel_min_items(term_count)) {
    large_window4 = false;
  } else if (bbs_large_shape_kernel_supported(term_count)) {
    large_window4 = true;
  } else if (bbs_universal_tiled_common_supported(term_count)) {
    large_window4 = true;
    universal_window4 = true;
  } else {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  const bool affine_common =
      !large_window4 && term_count >= 18u && item_count >= 16u &&
      bbs_homogeneous_affine_comb_enabled();
  int32_t cache_status = get_fused_common_points_cache(
      item->common_points, term_count, &cached, large_window4, affine_common,
      0u, universal_window4);
  if (cache_status != CUBBS_BBS_GPU_SUCCESS) {
    return cache_status;
  }

  auto &host_ws = cubbs_fused_pinned_host_workspace();
  auto &pipe_ws = cubbs_fused_pipeline_pinned_workspace();
  const cudaStream_t stream = host_ws.execution_stream(
      cached->stream != nullptr ? cached->stream : static_cast<cudaStream_t>(0));
  configure_cubbs_async_mempool();

  const size_t sk_bytes = item->secret_key_stride == 0u ? 32u : item_count * 32u;
  const size_t dom_bytes = item->domain_stride == 0u ? 32u : item_count * 32u;
  const size_t aux_in_bytes = sk_bytes + dom_bytes + item->message_dst.len + item->e_dst.len;
  const size_t sig_bytes = item_count * 80u;

  cudaError_t err = pipe_ws.ensure_messages(item->packed_messages_len);
  if (err == cudaSuccess && meta_u64) {
    err = pipe_ws.ensure_msg_meta(total_messages);
  } else if (err == cudaSuccess && meta_u32) {
    err = pipe_ws.ensure_msg_meta((total_messages + 2u) / 2u);
  }
  if (err == cudaSuccess) {
    err = pipe_ws.ensure_aux(aux_in_bytes, sig_bytes);
  }
  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  // Stage inputs into pinned host memory for zero-bounce async H2D DMA.
  // Skip memcpy when caller already packed directly into thread-local pinned scratch!
  if (item->packed_messages_len > 0u && pipe_ws.messages != item->packed_messages) {
    std::memcpy(pipe_ws.messages, item->packed_messages, item->packed_messages_len);
  }
  if (meta_u64) {
    if (pipe_ws.offsets != item->message_offsets) {
      std::memcpy(pipe_ws.offsets, item->message_offsets, total_messages * sizeof(uint64_t));
    }
    if (pipe_ws.lengths != item->message_lengths) {
      std::memcpy(pipe_ws.lengths, item->message_lengths, total_messages * sizeof(uint64_t));
    }
  } else if (meta_u32) {
    if (pipe_ws.offsets != item->message_offsets) {
      std::memcpy(pipe_ws.offsets, item->message_offsets, (total_messages + 1u) * sizeof(uint32_t));
    }
  }
  uint8_t *h_sk = pipe_ws.aux_in;
  uint8_t *h_dom = h_sk + sk_bytes;
  uint8_t *h_msg_dst = h_dom + dom_bytes;
  uint8_t *h_e_dst = h_msg_dst + item->message_dst.len;
  std::memcpy(h_sk, item->secret_keys_be, sk_bytes);
  std::memcpy(h_dom, item->domains_be, dom_bytes);
  if (item->message_dst.len > 0u)
    std::memcpy(h_msg_dst, item->message_dst.data, item->message_dst.len);
  if (item->e_dst.len > 0u)
    std::memcpy(h_e_dst, item->e_dst.data, item->e_dst.len);

  uint8_t *d_messages = nullptr;
  uint64_t *d_offsets = nullptr;
  uint64_t *d_lengths = nullptr;
  uint8_t *d_aux_in = nullptr;
  uint8_t *d_scalars_le = nullptr;
  uint8_t *d_row_scales_le = nullptr;
  uint32_t *d_invalid = nullptr;
  cubbs_sppark_bucket_t *d_outputs = nullptr;
  uint8_t *d_signatures = nullptr;
  CuBbsCommonMsmTransient msm_transient;

  const size_t total_scalars = item_count * term_count;
  const size_t d_off_bytes = meta_u64
      ? (total_messages * sizeof(uint64_t))
      : (meta_u32 ? ((total_messages + 1u) * sizeof(uint32_t)) : 0u);
  const size_t d_len_bytes = meta_u64 ? (total_messages * sizeof(uint64_t)) : 0u;
  auto &arena = cubbs_bbs_thread_device_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(item->packed_messages_len) +
      CuBbsDeviceArena::align_up(d_off_bytes) +
      CuBbsDeviceArena::align_up(d_len_bytes) +
      CuBbsDeviceArena::align_up(std::max<size_t>(1u, aux_in_bytes)) +
      CuBbsDeviceArena::align_up(total_scalars * 32u) +
      CuBbsDeviceArena::align_up(item_count * 32u) +
      CuBbsDeviceArena::align_up(sizeof(uint32_t)) +
      CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(sig_bytes);
  if (err == cudaSuccess) err = arena.reserve_and_reset(arena_bytes);
  if (err == cudaSuccess) {
    d_messages = arena.alloc<uint8_t>(item->packed_messages_len);
    d_offsets = d_off_bytes > 0u ? reinterpret_cast<uint64_t *>(arena.alloc<uint8_t>(d_off_bytes)) : nullptr;
    d_lengths = d_len_bytes > 0u ? arena.alloc<uint64_t>(total_messages) : nullptr;
    d_aux_in = arena.alloc<uint8_t>(std::max<size_t>(1u, aux_in_bytes));
    d_scalars_le = arena.alloc<uint8_t>(total_scalars * 32u);
    d_row_scales_le = arena.alloc<uint8_t>(item_count * 32u);
    d_invalid = arena.alloc<uint32_t>(1u);
    d_outputs = arena.alloc<cubbs_sppark_bucket_t>(item_count);
    d_signatures = arena.alloc<uint8_t>(sig_bytes);
    err = msm_transient.alloc(*cached, term_count, item_count, large_window4, stream);
  }

  // Async H2D transfers.
  if (err == cudaSuccess && item->packed_messages_len > 0u)
    err = cudaMemcpyAsync(d_messages, pipe_ws.messages, item->packed_messages_len,
                          cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess && d_off_bytes > 0u) {
    err = cudaMemcpyAsync(d_offsets, pipe_ws.offsets, d_off_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess && d_len_bytes > 0u) {
    err = cudaMemcpyAsync(d_lengths, pipe_ws.lengths, d_len_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(d_aux_in, pipe_ws.aux_in, aux_in_bytes,
                          cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(d_invalid, 0, sizeof(uint32_t), stream);

  const uint8_t *d_sk = d_aux_in;
  const uint8_t *d_dom = d_sk + sk_bytes;
  const uint8_t *d_msg_dst = d_dom + dom_bytes;
  const uint8_t *d_e_dst = d_msg_dst + item->message_dst.len;

  const bool sign_profile = env_flag_enabled("CUBBS_BBS_SIGN_KERNEL_PROFILE");
  cudaEvent_t ev[7]{};
  if (sign_profile) {
    for (int i = 0; i < 7; ++i) (void)cudaEventCreate(&ev[i]);
    (void)cudaEventRecord(ev[0], stream);
  }

  // Kernel 1: GPU H2S directly into MSM LE scalar slots [2 .. M+1].
  if (err == cudaSuccess && total_messages > 0u) {
    constexpr uint32_t h2s_threads = 128u;
    const uint32_t h2s_blocks = static_cast<uint32_t>(
        (total_messages + h2s_threads - 1u) / h2s_threads);
    cubbs_bbs_sign_h2s_messages_to_le_kernel<<<h2s_blocks, h2s_threads, 0, stream>>>(
        d_messages, d_offsets, d_lengths, uniform_msg_len, item_count, message_count, term_count,
        d_msg_dst, item->message_dst.len, d_scalars_le);
    err = cudaGetLastError();
  }
  if (sign_profile && err == cudaSuccess) (void)cudaEventRecord(ev[1], stream);

  // Kernel 2: GPU e + (sk + e)^-1 + slot 0/1 setup.
  // Use 64 threads/block so B=2048 launches 32 blocks (filling RTX 5060 Ti's 36 SMs).
  if (err == cudaSuccess) {
    constexpr uint32_t e_threads = 64u;
    const uint32_t e_blocks = static_cast<uint32_t>(
        (item_count + e_threads - 1u) / e_threads);
    cubbs_bbs_sign_e_and_inv_kernel<<<e_blocks, e_threads, 0, stream>>>(
        d_sk, item->secret_key_stride, d_dom, item->domain_stride,
        item_count, message_count, term_count,
        d_e_dst, item->e_dst.len, d_scalars_le, d_row_scales_le,
        d_signatures, d_invalid);
    err = cudaGetLastError();
  }
  if (sign_profile && err == cudaSuccess) (void)cudaEventRecord(ev[2], stream);

  // Kernel 3: Scale each scalar row by (sk + e)^-1 on GPU.
  if (err == cudaSuccess) {
    constexpr uint32_t scale_threads = 256u;
    const uint32_t scale_blocks = static_cast<uint32_t>(
        (total_scalars + scale_threads - 1u) / scale_threads);
    cubbs_bbs_scale_scalar_rows_kernel<<<scale_blocks, scale_threads, 0, stream>>>(
        d_scalars_le, d_row_scales_le, term_count, total_scalars);
    err = cudaGetLastError();
  }
  if (sign_profile && err == cudaSuccess) (void)cudaEventRecord(ev[3], stream);

  // Kernel 4: Cached common-points MSM -> A_i in xyzz_t.
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_cached_common_msm_range(
        *cached, msm_transient, d_scalars_le, term_count, item_count,
        d_outputs, stream);
  }
  if (sign_profile && err == cudaSuccess) (void)cudaEventRecord(ev[4], stream);

  // Kernel 5: GPU Montgomery batch inversion + BLS12-381 G1 compress directly into
  // d_signatures[row * 80 + 0..48].
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_xyzz_to_compressed(
        d_outputs, item_count, d_signatures, 80u, 0u, stream);
  }
  if (sign_profile && err == cudaSuccess) (void)cudaEventRecord(ev[5], stream);

  // Single D2H of B * 80 bytes + 4-byte invalid flag.
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.aux_out, d_signatures, sig_bytes,
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.invalid_flag, d_invalid, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (sign_profile && err == cudaSuccess) (void)cudaEventRecord(ev[6], stream);
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (sign_profile) {
    if (err == cudaSuccess) {
      float t_h2s = 0, t_e = 0, t_scale = 0, t_msm = 0, t_cmp = 0, t_d2h = 0;
      (void)cudaEventElapsedTime(&t_h2s, ev[0], ev[1]);
      (void)cudaEventElapsedTime(&t_e, ev[1], ev[2]);
      (void)cudaEventElapsedTime(&t_scale, ev[2], ev[3]);
      (void)cudaEventElapsedTime(&t_msm, ev[3], ev[4]);
      (void)cudaEventElapsedTime(&t_cmp, ev[4], ev[5]);
      (void)cudaEventElapsedTime(&t_d2h, ev[5], ev[6]);
      std::fprintf(stderr,
                   "cubbs_sign_gpu_events B=%zu M=%zu h2s=%.3fms e_inv=%.3fms "
                   "scale=%.3fms msm=%.3fms compress=%.3fms d2h=%.3fms\n",
                   item_count, message_count, t_h2s, t_e, t_scale, t_msm, t_cmp, t_d2h);
    }
    for (int i = 0; i < 7; ++i) (void)cudaEventDestroy(ev[i]);
  }

  // Free legacy MSM transient allocations if any (arena buffers stay persistent).
  msm_transient.free_async(stream);

  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
  if (*pipe_ws.invalid_flag != 0u) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;

  std::memcpy(item->signatures_out, pipe_ws.aux_out, sig_bytes);
  const uint8_t route = universal_window4
      ? CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL
      : (large_window4 ? CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE
                       : CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS);
  for (size_t row = 0u; row < item_count; ++row) {
    set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 80u, route);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

// ============================================================================
// GPU-Resident Fused BBS Pre-B Aggregate Verify Kernels
// ============================================================================

// Hash every raw message into flat LE scalar array d_msg_scalars_le[total_messages * 32].
// Supports 0B uniform, 4B/msg compact uint32_t prefix offsets, and legacy uint64_t modes.
__global__ static void cubbs_bbs_verify_h2s_flat_to_le_kernel(
    const uint8_t *messages, const uint64_t *offsets, const uint64_t *lengths,
    size_t uniform_len, size_t total_messages,
    const uint8_t *message_dst, size_t message_dst_len,
    uint8_t *msg_scalars_le) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_messages) return;
  size_t msg_off = 0u;
  size_t msg_len = uniform_len;
  if (lengths != nullptr) {
    msg_off = static_cast<size_t>(offsets[idx]);
    msg_len = static_cast<size_t>(lengths[idx]);
  } else if (offsets != nullptr) {
    const auto *offs32 = reinterpret_cast<const uint32_t *>(offsets);
    msg_off = static_cast<size_t>(offs32[idx]);
    msg_len = static_cast<size_t>(offs32[idx + 1u] - offs32[idx]);
  } else {
    msg_off = idx * uniform_len;
  }
  uint32_t uniform_words[12];
  expand_message_xmd_sha256_48_words_device(
      messages + msg_off, msg_len,
      message_dst, message_dst_len, uniform_words);
  uint64_t le[4];
  bls12_381_fr_reduce_words12_to_le64(uniform_words, le);
  bls12_381_fr_store_le(le, msg_scalars_le + (idx * 32u));
}

// Parallel Fr reduction across B items into agg_b_scalars_le[term_count * 32]:
//   block 0:     sum_i c_i
//   block 1:     sum_i c_i * domain_i
//   block 2 + j: sum_{i : j < M_i} c_i * msg_scalar[i, j]
template <uint32_t kThreads>
__global__ static void cubbs_bbs_verify_reduce_b_scalars_kernel(
    const uint8_t *coefficients_le,
    const uint8_t *domains_be, size_t domain_stride,
    const uint8_t *msg_scalars_le,
    const uint32_t *item_message_counts,
    const uint64_t *item_message_starts,
    size_t item_count, size_t uniform_msg_count, size_t term_count,
    uint8_t *agg_b_scalars_le, uint32_t *invalid_out) {
  __shared__ __align__(16) uint8_t smem[kThreads * sizeof(cubbs_sppark_scalar_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_scalar_t *>(smem);

  const size_t slot = blockIdx.x;
  if (slot >= term_count) return;
  const uint32_t tid = threadIdx.x;

  cubbs_sppark_scalar_t acc;
  acc.zero();

  if (slot == 0u) {
    for (size_t i = tid; i < item_count; i += kThreads) {
      cubbs_sppark_scalar_t c;
      memcpy(&c, coefficients_le + (i * 32u), 32u);
      c.to();
      acc += c;
    }
  } else if (slot == 1u) {
    if (domain_stride == 0u) {
      if (tid == 0u && !cubbs_bbs_scalar_be_is_canonical(domains_be)) {
        atomicExch(invalid_out, 1u);
      }
      uint8_t dom_le[32];
#pragma unroll
      for (int b = 0; b < 32; ++b) dom_le[b] = domains_be[31 - b];
      cubbs_sppark_scalar_t dom;
      memcpy(&dom, dom_le, 32u);
      dom.to();
      for (size_t i = tid; i < item_count; i += kThreads) {
        cubbs_sppark_scalar_t c;
        memcpy(&c, coefficients_le + (i * 32u), 32u);
        c.to();
        acc += c;
      }
      acc *= dom;
    } else {
      for (size_t i = tid; i < item_count; i += kThreads) {
        const uint8_t *dom_be = domains_be + (i * domain_stride);
        if (!cubbs_bbs_scalar_be_is_canonical(dom_be)) {
          atomicExch(invalid_out, 1u);
          continue;
        }
        uint8_t dom_le[32];
#pragma unroll
        for (int b = 0; b < 32; ++b) dom_le[b] = dom_be[31 - b];
        cubbs_sppark_scalar_t dom;
        cubbs_sppark_scalar_t c;
        memcpy(&dom, dom_le, 32u);
        memcpy(&c, coefficients_le + (i * 32u), 32u);
        dom.to();
        c.to();
        acc += dom * c;
      }
    }
  } else {
    const size_t msg_idx = slot - 2u;
    if (item_message_counts == nullptr) {
      for (size_t i = tid; i < item_count; i += kThreads) {
        cubbs_sppark_scalar_t m;
        cubbs_sppark_scalar_t c;
        memcpy(&m, msg_scalars_le + ((i * uniform_msg_count + msg_idx) * 32u), 32u);
        memcpy(&c, coefficients_le + (i * 32u), 32u);
        m.to();
        c.to();
        acc += m * c;
      }
    } else {
      for (size_t i = tid; i < item_count; i += kThreads) {
        if (msg_idx < static_cast<size_t>(item_message_counts[i])) {
          const size_t flat_idx =
              static_cast<size_t>(item_message_starts[i]) + msg_idx;
          cubbs_sppark_scalar_t m;
          cubbs_sppark_scalar_t c;
          memcpy(&m, msg_scalars_le + (flat_idx * 32u), 32u);
          memcpy(&c, coefficients_le + (i * 32u), 32u);
          m.to();
          c.to();
          acc += m * c;
        }
      }
    }
  }

  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreads >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid] += partials[tid + stride];
    __syncthreads();
  }
  if (tid == 0u) {
    cubbs_sppark_scalar_t sum = partials[0];
    sum.from();
    memcpy(agg_b_scalars_le + (slot * 32u), &sum, 32u);
  }
}

// Fused 100%-coalesced H2S + random-coefficient Fr accumulation for uniform-M Verify.
// Adjacent threads handle contiguous message slots (j, j+1, ..., j+31) so every
// warp reads contiguous message bytes while broadcasting a single c_i across all
// 32 lanes — eliminating the 134 MB d_msg_scalars_le allocation and 268 MB GDDR7
// round-trip per batch!
__global__ static void cubbs_bbs_verify_fused_h2s_reduce_tiled_kernel(
    const uint8_t *messages, const uint64_t *offsets, const uint64_t *lengths,
    size_t uniform_len, const uint8_t *coefficients_le,
    size_t item_count, size_t uniform_msg_count, uint32_t num_chunks,
    const uint8_t *message_dst, size_t message_dst_len,
    cubbs_sppark_scalar_t *partials_out,
    uint8_t *agg_b_scalars_le) {
  const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= uniform_msg_count) return;
  const uint32_t chunk = blockIdx.y;
  const size_t per_chunk = (item_count + num_chunks - 1u) / num_chunks;
  const size_t i_begin = static_cast<size_t>(chunk) * per_chunk;
  const size_t i_end = min(item_count, i_begin + per_chunk);

  cubbs_sppark_scalar_t acc;
  acc.zero();
  for (size_t i = i_begin; i < i_end; ++i) {
    const size_t idx = i * uniform_msg_count + j;
    size_t msg_off = 0u;
    size_t msg_len = uniform_len;
    if (lengths != nullptr) {
      msg_off = static_cast<size_t>(offsets[idx]);
      msg_len = static_cast<size_t>(lengths[idx]);
    } else if (offsets != nullptr) {
      const auto *offs32 = reinterpret_cast<const uint32_t *>(offsets);
      msg_off = static_cast<size_t>(offs32[idx]);
      msg_len = static_cast<size_t>(offs32[idx + 1u] - offs32[idx]);
    } else {
      msg_off = idx * uniform_len;
    }
    uint32_t uniform_words[12];
    expand_message_xmd_sha256_48_words_device(
        messages + msg_off, msg_len,
        message_dst, message_dst_len, uniform_words);
    uint64_t le[4];
    bls12_381_fr_reduce_words12_to_le64(uniform_words, le);
    cubbs_sppark_scalar_t m;
    cubbs_sppark_scalar_t c;
    memcpy(&m, le, 32u);
    memcpy(&c, coefficients_le + (i * 32u), 32u);
    m.to();
    c.to();
    acc += m * c;
  }

  if (num_chunks == 1u) {
    acc.from();
    memcpy(agg_b_scalars_le + ((2u + j) * 32u), &acc, 32u);
  } else {
    partials_out[static_cast<size_t>(chunk) * uniform_msg_count + j] = acc;
  }
}

__global__ static void cubbs_bbs_verify_reduce_tiled_partials_kernel(
    const cubbs_sppark_scalar_t *partials_in,
    size_t uniform_msg_count, uint32_t num_chunks,
    uint8_t *agg_b_scalars_le) {
  const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= uniform_msg_count) return;
  cubbs_sppark_scalar_t acc = partials_in[j];
  for (uint32_t c = 1u; c < num_chunks; ++c) {
    acc += partials_in[static_cast<size_t>(c) * uniform_msg_count + j];
  }
  acc.from();
  memcpy(agg_b_scalars_le + ((2u + j) * 32u), &acc, 32u);
}

// Builds dual A-MSM scalar buffer on GPU:
//   dual_scalars_le[0 .. B-1]     = c_i
//   dual_scalars_le[B .. 2*B-1]   = -(e_i * c_i) mod r
__global__ static void cubbs_bbs_verify_build_dual_a_scalars_kernel(
    const uint8_t *signatures_e_be, const uint8_t *coefficients_le,
    size_t item_count, uint8_t *dual_scalars_le, uint32_t *invalid_out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= item_count) return;

  const uint8_t *e_be = signatures_e_be + (i * 32u);
  if (!cubbs_bbs_scalar_be_is_canonical(e_be)) {
    atomicExch(invalid_out, 1u);
    memset(dual_scalars_le + (i * 32u), 0, 32u);
    memset(dual_scalars_le + ((item_count + i) * 32u), 0, 32u);
    return;
  }
  uint8_t e_le[32];
#pragma unroll
  for (int b = 0; b < 32; ++b) e_le[b] = e_be[31 - b];

  cubbs_sppark_scalar_t fr_e;
  cubbs_sppark_scalar_t fr_c;
  memcpy(&fr_e, e_le, 32u);
  memcpy(&fr_c, coefficients_le + (i * 32u), 32u);
  // Write lhs scalar c_i verbatim (already LE).
  memcpy(dual_scalars_le + (i * 32u), &fr_c, 32u);

  fr_e.to();
  fr_c.to();
  cubbs_sppark_scalar_t prod = fr_e * fr_c;
  prod.cneg(true);
  prod.from();
  memcpy(dual_scalars_le + ((item_count + i) * 32u), &prod, 32u);
}

// ============================================================================
// Signed 8-bit (17-window, 128-bucket) GLV Pippenger Dual A-MSM for Verify
// ============================================================================
// Computes:
//   outputs[0] = sum_i c_i * A_i             (c_i is 128-bit -> 17 signed 8-bit windows)
//   outputs[1] = sum_i (-(c_i * e_i)) * A_i  (GLV split into k1, k2 -> 2x17 windows)
// Uses shared-memory counting sort so all 128 threads/block perform mixed-affine
// uadd() in lockstep with zero per-point table builds (optimal for both cached
// and unique signatures).

__device__ __forceinline__ static void cubbs_bbs_recode_s8_128_store(
    const uint8_t scalar16_le[16], size_t item_idx, size_t item_count,
    int16_t *stream_digits_out) {
  int32_t carry = 0;
  #pragma unroll
  for (uint32_t w = 0u; w < 16u; ++w) {
    const int32_t v = static_cast<int32_t>(scalar16_le[w]) + carry;
    int32_t digit = v;
    if (v > 128) {
      digit = v - 256;
      carry = 1;
    } else {
      carry = 0;
    }
    stream_digits_out[static_cast<size_t>(w) * item_count + item_idx] =
        static_cast<int16_t>(digit);
  }
  stream_digits_out[static_cast<size_t>(16u) * item_count + item_idx] =
      static_cast<int16_t>(carry);
}

__global__ static void cubbs_bbs_verify_pippenger_recode_kernel(
    const uint8_t *signatures_e_be, const uint8_t *coefficients_le,
    size_t item_count, int16_t *digits_out, uint32_t *invalid_out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= item_count) return;

  const uint8_t *e_be = signatures_e_be + (i * 32u);
  const uint8_t *c_le = coefficients_le + (i * 32u);
  uint64_t c_hi0 = 0ull;
  uint64_t c_hi1 = 0ull;
  memcpy(&c_hi0, c_le + 16u, 8u);
  memcpy(&c_hi1, c_le + 24u, 8u);
  if (!cubbs_bbs_scalar_be_is_canonical(e_be) || (c_hi0 | c_hi1) != 0ull) {
    atomicExch(invalid_out, 1u);
    const uint8_t zero16[16] = {};
    cubbs_bbs_recode_s8_128_store(zero16, i, item_count, digits_out);
    cubbs_bbs_recode_s8_128_store(zero16, i, item_count, digits_out + (17u * item_count));
    cubbs_bbs_recode_s8_128_store(zero16, i, item_count, digits_out + (34u * item_count));
    return;
  }

  uint8_t e_le[32];
  #pragma unroll
  for (int b = 0; b < 32; ++b) e_le[b] = e_be[31 - b];

  cubbs_sppark_scalar_t fr_e;
  cubbs_sppark_scalar_t fr_c;
  memcpy(&fr_e, e_le, 32u);
  memcpy(&fr_c, c_le, 32u);
  fr_e.to();
  fr_c.to();
  cubbs_sppark_scalar_t prod = fr_e * fr_c;
  prod.cneg(true);
  prod.from();
  uint8_t prod_le[32];
  memcpy(prod_le, &prod, 32u);

  uint8_t k1_le[16];
  uint8_t k2_le[16];
  cubbs_bbs_scalar_div_by_z2_device(prod_le, k1_le, k2_le);

  cubbs_bbs_recode_s8_128_store(c_le, i, item_count, digits_out);
  cubbs_bbs_recode_s8_128_store(k1_le, i, item_count, digits_out + (17u * item_count));
  cubbs_bbs_recode_s8_128_store(k2_le, i, item_count, digits_out + (34u * item_count));
}

// Generic Signed 8-bit (17-window, 128-bucket) Pippenger bucket accumulator.
// Works for both Verify (3 streams sharing one point array) and ProofVerify
// aggregate pairing (2 point arrays sharing one 128-bit scalar stream).
__global__ static void __launch_bounds__(128, 2)
cubbs_bbs_verify_pippenger_buckets_s8_kernel(
    const cubbs_sppark_plain_affine_t *points,
    size_t point_stream_stride,
    const int16_t *digits,
    size_t digit_stream_stride,
    size_t item_count, uint32_t chunks,
    cubbs_sppark_bucket_t *win_chunk_sums) {
  constexpr uint32_t kMaxTile = 2048u;
  __shared__ uint32_t s_count[129];
  __shared__ uint32_t s_cursor[129];
  __shared__ uint32_t s_warp_totals[4];
  __shared__ uint32_t s_warp_offsets[4];
  __shared__ uint16_t s_sorted[kMaxTile];
  __shared__ __align__(16) uint8_t s_buckets_raw[128u * sizeof(cubbs_sppark_bucket_t)];
  auto *s_buckets = reinterpret_cast<cubbs_sppark_bucket_t *>(s_buckets_raw);

  const uint32_t w = blockIdx.x;          // 0..16
  const uint32_t stream_idx = blockIdx.y; // 0..num_streams-1
  const uint32_t chunk_id = blockIdx.z;   // 0..chunks-1
  const uint32_t tid = threadIdx.x;       // 0..127 -> bucket mag = tid + 1
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;

  const size_t tile_stride = (item_count + chunks - 1u) / chunks;
  const size_t tile_begin = static_cast<size_t>(chunk_id) * tile_stride;
  const size_t tile_end = min(item_count, tile_begin + tile_stride);
  const uint32_t tile_len =
      (tile_begin < tile_end) ? static_cast<uint32_t>(tile_end - tile_begin) : 0u;

  s_count[tid + 1u] = 0u;
  if (tid == 0u) s_count[0] = 0u;
  __syncthreads();

  const cubbs_sppark_plain_affine_t *stream_points =
      points + (static_cast<size_t>(stream_idx) * point_stream_stride);
  const int16_t *win_digits =
      digits + (static_cast<size_t>(stream_idx) * digit_stream_stride) +
      (static_cast<size_t>(w) * item_count) + tile_begin;

  // Pass 1a: Histogram bucket magnitudes (1..128) into shared memory.
  for (uint32_t i = tid; i < tile_len; i += 128u) {
    const int32_t d = static_cast<int32_t>(win_digits[i]);
    if (d != 0) {
      const uint32_t mag = static_cast<uint32_t>(d < 0 ? -d : d);
      atomicAdd(&s_count[mag], 1u);
    }
  }
  __syncthreads();

  // Pass 1b: 128-element exclusive prefix sum across 4 warps.
  const uint32_t my_cnt = s_count[tid + 1u];
  uint32_t scan = my_cnt;
  #pragma unroll
  for (uint32_t off = 1u; off < 32u; off <<= 1u) {
    const uint32_t up = __shfl_up_sync(0xffffffffu, scan, off);
    if (lane >= off) scan += up;
  }
  if (lane == 31u) s_warp_totals[warp] = scan;
  __syncthreads();
  if (tid == 0u) {
    uint32_t acc_off = 0u;
    for (uint32_t wp = 0u; wp < 4u; ++wp) {
      s_warp_offsets[wp] = acc_off;
      acc_off += s_warp_totals[wp];
    }
  }
  __syncthreads();
  const uint32_t my_off = s_warp_offsets[warp] + (scan - my_cnt);
  s_cursor[tid + 1u] = my_off;
  __syncthreads();

  // Pass 1c: Scatter (local_idx << 1 | neg) into contiguous per-bucket slices.
  for (uint32_t i = tid; i < tile_len; i += 128u) {
    const int32_t d = static_cast<int32_t>(win_digits[i]);
    if (d != 0) {
      const bool neg = d < 0;
      const uint32_t mag = static_cast<uint32_t>(neg ? -d : d);
      const uint32_t pos = atomicAdd(&s_cursor[mag], 1u);
      s_sorted[pos] = static_cast<uint16_t>((i << 1u) | (neg ? 1u : 0u));
    }
  }
  __syncthreads();

  // Pass 2: All 128 threads accumulate their bucket via mixed-affine uadd!
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (uint32_t k = 0u; k < my_cnt; ++k) {
    const uint16_t packed = s_sorted[my_off + k];
    const size_t pt_idx = tile_begin + static_cast<size_t>(packed >> 1u);
    acc.uadd(stream_points[pt_idx], (packed & 1u) != 0u);
  }
  s_buckets[tid] = acc;
  __syncthreads();

  // Pass 3: 7-step parallel reverse inclusive suffix scan + 7-step tree reduce:
  //   sum_{i=0}^{127} sum_{j=i}^{127} B_j = sum_{b=1}^{128} b * B_b.
  for (uint32_t step = 1u; step < 128u; step <<= 1u) {
    cubbs_sppark_bucket_t other;
    other.inf();
    if (tid + step < 128u) other = s_buckets[tid + step];
    __syncthreads();
    if (tid + step < 128u) s_buckets[tid].uadd(other);
    __syncthreads();
  }
  for (uint32_t stride = 64u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      s_buckets[tid].uadd(s_buckets[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0u) {
    win_chunk_sums[(static_cast<size_t>(stream_idx) * 17u + w) * chunks + chunk_id] =
        s_buckets[0];
  }
}

__global__ static void cubbs_bbs_verify_pippenger_combine_s8_kernel(
    const cubbs_sppark_bucket_t *win_chunk_sums,
    uint32_t chunks,
    uint32_t num_streams,
    bool combine_glv_12,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t s_win_raw[3u * 17u * sizeof(cubbs_sppark_bucket_t)];
  __shared__ __align__(16) uint8_t s_out_raw[3u * sizeof(cubbs_sppark_bucket_t)];
  auto *s_win = reinterpret_cast<cubbs_sppark_bucket_t *>(s_win_raw);
  auto *s_out = reinterpret_cast<cubbs_sppark_bucket_t *>(s_out_raw);

  const uint32_t tid = threadIdx.x;
  if (tid < 17u) {
    for (uint32_t s = 0u; s < num_streams; ++s) {
      const size_t base = (static_cast<size_t>(s) * 17u + tid) * chunks;
      cubbs_sppark_bucket_t acc = win_chunk_sums[base];
      for (uint32_t c = 1u; c < chunks; ++c) {
        acc.uadd(win_chunk_sums[base + c]);
      }
      s_win[s * 17u + tid] = acc;
    }
  }
  __syncwarp();

  if (tid < num_streams) {
    cubbs_sppark_bucket_t acc;
    acc.inf();
    for (int w = 16; w >= 0; --w) {
      if (!acc.is_inf()) {
        #pragma unroll
        for (uint32_t d = 0u; d < 8u; ++d) cubbs_bbs_xyzz_dbl_u(acc);
      }
      acc.uadd(s_win[tid * 17u + static_cast<uint32_t>(w)]);
    }
    if (combine_glv_12 && tid == 2u) {
      cubbs_bbs_apply_glv_endomorphism(acc);
    }
    s_out[tid] = acc;
  }
  __syncwarp();

  if (tid == 0u) {
    outputs[0] = s_out[0];
  } else if (tid == 1u) {
    cubbs_sppark_bucket_t second = s_out[1];
    if (combine_glv_12) second.uadd(s_out[2]);
    outputs[1] = second;
  }
}

// Adds rhs = b_agg + ae_agg and normalizes (a_agg, rhs_agg) to two plain affine points
// in Montgomery form for direct consumption by cubbs_bbs_cuda_launch_pairing_eq_single_on_stream.
__global__ static void cubbs_bbs_verify_finalize_pair_affines_kernel(
    const cubbs_sppark_bucket_t *a_and_ae_outputs,
    const cubbs_sppark_bucket_t *b_output,
    cubbs_sppark_plain_affine_t *pair_affines_out,
    uint32_t *inf_flag_out) {
  const uint32_t tid = threadIdx.x;
  __shared__ __align__(16) uint8_t pts_smem[2u * sizeof(cubbs_sppark_bucket_t)];
  auto *pts = reinterpret_cast<cubbs_sppark_bucket_t *>(pts_smem);
  if (tid == 0u) {
    pts[0] = a_and_ae_outputs[0];
    cubbs_sppark_bucket_t rhs = *b_output;
    rhs.add(a_and_ae_outputs[1]);
    pts[1] = rhs;
    if (pts[0].is_inf() || pts[1].is_inf()) {
      *inf_flag_out = 1u;
    } else {
      *inf_flag_out = 0u;
    }
  }
  __syncthreads();

  // Warp 0 (32 threads) runs reciprocal on ZZZ of point 0 (tid=0) and point 1 (tid=1)!
  const fp_t one = fp_t::one();
  fp_t zzz = one;
  if (tid < 2u && !pts[tid].is_inf()) {
    const fp_t *coords = reinterpret_cast<const fp_t *>(&pts[tid]);
    zzz = coords[2];
  }
  const fp_t inv_zzz = zzz.reciprocal();
  if (tid < 2u) {
    if (pts[tid].is_inf()) {
      fp_t zero;
      zero.zero();
      pair_affines_out[tid] = cubbs_sppark_plain_affine_t{zero, zero};
    } else {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pts[tid]);
      const fp_t affine_y = inv_zzz * coords[1];
      const fp_t inv_z = inv_zzz * coords[3];
      fp_t affine_x = inv_z * inv_z;
      affine_x *= coords[0];
      pair_affines_out[tid] = cubbs_sppark_plain_affine_t{affine_x, affine_y};
    }
  }
}

int32_t cubbs_bbs_cuda_verify_pre_b_aggregate_fused(
    const CuBbsBbsGpuVerifyPreBAggregateFusedItem *item,
    CuBbsBbsGpuResult *result) {
  if (item == nullptr || result == nullptr) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  const size_t item_count = item->item_count;
  const size_t term_count = item->common_term_count;
  if (item_count == 0u) {
    result->code = CUBBS_BBS_GPU_SUCCESS;
    result->value = 1u;
    result->bytes_written = 0u;
    return CUBBS_BBS_GPU_SUCCESS;
  }
  const bool unique_pk = (item_count > 1u && item->public_key.len == item_count * 96u);
  if (term_count < 2u || item->common_points == nullptr ||
      item->signatures == nullptr || item->coefficients_le == nullptr ||
      item->domains_be == nullptr ||
      (item->domain_stride != 0u && item->domain_stride != 32u) ||
      item->message_dst.len > 255u ||
      (item->message_dst.len > 0u && item->message_dst.data == nullptr) ||
      (!unique_pk && item->public_key.len != 96u) || item->public_key.data == nullptr ||
      item->g2_generator.len != 96u || item->g2_generator.data == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const bool mixed_counts = item->item_message_counts != nullptr;
  if (mixed_counts && item->item_message_starts == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t total_messages = item->total_message_count;
  const bool meta_uniform =
      total_messages > 0u && item->message_offsets == nullptr && item->message_lengths == nullptr;
  const bool meta_u32 =
      total_messages > 0u && item->message_offsets != nullptr && item->message_lengths == nullptr;
  const bool meta_u64 =
      total_messages > 0u && item->message_offsets != nullptr && item->message_lengths != nullptr;
  if (total_messages > 0u &&
      ((!meta_uniform && !meta_u32 && !meta_u64) ||
       (meta_uniform && (item->packed_messages_len % total_messages) != 0u) ||
       (meta_u32 && item->packed_messages_len > 0xFFFFFFFFull) ||
       (item->packed_messages_len > 0u && item->packed_messages == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t uniform_msg_len = meta_uniform ? (item->packed_messages_len / total_messages) : 0u;

  cubbs_bbs_ensure_cuda_device_schedule();

  // Check cached common-points route for 1-row B-MSM.
  bool large_window4 = false;
  bool universal_window4 = false;
  if (bbs_homogeneous_kernel_enabled() &&
      bbs_specialized_common_points_supported(term_count)) {
    large_window4 = false;
  } else if (bbs_large_shape_kernel_supported(term_count)) {
    large_window4 = true;
  } else if (bbs_universal_tiled_common_supported(term_count)) {
    large_window4 = true;
    universal_window4 = true;
  } else {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  const bool affine_common =
      !large_window4 && term_count >= 18u && bbs_homogeneous_affine_comb_enabled();
  int32_t cache_status = get_fused_common_points_cache(
      item->common_points, term_count, &cached, large_window4, affine_common,
      0u, universal_window4);
  if (cache_status != CUBBS_BBS_GPU_SUCCESS) return cache_status;

  auto &host_ws = cubbs_fused_pinned_host_workspace();
  auto &pipe_ws = cubbs_fused_pipeline_pinned_workspace();
  auto &dual_ws = cubbs_same_points_dual_msm_workspace();
  const cudaStream_t stream_b = host_ws.execution_stream(
      cached->stream != nullptr ? cached->stream : static_cast<cudaStream_t>(0));

  // Prepare G2 lines (for Same-PK cache pk_lines; for Unique-PK only G2 generator lines!).
  if (!unique_pk) {
    if (!pipe_ws.pk_valid ||
        std::memcmp(pipe_ws.pk_key, item->public_key.data, 96u) != 0) {
      int32_t g2_st = cubbs_bbs_blst_get_prepared_g2_lines(
          item->public_key, pipe_ws.pk_lines);
      if (g2_st != CUBBS_BBS_GPU_SUCCESS) return g2_st;
      std::memcpy(pipe_ws.pk_key, item->public_key.data, 96u);
      pipe_ws.pk_valid = true;
    }
  }
  if (!pipe_ws.g2_valid ||
      std::memcmp(pipe_ws.g2_key, item->g2_generator.data, 96u) != 0) {
    int32_t g2_st = cubbs_bbs_blst_get_prepared_g2_lines(
        item->g2_generator, pipe_ws.g2_lines);
    if (g2_st != CUBBS_BBS_GPU_SUCCESS) return g2_st;
    std::memcpy(pipe_ws.g2_key, item->g2_generator.data, 96u);
    pipe_ws.g2_valid = true;
  }

  // Fast zero-alloc check whether A_0..A_{B-1} match the GPU-cached affine points.
  bool matches_cached = (dual_ws.point_key.size() == item_count * 48u &&
                         dual_ws.capacity >= item_count);
  if (matches_cached) {
    const char *cached_bytes = dual_ws.point_key.data();
    for (size_t i = 0u; i < item_count; ++i) {
      if (std::memcmp(cached_bytes + (i * 48u), item->signatures + (i * 80u), 48u) != 0) {
        matches_cached = false;
        break;
      }
    }
  }
  const bool need_gpu_decode_a = !matches_cached;

  const size_t dom_bytes = item->domain_stride == 0u ? 32u : item_count * 32u;
  const size_t coef_bytes = item_count * 32u;
  const size_t e_bytes = item_count * 32u;
  const size_t a_comp_bytes = need_gpu_decode_a ? (item_count * 48u) : 0u;
  const size_t pks_comp_bytes = unique_pk ? (item_count * 96u) : 0u;
  const size_t mixed_meta_bytes = mixed_counts
      ? (item_count * (sizeof(uint32_t) + sizeof(uint64_t)))
      : 0u;
  const size_t aux_in_bytes =
      coef_bytes + dom_bytes + e_bytes + a_comp_bytes + pks_comp_bytes +
      item->message_dst.len + mixed_meta_bytes + 64u;
  const size_t aux_out_bytes = 32u;

  cudaError_t err = pipe_ws.ensure_messages(item->packed_messages_len);
  if (err == cudaSuccess && meta_u64)
    err = pipe_ws.ensure_msg_meta(total_messages);
  else if (err == cudaSuccess && meta_u32)
    err = pipe_ws.ensure_msg_meta((total_messages + 2u) / 2u);
  if (err == cudaSuccess)
    err = pipe_ws.ensure_aux(aux_in_bytes, aux_out_bytes);
  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  if (item->packed_messages_len > 0u && pipe_ws.messages != item->packed_messages)
    std::memcpy(pipe_ws.messages, item->packed_messages, item->packed_messages_len);
  if (meta_u64) {
    if (pipe_ws.offsets != item->message_offsets)
      std::memcpy(pipe_ws.offsets, item->message_offsets, total_messages * sizeof(uint64_t));
    if (pipe_ws.lengths != item->message_lengths)
      std::memcpy(pipe_ws.lengths, item->message_lengths, total_messages * sizeof(uint64_t));
  } else if (meta_u32) {
    if (pipe_ws.offsets != item->message_offsets)
      std::memcpy(pipe_ws.offsets, item->message_offsets, (total_messages + 1u) * sizeof(uint32_t));
  }

  size_t aux_off = 0u;
  auto push_aux = [&](size_t bytes, size_t align) -> size_t {
    aux_off = (aux_off + align - 1u) & ~(align - 1u);
    const size_t start = aux_off;
    aux_off += bytes;
    return start;
  };
  const size_t off_coef = push_aux(coef_bytes, 8u);
  const size_t off_dom = push_aux(dom_bytes, 8u);
  const size_t off_e = push_aux(e_bytes, 8u);
  const size_t off_a_comp = push_aux(a_comp_bytes, 8u);
  const size_t off_pks_comp = push_aux(pks_comp_bytes, 8u);
  const size_t off_item_starts = push_aux(mixed_counts ? item_count * sizeof(uint64_t) : 0u, 8u);
  const size_t off_item_counts = push_aux(mixed_counts ? item_count * sizeof(uint32_t) : 0u, 4u);
  const size_t off_msg_dst = push_aux(item->message_dst.len, 1u);

  std::memcpy(pipe_ws.aux_in + off_coef, item->coefficients_le, coef_bytes);
  std::memcpy(pipe_ws.aux_in + off_dom, item->domains_be, dom_bytes);
  for (size_t i = 0u; i < item_count; ++i) {
    const uint8_t *sig = item->signatures + (i * 80u);
    std::memcpy(pipe_ws.aux_in + off_e + (i * 32u), sig + 48u, 32u);
    if (need_gpu_decode_a) {
      std::memcpy(pipe_ws.aux_in + off_a_comp + (i * 48u), sig, 48u);
    }
  }
  if (unique_pk) {
    std::memcpy(pipe_ws.aux_in + off_pks_comp, item->public_key.data, pks_comp_bytes);
  }
  if (mixed_counts) {
    std::memcpy(pipe_ws.aux_in + off_item_starts, item->item_message_starts, item_count * sizeof(uint64_t));
    std::memcpy(pipe_ws.aux_in + off_item_counts, item->item_message_counts, item_count * sizeof(uint32_t));
  }
  if (item->message_dst.len > 0u) {
    std::memcpy(pipe_ws.aux_in + off_msg_dst, item->message_dst.data, item->message_dst.len);
  }

  static const bool pippenger_enabled = []() {
    const char *env = std::getenv("CUBBS_BBS_VERIFY_PIPPENGER");
    return env == nullptr || env_flag_enabled("CUBBS_BBS_VERIFY_PIPPENGER");
  }();
  const bool use_pippenger = pippenger_enabled && item_count >= 128u;
  const uint32_t chunks = static_cast<uint32_t>((item_count + 2047u) / 2048u);

  // Ensure persistent dual_ws buffers for GPU-cached A_i points + small-B tables.
  if (dual_ws.capacity < item_count) {
    if (dual_ws.points != nullptr) (void)cudaFree(dual_ws.points);
    if (dual_ws.scalars != nullptr) (void)cudaFree(dual_ws.scalars);
    if (dual_ws.components != nullptr) (void)cudaFree(dual_ws.components);
    if (dual_ws.products != nullptr) (void)cudaFree(dual_ws.products);
    if (dual_ws.tables != nullptr) (void)cudaFree(dual_ws.tables);
    dual_ws.points = nullptr;
    dual_ws.scalars = nullptr;
    dual_ws.components = nullptr;
    dual_ws.products = nullptr;
    dual_ws.tables = nullptr;
    dual_ws.capacity = 0u;
    dual_ws.point_key.clear();
    err = cudaMalloc(&dual_ws.points, item_count * sizeof(*dual_ws.points));
    if (err == cudaSuccess) err = cudaMalloc(&dual_ws.scalars, item_count * 64u);
    if (err == cudaSuccess) err = cudaMalloc(&dual_ws.components, item_count * 64u);
    if (err == cudaSuccess) err = cudaMalloc(&dual_ws.products, item_count * 2u * sizeof(*dual_ws.products));
    if (err == cudaSuccess) err = cudaMalloc(&dual_ws.tables, item_count * 9u * sizeof(*dual_ws.tables));
    if (err == cudaSuccess) dual_ws.capacity = item_count;
  }
  if (err == cudaSuccess && dual_ws.aggregate_lane_capacity < 1u) {
    if (dual_ws.buckets != nullptr) (void)cudaFree(dual_ws.buckets);
    if (dual_ws.window_sums != nullptr) (void)cudaFree(dual_ws.window_sums);
    if (dual_ws.outputs != nullptr) (void)cudaFree(dual_ws.outputs);
    dual_ws.buckets = nullptr;
    dual_ws.window_sums = nullptr;
    dual_ws.outputs = nullptr;
    dual_ws.aggregate_lane_capacity = 0u;
    const size_t bucket_count = 2u * CUBBS_DUAL_MSM_WINDOWS * CUBBS_DUAL_MSM_BUCKETS;
    err = cudaMalloc(&dual_ws.buckets, bucket_count * sizeof(*dual_ws.buckets));
    if (err == cudaSuccess)
      err = cudaMalloc(&dual_ws.window_sums, 2u * CUBBS_DUAL_MSM_WINDOWS * sizeof(*dual_ws.window_sums));
    if (err == cudaSuccess)
      err = cudaMalloc(&dual_ws.outputs, 2u * sizeof(*dual_ws.outputs));
    if (err == cudaSuccess) dual_ws.aggregate_lane_capacity = 1u;
  }

  // Allocate all transient device buffers from persistent thread-local arena (0 cudaMallocAsync!).
  const size_t uniform_msg_count = term_count - 2u;
  const bool use_fused_tiled_h2s = (!mixed_counts && total_messages > 0u);
  uint32_t h2s_chunks = 1u;
  if (use_fused_tiled_h2s) {
    while (h2s_chunks < 64u && (h2s_chunks * 2u) <= item_count &&
           (uniform_msg_count * h2s_chunks) < 16384u) {
      h2s_chunks <<= 1u;
    }
  }
  const size_t d_off_bytes = meta_u64
      ? (total_messages * sizeof(uint64_t))
      : (meta_u32 ? ((total_messages + 1u) * sizeof(uint32_t)) : 0u);
  const size_t d_len_bytes = meta_u64 ? (total_messages * sizeof(uint64_t)) : 0u;
  const size_t d_msg_scalars_bytes =
      use_fused_tiled_h2s ? 0u : (total_messages * 32u);
  const size_t d_h2s_partials_count =
      (use_fused_tiled_h2s && h2s_chunks > 1u) ? (static_cast<size_t>(h2s_chunks) * uniform_msg_count) : 0u;
  auto &arena = cubbs_bbs_thread_device_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(item->packed_messages_len) +
      CuBbsDeviceArena::align_up(d_off_bytes) +
      CuBbsDeviceArena::align_up(d_len_bytes) +
      CuBbsDeviceArena::align_up(d_msg_scalars_bytes) +
      CuBbsDeviceArena::align_up(d_h2s_partials_count * sizeof(cubbs_sppark_scalar_t)) +
      CuBbsDeviceArena::align_up(std::max<size_t>(1u, aux_off)) +
      CuBbsDeviceArena::align_up(term_count * 32u) +
      CuBbsDeviceArena::align_up(sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(sizeof(uint32_t)) * 2u +
      CuBbsDeviceArena::align_up(2u * sizeof(cubbs_sppark_plain_affine_t)) +
      CuBbsDeviceArena::align_up(sizeof(uint8_t)) +
      (use_pippenger
           ? (CuBbsDeviceArena::align_up(51u * item_count * sizeof(int16_t)) +
              CuBbsDeviceArena::align_up(51u * chunks * sizeof(cubbs_sppark_bucket_t)))
           : 0u) +
      (unique_pk
           ? (CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_bucket_t)) +
              CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_plain_affine_t)))
           : 0u);
  if (err == cudaSuccess) err = arena.reserve_and_reset(arena_bytes);

  uint8_t *d_messages = nullptr;
  uint64_t *d_offsets = nullptr;
  uint64_t *d_lengths = nullptr;
  uint8_t *d_msg_scalars_le = nullptr;
  cubbs_sppark_scalar_t *d_h2s_partials = nullptr;
  uint8_t *d_aux_in = nullptr;
  uint8_t *d_agg_b_scalars_le = nullptr;
  cubbs_sppark_bucket_t *d_b_output = nullptr;
  uint32_t *d_invalid = nullptr;
  cubbs_sppark_plain_affine_t *d_pair_affines = nullptr;
  uint32_t *d_inf_flag = nullptr;
  uint8_t *d_eq_out = nullptr;
  int16_t *d_digits = nullptr;
  cubbs_sppark_bucket_t *d_win_chunk_sums = nullptr;
  cubbs_sppark_bucket_t *d_scaled_a_xyzz = nullptr;
  cubbs_sppark_plain_affine_t *d_scaled_a_affines = nullptr;
  CuBbsCommonMsmTransient msm_transient;

  if (err == cudaSuccess) {
    d_messages = arena.alloc<uint8_t>(item->packed_messages_len);
    d_offsets = d_off_bytes > 0u ? reinterpret_cast<uint64_t *>(arena.alloc<uint8_t>(d_off_bytes)) : nullptr;
    d_lengths = d_len_bytes > 0u ? arena.alloc<uint64_t>(total_messages) : nullptr;
    d_msg_scalars_le = d_msg_scalars_bytes > 0u ? arena.alloc<uint8_t>(d_msg_scalars_bytes) : nullptr;
    d_h2s_partials = d_h2s_partials_count > 0u ? arena.alloc<cubbs_sppark_scalar_t>(d_h2s_partials_count) : nullptr;
    d_aux_in = arena.alloc<uint8_t>(std::max<size_t>(1u, aux_off));
    d_agg_b_scalars_le = arena.alloc<uint8_t>(term_count * 32u);
    d_b_output = arena.alloc<cubbs_sppark_bucket_t>(1u);
    d_invalid = arena.alloc<uint32_t>(1u);
    d_pair_affines = arena.alloc<cubbs_sppark_plain_affine_t>(2u);
    d_inf_flag = arena.alloc<uint32_t>(1u);
    d_eq_out = arena.alloc<uint8_t>(1u);
    if (use_pippenger) {
      d_digits = arena.alloc<int16_t>(51u * item_count);
      d_win_chunk_sums = arena.alloc<cubbs_sppark_bucket_t>(51u * chunks);
    }
    if (unique_pk) {
      d_scaled_a_xyzz = arena.alloc<cubbs_sppark_bucket_t>(item_count);
      d_scaled_a_affines = arena.alloc<cubbs_sppark_plain_affine_t>(item_count);
    }
    err = msm_transient.alloc(*cached, term_count, 1u, large_window4, stream_b);
  }

  if (err == cudaSuccess && item->packed_messages_len > 0u)
    err = cudaMemcpyAsync(d_messages, pipe_ws.messages, item->packed_messages_len,
                          cudaMemcpyHostToDevice, stream_b);
  if (err == cudaSuccess && d_off_bytes > 0u) {
    err = cudaMemcpyAsync(d_offsets, pipe_ws.offsets, d_off_bytes,
                          cudaMemcpyHostToDevice, stream_b);
  }
  if (err == cudaSuccess && d_len_bytes > 0u) {
    err = cudaMemcpyAsync(d_lengths, pipe_ws.lengths, d_len_bytes,
                          cudaMemcpyHostToDevice, stream_b);
  }
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(d_aux_in, pipe_ws.aux_in, aux_off,
                          cudaMemcpyHostToDevice, stream_b);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(d_invalid, 0, sizeof(uint32_t), stream_b);

  const uint8_t *d_coef = d_aux_in + off_coef;
  const uint8_t *d_dom = d_aux_in + off_dom;
  const uint8_t *d_e = d_aux_in + off_e;
  const uint8_t *d_a_comp = d_aux_in + off_a_comp;
  const uint8_t *d_pks_comp = d_aux_in + off_pks_comp;
  const uint64_t *d_item_starts = mixed_counts
      ? reinterpret_cast<const uint64_t *>(d_aux_in + off_item_starts)
      : nullptr;
  const uint32_t *d_item_counts = mixed_counts
      ? reinterpret_cast<const uint32_t *>(d_aux_in + off_item_counts)
      : nullptr;
  const uint8_t *d_msg_dst = d_aux_in + off_msg_dst;

  // Stage 1: GPU H2S + random-coefficient B-scalar reduction + 1-row B-MSM.
  if (err == cudaSuccess && use_fused_tiled_h2s) {
    constexpr uint32_t red_threads = 256u;
    cubbs_bbs_verify_reduce_b_scalars_kernel<red_threads>
        <<<2u, red_threads, 0, stream_b>>>(
            d_coef, d_dom, item->domain_stride, nullptr,
            nullptr, nullptr, item_count, uniform_msg_count,
            2u, d_agg_b_scalars_le, d_invalid);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      constexpr uint32_t h2s_t = 128u;
      const dim3 h2s_grid(static_cast<uint32_t>((uniform_msg_count + h2s_t - 1u) / h2s_t),
                          h2s_chunks, 1u);
      cubbs_bbs_verify_fused_h2s_reduce_tiled_kernel<<<h2s_grid, h2s_t, 0, stream_b>>>(
          d_messages, d_offsets, d_lengths, uniform_msg_len, d_coef,
          item_count, uniform_msg_count, h2s_chunks,
          d_msg_dst, item->message_dst.len,
          d_h2s_partials, d_agg_b_scalars_le);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess && h2s_chunks > 1u) {
      constexpr uint32_t p_t = 128u;
      const uint32_t p_blocks =
          static_cast<uint32_t>((uniform_msg_count + p_t - 1u) / p_t);
      cubbs_bbs_verify_reduce_tiled_partials_kernel<<<p_blocks, p_t, 0, stream_b>>>(
          d_h2s_partials, uniform_msg_count, h2s_chunks, d_agg_b_scalars_le);
      err = cudaGetLastError();
    }
  } else {
    if (err == cudaSuccess && total_messages > 0u) {
      constexpr uint32_t h2s_threads = 128u;
      const uint32_t h2s_blocks = static_cast<uint32_t>(
          (total_messages + h2s_threads - 1u) / h2s_threads);
      cubbs_bbs_verify_h2s_flat_to_le_kernel<<<h2s_blocks, h2s_threads, 0, stream_b>>>(
          d_messages, d_offsets, d_lengths, uniform_msg_len, total_messages,
          d_msg_dst, item->message_dst.len, d_msg_scalars_le);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
      constexpr uint32_t red_threads = 256u;
      cubbs_bbs_verify_reduce_b_scalars_kernel<red_threads>
          <<<static_cast<uint32_t>(term_count), red_threads, 0, stream_b>>>(
              d_coef, d_dom, item->domain_stride, d_msg_scalars_le,
              d_item_counts, d_item_starts, item_count, uniform_msg_count,
              term_count, d_agg_b_scalars_le, d_invalid);
      err = cudaGetLastError();
    }
  }
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_cached_common_msm_range(
        *cached, msm_transient, d_agg_b_scalars_le, term_count, 1u,
        d_b_output, stream_b);
  }

  // Stage 2: 100% GPU G1 decompression + Scott subgroup check on cache miss (0 CPU!).
  if (err == cudaSuccess && need_gpu_decode_a) {
    constexpr uint32_t dec_threads = 64u;
    const uint32_t dec_blocks = static_cast<uint32_t>((item_count + dec_threads - 1u) / dec_threads);
    cubbs_bbs_g1_uncompress_strided_kernel<<<dec_blocks, dec_threads, 0, stream_b>>>(
        d_a_comp, item_count, 48u, 1u, true, dual_ws.points, d_invalid);
    err = cudaGetLastError();
    if (err == cudaSuccess && !use_pippenger) {
      constexpr uint32_t table_threads = 128u;
      const uint32_t table_blocks = static_cast<uint32_t>(
          (item_count + table_threads - 1u) / table_threads);
      cubbs_bbs_proof_verify_radix16_tables_kernel
          <<<table_blocks, table_threads, 0u, stream_b>>>(
              dual_ws.points, item_count, dual_ws.tables);
      err = cudaGetLastError();
    }
  }

  // Stage 3: Dual A-MSM (sum r_i A_i, sum (-r_i e_i) A_i).
  if (err == cudaSuccess && use_pippenger) {
    constexpr uint32_t r_threads = 128u;
    const uint32_t r_blocks = static_cast<uint32_t>(
        (item_count + r_threads - 1u) / r_threads);
    cubbs_bbs_verify_pippenger_recode_kernel<<<r_blocks, r_threads, 0u, stream_b>>>(
        d_e, d_coef, item_count, d_digits, d_invalid);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      const dim3 bucket_grid(17u, 3u, chunks);
      cubbs_bbs_verify_pippenger_buckets_s8_kernel<<<bucket_grid, 128u, 0u, stream_b>>>(
          dual_ws.points, 0u, d_digits, 17u * item_count, item_count, chunks,
          d_win_chunk_sums);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
      cubbs_bbs_verify_pippenger_combine_s8_kernel<<<1u, 32u, 0u, stream_b>>>(
          d_win_chunk_sums, chunks, 3u, true, dual_ws.outputs);
      err = cudaGetLastError();
    }
  } else if (err == cudaSuccess) {
    constexpr uint32_t s_threads = 256u;
    const uint32_t s_blocks = static_cast<uint32_t>(
        (item_count + s_threads - 1u) / s_threads);
    cubbs_bbs_verify_build_dual_a_scalars_kernel<<<s_blocks, s_threads, 0u, stream_b>>>(
        d_e, d_coef, item_count, dual_ws.scalars, d_invalid);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      constexpr uint32_t threads = 128u;
      const size_t product_count = item_count * 2u;
      const uint32_t blocks = static_cast<uint32_t>(
          (product_count + threads - 1u) / threads);
      cubbs_bbs_g1_dual_msm_decompose_kernel<<<blocks, threads, 0u, stream_b>>>(
          dual_ws.scalars, product_count, dual_ws.components);
      err = cudaGetLastError();
      if (err == cudaSuccess) {
        cubbs_bbs_g1_dual_msm_same_points_table_mul_kernel<<<blocks, threads, 0u, stream_b>>>(
            dual_ws.tables, dual_ws.components, item_count, dual_ws.products);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) {
        const dim3 reduce_grid(1u, 2u, 1u);
        cubbs_bbs_g1_dual_msm_reduce_all_kernel<128u><<<reduce_grid, 128u, 0u, stream_b>>>(
            dual_ws.products, item_count, dual_ws.outputs);
        err = cudaGetLastError();
      }
    }
  }

  // Stage 4: Finalize (a_agg, S_g2 = sum r_i*(B_i - e_i*A_i)) and run GPU pairing!
  if (err == cudaSuccess) {
    cubbs_bbs_verify_finalize_pair_affines_kernel<<<1u, 32u, 0u, stream_b>>>(
        dual_ws.outputs, d_b_output, d_pair_affines, d_inf_flag);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    if (!unique_pk) {
      const int32_t pair_launch = cubbs_bbs_cuda_launch_pairing_eq_single_on_stream(
          &d_pair_affines[0], &d_pair_affines[1],
          pipe_ws.pk_lines, pipe_ws.g2_lines,
          item->public_key.data, item->g2_generator.data,
          d_eq_out, stream_b);
      if (pair_launch != CUBBS_BBS_GPU_SUCCESS) err = cudaErrorUnknown;
    } else {
      // Unique-PK GPU-resident randomized multi-pairing:
      //   prod_i e(r_i * A_i, pk_i) * e(-S_g2, G_2) == 1 (ONE FinalExp, 0 CPU G2 decode!).
      constexpr uint32_t sc_threads = 64u;
      const uint32_t sc_blocks = static_cast<uint32_t>((item_count + sc_threads - 1u) / sc_threads);
      cubbs_bbs_g1_scale_128bit_to_xyzz_kernel<<<sc_blocks, sc_threads, 0, stream_b>>>(
          dual_ws.points, 1u, d_coef, item_count, d_scaled_a_xyzz);
      err = cudaGetLastError();
      if (err == cudaSuccess) {
        cubbs_bbs_xyzz_to_plain_affine_batch64_kernel<<<sc_blocks, 64u, 0, stream_b>>>(
            d_scaled_a_xyzz, d_scaled_a_affines, item_count);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) {
        const int32_t pair_launch = cubbs_bbs_cuda_launch_pairing_unique_pk_on_stream(
            d_scaled_a_affines, &d_pair_affines[1], d_pks_comp,
            item->public_key.data, item_count,
            pipe_ws.g2_lines, item->g2_generator.data, d_invalid, d_eq_out, stream_b);
        if (pair_launch != CUBBS_BBS_GPU_SUCCESS) err = cudaErrorUnknown;
      }
    }
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.aux_out, d_inf_flag, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream_b);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.aux_out + sizeof(uint32_t), d_eq_out,
                          sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_b);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.invalid_flag, d_invalid, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream_b);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream_b);
  }

  msm_transient.free_async(stream_b);

  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
  if (*pipe_ws.invalid_flag != 0u) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  if (unique_pk) {
    cubbs_bbs_cuda_confirm_unique_pk_lines_cache(item->public_key.data, item_count);
  }
  if (need_gpu_decode_a) {
    dual_ws.point_key.resize(item_count * 48u);
    std::memcpy(&dual_ws.point_key[0], pipe_ws.aux_in + off_a_comp, item_count * 48u);
  }

  uint32_t inf_flag = 0u;
  std::memcpy(&inf_flag, pipe_ws.aux_out, sizeof(uint32_t));
  if (inf_flag != 0u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  const uint8_t eq_value = pipe_ws.aux_out[sizeof(uint32_t)];
  result->code = CUBBS_BBS_GPU_SUCCESS;
  result->value = eq_value;
  result->bytes_written = 0u;
  return CUBBS_BBS_GPU_SUCCESS;
}

#endif // CUBBS_BBS_FUSED_SIGN_VERIFY_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
