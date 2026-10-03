#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

static int32_t compute_g1_msm_fused_common_points_with_cuda(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results,
    const CuBbsBbsGpuByteSlice *variable_points = nullptr,
    const CuBbsBbsGpuByteSlice *variable_scalars = nullptr,
    bool large_window4 = false, bool universal_window4 = false) {
  if (output_count == 0u) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  const bool affine_common =
      !large_window4 && term_count >= 18u && output_count > 512u &&
      bbs_homogeneous_affine_comb_enabled();
  int32_t status = get_fused_common_points_cache(
      points, term_count, &cached, large_window4, affine_common, 0u,
      universal_window4);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t row = 0; row < output_count; ++row) {
      set_result(results, row, status, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t total_scalars = output_count * term_count;
  const size_t scalar_bytes = total_scalars * 32u;
  const bool has_device_row_scale =
      scalar_rows_be != nullptr &&
      scalar_row_stride == (term_count + 1u) * 32u;
  const char *packed_decode_config =
      std::getenv("CUBBS_BBS_GPU_PACKED_SCALAR_DECODE");
  const bool gpu_packed_scalar_decode =
      scalar_rows_be != nullptr &&
      (packed_decode_config != nullptr
           ? env_flag_enabled("CUBBS_BBS_GPU_PACKED_SCALAR_DECODE")
           : total_scalars >= 8192u && variable_points == nullptr);
  const size_t row_scale_bytes = has_device_row_scale ? output_count * 32u : 0u;
  const size_t scalar_buffer_bytes = scalar_bytes + row_scale_bytes;
  const size_t packed_scalar_bytes =
      gpu_packed_scalar_decode ? output_count * scalar_row_stride : 0u;
  // Outputs are projective (xyz) points; the host compress step converts and
  // encodes them via blst_p1_compress, which is already highly optimized
  // (assembly-accelerated to-affine + compress). A device-side affine
  // conversion was prototyped but did not beat blst's host compress on this
  // target, so the projective kernel + parallel host compress is kept.
  const size_t output_bytes = output_count * sizeof(cubbs_sppark_bucket_t);
  auto &host_workspace = cubbs_fused_pinned_host_workspace();
  if (host_workspace.ensure(
          gpu_packed_scalar_decode ? 0u : scalar_buffer_bytes,
          output_count) != cudaSuccess) {
    for (size_t row = 0u; row < output_count; ++row)
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }
  uint8_t *decoded_scalars = host_workspace.scalars;
  cubbs_sppark_bucket_t *host_outputs = host_workspace.outputs;

  // Decode scalars in parallel (overlaps with any prior group still on GPU).
  if (!gpu_packed_scalar_decode) {
    status = scalar_rows_be != nullptr
        ? cubbs_fused_decode_packed_scalars_parallel(
              scalar_rows_be + (has_device_row_scale ? 32u : 0u),
              output_count, term_count, scalar_row_stride, decoded_scalars)
        : cubbs_fused_decode_scalars_parallel(
              scalar_rows, total_scalars, decoded_scalars);
    if (status == CUBBS_BBS_GPU_SUCCESS && has_device_row_scale) {
      status = cubbs_fused_decode_packed_scalars_parallel(
          scalar_rows_be, output_count, 1u, scalar_row_stride,
          decoded_scalars + scalar_bytes);
    }
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      for (size_t row = 0; row < output_count; ++row) {
        set_result(results, row, status, 0);
      }
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  const bool large_glv = large_window4 &&
                         cached->device_point_glv_multiples != nullptr &&
                         !bbs_large_shape_pippenger_enabled() &&
                         !bbs_large_shape_window8_enabled();
  const bool large_glv_wnaf =
      large_glv && cached->device_point_glv_wnaf_multiples != nullptr;
  const uint8_t result_route = universal_window4
      ? CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL
      : (large_window4 ? CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE
                       : CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS);

  const bool has_variable_terms =
      variable_points != nullptr && variable_scalars != nullptr;
  std::vector<cubbs_sppark_plain_affine_t> variable_bases;
  std::vector<uint8_t> variable_scalar_le;
  if (has_variable_terms) {
    for (size_t row = 0; row < output_count; ++row) {
      if (variable_points[row].data == nullptr ||
          variable_points[row].len != 48u ||
          variable_scalars[row].data == nullptr ||
          variable_scalars[row].len != 32u) {
        for (size_t i = 0; i < output_count; ++i) {
          set_result(results, i, CUBBS_BBS_GPU_ERR_INVALID_INPUT, 0);
        }
        return CUBBS_BBS_GPU_SUCCESS;
      }
    }
    variable_bases.resize(output_count);
    status = cubbs_bbs_blst_decode_g1_points(
        variable_points, output_count,
        reinterpret_cast<uint8_t *>(variable_bases.data()),
        sizeof(cubbs_sppark_plain_affine_t));
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      for (size_t row = 0; row < output_count; ++row) {
        set_result(results, row, status, 0);
      }
      return CUBBS_BBS_GPU_SUCCESS;
    }
    variable_scalar_le.resize(output_count * 32u);
    status = cubbs_fused_decode_scalars_parallel(
        variable_scalars, output_count, variable_scalar_le.data());
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      for (size_t row = 0; row < output_count; ++row) {
        set_result(results, row, status, 0);
      }
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  // The cached point tables (device_point_multiples8) and the per-point-set
  // stream are read-only / owned by the cache and safe to share across
  // concurrent calls. The per-call scalar and output device buffers are
  // transient: allocating them per call (instead of holding the cache mutex
  // across the whole launch+sync) lets multiple threads dispatch the same
  // point set concurrently and overlap their kernel/compress tails.
  // Calls sharing one point table may run concurrently.  Keep their command
  // sequences on per-calling-thread streams so one caller's synchronize does
  // not drain or phase-lock every pipeline using the same cached points.
  const cudaStream_t stream = host_workspace.execution_stream(
      cached->stream != nullptr ? cached->stream : static_cast<cudaStream_t>(0));

  uint8_t *device_scalars = nullptr;
  uint8_t *device_packed_scalars = nullptr;
  uint32_t *device_scalar_invalid = nullptr;
  int8_t *device_large_wnaf_digits = nullptr;
  cubbs_sppark_bucket_t *device_outputs = nullptr;
  cubbs_sppark_bucket_t *device_large_partials = nullptr;
  cubbs_sppark_bucket_t *device_large_bucket_partials = nullptr;
  cubbs_sppark_bucket_t *device_large_window_partials = nullptr;
  cubbs_sppark_plain_affine_t *device_variable_bases = nullptr;
  uint8_t *device_variable_scalars = nullptr;
  cubbs_sppark_bucket_t *device_variable_tables = nullptr;
  const bool variable_radix16_tables =
      has_variable_terms &&
      output_count >= env_size_or_default(
                          "CUBBS_BBS_GPU_VARIABLE_RADIX16_MIN_BATCH", 128u);
  // cudaMallocAsync draws from a per-stream memory pool, so concurrent calls
  // reuse previously freed allocations without a global lock. This removes
  // both the cache mutex and the per-call cudaMalloc/cudaFree overhead.
  configure_cubbs_async_mempool();
  cudaError_t cuda_status =
      cudaMallocAsync(&device_scalars, scalar_buffer_bytes, stream);
  if (cuda_status == cudaSuccess && gpu_packed_scalar_decode) {
    cuda_status = cudaMallocAsync(
        &device_packed_scalars, packed_scalar_bytes, stream);
  }
  if (cuda_status == cudaSuccess && gpu_packed_scalar_decode) {
    cuda_status = cudaMallocAsync(&device_scalar_invalid,
                                  sizeof(*device_scalar_invalid), stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMallocAsync(&device_outputs, output_bytes, stream);
  }
  if (cuda_status == cudaSuccess && large_glv_wnaf) {
    cuda_status = cudaMallocAsync(
        &device_large_wnaf_digits, total_scalars * 260u, stream);
  }
  const size_t large_tile_terms = bbs_large_shape_tile_terms();
  const size_t large_tile_count =
      large_window4 ? (term_count + large_tile_terms - 1u) / large_tile_terms
                    : 0u;
  const bool large_pippenger =
      large_window4 && bbs_large_shape_pippenger_enabled();
  const bool large_window8 = large_window4 && !large_pippenger &&
                             cached->device_point_multiples8 != nullptr;
  if (cuda_status == cudaSuccess && large_window4 && !large_pippenger &&
      !large_window8) {
    cuda_status = cudaMallocAsync(
        &device_large_partials,
        output_count * large_tile_count * sizeof(*device_large_partials),
        stream);
  }
  if (cuda_status == cudaSuccess && large_pippenger) {
    cuda_status = cudaMallocAsync(
        &device_large_bucket_partials,
        output_count * 64u * large_tile_count * 15u *
            sizeof(*device_large_bucket_partials),
        stream);
  }
  if (cuda_status == cudaSuccess && large_pippenger) {
    cuda_status = cudaMallocAsync(
        &device_large_window_partials,
        output_count * 64u * sizeof(*device_large_window_partials), stream);
  }
  if (cuda_status == cudaSuccess && has_variable_terms) {
    cuda_status = cudaMallocAsync(
        &device_variable_bases,
        output_count * sizeof(*device_variable_bases), stream);
  }
  if (cuda_status == cudaSuccess && has_variable_terms) {
    cuda_status = cudaMallocAsync(
        &device_variable_scalars, output_count * 32u, stream);
  }
  if (cuda_status == cudaSuccess && variable_radix16_tables) {
    cuda_status = cudaMallocAsync(
        &device_variable_tables,
        output_count * 9u * sizeof(*device_variable_tables), stream);
  }
  bool scalar_input_invalid = false;
  if (cuda_status == cudaSuccess && gpu_packed_scalar_decode) {
    cuda_status = cudaMemcpyAsync(
        device_packed_scalars, scalar_rows_be, packed_scalar_bytes,
        cudaMemcpyHostToDevice, stream);
  }
  if (cuda_status == cudaSuccess && gpu_packed_scalar_decode) {
    cuda_status =
        cudaMemsetAsync(device_scalar_invalid, 0, sizeof(*device_scalar_invalid),
                        stream);
  }
  if (cuda_status == cudaSuccess && gpu_packed_scalar_decode) {
    cuda_status = cubbs_bbs_launch_packed_scalar_decode(
        device_packed_scalars, output_count, term_count, scalar_row_stride,
        has_device_row_scale ? 32u : 0u, device_scalars,
        device_scalar_invalid, stream);
  }
  if (cuda_status == cudaSuccess && gpu_packed_scalar_decode &&
      has_device_row_scale) {
    cuda_status = cubbs_bbs_launch_packed_scalar_decode(
        device_packed_scalars, output_count, 1u, scalar_row_stride, 0u,
        device_scalars + scalar_bytes, device_scalar_invalid, stream);
  }
  uint32_t host_scalar_invalid = 0u;
  const bool eager_scalar_validation =
      gpu_packed_scalar_decode && has_device_row_scale;
  if (cuda_status == cudaSuccess && eager_scalar_validation) {
    cuda_status = cudaMemcpyAsync(
        &host_scalar_invalid, device_scalar_invalid,
        sizeof(host_scalar_invalid), cudaMemcpyDeviceToHost, stream);
  }
  if (cuda_status == cudaSuccess && eager_scalar_validation) {
    cuda_status = cudaStreamSynchronize(stream);
    scalar_input_invalid = host_scalar_invalid != 0u;
    if (scalar_input_invalid) cuda_status = cudaErrorInvalidValue;
  }
  if (cuda_status == cudaSuccess && !gpu_packed_scalar_decode) {
    cuda_status = cudaMemcpyAsync(device_scalars,
                                  decoded_scalars, scalar_buffer_bytes,
                                  cudaMemcpyHostToDevice, stream);
  }
  if (cuda_status == cudaSuccess && has_device_row_scale) {
    constexpr uint32_t scale_threads = 256u;
    const uint32_t scale_blocks = static_cast<uint32_t>(
        (total_scalars + scale_threads - 1u) / scale_threads);
    cubbs_bbs_scale_scalar_rows_kernel<<<scale_blocks, scale_threads, 0, stream>>>(
        device_scalars, device_scalars + scalar_bytes, term_count,
        total_scalars);
    cuda_status = cudaGetLastError();
  }
  if (cuda_status == cudaSuccess && large_glv_wnaf) {
    constexpr uint32_t recode_threads = 128u;
    const uint32_t recode_blocks = static_cast<uint32_t>(
        (total_scalars + recode_threads - 1u) / recode_threads);
    cubbs_bbs_glv_wnaf8_recode_kernel
        <<<recode_blocks, recode_threads, 0, stream>>>(
            device_scalars, total_scalars, device_large_wnaf_digits);
    cuda_status = cudaGetLastError();
  }
  if (cuda_status == cudaSuccess && has_variable_terms) {
    cuda_status = cudaMemcpyAsync(
        device_variable_bases, variable_bases.data(),
        output_count * sizeof(*device_variable_bases),
        cudaMemcpyHostToDevice, stream);
  }
  if (cuda_status == cudaSuccess && has_variable_terms) {
    cuda_status = cudaMemcpyAsync(
        device_variable_scalars, variable_scalar_le.data(),
        output_count * 32u, cudaMemcpyHostToDevice, stream);
  }
  if (cuda_status == cudaSuccess && variable_radix16_tables) {
    constexpr uint32_t table_threads = 128u;
    const uint32_t table_blocks = static_cast<uint32_t>(
        (output_count + table_threads - 1u) / table_threads);
    cubbs_bbs_proof_verify_radix16_tables_kernel
        <<<table_blocks, table_threads, 0, stream>>>(
            device_variable_bases, output_count, device_variable_tables);
    cuda_status = cudaGetLastError();
  }

  // Adaptive thread count: scales the warp footprint to the VC shape's term
  // count so small common shapes (e.g. proof-T2's 8 terms) don't waste 7/8
  // of the threads, while large shapes (130 terms) still get 128 threads.
  const uint32_t threads =
      fused_common_points_adaptive_threads(term_count, output_count);

  // Pipelined chunked launch: split the output range into chunks so the host
  // can compress chunk K-1 (after its D2H) while chunk K's kernel is still
  // running on the device. This keeps the GPU busy across the whole batch
  // instead of idling during the serial host compress tail. The chunk size is
  // chosen so each kernel launch has enough blocks to saturate the GPU while
  // keeping the host compress of one chunk comparable to one kernel+copy.
  // For small batches (<= chunk) this collapses to a single launch+compress,
  // same as before. CUBBS_BBS_GPU_FUSED_CHUNK_SIZE overrides (0 = auto).
  // Several host streams already fill the device. Keeping up to 2048 rows in
  // one launch avoids short kernels while preserving kernel/D2H/compress
  // overlap for huge mixed-shape buckets.
  const size_t auto_chunk = std::min<size_t>(output_count, 2048u);
  const size_t chunk_size = std::max<size_t>(
      1u, env_size_or_default("CUBBS_BBS_GPU_FUSED_CHUNK_SIZE", auto_chunk));

  auto launch_kernel = [&](size_t row_begin, size_t row_end) -> cudaError_t {
    const size_t n = row_end - row_begin;
    if (n == 0u) return cudaSuccess;
    const uint32_t blocks = static_cast<uint32_t>(n);
    const size_t scalar_offset = row_begin * term_count * 32u;
    const size_t out_offset = row_begin;
    const auto add_variable_term = [&](cudaError_t status) -> cudaError_t {
      if (status != cudaSuccess || !has_variable_terms) return status;
      constexpr uint32_t variable_threads = 128u;
      const uint32_t variable_blocks = static_cast<uint32_t>(
          (n + variable_threads - 1u) / variable_threads);
      if (device_variable_tables != nullptr) {
        cubbs_bbs_g1_add_variable_term_radix16_kernel
            <<<variable_blocks, variable_threads, 0, stream>>>(
                device_variable_tables + row_begin * 9u,
                device_variable_scalars + (row_begin * 32u), n,
                device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_add_variable_term_kernel
            <<<variable_blocks, variable_threads, 0, stream>>>(
                device_variable_bases + row_begin,
                device_variable_scalars + (row_begin * 32u), n,
                device_outputs + out_offset);
      }
      return cudaGetLastError();
    };
    if (term_count >= 18u &&
        cached->device_fixed_affine_comb9 != nullptr) {
      return add_variable_term(cubbs_bbs_launch_affine_comb<9u>(
          cached->device_fixed_affine_comb9,
          device_scalars + scalar_offset, term_count, n, threads, stream,
          device_outputs + out_offset));
    }
    if (term_count >= 18u &&
        cached->device_fixed_affine_comb10 != nullptr) {
      return add_variable_term(cubbs_bbs_launch_affine_comb<10u>(
          cached->device_fixed_affine_comb10,
          device_scalars + scalar_offset, term_count, n, threads, stream,
          device_outputs + out_offset));
    }
    if (term_count >= 18u &&
        cached->device_fixed_affine_comb8 != nullptr) {
      return add_variable_term(cubbs_bbs_launch_affine_comb<8u>(
          cached->device_fixed_affine_comb8,
          device_scalars + scalar_offset, term_count, n, threads, stream,
          device_outputs + out_offset));
    }
    if (large_glv) {
      constexpr uint32_t glv_threads = 128u;
      const dim3 tiled_grid(blocks, static_cast<uint32_t>(large_tile_count));
      if (large_glv_wnaf) {
        cubbs_bbs_g1_msm_large_glv_wnaf8_tiled_kernel<glv_threads>
            <<<tiled_grid, glv_threads, 0, stream>>>(
                cached->device_point_glv_wnaf_multiples,
                device_large_wnaf_digits + row_begin * term_count * 260u,
                term_count, large_tile_terms,
                large_tile_count, n,
                device_large_partials + row_begin * large_tile_count);
      } else {
        cubbs_bbs_g1_msm_large_glv_window4_tiled_kernel<glv_threads>
            <<<tiled_grid, glv_threads, 0, stream>>>(
                cached->device_point_glv_multiples,
                device_scalars + scalar_offset, term_count, large_tile_terms,
                large_tile_count, n,
                device_large_partials + row_begin * large_tile_count);
      }
      cudaError_t large_status = cudaGetLastError();
      if (large_status != cudaSuccess) return large_status;
      constexpr uint32_t reduce_threads = 32u;
      cubbs_bbs_g1_msm_large_reduce_kernel<reduce_threads>
          <<<blocks, reduce_threads, 0, stream>>>(
              device_large_partials + row_begin * large_tile_count,
              large_tile_count, n, device_outputs + out_offset);
    } else if (large_window8) {
      constexpr uint32_t large_threads = 256u;
      cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<large_threads>
          <<<blocks, large_threads, 0, stream>>>(
              cached->device_point_multiples8,
              device_scalars + scalar_offset, term_count, n,
              device_outputs + out_offset);
    } else if (large_pippenger) {
      constexpr uint32_t bucket_threads = 128u;
      const dim3 bucket_grid(
          blocks, 64u, static_cast<uint32_t>(large_tile_count));
      cubbs_bbs_g1_msm_large_pippenger_locked_buckets_kernel<bucket_threads>
          <<<bucket_grid, bucket_threads, 0, stream>>>(
              cached->device_points, device_scalars + scalar_offset,
              term_count, large_tile_terms, large_tile_count, n,
              device_large_bucket_partials +
                  row_begin * 64u * large_tile_count * 15u);
      cudaError_t large_status = cudaGetLastError();
      if (large_status != cudaSuccess) return large_status;
      constexpr uint32_t window_threads = 32u;
      const dim3 window_grid(blocks, 64u);
      cubbs_bbs_g1_msm_large_pippenger_windows_kernel<window_threads>
          <<<window_grid, window_threads, 0, stream>>>(
              device_large_bucket_partials +
                  row_begin * 64u * large_tile_count * 15u,
              large_tile_count, n,
              device_large_window_partials + row_begin * 64u);
      large_status = cudaGetLastError();
      if (large_status != cudaSuccess) return large_status;
      cubbs_bbs_g1_msm_large_pippenger_rows_kernel<<<blocks, 1u, 0, stream>>>(
          device_large_window_partials + row_begin * 64u, n,
          device_outputs + out_offset);
    } else if (large_window4) {
      constexpr uint32_t tile_threads = 128u;
      const dim3 tiled_grid(blocks, static_cast<uint32_t>(large_tile_count));
      cubbs_bbs_g1_msm_large_window4_tiled_kernel<tile_threads>
          <<<tiled_grid, tile_threads, 0, stream>>>(
              cached->device_point_multiples,
              device_scalars + scalar_offset, term_count, large_tile_terms,
              large_tile_count, n,
              device_large_partials + row_begin * large_tile_count);
      cudaError_t large_status = cudaGetLastError();
      if (large_status != cudaSuccess) return large_status;
      constexpr uint32_t reduce_threads = 32u;
      cubbs_bbs_g1_msm_large_reduce_kernel<reduce_threads>
          <<<blocks, reduce_threads, 0, stream>>>(
              device_large_partials + row_begin * large_tile_count,
              large_tile_count, n, device_outputs + out_offset);
    } else if (cached->device_fixed_comb9 != nullptr) {
      if (threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<32u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb9, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<64u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb9, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<128u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb9, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      }
    } else if (cached->device_fixed_comb10 != nullptr) {
      if (threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<32u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb10, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<64u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb10, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<128u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb10, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      }
    } else if (cached->device_fixed_comb8 != nullptr) {
      if (threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<32u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb8, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<64u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb8, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<128u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb8, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      }
    } else if (cached->device_fixed_comb6 != nullptr) {
      if (threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<32u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb6, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<64u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb6, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<128u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb6, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      }
    } else if (cached->device_fixed_comb5 != nullptr) {
      if (threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb5_kernel<32u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb5, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb5_kernel<64u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb5, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb5_kernel<128u>
            <<<blocks, threads, 0, stream>>>(
                cached->device_fixed_comb5, device_scalars + scalar_offset,
                term_count, n, device_outputs + out_offset);
      }
    } else if (cached->device_fixed_comb4 != nullptr) {
      if (threads == 32u) {
        cubbs_bbs_g1_msm_fixed_comb4_kernel<32u><<<blocks, threads, 0, stream>>>(
            cached->device_fixed_comb4, device_scalars + scalar_offset,
            term_count, n, device_outputs + out_offset);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_fixed_comb4_kernel<64u><<<blocks, threads, 0, stream>>>(
            cached->device_fixed_comb4, device_scalars + scalar_offset,
            term_count, n, device_outputs + out_offset);
      } else {
        cubbs_bbs_g1_msm_fixed_comb4_kernel<128u><<<blocks, threads, 0, stream>>>(
            cached->device_fixed_comb4, device_scalars + scalar_offset,
            term_count, n, device_outputs + out_offset);
      }
    } else if (threads == 32u) {
      cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<32u>
          <<<blocks, threads, 0, stream>>>(
              cached->device_point_multiples8,
              device_scalars + scalar_offset, term_count, n,
              device_outputs + out_offset);
    } else if (threads == 64u) {
      cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<64u>
          <<<blocks, threads, 0, stream>>>(
              cached->device_point_multiples8,
              device_scalars + scalar_offset, term_count, n,
              device_outputs + out_offset);
    } else if (threads == 256u) {
      cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<256u>
          <<<blocks, threads, 0, stream>>>(
              cached->device_point_multiples8,
              device_scalars + scalar_offset, term_count, n,
              device_outputs + out_offset);
    } else {
      cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<128u>
          <<<blocks, threads, 0, stream>>>(
              cached->device_point_multiples8,
              device_scalars + scalar_offset, term_count, n,
              device_outputs + out_offset);
    }
    return add_variable_term(cudaGetLastError());
  };

  auto compress_range = [&](size_t row_begin, size_t row_end) {
    const size_t worker_count = fused_compress_worker_count(row_end - row_begin);
    if (worker_count <= 1u) {
      for (size_t row = row_begin; row < row_end; ++row) {
        const cubbs_sppark_point_t projective =
            static_cast<cubbs_sppark_point_t>(host_outputs[row]);
        const int32_t rs = cubbs_bbs_blst_compress_g1_projective(
            &projective, point_outs[row].data);
        if (rs == CUBBS_BBS_GPU_SUCCESS) {
          set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                           result_route);
        } else {
          set_result(results, row, rs, 0);
        }
      }
      return;
    }
    std::atomic<size_t> next_row{row_begin};
    cubbs_run_host_tasks(worker_count, [&](size_t) {
      while (true) {
        const size_t row = next_row.fetch_add(1u);
        if (row >= row_end) break;
        const cubbs_sppark_point_t projective =
            static_cast<cubbs_sppark_point_t>(host_outputs[row]);
        const int32_t rs = cubbs_bbs_blst_compress_g1_projective(
            &projective, point_outs[row].data);
        if (rs == CUBBS_BBS_GPU_SUCCESS) {
          set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                           result_route);
        } else {
          set_result(results, row, rs, 0);
        }
      }
    });
  };

  if (cuda_status == cudaSuccess) {
    // Walk chunks. For each chunk K>0: launch K's kernel, then D2H+compress
    // chunk K-1 (overlapping host work with K's kernel). This requires the
    // per-chunk D2H to be ordered after its own kernel on the stream, which a
    // single stream guarantees. We sync once per chunk to read back.
    size_t prev_begin = 0u;
    size_t prev_end = 0u;
    bool have_prev = false;
    for (size_t begin = 0u; begin < output_count; begin += chunk_size) {
      const size_t end = std::min(output_count, begin + chunk_size);
      // Overlap: compress the previous chunk on the host while this chunk's
      // kernel runs on the device. The previous chunk's D2H already completed
      // (we synced after it), so host_outputs[prev] is valid.
      std::thread compress_thread;
      if (have_prev) {
        compress_thread = std::thread(compress_range, prev_begin, prev_end);
      }
      cuda_status = launch_kernel(begin, end);
      if (cuda_status == cudaSuccess) {
        const size_t bytes = (end - begin) * sizeof(cubbs_sppark_bucket_t);
        cuda_status = cudaMemcpyAsync(host_outputs + begin,
                                      device_outputs + begin, bytes,
                                      cudaMemcpyDeviceToHost, stream);
      }
      // The first chunk already needs a stream synchronization for its output.
      // Read the canonical-Fr validation flag at the same boundary instead of
      // inserting a separate decode-only synchronization before the MSM.
      if (cuda_status == cudaSuccess && gpu_packed_scalar_decode &&
          !eager_scalar_validation && begin == 0u) {
        cuda_status = cudaMemcpyAsync(
            &host_scalar_invalid, device_scalar_invalid,
            sizeof(host_scalar_invalid), cudaMemcpyDeviceToHost, stream);
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaStreamSynchronize(stream);
      }
      if (cuda_status == cudaSuccess && gpu_packed_scalar_decode &&
          !eager_scalar_validation && begin == 0u) {
        scalar_input_invalid = host_scalar_invalid != 0u;
      }
      if (have_prev) compress_thread.join();
      if (cuda_status != cudaSuccess || scalar_input_invalid) break;
      prev_begin = begin;
      prev_end = end;
      have_prev = true;
    }
    // Compress the final chunk (no following kernel to overlap with).
    if (cuda_status == cudaSuccess && have_prev) {
      compress_range(prev_begin, prev_end);
    }
  }

  if (device_scalars != nullptr) {
    (void)cudaFreeAsync(device_scalars, stream);
  }
  if (device_packed_scalars != nullptr) {
    (void)cudaFreeAsync(device_packed_scalars, stream);
  }
  if (device_scalar_invalid != nullptr) {
    (void)cudaFreeAsync(device_scalar_invalid, stream);
  }
  if (device_large_wnaf_digits != nullptr) {
    (void)cudaFreeAsync(device_large_wnaf_digits, stream);
  }
  if (device_outputs != nullptr) {
    (void)cudaFreeAsync(device_outputs, stream);
  }
  if (device_large_partials != nullptr) {
    (void)cudaFreeAsync(device_large_partials, stream);
  }
  if (device_large_bucket_partials != nullptr) {
    (void)cudaFreeAsync(device_large_bucket_partials, stream);
  }
  if (device_large_window_partials != nullptr) {
    (void)cudaFreeAsync(device_large_window_partials, stream);
  }
  if (device_variable_bases != nullptr) {
    (void)cudaFreeAsync(device_variable_bases, stream);
  }
  if (device_variable_scalars != nullptr) {
    (void)cudaFreeAsync(device_variable_scalars, stream);
  }
  if (device_variable_tables != nullptr) {
    (void)cudaFreeAsync(device_variable_tables, stream);
  }
  if (scalar_input_invalid) {
    for (size_t row = 0u; row < output_count; ++row) {
      set_result(results, row, CUBBS_BBS_GPU_ERR_INVALID_INPUT, 0u);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }
  if (cuda_status != cudaSuccess) {
    for (size_t row = 0; row < output_count; ++row) {
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}
#endif
