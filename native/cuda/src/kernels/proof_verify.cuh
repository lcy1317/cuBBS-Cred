#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

int32_t cubbs_bbs_cuda_proof_verify_msm_pair_matrix(
    const CuBbsBbsGpuByteSlice *common_points, size_t common_term_count,
    const CuBbsBbsGpuByteSlice *common_scalar_rows,
    const CuBbsBbsGpuByteSlice *proof_points,
    const CuBbsBbsGpuByteSlice *proof_scalars, size_t output_count,
    CuBbsBbsGpuByteBuffer *t1_outs, CuBbsBbsGpuByteBuffer *t2_outs,
    CuBbsBbsGpuResult *results) {
  using ProfileClock = std::chrono::steady_clock;
  const bool profile =
      std::getenv("CUBBS_BBS_GPU_PROOF_VERIFY_PAIR_PROFILE") != nullptr;
  const auto profile_started = ProfileClock::now();
  if (output_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  if (common_term_count == 0u || common_points == nullptr ||
      common_scalar_rows == nullptr || proof_points == nullptr ||
      proof_scalars == nullptr || t1_outs == nullptr || t2_outs == nullptr ||
      results == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t row = 0u; row < output_count; ++row) {
    if (t1_outs[row].data == nullptr || t1_outs[row].capacity < 48u ||
        t2_outs[row].data == nullptr || t2_outs[row].capacity < 48u) {
      return CUBBS_BBS_GPU_ERR_INSUFFICIENT_OUTPUT;
    }
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  const bool large_common =
      bbs_large_shape_kernel_supported(common_term_count);
  const bool universal_common =
      !large_common &&
      !bbs_specialized_common_points_supported(common_term_count) &&
      bbs_universal_tiled_common_supported(common_term_count);
  const bool tiled_common = large_common || universal_common;
  const bool affine_common =
      !tiled_common && common_term_count >= 18u && output_count > 512u &&
      bbs_homogeneous_affine_comb_enabled();
  int32_t status = get_fused_common_points_cache(
      common_points, common_term_count, &cached, tiled_common, affine_common,
      affine_common ? 9u : 0u, universal_common);
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      (cached->device_fixed_comb6 == nullptr &&
       cached->device_fixed_comb8 == nullptr &&
       cached->device_fixed_comb9 == nullptr &&
       cached->device_fixed_comb10 == nullptr &&
       cached->device_point_glv_multiples == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INTERNAL;
  }

  const size_t common_scalar_count = output_count * common_term_count;
  const size_t proof_point_count = output_count * 3u;
  const size_t proof_scalar_count = output_count * 4u;
  std::vector<uint8_t> common_scalars_le(common_scalar_count * 32u);
  std::vector<uint8_t> proof_scalars_le(proof_scalar_count * 32u);
  std::vector<cubbs_sppark_plain_affine_t> proof_bases(proof_point_count);
  const char *decode_workers_config =
      std::getenv("CUBBS_BBS_GPU_PROOF_VERIFY_DECODE_WORKERS");
  const size_t decode_workers_default = output_count < 128u
      ? 8u
      : (output_count == 128u
             ? 4u
             : (output_count < 256u
                    ? 1u
                    : (output_count < 512u ? 8u : 1u)));
  const size_t decode_workers = decode_workers_config != nullptr
      ? env_size_or_default("CUBBS_BBS_GPU_PROOF_VERIFY_DECODE_WORKERS", 0u)
      : decode_workers_default;
  const bool host_glv_split =
      env_flag_enabled("CUBBS_BBS_GPU_PROOF_VERIFY_HOST_GLV_SPLIT");
  status = cubbs_fused_decode_scalars_parallel(
      common_scalar_rows, common_scalar_count, common_scalars_le.data(),
      decode_workers);
  if (status == CUBBS_BBS_GPU_SUCCESS) {
    status = cubbs_fused_decode_scalars_parallel(
        proof_scalars, proof_scalar_count, proof_scalars_le.data(),
        decode_workers);
  }
  if (status == CUBBS_BBS_GPU_SUCCESS && host_glv_split) {
    const size_t split_workers = std::min<size_t>(
        decode_workers > 0u ? decode_workers
                            : blst_msm_worker_count(proof_scalar_count),
        proof_scalar_count);
    const size_t split_chunk =
        (proof_scalar_count + split_workers - 1u) / split_workers;
    cubbs_run_host_tasks(split_workers, [&](size_t worker) {
      const size_t begin = worker * split_chunk;
      const size_t end = std::min(proof_scalar_count, begin + split_chunk);
      for (size_t scalar = begin; scalar < end; ++scalar) {
        uint8_t *value = proof_scalars_le.data() + scalar * 32u;
        cubbs_bbs_scalar_div_by_z2_host(value, value);
      }
    });
  }
  const bool proof_points_cached = status == CUBBS_BBS_GPU_SUCCESS &&
      load_cached_g1_affines(proof_points, proof_point_count, proof_bases.data());
  if (status == CUBBS_BBS_GPU_SUCCESS && !proof_points_cached) {
    status = cubbs_bbs_blst_decode_g1_points(
        proof_points, proof_point_count,
        reinterpret_cast<uint8_t *>(proof_bases.data()),
        sizeof(cubbs_sppark_plain_affine_t));
  }
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;
  if (!proof_points_cached)
    cache_proof_pair_affines(proof_points, proof_bases.data(), output_count);
  const auto decode_finished = ProfileClock::now();

  const cudaStream_t stream =
      cached->stream != nullptr ? cached->stream : static_cast<cudaStream_t>(0);
  const char *overlap_configured =
      std::getenv("CUBBS_BBS_GPU_PROOF_VERIFY_MSM_OVERLAP");
  bool overlap = overlap_configured != nullptr
                     ? env_flag_enabled("CUBBS_BBS_GPU_PROOF_VERIFY_MSM_OVERLAP")
                     : output_count >= 128u;
  if (profile) overlap = false;
  const char *direct_variable_config =
      std::getenv("CUBBS_BBS_GPU_PROOF_VERIFY_DIRECT_MUL");
  const bool direct_variable = direct_variable_config != nullptr
      ? env_flag_enabled("CUBBS_BBS_GPU_PROOF_VERIFY_DIRECT_MUL")
      : output_count > 512u;
  const char *global_radix16_configured =
      std::getenv("CUBBS_BBS_GPU_PROOF_VERIFY_GLOBAL_RADIX16_TABLES");
  const bool global_radix16_tables = global_radix16_configured != nullptr
      ? env_flag_enabled("CUBBS_BBS_GPU_PROOF_VERIFY_GLOBAL_RADIX16_TABLES")
      : output_count >= 512u;
  if (overlap_configured == nullptr && global_radix16_tables) overlap = false;
  struct OverlapResources {
    cudaStream_t stream{nullptr};
    cudaEvent_t setup{nullptr};
    cudaEvent_t done{nullptr};
    void clear() {
      if (stream != nullptr) (void)cudaStreamSynchronize(stream);
      if (done != nullptr) (void)cudaEventDestroy(done);
      if (setup != nullptr) (void)cudaEventDestroy(setup);
      if (stream != nullptr) (void)cudaStreamDestroy(stream);
      stream = nullptr; setup = nullptr; done = nullptr;
    }
    ~OverlapResources() { clear(); }
  };
  static thread_local OverlapResources overlap_resources;
  auto &variable_stream = overlap_resources.stream;
  auto &setup_event = overlap_resources.setup;
  auto &variable_done_event = overlap_resources.done;
  if (overlap && variable_stream == nullptr) {
    cudaError_t setup_status = cudaStreamCreateWithFlags(
        &variable_stream, cudaStreamNonBlocking);
    if (setup_status == cudaSuccess)
      setup_status = cudaEventCreateWithFlags(&setup_event, cudaEventDisableTiming);
    if (setup_status == cudaSuccess)
      setup_status = cudaEventCreateWithFlags(&variable_done_event,
                                              cudaEventDisableTiming);
    if (setup_status != cudaSuccess) {
      overlap_resources.clear();
      overlap = false;
    }
  }
  uint8_t *device_common_scalars = nullptr;
  uint8_t *device_proof_scalars = nullptr;
  cubbs_sppark_plain_affine_t *device_proof_bases = nullptr;
  cubbs_sppark_bucket_t *device_proof_tables = nullptr;
  cubbs_sppark_bucket_t *device_t1 = nullptr;
  cubbs_sppark_bucket_t *device_t2 = nullptr;
  cubbs_sppark_bucket_t *device_t2_variable = nullptr;
  cubbs_sppark_bucket_t *device_common_partials = nullptr;
  cudaEvent_t profile_events[5] = {};
  if (profile) {
    for (cudaEvent_t &event : profile_events) {
      if (cudaEventCreate(&event) != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
    }
  }
  configure_cubbs_async_mempool();
  cudaError_t error = cudaMallocAsync(
      &device_common_scalars, common_scalars_le.size(), stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&device_proof_scalars, proof_scalars_le.size(), stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&device_proof_bases,
                            proof_point_count * sizeof(*device_proof_bases), stream);
  if (error == cudaSuccess && global_radix16_tables)
    error = cudaMallocAsync(
        &device_proof_tables,
        proof_point_count * 9u * sizeof(*device_proof_tables), stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&device_t1,
                            output_count * sizeof(*device_t1), stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&device_t2,
                            output_count * sizeof(*device_t2), stream);
  const size_t large_tile_terms = bbs_large_shape_tile_terms();
  const size_t large_tile_count = tiled_common
      ? (common_term_count + large_tile_terms - 1u) / large_tile_terms
      : 0u;
  if (error == cudaSuccess && tiled_common)
    error = cudaMallocAsync(
        &device_common_partials,
        output_count * large_tile_count * sizeof(*device_common_partials),
        stream);
  if (error == cudaSuccess && overlap)
    error = cudaMallocAsync(&device_t2_variable,
                            output_count * sizeof(*device_t2_variable), stream);
  if (error == cudaSuccess && profile)
    error = cudaEventRecord(profile_events[0], stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(device_common_scalars, common_scalars_le.data(),
                            common_scalars_le.size(), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(device_proof_scalars, proof_scalars_le.data(),
                            proof_scalars_le.size(), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(device_proof_bases, proof_bases.data(),
                            proof_point_count * sizeof(*device_proof_bases),
                            cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess && global_radix16_tables) {
    constexpr uint32_t table_threads = 128u;
    const uint32_t table_blocks = static_cast<uint32_t>(
        (proof_point_count + table_threads - 1u) / table_threads);
    cubbs_bbs_proof_verify_radix16_tables_kernel
        <<<table_blocks, table_threads, 0, stream>>>(
            device_proof_bases, proof_point_count, device_proof_tables);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && profile)
    error = cudaEventRecord(profile_events[1], stream);
  if (error == cudaSuccess && overlap)
    error = cudaEventRecord(setup_event, stream);
  if (error == cudaSuccess) {
    const uint32_t common_threads =
        fused_common_points_adaptive_threads(common_term_count, output_count);
    if (tiled_common) {
      constexpr uint32_t glv_threads = 128u;
      const dim3 tiled_grid(static_cast<uint32_t>(output_count),
                            static_cast<uint32_t>(large_tile_count));
      cubbs_bbs_g1_msm_large_glv_window4_tiled_kernel<glv_threads>
          <<<tiled_grid, glv_threads, 0, stream>>>(
              cached->device_point_glv_multiples, device_common_scalars,
              common_term_count, large_tile_terms, large_tile_count,
              output_count, device_common_partials);
      error = cudaGetLastError();
      if (error == cudaSuccess) {
        constexpr uint32_t reduce_threads = 32u;
        cubbs_bbs_g1_msm_large_reduce_kernel<reduce_threads>
            <<<static_cast<uint32_t>(output_count), reduce_threads, 0,
               stream>>>(device_common_partials, large_tile_count,
                         output_count, device_t2);
      }
    } else if (cached->device_fixed_affine_comb9 != nullptr) {
      error = cubbs_bbs_launch_affine_comb<9u>(
          cached->device_fixed_affine_comb9, device_common_scalars,
          common_term_count, output_count, common_threads, stream, device_t2);
    } else if (cached->device_fixed_affine_comb10 != nullptr) {
      error = cubbs_bbs_launch_affine_comb<10u>(
          cached->device_fixed_affine_comb10, device_common_scalars,
          common_term_count, output_count, common_threads, stream, device_t2);
    } else if (cached->device_fixed_affine_comb8 != nullptr) {
      error = cubbs_bbs_launch_affine_comb<8u>(
          cached->device_fixed_affine_comb8, device_common_scalars,
          common_term_count, output_count, common_threads, stream, device_t2);
    } else if (cached->device_fixed_comb9 != nullptr) {
      if (common_threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<32u>
            <<<static_cast<uint32_t>(output_count), 32u, 0, stream>>>(
                cached->device_fixed_comb9, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else if (common_threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<64u>
            <<<static_cast<uint32_t>(output_count), 64u, 0, stream>>>(
                cached->device_fixed_comb9, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb9_kernel<128u>
            <<<static_cast<uint32_t>(output_count), 128u, 0, stream>>>(
                cached->device_fixed_comb9, device_common_scalars,
                common_term_count, output_count, device_t2);
      }
    } else if (cached->device_fixed_comb10 != nullptr) {
      if (common_threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<32u>
            <<<static_cast<uint32_t>(output_count), 32u, 0, stream>>>(
                cached->device_fixed_comb10, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else if (common_threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<64u>
            <<<static_cast<uint32_t>(output_count), 64u, 0, stream>>>(
                cached->device_fixed_comb10, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb10_kernel<128u>
            <<<static_cast<uint32_t>(output_count), 128u, 0, stream>>>(
                cached->device_fixed_comb10, device_common_scalars,
                common_term_count, output_count, device_t2);
      }
    } else if (cached->device_fixed_comb8 != nullptr) {
      if (common_threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<32u>
            <<<static_cast<uint32_t>(output_count), 32u, 0, stream>>>(
                cached->device_fixed_comb8, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else if (common_threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<64u>
            <<<static_cast<uint32_t>(output_count), 64u, 0, stream>>>(
                cached->device_fixed_comb8, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb8_kernel<128u>
            <<<static_cast<uint32_t>(output_count), 128u, 0, stream>>>(
                cached->device_fixed_comb8, device_common_scalars,
                common_term_count, output_count, device_t2);
      }
    } else {
      if (common_threads == 32u) {
        cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<32u>
            <<<static_cast<uint32_t>(output_count), 32u, 0, stream>>>(
                cached->device_fixed_comb6, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else if (common_threads == 64u) {
        cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<64u>
            <<<static_cast<uint32_t>(output_count), 64u, 0, stream>>>(
                cached->device_fixed_comb6, device_common_scalars,
                common_term_count, output_count, device_t2);
      } else {
        cubbs_bbs_g1_msm_fixed_signed_comb6_kernel<128u>
            <<<static_cast<uint32_t>(output_count), 128u, 0, stream>>>(
                cached->device_fixed_comb6, device_common_scalars,
                common_term_count, output_count, device_t2);
      }
    }
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && profile)
    error = cudaEventRecord(profile_events[2], stream);
  if (error == cudaSuccess) {
    if (overlap) {
      error = cudaStreamWaitEvent(variable_stream, setup_event, 0u);
      if (error == cudaSuccess) {
        error = cubbs_bbs_launch_proof_verify_variable(
            direct_variable, host_glv_split, device_proof_bases,
            device_proof_tables, device_proof_scalars, output_count,
            device_t1, device_t2_variable, false, variable_stream);
      }
      if (error == cudaSuccess)
        error = cudaEventRecord(variable_done_event, variable_stream);
      if (error == cudaSuccess)
        error = cudaStreamWaitEvent(stream, variable_done_event, 0u);
      if (error == cudaSuccess) {
        constexpr uint32_t add_threads = 128u;
        const uint32_t add_blocks = static_cast<uint32_t>(
            (output_count + add_threads - 1u) / add_threads);
        cubbs_bbs_g1_add_projective_outputs_kernel
            <<<add_blocks, add_threads, 0, stream>>>(
                device_t2, device_t2_variable, output_count);
        error = cudaGetLastError();
      }
    } else {
      error = cubbs_bbs_launch_proof_verify_variable(
          direct_variable, host_glv_split, device_proof_bases,
          device_proof_tables, device_proof_scalars, output_count,
          device_t1, device_t2, true, stream);
    }
  }
  if (error == cudaSuccess && profile)
    error = cudaEventRecord(profile_events[3], stream);

  std::vector<cubbs_sppark_bucket_t> host_t1(output_count);
  std::vector<cubbs_sppark_bucket_t> host_t2(output_count);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(host_t1.data(), device_t1,
                            output_count * sizeof(host_t1[0]),
                            cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(host_t2.data(), device_t2,
                            output_count * sizeof(host_t2[0]),
                            cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess && profile)
    error = cudaEventRecord(profile_events[4], stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);

  float h2d_ms = 0.0f;
  float common_ms = 0.0f;
  float variable_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (error == cudaSuccess && profile) {
    cudaEventElapsedTime(&h2d_ms, profile_events[0], profile_events[1]);
    cudaEventElapsedTime(&common_ms, profile_events[1], profile_events[2]);
    cudaEventElapsedTime(&variable_ms, profile_events[2], profile_events[3]);
    cudaEventElapsedTime(&d2h_ms, profile_events[3], profile_events[4]);
  }

  if (device_common_scalars != nullptr) cudaFreeAsync(device_common_scalars, stream);
  if (device_proof_scalars != nullptr) cudaFreeAsync(device_proof_scalars, stream);
  if (device_proof_bases != nullptr) cudaFreeAsync(device_proof_bases, stream);
  if (device_proof_tables != nullptr) cudaFreeAsync(device_proof_tables, stream);
  if (device_t1 != nullptr) cudaFreeAsync(device_t1, stream);
  if (device_t2 != nullptr) cudaFreeAsync(device_t2, stream);
  if (device_t2_variable != nullptr) cudaFreeAsync(device_t2_variable, stream);
  if (device_common_partials != nullptr)
    cudaFreeAsync(device_common_partials, stream);
  if (profile) {
    for (cudaEvent_t event : profile_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
  }
  if (error != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  std::vector<cubbs_sppark_point_t> projectives(output_count * 2u);
  std::vector<const void *> projective_ptrs(output_count * 2u);
  std::vector<uint8_t *> compressed_outs(output_count * 2u);
  for (size_t row = 0u; row < output_count; ++row) {
    projectives[row * 2u] = static_cast<cubbs_sppark_point_t>(host_t1[row]);
    projectives[row * 2u + 1u] = static_cast<cubbs_sppark_point_t>(host_t2[row]);
    projective_ptrs[row * 2u] = &projectives[row * 2u];
    projective_ptrs[row * 2u + 1u] = &projectives[row * 2u + 1u];
    compressed_outs[row * 2u] = t1_outs[row].data;
    compressed_outs[row * 2u + 1u] = t2_outs[row].data;
  }
  const auto compression_started = ProfileClock::now();
  status = cubbs_bbs_blst_compress_g1_projective_batch(
      projective_ptrs.data(), compressed_outs.data(), projective_ptrs.size());
  const auto compression_finished = ProfileClock::now();
  if (profile) {
    const auto host_ms = [](auto started, auto finished) {
      return std::chrono::duration<double, std::milli>(finished - started).count();
    };
    std::fprintf(stderr,
                 "cubbs_proof_verify_pair_profile items=%zu decode_ms=%.6f "
                 "h2d_ms=%.6f common_ms=%.6f variable_ms=%.6f d2h_ms=%.6f "
                 "compress_ms=%.6f total_ms=%.6f\n",
                 output_count, host_ms(profile_started, decode_finished), h2d_ms,
                 common_ms, variable_ms, d2h_ms,
                 host_ms(compression_started, compression_finished),
                 host_ms(profile_started, compression_finished));
  }
  for (size_t row = 0u; row < output_count; ++row) {
    set_result_route(results, row, status, status == CUBBS_BBS_GPU_SUCCESS ? 96u : 0u,
                     universal_common
                         ? CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL
                         : CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

#endif
