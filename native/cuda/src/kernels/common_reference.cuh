#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

static int32_t compute_g1_msm_fused_common_points_with_naive_cuda(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  if (output_count == 0u) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  int32_t status =
      get_fused_common_points_cache(points, term_count, &cached, false, false);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t row = 0; row < output_count; ++row) {
      set_result(results, row, status, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::vector<CuBbsBbsGpuByteSlice> packed_scalar_slices;
  if (scalar_rows_be != nullptr) {
    packed_scalar_slices.resize(output_count * term_count);
    for (size_t row = 0; row < output_count; ++row) {
      const uint8_t *row_base = scalar_rows_be + (row * scalar_row_stride);
      for (size_t term = 0; term < term_count; ++term) {
        packed_scalar_slices[(row * term_count) + term] =
            CuBbsBbsGpuByteSlice{row_base + (term * 32u), 32u};
      }
    }
    scalar_rows = packed_scalar_slices.data();
  }

  std::vector<uint8_t> decoded_scalars(output_count * term_count * 32u);
  status = cubbs_bbs_blst_decode_scalars(
      scalar_rows, output_count * term_count, decoded_scalars.data(),
      32u);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t row = 0; row < output_count; ++row) {
      set_result(results, row, status, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t scalar_bytes = output_count * term_count * 32u;
  const size_t output_bytes = output_count * sizeof(cubbs_sppark_bucket_t);
  std::vector<cubbs_sppark_bucket_t> host_outputs(output_count);
  std::lock_guard<std::mutex> cache_lock(cached->mutex);
  cudaError_t cuda_status = ensure_fused_common_points_workspace(
      cached.get(), scalar_bytes, output_count);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpy(cached->device_scalars, decoded_scalars.data(),
                             scalar_bytes, cudaMemcpyHostToDevice);
  }
  if (cuda_status == cudaSuccess) {
    const uint32_t blocks = static_cast<uint32_t>(output_count);
    if (env_flag_enabled("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_BUCKET")) {
      const uint32_t threads = 128u;
      cubbs_bbs_g1_msm_common_points_bucket4_kernel<<<blocks, threads>>>(
          cached->device_points, cached->device_scalars, term_count,
          output_count, cached->device_outputs);
    } else if (env_flag_enabled(
                   "CUBBS_BBS_GPU_FUSED_COMMON_POINTS_NO_PRECOMPUTE")) {
      const uint32_t threads = 128u;
      cubbs_bbs_g1_msm_common_points_window4_kernel<<<blocks, threads>>>(
          cached->device_points, cached->device_scalars, term_count,
          output_count, cached->device_outputs);
    } else if (env_flag_enabled("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_WINDOW4")) {
      const uint32_t threads = 128u;
      cubbs_bbs_g1_msm_common_points_window4_precomputed_kernel<<<blocks,
                                                                  threads>>>(
          cached->device_point_multiples, cached->device_scalars, term_count,
          output_count, cached->device_outputs);
    } else {
      const uint32_t threads = fused_common_points_threads_per_row();
      if (threads == 32u) {
        cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<32u>
            <<<blocks, threads>>>(cached->device_point_multiples8,
                                  cached->device_scalars, term_count,
                                  output_count, cached->device_outputs);
      } else if (threads == 64u) {
        cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<64u>
            <<<blocks, threads>>>(cached->device_point_multiples8,
                                  cached->device_scalars, term_count,
                                  output_count, cached->device_outputs);
      } else if (threads == 256u) {
        cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<256u>
            <<<blocks, threads>>>(cached->device_point_multiples8,
                                  cached->device_scalars, term_count,
                                  output_count, cached->device_outputs);
      } else {
        cubbs_bbs_g1_msm_common_points_window8_precomputed_kernel<128u>
            <<<blocks, threads>>>(cached->device_point_multiples8,
                                  cached->device_scalars, term_count,
                                  output_count, cached->device_outputs);
      }
    }
    cuda_status = cudaGetLastError();
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpy(host_outputs.data(), cached->device_outputs,
                             output_bytes, cudaMemcpyDeviceToHost);
  }
  if (cuda_status != cudaSuccess) {
    for (size_t row = 0; row < output_count; ++row) {
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  auto compress_row = [&](size_t row) {
    const cubbs_sppark_point_t projective =
        static_cast<cubbs_sppark_point_t>(host_outputs[row]);
    const int32_t row_status = cubbs_bbs_blst_compress_g1_projective(
        &projective, point_outs[row].data);
    if (row_status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS);
    } else {
      set_result(results, row, row_status, 0);
    }
  };

  const size_t worker_count = fused_compress_worker_count(output_count);
  if (worker_count <= 1u) {
    for (size_t row = 0; row < output_count; ++row) {
      compress_row(row);
    }
  } else {
    std::atomic<size_t> next_row{0u};
    cubbs_run_host_tasks(worker_count, [&](size_t) {
      while (true) {
        const size_t row = next_row.fetch_add(1u);
        if (row >= output_count) {
          break;
        }
        compress_row(row);
      }
    });
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

// Parallel scalar decode: byte-swap + range-check the whole scalar matrix on
// the host using a worker pool instead of the serial cubbs_bbs_blst_decode_scalars
// loop. Each scalar is independent, so this removes a serial tail proportional
// to output_count * term_count. Scalars are range-checked against the Fr
// modulus while decoding directly from big-endian to little-endian form.
static int32_t cubbs_bbs_decode_scalar_be_fast(
    const CuBbsBbsGpuByteSlice &scalar, uint8_t *scalar_le) {
  static constexpr uint8_t fr_modulus_be[32] = {
      0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48,
      0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
      0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe,
      0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01};
  if (scalar.data == nullptr || scalar.len != 32u || scalar_le == nullptr ||
      std::memcmp(scalar.data, fr_modulus_be, sizeof(fr_modulus_be)) >= 0) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
#pragma unroll
  for (size_t byte = 0u; byte < 32u; ++byte) {
    scalar_le[byte] = scalar.data[31u - byte];
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t cubbs_fused_decode_packed_scalars_parallel(
    const uint8_t *scalar_rows_be, size_t row_count, size_t term_count,
    size_t scalar_row_stride, uint8_t *scalar_out) {
  if ((row_count > 0u &&
       (scalar_rows_be == nullptr || scalar_out == nullptr)) ||
      scalar_row_stride < term_count * 32u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t scalar_count = row_count * term_count;
  if (scalar_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  const size_t worker_count =
      std::min<size_t>(blst_msm_worker_count(scalar_count), scalar_count);
  const size_t chunk = (scalar_count + worker_count - 1u) / worker_count;
  std::atomic<int32_t> first_error{0};
  cubbs_run_host_tasks(worker_count, [&](size_t worker) {
    const size_t begin = worker * chunk;
    const size_t end = std::min(scalar_count, begin + chunk);
    if (begin < end) {
      for (size_t scalar = begin; scalar < end; ++scalar) {
        const size_t row = scalar / term_count;
        const size_t term = scalar - row * term_count;
        const CuBbsBbsGpuByteSlice input{
            scalar_rows_be + row * scalar_row_stride + term * 32u, 32u};
        const int32_t status = cubbs_bbs_decode_scalar_be_fast(
            input, scalar_out + scalar * 32u);
        if (status != CUBBS_BBS_GPU_SUCCESS) {
          int32_t expected = 0;
          first_error.compare_exchange_strong(expected, status);
          break;
        }
      }
    }
  });
  const int32_t status = first_error.load();
  return status == 0 ? CUBBS_BBS_GPU_SUCCESS : status;
}

static int32_t cubbs_fused_decode_scalars_parallel(
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t scalar_count,
    uint8_t *scalar_out, size_t requested_workers = 0u) {
  if (scalar_count > 0 && (scalar_rows == nullptr || scalar_out == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (scalar_count == 0u) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t worker_count = std::min<size_t>(
      requested_workers > 0u ? requested_workers
                             : blst_msm_worker_count(scalar_count),
      scalar_count);
  if (worker_count <= 1u) {
    for (size_t scalar = 0u; scalar < scalar_count; ++scalar) {
      const int32_t status = cubbs_bbs_decode_scalar_be_fast(
          scalar_rows[scalar], scalar_out + scalar * 32u);
      if (status != CUBBS_BBS_GPU_SUCCESS) return status;
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t chunk =
      (scalar_count + worker_count - 1u) / worker_count;
  std::atomic<int32_t> first_error{0};
  cubbs_run_host_tasks(worker_count, [&](size_t worker) {
    const size_t begin = worker * chunk;
    const size_t end = std::min(scalar_count, begin + chunk);
    if (begin < end) {
      for (size_t scalar = begin; scalar < end; ++scalar) {
        const int32_t status = cubbs_bbs_decode_scalar_be_fast(
            scalar_rows[scalar], scalar_out + scalar * 32u);
        if (status != CUBBS_BBS_GPU_SUCCESS) {
          int32_t expected = 0;
          first_error.compare_exchange_strong(expected, status);
          break;
        }
      }
    }
  });
  const int32_t err = first_error.load();
  return err == 0 ? CUBBS_BBS_GPU_SUCCESS : err;
}


#endif
