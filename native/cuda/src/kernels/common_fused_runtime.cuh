#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_COMMON_FUSED_RUNTIME_CUH
#define CUBBS_BBS_COMMON_FUSED_RUNTIME_CUH

// ============================================================================
// Pinned Host Workspace, Parallel Scalar Decoding, & Standalone Common MSM
// ============================================================================

static cudaError_t cubbs_bbs_launch_xyzz_to_compressed(
    const cubbs_sppark_bucket_t *device_outputs, size_t count,
    uint8_t *device_compressed_out, size_t out_stride, size_t out_offset,
    cudaStream_t stream);

struct CuBbsFusedPinnedHostWorkspace {
  uint8_t *scalars{nullptr};
  size_t scalar_capacity{0u};
  cubbs_sppark_bucket_t *outputs{nullptr};
  size_t output_capacity{0u};
  cudaStream_t stream{nullptr};

  ~CuBbsFusedPinnedHostWorkspace() {
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    if (scalars != nullptr) (void)cudaFreeHost(scalars);
    if (outputs != nullptr) (void)cudaFreeHost(outputs);
  }

  cudaStream_t execution_stream(cudaStream_t fallback) {
    if (stream == nullptr &&
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      stream = nullptr;
    }
    return stream != nullptr ? stream : fallback;
  }

  cudaError_t ensure(size_t scalar_bytes, size_t output_count) {
    if (scalar_bytes > scalar_capacity) {
      if (scalars != nullptr) {
        (void)cudaFreeHost(scalars);
        scalars = nullptr;
        scalar_capacity = 0u;
      }
      cudaError_t status = cudaHostAlloc(
          reinterpret_cast<void **>(&scalars), scalar_bytes, cudaHostAllocPortable);
      if (status != cudaSuccess) return status;
      scalar_capacity = scalar_bytes;
    }
    if (output_count > output_capacity) {
      if (outputs != nullptr) {
        (void)cudaFreeHost(outputs);
        outputs = nullptr;
        output_capacity = 0u;
      }
      cudaError_t status = cudaHostAlloc(
          reinterpret_cast<void **>(&outputs),
          output_count * sizeof(*outputs), cudaHostAllocPortable);
      if (status != cudaSuccess) return status;
      output_capacity = output_count;
    }
    return cudaSuccess;
  }
};

static CuBbsFusedPinnedHostWorkspace &cubbs_fused_pinned_host_workspace() {
  static thread_local CuBbsFusedPinnedHostWorkspace workspace;
  return workspace;
}

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
  if ((row_count > 0u && (scalar_rows_be == nullptr || scalar_out == nullptr)) ||
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
  });
  const int32_t status = first_error.load();
  return status == 0 ? CUBBS_BBS_GPU_SUCCESS : status;
}

static int32_t cubbs_fused_decode_scalars_parallel(
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t scalar_count,
    uint8_t *scalar_out, size_t requested_workers = 0u) {
  if (scalar_count > 0u && (scalar_rows == nullptr || scalar_out == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (scalar_count == 0u) return CUBBS_BBS_GPU_SUCCESS;

  const size_t worker_count = std::min<size_t>(
      requested_workers > 0u ? requested_workers : blst_msm_worker_count(scalar_count),
      scalar_count);
  if (worker_count <= 1u) {
    for (size_t scalar = 0u; scalar < scalar_count; ++scalar) {
      const int32_t status = cubbs_bbs_decode_scalar_be_fast(
          scalar_rows[scalar], scalar_out + scalar * 32u);
      if (status != CUBBS_BBS_GPU_SUCCESS) return status;
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }
  const size_t chunk = (scalar_count + worker_count - 1u) / worker_count;
  std::atomic<int32_t> first_error{0};
  cubbs_run_host_tasks(worker_count, [&](size_t worker) {
    const size_t begin = worker * chunk;
    const size_t end = std::min(scalar_count, begin + chunk);
    for (size_t scalar = begin; scalar < end; ++scalar) {
      const int32_t status = cubbs_bbs_decode_scalar_be_fast(
          scalar_rows[scalar], scalar_out + scalar * 32u);
      if (status != CUBBS_BBS_GPU_SUCCESS) {
        int32_t expected = 0;
        first_error.compare_exchange_strong(expected, status);
        break;
      }
    }
  });
  const int32_t err = first_error.load();
  return err == 0 ? CUBBS_BBS_GPU_SUCCESS : err;
}

static int32_t compute_g1_msm_fused_common_points_with_cuda(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  if (output_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  cubbs_bbs_ensure_cuda_device_schedule();

  if (term_count == 0u) {
    static constexpr uint8_t inf_g1[48] = {0xc0u};
    for (size_t row = 0u; row < output_count; ++row) {
      std::memcpy(point_outs[row].data, inf_g1, 48u);
      set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  int32_t status = get_fused_common_points_cache(points, term_count, &cached);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t row = 0u; row < output_count; ++row) set_result(results, row, status, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t total_scalars = output_count * term_count;
  const size_t scalar_bytes = total_scalars * 32u;
  const bool has_device_row_scale =
      scalar_rows_be != nullptr && scalar_row_stride == (term_count + 1u) * 32u;
  const size_t row_scale_bytes = has_device_row_scale ? output_count * 32u : 0u;
  const size_t scalar_buffer_bytes = scalar_bytes + row_scale_bytes;
  auto &host_ws = cubbs_fused_pinned_host_workspace();
  if (host_ws.ensure(scalar_buffer_bytes, output_count) != cudaSuccess) {
    for (size_t row = 0u; row < output_count; ++row)
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  status = scalar_rows_be != nullptr
      ? cubbs_fused_decode_packed_scalars_parallel(
            scalar_rows_be + (has_device_row_scale ? 32u : 0u),
            output_count, term_count, scalar_row_stride, host_ws.scalars)
      : cubbs_fused_decode_scalars_parallel(
            scalar_rows, total_scalars, host_ws.scalars);
  if (status == CUBBS_BBS_GPU_SUCCESS && has_device_row_scale) {
    status = cubbs_fused_decode_packed_scalars_parallel(
        scalar_rows_be, output_count, 1u, scalar_row_stride,
        host_ws.scalars + scalar_bytes);
  }
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t row = 0u; row < output_count; ++row) set_result(results, row, status, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const cudaStream_t stream = host_ws.execution_stream(cached->stream);
  auto &arena = cubbs_bbs_thread_device_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(scalar_buffer_bytes) +
      CuBbsDeviceArena::align_up(output_count * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(output_count * 48u);
  cudaError_t err = arena.reserve_and_reset(arena_bytes);
  uint8_t *d_scalars = nullptr;
  cubbs_sppark_bucket_t *d_outputs = nullptr;
  uint8_t *d_compressed = nullptr;
  if (err == cudaSuccess) {
    d_scalars = arena.alloc<uint8_t>(scalar_buffer_bytes);
    d_outputs = arena.alloc<cubbs_sppark_bucket_t>(output_count);
    d_compressed = arena.alloc<uint8_t>(output_count * 48u);
    err = cudaMemcpyAsync(d_scalars, host_ws.scalars, scalar_buffer_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess && has_device_row_scale) {
    constexpr uint32_t st = 256u;
    const uint32_t sb = static_cast<uint32_t>((total_scalars + st - 1u) / st);
    cubbs_bbs_scale_scalar_rows_kernel<<<sb, st, 0, stream>>>(
        d_scalars, d_scalars + scalar_bytes, term_count, total_scalars);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_cached_common_msm_range(
        *cached, d_scalars, term_count, output_count, d_outputs, stream);
  }
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_xyzz_to_compressed(
        d_outputs, output_count, d_compressed, 48u, 0u, stream);
  }
  std::vector<uint8_t> host_compressed(output_count * 48u);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_compressed.data(), d_compressed, output_count * 48u,
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    for (size_t row = 0u; row < output_count; ++row)
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const uint8_t route = cached->use_pippenger_large
      ? CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE
      : CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS;
  for (size_t row = 0u; row < output_count; ++row) {
    std::memcpy(point_outs[row].data, host_compressed.data() + (row * 48u), 48u);
    set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u, route);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

#endif // CUBBS_BBS_COMMON_FUSED_RUNTIME_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
