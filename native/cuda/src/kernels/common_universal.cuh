#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

// Resource-bounded common-points fallback for shapes outside the tuned comb
// and cached tiled ranges. Points are decoded and uploaded once, while scalar
// rows are processed in bounded chunks. The executable MSM kernel is owned by
// cuBBS-Cred; BLST is used only for point encoding/decoding at the ABI boundary.
static int32_t compute_g1_msm_universal_common_points_with_cuda(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  if (output_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  if (output_count > static_cast<size_t>(UINT32_MAX) ||
      term_count > static_cast<size_t>(-1) / 32u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  const size_t row_scalar_bytes = term_count * 32u;
  const bool row_scale = scalar_rows_be != nullptr &&
      scalar_row_stride == (term_count + 1u) * 32u;
  if (term_count > static_cast<size_t>(-1) /
                       sizeof(cubbs_sppark_plain_affine_t)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  std::vector<cubbs_sppark_plain_affine_t> host_points(term_count);
  if (term_count > 0u) {
    const int32_t decode_status = cubbs_bbs_blst_decode_g1_points(
        points, term_count, reinterpret_cast<uint8_t *>(host_points.data()),
        sizeof(cubbs_sppark_plain_affine_t));
    if (decode_status != CUBBS_BBS_GPU_SUCCESS) return decode_status;
  }

  const size_t scalar_budget = env_size_or_default(
      "CUBBS_BBS_GPU_GENERIC_SCALAR_BUDGET_BYTES", 256u * 1024u * 1024u);
  const size_t rows_per_chunk = row_scalar_bytes == 0u
      ? output_count
      : std::max<size_t>(
            1u, std::min(output_count,
                         std::max<size_t>(1u, scalar_budget) /
                             row_scalar_bytes));
  const size_t max_scalar_bytes = rows_per_chunk * row_scalar_bytes;
  const size_t max_scale_bytes = row_scale ? rows_per_chunk * 32u : 0u;
  if (max_scale_bytes > SIZE_MAX - max_scalar_bytes) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;

  cubbs_sppark_plain_affine_t *device_points = nullptr;
  uint8_t *device_scalars = nullptr;
  cubbs_sppark_bucket_t *device_outputs = nullptr;
  const size_t point_bytes =
      term_count * sizeof(cubbs_sppark_plain_affine_t);
  cudaError_t error = cudaMalloc(
      &device_points, std::max<size_t>(1u, point_bytes));
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_scalars,
                       std::max<size_t>(1u, max_scalar_bytes + max_scale_bytes));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_outputs,
                       rows_per_chunk * sizeof(*device_outputs));
  }
  if (error == cudaSuccess && point_bytes > 0u) {
    error = cudaMemcpy(device_points, host_points.data(), point_bytes,
                       cudaMemcpyHostToDevice);
  }

  std::vector<uint8_t> decoded_scalars(max_scalar_bytes + max_scale_bytes);
  std::vector<cubbs_sppark_bucket_t> host_outputs(rows_per_chunk);
  for (size_t row_begin = 0u;
       error == cudaSuccess && row_begin < output_count;
       row_begin += rows_per_chunk) {
    const size_t rows = std::min(rows_per_chunk, output_count - row_begin);
    const size_t scalar_bytes = rows * row_scalar_bytes;
    int32_t decode_status = CUBBS_BBS_GPU_SUCCESS;
    if (scalar_rows_be != nullptr) {
      decode_status = cubbs_fused_decode_packed_scalars_parallel(
          scalar_rows_be + row_begin * scalar_row_stride + (row_scale ? 32u : 0u), rows, term_count,
          scalar_row_stride, decoded_scalars.data());
      if (decode_status == CUBBS_BBS_GPU_SUCCESS && row_scale) {
        decode_status = cubbs_fused_decode_packed_scalars_parallel(
            scalar_rows_be + row_begin * scalar_row_stride, rows, 1u,
            scalar_row_stride, decoded_scalars.data() + scalar_bytes);
      }
    } else {
      decode_status = cubbs_fused_decode_scalars_parallel(
          scalar_rows + row_begin * term_count, rows * term_count,
          decoded_scalars.data());
    }
    if (decode_status != CUBBS_BBS_GPU_SUCCESS) {
      for (size_t row = row_begin; row < row_begin + rows; ++row) {
        set_result(results, row, decode_status, 0u);
      }
      error = cudaErrorInvalidValue;
      break;
    }
    if (scalar_bytes > 0u) {
      const size_t upload_bytes = scalar_bytes + (row_scale ? rows * 32u : 0u);
      error = cudaMemcpy(device_scalars, decoded_scalars.data(), upload_bytes,
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess && row_scale && scalar_bytes > 0u) {
      const size_t count = rows * term_count;
      cubbs_bbs_scale_scalar_rows_kernel<<<static_cast<uint32_t>((count + 255u) / 256u), 256u>>>(
          device_scalars, device_scalars + scalar_bytes, term_count, count);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      cubbs_bbs_g1_msm_common_points_window4_kernel<<<
          static_cast<uint32_t>(rows), 128u>>>(
          device_points, device_scalars, term_count, rows, device_outputs);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(host_outputs.data(), device_outputs,
                         rows * sizeof(host_outputs[0]),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) break;
    for (size_t local_row = 0u; local_row < rows; ++local_row) {
      const size_t row = row_begin + local_row;
      const cubbs_sppark_point_t projective =
          static_cast<cubbs_sppark_point_t>(host_outputs[local_row]);
      const int32_t compress_status = cubbs_bbs_blst_compress_g1_projective(
          &projective, point_outs[row].data);
      if (compress_status == CUBBS_BBS_GPU_SUCCESS) {
        set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                         CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC);
      } else {
        set_result(results, row, compress_status, 0u);
      }
    }
  }

  if (device_outputs != nullptr) (void)cudaFree(device_outputs);
  if (device_scalars != nullptr) (void)cudaFree(device_scalars);
  if (device_points != nullptr) (void)cudaFree(device_points);
  if (error != cudaSuccess) {
    for (size_t row = 0u; row < output_count; ++row) {
      if (results[row].bytes_written == 0u && results[row].code == 0) {
        set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
      }
    }
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

// Production multi-output fused common-points MSM body. Same window8
// precomputed Pippenger kernel as the naive scaffold, but with three
// improvements that lift GPU utilization from ~15% to ~33%+ and remove the
// serial host tail:
//   1. default 128 threads/row (vs 32) for higher occupancy;
//   2. launches on a per-point-set CUDA stream so successive VC-shape groups
//      overlap their H2D/kernel/D2H/compress stages instead of serializing;
//   3. parallel host-side scalar decode and output compression via worker
//      pools, overlapping host work with the device kernel.
__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_scalar_mul_glv_device(cubbs_sppark_bucket_t base,
                                    const uint8_t remainder_le[16],
                                    const uint8_t quotient_le[16]);

__global__ static void cubbs_bbs_g1_add_variable_term_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    size_t item_count, cubbs_sppark_bucket_t *outputs);

__global__ static void cubbs_bbs_proof_verify_radix16_tables_kernel(
    const cubbs_sppark_plain_affine_t *proof_bases, size_t proof_point_count,
    cubbs_sppark_bucket_t *tables);

__global__ static void cubbs_bbs_g1_add_variable_term_radix16_kernel(
    const cubbs_sppark_bucket_t *tables, const uint8_t *scalars_le,
    size_t item_count, cubbs_sppark_bucket_t *outputs);

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
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess) {
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
          reinterpret_cast<void **>(&scalars), scalar_bytes,
          cudaHostAllocPortable);
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


#endif
