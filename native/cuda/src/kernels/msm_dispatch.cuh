#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

// 1-thread-per-row GLV short item MSM (Route 1 ablation for K <= 3 variable-base).
__global__ static void cubbs_bbs_g1_msm_short_item_glv_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    const size_t *row_offsets, size_t item_count,
    cubbs_sppark_bucket_t *outputs) {
  const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= item_count) return;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const size_t begin = row_offsets[item];
  const size_t end = row_offsets[item + 1u];
  for (size_t term = begin; term < end; ++term) {
    cubbs_sppark_bucket_t base;
    base = bases[term];
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    cubbs_bbs_scalar_div_by_z2_device(
        scalars_le + term * 32u, remainder_le, quotient_le);
    acc.add(cubbs_bbs_g1_scalar_mul_glv_device(base, remainder_le, quotient_le));
  }
  outputs[item] = acc;
}

// Generic 128t/row GLV common-point MSM (Route 4 ablation baseline on shared bases).
__global__ static void cubbs_bbs_g1_msm_common_generic_glv_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    size_t term_count, size_t output_count, cubbs_sppark_bucket_t *outputs) {
  constexpr uint32_t kThreads = 128u;
  __shared__ __align__(16) uint8_t
      partial_storage[kThreads * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t row = blockIdx.x;
  if (row >= output_count) return;

  const uint32_t tid = threadIdx.x;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint8_t *row_scalars = scalars_le + row * term_count * 32u;
  for (size_t term = tid; term < term_count; term += blockDim.x) {
    cubbs_sppark_bucket_t base;
    base = bases[term];
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    cubbs_bbs_scalar_div_by_z2_device(
        row_scalars + term * 32u, remainder_le, quotient_le);
    acc.add(cubbs_bbs_g1_scalar_mul_glv_device(base, remainder_le, quotient_le));
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) outputs[row] = partials[0];
}

static int32_t cubbs_bbs_cuda_g1_variable_base_batch_impl(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results, bool use_short_item_route1) {
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  cubbs_bbs_ensure_cuda_device_schedule();
  const uint8_t route_code = use_short_item_route1
      ? CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM
      : CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC;

  std::vector<size_t> row_offsets(item_count + 1u, 0u);
  size_t total_terms = 0u;
  for (size_t i = 0u; i < item_count; ++i) {
    total_terms += items[i].term_count;
    row_offsets[i + 1u] = total_terms;
  }
  if (total_terms == 0u) {
    static constexpr uint8_t inf_g1[48] = {0xc0u};
    for (size_t i = 0u; i < item_count; ++i) {
      std::memcpy(items[i].compressed_point_out, inf_g1, 48u);
      set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u, route_code);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::vector<CuBbsBbsGpuByteSlice> flat_points;
  std::vector<CuBbsBbsGpuByteSlice> flat_scalars;
  flat_points.reserve(total_terms);
  flat_scalars.reserve(total_terms);
  for (size_t i = 0u; i < item_count; ++i) {
    flat_points.insert(flat_points.end(), items[i].points, items[i].points + items[i].term_count);
    flat_scalars.insert(flat_scalars.end(), items[i].scalars, items[i].scalars + items[i].term_count);
  }

  std::vector<cubbs_sppark_plain_affine_t> host_bases(total_terms);
  int32_t status = cubbs_bbs_blst_decode_g1_points(
      flat_points.data(), total_terms,
      reinterpret_cast<uint8_t *>(host_bases.data()),
      sizeof(cubbs_sppark_plain_affine_t));
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t i = 0u; i < item_count; ++i) set_result(results, i, status, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::vector<uint8_t> host_scalars(total_terms * 32u);
  status = cubbs_fused_decode_scalars_parallel(
      flat_scalars.data(), total_terms, host_scalars.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t i = 0u; i < item_count; ++i) set_result(results, i, status, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const cudaStream_t stream = cubbs_fused_pinned_host_workspace().execution_stream(nullptr);
  auto &arena = cubbs_bbs_thread_device_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(total_terms * sizeof(cubbs_sppark_plain_affine_t)) +
      CuBbsDeviceArena::align_up(total_terms * 32u) +
      CuBbsDeviceArena::align_up((item_count + 1u) * sizeof(size_t)) +
      CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(item_count * 48u);
  cudaError_t err = arena.reserve_and_reset(arena_bytes);
  cubbs_sppark_plain_affine_t *d_bases = nullptr;
  uint8_t *d_scalars = nullptr;
  size_t *d_offsets = nullptr;
  cubbs_sppark_bucket_t *d_outputs = nullptr;
  uint8_t *d_compressed = nullptr;
  if (err == cudaSuccess) {
    d_bases = arena.alloc<cubbs_sppark_plain_affine_t>(total_terms);
    d_scalars = arena.alloc<uint8_t>(total_terms * 32u);
    d_offsets = arena.alloc<size_t>(item_count + 1u);
    d_outputs = arena.alloc<cubbs_sppark_bucket_t>(item_count);
    d_compressed = arena.alloc<uint8_t>(item_count * 48u);
    err = cudaMemcpyAsync(d_bases, host_bases.data(),
                          total_terms * sizeof(cubbs_sppark_plain_affine_t),
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(d_scalars, host_scalars.data(), total_terms * 32u,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(d_offsets, row_offsets.data(),
                          (item_count + 1u) * sizeof(size_t),
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    if (use_short_item_route1) {
      constexpr uint32_t kThreads = 32u;
      const uint32_t blocks =
          static_cast<uint32_t>((item_count + kThreads - 1u) / kThreads);
      cubbs_bbs_g1_msm_short_item_glv_kernel<<<blocks, kThreads, 0, stream>>>(
          d_bases, d_scalars, d_offsets, item_count, d_outputs);
    } else {
      cubbs_bbs_g1_msm_generic_rows_glv_kernel<<<static_cast<uint32_t>(item_count), 128u, 0, stream>>>(
          d_bases, d_scalars, d_offsets, item_count, d_outputs);
    }
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_xyzz_to_compressed(
        d_outputs, item_count, d_compressed, 48u, 0u, stream);
  }
  std::vector<uint8_t> host_compressed(item_count * 48u);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_compressed.data(), d_compressed, item_count * 48u,
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    for (size_t i = 0u; i < item_count; ++i)
      set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }
  for (size_t i = 0u; i < item_count; ++i) {
    std::memcpy(items[i].compressed_point_out, host_compressed.data() + (i * 48u), 48u);
    set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u, route_code);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t compute_g1_msm_common_generic_route4_with_cuda(
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
                       CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC);
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
  auto &host_ws = cubbs_fused_pinned_host_workspace();
  if (host_ws.ensure(scalar_bytes, output_count) != cudaSuccess) {
    for (size_t row = 0u; row < output_count; ++row)
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }
  status = scalar_rows_be != nullptr
      ? cubbs_fused_decode_packed_scalars_parallel(
            scalar_rows_be, output_count, term_count, scalar_row_stride,
            host_ws.scalars)
      : cubbs_fused_decode_scalars_parallel(
            scalar_rows, total_scalars, host_ws.scalars);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t row = 0u; row < output_count; ++row) set_result(results, row, status, 0u);
    return CUBBS_BBS_GPU_SUCCESS;
  }
  const cudaStream_t stream = host_ws.execution_stream(cached->stream);
  auto &arena = cubbs_bbs_thread_device_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(scalar_bytes) +
      CuBbsDeviceArena::align_up(output_count * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(output_count * 48u);
  cudaError_t err = arena.reserve_and_reset(arena_bytes);
  uint8_t *d_scalars = nullptr;
  cubbs_sppark_bucket_t *d_outputs = nullptr;
  uint8_t *d_compressed = nullptr;
  if (err == cudaSuccess) {
    d_scalars = arena.alloc<uint8_t>(scalar_bytes);
    d_outputs = arena.alloc<cubbs_sppark_bucket_t>(output_count);
    d_compressed = arena.alloc<uint8_t>(output_count * 48u);
    err = cudaMemcpyAsync(d_scalars, host_ws.scalars, scalar_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    cubbs_bbs_g1_msm_common_generic_glv_kernel<<<static_cast<uint32_t>(output_count), 128u, 0, stream>>>(
        cached->device_points, d_scalars, term_count, output_count, d_outputs);
    err = cudaGetLastError();
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
  for (size_t row = 0u; row < output_count; ++row) {
    std::memcpy(point_outs[row].data, host_compressed.data() + (row * 48u), 48u);
    point_outs[row].len = 48u;
    set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                     CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

#endif

int32_t cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  if ((term_count > 0u && points == nullptr) ||
      (output_count > 0u && (point_outs == nullptr || results == nullptr)) ||
      (output_count > 0u && term_count > 0u && scalar_rows == nullptr &&
       scalar_rows_be == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  return compute_g1_msm_fused_common_points_with_cuda(
      points, term_count, scalar_rows, output_count, scalar_rows_be,
      scalar_row_stride, point_outs, results);
#else
  for (size_t row = 0u; row < output_count; ++row) {
    set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
  }
  return CUBBS_BBS_GPU_SUCCESS;
#endif
}

int32_t cubbs_bbs_cuda_g1_msm_common_points_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  const uint8_t benchmark_route = explicit_msm_benchmark_route();
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS ||
      benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_ITEM) {
    std::vector<CuBbsBbsGpuByteSlice> flat_scalars;
    if (scalar_rows == nullptr && scalar_rows_be != nullptr) {
      flat_scalars.resize(output_count * term_count);
      for (size_t row = 0u; row < output_count; ++row) {
        const uint8_t *row_ptr = scalar_rows_be + row * scalar_row_stride;
        for (size_t term = 0u; term < term_count; ++term) {
          flat_scalars[row * term_count + term] =
              CuBbsBbsGpuByteSlice{row_ptr + term * 32u, 32u};
        }
      }
    }
    const CuBbsBbsGpuByteSlice *scalars_ptr =
        scalar_rows != nullptr ? scalar_rows : flat_scalars.data();
    std::vector<CuBbsG1MsmInput> blst_items(output_count);
    for (size_t row = 0u; row < output_count; ++row) {
      blst_items[row] = CuBbsG1MsmInput{
          points, scalars_ptr + row * term_count, term_count,
          point_outs[row].data};
    }
    const int32_t status =
        benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS
            ? cubbs_bbs_blst_g1_msm_common_points_batch(
                  blst_items.data(), output_count, results)
            : cubbs_bbs_blst_g1_msm_batch(
                  blst_items.data(), output_count, results);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      for (size_t row = 0u; row < output_count; ++row) {
        if (results[row].code == CUBBS_BBS_GPU_SUCCESS) {
          point_outs[row].len = results[row].bytes_written;
        }
      }
    }
    return status;
  }
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL) {
    return compute_g1_msm_common_generic_route4_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results);
  }
#endif
  return cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
      points, term_count, scalar_rows, output_count, scalar_rows_be,
      scalar_row_stride, point_outs, results);
}

int32_t cubbs_bbs_cuda_g1_msm_batch(const CuBbsG1MsmInput *items,
                                     size_t item_count,
                                     CuBbsBbsGpuResult *results) {
  if (item_count > 0u && (items == nullptr || results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t i = 0u; i < item_count; ++i) {
    if ((items[i].term_count > 0u &&
         (items[i].points == nullptr || items[i].scalars == nullptr)) ||
        items[i].compressed_point_out == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t j = 0u; j < items[i].term_count; ++j) {
      if (items[i].points[j].len != 48u || items[i].points[j].data == nullptr ||
          items[i].scalars[j].len != 32u || items[i].scalars[j].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
    }
  }
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;

  const uint8_t benchmark_route = explicit_msm_benchmark_route();
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_ITEM) {
    return cubbs_bbs_blst_g1_msm_batch(items, item_count, results);
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS) {
    return cubbs_bbs_blst_g1_msm_common_points_batch(items, item_count, results);
  }

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  if (benchmark_route != CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL &&
      benchmark_route != CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM &&
      g1_msm_items_have_common_points(items, item_count)) {
    const size_t term_count = items[0].term_count;
    std::vector<CuBbsBbsGpuByteSlice> scalar_rows;
    scalar_rows.reserve(item_count * term_count);
    std::vector<CuBbsBbsGpuByteBuffer> point_outs;
    point_outs.reserve(item_count);
    for (size_t row = 0u; row < item_count; ++row) {
      scalar_rows.insert(scalar_rows.end(), items[row].scalars,
                         items[row].scalars + term_count);
      point_outs.push_back(CuBbsBbsGpuByteBuffer{
          items[row].compressed_point_out, 48u, 0u});
    }
    return cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
        items[0].points, term_count, scalar_rows.data(), item_count, nullptr,
        0u, point_outs.data(), results);
  }
  bool all_short = true;
  bool all_k3 = true;
  for (size_t i = 0u; i < item_count; ++i) {
    if (items[i].term_count > 3u) all_short = false;
    if (items[i].term_count != 3u) all_k3 = false;
  }
  const bool use_route1 =
      (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM && all_short) ||
      (benchmark_route == 0u && all_k3 && item_count >= 128u);
  return cubbs_bbs_cuda_g1_variable_base_batch_impl(
      items, item_count, results, use_route1);
#else
  return cubbs_bbs_blst_g1_msm_batch(items, item_count, results);
#endif
}

int32_t cubbs_bbs_cuda_prepare_b_msm_batch(const CuBbsPrepareBInput *items,
                                            size_t item_count,
                                            CuBbsBbsGpuResult *results) {
  if (item_count > 0u && (items == nullptr || results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  (void)items;
  for (size_t i = 0u; i < item_count; ++i) {
    set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}
