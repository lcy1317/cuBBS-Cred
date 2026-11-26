#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

static int32_t cubbs_bbs_cuda_g1_variable_base_generic_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  cubbs_bbs_ensure_cuda_device_schedule();

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
      set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC);
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
    cubbs_bbs_g1_msm_generic_rows_glv_kernel<<<static_cast<uint32_t>(item_count), 128u, 0, stream>>>(
        d_bases, d_scalars, d_offsets, item_count, d_outputs);
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
    set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
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
  if (g1_msm_items_have_common_points(items, item_count)) {
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
  return cubbs_bbs_cuda_g1_variable_base_generic_batch(items, item_count, results);
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
