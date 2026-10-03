static int32_t cubbs_bbs_cuda_g1_variable_base_three_term_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_g1_msm_batch(const CuBbsG1MsmInput *items,
                                     size_t item_count,
                                     CuBbsBbsGpuResult *results) {
  if (item_count > 0 && (items == nullptr || results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  for (size_t i = 0; i < item_count; ++i) {
    if ((items[i].term_count > 0 &&
         (items[i].points == nullptr || items[i].scalars == nullptr)) ||
        items[i].compressed_point_out == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t j = 0; j < items[i].term_count; ++j) {
      if (items[i].points[j].len != 48u || items[i].points[j].data == nullptr ||
          items[i].scalars[j].len != 32u || items[i].scalars[j].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
    }
  }

  const uint8_t benchmark_route = explicit_msm_benchmark_route();
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM) {
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
    if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
    const size_t term_count = items[0].term_count;
    const bool uniform_short_shape =
        (term_count == 1u || term_count == 2u || term_count == 3u) &&
        std::all_of(items, items + item_count, [term_count](const auto &item) {
          return item.term_count == term_count;
        });
    if (!uniform_short_shape ||
        g1_msm_items_have_common_points(items, item_count)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    return term_count == 3u
               ? cubbs_bbs_cuda_g1_variable_base_three_term_batch(
                     items, item_count, results)
               : cubbs_bbs_cuda_g1_variable_base_single_term_batch(
                     items, item_count, results);
#else
    return CUBBS_BBS_GPU_ERR_INTERNAL;
#endif
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_ITEM) {
    return cubbs_bbs_blst_g1_msm_batch(items, item_count, results);
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS) {
    return cubbs_bbs_blst_g1_msm_common_points_batch(items, item_count, results);
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_SPPARK_ITEM_REFERENCE ||
      benchmark_route ==
          CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE) {
#if defined(CUBBS_BBS_ENABLE_SPPARK_REFERENCE)
    return compute_g1_msm_sppark_reference_batch(
        items, item_count, results, benchmark_route);
#else
    for (size_t item = 0u; item < item_count; ++item) {
      set_result(results, item, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    }
    return CUBBS_BBS_GPU_SUCCESS;
#endif
  }

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;

  // Every common-point shape stays on an cuBBS-Cred kernel. Tuned shapes use
  // comb/GLV tiled kernels; all other sizes are handled by the bounded
  // universal common-points kernel selected inside the fused ABI.
  if (g1_msm_items_have_common_points(items, item_count)) {
    const size_t term_count = items[0].term_count;
    if (term_count > 0u && item_count > static_cast<size_t>(-1) / term_count) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    std::vector<CuBbsBbsGpuByteSlice> scalar_rows;
    scalar_rows.reserve(item_count * term_count);
    std::vector<CuBbsBbsGpuByteBuffer> point_outs;
    point_outs.reserve(item_count);
    for (size_t row = 0; row < item_count; ++row) {
      scalar_rows.insert(scalar_rows.end(), items[row].scalars,
                         items[row].scalars + term_count);
      point_outs.push_back(CuBbsBbsGpuByteBuffer{
          items[row].compressed_point_out, 48u, 0u});
    }
    if (benchmark_route >= CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS &&
        benchmark_route <= CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL) {
      return cubbs_bbs_cuda_g1_msm_common_points_matrix(
          items[0].points, term_count, scalar_rows.data(), item_count, nullptr,
          0u, point_outs.data(), results);
    }
    return cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
        items[0].points, term_count, scalar_rows.data(), item_count, nullptr,
        0u, point_outs.data(), results);
  }

  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS ||
      benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL) {
    return cubbs_bbs_cuda_g1_variable_base_generic_batch(
        items, item_count, results);
  }

  size_t max_term_count = 0u;
  for (size_t i = 0u; i < item_count; ++i) {
    max_term_count = std::max(max_term_count, items[i].term_count);
  }
  const bool all_single_term = max_term_count == 1u &&
                               std::all_of(items, items + item_count, [](const auto &item) {
                                 return item.term_count == 1u;
                               });
  const bool all_two_terms = max_term_count == 2u &&
                             std::all_of(items, items + item_count, [](const auto &item) {
                               return item.term_count == 2u;
                             });
  const bool all_three_terms = max_term_count == 3u &&
                               std::all_of(items, items + item_count, [](const auto &item) {
                                 return item.term_count == 3u;
                               });
  const bool proof_verify_t1_fused =
      all_three_terms &&
      (std::getenv("CUBBS_BBS_GPU_PROOF_VERIFY_T1_FUSED") != nullptr
           ? env_flag_enabled("CUBBS_BBS_GPU_PROOF_VERIFY_T1_FUSED")
           : item_count >= 128u);
  if (proof_verify_t1_fused) {
    const int32_t status = cubbs_bbs_cuda_g1_variable_base_three_term_batch(
        items, item_count, results);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
  }
  const bool variable_base_enabled =
      (all_single_term && env_flag_enabled("CUBBS_BBS_GPU_VARIABLE_BASE")) ||
      (all_two_terms &&
       (std::getenv("CUBBS_BBS_GPU_VERIFY_FUSED_G1") != nullptr
            ? env_flag_enabled("CUBBS_BBS_GPU_VERIFY_FUSED_G1")
            : item_count >= 512u));
  if (variable_base_enabled &&
      item_count >= env_size_or_default(
                        "CUBBS_BBS_GPU_VARIABLE_BASE_MIN_ITEMS", 128u)) {
    const int32_t status = cubbs_bbs_cuda_g1_variable_base_single_term_batch(
        items, item_count, results);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
  }

  return cubbs_bbs_cuda_g1_variable_base_generic_batch(
      items, item_count, results);
#else
  for (size_t i = 0u; i < item_count; ++i) {
    set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
  }
  return CUBBS_BBS_GPU_SUCCESS;
#endif
}
int32_t cubbs_bbs_cuda_g1_msm_common_points_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  const bool has_packed_scalars = scalar_rows_be != nullptr;
  if ((term_count > 0 && points == nullptr) ||
      (output_count > 0 &&
       ((!has_packed_scalars && scalar_rows == nullptr) || point_outs == nullptr ||
        results == nullptr)) ||
      (has_packed_scalars && scalar_row_stride < term_count * 32u)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t term = 0; term < term_count; ++term) {
    if (points[term].len != 48u || points[term].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }
  for (size_t row = 0; row < output_count; ++row) {
    if (point_outs[row].data == nullptr || point_outs[row].capacity < 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (!has_packed_scalars) {
      for (size_t term = 0; term < term_count; ++term) {
        const size_t scalar_index = (row * term_count) + term;
        if (scalar_rows[scalar_index].len != 32u ||
            scalar_rows[scalar_index].data == nullptr) {
          return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
        }
      }
    }
  }

  const uint8_t benchmark_route = explicit_msm_benchmark_route();
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS) {
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
    if (output_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
    if (!bbs_specialized_common_points_supported(term_count)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    return compute_g1_msm_fused_common_points_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results);
#else
    return CUBBS_BBS_GPU_ERR_INTERNAL;
#endif
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE) {
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
    if (output_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
    if (!bbs_large_shape_kernel_supported(term_count)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    return compute_g1_msm_fused_common_points_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results, nullptr, nullptr, true);
#else
    return CUBBS_BBS_GPU_ERR_INTERNAL;
#endif
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL) {
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
    return compute_g1_msm_universal_common_points_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results);
#else
    return CUBBS_BBS_GPU_ERR_INTERNAL;
#endif
  }
  if (benchmark_route == CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS) {
    if (has_packed_scalars) {
      return cubbs_bbs_blst_g1_msm_common_points_matrix_packed(
          points, term_count, scalar_rows_be, output_count, scalar_row_stride,
          point_outs, results);
    }
    std::vector<CuBbsG1MsmInput> blst_items(output_count);
    for (size_t row = 0u; row < output_count; ++row) {
      blst_items[row] = CuBbsG1MsmInput{
          points, scalar_rows + row * term_count, term_count,
          point_outs[row].data};
    }
    return cubbs_bbs_blst_g1_msm_common_points_batch(
        blst_items.data(), output_count, results);
  }
  if (benchmark_route ==
      CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE) {
#if defined(CUBBS_BBS_ENABLE_SPPARK_REFERENCE)
    std::vector<CuBbsBbsGpuByteSlice> reference_scalar_rows;
    const CuBbsBbsGpuByteSlice *reference_scalars = scalar_rows;
    if (has_packed_scalars) {
      reference_scalar_rows.resize(output_count * term_count);
      for (size_t row = 0u; row < output_count; ++row) {
        for (size_t term = 0u; term < term_count; ++term) {
          reference_scalar_rows[row * term_count + term] =
              CuBbsBbsGpuByteSlice{
                  scalar_rows_be + row * scalar_row_stride + term * 32u, 32u};
        }
      }
      reference_scalars = reference_scalar_rows.data();
    }
    std::vector<CuBbsG1MsmInput> reference_items(output_count);
    for (size_t row = 0u; row < output_count; ++row) {
      reference_items[row] = CuBbsG1MsmInput{
          points, reference_scalars + row * term_count, term_count,
          point_outs[row].data};
    }
    return compute_g1_msm_sppark_reference_batch(
        reference_items.data(), output_count, results, benchmark_route);
#else
    for (size_t row = 0u; row < output_count; ++row) {
      set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
    }
    return CUBBS_BBS_GPU_SUCCESS;
#endif
  }

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  if (bbs_homogeneous_kernel_enabled() && output_count > 0u &&
      bbs_specialized_common_points_supported(term_count) &&
      output_count >= bbs_homogeneous_kernel_min_items(term_count)) {
    return compute_g1_msm_fused_common_points_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results);
  }

  if (bbs_large_shape_kernel_supported(term_count) && output_count > 0u) {
    return compute_g1_msm_fused_common_points_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results, nullptr, nullptr, true);
  }
  if (bbs_universal_tiled_common_supported(term_count) && output_count > 0u) {
    return compute_g1_msm_fused_common_points_with_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results, nullptr, nullptr, true, true);
  }
  return compute_g1_msm_universal_common_points_with_cuda(
      points, term_count, scalar_rows, output_count, scalar_rows_be,
      scalar_row_stride, point_outs, results);
#else
  for (size_t row = 0u; row < output_count; ++row) {
    set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
  }
  return CUBBS_BBS_GPU_SUCCESS;
#endif
}

int32_t cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  const bool has_packed_scalars = scalar_rows_be != nullptr;
  if ((term_count > 0 && points == nullptr) ||
      (output_count > 0 &&
       ((!has_packed_scalars && scalar_rows == nullptr) || point_outs == nullptr ||
        results == nullptr)) ||
      (has_packed_scalars && scalar_row_stride < term_count * 32u)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t term = 0; term < term_count; ++term) {
    if (points[term].len != 48u || points[term].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }
  for (size_t row = 0; row < output_count; ++row) {
    if (point_outs[row].data == nullptr || point_outs[row].capacity < 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (!has_packed_scalars) {
      for (size_t term = 0; term < term_count; ++term) {
        const size_t scalar_index = (row * term_count) + term;
        if (scalar_rows[scalar_index].len != 32u ||
            scalar_rows[scalar_index].data == nullptr) {
          return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
        }
      }
    }
  }

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  const bool large_window4 = bbs_large_shape_kernel_supported(term_count);
  const bool universal_window4 =
      !large_window4 &&
      !bbs_specialized_common_points_supported(term_count) &&
      bbs_universal_tiled_common_supported(term_count);
  if (!bbs_specialized_common_points_supported(term_count) && !large_window4 &&
      !universal_window4) {
    return cubbs_bbs_cuda_g1_msm_common_points_matrix(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results);
  }
  // The naive scaffold (32-thread, default-stream, serial decode/compress) is
  // kept as an opt-in correctness reference. The production body is the
  // default: higher occupancy, per-point-set CUDA stream, and parallel
  // host-side scalar decode / output compression.
  if (!large_window4 && !universal_window4 &&
      env_flag_enabled("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_NAIVE")) {
    return compute_g1_msm_fused_common_points_with_naive_cuda(
        points, term_count, scalar_rows, output_count, scalar_rows_be,
        scalar_row_stride, point_outs, results);
  }
  return compute_g1_msm_fused_common_points_with_cuda(
      points, term_count, scalar_rows, output_count, scalar_rows_be,
      scalar_row_stride, point_outs, results, nullptr, nullptr,
      large_window4 || universal_window4, universal_window4);
#else
  for (size_t row = 0; row < output_count; ++row) {
    set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
  }
  return CUBBS_BBS_GPU_SUCCESS;
#endif
}

int32_t cubbs_bbs_cuda_g1_msm_fused_common_points_plus_variable_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const CuBbsBbsGpuByteSlice *variable_points,
    const CuBbsBbsGpuByteSlice *variable_scalars,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  if ((term_count > 0u && (points == nullptr || scalar_rows == nullptr)) ||
      (output_count > 0u &&
       (variable_points == nullptr || variable_scalars == nullptr ||
        point_outs == nullptr || results == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  const bool large_window4 = bbs_large_shape_kernel_supported(term_count);
  const bool universal_window4 =
      !large_window4 &&
      !bbs_specialized_common_points_supported(term_count) &&
      bbs_universal_tiled_common_supported(term_count);
  if (!bbs_specialized_common_points_supported(term_count) && !large_window4 &&
      !universal_window4) {
    const size_t combined_term_count = term_count + 1u;
    if (combined_term_count <= term_count ||
        (output_count > 0u &&
         combined_term_count > static_cast<size_t>(-1) / output_count)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    std::vector<CuBbsBbsGpuByteSlice> combined_points(
        output_count * combined_term_count);
    std::vector<CuBbsBbsGpuByteSlice> combined_scalars(
        output_count * combined_term_count);
    std::vector<CuBbsG1MsmInput> msm_inputs(output_count);
    for (size_t row = 0u; row < output_count; ++row) {
      const size_t offset = row * combined_term_count;
      std::copy(points, points + term_count,
                combined_points.begin() + offset);
      std::copy(scalar_rows + row * term_count,
                scalar_rows + (row + 1u) * term_count,
                combined_scalars.begin() + offset);
      combined_points[offset + term_count] = variable_points[row];
      combined_scalars[offset + term_count] = variable_scalars[row];
      msm_inputs[row] = CuBbsG1MsmInput{
          combined_points.data() + offset,
          combined_scalars.data() + offset,
          combined_term_count,
          point_outs[row].data,
      };
    }
    return cubbs_bbs_cuda_g1_msm_batch(msm_inputs.data(), output_count,
                                        results);
  }
  return compute_g1_msm_fused_common_points_with_cuda(
      points, term_count, scalar_rows, output_count, nullptr, 0u, point_outs,
      results, variable_points, variable_scalars,
      large_window4 || universal_window4, universal_window4);
#else
  for (size_t row = 0; row < output_count; ++row) {
    set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
  }
  return CUBBS_BBS_GPU_SUCCESS;
#endif
}

int32_t cubbs_bbs_cuda_prepare_b_msm_batch(const CuBbsPrepareBInput *items,
                                            size_t item_count,
                                            CuBbsBbsGpuResult *results) {
  if (item_count > 0 && (items == nullptr || results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
  // TODO: Bridge cuBBS-Cred BBS data into sppark:
  // 1. derive or receive P1/Q1/H_i as compressed G1;
  // 2. decompress into cubbs_sppark_affine_t layout;
  // 3. convert big-endian scalar bytes into cubbs_sppark_scalar_t;
  // 4. launch the cuBBS-Cred row-wise GLV accumulation kernel;
  // 5. compress cubbs_sppark_point_t back to 48-byte G1.
  //
  // Keeping this branch explicit prevents the ABI from claiming acceleration
  // before the point/scalar format bridge is byte-for-byte compatible.
  (void)items;
#else
  (void)items;
#endif

  for (size_t i = 0; i < item_count; ++i) {
    set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}
