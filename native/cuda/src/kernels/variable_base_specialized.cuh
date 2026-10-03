#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

__global__ static void cubbs_bbs_g1_add_variable_term_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    size_t item_count, cubbs_sppark_bucket_t *outputs) {
  const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= item_count) return;
  cubbs_sppark_bucket_t base;
  base = bases[item];
  uint8_t remainder_le[16];
  uint8_t quotient_le[16];
  cubbs_bbs_scalar_div_by_z2_device(
      scalars_le + (item * 32u), remainder_le, quotient_le);
  auto variable = cubbs_bbs_g1_scalar_mul_glv_device(
      base, remainder_le, quotient_le);
  outputs[item].add(variable);
}

__global__ static void cubbs_bbs_g1_add_variable_term_radix16_kernel(
    const cubbs_sppark_bucket_t *tables, const uint8_t *scalars_le,
    size_t item_count, cubbs_sppark_bucket_t *outputs) {
  const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= item_count) return;
  uint8_t remainder_le[16];
  uint8_t quotient_le[16];
  cubbs_bbs_scalar_div_by_z2_device(
      scalars_le + (item * 32u), remainder_le, quotient_le);
  auto variable = cubbs_bbs_g1_scalar_mul_glv_signed_radix16_shared(
      tables + item * 9u, remainder_le, quotient_le);
  outputs[item].add(variable);
}

__global__ static void cubbs_bbs_g1_variable_base_single_term_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    size_t term_count, size_t item_count, cubbs_sppark_bucket_t *outputs) {
  const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= item_count) return;
  cubbs_sppark_bucket_t base;
  base = bases[(item * term_count) + (term_count - 1u)];
  uint8_t remainder_le[16];
  uint8_t quotient_le[16];
  const uint8_t *scalar = scalars_le + (item * 32u);
  cubbs_bbs_scalar_div_by_z2_device(scalar, remainder_le, quotient_le);
  auto result = cubbs_bbs_g1_scalar_mul_glv_device(
      base, remainder_le, quotient_le);
  if (term_count == 2u) {
    cubbs_sppark_bucket_t first;
    first = bases[item * 2u];
    first.add(result);
    result = first;
  }
  outputs[item] = result;
}

struct CuBbsVariableBaseDeviceWorkspace {
  std::mutex mutex;
  size_t capacity{0u};
  cubbs_sppark_plain_affine_t *bases{nullptr};
  uint8_t *scalars{nullptr};
  cubbs_sppark_bucket_t *outputs{nullptr};
};

static CuBbsVariableBaseDeviceWorkspace &cubbs_variable_base_workspace() {
  static auto *workspace = new CuBbsVariableBaseDeviceWorkspace();
  return *workspace;
}

static int32_t cubbs_bbs_cuda_g1_variable_base_single_term_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  const size_t term_count = item_count == 0u ? 0u : items[0].term_count;
  if (term_count != 1u && term_count != 2u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t point_count = item_count * term_count;
  std::vector<CuBbsBbsGpuByteSlice> point_slices(point_count);
  std::vector<cubbs_sppark_plain_affine_t> host_bases(point_count);
  std::vector<uint8_t> host_scalars(item_count * 32u);
  for (size_t i = 0; i < item_count; ++i) {
    if (items[i].term_count != term_count || items[i].points == nullptr ||
        items[i].scalars == nullptr || items[i].compressed_point_out == nullptr ||
        items[i].scalars[term_count - 1u].len != 32u ||
        items[i].scalars[term_count - 1u].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t term = 0; term < term_count; ++term) {
      if (items[i].points[term].len != 48u ||
          items[i].points[term].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      point_slices[(i * term_count) + term] = items[i].points[term];
    }
    const auto &scalar = items[i].scalars[term_count - 1u];
    std::reverse_copy(scalar.data, scalar.data + 32u,
                      host_scalars.data() + (i * 32u));
  }
  const int32_t decode_status = cubbs_bbs_blst_decode_g1_points(
      point_slices.data(), point_count,
      reinterpret_cast<uint8_t *>(host_bases.data()),
      sizeof(cubbs_sppark_plain_affine_t));
  if (decode_status != CUBBS_BBS_GPU_SUCCESS) return decode_status;

  auto &workspace = cubbs_variable_base_workspace();
  std::lock_guard<std::mutex> lock(workspace.mutex);
  cudaError_t error = cudaSuccess;
  if (workspace.capacity < item_count) {
    if (workspace.bases != nullptr) cudaFree(workspace.bases);
    if (workspace.scalars != nullptr) cudaFree(workspace.scalars);
    if (workspace.outputs != nullptr) cudaFree(workspace.outputs);
    workspace.capacity = 0u;
    workspace.bases = nullptr;
    workspace.scalars = nullptr;
    workspace.outputs = nullptr;
    error = cudaMalloc(&workspace.bases,
                       2u * item_count * sizeof(*workspace.bases));
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.scalars, host_scalars.size());
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.outputs,
                         item_count * sizeof(*workspace.outputs));
    if (error == cudaSuccess) workspace.capacity = item_count;
  }
  if (error == cudaSuccess)
    error = cudaMemcpy(workspace.bases, host_bases.data(),
                       point_count * sizeof(*workspace.bases),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(workspace.scalars, host_scalars.data(),
                       host_scalars.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    constexpr uint32_t threads = 128u;
    const uint32_t blocks =
        static_cast<uint32_t>((item_count + threads - 1u) / threads);
    cubbs_bbs_g1_variable_base_single_term_kernel<<<blocks, threads>>>(
        workspace.bases, workspace.scalars, term_count, item_count,
        workspace.outputs);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<cubbs_sppark_bucket_t> host_outputs(item_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(host_outputs.data(), workspace.outputs,
                       item_count * sizeof(host_outputs[0]),
                       cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  std::vector<cubbs_sppark_point_t> projectives(item_count);
  std::vector<const void *> projective_ptrs(item_count);
  std::vector<uint8_t *> compressed_outs(item_count);
  for (size_t i = 0; i < item_count; ++i) {
    projectives[i] = static_cast<cubbs_sppark_point_t>(host_outputs[i]);
    projective_ptrs[i] = &projectives[i];
    compressed_outs[i] = items[i].compressed_point_out;
  }
  const int32_t compression_status =
      cubbs_bbs_blst_compress_g1_projective_batch(
          projective_ptrs.data(), compressed_outs.data(), item_count);
  for (size_t i = 0; i < item_count; ++i) {
    if (compression_status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_FUSED_ITEM);
    } else {
      set_result(results, i, compression_status, 0u);
    }
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

// BBS proof verification T1 has exactly three variable-base terms:
//   challenge*Bbar + e_cap*Abar + r1_cap*D.
// One warp handles ten proofs. Three lanes per proof perform independent GLV
// scalar multiplications and the first lane merges their results. This packs
// 30/32 lanes instead of the old 24/32 layout.
__global__ static void cubbs_bbs_g1_variable_base_three_term_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    size_t item_count, cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) unsigned char
      stage_storage[32u * sizeof(cubbs_sppark_bucket_t)];
  auto *stage = reinterpret_cast<cubbs_sppark_bucket_t *>(stage_storage);
  const uint32_t warp_lane = threadIdx.x & 31u;
  const uint32_t proof_lane = warp_lane % 3u;
  const uint32_t local_proof = warp_lane / 3u;
  const size_t item = static_cast<size_t>(blockIdx.x) * 10u + local_proof;
  stage[warp_lane].inf();
  if (local_proof < 10u && item < item_count) {
    cubbs_sppark_bucket_t base;
    base = bases[(item * 3u) + proof_lane];
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    const uint8_t *scalar =
        scalars_le + (((item * 3u) + proof_lane) * 32u);
    cubbs_bbs_scalar_div_by_z2_device(scalar, remainder_le, quotient_le);
    stage[warp_lane] = cubbs_bbs_g1_scalar_mul_glv_device(
        base, remainder_le, quotient_le);
  }
  __syncwarp();
  if (local_proof < 10u && item < item_count && proof_lane == 0u) {
    auto result = stage[warp_lane];
    result.add(stage[warp_lane + 1u]);
    result.add(stage[warp_lane + 2u]);
    outputs[item] = result;
  }
}

struct CuBbsThreeTermDeviceWorkspace {
  std::mutex mutex;
  size_t capacity{0u};
  cubbs_sppark_plain_affine_t *bases{nullptr};
  uint8_t *scalars{nullptr};
  cubbs_sppark_bucket_t *outputs{nullptr};
};

static CuBbsThreeTermDeviceWorkspace &cubbs_three_term_workspace() {
  static auto *workspace = new CuBbsThreeTermDeviceWorkspace();
  return *workspace;
}

static int32_t cubbs_bbs_cuda_g1_variable_base_three_term_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  constexpr size_t term_count = 3u;
  const size_t point_count = item_count * term_count;
  std::vector<CuBbsBbsGpuByteSlice> point_slices(point_count);
  std::vector<cubbs_sppark_plain_affine_t> host_bases(point_count);
  std::vector<uint8_t> host_scalars(point_count * 32u);
  for (size_t i = 0; i < item_count; ++i) {
    if (items[i].term_count != term_count || items[i].points == nullptr ||
        items[i].scalars == nullptr || items[i].compressed_point_out == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t term = 0; term < term_count; ++term) {
      const size_t offset = (i * term_count) + term;
      if (items[i].points[term].len != 48u ||
          items[i].points[term].data == nullptr ||
          items[i].scalars[term].len != 32u ||
          items[i].scalars[term].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      point_slices[offset] = items[i].points[term];
      std::reverse_copy(items[i].scalars[term].data,
                        items[i].scalars[term].data + 32u,
                        host_scalars.data() + (offset * 32u));
    }
  }
  const int32_t decode_status = cubbs_bbs_blst_decode_g1_points(
      point_slices.data(), point_count,
      reinterpret_cast<uint8_t *>(host_bases.data()),
      sizeof(cubbs_sppark_plain_affine_t));
  if (decode_status != CUBBS_BBS_GPU_SUCCESS) return decode_status;

  auto &workspace = cubbs_three_term_workspace();
  std::lock_guard<std::mutex> lock(workspace.mutex);
  cudaError_t error = cudaSuccess;
  if (workspace.capacity < item_count) {
    if (workspace.bases != nullptr) cudaFree(workspace.bases);
    if (workspace.scalars != nullptr) cudaFree(workspace.scalars);
    if (workspace.outputs != nullptr) cudaFree(workspace.outputs);
    workspace.capacity = 0u;
    workspace.bases = nullptr;
    workspace.scalars = nullptr;
    workspace.outputs = nullptr;
    error = cudaMalloc(&workspace.bases,
                       point_count * sizeof(*workspace.bases));
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.scalars, host_scalars.size());
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.outputs,
                         item_count * sizeof(*workspace.outputs));
    if (error == cudaSuccess) workspace.capacity = item_count;
  }
  if (error == cudaSuccess)
    error = cudaMemcpy(workspace.bases, host_bases.data(),
                       point_count * sizeof(*workspace.bases),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(workspace.scalars, host_scalars.data(),
                       host_scalars.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    const uint32_t blocks = static_cast<uint32_t>((item_count + 9u) / 10u);
    cubbs_bbs_g1_variable_base_three_term_kernel<<<blocks, 32u>>>(
        workspace.bases, workspace.scalars, item_count, workspace.outputs);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<cubbs_sppark_bucket_t> host_outputs(item_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(host_outputs.data(), workspace.outputs,
                       item_count * sizeof(host_outputs[0]),
                       cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  std::vector<cubbs_sppark_point_t> projectives(item_count);
  std::vector<const void *> projective_ptrs(item_count);
  std::vector<uint8_t *> compressed_outs(item_count);
  for (size_t i = 0; i < item_count; ++i) {
    projectives[i] = static_cast<cubbs_sppark_point_t>(host_outputs[i]);
    projective_ptrs[i] = &projectives[i];
    compressed_outs[i] = items[i].compressed_point_out;
  }
  const int32_t compression_status =
      cubbs_bbs_blst_compress_g1_projective_batch(
          projective_ptrs.data(), compressed_outs.data(), item_count);
  for (size_t i = 0; i < item_count; ++i) {
    if (compression_status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_FUSED_ITEM);
    } else {
      set_result(results, i, compression_status, 0u);
    }
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

// Fused BBS proof-verify MSM. One warp handles eight proofs. Each proof builds
// small shared window tables for Bbar, Abar and D; four lanes then reuse those
// tables for challenge*Bbar, e_cap*Abar, r1_cap*D and r3_cap*D.
template <bool kHostGlvSplit>
__global__ static void cubbs_bbs_proof_verify_msm_pair_direct_kernel(
    const cubbs_sppark_plain_affine_t *proof_bases,
    const uint8_t *proof_scalars_le, size_t item_count,
    cubbs_sppark_bucket_t *t1_outputs,
    cubbs_sppark_bucket_t *t2_outputs, bool add_to_t2) {
  __shared__ __align__(16) unsigned char stage_storage[
      32u * sizeof(cubbs_sppark_bucket_t)];
  auto *stage = reinterpret_cast<cubbs_sppark_bucket_t *>(stage_storage);
  const uint32_t warp_lane = threadIdx.x & 31u;
  const uint32_t proof_lane = warp_lane & 3u;
  const size_t item =
      static_cast<size_t>(blockIdx.x) * 8u + (warp_lane >> 2u);
  stage[warp_lane].inf();
  if (item < item_count) {
    const uint32_t base_index = proof_lane < 3u ? proof_lane : 2u;
    cubbs_sppark_bucket_t base;
    base = proof_bases[item * 3u + base_index];
    const uint8_t *scalar =
        proof_scalars_le + (((item * 4u) + proof_lane) * 32u);
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    if constexpr (kHostGlvSplit) {
      memcpy(remainder_le, scalar, 16u);
      memcpy(quotient_le, scalar + 16u, 16u);
    } else {
      cubbs_bbs_scalar_div_by_z2_device(scalar, remainder_le, quotient_le);
    }
    stage[warp_lane] = cubbs_bbs_g1_scalar_mul_glv_device(
        base, remainder_le, quotient_le);
  }
  __syncwarp();
  if (item < item_count && proof_lane == 0u) {
    auto t1 = stage[warp_lane];
    t1.add(stage[warp_lane + 1u]);
    t1.add(stage[warp_lane + 2u]);
    t1_outputs[item] = t1;
    if (add_to_t2) t2_outputs[item].add(stage[warp_lane + 3u]);
    else t2_outputs[item] = stage[warp_lane + 3u];
  }
}

template <bool kHostGlvSplit>
__global__ static void cubbs_bbs_proof_verify_msm_pair_kernel(
    const cubbs_sppark_plain_affine_t *proof_bases,
    const uint8_t *proof_scalars_le, size_t item_count,
    cubbs_sppark_bucket_t *t1_outputs,
    cubbs_sppark_bucket_t *t2_outputs, bool add_to_t2) {
  constexpr uint32_t proofs_per_block = 8u;
  constexpr uint32_t bases_per_proof = 3u;
  constexpr uint32_t table_size = 9u;
  __shared__ __align__(16) unsigned char table_storage[
      proofs_per_block * bases_per_proof * table_size *
      sizeof(cubbs_sppark_bucket_t)];
  __shared__ __align__(16) unsigned char stage_storage[
      32u * sizeof(cubbs_sppark_bucket_t)];
  auto *tables = reinterpret_cast<cubbs_sppark_bucket_t *>(table_storage);
  auto *stage = reinterpret_cast<cubbs_sppark_bucket_t *>(stage_storage);
  const uint32_t warp_lane = threadIdx.x & 31u;
  const uint32_t proof_lane = warp_lane & 3u;
  const size_t item =
      static_cast<size_t>(blockIdx.x) * proofs_per_block + (warp_lane >> 2u);
  if (warp_lane < proofs_per_block * bases_per_proof) {
    const uint32_t local_proof = warp_lane / bases_per_proof;
    const uint32_t base_index = warp_lane % bases_per_proof;
    const size_t table_item =
        static_cast<size_t>(blockIdx.x) * proofs_per_block + local_proof;
    if (table_item < item_count) {
      auto *table = tables +
          ((local_proof * bases_per_proof + base_index) * table_size);
      table[0].inf();
      table[1] = proof_bases[table_item * bases_per_proof + base_index];
      #pragma unroll
      for (uint32_t digit = 2u; digit < table_size; ++digit) {
        table[digit] = table[digit - 1u];
        table[digit].add(table[1]);
      }
    }
  }
  __syncwarp();
  stage[warp_lane].inf();
  if (item < item_count) {
    const uint32_t base_index = proof_lane < 3u ? proof_lane : 2u;
    const uint8_t *scalar =
        proof_scalars_le + (((item * 4u) + proof_lane) * 32u);
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    if constexpr (kHostGlvSplit) {
      memcpy(remainder_le, scalar, 16u);
      memcpy(quotient_le, scalar + 16u, 16u);
    } else {
      cubbs_bbs_scalar_div_by_z2_device(scalar, remainder_le, quotient_le);
    }
    const uint32_t local_proof = warp_lane >> 2u;
    stage[warp_lane] = cubbs_bbs_g1_scalar_mul_glv_signed_radix16_shared(
        tables + ((local_proof * bases_per_proof + base_index) * table_size),
        remainder_le, quotient_le);
  }
  __syncwarp();
  if (item < item_count && proof_lane == 0u) {
    auto t1 = stage[warp_lane];
    t1.add(stage[warp_lane + 1u]);
    t1.add(stage[warp_lane + 2u]);
    t1_outputs[item] = t1;
    if (add_to_t2) t2_outputs[item].add(stage[warp_lane + 3u]);
    else t2_outputs[item] = stage[warp_lane + 3u];
  }
}

__global__ static void cubbs_bbs_proof_verify_radix16_tables_kernel(
    const cubbs_sppark_plain_affine_t *proof_bases, size_t proof_point_count,
    cubbs_sppark_bucket_t *tables) {
  constexpr uint32_t table_size = 9u;
  const size_t point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= proof_point_count) return;
  auto *table = tables + point * table_size;
  table[0].inf();
  table[1] = proof_bases[point];
  #pragma unroll
  for (uint32_t digit = 2u; digit < table_size; ++digit) {
    table[digit] = table[digit - 1u];
    table[digit].add(table[1]);
  }
}

template <bool kHostGlvSplit>
__global__ static void cubbs_bbs_proof_verify_msm_pair_global_table_kernel(
    const cubbs_sppark_bucket_t *tables,
    const uint8_t *proof_scalars_le, size_t item_count,
    cubbs_sppark_bucket_t *t1_outputs,
    cubbs_sppark_bucket_t *t2_outputs, bool add_to_t2) {
  constexpr uint32_t table_size = 9u;
  __shared__ __align__(16) unsigned char stage_storage[
      32u * sizeof(cubbs_sppark_bucket_t)];
  auto *stage = reinterpret_cast<cubbs_sppark_bucket_t *>(stage_storage);
  const uint32_t warp_lane = threadIdx.x & 31u;
  const uint32_t proof_lane = warp_lane & 3u;
  const size_t item =
      static_cast<size_t>(blockIdx.x) * 8u + (warp_lane >> 2u);
  stage[warp_lane].inf();
  if (item < item_count) {
    const uint32_t base_index = proof_lane < 3u ? proof_lane : 2u;
    const uint8_t *scalar =
        proof_scalars_le + (((item * 4u) + proof_lane) * 32u);
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    if constexpr (kHostGlvSplit) {
      memcpy(remainder_le, scalar, 16u);
      memcpy(quotient_le, scalar + 16u, 16u);
    } else {
      cubbs_bbs_scalar_div_by_z2_device(scalar, remainder_le, quotient_le);
    }
    stage[warp_lane] = cubbs_bbs_g1_scalar_mul_glv_signed_radix16_shared(
        tables + ((item * 3u + base_index) * table_size),
        remainder_le, quotient_le);
  }
  __syncwarp();
  if (item < item_count && proof_lane == 0u) {
    auto t1 = stage[warp_lane];
    t1.add(stage[warp_lane + 1u]);
    t1.add(stage[warp_lane + 2u]);
    t1_outputs[item] = t1;
    if (add_to_t2) t2_outputs[item].add(stage[warp_lane + 3u]);
    else t2_outputs[item] = stage[warp_lane + 3u];
  }
}

static cudaError_t cubbs_bbs_launch_proof_verify_variable(
    bool direct_variable, bool host_glv_split,
    const cubbs_sppark_plain_affine_t *proof_bases,
    const cubbs_sppark_bucket_t *global_tables,
    const uint8_t *proof_scalars, size_t output_count,
    cubbs_sppark_bucket_t *t1, cubbs_sppark_bucket_t *t2,
    bool add_to_t2, cudaStream_t stream) {
  const uint32_t blocks = static_cast<uint32_t>((output_count + 7u) / 8u);
  if (global_tables != nullptr) {
    if (host_glv_split) {
      cubbs_bbs_proof_verify_msm_pair_global_table_kernel<true>
          <<<blocks, 32u, 0, stream>>>(
              global_tables, proof_scalars, output_count, t1, t2, add_to_t2);
    } else {
      cubbs_bbs_proof_verify_msm_pair_global_table_kernel<false>
          <<<blocks, 32u, 0, stream>>>(
              global_tables, proof_scalars, output_count, t1, t2, add_to_t2);
    }
  } else if (direct_variable) {
    if (host_glv_split) {
      cubbs_bbs_proof_verify_msm_pair_direct_kernel<true>
          <<<blocks, 32u, 0, stream>>>(
              proof_bases, proof_scalars, output_count, t1, t2, add_to_t2);
    } else {
      cubbs_bbs_proof_verify_msm_pair_direct_kernel<false>
          <<<blocks, 32u, 0, stream>>>(
              proof_bases, proof_scalars, output_count, t1, t2, add_to_t2);
    }
  } else if (host_glv_split) {
    cubbs_bbs_proof_verify_msm_pair_kernel<true>
        <<<blocks, 32u, 0, stream>>>(
            proof_bases, proof_scalars, output_count, t1, t2, add_to_t2);
  } else {
    cubbs_bbs_proof_verify_msm_pair_kernel<false>
        <<<blocks, 32u, 0, stream>>>(
            proof_bases, proof_scalars, output_count, t1, t2, add_to_t2);
  }
  return cudaGetLastError();
}

__global__ static void cubbs_bbs_g1_add_projective_outputs_kernel(
    cubbs_sppark_bucket_t *outputs,
    const cubbs_sppark_bucket_t *terms, size_t item_count) {
  const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item < item_count) outputs[item].add(terms[item]);
}

struct CuBbsG1AffineCache {
  std::shared_mutex mutex;
  std::unordered_map<std::string, cubbs_sppark_plain_affine_t> points;
};

static CuBbsG1AffineCache &cubbs_g1_affine_cache() {
  static auto *cache = new CuBbsG1AffineCache();
  return *cache;
}

static constexpr size_t CUBBS_G1_AFFINE_CACHE_MAX_POINTS = 16384u;

static void cache_proof_pair_affines(
    const CuBbsBbsGpuByteSlice *proof_points,
    const cubbs_sppark_plain_affine_t *proof_bases, size_t item_count) {
  auto &cache = cubbs_g1_affine_cache();
  std::unique_lock<std::shared_mutex> lock(cache.mutex);
  // Count only keys that are actually absent. The previous conservative
  // `size + batch_size` test cleared a warm cache even when every point in the
  // incoming proof batch was already present. A mixed 4096-item workload then
  // repeatedly evicted and decoded the same 12K instance points.
  std::unordered_set<std::string> missing;
  missing.reserve(item_count * 3u);
  for (size_t point = 0u; point < item_count * 3u; ++point) {
    const auto &slice = proof_points[point];
    if (slice.data == nullptr || slice.len != 48u) continue;
    std::string key(reinterpret_cast<const char *>(slice.data), 48u);
    if (cache.points.find(key) == cache.points.end()) {
      missing.insert(std::move(key));
    }
  }
  if (cache.points.size() + missing.size() >
      CUBBS_G1_AFFINE_CACHE_MAX_POINTS) {
    cache.points.clear();
  }
  for (size_t item = 0u; item < item_count; ++item) {
    const auto &bbar = proof_points[item * 3u];
    const auto &abar = proof_points[item * 3u + 1u];
    const auto &d = proof_points[item * 3u + 2u];
    cache.points.insert_or_assign(
        std::string(reinterpret_cast<const char *>(abar.data), 48u),
        proof_bases[item * 3u + 1u]);
    cache.points.insert_or_assign(
        std::string(reinterpret_cast<const char *>(bbar.data), 48u),
        proof_bases[item * 3u]);
    cache.points.insert_or_assign(
        std::string(reinterpret_cast<const char *>(d.data), 48u),
        proof_bases[item * 3u + 2u]);
  }
}

static bool load_cached_g1_affines(
    const CuBbsBbsGpuByteSlice *points, size_t point_count,
    cubbs_sppark_plain_affine_t *points_out) {
  auto &cache = cubbs_g1_affine_cache();
  std::shared_lock<std::shared_mutex> lock(cache.mutex);
  for (size_t index = 0u; index < point_count; ++index) {
    if (points[index].len != 48u || points[index].data == nullptr) return false;
    const auto cached = cache.points.find(std::string(
        reinterpret_cast<const char *>(points[index].data), 48u));
    if (cached == cache.points.end()) return false;
    points_out[index] = cached->second;
  }
  return true;
}

// Resolve an arbitrary batch of compressed G1 points through the process-wide
// affine cache. Cache misses are deduplicated before BLST validation, so a
// repeated issuer/signature point is decoded only once even on the cold path.
static int32_t load_or_decode_cached_g1_affines(
    const CuBbsBbsGpuByteSlice *points, size_t point_count,
    cubbs_sppark_plain_affine_t *points_out) {
  if ((point_count > 0u && (points == nullptr || points_out == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  std::vector<std::string> miss_keys;
  std::vector<CuBbsBbsGpuByteSlice> miss_points;
  std::vector<std::vector<size_t>> miss_destinations;
  std::unordered_map<std::string, size_t> misses_by_key;
  miss_keys.reserve(point_count);
  miss_points.reserve(point_count);
  miss_destinations.reserve(point_count);
  misses_by_key.reserve(point_count);

  auto &cache = cubbs_g1_affine_cache();
  {
    std::shared_lock<std::shared_mutex> lock(cache.mutex);
    for (size_t index = 0u; index < point_count; ++index) {
      if (points[index].data == nullptr || points[index].len != 48u) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      std::string key(reinterpret_cast<const char *>(points[index].data), 48u);
      const auto cached = cache.points.find(key);
      if (cached != cache.points.end()) {
        points_out[index] = cached->second;
        continue;
      }
      const auto [found, inserted] =
          misses_by_key.emplace(key, miss_keys.size());
      if (inserted) {
        miss_keys.push_back(std::move(key));
        miss_points.push_back(points[index]);
        miss_destinations.emplace_back();
      }
      miss_destinations[found->second].push_back(index);
    }
  }

  if (miss_points.empty()) return CUBBS_BBS_GPU_SUCCESS;

  std::vector<cubbs_sppark_plain_affine_t> decoded(miss_points.size());
  const int32_t status = cubbs_bbs_blst_decode_g1_points(
      miss_points.data(), miss_points.size(),
      reinterpret_cast<uint8_t *>(decoded.data()),
      sizeof(cubbs_sppark_plain_affine_t));
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;

  for (size_t miss = 0u; miss < decoded.size(); ++miss) {
    for (const size_t destination : miss_destinations[miss]) {
      points_out[destination] = decoded[miss];
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(cache.mutex);
    if (cache.points.size() + decoded.size() >
        CUBBS_G1_AFFINE_CACHE_MAX_POINTS) {
      cache.points.clear();
    }
    for (size_t miss = 0u; miss < decoded.size(); ++miss) {
      cache.points.insert_or_assign(std::move(miss_keys[miss]), decoded[miss]);
    }
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

static bool load_cached_proof_pair_affines(
    const CuBbsBbsGpuByteSlice *lhs_points,
    const CuBbsBbsGpuByteSlice *rhs_points, size_t item_count,
    std::vector<cubbs_sppark_plain_affine_t> *points_out) {
  auto &cache = cubbs_g1_affine_cache();
  std::shared_lock<std::shared_mutex> lock(cache.mutex);
  points_out->resize(item_count * 2u);
  for (size_t item = 0u; item < item_count; ++item) {
    if (lhs_points[item].len != 48u || rhs_points[item].len != 48u ||
        lhs_points[item].data == nullptr || rhs_points[item].data == nullptr) {
      return false;
    }
    const auto lhs = cache.points.find(std::string(
        reinterpret_cast<const char *>(lhs_points[item].data), 48u));
    const auto rhs = cache.points.find(std::string(
        reinterpret_cast<const char *>(rhs_points[item].data), 48u));
    if (lhs == cache.points.end() || rhs == cache.points.end()) return false;
    (*points_out)[item] = lhs->second;
    (*points_out)[item_count + item] = rhs->second;
  }
  return true;
}

#endif
