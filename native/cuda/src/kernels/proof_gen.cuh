#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

// Device kernel: one warp handles five proofs. Each proof builds window-4
// tables for B and A once, then reuses them across six GLV multiplications.
// Per-proof scalar layout (7 x 32 bytes LE):
//   [0] r2                 [1] r1r2
//   [2] r1r2               [3] r1r2*e
//   [4] r1r2*e_tilde       [5] r2*r1_tilde
//   [6] r3_tilde (returned to host, not used here)
//
// Computed points:
//   d    = b * r2
//   abar = A * r1r2
//   bbar = b*(r1r2) - A*(r1r2*e)
//   t1   = A*(r1r2*e_tilde) + b*(r2*r1_tilde)
template <uint32_t WINDOW_BITS>
__global__ static void cubbs_bbs_proof_gen_fixed_muls_kernel(
    const cubbs_sppark_plain_affine_t *bases,
    const uint8_t *glv_splits,  // item_count * 6 * (remainder || quotient)
    cubbs_sppark_bucket_t *d_out,
    cubbs_sppark_bucket_t *abar_out,
    cubbs_sppark_bucket_t *bbar_out,
    cubbs_sppark_bucket_t *t1_out,
    size_t item_count) {
  constexpr uint32_t proofs_per_block = 5u;
  constexpr uint32_t table_size = 1u << WINDOW_BITS;
  const uint32_t local_proof = threadIdx.x / 6u;
  const uint32_t lane = threadIdx.x % 6u;
  const size_t proof = blockIdx.x * proofs_per_block + local_proof;
  const bool valid = local_proof < proofs_per_block && proof < item_count;
  const uint8_t *s = nullptr;
  if (valid) {
    s = glv_splits + (proof * 6u * 32u);
  }

  __shared__ __align__(16) uint8_t table_storage[
      proofs_per_block * 2u * table_size * sizeof(cubbs_sppark_bucket_t)];
  __shared__ __align__(16) uint8_t
      stage_storage[proofs_per_block * 4u * sizeof(cubbs_sppark_bucket_t)];
  auto *tables = reinterpret_cast<cubbs_sppark_bucket_t *>(table_storage);
  auto *all_stages = reinterpret_cast<cubbs_sppark_bucket_t *>(stage_storage);
  auto *stage = all_stages + local_proof * 4u;

  if (threadIdx.x < proofs_per_block * 2u) {
    const uint32_t table_proof = threadIdx.x / 2u;
    const uint32_t base_index = threadIdx.x & 1u;
    const size_t table_item = blockIdx.x * proofs_per_block + table_proof;
    if (table_item < item_count) {
      auto *table = tables + ((table_proof * 2u + base_index) * table_size);
      table[0].inf();
      table[1] = bases[table_item * 2u + base_index];
      #pragma unroll
      for (uint32_t digit = 2u; digit < table_size; ++digit) {
        table[digit] = table[digit - 1u];
        table[digit].add(table[1]);
      }
    }
  }
  __syncwarp();

  if (valid) {
    const uint32_t base_index =
        (lane == 0u || lane == 2u || lane == 5u) ? 0u : 1u;
    const uint8_t *remainder_le = s + (lane * 32u);
    const uint8_t *quotient_le = remainder_le + 16u;
    auto product = cubbs_bbs_g1_scalar_mul_glv_window_shared<WINDOW_BITS>(
        tables + ((local_proof * 2u + base_index) * table_size),
        remainder_le, quotient_le);
    if (lane == 0u) {
      d_out[proof] = product;
    } else if (lane == 1u) {
      abar_out[proof] = product;
    } else {
      if (lane == 3u) product.cneg(true);
      stage[lane - 2u] = product;
    }
  }
  __syncwarp();

  if (valid && lane == 2u) {
    stage[0].add(stage[1]);
    bbar_out[proof] = stage[0];
  } else if (valid && lane == 4u) {
    stage[2].add(stage[3]);
    t1_out[proof] = stage[2];
  }
}

__global__ static void cubbs_bbs_proof_gen_radix256_tables_kernel(
    const cubbs_sppark_plain_affine_t *bases, size_t point_count,
    cubbs_sppark_bucket_t *tables) {
  constexpr uint32_t table_size = 129u;
  const size_t point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= point_count) return;
  auto *table = tables + point * table_size;
  table[0].inf();
  table[1] = bases[point];
  #pragma unroll 1
  for (uint32_t digit = 2u; digit < table_size; ++digit) {
    table[digit] = table[digit - 1u];
    table[digit].add(table[1]);
  }
}

__global__ static void cubbs_bbs_proof_gen_fixed_muls_global_table_kernel(
    const cubbs_sppark_bucket_t *tables,
    const uint8_t *glv_splits,
    cubbs_sppark_bucket_t *d_out,
    cubbs_sppark_bucket_t *abar_out,
    cubbs_sppark_bucket_t *bbar_out,
    cubbs_sppark_bucket_t *t1_out,
    size_t item_count) {
  constexpr uint32_t proofs_per_block = 5u;
  const uint32_t local_proof = threadIdx.x / 6u;
  const uint32_t lane = threadIdx.x % 6u;
  const size_t proof =
      static_cast<size_t>(blockIdx.x) * proofs_per_block + local_proof;
  const bool valid = local_proof < proofs_per_block && proof < item_count;
  __shared__ __align__(16) uint8_t
      stage_storage[proofs_per_block * 4u *
                    sizeof(cubbs_sppark_bucket_t)];
  auto *stages =
      reinterpret_cast<cubbs_sppark_bucket_t *>(stage_storage);
  auto *stage = stages + local_proof * 4u;
  if (valid) {
    const uint32_t base_index =
        (lane == 0u || lane == 2u || lane == 5u) ? 0u : 1u;
    const uint8_t *components =
        glv_splits + ((proof * 6u + lane) * 32u);
    auto product =
        cubbs_bbs_g1_scalar_mul_glv_signed_radix256_shared(
            tables + ((proof * 2u + base_index) * 129u),
            components, components + 16u);
    if (lane == 0u) {
      d_out[proof] = product;
    } else if (lane == 1u) {
      abar_out[proof] = product;
    } else {
      if (lane == 3u) product.cneg(true);
      stage[lane - 2u] = product;
    }
  }
  __syncwarp();
  if (valid && lane == 2u) {
    stage[0].add(stage[1]);
    bbar_out[proof] = stage[0];
  } else if (valid && lane == 4u) {
    stage[2].add(stage[3]);
    t1_out[proof] = stage[2];
  }
}

// Host orchestration for the GPU proof-gen t2Prepare. The Rust caller
// precomputes the 7 host-side scalar multipliers (including r1*r2) and passes
// them as big-endian 32-byte blobs, so the device only needs G1 scalar
// multiplication (no device-side scalar-field arithmetic). Random scalars are
// also generated on the host (cost is ~3us/proof, negligible).
struct CuBbsProofGenT2DeviceWorkspace {
  std::mutex mutex;
  size_t capacity{0u};
  size_t table_size{0u};
  cudaStream_t stream{nullptr};
  cubbs_sppark_plain_affine_t *bases{nullptr};
  uint8_t *scalars{nullptr};
  cubbs_sppark_bucket_t *tables{nullptr};
  cubbs_sppark_bucket_t *outputs{nullptr};
  cubbs_sppark_plain_affine_t *host_bases{nullptr};
  uint8_t *host_scalars{nullptr};
  cubbs_sppark_bucket_t *host_outputs{nullptr};
  std::string point_key;
  std::string next_point_key;
  std::vector<CuBbsBbsGpuByteSlice> compressed_points;
  std::vector<cubbs_sppark_point_t> projectives;
  std::vector<const void *> projective_ptrs;
  std::vector<uint8_t *> compressed_outs;
};

static CuBbsProofGenT2DeviceWorkspace &cubbs_proof_gen_t2_workspace() {
  static auto *workspaces =
      new std::array<CuBbsProofGenT2DeviceWorkspace, 8u>();
  static std::atomic<size_t> next_workspace{0u};
  const size_t active_workspaces = std::min<size_t>(
      workspaces->size(),
      std::max<size_t>(
          1u, env_size_or_default(
                  "CUBBS_BBS_GPU_PROOF_GEN_WORKSPACES", 3u)));
  return (*workspaces)[next_workspace.fetch_add(1u, std::memory_order_relaxed) %
                       active_workspaces];
}

int32_t cubbs_bbs_cuda_proof_gen_t2_prepare(
    CuBbsBbsGpuProofGenT2PrepareItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (item_count > 0u && (items == nullptr || results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0u) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t n = item_count;
  const char *radix256_config =
      std::getenv("CUBBS_BBS_GPU_PROOF_GEN_RADIX256");
  const bool radix256_enabled = radix256_config == nullptr ||
      env_flag_enabled("CUBBS_BBS_GPU_PROOF_GEN_RADIX256");
  constexpr size_t radix256_table_size = 129u;
  constexpr size_t radix16_table_size = 9u;
  const size_t radix256_budget = env_size_or_default(
      "CUBBS_BBS_GPU_PROOF_GEN_RADIX256_MAX_BYTES",
      256u * 1024u * 1024u);
  const size_t radix256_max_items =
      radix256_budget /
      (2u * radix256_table_size * sizeof(cubbs_sppark_bucket_t));
  const bool use_radix256 = radix256_enabled && n <= radix256_max_items;
  const size_t table_size =
      use_radix256 ? radix256_table_size : radix16_table_size;
  auto &workspace = cubbs_proof_gen_t2_workspace();
  std::lock_guard<std::mutex> workspace_lock(workspace.mutex);
  cudaError_t err = cudaSuccess;
  if (workspace.stream == nullptr) {
    err = cudaStreamCreateWithFlags(&workspace.stream, cudaStreamNonBlocking);
  }
  if (err == cudaSuccess &&
      (workspace.capacity < n || workspace.table_size != table_size)) {
    if (workspace.bases != nullptr) cudaFree(workspace.bases);
    if (workspace.scalars != nullptr) cudaFree(workspace.scalars);
    if (workspace.tables != nullptr) cudaFree(workspace.tables);
    if (workspace.outputs != nullptr) cudaFree(workspace.outputs);
    if (workspace.host_bases != nullptr) cudaFreeHost(workspace.host_bases);
    if (workspace.host_scalars != nullptr) cudaFreeHost(workspace.host_scalars);
    if (workspace.host_outputs != nullptr) cudaFreeHost(workspace.host_outputs);
    workspace.capacity = 0u;
    workspace.table_size = 0u;
    workspace.bases = nullptr;
    workspace.scalars = nullptr;
    workspace.tables = nullptr;
    workspace.outputs = nullptr;
    workspace.host_bases = nullptr;
    workspace.host_scalars = nullptr;
    workspace.host_outputs = nullptr;
    workspace.point_key.clear();
    const size_t bases_bytes = 2u * n * sizeof(*workspace.bases);
    const size_t scalars_bytes = n * 6u * 32u;
    const size_t tables_bytes =
        2u * n * table_size * sizeof(*workspace.tables);
    const size_t outputs_bytes = 4u * n * sizeof(*workspace.outputs);
    err = cudaMalloc(&workspace.bases, bases_bytes);
    if (err == cudaSuccess)
      err = cudaMalloc(&workspace.scalars, scalars_bytes);
    if (err == cudaSuccess)
      err = cudaMalloc(&workspace.tables, tables_bytes);
    if (err == cudaSuccess)
      err = cudaMalloc(&workspace.outputs, outputs_bytes);
    if (err == cudaSuccess)
      err = cudaMallocHost(&workspace.host_bases, bases_bytes);
    if (err == cudaSuccess)
      err = cudaMallocHost(&workspace.host_scalars, scalars_bytes);
    if (err == cudaSuccess)
      err = cudaMallocHost(&workspace.host_outputs, outputs_bytes);
    if (err == cudaSuccess) {
      workspace.capacity = n;
      workspace.table_size = table_size;
    }
  }
  if (err != cudaSuccess) {
    for (size_t i = 0; i < n; ++i) {
      set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  // Per-proof inputs: 2 interleaved affine base points (b, A) and 6 GLV scalar
  // splits. These workspace-owned host buffers are pinned so transfers from
  // different shape streams can overlap without driver-side pageable staging.
  workspace.compressed_points.resize(2u * n);

  // Reuse validated B/A affines across repeated batches. Cold misses are
  // deduplicated and decoded as one parallel BLST batch.
  for (size_t i = 0; i < n; ++i) {
    workspace.compressed_points[2u * i] = items[i].compressed_b;
    workspace.compressed_points[2u * i + 1u] = items[i].signature_a;
  }
  const int32_t decode_status = load_or_decode_cached_g1_affines(
      workspace.compressed_points.data(), workspace.compressed_points.size(),
      workspace.host_bases);
  if (decode_status != CUBBS_BBS_GPU_SUCCESS) {
    for (size_t i = 0; i < n; ++i) {
      set_result(results, i, decode_status, 0);
    }
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const char *global_tables_config =
      std::getenv("CUBBS_BBS_GPU_PROOF_GEN_GLOBAL_TABLES");
  const bool global_tables = global_tables_config == nullptr ||
      env_flag_enabled("CUBBS_BBS_GPU_PROOF_GEN_GLOBAL_TABLES");
  workspace.next_point_key.clear();
  if (global_tables) {
    workspace.next_point_key.reserve(2u * n * 48u);
    for (const auto &point : workspace.compressed_points) {
      workspace.next_point_key.append(
          reinterpret_cast<const char *>(point.data), point.len);
    }
  }

  // Decode each big-endian scalar and perform the exact BLST GLV split on the
  // host. The kernel consumes only the first six multipliers; r3_tilde remains
  // part of Rust state assembly and does not need a device transfer.
  for (size_t i = 0; i < n; ++i) {
    // Convert the six device multipliers from Rust's big-endian encoding.
    for (size_t s = 0; s < 6u; ++s) {
      const uint8_t *src = items[i].scalars_be[s].data;
      uint8_t *split = &workspace.host_scalars[(i * 6u + s) * 32u];
      if (s == 2u &&
          std::memcmp(src, items[i].scalars_be[1].data, 32u) == 0) {
        std::memcpy(split, split - 32u, 32u);
        continue;
      }
      uint8_t scalar_le[32];
      for (size_t b = 0; b < 32u; ++b) {
        scalar_le[b] = src[31u - b];
      }
      cubbs_bbs_scalar_div_by_z2_host(scalar_le, split);
    }
  }

  auto *dev_d = workspace.outputs;
  auto *dev_abar = workspace.outputs + n;
  auto *dev_bbar = workspace.outputs + (2u * n);
  auto *dev_t1 = workspace.outputs + (3u * n);
  const bool upload_bases =
      !global_tables || workspace.point_key != workspace.next_point_key;
  if (err == cudaSuccess && upload_bases)
    err = cudaMemcpyAsync(workspace.bases, workspace.host_bases,
                          2u * n * sizeof(*workspace.bases),
                          cudaMemcpyHostToDevice, workspace.stream);
  if (err == cudaSuccess && global_tables && upload_bases) {
    constexpr uint32_t table_threads = 128u;
    const size_t point_count = 2u * n;
    const uint32_t table_blocks = static_cast<uint32_t>(
        (point_count + table_threads - 1u) / table_threads);
    if (use_radix256) {
      cubbs_bbs_proof_gen_radix256_tables_kernel
          <<<table_blocks, table_threads, 0u, workspace.stream>>>(
              workspace.bases, point_count, workspace.tables);
    } else {
      cubbs_bbs_proof_verify_radix16_tables_kernel
          <<<table_blocks, table_threads, 0u, workspace.stream>>>(
              workspace.bases, point_count, workspace.tables);
    }
    err = cudaGetLastError();
  }
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(workspace.scalars, workspace.host_scalars,
                          n * 6u * 32u, cudaMemcpyHostToDevice,
                          workspace.stream);

  if (err == cudaSuccess) {
    if (global_tables) {
      const uint32_t blocks = static_cast<uint32_t>((n + 4u) / 5u);
      if (use_radix256) {
        cubbs_bbs_proof_gen_fixed_muls_global_table_kernel
            <<<blocks, 32u, 0u, workspace.stream>>>(
                workspace.tables, workspace.scalars, dev_d, dev_abar,
                dev_bbar, dev_t1, n);
      } else {
        cubbs_bbs_proof_gen_fixed_muls_kernel<4u>
            <<<blocks, 32u, 0u, workspace.stream>>>(
                workspace.bases, workspace.scalars, dev_d, dev_abar,
                dev_bbar, dev_t1, n);
      }
    } else if (env_size_or_default(
                   "CUBBS_BBS_GPU_PROOF_GEN_WINDOW_BITS", 3u) <= 3u) {
      const uint32_t blocks = static_cast<uint32_t>((n + 4u) / 5u);
      cubbs_bbs_proof_gen_fixed_muls_kernel<3u>
          <<<blocks, 32u, 0u, workspace.stream>>>(
          workspace.bases, workspace.scalars, dev_d, dev_abar, dev_bbar,
          dev_t1, n);
    } else {
      const uint32_t blocks = static_cast<uint32_t>((n + 4u) / 5u);
      cubbs_bbs_proof_gen_fixed_muls_kernel<4u>
          <<<blocks, 32u, 0u, workspace.stream>>>(
          workspace.bases, workspace.scalars, dev_d, dev_abar, dev_bbar,
          dev_t1, n);
    }
    err = cudaGetLastError();
  }
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(workspace.host_outputs, workspace.outputs,
                          4u * n * sizeof(*workspace.host_outputs),
                          cudaMemcpyDeviceToHost, workspace.stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(workspace.stream);

  if (err != cudaSuccess) {
    for (size_t i = 0; i < n; ++i) {
      set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }
  if (global_tables && upload_bases) {
    workspace.point_key.swap(workspace.next_point_key);
  }

  // Normalize all 4*n projective outputs with one Montgomery batch inversion,
  // then perform only the cheap affine compression in parallel.
  workspace.projectives.resize(4u * n);
  workspace.projective_ptrs.resize(4u * n);
  workspace.compressed_outs.resize(4u * n);
  for (size_t i = 0; i < n; ++i) {
    const size_t offset = i * 4u;
    workspace.projectives[offset] =
        static_cast<cubbs_sppark_point_t>(workspace.host_outputs[i]);
    workspace.projectives[offset + 1u] =
        static_cast<cubbs_sppark_point_t>(workspace.host_outputs[n + i]);
    workspace.projectives[offset + 2u] =
        static_cast<cubbs_sppark_point_t>(workspace.host_outputs[2u * n + i]);
    workspace.projectives[offset + 3u] =
        static_cast<cubbs_sppark_point_t>(workspace.host_outputs[3u * n + i]);
    workspace.projective_ptrs[offset] = &workspace.projectives[offset];
    workspace.projective_ptrs[offset + 1u] = &workspace.projectives[offset + 1u];
    workspace.projective_ptrs[offset + 2u] = &workspace.projectives[offset + 2u];
    workspace.projective_ptrs[offset + 3u] = &workspace.projectives[offset + 3u];
    workspace.compressed_outs[offset] = items[i].d_out.data;
    workspace.compressed_outs[offset + 1u] = items[i].abar_out.data;
    workspace.compressed_outs[offset + 2u] = items[i].bbar_out.data;
    workspace.compressed_outs[offset + 3u] = items[i].t1_out.data;
  }
  const int32_t compression_status =
      cubbs_bbs_blst_compress_g1_projective_batch(
          workspace.projective_ptrs.data(), workspace.compressed_outs.data(),
          workspace.projective_ptrs.size());
  for (size_t i = 0; i < n; ++i) {
    if (compression_status != CUBBS_BBS_GPU_SUCCESS) {
      set_result(results, i, compression_status, 0);
      continue;
    }
    items[i].d_out.len = 48u;
    items[i].abar_out.len = 48u;
    items[i].bbar_out.len = 48u;
    items[i].t1_out.len = 48u;
    set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
                     CUBBS_BBS_MSM_ROUTE_FUSED_ITEM);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

#endif
