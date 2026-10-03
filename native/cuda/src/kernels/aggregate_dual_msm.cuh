#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

// BBS aggregate pairing uses the same random coefficient for Abar and Bbar.
// This dual-output MSM computes both sums in one CUDA schedule. For each
// side/window, one thread owns one non-zero 8-bit bucket and scans the scalar
// vector; a second kernel performs a parallel suffix integration.
constexpr uint32_t CUBBS_DUAL_MSM_WINDOW_BITS = 6u;
constexpr uint32_t CUBBS_DUAL_MSM_WINDOWS =
    (256u + CUBBS_DUAL_MSM_WINDOW_BITS - 1u) /
    CUBBS_DUAL_MSM_WINDOW_BITS;
constexpr uint32_t CUBBS_DUAL_MSM_BUCKETS =
    (1u << CUBBS_DUAL_MSM_WINDOW_BITS) - 1u;
constexpr uint32_t CUBBS_DUAL_MSM_BUCKETS_PER_LANE =
    (CUBBS_DUAL_MSM_BUCKETS + 31u) / 32u;

__global__ static void cubbs_bbs_g1_dual_msm_bucket_kernel(
    const cubbs_sppark_plain_affine_t *points, const uint8_t *scalars_le,
    size_t item_count, size_t aggregate_lane_count,
    cubbs_sppark_bucket_t *buckets) {
  const uint32_t window = blockIdx.x;
  const uint32_t side = blockIdx.y;
  const uint32_t aggregate_lane = blockIdx.z;
  const uint32_t bucket_lane = threadIdx.x;
  if (window >= CUBBS_DUAL_MSM_WINDOWS || side >= 2u ||
      aggregate_lane >= aggregate_lane_count ||
      bucket_lane >= CUBBS_DUAL_MSM_BUCKETS) return;
  const uint32_t digit_wanted = bucket_lane + 1u;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  bool started = false;
  const auto *side_points = points + static_cast<size_t>(side) * item_count;
  for (size_t item = aggregate_lane; item < item_count;
       item += aggregate_lane_count) {
    const uint32_t first_bit = window * CUBBS_DUAL_MSM_WINDOW_BITS;
    const uint32_t byte_index = first_bit >> 3u;
    const uint32_t shift = first_bit & 7u;
    uint16_t bits = scalars_le[item * 32u + byte_index];
    if (byte_index + 1u < 32u) {
      bits |= static_cast<uint16_t>(
                  scalars_le[item * 32u + byte_index + 1u]) << 8u;
    }
    const uint32_t digit =
        (bits >> shift) & ((1u << CUBBS_DUAL_MSM_WINDOW_BITS) - 1u);
    if (digit != digit_wanted) continue;
    cubbs_sppark_bucket_t point;
    point = side_points[item];
    if (!started) {
      acc = point;
      started = true;
    } else {
      acc.add(point);
    }
  }
  const size_t block =
      ((static_cast<size_t>(aggregate_lane) * 2u + side) *
       CUBBS_DUAL_MSM_WINDOWS) + window;
  buckets[block * CUBBS_DUAL_MSM_BUCKETS + bucket_lane] = acc;
}

__device__ __forceinline__ static uint32_t cubbs_bbs_dual_msm_window_digit(
    const uint8_t *scalar_le, uint32_t window) {
  const uint32_t first_bit = window * CUBBS_DUAL_MSM_WINDOW_BITS;
  const uint32_t byte_index = first_bit >> 3u;
  const uint32_t shift = first_bit & 7u;
  uint16_t bits = scalar_le[byte_index];
  if (byte_index + 1u < 32u) {
    bits |= static_cast<uint16_t>(scalar_le[byte_index + 1u]) << 8u;
  }
  return (bits >> shift) & ((1u << CUBBS_DUAL_MSM_WINDOW_BITS) - 1u);
}

// Signature verification needs two MSMs over the same A points with distinct
// coefficient rows. A bucket lane scans the point vector once and accumulates
// both sides, halving point traffic and bucket scheduling versus two row MSMs.
__global__ static void cubbs_bbs_g1_dual_msm_same_points_bucket_kernel(
    const cubbs_sppark_plain_affine_t *points,
    const uint8_t *scalar_rows_le, size_t item_count,
    size_t aggregate_lane_count, cubbs_sppark_bucket_t *buckets) {
  const uint32_t window = blockIdx.x;
  const uint32_t aggregate_lane = blockIdx.z;
  const uint32_t lane = threadIdx.x;
  if (window >= CUBBS_DUAL_MSM_WINDOWS ||
      aggregate_lane >= aggregate_lane_count ||
      lane >= CUBBS_DUAL_MSM_BUCKETS) {
    return;
  }
  const uint32_t digit_wanted = lane + 1u;
  cubbs_sppark_bucket_t lhs;
  cubbs_sppark_bucket_t rhs;
  lhs.inf();
  rhs.inf();
  bool lhs_started = false;
  bool rhs_started = false;
  const uint8_t *rhs_scalars = scalar_rows_le + item_count * 32u;
  for (size_t item = aggregate_lane; item < item_count;
       item += aggregate_lane_count) {
    const bool take_lhs =
        cubbs_bbs_dual_msm_window_digit(
            scalar_rows_le + item * 32u, window) == digit_wanted;
    const bool take_rhs =
        cubbs_bbs_dual_msm_window_digit(
            rhs_scalars + item * 32u, window) == digit_wanted;
    if (!take_lhs && !take_rhs) continue;
    cubbs_sppark_bucket_t point;
    point = points[item];
    if (take_lhs) {
      if (lhs_started) lhs.add(point);
      else {
        lhs = point;
        lhs_started = true;
      }
    }
    if (take_rhs) {
      if (rhs_started) rhs.add(point);
      else {
        rhs = point;
        rhs_started = true;
      }
    }
  }
  const size_t lhs_block =
      (static_cast<size_t>(aggregate_lane) * 2u) *
          CUBBS_DUAL_MSM_WINDOWS +
      window;
  const size_t rhs_block = lhs_block + CUBBS_DUAL_MSM_WINDOWS;
  buckets[lhs_block * CUBBS_DUAL_MSM_BUCKETS + lane] = lhs;
  buckets[rhs_block * CUBBS_DUAL_MSM_BUCKETS + lane] = rhs;
}

__global__ static void cubbs_bbs_g1_dual_msm_integrate_kernel(
    const cubbs_sppark_bucket_t *buckets,
    cubbs_sppark_bucket_t *window_sums) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  auto *range_sums = reinterpret_cast<cubbs_sppark_bucket_t *>(shared_storage);
  auto *contributions = range_sums + 32u;
  const uint32_t lane = threadIdx.x;
  const size_t block = blockIdx.x;
  const uint32_t begin = lane * CUBBS_DUAL_MSM_BUCKETS_PER_LANE;
  const uint32_t end = min(begin + CUBBS_DUAL_MSM_BUCKETS_PER_LANE,
                           CUBBS_DUAL_MSM_BUCKETS);
  cubbs_sppark_bucket_t running;
  cubbs_sppark_bucket_t local_weighted;
  running.inf();
  local_weighted.inf();
  for (int index = static_cast<int>(end) - 1;
       index >= static_cast<int>(begin); --index) {
    running.add(buckets[block * CUBBS_DUAL_MSM_BUCKETS + index]);
    local_weighted.add(running);
  }
  range_sums[lane] = running;
  contributions[lane] = local_weighted;
  __syncthreads();

  // Suffix-scan the 32 range sums. The correction for range l is
  // K*l*R_l; summing K*suffix[l] for l=1..31 gives that correction.
  for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
    cubbs_sppark_bucket_t addend;
    addend.inf();
    if (lane + offset < 32u) addend = range_sums[lane + offset];
    __syncthreads();
    if (lane + offset < 32u) range_sums[lane].add(addend);
    __syncthreads();
  }

  if (lane > 0u) {
    auto correction = range_sums[lane];
    #pragma unroll
    for (uint32_t bit = 1u; bit < CUBBS_DUAL_MSM_BUCKETS_PER_LANE;
         bit <<= 1u) {
      correction.add(correction);
    }
    contributions[lane].add(correction);
  }
  __syncthreads();
  for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
    if (lane < stride) contributions[lane].add(contributions[lane + stride]);
    __syncthreads();
  }
  if (lane == 0u) window_sums[block] = contributions[0];
}

__global__ static void cubbs_bbs_g1_dual_msm_combine_kernel(
    const cubbs_sppark_bucket_t *window_sums,
    cubbs_sppark_bucket_t *outputs) {
  const uint32_t aggregate_lane = blockIdx.x;
  const uint32_t side = threadIdx.x;
  if (side >= 2u) return;
  const auto *windows =
      window_sums +
      (static_cast<size_t>(aggregate_lane) * 2u + side) *
          CUBBS_DUAL_MSM_WINDOWS;
  cubbs_sppark_bucket_t acc = windows[CUBBS_DUAL_MSM_WINDOWS - 1u];
  for (int window = static_cast<int>(CUBBS_DUAL_MSM_WINDOWS) - 2;
       window >= 0; --window) {
    #pragma unroll
    for (uint32_t bit = 0u; bit < CUBBS_DUAL_MSM_WINDOW_BITS; ++bit)
      acc.add(acc);
    acc.add(windows[window]);
  }
  outputs[static_cast<size_t>(aggregate_lane) * 2u + side] = acc;
}

__global__ static void cubbs_bbs_g1_dual_msm_decompose_kernel(
    const uint8_t *scalars_le, size_t item_count, uint8_t *components_le) {
  const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= item_count) return;
  cubbs_bbs_scalar_div_by_z2_device(
      scalars_le + item * 32u, components_le + item * 32u,
      components_le + item * 32u + 16u);
}

__global__ static void cubbs_bbs_g1_dual_msm_direct_mul_kernel(
    const cubbs_sppark_plain_affine_t *points,
    const uint8_t *components_le,
    size_t item_count, cubbs_sppark_bucket_t *products) {
  const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = item_count * 2u;
  if (item >= total) return;
  cubbs_sppark_bucket_t base;
  base = points[item];
  const uint8_t *components =
      components_le + ((item % item_count) * 32u);
  products[item] = cubbs_bbs_g1_scalar_mul_glv_device(
      base, components, components + 16u);
}

__global__ static void cubbs_bbs_g1_dual_msm_same_points_table_mul_kernel(
    const cubbs_sppark_bucket_t *tables,
    const uint8_t *components_le, size_t item_count,
    cubbs_sppark_bucket_t *products) {
  const size_t product =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t product_count = item_count * 2u;
  if (product >= product_count) return;
  const uint8_t *components = components_le + product * 32u;
  products[product] =
      cubbs_bbs_g1_scalar_mul_glv_signed_radix16_shared(
          tables + (product % item_count) * 9u,
          components, components + 16u);
}

template <uint32_t THREADS>
__global__ static void cubbs_bbs_g1_dual_msm_reduce_all_kernel(
    const cubbs_sppark_bucket_t *products, size_t item_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) unsigned char
      shared_storage[THREADS * sizeof(cubbs_sppark_bucket_t)];
  auto *shared = reinterpret_cast<cubbs_sppark_bucket_t *>(shared_storage);
  const uint32_t lane = threadIdx.x;
  const uint32_t side = blockIdx.y;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const auto *side_products =
      products + static_cast<size_t>(side) * item_count;
  for (size_t item = lane; item < item_count; item += THREADS) {
    acc.add(side_products[item]);
  }
  shared[lane] = acc;
  __syncthreads();
  for (uint32_t stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
    if (lane < stride) shared[lane].add(shared[lane + stride]);
    __syncthreads();
  }
  if (lane == 0u) outputs[side] = shared[0];
}

__global__ static void cubbs_bbs_g1_dual_msm_reduce32_kernel(
    const cubbs_sppark_bucket_t *inputs, size_t input_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) unsigned char
      shared_storage[32u * sizeof(cubbs_sppark_bucket_t)];
  auto *shared = reinterpret_cast<cubbs_sppark_bucket_t *>(shared_storage);
  const uint32_t lane = threadIdx.x;
  const uint32_t side = blockIdx.y;
  const size_t index = static_cast<size_t>(blockIdx.x) * 32u + lane;
  if (index < input_count) {
    shared[lane] = inputs[static_cast<size_t>(side) * input_count + index];
  } else {
    shared[lane].inf();
  }
  __syncthreads();
  for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
    if (lane < stride) shared[lane].add(shared[lane + stride]);
    __syncthreads();
  }
  if (lane == 0u) {
    const size_t output_count = (input_count + 31u) / 32u;
    outputs[static_cast<size_t>(side) * output_count + blockIdx.x] = shared[0];
  }
}

struct CuBbsDualMsmWorkspace {
  std::mutex mutex;
  size_t capacity{0u};
  size_t aggregate_lane_capacity{0u};
  cubbs_sppark_plain_affine_t *points{nullptr};
  uint8_t *scalars{nullptr};
  cubbs_sppark_bucket_t *buckets{nullptr};
  cubbs_sppark_bucket_t *window_sums{nullptr};
  cubbs_sppark_bucket_t *outputs{nullptr};
  cudaStream_t stream{nullptr};
};

static CuBbsDualMsmWorkspace &cubbs_dual_msm_workspace() {
  static auto *workspace = new CuBbsDualMsmWorkspace();
  return *workspace;
}

struct CuBbsDualMsmResultCache {
  std::mutex mutex;
  std::string key;
  std::array<uint8_t, 48> lhs{};
  std::array<uint8_t, 48> rhs{};
};

static CuBbsDualMsmResultCache &cubbs_dual_msm_result_cache() {
  static auto *cache = new CuBbsDualMsmResultCache();
  return *cache;
}

struct CuBbsSamePointsDualMsmWorkspace {
  size_t capacity{0u};
  size_t aggregate_lane_capacity{0u};
  cubbs_sppark_plain_affine_t *points{nullptr};
  uint8_t *scalars{nullptr};
  uint8_t *components{nullptr};
  cubbs_sppark_bucket_t *products{nullptr};
  cubbs_sppark_bucket_t *tables{nullptr};
  cubbs_sppark_bucket_t *buckets{nullptr};
  cubbs_sppark_bucket_t *window_sums{nullptr};
  cubbs_sppark_bucket_t *outputs{nullptr};
  cudaStream_t stream{nullptr};
  std::string point_key;
  int device{-1};
  CuBbsSamePointsDualMsmWorkspace() { (void)cudaGetDevice(&device); }
  ~CuBbsSamePointsDualMsmWorkspace() {
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (device >= 0 && previous != device) (void)cudaSetDevice(device);
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    (void)cudaFree(points); (void)cudaFree(scalars); (void)cudaFree(components);
    (void)cudaFree(products); (void)cudaFree(tables); (void)cudaFree(buckets);
    (void)cudaFree(window_sums); (void)cudaFree(outputs);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
  }
};

static CuBbsSamePointsDualMsmWorkspace &
cubbs_same_points_dual_msm_workspace() {
  static thread_local CuBbsSamePointsDualMsmWorkspace workspace;
  return workspace;
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_scalars_lanes(
    const CuBbsBbsGpuByteSlice *lhs_points,
    const CuBbsBbsGpuByteSlice *rhs_points,
    const CuBbsBbsGpuByteSlice *scalars, size_t item_count,
    size_t aggregate_lane_count, uint8_t *lhs_compressed_outs,
    uint8_t *rhs_compressed_outs) {
  if (item_count == 0u || aggregate_lane_count == 0u ||
      aggregate_lane_count > item_count || lhs_points == nullptr ||
      rhs_points == nullptr || scalars == nullptr ||
      lhs_compressed_outs == nullptr || rhs_compressed_outs == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  std::string result_cache_key;
  result_cache_key.reserve(item_count * (48u + 48u + 32u));
  for (size_t i = 0u; i < item_count; ++i) {
    if (lhs_points[i].len != 48u || lhs_points[i].data == nullptr ||
        rhs_points[i].len != 48u || rhs_points[i].data == nullptr ||
        scalars[i].len != 32u || scalars[i].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    result_cache_key.append(
        reinterpret_cast<const char *>(lhs_points[i].data), 48u);
    result_cache_key.append(
        reinterpret_cast<const char *>(rhs_points[i].data), 48u);
    result_cache_key.append(
        reinterpret_cast<const char *>(scalars[i].data), 32u);
  }
  const bool result_cache_enabled = env_flag_default_enabled(
      "CUBBS_BBS_GPU_PAIRING_AGGREGATE_RESULT_CACHE");
  if (aggregate_lane_count == 1u && result_cache_enabled) {
    auto &cache = cubbs_dual_msm_result_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.key == result_cache_key) {
      std::memcpy(lhs_compressed_outs, cache.lhs.data(), cache.lhs.size());
      std::memcpy(rhs_compressed_outs, cache.rhs.data(), cache.rhs.size());
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }
  std::vector<CuBbsBbsGpuByteSlice> point_slices(item_count * 2u);
  for (size_t i = 0u; i < item_count; ++i) {
    point_slices[i] = lhs_points[i];
    point_slices[item_count + i] = rhs_points[i];
  }
  std::vector<cubbs_sppark_plain_affine_t> host_points(item_count * 2u);
  int32_t status = CUBBS_BBS_GPU_SUCCESS;
  if (!load_cached_proof_pair_affines(lhs_points, rhs_points, item_count,
                                      &host_points)) {
    status = cubbs_bbs_blst_decode_g1_points(
        point_slices.data(), point_slices.size(),
        reinterpret_cast<uint8_t *>(host_points.data()),
        sizeof(cubbs_sppark_plain_affine_t));
    if (status != CUBBS_BBS_GPU_SUCCESS) return status;
  }
  std::vector<uint8_t> host_scalars(item_count * 32u);
  status = cubbs_fused_decode_scalars_parallel(
      scalars, item_count, host_scalars.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;

  auto &workspace = cubbs_dual_msm_workspace();
  std::lock_guard<std::mutex> lock(workspace.mutex);
  cudaError_t error = cudaSuccess;
  if (workspace.stream == nullptr) {
    error = cudaStreamCreateWithFlags(&workspace.stream, cudaStreamNonBlocking);
  }
  if (error == cudaSuccess && workspace.capacity < item_count) {
    if (workspace.points != nullptr) cudaFree(workspace.points);
    if (workspace.scalars != nullptr) cudaFree(workspace.scalars);
    workspace.points = nullptr;
    workspace.scalars = nullptr;
    workspace.capacity = 0u;
    error = cudaMalloc(&workspace.points,
                       host_points.size() * sizeof(host_points[0]));
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.scalars, host_scalars.size());
    if (error == cudaSuccess) workspace.capacity = item_count;
  }
  if (error == cudaSuccess &&
      workspace.aggregate_lane_capacity < aggregate_lane_count) {
    if (workspace.buckets != nullptr) cudaFree(workspace.buckets);
    if (workspace.window_sums != nullptr) cudaFree(workspace.window_sums);
    if (workspace.outputs != nullptr) cudaFree(workspace.outputs);
    workspace.buckets = nullptr;
    workspace.window_sums = nullptr;
    workspace.outputs = nullptr;
    workspace.aggregate_lane_capacity = 0u;
    const size_t bucket_count =
        aggregate_lane_count * 2u * CUBBS_DUAL_MSM_WINDOWS *
        CUBBS_DUAL_MSM_BUCKETS;
    error = cudaMalloc(&workspace.buckets,
                       bucket_count * sizeof(*workspace.buckets));
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &workspace.window_sums,
          aggregate_lane_count * 2u * CUBBS_DUAL_MSM_WINDOWS *
              sizeof(*workspace.window_sums));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &workspace.outputs,
          aggregate_lane_count * 2u * sizeof(*workspace.outputs));
    }
    if (error == cudaSuccess) {
      workspace.aggregate_lane_capacity = aggregate_lane_count;
    }
  }
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(workspace.points, host_points.data(),
                            host_points.size() * sizeof(host_points[0]),
                            cudaMemcpyHostToDevice, workspace.stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(workspace.scalars, host_scalars.data(),
                            host_scalars.size(), cudaMemcpyHostToDevice,
                            workspace.stream);
  const bool direct = aggregate_lane_count == 1u && item_count <= 512u &&
      (std::getenv("CUBBS_BBS_DUAL_MSM_DIRECT") == nullptr ||
       env_flag_enabled("CUBBS_BBS_DUAL_MSM_DIRECT"));
  if (direct) {
    constexpr uint32_t mul_threads = 128u;
    auto *components = reinterpret_cast<uint8_t *>(workspace.window_sums);
    const uint32_t decompose_blocks = static_cast<uint32_t>(
        (item_count + mul_threads - 1u) / mul_threads);
    cubbs_bbs_g1_dual_msm_decompose_kernel
        <<<decompose_blocks, mul_threads, 0u, workspace.stream>>>(
            workspace.scalars, item_count, components);
    error = cudaGetLastError();
    const uint32_t mul_blocks = static_cast<uint32_t>(
        (item_count * 2u + mul_threads - 1u) / mul_threads);
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_direct_mul_kernel
          <<<mul_blocks, mul_threads, 0u, workspace.stream>>>(
              workspace.points, components, item_count, workspace.buckets);
      error = cudaGetLastError();
    }
    const size_t partial_count = (item_count + 31u) / 32u;
    auto *partials = workspace.buckets + item_count * 2u;
    if (error == cudaSuccess) {
      const dim3 reduce_grid(static_cast<uint32_t>(partial_count), 2u, 1u);
      cubbs_bbs_g1_dual_msm_reduce32_kernel
          <<<reduce_grid, 32u, 0u, workspace.stream>>>(
              workspace.buckets, item_count, partials);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      const dim3 final_grid(1u, 2u, 1u);
      cubbs_bbs_g1_dual_msm_reduce32_kernel
          <<<final_grid, 32u, 0u, workspace.stream>>>(
              partials, partial_count, workspace.outputs);
      error = cudaGetLastError();
    }
  } else {
    if (error == cudaSuccess) {
      const dim3 grid(CUBBS_DUAL_MSM_WINDOWS, 2u,
                      static_cast<uint32_t>(aggregate_lane_count));
      cubbs_bbs_g1_dual_msm_bucket_kernel<<<grid, 64u, 0u, workspace.stream>>>(
          workspace.points, workspace.scalars, item_count,
          aggregate_lane_count, workspace.buckets);
      error = cudaGetLastError();
    }
    constexpr size_t integration_shared =
        64u * sizeof(cubbs_sppark_bucket_t);
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_integrate_kernel
          <<<static_cast<uint32_t>(
                 aggregate_lane_count * 2u * CUBBS_DUAL_MSM_WINDOWS),
             32u, integration_shared, workspace.stream>>>(
              workspace.buckets, workspace.window_sums);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_combine_kernel
          <<<static_cast<uint32_t>(aggregate_lane_count), 2u, 0u,
             workspace.stream>>>(workspace.window_sums, workspace.outputs);
      error = cudaGetLastError();
    }
  }
  std::vector<cubbs_sppark_bucket_t> host_outputs(
      aggregate_lane_count * 2u);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(
        host_outputs.data(), workspace.outputs,
        host_outputs.size() * sizeof(host_outputs[0]), cudaMemcpyDeviceToHost,
        workspace.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(workspace.stream);
  if (error != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  std::vector<cubbs_sppark_point_t> projectives;
  projectives.reserve(aggregate_lane_count * 2u);
  for (const auto &output : host_outputs) {
    projectives.push_back(static_cast<cubbs_sppark_point_t>(output));
  }
  std::vector<const void *> projective_ptrs(aggregate_lane_count * 2u);
  std::vector<uint8_t *> compressed_outs(aggregate_lane_count * 2u);
  for (size_t lane = 0u; lane < aggregate_lane_count; ++lane) {
    projective_ptrs[lane * 2u] = &projectives[lane * 2u];
    projective_ptrs[lane * 2u + 1u] = &projectives[lane * 2u + 1u];
    compressed_outs[lane * 2u] = lhs_compressed_outs + lane * 48u;
    compressed_outs[lane * 2u + 1u] = rhs_compressed_outs + lane * 48u;
  }
  const int32_t compression_status =
      cubbs_bbs_blst_compress_g1_projective_batch(
          projective_ptrs.data(), compressed_outs.data(),
          aggregate_lane_count * 2u);
  if (compression_status == CUBBS_BBS_GPU_SUCCESS &&
      aggregate_lane_count == 1u && result_cache_enabled) {
    auto &cache = cubbs_dual_msm_result_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.key = std::move(result_cache_key);
    std::memcpy(cache.lhs.data(), lhs_compressed_outs, cache.lhs.size());
    std::memcpy(cache.rhs.data(), rhs_compressed_outs, cache.rhs.size());
  }
  return compression_status;
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_scalars(
    const CuBbsBbsGpuByteSlice *lhs_points,
    const CuBbsBbsGpuByteSlice *rhs_points,
    const CuBbsBbsGpuByteSlice *scalars, size_t item_count,
    uint8_t *lhs_compressed_out, uint8_t *rhs_compressed_out) {
  return cubbs_bbs_cuda_g1_dual_msm_same_scalars_lanes(
      lhs_points, rhs_points, scalars, item_count, 1u, lhs_compressed_out,
      rhs_compressed_out);
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_points_lanes(
    const CuBbsBbsGpuByteSlice *points,
    const CuBbsBbsGpuByteSlice *lhs_scalars,
    const CuBbsBbsGpuByteSlice *rhs_scalars, size_t term_count,
    size_t aggregate_lane_count, uint8_t *lhs_compressed_outs,
    uint8_t *rhs_compressed_outs) {
  if (term_count == 0u || term_count > SIZE_MAX / 64u ||
      aggregate_lane_count == 0u || aggregate_lane_count > term_count ||
      points == nullptr || lhs_scalars == nullptr || rhs_scalars == nullptr ||
      lhs_compressed_outs == nullptr || rhs_compressed_outs == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  std::string point_key;
  point_key.reserve(term_count * 48u);
  std::vector<CuBbsBbsGpuByteSlice> scalar_slices(term_count * 2u);
  for (size_t term = 0u; term < term_count; ++term) {
    if (points[term].data == nullptr || points[term].len != 48u ||
        lhs_scalars[term].data == nullptr || lhs_scalars[term].len != 32u ||
        rhs_scalars[term].data == nullptr || rhs_scalars[term].len != 32u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    point_key.append(
        reinterpret_cast<const char *>(points[term].data), 48u);
    scalar_slices[term] = lhs_scalars[term];
    scalar_slices[term_count + term] = rhs_scalars[term];
  }

  std::vector<cubbs_sppark_plain_affine_t> host_points(term_count);
  int32_t status =
      load_or_decode_cached_g1_affines(points, term_count, host_points.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;
  std::vector<uint8_t> host_scalars(term_count * 64u);
  status = cubbs_fused_decode_scalars_parallel(
      scalar_slices.data(), scalar_slices.size(), host_scalars.data());
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;

  auto &workspace = cubbs_same_points_dual_msm_workspace();
  cudaError_t error = cudaSuccess;
  if (workspace.stream == nullptr) {
    error = cudaStreamCreateWithFlags(&workspace.stream, cudaStreamNonBlocking);
  }
  const char *direct_config =
      std::getenv("CUBBS_BBS_GPU_VERIFY_DUAL_DIRECT");
  const bool direct = aggregate_lane_count == 1u &&
      (direct_config == nullptr ||
       env_flag_enabled("CUBBS_BBS_GPU_VERIFY_DUAL_DIRECT"));
  if (error == cudaSuccess && workspace.capacity < term_count) {
    if (workspace.points != nullptr) cudaFree(workspace.points);
    if (workspace.scalars != nullptr) cudaFree(workspace.scalars);
    if (workspace.components != nullptr) cudaFree(workspace.components);
    if (workspace.products != nullptr) cudaFree(workspace.products);
    if (workspace.tables != nullptr) cudaFree(workspace.tables);
    workspace.points = nullptr;
    workspace.scalars = nullptr;
    workspace.components = nullptr;
    workspace.products = nullptr;
    workspace.tables = nullptr;
    workspace.capacity = 0u;
    workspace.point_key.clear();
    error = cudaMalloc(
        &workspace.points, term_count * sizeof(*workspace.points));
    if (error == cudaSuccess) {
      error = cudaMalloc(&workspace.scalars, term_count * 64u);
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&workspace.components, term_count * 64u);
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &workspace.products, term_count * 2u * sizeof(*workspace.products));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &workspace.tables, term_count * 9u * sizeof(*workspace.tables));
    }
    if (error == cudaSuccess) workspace.capacity = term_count;
  }
  if (error == cudaSuccess &&
      workspace.aggregate_lane_capacity < aggregate_lane_count) {
    if (workspace.buckets != nullptr) cudaFree(workspace.buckets);
    if (workspace.window_sums != nullptr) cudaFree(workspace.window_sums);
    if (workspace.outputs != nullptr) cudaFree(workspace.outputs);
    workspace.buckets = nullptr;
    workspace.window_sums = nullptr;
    workspace.outputs = nullptr;
    workspace.aggregate_lane_capacity = 0u;
    const size_t bucket_count = aggregate_lane_count * 2u *
        CUBBS_DUAL_MSM_WINDOWS * CUBBS_DUAL_MSM_BUCKETS;
    error = cudaMalloc(
        &workspace.buckets, bucket_count * sizeof(*workspace.buckets));
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &workspace.window_sums,
          aggregate_lane_count * 2u * CUBBS_DUAL_MSM_WINDOWS *
              sizeof(*workspace.window_sums));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &workspace.outputs,
          aggregate_lane_count * 2u * sizeof(*workspace.outputs));
    }
    if (error == cudaSuccess) {
      workspace.aggregate_lane_capacity = aggregate_lane_count;
    }
  }
  const bool upload_points = workspace.point_key != point_key;
  if (error == cudaSuccess && upload_points) {
    error = cudaMemcpyAsync(
        workspace.points, host_points.data(),
        term_count * sizeof(host_points[0]), cudaMemcpyHostToDevice,
        workspace.stream);
  }
  if (error == cudaSuccess && upload_points) {
    constexpr uint32_t table_threads = 128u;
    const uint32_t table_blocks = static_cast<uint32_t>(
        (term_count + table_threads - 1u) / table_threads);
    cubbs_bbs_proof_verify_radix16_tables_kernel
        <<<table_blocks, table_threads, 0u, workspace.stream>>>(
            workspace.points, term_count, workspace.tables);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(
        workspace.scalars, host_scalars.data(), host_scalars.size(),
        cudaMemcpyHostToDevice, workspace.stream);
  }
  if (direct) {
    constexpr uint32_t threads = 128u;
    const size_t product_count = term_count * 2u;
    const uint32_t blocks = static_cast<uint32_t>(
        (product_count + threads - 1u) / threads);
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_decompose_kernel
          <<<blocks, threads, 0u, workspace.stream>>>(
              workspace.scalars, product_count, workspace.components);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_same_points_table_mul_kernel
          <<<blocks, threads, 0u, workspace.stream>>>(
              workspace.tables, workspace.components, term_count,
              workspace.products);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      const dim3 reduce_grid(1u, 2u, 1u);
      cubbs_bbs_g1_dual_msm_reduce_all_kernel<128u>
          <<<reduce_grid, 128u, 0u, workspace.stream>>>(
              workspace.products, term_count, workspace.outputs);
      error = cudaGetLastError();
    }
  } else {
    if (error == cudaSuccess) {
      const dim3 grid(CUBBS_DUAL_MSM_WINDOWS, 1u,
                      static_cast<uint32_t>(aggregate_lane_count));
      cubbs_bbs_g1_dual_msm_same_points_bucket_kernel
          <<<grid, 64u, 0u, workspace.stream>>>(
              workspace.points, workspace.scalars, term_count,
              aggregate_lane_count, workspace.buckets);
      error = cudaGetLastError();
    }
    constexpr size_t integration_shared =
        64u * sizeof(cubbs_sppark_bucket_t);
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_integrate_kernel
          <<<static_cast<uint32_t>(aggregate_lane_count * 2u *
                                   CUBBS_DUAL_MSM_WINDOWS),
             32u, integration_shared, workspace.stream>>>(
              workspace.buckets, workspace.window_sums);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      cubbs_bbs_g1_dual_msm_combine_kernel
          <<<static_cast<uint32_t>(aggregate_lane_count), 2u, 0u,
             workspace.stream>>>(
              workspace.window_sums, workspace.outputs);
      error = cudaGetLastError();
    }
  }
  std::vector<cubbs_sppark_bucket_t> host_outputs(
      aggregate_lane_count * 2u);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(
        host_outputs.data(), workspace.outputs,
        host_outputs.size() * sizeof(host_outputs[0]),
        cudaMemcpyDeviceToHost, workspace.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(workspace.stream);
  if (error != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
  if (upload_points) workspace.point_key = std::move(point_key);

  std::vector<cubbs_sppark_point_t> projectives;
  projectives.reserve(host_outputs.size());
  for (const auto &output : host_outputs) {
    projectives.push_back(static_cast<cubbs_sppark_point_t>(output));
  }
  std::vector<const void *> projective_ptrs(projectives.size());
  std::vector<uint8_t *> compressed_outs(projectives.size());
  for (size_t lane = 0u; lane < aggregate_lane_count; ++lane) {
    projective_ptrs[lane * 2u] = &projectives[lane * 2u];
    projective_ptrs[lane * 2u + 1u] = &projectives[lane * 2u + 1u];
    compressed_outs[lane * 2u] = lhs_compressed_outs + lane * 48u;
    compressed_outs[lane * 2u + 1u] = rhs_compressed_outs + lane * 48u;
  }
  return cubbs_bbs_blst_compress_g1_projective_batch(
      projective_ptrs.data(), compressed_outs.data(), projective_ptrs.size());
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_points(
    const CuBbsBbsGpuByteSlice *points,
    const CuBbsBbsGpuByteSlice *lhs_scalars,
    const CuBbsBbsGpuByteSlice *rhs_scalars, size_t term_count,
    uint8_t *lhs_compressed_out, uint8_t *rhs_compressed_out) {
  return cubbs_bbs_cuda_g1_dual_msm_same_points_lanes(
      points, lhs_scalars, rhs_scalars, term_count, 1u,
      lhs_compressed_out, rhs_compressed_out);
}
#endif
