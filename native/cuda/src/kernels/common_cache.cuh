#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_COMMON_CACHE_CUH
#define CUBBS_BBS_COMMON_CACHE_CUH

// ============================================================================
// Common-Points Generator Cache (GLV Lim-Lee for 1..2100, 10-bit Pippenger >2100)
// ============================================================================

struct CuBbsFusedCommonPointsCache {
  cubbs_sppark_plain_affine_t *device_points{nullptr};
  cubbs_sppark_plain_affine_t *device_limlee_affines{nullptr};
  size_t *device_limlee_offsets{nullptr};
  cubbs_sppark_bucket_t *device_limlee_c_total{nullptr};
  uint32_t limlee_num_groups{0u};
  uint32_t limlee_group_size{0u};
  uint32_t limlee_v_sub{0u};
  size_t limlee_total_entries{0u};
  bool use_pippenger_large{false};
  cudaStream_t stream{nullptr};
  size_t term_count{0u};
  int device{0};

  CuBbsFusedCommonPointsCache() { (void)cudaGetDevice(&device); }

  size_t table_bytes() const {
    const size_t base_pts = use_pippenger_large ? (2u * term_count) : term_count;
    const size_t base_bytes =
        (device_points != nullptr ? base_pts * sizeof(cubbs_sppark_plain_affine_t) : 0u);
    return base_bytes + limlee_total_entries * sizeof(cubbs_sppark_plain_affine_t);
  }

  ~CuBbsFusedCommonPointsCache() {
    int previous = device;
    (void)cudaGetDevice(&previous);
    if (previous != device) (void)cudaSetDevice(device);
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    if (device_points != nullptr) (void)cudaFree(device_points);
    if (device_limlee_affines != nullptr) (void)cudaFree(device_limlee_affines);
    if (device_limlee_offsets != nullptr) (void)cudaFree(device_limlee_offsets);
    if (device_limlee_c_total != nullptr) (void)cudaFree(device_limlee_c_total);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }
};

static std::mutex &fused_common_points_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::unordered_map<std::string, std::shared_ptr<CuBbsFusedCommonPointsCache>> &
fused_common_points_cache() {
  static auto *cache = new std::unordered_map<
      std::string, std::shared_ptr<CuBbsFusedCommonPointsCache>>();
  return *cache;
}

static std::mutex &fused_common_points_build_mutex(const std::string &key) {
  static std::mutex map_mutex;
  static auto *build_mutexes =
      new std::unordered_map<std::string, std::unique_ptr<std::mutex>>();
  std::lock_guard<std::mutex> lock(map_mutex);
  auto &entry = (*build_mutexes)[key];
  if (entry == nullptr) entry = std::make_unique<std::mutex>();
  return *entry;
}

static size_t cubbs_gpu_table_cache_budget_bytes() {
  size_t default_budget = 2048ull * 1024ull * 1024ull;
  size_t free_vram = 0u, total_vram = 0u;
  if (cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess &&
      total_vram >= 12ull * 1024ull * 1024ull * 1024ull) {
    default_budget = 3072ull * 1024ull * 1024ull;
  }
  return env_size_or_default(
      "CUBBS_BBS_GPU_TABLE_CACHE_BUDGET_BYTES", default_budget);
}

static void retain_fused_common_points_cache(
    const std::string &key,
    const std::shared_ptr<CuBbsFusedCommonPointsCache> &value) {
  const size_t budget = cubbs_gpu_table_cache_budget_bytes();
  const size_t incoming = value->table_bytes();
  if (incoming > budget) return;
  std::lock_guard<std::mutex> lock(fused_common_points_cache_mutex());
  auto &cache = fused_common_points_cache();
  if (cache.find(key) != cache.end()) return;
  size_t retained = 0u;
  for (const auto &entry : cache) retained += entry.second->table_bytes();
  for (auto it = cache.begin(); retained > budget - incoming && it != cache.end();) {
    if (it->second.use_count() == 1) {
      retained -= it->second->table_bytes();
      it = cache.erase(it);
    } else {
      ++it;
    }
  }
  if (retained <= budget - incoming) cache.emplace(key, value);
}

static std::string fused_common_points_cache_key(
    const CuBbsBbsGpuByteSlice *points, size_t term_count) {
  std::string key;
  int device = 0;
  (void)cudaGetDevice(&device);
  key.reserve(sizeof(device) + sizeof(term_count) + (term_count * 48u));
  key.append(reinterpret_cast<const char *>(&device), sizeof(device));
  key.append(reinterpret_cast<const char *>(&term_count), sizeof(term_count));
  for (size_t i = 0u; i < term_count; ++i) {
    key.append(reinterpret_cast<const char *>(points[i].data), points[i].len);
  }
  return key;
}

static cudaError_t build_limlee_table_for_cache(
    CuBbsFusedCommonPointsCache *cached, size_t term_count) {
  uint32_t group_size = (term_count <= 150u) ? 17u : 18u;
  if (const char *genv = std::getenv("CUBBS_BBS_LIMLEE_GROUP_SIZE")) {
    const unsigned long parsed = std::strtoul(genv, nullptr, 10);
    if (parsed >= 1ul && parsed <= CUBBS_LIMLEE_MAX_GROUP_SIZE) {
      group_size = static_cast<uint32_t>(parsed);
    }
  }
  if (group_size > term_count) {
    group_size = static_cast<uint32_t>(term_count);
  }
  uint32_t v_sub =
      (term_count <= 320u) ? 4u : ((term_count <= 512u) ? 2u : 1u);
  if (const char *venv = std::getenv("CUBBS_BBS_LIMLEE_V_SUB")) {
    const unsigned long parsed = std::strtoul(venv, nullptr, 10);
    if (parsed == 1ul || parsed == 2ul || parsed == 4ul) {
      v_sub = static_cast<uint32_t>(parsed);
    }
  }
  size_t free_vram = 0u, total_vram = 0u;
  if (cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess && free_vram > 0u) {
    while (group_size > 14u) {
      const uint32_t ng =
          static_cast<uint32_t>((term_count + group_size - 1u) / group_size);
      const size_t approx_entries =
          static_cast<size_t>(ng) * v_sub * (size_t{1u} << (group_size - 1u));
      if (approx_entries * 288u <= (free_vram * 55u) / 100u) break;
      if (v_sub > 1u) v_sub >>= 1u;
      else --group_size;
    }
  }
  const uint32_t num_groups =
      static_cast<uint32_t>((term_count + group_size - 1u) / group_size);
  const uint32_t total_subtables = num_groups * v_sub;
  std::vector<size_t> offsets(total_subtables);
  size_t total_entries = 0u;
  for (uint32_t g = 0u; g < num_groups; ++g) {
    const size_t start = static_cast<size_t>(g) * group_size;
    const uint32_t gl = static_cast<uint32_t>(
        std::min<size_t>(group_size, term_count - start));
    const size_t entries = size_t{1u} << (gl - 1u);
    for (uint32_t sv = 0u; sv < v_sub; ++sv) {
      offsets[g * v_sub + sv] = total_entries;
      total_entries += entries;
    }
  }

  cubbs_sppark_bucket_t *tmp_q = nullptr;
  cubbs_sppark_bucket_t *tmp_xyzz = nullptr;
  cudaError_t err = cudaMalloc(&cached->device_limlee_offsets,
                               total_subtables * sizeof(size_t));
  if (err == cudaSuccess) {
    err = cudaMalloc(&cached->device_limlee_c_total, sizeof(cubbs_sppark_bucket_t));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&cached->device_limlee_affines,
                     total_entries * sizeof(cubbs_sppark_plain_affine_t));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&tmp_q, term_count * v_sub * sizeof(cubbs_sppark_bucket_t));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&tmp_xyzz, total_entries * sizeof(cubbs_sppark_bucket_t));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(cached->device_limlee_offsets, offsets.data(),
                     total_subtables * sizeof(size_t), cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    constexpr uint32_t q_threads = 128u;
    const uint32_t q_blocks =
        static_cast<uint32_t>((term_count + q_threads - 1u) / q_threads);
    cubbs_bbs_limlee_prepare_q_kernel<<<q_blocks, q_threads>>>(
        cached->device_points, term_count, v_sub, tmp_q);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    cubbs_bbs_limlee_prepare_c_total_kernel<<<1u, 128u>>>(
        cached->device_points, term_count, cached->device_limlee_c_total);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    cubbs_bbs_limlee_build_subtables_kernel<<<total_subtables, 256u>>>(
        tmp_q, term_count, group_size, v_sub, cached->device_limlee_offsets, tmp_xyzz);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    constexpr uint32_t norm_threads = 256u;
    const uint32_t norm_blocks =
        static_cast<uint32_t>((total_entries + norm_threads - 1u) / norm_threads);
    cubbs_bbs_g1_comb_xyzz_to_affine_kernel<<<norm_blocks, norm_threads>>>(
        tmp_xyzz, cached->device_limlee_affines, total_entries);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (tmp_xyzz != nullptr) (void)cudaFree(tmp_xyzz);
  if (tmp_q != nullptr) (void)cudaFree(tmp_q);
  if (err == cudaSuccess) {
    cached->limlee_num_groups = num_groups;
    cached->limlee_group_size = group_size;
    cached->limlee_v_sub = v_sub;
    cached->limlee_total_entries = total_entries;
  }
  return err;
}

static int32_t get_fused_common_points_cache(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    std::shared_ptr<CuBbsFusedCommonPointsCache> *cached_out) {
  if (points == nullptr || cached_out == nullptr || term_count == 0u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t term = 0u; term < term_count; ++term) {
    if (points[term].data == nullptr || points[term].len != 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }

  // Lock-free 4-slot thread-local L1 cache avoids mutex & key allocation on steady-state hits.
  struct CuBbsTlsCommonPointsEntry {
    int device{-1};
    size_t term_count{0u};
    std::vector<uint8_t> raw_points;
    std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  };
  static thread_local std::array<CuBbsTlsCommonPointsEntry, 4> tls_l1;
  static thread_local size_t tls_l1_next = 0u;
  int current_device = 0;
  (void)cudaGetDevice(&current_device);
  for (const auto &entry : tls_l1) {
    if (entry.cached != nullptr && entry.device == current_device &&
        entry.term_count == term_count &&
        entry.raw_points.size() == term_count * 48u) {
      bool exact = true;
      const uint8_t *expected = entry.raw_points.data();
      for (size_t term = 0u; term < term_count; ++term) {
        if (std::memcmp(expected + (term * 48u), points[term].data, 48u) != 0) {
          exact = false;
          break;
        }
      }
      if (exact) {
        *cached_out = entry.cached;
        return CUBBS_BBS_GPU_SUCCESS;
      }
    }
  }

  auto store_tls_l1 = [&](const std::shared_ptr<CuBbsFusedCommonPointsCache> &val) {
    auto &slot = tls_l1[tls_l1_next & 3u];
    tls_l1_next = (tls_l1_next + 1u) & 3u;
    slot.device = current_device;
    slot.term_count = term_count;
    slot.raw_points.resize(term_count * 48u);
    for (size_t term = 0u; term < term_count; ++term) {
      std::memcpy(slot.raw_points.data() + (term * 48u), points[term].data, 48u);
    }
    slot.cached = val;
  };

  const std::string key = fused_common_points_cache_key(points, term_count);
  {
    std::lock_guard<std::mutex> lock(fused_common_points_cache_mutex());
    auto &cache = fused_common_points_cache();
    auto found = cache.find(key);
    if (found != cache.end()) {
      store_tls_l1(found->second);
      *cached_out = found->second;
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  std::lock_guard<std::mutex> build_lock(fused_common_points_build_mutex(key));
  {
    std::lock_guard<std::mutex> lock(fused_common_points_cache_mutex());
    auto &cache = fused_common_points_cache();
    auto found = cache.find(key);
    if (found != cache.end()) {
      store_tls_l1(found->second);
      *cached_out = found->second;
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  std::vector<uint8_t> decoded_points(term_count * sizeof(cubbs_sppark_plain_affine_t));
  const int32_t status = cubbs_bbs_blst_decode_g1_points(
      points, term_count, decoded_points.data(), sizeof(cubbs_sppark_plain_affine_t));
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;

  auto cached = std::make_shared<CuBbsFusedCommonPointsCache>();
  cached->term_count = term_count;
  const bool use_large_pip = (term_count > 2100u);
  const size_t point_bytes = term_count * sizeof(cubbs_sppark_plain_affine_t);
  const size_t alloc_point_bytes = use_large_pip ? (2u * point_bytes) : point_bytes;
  cudaError_t cuda_status = cudaMalloc(&cached->device_points, alloc_point_bytes);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpy(cached->device_points, decoded_points.data(),
                             point_bytes, cudaMemcpyHostToDevice);
  }
  if (cuda_status == cudaSuccess) {
    if (use_large_pip) {
      cached->use_pippenger_large = true;
      constexpr uint32_t glv_t = 256u;
      const uint32_t glv_b = static_cast<uint32_t>((term_count + glv_t - 1u) / glv_t);
      cubbs_large_pip_fill_glv_points_kernel<<<glv_b, glv_t>>>(
          cached->device_points, term_count);
      cuda_status = cudaGetLastError();
      if (cuda_status == cudaSuccess) cuda_status = cudaDeviceSynchronize();
    } else {
      cuda_status = build_limlee_table_for_cache(cached.get(), term_count);
    }
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaStreamCreateWithFlags(&cached->stream, cudaStreamNonBlocking);
  }
  if (cuda_status != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  retain_fused_common_points_cache(key, cached);
  store_tls_l1(cached);
  *cached_out = cached;
  return CUBBS_BBS_GPU_SUCCESS;
}

static cudaError_t cubbs_bbs_launch_cached_common_msm_range(
    const CuBbsFusedCommonPointsCache &cached,
    const uint8_t *device_scalars_le,
    size_t term_count,
    size_t output_count,
    cubbs_sppark_bucket_t *device_outputs,
    cudaStream_t stream) {
  if (output_count == 0u || term_count == 0u) return cudaSuccess;
  if (cached.use_pippenger_large) {
    return cubbs_bbs_launch_pippenger_large_glv_msm(
        cached.device_points, device_scalars_le, term_count, output_count,
        device_outputs, stream);
  }
  return cubbs_bbs_launch_glv_limlee(
      cached.device_limlee_affines, cached.device_limlee_offsets,
      cached.device_limlee_c_total, device_scalars_le, term_count,
      output_count, cached.limlee_num_groups, cached.limlee_group_size,
      cached.limlee_v_sub, stream, device_outputs);
}

#endif // CUBBS_BBS_COMMON_CACHE_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
