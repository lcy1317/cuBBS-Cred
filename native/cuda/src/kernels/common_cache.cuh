#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

struct CuBbsFusedCommonPointsCache {
  std::mutex mutex;
  cubbs_sppark_plain_affine_t *device_points{nullptr};
  cubbs_sppark_bucket_t *device_point_multiples{nullptr};
  cubbs_sppark_bucket_t *device_point_glv_multiples{nullptr};
  cubbs_sppark_bucket_t *device_point_glv_wnaf_multiples{nullptr};
  cubbs_sppark_bucket_t *device_point_multiples8{nullptr};
  cubbs_sppark_bucket_t *device_fixed_comb4{nullptr};
  cubbs_sppark_bucket_t *device_fixed_comb5{nullptr};
  cubbs_sppark_bucket_t *device_fixed_comb6{nullptr};
  cubbs_sppark_bucket_t *device_fixed_comb8{nullptr};
  cubbs_sppark_bucket_t *device_fixed_comb9{nullptr};
  cubbs_sppark_bucket_t *device_fixed_comb10{nullptr};
  cubbs_sppark_plain_affine_t *device_fixed_affine_comb8{nullptr};
  cubbs_sppark_plain_affine_t *device_fixed_affine_comb9{nullptr};
  cubbs_sppark_plain_affine_t *device_fixed_affine_comb10{nullptr};
  cubbs_sppark_plain_affine_t *device_fixed_glv_affine_comb8{nullptr};
  cubbs_sppark_plain_affine_t *device_fixed_glv_affine_comb9{nullptr};
  cubbs_sppark_plain_affine_t *device_fixed_glv_affine_comb10{nullptr};
  cubbs_sppark_plain_affine_t *device_limlee_affines{nullptr};
  size_t *device_limlee_offsets{nullptr};
  cubbs_sppark_bucket_t *device_limlee_c_total{nullptr};
  uint32_t limlee_num_groups{0u};
  uint32_t limlee_group_size{0u};
  uint32_t limlee_v_sub{0u};
  size_t limlee_total_entries{0u};
  bool use_pippenger_large{false};
  uint8_t *device_scalars{nullptr};
  cubbs_sppark_bucket_t *device_outputs{nullptr};
  // Dedicated CUDA stream so launches for different point-set groups can
  // overlap on the device instead of serializing on the default stream.
  cudaStream_t stream{nullptr};
  size_t term_count{0u};
  size_t scalar_capacity{0u};
  size_t output_capacity{0u};
  int device{0};

  CuBbsFusedCommonPointsCache() { (void)cudaGetDevice(&device); }

  // Immutable table bytes only. Per-call/TLS workspaces have separate bounds.
  size_t table_bytes() const {
    size_t buckets = 0u;
    size_t affines = device_points != nullptr ? 1u : 0u;
    if (device_point_multiples) buckets += 16u;
    if (device_point_glv_multiples) buckets += 32u;
    if (device_point_glv_wnaf_multiples) buckets += 128u;
    if (device_point_multiples8) buckets += 256u;
    if (device_fixed_comb4) buckets += 64u * 16u;
    if (device_fixed_comb5) buckets += 52u * 17u;
    if (device_fixed_comb6) buckets += 43u * 33u;
    if (device_fixed_comb8) buckets += 32u * 129u;
    if (device_fixed_comb9) buckets += 29u * 257u;
    if (device_fixed_comb10) buckets += 26u * 513u;
    if (device_fixed_affine_comb8) affines += 32u * 129u;
    if (device_fixed_affine_comb9) affines += 29u * 257u;
    if (device_fixed_affine_comb10) affines += 26u * 513u;
    if (device_fixed_glv_affine_comb8) affines += 17u * 129u;
    if (device_fixed_glv_affine_comb9) affines += 15u * 257u;
    if (device_fixed_glv_affine_comb10) affines += 13u * 513u;
    return term_count * (buckets * sizeof(cubbs_sppark_bucket_t) +
                         affines * sizeof(cubbs_sppark_plain_affine_t)) +
           limlee_total_entries * sizeof(cubbs_sppark_plain_affine_t);
  }

  ~CuBbsFusedCommonPointsCache() {
    int previous = device;
    (void)cudaGetDevice(&previous);
    if (previous != device) (void)cudaSetDevice(device);
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    if (device_outputs != nullptr) {
      (void)cudaFree(device_outputs);
    }
    if (device_scalars != nullptr) {
      (void)cudaFree(device_scalars);
    }
    if (device_points != nullptr) {
      (void)cudaFree(device_points);
    }
    if (device_point_multiples != nullptr) {
      (void)cudaFree(device_point_multiples);
    }
    if (device_point_glv_multiples != nullptr) {
      (void)cudaFree(device_point_glv_multiples);
    }
    if (device_point_glv_wnaf_multiples != nullptr) {
      (void)cudaFree(device_point_glv_wnaf_multiples);
    }
    if (device_point_multiples8 != nullptr) {
      (void)cudaFree(device_point_multiples8);
    }
    if (device_fixed_comb4 != nullptr) {
      (void)cudaFree(device_fixed_comb4);
    }
    if (device_fixed_comb5 != nullptr) {
      (void)cudaFree(device_fixed_comb5);
    }
    if (device_fixed_comb6 != nullptr) {
      (void)cudaFree(device_fixed_comb6);
    }
    if (device_fixed_comb8 != nullptr) {
      (void)cudaFree(device_fixed_comb8);
    }
    if (device_fixed_comb9 != nullptr) {
      (void)cudaFree(device_fixed_comb9);
    }
    if (device_fixed_comb10 != nullptr) {
      (void)cudaFree(device_fixed_comb10);
    }
    if (device_fixed_affine_comb8 != nullptr) {
      (void)cudaFree(device_fixed_affine_comb8);
    }
    if (device_fixed_affine_comb9 != nullptr) {
      (void)cudaFree(device_fixed_affine_comb9);
    }
    if (device_fixed_affine_comb10 != nullptr) {
      (void)cudaFree(device_fixed_affine_comb10);
    }
    if (device_fixed_glv_affine_comb8 != nullptr) {
      (void)cudaFree(device_fixed_glv_affine_comb8);
    }
    if (device_fixed_glv_affine_comb9 != nullptr) {
      (void)cudaFree(device_fixed_glv_affine_comb9);
    }
    if (device_fixed_glv_affine_comb10 != nullptr) {
      (void)cudaFree(device_fixed_glv_affine_comb10);
    }
    if (device_limlee_affines != nullptr) {
      (void)cudaFree(device_limlee_affines);
    }
    if (device_limlee_offsets != nullptr) {
      (void)cudaFree(device_limlee_offsets);
    }
    if (device_limlee_c_total != nullptr) {
      (void)cudaFree(device_limlee_c_total);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
    if (previous != device) (void)cudaSetDevice(previous);
  }
};

static std::mutex &fused_common_points_cache_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<std::string,
                          std::shared_ptr<CuBbsFusedCommonPointsCache>>
    &fused_common_points_cache() {
  static auto *cache = new std::unordered_map<
      std::string, std::shared_ptr<CuBbsFusedCommonPointsCache>>();
  return *cache;
}

static std::mutex &fused_common_points_build_mutex(const std::string &key) {
  // Fixed metadata instead of one immortal mutex for every point set.
  // Hash collisions serialize cold builds only, never steady-state hits.
  static auto *mutexes = new std::array<std::mutex, 64>();
  return (*mutexes)[std::hash<std::string>{}(key) % mutexes->size()];
}

static void retain_fused_common_points_cache(
    const std::string &key, const std::shared_ptr<CuBbsFusedCommonPointsCache> &value) {
  const size_t budget = env_size_or_default(
      "CUBBS_BBS_GPU_TABLE_CACHE_BUDGET_BYTES", 4ull * 1024ull * 1024ull * 1024ull);
  const size_t incoming = value->table_bytes();
  if (incoming > budget || budget == 0u) return;
  std::lock_guard<std::mutex> lock(fused_common_points_cache_mutex());
  auto &cache = fused_common_points_cache();
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
  // Never invalidate active readers. If every table is in use, this new
  // table remains call-owned and is not retained past its last reader.
  if (retained <= budget - incoming) cache.emplace(key, value);
}

static std::string fused_common_points_cache_key(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    bool large_window4, bool affine_comb, size_t fixed_comb_bits,
    bool universal_window4) {
  std::string key;
  int device = 0;
  (void)cudaGetDevice(&device);
  key.reserve(sizeof(device) + 4u + sizeof(term_count) + (term_count * 48u));
  key.append(reinterpret_cast<const char *>(&device), sizeof(device));
  key.push_back(large_window4 ? '\x01' : '\x00');
  key.push_back(universal_window4 ? '\x01' : '\x00');
  key.push_back(affine_comb ? '\x01' : '\x00');
  key.push_back(static_cast<char>(fixed_comb_bits));
  key.append(reinterpret_cast<const char *>(&term_count), sizeof(term_count));
  for (size_t i = 0; i < term_count; ++i) {
    key.append(reinterpret_cast<const char *>(points[i].data), points[i].len);
  }
  return key;
}

static bool bbs_limlee_comb_enabled() {
  const char *env = std::getenv("CUBBS_BBS_GPU_LIMLEE_COMB");
  return env == nullptr || env_flag_enabled("CUBBS_BBS_GPU_LIMLEE_COMB");
}

static cudaError_t build_limlee_table_for_cache(
    CuBbsFusedCommonPointsCache *cached, size_t term_count) {
  // Throughput-optimal (g, v) on GDDR7 (448 GB/s, 16 GB VRAM, 32 MB L2):
  //   - <= 70 terms (M=64, U=32, U=64): g=17, v=4 -> 4 groups x 4 splits (32 doublings, ~78 MB)
  //   - 71..512 terms (M=128/256, U=128/192): g=16, v=2 -> compact uint16_t masks, 64 doublings (~100 MB)
  //   - > 512 terms (U=1024, M=2048): g=18, v=1 -> 114 groups (~1.37 GB, v=1 strictly enforced!)
  uint32_t group_size =
      (term_count <= 70u) ? 17u : ((term_count <= 512u) ? 16u : 18u);
  if (const char *genv = std::getenv("CUBBS_BBS_LIMLEE_GROUP_SIZE")) {
    const unsigned long parsed = std::strtoul(genv, nullptr, 10);
    if (parsed >= 8ul && parsed <= CUBBS_LIMLEE_MAX_GROUP_SIZE) {
      group_size = static_cast<uint32_t>(parsed);
    }
  }
  uint32_t v_sub =
      (term_count <= 70u) ? 4u : ((term_count <= 512u) ? 2u : 1u);
  if (const char *venv = std::getenv("CUBBS_BBS_LIMLEE_V_SUB")) {
    const unsigned long parsed = std::strtoul(venv, nullptr, 10);
    if (parsed == 1ul || parsed == 2ul || parsed == 4ul) {
      v_sub = static_cast<uint32_t>(parsed);
    }
  }
  // Automatically step down g/v if free VRAM cannot hold the 3x build footprint.
  size_t free_vram = 0u, total_vram = 0u;
  if (cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess && free_vram > 0u) {
    while (group_size > 14u) {
      const uint32_t ng = static_cast<uint32_t>((term_count + group_size - 1u) / group_size);
      const size_t approx_entries =
          static_cast<size_t>(ng) * v_sub * (size_t{1u} << (group_size - 1u));
      // Build needs affines (96B) + tmp_xyzz (192B) = 288B/entry; keep <= 55% of free VRAM.
      if (approx_entries * 288u <= (free_vram * 55u) / 100u) break;
      if (v_sub > 1u) {
        v_sub >>= 1u;
      } else {
        --group_size;
      }
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
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
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
    std::shared_ptr<CuBbsFusedCommonPointsCache> *cached_out,
    bool large_window4 = false, bool affine_comb = true,
    size_t fixed_comb_bits_override = 0u,
    bool universal_window4 = false) {
  if (points == nullptr || cached_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  // Validate slice boundaries before cache lookup. Otherwise differently
  // partitioned malformed slices could concatenate to an existing valid key
  // and bypass the BLST decode checks on a cache hit.
  for (size_t term = 0u; term < term_count; ++term) {
    if (points[term].data == nullptr || points[term].len != 48u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }
  if ((universal_window4 && !large_window4) ||
      (!large_window4 &&
       !bbs_specialized_common_points_supported(term_count)) ||
      (large_window4 && !universal_window4 &&
       !bbs_large_shape_kernel_supported(term_count)) ||
      (large_window4 && universal_window4 &&
       !bbs_universal_tiled_common_supported(term_count))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t fixed_comb_bits = large_window4
      ? 0u
      : (fixed_comb_bits_override == 0u
             ? bbs_homogeneous_comb_bits(term_count)
             : fixed_comb_bits_override);
  if (!large_window4 && fixed_comb_bits_override != 0u &&
      fixed_comb_bits != 8u && fixed_comb_bits != 9u &&
      fixed_comb_bits != 10u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  // Lock-free thread-local L1 cache (4-slot MRU) avoids allocating/hashing
  // up to 3.15 MB cache-key strings under the global mutex on steady-state hits
  // while still verifying every point slice byte-for-byte.
  struct CuBbsTlsCommonPointsEntry {
    int device{-1};
    bool large_window4{false};
    bool universal_window4{false};
    bool affine_comb{false};
    size_t fixed_comb_bits{0u};
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
        entry.large_window4 == large_window4 &&
        entry.universal_window4 == universal_window4 &&
        entry.affine_comb == affine_comb &&
        entry.fixed_comb_bits == fixed_comb_bits &&
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
    slot.large_window4 = large_window4;
    slot.universal_window4 = universal_window4;
    slot.affine_comb = affine_comb;
    slot.fixed_comb_bits = fixed_comb_bits;
    slot.term_count = term_count;
    slot.raw_points.resize(term_count * 48u);
    for (size_t term = 0u; term < term_count; ++term) {
      std::memcpy(slot.raw_points.data() + (term * 48u), points[term].data, 48u);
    }
    slot.cached = val;
  };

  const std::string key = fused_common_points_cache_key(
      points, term_count, large_window4, affine_comb, fixed_comb_bits,
      universal_window4);
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

  // Multiple host request streams often encounter the same generator set at
  // process start. Build each table once while still allowing unrelated VC
  // shapes to initialize concurrently.
  std::lock_guard<std::mutex> build_lock(fused_common_points_build_mutex(key));
  {
    std::lock_guard<std::mutex> lock(fused_common_points_cache_mutex());
    auto &cache = fused_common_points_cache();
    auto found = cache.find(key);
    if (found != cache.end()) {
      *cached_out = found->second;
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  std::vector<uint8_t> decoded_points(
      term_count * sizeof(cubbs_sppark_plain_affine_t));
  int32_t status = cubbs_bbs_blst_decode_g1_points(
      points, term_count, decoded_points.data(),
      sizeof(cubbs_sppark_plain_affine_t));
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    return status;
  }

  auto cached = std::make_shared<CuBbsFusedCommonPointsCache>();
  cached->term_count = term_count;
  const size_t point_bytes =
      term_count * sizeof(cubbs_sppark_plain_affine_t);
  const size_t multiple_bytes = term_count * 16u * sizeof(cubbs_sppark_bucket_t);
  const size_t glv_multiple_bytes =
      term_count * 32u * sizeof(cubbs_sppark_bucket_t);
  const size_t glv_wnaf_multiple_bytes =
      term_count * 128u * sizeof(cubbs_sppark_bucket_t);
  const size_t multiple8_bytes =
      term_count * 256u * sizeof(cubbs_sppark_bucket_t);
  if (large_window4) {
    const bool use_limlee_large =
        !universal_window4 && term_count >= 18u && term_count <= 2100u &&
        bbs_limlee_comb_enabled();
    if (use_limlee_large) {
      cudaError_t cuda_status = cudaMalloc(&cached->device_points, point_bytes);
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(cached->device_points, decoded_points.data(),
                                 point_bytes, cudaMemcpyHostToDevice);
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = build_limlee_table_for_cache(cached.get(), term_count);
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaStreamCreateWithFlags(&cached->stream,
                                                cudaStreamNonBlocking);
      }
      if (cuda_status != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
      retain_fused_common_points_cache(key, cached);
      *cached_out = cached;
      return CUBBS_BBS_GPU_SUCCESS;
    }
    const bool use_pip_large =
        term_count > 2100u &&
        (std::getenv("CUBBS_BBS_LARGE_PIPPENGER") == nullptr ||
         env_flag_enabled("CUBBS_BBS_LARGE_PIPPENGER"));
    if (use_pip_large) {
      cudaError_t cuda_status = cudaMalloc(&cached->device_points, point_bytes);
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(cached->device_points, decoded_points.data(),
                                 point_bytes, cudaMemcpyHostToDevice);
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaStreamCreateWithFlags(&cached->stream,
                                                cudaStreamNonBlocking);
      }
      if (cuda_status != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
      cached->use_pippenger_large = true;
      retain_fused_common_points_cache(key, cached);
      *cached_out = cached;
      return CUBBS_BBS_GPU_SUCCESS;
    }
    const bool build_window8 = bbs_large_shape_window8_enabled();
    cudaError_t cuda_status = cudaMalloc(&cached->device_points, point_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMalloc(&cached->device_point_multiples, multiple_bytes);
    }
    if (cuda_status == cudaSuccess && bbs_large_shape_glv_enabled()) {
      cuda_status = cudaMalloc(&cached->device_point_glv_multiples,
                               glv_multiple_bytes);
    }
    if (cuda_status == cudaSuccess && bbs_large_shape_glv_enabled() &&
        bbs_large_shape_glv_wnaf_enabled()) {
      cuda_status = cudaMalloc(&cached->device_point_glv_wnaf_multiples,
                               glv_wnaf_multiple_bytes);
    }
    if (cuda_status == cudaSuccess && build_window8) {
      cuda_status =
          cudaMalloc(&cached->device_point_multiples8, multiple8_bytes);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(cached->device_points, decoded_points.data(),
                               point_bytes, cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess) {
      cubbs_bbs_g1_window4_multiples_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 16u>>>(
              cached->device_points, term_count,
              cached->device_point_multiples);
      cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess &&
        cached->device_point_glv_multiples != nullptr) {
      cubbs_bbs_g1_glv_window4_multiples_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 16u>>>(
              cached->device_points, term_count,
              cached->device_point_glv_multiples);
      cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess &&
        cached->device_point_glv_wnaf_multiples != nullptr) {
      cubbs_bbs_g1_glv_wnaf8_multiples_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 64u>>>(
              cached->device_points, term_count,
              cached->device_point_glv_wnaf_multiples);
      cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess && build_window8) {
      constexpr uint32_t precompute_threads = 128u;
      const uint32_t precompute_blocks = static_cast<uint32_t>(
          (term_count + precompute_threads - 1u) / precompute_threads);
      cubbs_bbs_g1_window8_multiples_precompute_kernel
          <<<precompute_blocks, precompute_threads>>>(
              cached->device_points, term_count,
              cached->device_point_multiples8);
      cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess) cuda_status = cudaDeviceSynchronize();
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaStreamCreateWithFlags(&cached->stream,
                                              cudaStreamNonBlocking);
    }
    if (cuda_status != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
    retain_fused_common_points_cache(key, cached);
    *cached_out = cached;
    return CUBBS_BBS_GPU_SUCCESS;
  }
  std::vector<cubbs_sppark_bucket_t> point_multiples(term_count * 16u);
  std::vector<cubbs_sppark_bucket_t> point_multiples8(term_count * 256u);
  const bool build_fixed_comb = bbs_homogeneous_comb_enabled();
  const bool build_affine_comb =
      build_fixed_comb && affine_comb && bbs_homogeneous_affine_comb_enabled();
  const bool build_limlee =
      build_affine_comb && term_count >= 18u && term_count <= 2100u &&
      bbs_limlee_comb_enabled();
  if (build_limlee) {
    cudaError_t cuda_status = cudaMalloc(&cached->device_points, point_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(cached->device_points, decoded_points.data(),
                               point_bytes, cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = build_limlee_table_for_cache(cached.get(), term_count);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaStreamCreateWithFlags(&cached->stream,
                                              cudaStreamNonBlocking);
    }
    if (cuda_status != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
    retain_fused_common_points_cache(key, cached);
    *cached_out = cached;
    return CUBBS_BBS_GPU_SUCCESS;
  }
  const char *glv_comb_env = std::getenv("CUBBS_BBS_GPU_GLV_AFFINE_COMB");
  const bool build_glv_affine_comb =
      build_affine_comb &&
      (fixed_comb_bits == 8u || fixed_comb_bits == 9u || fixed_comb_bits == 10u) &&
      (glv_comb_env == nullptr || env_flag_enabled("CUBBS_BBS_GPU_GLV_AFFINE_COMB"));
  const bool build_legacy_fixed_comb = build_fixed_comb && !build_glv_affine_comb;
  const bool build_legacy_affine_comb = build_affine_comb && !build_glv_affine_comb;
  const size_t fixed_comb4_bytes =
      term_count * 64u * 16u * sizeof(cubbs_sppark_bucket_t);
  const size_t fixed_comb5_bytes =
      term_count * 52u * 17u * sizeof(cubbs_sppark_bucket_t);
  const size_t fixed_comb6_bytes =
      term_count * 43u * 33u * sizeof(cubbs_sppark_bucket_t);
  const size_t fixed_comb8_bytes =
      term_count * 32u * 129u * sizeof(cubbs_sppark_bucket_t);
  const size_t fixed_comb9_bytes =
      term_count * 29u * 257u * sizeof(cubbs_sppark_bucket_t);
  const size_t fixed_comb10_bytes =
      term_count * 26u * 513u * sizeof(cubbs_sppark_bucket_t);
  const size_t fixed_affine_comb8_count = term_count * 32u * 129u;
  const size_t fixed_affine_comb9_count = term_count * 29u * 257u;
  const size_t fixed_affine_comb10_count = term_count * 26u * 513u;
  const auto *affine_points =
      reinterpret_cast<const cubbs_sppark_plain_affine_t *>(decoded_points.data());
  for (size_t term = 0; term < term_count; ++term) {
    const size_t base = term * 16u;
    point_multiples[base].inf();
    point_multiples[base + 1u] = affine_points[term];
    for (uint32_t digit = 2u; digit < 16u; ++digit) {
      point_multiples[base + digit] = point_multiples[base + digit - 1u];
      point_multiples[base + digit].add(affine_points[term]);
    }
    const size_t base8 = term * 256u;
    point_multiples8[base8].inf();
    point_multiples8[base8 + 1u] = affine_points[term];
    for (uint32_t digit = 2u; digit < 256u; ++digit) {
      point_multiples8[base8 + digit] = point_multiples8[base8 + digit - 1u];
      point_multiples8[base8 + digit].add(affine_points[term]);
    }
  }
  cudaError_t cuda_status = cudaMalloc(&cached->device_points, point_bytes);
  if (cuda_status == cudaSuccess) {
    cuda_status =
        cudaMalloc(&cached->device_point_multiples, multiple_bytes);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status =
        cudaMalloc(&cached->device_point_multiples8, multiple8_bytes);
  }
  if (cuda_status == cudaSuccess && build_legacy_fixed_comb) {
    if (fixed_comb_bits == 4u) {
      cuda_status = cudaMalloc(&cached->device_fixed_comb4, fixed_comb4_bytes);
    } else if (fixed_comb_bits == 5u) {
      cuda_status = cudaMalloc(&cached->device_fixed_comb5, fixed_comb5_bytes);
    } else if (fixed_comb_bits == 6u) {
      cuda_status = cudaMalloc(&cached->device_fixed_comb6, fixed_comb6_bytes);
    } else if (fixed_comb_bits == 8u) {
      cuda_status = cudaMalloc(&cached->device_fixed_comb8, fixed_comb8_bytes);
    } else if (fixed_comb_bits == 9u) {
      cuda_status = cudaMalloc(&cached->device_fixed_comb9, fixed_comb9_bytes);
    } else {
      cuda_status = cudaMalloc(&cached->device_fixed_comb10, fixed_comb10_bytes);
    }
  }
  if (cuda_status == cudaSuccess && build_legacy_affine_comb) {
    if (fixed_comb_bits == 8u) {
      cuda_status = cudaMalloc(
          &cached->device_fixed_affine_comb8,
          fixed_affine_comb8_count * sizeof(cubbs_sppark_plain_affine_t));
    } else if (fixed_comb_bits == 9u) {
      cuda_status = cudaMalloc(
          &cached->device_fixed_affine_comb9,
          fixed_affine_comb9_count * sizeof(cubbs_sppark_plain_affine_t));
    } else if (fixed_comb_bits == 10u) {
      cuda_status = cudaMalloc(
          &cached->device_fixed_affine_comb10,
          fixed_affine_comb10_count * sizeof(cubbs_sppark_plain_affine_t));
    }
  }
  const size_t glv_comb_count = fixed_comb_bits == 8u
      ? (term_count * 17u * 129u)
      : (fixed_comb_bits == 9u ? (term_count * 15u * 257u)
                               : (term_count * 13u * 513u));
  if (cuda_status == cudaSuccess && build_glv_affine_comb) {
    if (fixed_comb_bits == 8u) {
      cuda_status = cudaMalloc(
          &cached->device_fixed_glv_affine_comb8,
          glv_comb_count * sizeof(cubbs_sppark_plain_affine_t));
    } else if (fixed_comb_bits == 9u) {
      cuda_status = cudaMalloc(
          &cached->device_fixed_glv_affine_comb9,
          glv_comb_count * sizeof(cubbs_sppark_plain_affine_t));
    } else if (fixed_comb_bits == 10u) {
      cuda_status = cudaMalloc(
          &cached->device_fixed_glv_affine_comb10,
          glv_comb_count * sizeof(cubbs_sppark_plain_affine_t));
    }
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpy(cached->device_points, decoded_points.data(),
                             point_bytes, cudaMemcpyHostToDevice);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpy(cached->device_point_multiples,
                             point_multiples.data(), multiple_bytes,
                             cudaMemcpyHostToDevice);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpy(cached->device_point_multiples8,
                             point_multiples8.data(), multiple8_bytes,
                             cudaMemcpyHostToDevice);
  }
  if (cuda_status == cudaSuccess && build_legacy_fixed_comb) {
    if (fixed_comb_bits == 4u) {
      cubbs_bbs_g1_fixed_comb4_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 16u>>>(
              cached->device_points, term_count, cached->device_fixed_comb4);
    } else if (fixed_comb_bits == 5u) {
      cubbs_bbs_g1_fixed_signed_comb5_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 32u>>>(
              cached->device_points, term_count, cached->device_fixed_comb5);
    } else if (fixed_comb_bits == 6u) {
      cubbs_bbs_g1_fixed_signed_comb6_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 64u>>>(
              cached->device_points, term_count, cached->device_fixed_comb6);
    } else if (fixed_comb_bits == 8u) {
      cubbs_bbs_g1_fixed_signed_comb8_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 256u>>>(
              cached->device_points, term_count, cached->device_fixed_comb8);
    } else if (fixed_comb_bits == 9u) {
      cubbs_bbs_g1_fixed_signed_comb9_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 256u>>>(
              cached->device_points, term_count, cached->device_fixed_comb9);
    } else {
      cubbs_bbs_g1_fixed_signed_comb10_precompute_kernel
          <<<static_cast<uint32_t>(term_count), 256u>>>(
              cached->device_points, term_count, cached->device_fixed_comb10);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaDeviceSynchronize();
    }
  }
  if (cuda_status == cudaSuccess && build_legacy_affine_comb) {
    const auto normalize = [&](const cubbs_sppark_bucket_t *input,
                               cubbs_sppark_plain_affine_t *output,
                               size_t count) {
      constexpr uint32_t threads = 256u;
      const uint32_t blocks =
          static_cast<uint32_t>((count + threads - 1u) / threads);
      cubbs_bbs_g1_comb_xyzz_to_affine_kernel<<<blocks, threads>>>(
          input, output, count);
      return cudaGetLastError();
    };
    if (fixed_comb_bits == 8u) {
      cuda_status = normalize(cached->device_fixed_comb8,
                              cached->device_fixed_affine_comb8,
                              fixed_affine_comb8_count);
    } else if (fixed_comb_bits == 9u) {
      cuda_status = normalize(cached->device_fixed_comb9,
                              cached->device_fixed_affine_comb9,
                              fixed_affine_comb9_count);
    } else if (fixed_comb_bits == 10u) {
      cuda_status = normalize(cached->device_fixed_comb10,
                              cached->device_fixed_affine_comb10,
                              fixed_affine_comb10_count);
    }
    if (cuda_status == cudaSuccess) cuda_status = cudaDeviceSynchronize();
  }
  if (cuda_status == cudaSuccess && build_glv_affine_comb) {
    cubbs_sppark_bucket_t *tmp_glv = nullptr;
    cuda_status = cudaMalloc(&tmp_glv, glv_comb_count * sizeof(cubbs_sppark_bucket_t));
    if (cuda_status == cudaSuccess) {
      cubbs_sppark_plain_affine_t *target_affine = nullptr;
      if (fixed_comb_bits == 8u) {
        cubbs_bbs_g1_fixed_glv_comb_precompute_kernel<8u, 17u, 129u>
            <<<static_cast<uint32_t>(term_count), 128u>>>(
                cached->device_points, term_count, tmp_glv);
        target_affine = cached->device_fixed_glv_affine_comb8;
      } else if (fixed_comb_bits == 9u) {
        cubbs_bbs_g1_fixed_glv_comb_precompute_kernel<9u, 15u, 257u>
            <<<static_cast<uint32_t>(term_count), 256u>>>(
                cached->device_points, term_count, tmp_glv);
        target_affine = cached->device_fixed_glv_affine_comb9;
      } else {
        cubbs_bbs_g1_fixed_glv_comb_precompute_kernel<10u, 13u, 513u>
            <<<static_cast<uint32_t>(term_count), 256u>>>(
                cached->device_points, term_count, tmp_glv);
        target_affine = cached->device_fixed_glv_affine_comb10;
      }
      cuda_status = cudaGetLastError();
      if (cuda_status == cudaSuccess) {
        constexpr uint32_t threads = 256u;
        const uint32_t blocks =
            static_cast<uint32_t>((glv_comb_count + threads - 1u) / threads);
        cubbs_bbs_g1_comb_xyzz_to_affine_kernel<<<blocks, threads>>>(
            tmp_glv, target_affine, glv_comb_count);
        cuda_status = cudaGetLastError();
      }
      if (cuda_status == cudaSuccess) cuda_status = cudaDeviceSynchronize();
      (void)cudaFree(tmp_glv);
    }
  }
  // Create a dedicated stream for this point set so launches and copies for
  // different VC-shape groups can overlap instead of serializing on the
  // default stream. This is created once per cached point set.
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaStreamCreateWithFlags(&cached->stream,
                                            cudaStreamNonBlocking);
  }
  if (cuda_status != cudaSuccess) {
    return CUBBS_BBS_GPU_ERR_INTERNAL;
  }

  retain_fused_common_points_cache(key, cached);
  *cached_out = cached;
  return CUBBS_BBS_GPU_SUCCESS;
}

static cudaError_t ensure_fused_common_points_workspace(
    CuBbsFusedCommonPointsCache *cached, size_t scalar_bytes,
    size_t output_count) {
  if (scalar_bytes > cached->scalar_capacity) {
    if (cached->device_scalars != nullptr) {
      (void)cudaFree(cached->device_scalars);
      cached->device_scalars = nullptr;
      cached->scalar_capacity = 0u;
    }
    cudaError_t status = cudaMalloc(&cached->device_scalars, scalar_bytes);
    if (status != cudaSuccess) {
      return status;
    }
    cached->scalar_capacity = scalar_bytes;
  }

  if (output_count > cached->output_capacity) {
    if (cached->device_outputs != nullptr) {
      (void)cudaFree(cached->device_outputs);
      cached->device_outputs = nullptr;
      cached->output_capacity = 0u;
    }
    cudaError_t status = cudaMalloc(
        &cached->device_outputs,
        output_count * sizeof(cubbs_sppark_bucket_t));
    if (status != cudaSuccess) {
      return status;
    }
    cached->output_capacity = output_count;
  }
  return cudaSuccess;
}


#endif
