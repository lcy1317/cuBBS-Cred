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
    return term_count * (buckets * sizeof(cubbs_sppark_bucket_t) +
                         affines * sizeof(cubbs_sppark_plain_affine_t));
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
      "CUBBS_BBS_GPU_TABLE_CACHE_BUDGET_BYTES", 2u * 1024u * 1024u * 1024u);
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
  const std::string key = fused_common_points_cache_key(
      points, term_count, large_window4, affine_comb, fixed_comb_bits,
      universal_window4);
  {
    std::lock_guard<std::mutex> lock(fused_common_points_cache_mutex());
    auto &cache = fused_common_points_cache();
    auto found = cache.find(key);
    if (found != cache.end()) {
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
  if (cuda_status == cudaSuccess && build_fixed_comb) {
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
  if (cuda_status == cudaSuccess && build_affine_comb) {
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
  if (cuda_status == cudaSuccess && build_fixed_comb) {
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
  if (cuda_status == cudaSuccess && build_affine_comb) {
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
