#include "prepare_b_msm.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <blst.h>

static void set_result(CuBbsBbsGpuResult *results, size_t index, int32_t code,
                       size_t bytes_written) {
  results[index].code = code;
  results[index].value = 0;
  results[index].bytes_written = bytes_written;
}

static void set_result_route(CuBbsBbsGpuResult *results, size_t index,
                             int32_t code, size_t bytes_written,
                             uint8_t route) {
  results[index].code = code;
  results[index].value = route;
  results[index].bytes_written = bytes_written;
}

static constexpr uint8_t CUBBS_BBS_MSM_ROUTE_FUSED_ITEM =
    CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM;
static constexpr uint8_t CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS =
    CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS;

static size_t env_size_or_default(const char *name, size_t default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || parsed == 0ul) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

static bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
         std::strcmp(value, "no") != 0 && std::strcmp(value, "NO") != 0;
}

static bool env_flag_default_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr || value[0] == '\0' || env_flag_enabled(name);
}

static bool gpu_miller_loop_enabled(size_t item_count, bool both_g2_common) {
  if (env_flag_enabled("CUBBS_BBS_FORCE_GPU")) {
    return true;
  }
  const char *configured = std::getenv("CUBBS_BBS_GPU_MILLER_LOOP");
  if (configured != nullptr && configured[0] != '\0') {
    return env_flag_enabled("CUBBS_BBS_GPU_MILLER_LOOP");
  }
  const size_t default_min_items = both_g2_common ? 256u : 512u;
  return item_count >= env_size_or_default(
                           "CUBBS_BBS_GPU_MILLER_MIN_ITEMS",
                           default_min_items);
}

static bool gpu_final_exponentiation_enabled(size_t item_count) {
  if (env_flag_enabled("CUBBS_BBS_FORCE_GPU")) {
    return true;
  }
  const char *configured = std::getenv("CUBBS_BBS_GPU_FINAL_EXP");
  if (configured != nullptr && configured[0] != '\0') {
    return env_flag_enabled("CUBBS_BBS_GPU_FINAL_EXP");
  }
  return item_count >= env_size_or_default(
                           "CUBBS_BBS_GPU_FINAL_EXP_MIN_ITEMS", 512u);
}

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point started, Clock::time_point finished) {
  return std::chrono::duration<double, std::milli>(finished - started).count();
}

static size_t blst_msm_worker_count(size_t item_count) {
  const unsigned int hardware = std::thread::hardware_concurrency();
  const size_t default_workers =
      hardware == 0u ? 8u : std::min<size_t>(32u, static_cast<size_t>(hardware));
  const size_t configured =
      env_size_or_default("CUBBS_BBS_BLST_MSM_WORKERS", default_workers);
  return std::max<size_t>(1u, std::min(item_count, configured));
}

static size_t blst_msm_worker_count_for_terms(size_t item_count,
                                              size_t term_count) {
  if (term_count >= 2u && term_count <= 4u &&
      std::getenv("CUBBS_BBS_BLST_MSM_WORKERS") == nullptr) {
    const unsigned int hardware = std::thread::hardware_concurrency();
    const size_t default_workers =
        hardware == 0u ? 16u : std::min<size_t>(16u, static_cast<size_t>(hardware));
    return std::max<size_t>(1u, std::min(item_count, default_workers));
  }
  return blst_msm_worker_count(item_count);
}

static size_t blst_common_points_msm_worker_count_for_terms(size_t item_count,
                                                            size_t term_count) {
  const char *configured = std::getenv("CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS");
  if (configured == nullptr) {
    return blst_msm_worker_count_for_terms(item_count, term_count);
  }
  (void)term_count;
  return std::max<size_t>(
      1u, std::min(item_count, env_size_or_default(
                                   "CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS",
                                   blst_msm_worker_count_for_terms(item_count, term_count))));
}

static size_t blst_pairing_worker_count(size_t item_count) {
  const unsigned int hardware = std::thread::hardware_concurrency();
  const size_t default_workers =
      hardware == 0u ? 16u : std::min<size_t>(32u, static_cast<size_t>(hardware));
  const size_t configured =
      env_size_or_default("CUBBS_BBS_BLST_PAIRING_WORKERS", default_workers);
  return std::max<size_t>(1u, std::min(item_count, configured));
}

static size_t blst_pippenger_min_terms() {
  return env_size_or_default("CUBBS_BBS_BLST_PIPPENGER_MIN_TERMS", 16u);
}

static size_t blst_pippenger_scratch_limb_count(size_t term_count) {
  const size_t scratch_bytes = blst_p1s_mult_pippenger_scratch_sizeof(term_count);
  return (scratch_bytes + sizeof(limb_t) - 1u) / sizeof(limb_t);
}

struct BlstCommonPointsCacheEntry {
  std::vector<blst_p1_affine> affine_points;
  std::vector<const blst_p1_affine *> affine_point_ptrs;
};

static std::mutex &blst_common_points_cache_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<std::string, std::shared_ptr<BlstCommonPointsCacheEntry>>
    &blst_common_points_cache() {
  static auto *cache =
      new std::unordered_map<std::string, std::shared_ptr<BlstCommonPointsCacheEntry>>();
  return *cache;
}

static std::string blst_common_points_cache_key(const CuBbsBbsGpuByteSlice *points,
                                                size_t term_count) {
  std::string key;
  key.reserve(sizeof(term_count) + (term_count * 48u));
  key.append(reinterpret_cast<const char *>(&term_count), sizeof(term_count));
  for (size_t i = 0; i < term_count; ++i) {
    key.append(reinterpret_cast<const char *>(points[i].data), points[i].len);
  }
  return key;
}

static int32_t get_cached_blst_common_points(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    std::shared_ptr<BlstCommonPointsCacheEntry> *entry_out) {
  const std::string key = blst_common_points_cache_key(points, term_count);
  {
    std::lock_guard<std::mutex> lock(blst_common_points_cache_mutex());
    auto &cache = blst_common_points_cache();
    auto found = cache.find(key);
    if (found != cache.end()) {
      *entry_out = found->second;
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  auto entry = std::make_shared<BlstCommonPointsCacheEntry>();
  entry->affine_points.resize(term_count);
  entry->affine_point_ptrs.resize(term_count);
  for (size_t i = 0; i < term_count; ++i) {
    if (points[i].len != 48u || points[i].data == nullptr ||
        blst_p1_uncompress(&entry->affine_points[i], points[i].data) != BLST_SUCCESS ||
        !blst_p1_affine_in_g1(&entry->affine_points[i])) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    entry->affine_point_ptrs[i] = &entry->affine_points[i];
  }

  std::lock_guard<std::mutex> lock(blst_common_points_cache_mutex());
  auto [inserted, _] = blst_common_points_cache().emplace(key, entry);
  *entry_out = inserted->second;
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t compute_g1_msm_with_blst_pippenger_point_ptrs(
    const CuBbsG1MsmInput &item,
    const std::vector<const blst_p1_affine *> &point_ptrs,
    limb_t *scratch) {
  if (item.term_count != point_ptrs.size() || item.compressed_point_out == nullptr ||
      scratch == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item.term_count == 0) {
    blst_p1 zero{};
    std::memset(&zero, 0, sizeof(zero));
    blst_p1_compress(item.compressed_point_out, &zero);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::vector<std::array<uint8_t, 32>> scalar_le(item.term_count);
  std::vector<const byte *> scalar_ptrs(item.term_count);
  for (size_t i = 0; i < item.term_count; ++i) {
    if (item.scalars[i].len != 32u || item.scalars[i].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    std::reverse_copy(item.scalars[i].data, item.scalars[i].data + 32u, scalar_le[i].data());
    scalar_ptrs[i] = reinterpret_cast<const byte *>(scalar_le[i].data());
  }

  blst_p1 out{};
  blst_p1s_mult_pippenger(&out, point_ptrs.data(), item.term_count, scalar_ptrs.data(), 256u,
                          scratch);
  blst_p1_compress(item.compressed_point_out, &out);
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t compute_g1_msm_with_blst_pippenger_decoded_scalars(
    const CuBbsG1MsmInput &item,
    const std::vector<const blst_p1_affine *> &point_ptrs,
    const byte *const *scalar_ptrs,
    limb_t *scratch) {
  if (item.term_count != point_ptrs.size() || item.compressed_point_out == nullptr ||
      scalar_ptrs == nullptr || scratch == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item.term_count == 0) {
    blst_p1 zero{};
    std::memset(&zero, 0, sizeof(zero));
    blst_p1_compress(item.compressed_point_out, &zero);
    return CUBBS_BBS_GPU_SUCCESS;
  }

  blst_p1 out{};
  blst_p1s_mult_pippenger(&out, point_ptrs.data(), item.term_count, scalar_ptrs, 256u,
                          scratch);
  blst_p1_compress(item.compressed_point_out, &out);
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t compute_g1_msm_with_blst_pippenger_affines(
    const CuBbsG1MsmInput &item, const std::vector<blst_p1_affine> &affines) {
  if (item.term_count != affines.size()) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  std::vector<const blst_p1_affine *> point_ptrs(item.term_count);
  for (size_t i = 0; i < item.term_count; ++i) {
    point_ptrs[i] = &affines[i];
  }
  std::unique_ptr<limb_t[]> scratch{new limb_t[blst_pippenger_scratch_limb_count(item.term_count)]};
  return compute_g1_msm_with_blst_pippenger_point_ptrs(item, point_ptrs, scratch.get());
}

static int32_t compute_g1_msm_with_blst_pippenger(const CuBbsG1MsmInput &item) {
  if ((item.term_count > 0 && (item.points == nullptr || item.scalars == nullptr)) ||
      item.compressed_point_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  std::vector<blst_p1_affine> affines(item.term_count);
  for (size_t i = 0; i < item.term_count; ++i) {
    if (item.points[i].len != 48u || item.points[i].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (blst_p1_uncompress(&affines[i], item.points[i].data) != BLST_SUCCESS ||
        !blst_p1_affine_in_g1(&affines[i])) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }
  return compute_g1_msm_with_blst_pippenger_affines(item, affines);
}

static int32_t compute_g1_msm_with_blst(const CuBbsG1MsmInput &item) {
  if (item.term_count >= blst_pippenger_min_terms()) {
    return compute_g1_msm_with_blst_pippenger(item);
  }

  blst_p1 accumulator{};
  std::memset(&accumulator, 0, sizeof(accumulator));

  for (size_t i = 0; i < item.term_count; ++i) {
    blst_p1_affine affine{};
    if (blst_p1_uncompress(&affine, item.points[i].data) != BLST_SUCCESS) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (!blst_p1_affine_in_g1(&affine)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }

    blst_p1 point{};
    blst_p1_from_affine(&point, &affine);

    uint8_t scalar_le[32];
    std::reverse_copy(item.scalars[i].data, item.scalars[i].data + 32u, scalar_le);

    blst_p1 term{};
    blst_p1_mult(&term, &point, scalar_le, 256u);
    blst_p1_add_or_double(&accumulator, &accumulator, &term);
    std::memset(scalar_le, 0, sizeof(scalar_le));
  }

  blst_p1_compress(item.compressed_point_out, &accumulator);
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t compute_g1_msm_with_blst_projective_points(
    const CuBbsG1MsmInput &item, const std::vector<blst_p1> &points) {
  blst_p1 accumulator{};
  std::memset(&accumulator, 0, sizeof(accumulator));

  for (size_t i = 0; i < item.term_count; ++i) {
    uint8_t scalar_le[32];
    std::reverse_copy(item.scalars[i].data, item.scalars[i].data + 32u, scalar_le);

    blst_p1 term{};
    blst_p1_mult(&term, &points[i], scalar_le, 256u);
    blst_p1_add_or_double(&accumulator, &accumulator, &term);
    std::memset(scalar_le, 0, sizeof(scalar_le));
  }

  blst_p1_compress(item.compressed_point_out, &accumulator);
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t add_g1_points_with_blst(const CuBbsBbsGpuG1AddItem &item) {
  if (item.lhs.len != 48u || item.lhs.data == nullptr ||
      item.rhs.len != 48u || item.rhs.data == nullptr ||
      item.point_out.data == nullptr || item.point_out.capacity < 48u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_p1_affine lhs_affine{};
  blst_p1_affine rhs_affine{};
  if (blst_p1_uncompress(&lhs_affine, item.lhs.data) != BLST_SUCCESS ||
      blst_p1_uncompress(&rhs_affine, item.rhs.data) != BLST_SUCCESS ||
      !blst_p1_affine_in_g1(&lhs_affine) || !blst_p1_affine_in_g1(&rhs_affine)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_p1 lhs{};
  blst_p1 rhs{};
  blst_p1_from_affine(&lhs, &lhs_affine);
  blst_p1_from_affine(&rhs, &rhs_affine);
  blst_p1_add_or_double(&lhs, &lhs, &rhs);
  blst_p1_compress(item.point_out.data, &lhs);
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t pairing_eq_with_blst(const CuBbsBbsGpuPairingEqItem &item,
                                    uint8_t *value_out) {
  if (item.lhs_g1.len != 48u || item.lhs_g1.data == nullptr ||
      item.rhs_g1.len != 48u || item.rhs_g1.data == nullptr ||
      item.lhs_g2.len != 96u || item.lhs_g2.data == nullptr ||
      item.rhs_g2.len != 96u || item.rhs_g2.data == nullptr ||
      value_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_p1_affine lhs_g1{};
  blst_p1_affine rhs_g1{};
  blst_p2_affine lhs_g2{};
  blst_p2_affine rhs_g2{};
  if (blst_p1_uncompress(&lhs_g1, item.lhs_g1.data) != BLST_SUCCESS ||
      blst_p1_uncompress(&rhs_g1, item.rhs_g1.data) != BLST_SUCCESS ||
      blst_p2_uncompress(&lhs_g2, item.lhs_g2.data) != BLST_SUCCESS ||
      blst_p2_uncompress(&rhs_g2, item.rhs_g2.data) != BLST_SUCCESS) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (!blst_p1_affine_in_g1(&lhs_g1) || !blst_p1_affine_in_g1(&rhs_g1) ||
      !blst_p2_affine_in_g2(&lhs_g2) || !blst_p2_affine_in_g2(&rhs_g2)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_fp12 lhs{};
  blst_fp12 rhs{};
  blst_miller_loop(&lhs, &lhs_g2, &lhs_g1);
  blst_miller_loop(&rhs, &rhs_g2, &rhs_g1);
  *value_out = blst_fp12_finalverify(&lhs, &rhs) ? 1u : 0u;
  return CUBBS_BBS_GPU_SUCCESS;
}

static bool same_g2_bytes(const CuBbsBbsGpuByteSlice &lhs,
                          const CuBbsBbsGpuByteSlice &rhs) {
  return lhs.len == 96u && rhs.len == 96u && lhs.data != nullptr &&
         rhs.data != nullptr && std::memcmp(lhs.data, rhs.data, 96u) == 0;
}

static int32_t decode_g2_checked(const CuBbsBbsGpuByteSlice &slice,
                                 blst_p2_affine *out) {
  if (slice.len != 96u || slice.data == nullptr || out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (blst_p2_uncompress(out, slice.data) != BLST_SUCCESS ||
      !blst_p2_affine_in_g2(out)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

struct BlstPreparedG2Lines {
  blst_p2_affine affine{};
  std::array<blst_fp6, 68> lines{};
};

static std::mutex &blst_g2_lines_cache_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<std::string, std::shared_ptr<BlstPreparedG2Lines>> &
blst_g2_lines_cache() {
  static auto *cache = new std::unordered_map<
      std::string, std::shared_ptr<BlstPreparedG2Lines>>();
  return *cache;
}

static int32_t prepare_g2_lines_checked(const CuBbsBbsGpuByteSlice &slice,
                                        BlstPreparedG2Lines *out) {
  if (out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const int32_t status = decode_g2_checked(slice, &out->affine);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    return status;
  }
  blst_precompute_lines(out->lines.data(), &out->affine);
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t get_prepared_g2_lines(
    const CuBbsBbsGpuByteSlice &slice,
    std::shared_ptr<BlstPreparedG2Lines> *out) {
  if (slice.len != 96u || slice.data == nullptr || out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const std::string key(reinterpret_cast<const char *>(slice.data), slice.len);
  {
    std::lock_guard<std::mutex> lock(blst_g2_lines_cache_mutex());
    const auto found = blst_g2_lines_cache().find(key);
    if (found != blst_g2_lines_cache().end()) {
      *out = found->second;
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }

  auto prepared = std::make_shared<BlstPreparedG2Lines>();
  const int32_t status = prepare_g2_lines_checked(slice, prepared.get());
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    return status;
  }
  {
    std::lock_guard<std::mutex> lock(blst_g2_lines_cache_mutex());
    const size_t max_entries =
        env_size_or_default("CUBBS_BBS_BLST_G2_LINES_CACHE_ENTRIES", 4096u);
    if (blst_g2_lines_cache().size() >= max_entries) {
      blst_g2_lines_cache().clear();
    }
    const auto inserted = blst_g2_lines_cache().emplace(key, prepared);
    *out = inserted.first->second;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t pairing_eq_with_blst_cached_g2(
    const CuBbsBbsGpuPairingEqItem &item,
    const BlstPreparedG2Lines *common_lhs_g2,
    const BlstPreparedG2Lines *common_rhs_g2, uint8_t *value_out) {
  if (item.lhs_g1.len != 48u || item.lhs_g1.data == nullptr ||
      item.rhs_g1.len != 48u || item.rhs_g1.data == nullptr ||
      value_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_p1_affine lhs_g1{};
  blst_p1_affine rhs_g1{};
  if (blst_p1_uncompress(&lhs_g1, item.lhs_g1.data) != BLST_SUCCESS ||
      blst_p1_uncompress(&rhs_g1, item.rhs_g1.data) != BLST_SUCCESS ||
      !blst_p1_affine_in_g1(&lhs_g1) || !blst_p1_affine_in_g1(&rhs_g1)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_p2_affine lhs_g2{};
  blst_p2_affine rhs_g2{};
  std::shared_ptr<BlstPreparedG2Lines> cached_lhs_g2;
  std::shared_ptr<BlstPreparedG2Lines> cached_rhs_g2;
  const bool cache_per_item =
      env_flag_default_enabled("CUBBS_BBS_BLST_PAIRING_PRECOMPUTE_LINES");
  if (common_lhs_g2 == nullptr) {
    const int32_t status = cache_per_item
                               ? get_prepared_g2_lines(item.lhs_g2, &cached_lhs_g2)
                               : decode_g2_checked(item.lhs_g2, &lhs_g2);
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
  }
  if (common_rhs_g2 == nullptr) {
    const int32_t status = cache_per_item
                               ? get_prepared_g2_lines(item.rhs_g2, &cached_rhs_g2)
                               : decode_g2_checked(item.rhs_g2, &rhs_g2);
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      return status;
    }
  }

  blst_fp12 lhs{};
  blst_fp12 rhs{};
  if (common_lhs_g2 != nullptr) {
    blst_miller_loop_lines(&lhs, common_lhs_g2->lines.data(), &lhs_g1);
  } else if (cached_lhs_g2 != nullptr) {
    blst_miller_loop_lines(&lhs, cached_lhs_g2->lines.data(), &lhs_g1);
  } else {
    blst_miller_loop(&lhs, &lhs_g2, &lhs_g1);
  }
  if (common_rhs_g2 != nullptr) {
    blst_miller_loop_lines(&rhs, common_rhs_g2->lines.data(), &rhs_g1);
  } else if (cached_rhs_g2 != nullptr) {
    blst_miller_loop_lines(&rhs, cached_rhs_g2->lines.data(), &rhs_g1);
  } else {
    blst_miller_loop(&rhs, &rhs_g2, &rhs_g1);
  }
  *value_out = blst_fp12_finalverify(&lhs, &rhs) ? 1u : 0u;
  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t decode_pairing_eq_scaled_item(
    const CuBbsBbsGpuPairingEqScaledItem &item, blst_p1_affine *lhs_g1_out,
    blst_p2_affine *lhs_g2_out, blst_p1_affine *rhs_g1_out,
    blst_p2_affine *rhs_g2_out) {
  if (item.lhs_g1.len != 48u || item.lhs_g1.data == nullptr ||
      item.rhs_g1.len != 48u || item.rhs_g1.data == nullptr ||
      item.lhs_g2.len != 96u || item.lhs_g2.data == nullptr ||
      item.rhs_g2.len != 96u || item.rhs_g2.data == nullptr ||
      item.coefficient.len != 32u || item.coefficient.data == nullptr ||
      lhs_g1_out == nullptr || lhs_g2_out == nullptr || rhs_g1_out == nullptr ||
      rhs_g2_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  blst_p1_affine lhs_g1{};
  blst_p1_affine rhs_g1{};
  blst_p2_affine lhs_g2{};
  blst_p2_affine rhs_g2{};
  if (blst_p1_uncompress(&lhs_g1, item.lhs_g1.data) != BLST_SUCCESS ||
      blst_p1_uncompress(&rhs_g1, item.rhs_g1.data) != BLST_SUCCESS ||
      blst_p2_uncompress(&lhs_g2, item.lhs_g2.data) != BLST_SUCCESS ||
      blst_p2_uncompress(&rhs_g2, item.rhs_g2.data) != BLST_SUCCESS) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (!blst_p1_affine_in_g1(&lhs_g1) || !blst_p1_affine_in_g1(&rhs_g1) ||
      !blst_p2_affine_in_g2(&lhs_g2) || !blst_p2_affine_in_g2(&rhs_g2)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  uint8_t scalar_le[32];
  std::reverse_copy(item.coefficient.data, item.coefficient.data + 32u, scalar_le);

  blst_p1 lhs_projective{};
  blst_p1 rhs_projective{};
  blst_p1_from_affine(&lhs_projective, &lhs_g1);
  blst_p1_from_affine(&rhs_projective, &rhs_g1);
  blst_p1_mult(&lhs_projective, &lhs_projective, scalar_le, 256u);
  blst_p1_mult(&rhs_projective, &rhs_projective, scalar_le, 256u);
  std::memset(scalar_le, 0, sizeof(scalar_le));

  blst_p1_to_affine(lhs_g1_out, &lhs_projective);
  blst_p1_to_affine(rhs_g1_out, &rhs_projective);
  *lhs_g2_out = lhs_g2;
  *rhs_g2_out = rhs_g2;
  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_blst_decode_g1_msm_terms(const CuBbsG1MsmInput *item,
                                            uint8_t *affine_out,
                                            size_t affine_stride,
                                            uint8_t *scalar_out,
                                            size_t scalar_stride) {
  if (item == nullptr || (item->term_count > 0 && (affine_out == nullptr || scalar_out == nullptr)) ||
      affine_stride < sizeof(blst_p1_affine) || scalar_stride < sizeof(blst_scalar)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  int32_t status =
      cubbs_bbs_blst_decode_g1_points(item->points, item->term_count, affine_out, affine_stride);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    return status;
  }
  return cubbs_bbs_blst_decode_scalars(item->scalars, item->term_count, scalar_out, scalar_stride);
}

int32_t cubbs_bbs_blst_decode_g1_points(const CuBbsBbsGpuByteSlice *points,
                                         size_t point_count,
                                         uint8_t *affine_out,
                                         size_t affine_stride) {
  if ((point_count > 0 && (points == nullptr || affine_out == nullptr)) ||
      affine_stride < sizeof(blst_p1_affine)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  auto decode_one = [&](size_t i) -> int32_t {
    if (points[i].len != 48u || points[i].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }

    auto *affine = reinterpret_cast<blst_p1_affine *>(affine_out + (i * affine_stride));
    if (blst_p1_uncompress(affine, points[i].data) != BLST_SUCCESS) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    if (!blst_p1_affine_in_g1(affine)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    return CUBBS_BBS_GPU_SUCCESS;
  };

  if (point_count < 32u) {
    for (size_t i = 0; i < point_count; ++i) {
      const int32_t status = decode_one(i);
      if (status != CUBBS_BBS_GPU_SUCCESS) {
        return status;
      }
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  std::atomic<size_t> next_index{0u};
  std::atomic<int32_t> first_error{CUBBS_BBS_GPU_SUCCESS};
  const size_t worker_count = blst_msm_worker_count(point_count);
  cubbs_bbs_run_host_tasks(worker_count, [&](size_t) {
      while (first_error.load(std::memory_order_relaxed) ==
             CUBBS_BBS_GPU_SUCCESS) {
        const size_t i = next_index.fetch_add(1u, std::memory_order_relaxed);
        if (i >= point_count) {
          break;
        }
        const int32_t status = decode_one(i);
        if (status != CUBBS_BBS_GPU_SUCCESS) {
          int32_t expected = CUBBS_BBS_GPU_SUCCESS;
          first_error.compare_exchange_strong(expected, status);
          break;
        }
      }
    });
  return first_error.load();
}

int32_t cubbs_bbs_blst_decode_scalars(const CuBbsBbsGpuByteSlice *scalars,
                                       size_t scalar_count,
                                       uint8_t *scalar_out,
                                       size_t scalar_stride) {
  if ((scalar_count > 0 && (scalars == nullptr || scalar_out == nullptr)) ||
      scalar_stride < sizeof(blst_scalar)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  for (size_t i = 0; i < scalar_count; ++i) {
    if (scalars[i].len != 32u || scalars[i].data == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }

    auto *scalar = reinterpret_cast<blst_scalar *>(scalar_out + (i * scalar_stride));
    blst_scalar_from_bendian(scalar, scalars[i].data);
    if (!blst_scalar_fr_check(scalar)) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
  }

  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_blst_compress_g1_projective(const void *projective,
                                               uint8_t *compressed_out) {
  if (projective == nullptr || compressed_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  blst_p1_compress(compressed_out, reinterpret_cast<const blst_p1 *>(projective));
  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_blst_compress_g1_projective_batch(
    const void *const *projectives, uint8_t *const *compressed_out,
    size_t point_count) {
  if (point_count > 0u &&
      (projectives == nullptr || compressed_out == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  std::vector<const blst_p1 *> point_ptrs(point_count);
  for (size_t i = 0; i < point_count; ++i) {
    if (projectives[i] == nullptr || compressed_out[i] == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    point_ptrs[i] = reinterpret_cast<const blst_p1 *>(projectives[i]);
  }

  // Normalize in worker-sized chunks. Each chunk uses one Montgomery batch
  // inversion, reducing hundreds of independent inversions while retaining
  // CPU parallelism across the available cores.
  std::vector<blst_p1_affine> affine_points(point_count);
  const size_t worker_count = blst_msm_worker_count(point_count);
  cubbs_bbs_run_host_tasks(worker_count, [&](size_t worker) {
    const size_t begin = (point_count * worker) / worker_count;
    const size_t end = (point_count * (worker + 1u)) / worker_count;
    blst_p1s_to_affine(affine_points.data() + begin,
                       point_ptrs.data() + begin, end - begin);
    for (size_t i = begin; i < end; ++i) {
      blst_p1_affine_compress(compressed_out[i], &affine_points[i]);
    }
  });
  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_blst_g1_msm_batch(const CuBbsG1MsmInput *items,
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

    const int32_t status = compute_g1_msm_with_blst(items[i]);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_BLST_ITEM);
    } else {
      set_result(results, i, status, 0);
    }
  }

  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_blst_g1_msm_common_points_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  const bool profile = env_flag_enabled("CUBBS_BBS_BLST_MSM_PROFILE");
  const auto total_started = profile ? Clock::now() : Clock::time_point{};
  if (item_count > 0 && (items == nullptr || results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (item_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  const size_t term_count = items[0].term_count;
  for (size_t i = 0; i < item_count; ++i) {
    if (items[i].term_count != term_count ||
        (term_count > 0 && (items[i].points == nullptr || items[i].scalars == nullptr)) ||
        items[i].compressed_point_out == nullptr) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    for (size_t j = 0; j < term_count; ++j) {
      if (items[i].points[j].len != 48u || items[i].points[j].data == nullptr ||
          items[i].scalars[j].len != 32u || items[i].scalars[j].data == nullptr) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      if (i > 0 &&
          std::memcmp(items[i].points[j].data, items[0].points[j].data, 48u) != 0) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
    }
  }

  const bool use_pippenger = term_count >= blst_pippenger_min_terms();
  std::shared_ptr<BlstCommonPointsCacheEntry> cached_points;
  std::vector<std::array<uint8_t, 32>> scalar_le;
  std::vector<const byte *> scalar_ptrs;
  const auto points_started = profile ? Clock::now() : Clock::time_point{};
  std::vector<blst_p1> projective_points;
  if (use_pippenger) {
    const int32_t cached_status =
        get_cached_blst_common_points(items[0].points, term_count, &cached_points);
    if (cached_status != CUBBS_BBS_GPU_SUCCESS) {
      return cached_status;
    }
  } else {
    projective_points.resize(term_count);
    for (size_t i = 0; i < term_count; ++i) {
      blst_p1_affine affine{};
      if (blst_p1_uncompress(&affine, items[0].points[i].data) != BLST_SUCCESS ||
          !blst_p1_affine_in_g1(&affine)) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      blst_p1_from_affine(&projective_points[i], &affine);
    }
  }
  const auto points_finished = profile ? Clock::now() : Clock::time_point{};

  const auto scalars_started = profile ? Clock::now() : Clock::time_point{};
  if (use_pippenger) {
    scalar_le.resize(item_count * term_count);
    scalar_ptrs.resize(item_count * term_count);
    for (size_t i = 0; i < item_count; ++i) {
      for (size_t j = 0; j < term_count; ++j) {
        const size_t offset = (i * term_count) + j;
        std::reverse_copy(items[i].scalars[j].data, items[i].scalars[j].data + 32u,
                          scalar_le[offset].data());
        scalar_ptrs[offset] = reinterpret_cast<const byte *>(scalar_le[offset].data());
      }
    }
  }
  const auto scalars_finished = profile ? Clock::now() : Clock::time_point{};

  auto run_item = [&](size_t i, limb_t *scratch) {
    const int32_t status =
        use_pippenger
            ? compute_g1_msm_with_blst_pippenger_decoded_scalars(
                  items[i], cached_points->affine_point_ptrs,
                  &scalar_ptrs[i * term_count], scratch)
            : compute_g1_msm_with_blst_projective_points(items[i], projective_points);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS);
    } else {
      set_result(results, i, status, 0);
    }
  };

  const size_t worker_count =
      blst_common_points_msm_worker_count_for_terms(item_count, term_count);
  const auto msm_started = profile ? Clock::now() : Clock::time_point{};
  if (worker_count > 1u) {
    std::atomic<size_t> next_index{0u};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        std::unique_ptr<limb_t[]> scratch;
        if (use_pippenger) {
          scratch.reset(new limb_t[blst_pippenger_scratch_limb_count(term_count)]);
        }
        while (true) {
          const size_t i = next_index.fetch_add(1u);
          if (i >= item_count) {
            break;
          }
          run_item(i, scratch.get());
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
  } else {
    std::unique_ptr<limb_t[]> scratch;
    if (use_pippenger) {
      scratch.reset(new limb_t[blst_pippenger_scratch_limb_count(term_count)]);
    }
    for (size_t i = 0; i < item_count; ++i) {
      run_item(i, scratch.get());
    }
  }
  const auto msm_finished = profile ? Clock::now() : Clock::time_point{};
  if (profile) {
    const double point_ms = elapsed_ms(points_started, points_finished);
    const double scalar_ms = elapsed_ms(scalars_started, scalars_finished);
    const double msm_ms = elapsed_ms(msm_started, msm_finished);
    const double total_ms = elapsed_ms(total_started, msm_finished);
    std::fprintf(stderr,
                 "cubbs_blst_common_points_profile mode=slices "
                 "output_count=%zu term_count=%zu workers=%zu "
                 "point_decode_ms=%.6f scalar_transpose_ms=%.6f "
                 "row_msm_ms=%.6f total_ms=%.6f item_ms=%.6f\n",
                 item_count, term_count, worker_count, point_ms, scalar_ms,
                 msm_ms, total_ms, item_count == 0 ? 0.0 : total_ms / item_count);
  }

  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_blst_g1_msm_common_points_matrix_packed(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const uint8_t *scalar_rows_be, size_t output_count, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results) {
  const bool profile = env_flag_enabled("CUBBS_BBS_BLST_MSM_PROFILE");
  const auto total_started = profile ? Clock::now() : Clock::time_point{};
  if ((term_count > 0 && points == nullptr) ||
      (output_count > 0 &&
       (scalar_rows_be == nullptr || point_outs == nullptr || results == nullptr)) ||
      scalar_row_stride < term_count * 32u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (output_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }
  if (term_count < blst_pippenger_min_terms()) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  const auto points_started = profile ? Clock::now() : Clock::time_point{};
  std::shared_ptr<BlstCommonPointsCacheEntry> cached_points;
  const int32_t cached_status =
      get_cached_blst_common_points(points, term_count, &cached_points);
  if (cached_status != CUBBS_BBS_GPU_SUCCESS) {
    return cached_status;
  }
  const auto points_finished = profile ? Clock::now() : Clock::time_point{};

  std::vector<std::array<uint8_t, 32>> scalar_le(output_count * term_count);
  std::vector<const byte *> scalar_ptrs(output_count * term_count);
  const auto scalars_started = profile ? Clock::now() : Clock::time_point{};
  for (size_t row = 0; row < output_count; ++row) {
    const uint8_t *row_base = scalar_rows_be + (row * scalar_row_stride);
    for (size_t term = 0; term < term_count; ++term) {
      const size_t offset = (row * term_count) + term;
      const uint8_t *scalar_be = row_base + (term * 32u);
      std::reverse_copy(scalar_be, scalar_be + 32u, scalar_le[offset].data());
      scalar_ptrs[offset] = reinterpret_cast<const byte *>(scalar_le[offset].data());
    }
  }
  const auto scalars_finished = profile ? Clock::now() : Clock::time_point{};

  auto run_item = [&](size_t row, limb_t *scratch) {
    if (point_outs[row].data == nullptr || point_outs[row].capacity < 48u ||
        scratch == nullptr) {
      set_result(results, row, CUBBS_BBS_GPU_ERR_INVALID_INPUT, 0);
      return;
    }
    CuBbsG1MsmInput item{points, nullptr, term_count, point_outs[row].data};
    const int32_t status = compute_g1_msm_with_blst_pippenger_decoded_scalars(
        item, cached_points->affine_point_ptrs, &scalar_ptrs[row * term_count], scratch);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                       CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS);
    } else {
      set_result(results, row, status, 0);
    }
  };

  const size_t worker_count =
      blst_common_points_msm_worker_count_for_terms(output_count, term_count);
  const auto msm_started = profile ? Clock::now() : Clock::time_point{};
  if (worker_count > 1u) {
    std::atomic<size_t> next_index{0u};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        std::unique_ptr<limb_t[]> scratch{
            new limb_t[blst_pippenger_scratch_limb_count(term_count)]};
        while (true) {
          const size_t row = next_index.fetch_add(1u);
          if (row >= output_count) {
            break;
          }
          run_item(row, scratch.get());
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
  } else {
    std::unique_ptr<limb_t[]> scratch{
        new limb_t[blst_pippenger_scratch_limb_count(term_count)]};
    for (size_t row = 0; row < output_count; ++row) {
      run_item(row, scratch.get());
    }
  }
  const auto msm_finished = profile ? Clock::now() : Clock::time_point{};
  if (profile) {
    const double point_ms = elapsed_ms(points_started, points_finished);
    const double scalar_ms = elapsed_ms(scalars_started, scalars_finished);
    const double msm_ms = elapsed_ms(msm_started, msm_finished);
    const double total_ms = elapsed_ms(total_started, msm_finished);
    std::fprintf(stderr,
                 "cubbs_blst_common_points_profile mode=packed "
                 "output_count=%zu term_count=%zu "
                 "workers=%zu point_decode_ms=%.6f scalar_transpose_ms=%.6f "
                 "row_msm_ms=%.6f total_ms=%.6f item_ms=%.6f\n",
                 output_count, term_count, worker_count, point_ms, scalar_ms,
                 msm_ms, total_ms, output_count == 0 ? 0.0 : total_ms / output_count);
  }

  return CUBBS_BBS_GPU_SUCCESS;
}

extern "C" int32_t cubbs_bbs_gpu_g1_add_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1AddItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (abi_version != CUBBS_BBS_GPU_ABI_VERSION ||
      (item_count > 0 && items == nullptr) ||
      (item_count > 0 && results == nullptr)) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  auto run_item = [&](size_t i) {
    const int32_t status = add_g1_points_with_blst(items[i]);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      items[i].point_out.len = 48u;
      set_result(results, i, CUBBS_BBS_GPU_SUCCESS, 48u);
    } else {
      set_result(results, i, status, 0);
    }
  };

  const size_t worker_count = blst_msm_worker_count(item_count);
  if (worker_count > 1u) {
    std::atomic<size_t> next_index{0u};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        while (true) {
          const size_t i = next_index.fetch_add(1u);
          if (i >= item_count) {
            break;
          }
          run_item(i);
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
  } else {
    for (size_t i = 0; i < item_count; ++i) {
      run_item(i);
    }
  }

  return CUBBS_BBS_GPU_SUCCESS;
}

static int32_t cubbs_bbs_gpu_pairing_eq_batch_direct_v2(
    uint32_t abi_version, const CuBbsBbsGpuPairingEqItem *items,
    size_t item_count, CuBbsBbsGpuResult *results) {
  if (abi_version != CUBBS_BBS_GPU_ABI_VERSION ||
      (item_count > 0 && (items == nullptr || results == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  std::shared_ptr<BlstPreparedG2Lines> common_lhs_g2;
  std::shared_ptr<BlstPreparedG2Lines> common_rhs_g2;
  bool gpu_decode_invalid = false;
  if (item_count > 0u &&
      env_flag_default_enabled("CUBBS_BBS_BLST_PAIRING_PRECOMPUTE_LINES")) {
    bool lhs_common = true;
    bool rhs_common = true;
    for (size_t i = 1; i < item_count; ++i) {
      lhs_common = lhs_common && same_g2_bytes(items[0].lhs_g2, items[i].lhs_g2);
      rhs_common = rhs_common && same_g2_bytes(items[0].rhs_g2, items[i].rhs_g2);
    }
    if (lhs_common) {
      (void)get_prepared_g2_lines(items[0].lhs_g2, &common_lhs_g2);
    }
    if (rhs_common) {
      (void)get_prepared_g2_lines(items[0].rhs_g2, &common_rhs_g2);
    }
  }

  if (gpu_miller_loop_enabled(
          item_count,
          common_lhs_g2 != nullptr && common_rhs_g2 != nullptr)) {
    const bool profile_gpu_miller =
        env_flag_enabled("CUBBS_BBS_GPU_MILLER_PROFILE");
    const auto gpu_miller_started =
        profile_gpu_miller ? Clock::now() : Clock::time_point{};
    std::vector<blst_p1_affine> lhs_g1(item_count);
    std::vector<blst_p1_affine> rhs_g1(item_count);
    using G2LineTable = std::array<blst_fp6, 68>;
    std::vector<G2LineTable> lhs_item_lines(
        common_lhs_g2 == nullptr ? item_count : 0u);
    std::vector<G2LineTable> rhs_item_lines(
        common_rhs_g2 == nullptr ? item_count : 0u);
    std::atomic<size_t> decode_next{0u};
    std::atomic<bool> valid{true};
    const size_t decode_workers = blst_msm_worker_count(item_count);
    cubbs_bbs_run_host_tasks(decode_workers, [&](size_t) {
        while (valid.load()) {
          const size_t i = decode_next.fetch_add(1u);
          if (i >= item_count) {
            break;
          }
          if (items[i].lhs_g1.len != 48u || items[i].lhs_g1.data == nullptr ||
              items[i].rhs_g1.len != 48u || items[i].rhs_g1.data == nullptr ||
              blst_p1_uncompress(&lhs_g1[i], items[i].lhs_g1.data) != BLST_SUCCESS ||
              blst_p1_uncompress(&rhs_g1[i], items[i].rhs_g1.data) != BLST_SUCCESS ||
              !blst_p1_affine_in_g1(&lhs_g1[i]) ||
              !blst_p1_affine_in_g1(&rhs_g1[i])) {
            valid.store(false);
            break;
          }
          if (common_lhs_g2 == nullptr) {
            std::shared_ptr<BlstPreparedG2Lines> prepared;
            if (get_prepared_g2_lines(items[i].lhs_g2, &prepared) !=
                CUBBS_BBS_GPU_SUCCESS) {
              valid.store(false);
              break;
            }
            lhs_item_lines[i] = prepared->lines;
          }
          if (common_rhs_g2 == nullptr) {
            std::shared_ptr<BlstPreparedG2Lines> prepared;
            if (get_prepared_g2_lines(items[i].rhs_g2, &prepared) !=
                CUBBS_BBS_GPU_SUCCESS) {
              valid.store(false);
              break;
            }
            rhs_item_lines[i] = prepared->lines;
          }
        }
      });
    if (valid.load()) {
      const auto decode_finished =
          profile_gpu_miller ? Clock::now() : Clock::time_point{};
      std::vector<blst_fp12> lhs_values(item_count);
      std::vector<blst_fp12> rhs_values(item_count);
      const void *lhs_lines = common_lhs_g2 != nullptr
                                  ? static_cast<const void *>(common_lhs_g2->lines.data())
                                  : static_cast<const void *>(lhs_item_lines.data());
      const void *rhs_lines = common_rhs_g2 != nullptr
                                  ? static_cast<const void *>(common_rhs_g2->lines.data())
                                  : static_cast<const void *>(rhs_item_lines.data());
      const size_t lhs_lines_stride =
          common_lhs_g2 != nullptr ? 0u : sizeof(G2LineTable);
      const size_t rhs_lines_stride =
          common_rhs_g2 != nullptr ? 0u : sizeof(G2LineTable);
      const bool gpu_final_exp =
          gpu_final_exponentiation_enabled(item_count);
      std::vector<uint8_t> gpu_final_values(gpu_final_exp ? item_count : 0u);
      const int32_t cuda_status =
          cubbs_bbs_cuda_miller_loop_common_lines_batch(
              lhs_g1.data(), rhs_g1.data(), lhs_lines, lhs_lines_stride,
              rhs_lines, rhs_lines_stride, item_count, lhs_values.data(),
              rhs_values.data(),
              gpu_final_exp ? gpu_final_values.data() : nullptr,
              common_lhs_g2 != nullptr ? items[0].lhs_g2.data : nullptr,
              common_rhs_g2 != nullptr ? items[0].rhs_g2.data : nullptr);
      if (cuda_status == CUBBS_BBS_GPU_SUCCESS) {
        const auto cuda_finished =
            profile_gpu_miller ? Clock::now() : Clock::time_point{};
        if (gpu_final_exp) {
          for (size_t i = 0; i < item_count; ++i) {
            results[i].code = CUBBS_BBS_GPU_SUCCESS;
            results[i].value = gpu_final_values[i];
            results[i].bytes_written = 0u;
          }
        } else {
          std::atomic<size_t> next_index{0u};
          const size_t worker_count = blst_pairing_worker_count(item_count);
          std::vector<std::thread> workers;
          workers.reserve(worker_count);
          for (size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&]() {
              while (true) {
                const size_t i = next_index.fetch_add(1u);
                if (i >= item_count) {
                  break;
                }
                results[i].code = CUBBS_BBS_GPU_SUCCESS;
                results[i].value =
                    blst_fp12_finalverify(&lhs_values[i], &rhs_values[i]) ? 1u : 0u;
                results[i].bytes_written = 0u;
              }
            });
          }
          for (auto &worker : workers) {
            worker.join();
          }
        }
        if (profile_gpu_miller) {
          const auto verify_finished = Clock::now();
          std::fprintf(
              stderr,
              "cubbs_gpu_miller_profile items=%zu decode_ms=%.6f "
              "cuda_miller_ms=%.6f finalverify_ms=%.6f total_ms=%.6f\n",
              item_count, elapsed_ms(gpu_miller_started, decode_finished),
              elapsed_ms(decode_finished, cuda_finished),
              elapsed_ms(cuda_finished, verify_finished),
              elapsed_ms(gpu_miller_started, verify_finished));
        }
        return CUBBS_BBS_GPU_SUCCESS;
      }
      if (profile_gpu_miller) {
        std::fprintf(stderr,
                     "cubbs_gpu_miller_profile items=%zu fallback=cuda_status "
                     "status=%d\n",
                     item_count, cuda_status);
      }
    } else {
      gpu_decode_invalid = true;
      if (profile_gpu_miller) {
        std::fprintf(
            stderr,
            "cubbs_gpu_miller_profile items=%zu fallback=decode_or_lines\n",
            item_count);
      }
    }
  }

  // In strict GPU mode a CUDA/decode failure is observable. Never hide it by
  // executing the same pairing on BLST, otherwise a "GPU-only" result would
  // be contaminated by host cryptography.
  if (env_flag_enabled("CUBBS_BBS_FORCE_GPU")) {
    const int32_t failure = gpu_decode_invalid
                                ? CUBBS_BBS_GPU_ERR_INVALID_INPUT
                                : CUBBS_BBS_GPU_ERR_INTERNAL;
    for (size_t i = 0; i < item_count; ++i) {
      set_result(results, i, failure, 0u);
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  auto run_item = [&](size_t i) {
    uint8_t value = 0;
    const int32_t status = pairing_eq_with_blst_cached_g2(
        items[i], common_lhs_g2.get(), common_rhs_g2.get(), &value);
    results[i].code = status;
    results[i].value = status == CUBBS_BBS_GPU_SUCCESS ? value : 0;
    results[i].bytes_written = 0;
  };

  const size_t worker_count = blst_pairing_worker_count(item_count);
  if (worker_count > 1u) {
    std::atomic<size_t> next_index{0u};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        while (true) {
          const size_t i = next_index.fetch_add(1u);
          if (i >= item_count) {
            break;
          }
          run_item(i);
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
  } else {
    for (size_t i = 0; i < item_count; ++i) {
      run_item(i);
    }
  }

  return CUBBS_BBS_GPU_SUCCESS;
}

// Random-coefficient verification can reduce thousands of requests to one
// pairing equation per outer call. Coalesce those synchronous single-item
// calls across host streams before entering the serialized CUDA workspace.
struct CuBbsPairingMicrobatchRequest {
  std::array<uint8_t, 48> lhs_g1{};
  std::array<uint8_t, 48> rhs_g1{};
  std::vector<uint8_t> lhs_g2;
  std::vector<uint8_t> rhs_g2;
  std::mutex mutex;
  std::condition_variable ready;
  bool completed{false};
  int32_t call_status{CUBBS_BBS_GPU_ERR_INTERNAL};
  CuBbsBbsGpuResult result{CUBBS_BBS_GPU_ERR_INTERNAL, 0u, 0u};
};

class CuBbsPairingMicrobatcher {
 public:
  CuBbsPairingMicrobatcher() {
    std::thread([this]() { drain(); }).detach();
  }

  int32_t submit(uint32_t abi_version,
                 const CuBbsBbsGpuPairingEqItem &item,
                 CuBbsBbsGpuResult *result) {
    if (abi_version != CUBBS_BBS_GPU_ABI_VERSION || result == nullptr ||
        item.lhs_g1.data == nullptr ||
        item.lhs_g1.len != 48u || item.rhs_g1.data == nullptr ||
        item.rhs_g1.len != 48u || item.lhs_g2.data == nullptr ||
        item.lhs_g2.len == 0u || item.rhs_g2.data == nullptr ||
        item.rhs_g2.len == 0u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    auto request = std::make_shared<CuBbsPairingMicrobatchRequest>();
    std::memcpy(request->lhs_g1.data(), item.lhs_g1.data, 48u);
    std::memcpy(request->rhs_g1.data(), item.rhs_g1.data, 48u);
    request->lhs_g2.assign(item.lhs_g2.data,
                           item.lhs_g2.data + item.lhs_g2.len);
    request->rhs_g2.assign(item.rhs_g2.data,
                           item.rhs_g2.data + item.rhs_g2.len);

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_.push_back(request);
    }
    queue_changed_.notify_one();

    std::unique_lock<std::mutex> request_lock(request->mutex);
    request->ready.wait(request_lock,
                        [&request]() { return request->completed; });
    *result = request->result;
    return request->call_status;
  }

 private:
  void drain() {
    const size_t max_batch = env_size_or_default(
        "CUBBS_BBS_GPU_PAIRING_MICROBATCH_MAX", 64u);
    const size_t min_batch = std::min(
        max_batch, env_size_or_default(
                       "CUBBS_BBS_GPU_PAIRING_MICROBATCH_MIN", 4u));
    const size_t gather_us = env_size_or_default(
        "CUBBS_BBS_GPU_PAIRING_MICROBATCH_GATHER_US", 300u);
    const bool profile =
        env_flag_enabled("CUBBS_BBS_GPU_PAIRING_MICROBATCH_PROFILE");

    while (true) {
      std::vector<std::shared_ptr<CuBbsPairingMicrobatchRequest>> batch;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_changed_.wait(
            lock, [this]() { return !pending_.empty(); });
        if (pending_.size() < min_batch) {
          queue_changed_.wait_for(
              lock, std::chrono::microseconds(gather_us),
              [this, min_batch]() { return pending_.size() >= min_batch; });
        }
        const size_t count = std::min(max_batch, pending_.size());
        batch.reserve(count);
        for (size_t index = 0u; index < count; ++index) {
          batch.push_back(std::move(pending_.front()));
          pending_.pop_front();
        }
      }

      std::vector<CuBbsBbsGpuPairingEqItem> items;
      items.reserve(batch.size());
      for (const auto &request : batch) {
        items.push_back(CuBbsBbsGpuPairingEqItem{
            CuBbsBbsGpuByteSlice{request->lhs_g1.data(),
                                  request->lhs_g1.size()},
            CuBbsBbsGpuByteSlice{request->lhs_g2.data(),
                                  request->lhs_g2.size()},
            CuBbsBbsGpuByteSlice{request->rhs_g1.data(),
                                  request->rhs_g1.size()},
            CuBbsBbsGpuByteSlice{request->rhs_g2.data(),
                                  request->rhs_g2.size()}});
      }
      std::vector<CuBbsBbsGpuResult> results(batch.size());
      const auto started = profile ? Clock::now() : Clock::time_point{};
      const int32_t call_status = cubbs_bbs_gpu_pairing_eq_batch_direct_v2(
          CUBBS_BBS_GPU_ABI_VERSION, items.data(), items.size(),
          results.data());
      if (profile) {
        std::fprintf(
            stderr,
            "cubbs_gpu_pairing_microbatch items=%zu elapsed_ms=%.6f\n",
            batch.size(),
            elapsed_ms(started, Clock::now()));
      }
      for (size_t index = 0u; index < batch.size(); ++index) {
        const auto &request = batch[index];
        {
          std::lock_guard<std::mutex> request_lock(request->mutex);
          request->call_status = call_status;
          request->result = results[index];
          request->completed = true;
        }
        request->ready.notify_one();
      }

    }
  }

  std::mutex queue_mutex_;
  std::condition_variable queue_changed_;
  std::deque<std::shared_ptr<CuBbsPairingMicrobatchRequest>> pending_;
};

static CuBbsPairingMicrobatcher &cubbs_pairing_microbatcher() {
  static auto *batcher = new CuBbsPairingMicrobatcher();
  return *batcher;
}

extern "C" int32_t cubbs_bbs_gpu_pairing_eq_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuPairingEqItem *items,
    size_t item_count, CuBbsBbsGpuResult *results) {
  if (abi_version == CUBBS_BBS_GPU_ABI_VERSION &&
      env_flag_enabled("CUBBS_BBS_FORCE_GPU") && item_count == 1u &&
      items != nullptr && results != nullptr &&
      env_flag_enabled("CUBBS_BBS_GPU_PAIRING_MICROBATCH")) {
    return cubbs_pairing_microbatcher().submit(abi_version, items[0],
                                                &results[0]);
  }
  return cubbs_bbs_gpu_pairing_eq_batch_direct_v2(
      abi_version, items, item_count, results);
}

extern "C" int32_t cubbs_bbs_gpu_pairing_eq_aggregate_v2(
    uint32_t abi_version, const CuBbsBbsGpuPairingEqScaledItem *items,
    size_t item_count, CuBbsBbsGpuResult *result) {
  if (abi_version != CUBBS_BBS_GPU_ABI_VERSION ||
      (item_count > 0 && items == nullptr) || result == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  result->code = CUBBS_BBS_GPU_SUCCESS;
  result->value = 1;
  result->bytes_written = 0;
  if (item_count == 0) {
    return CUBBS_BBS_GPU_SUCCESS;
  }

  bool lhs_g2_common = true;
  bool rhs_g2_common = true;
  for (size_t i = 1; i < item_count; ++i) {
    lhs_g2_common = lhs_g2_common && same_g2_bytes(items[0].lhs_g2, items[i].lhs_g2);
    rhs_g2_common = rhs_g2_common && same_g2_bytes(items[0].rhs_g2, items[i].rhs_g2);
  }
  if (lhs_g2_common && rhs_g2_common) {
    const bool profile = env_flag_enabled("CUBBS_BBS_PAIRING_AGGREGATE_PROFILE");
    const auto started = profile ? Clock::now() : Clock::time_point{};
    std::vector<CuBbsBbsGpuByteSlice> lhs_points(item_count);
    std::vector<CuBbsBbsGpuByteSlice> rhs_points(item_count);
    std::vector<CuBbsBbsGpuByteSlice> scalars(item_count);
    for (size_t i = 0; i < item_count; ++i) {
      lhs_points[i] = items[i].lhs_g1;
      rhs_points[i] = items[i].rhs_g1;
      scalars[i] = items[i].coefficient;
    }
    const bool force_gpu = env_flag_enabled("CUBBS_BBS_FORCE_GPU");
    // A single randomized aggregate equation leaves the CUDA Miller loop with
    // only a handful of useful threads. Partition large FORCE_GPU batches into
    // independent aggregate lanes so wide devices can execute pairings in
    // parallel. AUTO retains one lane because it may choose the BLST fast path.
    const size_t configured_lanes = env_size_or_default(
        "CUBBS_BBS_GPU_PAIRING_AGGREGATE_LANES", force_gpu ? 32u : 1u);
    const size_t lane_count =
        force_gpu
            ? std::min(configured_lanes,
                       std::max<size_t>(1u, item_count / 128u))
            : 1u;
    if (lane_count > 1u) {
      std::vector<uint8_t> lane_lhs_sum(lane_count * 48u);
      std::vector<uint8_t> lane_rhs_sum(lane_count * 48u);
      const int32_t lane_msm_status =
          cubbs_bbs_cuda_g1_dual_msm_same_scalars_lanes(
              lhs_points.data(), rhs_points.data(), scalars.data(), item_count,
              lane_count, lane_lhs_sum.data(), lane_rhs_sum.data());
      if (lane_msm_status != CUBBS_BBS_GPU_SUCCESS) {
        result->code = lane_msm_status;
        result->value = 0u;
        return CUBBS_BBS_GPU_SUCCESS;
      }
      std::vector<CuBbsBbsGpuPairingEqItem> lane_pairings(lane_count);
      for (size_t lane = 0u; lane < lane_count; ++lane) {
        lane_pairings[lane] = CuBbsBbsGpuPairingEqItem{
            CuBbsBbsGpuByteSlice{lane_lhs_sum.data() + lane * 48u, 48u},
            items[0].lhs_g2,
            CuBbsBbsGpuByteSlice{lane_rhs_sum.data() + lane * 48u, 48u},
            items[0].rhs_g2};
      }
      std::vector<CuBbsBbsGpuResult> lane_pairing_results(lane_count);
      const int32_t lane_pairing_status =
          cubbs_bbs_gpu_pairing_eq_batch_v2(
              abi_version, lane_pairings.data(), lane_pairings.size(),
              lane_pairing_results.data());
      if (lane_pairing_status != CUBBS_BBS_GPU_SUCCESS) {
        result->code = lane_pairing_status;
        result->value = 0u;
        return CUBBS_BBS_GPU_SUCCESS;
      }
      for (const auto &lane_result : lane_pairing_results) {
        if (lane_result.code != CUBBS_BBS_GPU_SUCCESS) {
          result->code = lane_result.code;
          result->value = 0u;
          return CUBBS_BBS_GPU_SUCCESS;
        }
        if (lane_result.value != 1u) {
          result->value = 0u;
          return CUBBS_BBS_GPU_SUCCESS;
        }
      }
      result->value = 1u;
      return CUBBS_BBS_GPU_SUCCESS;
    }
    std::array<uint8_t, 48> lhs_sum_bytes{};
    std::array<uint8_t, 48> rhs_sum_bytes{};
    CuBbsG1MsmInput lhs_msm{lhs_points.data(), scalars.data(), item_count,
                             lhs_sum_bytes.data()};
    CuBbsG1MsmInput rhs_msm{rhs_points.data(), scalars.data(), item_count,
                             rhs_sum_bytes.data()};
    bool gpu_msm_ok = false;
    const bool use_dual_msm =
        std::getenv("CUBBS_BBS_PAIRING_AGGREGATE_DUAL_MSM") != nullptr
            ? env_flag_enabled("CUBBS_BBS_PAIRING_AGGREGATE_DUAL_MSM")
            : item_count >= 128u;
    if (use_dual_msm) {
      const int32_t dual_status = cubbs_bbs_cuda_g1_dual_msm_same_scalars(
          lhs_points.data(), rhs_points.data(), scalars.data(), item_count,
          lhs_sum_bytes.data(), rhs_sum_bytes.data());
      gpu_msm_ok = dual_status == CUBBS_BBS_GPU_SUCCESS;
    }
    if (!gpu_msm_ok) {
      std::array<CuBbsG1MsmInput, 2> msm_items{lhs_msm, rhs_msm};
      std::array<CuBbsBbsGpuResult, 2> msm_results{};
      const int32_t msm_status = cubbs_bbs_cuda_g1_msm_batch(
          msm_items.data(), msm_items.size(), msm_results.data());
      gpu_msm_ok = msm_status == CUBBS_BBS_GPU_SUCCESS &&
                   msm_results[0].code == CUBBS_BBS_GPU_SUCCESS &&
                   msm_results[1].code == CUBBS_BBS_GPU_SUCCESS;
      if (!gpu_msm_ok) {
        result->code = msm_status != CUBBS_BBS_GPU_SUCCESS
                           ? msm_status
                           : (msm_results[0].code != CUBBS_BBS_GPU_SUCCESS
                                  ? msm_results[0].code
                                  : msm_results[1].code);
        result->value = 0;
        return CUBBS_BBS_GPU_SUCCESS;
      }
    }
    const auto msm_finished = profile ? Clock::now() : Clock::time_point{};

    // Random-coefficient aggregation reduces the batch to one pairing
    // equation. AUTO mode intentionally lets BLST evaluate that single
    // equation because a two-thread CUDA launch cannot amortize its fixed
    // cost. FORCE_GPU is a strict execution-policy switch, however: it must
    // never report a GPU-only run while evaluating the reduced equation on
    // the host.
    if (env_flag_enabled("CUBBS_BBS_FORCE_GPU")) {
      const CuBbsBbsGpuPairingEqItem aggregate_item{
          CuBbsBbsGpuByteSlice{lhs_sum_bytes.data(), lhs_sum_bytes.size()},
          items[0].lhs_g2,
          CuBbsBbsGpuByteSlice{rhs_sum_bytes.data(), rhs_sum_bytes.size()},
          items[0].rhs_g2};
      CuBbsBbsGpuResult aggregate_result{};
      const bool microbatch_enabled =
          env_flag_default_enabled("CUBBS_BBS_GPU_PAIRING_MICROBATCH");
      const int32_t aggregate_status =
          microbatch_enabled
              ? cubbs_pairing_microbatcher().submit(
                    abi_version, aggregate_item, &aggregate_result)
              : cubbs_bbs_gpu_pairing_eq_batch_v2(
                    abi_version, &aggregate_item, 1u, &aggregate_result);
      if (aggregate_status != CUBBS_BBS_GPU_SUCCESS) {
        result->code = aggregate_status;
        result->value = 0u;
      } else {
        *result = aggregate_result;
      }
      return CUBBS_BBS_GPU_SUCCESS;
    }

    // AUTO mode evaluates the single reduced equation with BLST. Decode only
    // on that branch; FORCE_GPU validates the same inputs in the CUDA direct
    // path and should not pay for a redundant Host decode.
    blst_p1_affine lhs_g1{};
    blst_p1_affine rhs_g1{};
    blst_p2_affine lhs_g2{};
    blst_p2_affine rhs_g2{};
    if (blst_p1_uncompress(&lhs_g1, lhs_sum_bytes.data()) != BLST_SUCCESS ||
        blst_p1_uncompress(&rhs_g1, rhs_sum_bytes.data()) != BLST_SUCCESS ||
        !blst_p1_affine_in_g1(&lhs_g1) || !blst_p1_affine_in_g1(&rhs_g1) ||
        decode_g2_checked(items[0].lhs_g2, &lhs_g2) !=
            CUBBS_BBS_GPU_SUCCESS ||
        decode_g2_checked(items[0].rhs_g2, &rhs_g2) !=
            CUBBS_BBS_GPU_SUCCESS) {
      result->code = CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      result->value = 0u;
      return CUBBS_BBS_GPU_SUCCESS;
    }

    blst_fp12 lhs{};
    blst_fp12 rhs{};
    blst_miller_loop(&lhs, &lhs_g2, &lhs_g1);
    blst_miller_loop(&rhs, &rhs_g2, &rhs_g1);
    result->value = blst_fp12_finalverify(&lhs, &rhs) ? 1u : 0u;
    if (profile) {
      const auto finished = Clock::now();
      std::fprintf(stderr,
                   "cubbs_pairing_aggregate_profile items=%zu msm_ms=%.6f "
                   "pairing_ms=%.6f total_ms=%.6f\n",
                   item_count, elapsed_ms(started, msm_finished),
                   elapsed_ms(msm_finished, finished), elapsed_ms(started, finished));
    }
    return CUBBS_BBS_GPU_SUCCESS;
  }

  // Random-coefficient aggregation cannot collapse varying G2 inputs into two
  // pairings. Verify the equations individually through the CUDA Miller batch
  // instead; this is stronger than aggregate acceptance and avoids the old CPU
  // path's two G1 scalar multiplications plus two Miller loops per item.
  std::vector<CuBbsBbsGpuPairingEqItem> plain_items(item_count);
  for (size_t i = 0u; i < item_count; ++i) {
    plain_items[i] = CuBbsBbsGpuPairingEqItem{
        items[i].lhs_g1, items[i].lhs_g2, items[i].rhs_g1, items[i].rhs_g2};
  }
  std::vector<CuBbsBbsGpuResult> plain_results(item_count);
  const int32_t batch_status = cubbs_bbs_gpu_pairing_eq_batch_v2(
      abi_version, plain_items.data(), item_count, plain_results.data());
  if (batch_status != CUBBS_BBS_GPU_SUCCESS) {
    result->code = batch_status;
    result->value = 0u;
    return CUBBS_BBS_GPU_SUCCESS;
  }
  for (const auto &item_result : plain_results) {
    if (item_result.code != CUBBS_BBS_GPU_SUCCESS) {
      result->code = item_result.code;
      result->value = 0u;
      return CUBBS_BBS_GPU_SUCCESS;
    }
    if (item_result.value != 1u) {
      result->value = 0u;
      return CUBBS_BBS_GPU_SUCCESS;
    }
  }
  result->value = 1u;
  return CUBBS_BBS_GPU_SUCCESS;
}
