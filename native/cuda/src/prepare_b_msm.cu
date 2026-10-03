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
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Minimal compatible declarations derived from BLST bindings/blst.h and its
// exported low-level interface (Copyright Supranational LLC, Apache-2.0; see
// NOTICE and THIRD_PARTY.md). The full header is not included because it
// redeclares limb_t, which conflicts with the sppark vect.h header pulled in
// below. Only blst_p1_affine's layout {blst_fp x, y} (48 bytes each, 6x
// uint64 Montgomery limbs) and the required function declarations are kept.
// fp_t and blst_fp share the same 48-byte Montgomery limb layout on this
// little-endian target, so a bitwise copy between them is correct.
struct blst_fp { uint64_t l[384 / 8 / sizeof(uint64_t)]; };
struct blst_p1_affine { blst_fp x, y; };
extern "C" void blst_p1_affine_compress(uint8_t out[48],
                                        const blst_p1_affine *in);
extern "C" uint64_t div_3_limbs(const uint64_t dividend_top[2],
                                uint64_t divisor_lo, uint64_t divisor_hi);
extern "C" uint64_t quot_rem_128(uint64_t *quot_rem,
                                 const uint64_t divisor[2],
                                 uint64_t quotient);

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
#define FEATURE_BLS12_381
#include <ff/bls12-381-fp2.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> cubbs_sppark_point_t;
typedef xyzz_t<fp_t> cubbs_sppark_bucket_t;
typedef cubbs_sppark_bucket_t::affine_inf_t cubbs_sppark_affine_t;
typedef fr_t cubbs_sppark_scalar_t;

#if defined(CUBBS_BBS_ENABLE_SPPARK_REFERENCE)
#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>
#endif

static int32_t cubbs_bbs_cuda_g1_variable_base_single_term_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results);
static int32_t cubbs_bbs_cuda_g1_variable_base_generic_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results);
#endif

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

[[maybe_unused]] static constexpr uint8_t CUBBS_BBS_MSM_ROUTE_FUSED_ITEM =
    CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM;
[[maybe_unused]] static constexpr uint8_t
    CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS =
        CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS;
static constexpr uint8_t CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC =
    CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL;

#if defined(CUBBS_BBS_ENABLE_SPPARK_REFERENCE)
static int32_t compute_g1_msm_with_sppark_reference(
    const CuBbsG1MsmInput &item) {
  std::vector<uint8_t> decoded_points(
      item.term_count * sizeof(cubbs_sppark_bucket_t::affine_t));
  std::vector<uint8_t> decoded_scalars(
      item.term_count * sizeof(cubbs_sppark_scalar_t));
  int32_t status = cubbs_bbs_blst_decode_g1_msm_terms(
      &item, decoded_points.data(),
      sizeof(cubbs_sppark_bucket_t::affine_t), decoded_scalars.data(),
      sizeof(cubbs_sppark_scalar_t));
  if (status != CUBBS_BBS_GPU_SUCCESS) return status;

  cubbs_sppark_point_t out{};
  const RustError err = mult_pippenger<cubbs_sppark_bucket_t>(
      &out,
      reinterpret_cast<const cubbs_sppark_bucket_t::affine_t *>(
          decoded_points.data()),
      item.term_count,
      reinterpret_cast<const cubbs_sppark_scalar_t *>(decoded_scalars.data()),
      false);
  if (err.code != 0) return CUBBS_BBS_GPU_ERR_INTERNAL;
  return cubbs_bbs_blst_compress_g1_projective(
      &out, item.compressed_point_out);
}

static int32_t compute_g1_msm_sppark_reference_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results, uint8_t route) {
  for (size_t item = 0u; item < item_count; ++item) {
    const int32_t status = compute_g1_msm_with_sppark_reference(items[item]);
    if (status == CUBBS_BBS_GPU_SUCCESS) {
      set_result_route(results, item, status, 48u, route);
    } else {
      set_result(results, item, status, 0u);
    }
  }
  return CUBBS_BBS_GPU_SUCCESS;
}
#endif

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

static uint8_t explicit_msm_benchmark_route() {
  const char *value = std::getenv("CUBBS_BBS_MSM_BENCH_ROUTE");
  if (value == nullptr || value[0] == '\0') return 0u;
  char *end = nullptr;
  const unsigned long route = std::strtoul(value, &end, 10);
  // Benchmark-only override; production leaves it unset and auto-selects.
  if (end == value || *end != '\0' || route < 1u || route > 8u) {
    return 0u;
  }
  return static_cast<uint8_t>(route);
}

static size_t env_size_or_default(const char *name, size_t default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

// Decode and compression used to create a fresh set of std::threads for every
// shape bucket. Under throughput loads, outer streams * shape buckets * inner
// workers could briefly create hundreds of runnable threads. Keep a bounded
// process-wide pool instead so concurrent BBS batches share the host helpers.
static thread_local bool cubbs_inside_host_pool_worker = false;

class CuBbsHostTaskPool {
 public:
  CuBbsHostTaskPool() {
    const unsigned int hardware = std::thread::hardware_concurrency();
    const size_t default_workers = hardware == 0u
                                       ? 8u
                                       : std::min<size_t>(32u, hardware);
    const size_t worker_count = std::max<size_t>(
        1u, env_size_or_default("CUBBS_BBS_HOST_POOL_WORKERS", default_workers));
    workers_.reserve(worker_count);
    for (size_t worker = 0u; worker < worker_count; ++worker) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~CuBbsHostTaskPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_) worker.join();
  }

  void run(size_t task_count, const std::function<void(size_t)> &task) {
    if (task_count == 0u) return;
    // A helper may enter another decode/compression primitive. Running nested
    // shards inline prevents every pool worker from waiting for work that can
    // no longer be scheduled.
    if (task_count == 1u || cubbs_inside_host_pool_worker) {
      for (size_t index = 0u; index < task_count; ++index) task(index);
      return;
    }
    struct Group {
      explicit Group(size_t count) : remaining(count) {}
      std::mutex mutex;
      std::condition_variable done;
      size_t remaining;
    };
    // The submitting thread already owns a CPU and would otherwise sleep
    // while every helper is queued. Execute one shard inline and enqueue only
    // the remaining shards, reducing wakeups and critical-path queue latency.
    Group group(task_count - 1u);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t index = 1u; index < task_count; ++index) {
        tasks_.emplace_back([&group, &task, index]() {
          task(index);
          std::lock_guard<std::mutex> lock(group.mutex);
          if (--group.remaining == 0u) group.done.notify_one();
        });
      }
    }
    ready_.notify_all();
    cubbs_inside_host_pool_worker = true;
    task(0u);
    cubbs_inside_host_pool_worker = false;
    std::unique_lock<std::mutex> lock(group.mutex);
    group.done.wait(lock, [&]() { return group.remaining == 0u; });
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      cubbs_inside_host_pool_worker = true;
      task();
      cubbs_inside_host_pool_worker = false;
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};

static CuBbsHostTaskPool &cubbs_host_task_pool() {
  static CuBbsHostTaskPool pool;
  return pool;
}

void cubbs_bbs_run_host_tasks(
    size_t task_count, const std::function<void(size_t)> &task) {
  if (env_flag_default_enabled("CUBBS_BBS_HOST_POOL")) {
    cubbs_host_task_pool().run(task_count, task);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(task_count);
  for (size_t index = 0u; index < task_count; ++index) {
    workers.emplace_back([&, index]() { task(index); });
  }
  for (auto &worker : workers) worker.join();
}

static void cubbs_run_host_tasks(
    size_t task_count, const std::function<void(size_t)> &task) {
  cubbs_bbs_run_host_tasks(task_count, task);
}

// cudaMallocAsync uses the device's default pool. With the default zero
// release threshold, every stream synchronization can return cached pages to
// the driver, so the next BBS stage pays the mapping cost again. Retain a
// bounded amount across calls. 64 MiB covers the hot buffers of ordinary
// batches without pressuring the many transient groups in huge mixed batches.
static void configure_cubbs_async_mempool() {
  static std::once_flag configured;
  std::call_once(configured, []() {
    int device = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess ||
        pool == nullptr) {
      return;
    }
    uint64_t threshold = static_cast<uint64_t>(env_size_or_default(
        "CUBBS_BBS_GPU_MEMPOOL_RELEASE_THRESHOLD_BYTES",
        64u * 1024u * 1024u));
    (void)cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  });
}

static bool bbs_homogeneous_kernel_enabled() {
  return env_flag_default_enabled("CUBBS_BBS_BBS_HOMOGENEOUS_KERNEL");
}

static bool bbs_homogeneous_comb_enabled() {
  return env_flag_default_enabled("CUBBS_BBS_BBS_HOMOGENEOUS_COMB");
}

static bool bbs_homogeneous_affine_comb_enabled() {
  return env_flag_default_enabled("CUBBS_BBS_BBS_AFFINE_COMB");
}

static size_t bbs_homogeneous_comb_bits(size_t term_count) {
  const char *configured_env =
      std::getenv("CUBBS_BBS_BBS_HOMOGENEOUS_COMB_BITS");
  const size_t shape_default = term_count >= 18u && term_count <= 66u
      ? 10u
      : (term_count >= 67u && term_count <= 160u ? 9u : 8u);
  const size_t configured = configured_env == nullptr
      ? shape_default
      : env_size_or_default("CUBBS_BBS_BBS_HOMOGENEOUS_COMB_BITS",
                            shape_default);
  return configured == 4u || configured == 6u || configured == 8u ||
                 configured == 9u || configured == 10u
             ? configured
             : 5u;
}

static size_t bbs_specialized_max_terms() {
  // 258 terms covers a 256-message BBS shape plus the two fixed B terms.
  return env_size_or_default(
      "CUBBS_BBS_BBS_SPECIALIZED_MAX_TERMS", 258u);
}

static size_t bbs_specialized_cache_budget_bytes() {
  return env_size_or_default(
      "CUBBS_BBS_BBS_SPECIALIZED_CACHE_BUDGET_BYTES", 256u * 1024u * 1024u);
}

static bool bbs_large_shape_kernel_enabled() {
  return env_flag_default_enabled("CUBBS_BBS_GPU_LARGE_SHAPE_KERNEL");
}

static size_t bbs_large_shape_max_terms() {
  // 4098 terms covers 4096 BBS messages plus the two fixed B terms.
  return env_size_or_default("CUBBS_BBS_GPU_LARGE_SHAPE_MAX_TERMS", 4098u);
}

static size_t bbs_large_shape_tile_terms() {
  const size_t configured =
      env_size_or_default("CUBBS_BBS_GPU_LARGE_SHAPE_TILE_TERMS", 128u);
  return configured == 64u || configured == 128u || configured == 256u
             ? configured
             : 128u;
}

static bool bbs_large_shape_pippenger_enabled() {
  return env_flag_enabled("CUBBS_BBS_GPU_LARGE_SHAPE_PIPPENGER");
}

static bool bbs_large_shape_window8_enabled() {
  return env_flag_enabled("CUBBS_BBS_GPU_LARGE_SHAPE_WINDOW8");
}

static bool bbs_large_shape_glv_enabled() {
  return env_flag_default_enabled("CUBBS_BBS_GPU_LARGE_SHAPE_GLV");
}

static bool bbs_large_shape_glv_wnaf_enabled() {
  return env_flag_enabled("CUBBS_BBS_GPU_LARGE_SHAPE_GLV_WNAF");
}

static bool bbs_universal_tiled_kernel_enabled() {
  return env_flag_default_enabled("CUBBS_BBS_GPU_UNIVERSAL_TILED_KERNEL");
}

static size_t bbs_universal_tiled_cache_budget_bytes() {
  // The 65,538-term BBS common-point set needs about 607 MiB with both the
  // window-4 and GLV window-4 tables enabled. Keep enough headroom for that
  // publication shape without letting an arbitrary Route 4 request build an
  // unbounded precomputation table. Larger shapes retain the resource-bounded
  // universal fallback below this optimized tier.
  return env_size_or_default(
      "CUBBS_BBS_GPU_UNIVERSAL_TILED_CACHE_BUDGET_BYTES",
      768u * 1024u * 1024u);
}

static size_t bbs_universal_item_tile_terms() {
  const size_t configured = env_size_or_default(
      "CUBBS_BBS_GPU_UNIVERSAL_ITEM_TILE_TERMS", 128u);
  return configured == 64u || configured == 128u || configured == 256u
             ? configured
             : 128u;
}

static size_t bbs_universal_item_tiled_min_terms() {
  return env_size_or_default(
      "CUBBS_BBS_GPU_UNIVERSAL_ITEM_TILED_MIN_TERMS", 512u);
}

static size_t bbs_universal_item_partial_budget_bytes() {
  return env_size_or_default(
      "CUBBS_BBS_GPU_UNIVERSAL_ITEM_PARTIAL_BUDGET_BYTES",
      256u * 1024u * 1024u);
}

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
static bool bbs_specialized_common_points_supported(size_t term_count) {
  if (term_count > bbs_specialized_max_terms()) return false;

  const size_t comb_bits = bbs_homogeneous_comb_bits(term_count);
  size_t comb_entries = 0u;
  if (bbs_homogeneous_comb_enabled()) {
    if (comb_bits == 4u) comb_entries = 64u * 16u;
    else if (comb_bits == 5u) comb_entries = 52u * 17u;
    else if (comb_bits == 6u) comb_entries = 43u * 33u;
    else if (comb_bits == 8u) comb_entries = 32u * 129u;
    else if (comb_bits == 9u) comb_entries = 29u * 257u;
    else comb_entries = 26u * 513u;
  }
  const size_t bucket_entries = 16u + 256u + comb_entries;
  const size_t fixed_bytes =
      sizeof(cubbs_sppark_bucket_t::affine_t);
  if (bucket_entries >
      (static_cast<size_t>(-1) - fixed_bytes) /
          sizeof(cubbs_sppark_bucket_t)) {
    return false;
  }
  const size_t bytes_per_term =
      fixed_bytes + bucket_entries * sizeof(cubbs_sppark_bucket_t);
  const size_t budget = bbs_specialized_cache_budget_bytes();
  return bytes_per_term == 0u || term_count <= budget / bytes_per_term;
}

static bool bbs_large_shape_kernel_supported(size_t term_count) {
  return bbs_large_shape_kernel_enabled() &&
         !bbs_specialized_common_points_supported(term_count) &&
         term_count <= bbs_large_shape_max_terms();
}

static bool bbs_universal_tiled_common_supported(size_t term_count) {
  if (!bbs_universal_tiled_kernel_enabled() || term_count == 0u ||
      term_count > static_cast<size_t>(UINT32_MAX)) {
    return false;
  }
  const size_t tile_terms = bbs_large_shape_tile_terms();
  const size_t tile_count = (term_count + tile_terms - 1u) / tile_terms;
  // The tiled common-point kernels use grid.y for the term tile.
  if (tile_count == 0u || tile_count > 65535u) return false;

  size_t entries_per_term = 0u;
  const auto add_entries = [&](size_t entries) {
    if (entries_per_term > static_cast<size_t>(-1) - entries) return false;
    entries_per_term += entries;
    return true;
  };
  // The current large-table builder always retains affine points and the
  // ordinary window-4 table. Optional tables are counted exactly as built.
  if (!add_entries(16u)) return false;
  if (bbs_large_shape_glv_enabled() && !add_entries(32u)) return false;
  if (bbs_large_shape_glv_enabled() && bbs_large_shape_glv_wnaf_enabled() &&
      !add_entries(128u)) {
    return false;
  }
  if (bbs_large_shape_window8_enabled() && !add_entries(256u)) return false;

  if (entries_per_term >
      (static_cast<size_t>(-1) -
       sizeof(cubbs_sppark_bucket_t::affine_t)) /
          sizeof(cubbs_sppark_bucket_t)) {
    return false;
  }
  const size_t bytes_per_term = sizeof(cubbs_sppark_bucket_t::affine_t) +
      entries_per_term * sizeof(cubbs_sppark_bucket_t);
  const size_t budget = bbs_universal_tiled_cache_budget_bytes();
  return bytes_per_term > 0u && term_count <= budget / bytes_per_term;
}
#else
static bool bbs_large_shape_kernel_supported(size_t) { return false; }
static bool bbs_universal_tiled_common_supported(size_t) { return false; }
#endif

static size_t bbs_homogeneous_kernel_min_items(size_t /*term_count*/) {
  const size_t shape_default = 1u;
  return env_size_or_default("CUBBS_BBS_BBS_HOMOGENEOUS_KERNEL_MIN_ITEMS",
                             shape_default);
}

static size_t blst_msm_worker_count(size_t item_count) {
  const unsigned int hardware = std::thread::hardware_concurrency();
  const size_t default_workers =
      hardware == 0u ? 8u : std::min<size_t>(32u, static_cast<size_t>(hardware));
  const size_t configured =
      env_size_or_default("CUBBS_BBS_BLST_MSM_WORKERS", default_workers);
  if (configured == 0u) {
    return std::max<size_t>(1u, std::min(item_count, default_workers));
  }
  return std::max<size_t>(1u, std::min(item_count, configured));
}

static size_t fused_compress_worker_count(size_t item_count) {
  const char *configured_env = std::getenv("CUBBS_BBS_FUSED_COMPRESS_WORKERS");
  if (configured_env == nullptr || configured_env[0] == '\0' ||
      item_count < env_size_or_default("CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS",
                                       64u)) {
    return 1u;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  const size_t default_workers =
      hardware == 0u ? 8u : std::min<size_t>(8u, static_cast<size_t>(hardware));
  const size_t configured =
      env_size_or_default("CUBBS_BBS_FUSED_COMPRESS_WORKERS", default_workers);
  return std::max<size_t>(1u, std::min(item_count, configured));
}

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
using cubbs_sppark_plain_affine_t = cubbs_sppark_bucket_t::affine_t;

static bool g1_msm_items_have_common_points(const CuBbsG1MsmInput *items,
                                            size_t item_count) {
  if (item_count < 2u) return false;
  const size_t term_count = items[0].term_count;
  for (size_t item = 1u; item < item_count; ++item) {
    if (items[item].term_count != term_count) return false;
  }
  for (size_t term = 0u; term < term_count; ++term) {
    const auto &base = items[0].points[term];
    for (size_t item = 1u; item < item_count; ++item) {
      const auto &candidate = items[item].points[term];
      if (candidate.len != base.len ||
          std::memcmp(candidate.data, base.data, base.len) != 0) {
        return false;
      }
    }
  }
  return true;
}
#endif

// MSM implementation modules are included in dependency order. Pairing is a
// separate CUDA translation unit because it has an independent workspace,
// stream policy, and optimization lifecycle.
#include "kernels/common_base.cuh"
#include "kernels/large_msm.cuh"
#include "kernels/comb_msm.cuh"
#include "kernels/comb_precompute.cuh"
#include "kernels/affine_msm.cuh"
#include "kernels/common_cache.cuh"
#include "kernels/common_reference.cuh"
#include "kernels/scalar_decode.cuh"
#include "kernels/common_universal.cuh"
#include "kernels/common_fused_runtime.cuh"
#include "kernels/msm_dispatch.cuh"
#include "kernels/variable_base_universal.cuh"
#include "kernels/variable_base_specialized.cuh"
#include "kernels/proof_verify.cuh"
#include "kernels/proof_gen.cuh"
#include "kernels/aggregate_dual_msm.cuh"
#include "kernels/disabled_backend_stubs.cuh"
