#include "cubbs_bbs_gpu_abi.h"

#include <blst.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using PointBytes = std::array<uint8_t, 48>;
using ScalarBytes = std::array<uint8_t, 32>;

struct Options {
  std::string layout{"common"};
  std::string route{"auto"};
  size_t term_count{18u};
  size_t batch_size{1024u};
  double duration_secs{6.0};
  size_t warmup{2u};
};

uint64_t splitmix64(uint64_t &state) {
  uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

ScalarBytes scalar_from_seed(uint64_t seed) {
  uint64_t state = seed;
  uint64_t limbs[4] = {
      splitmix64(state), splitmix64(state), splitmix64(state),
      splitmix64(state) & 0x3fffffffffffffffULL};
  limbs[0] |= 1u;
  blst_scalar scalar{};
  blst_scalar_from_uint64(&scalar, limbs);
  ScalarBytes out{};
  blst_bendian_from_scalar(out.data(), &scalar);
  return out;
}

ScalarBytes scalar_one() {
  const uint64_t limbs[4] = {1u, 0u, 0u, 0u};
  blst_scalar scalar{};
  blst_scalar_from_uint64(&scalar, limbs);
  ScalarBytes out{};
  blst_bendian_from_scalar(out.data(), &scalar);
  return out;
}

PointBytes point_multiple(uint64_t multiplier) {
  std::array<uint8_t, 32> scalar_le{};
  for (size_t byte = 0u; byte < sizeof(multiplier); ++byte) {
    scalar_le[byte] =
        static_cast<uint8_t>((multiplier >> (byte * 8u)) & 0xffu);
  }
  blst_p1 point{};
  blst_p1_mult(&point, blst_p1_generator(), scalar_le.data(), 64u);
  PointBytes out{};
  blst_p1_compress(out.data(), &point);
  return out;
}

size_t parse_size(const char *value, const char *name) {
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0u) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return static_cast<size_t>(parsed);
}

double parse_double(const char *value, const char *name) {
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || parsed <= 0.0) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return parsed;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto take = [&](const char *name) -> const char * {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++index];
    };
    if (arg == "--layout") {
      options.layout = take("--layout");
    } else if (arg == "--route") {
      options.route = take("--route");
    } else if (arg == "--terms") {
      options.term_count = parse_size(take("--terms"), "--terms");
    } else if (arg == "--batch") {
      options.batch_size = parse_size(take("--batch"), "--batch");
    } else if (arg == "--duration-secs") {
      options.duration_secs =
          parse_double(take("--duration-secs"), "--duration-secs");
    } else if (arg == "--warmup") {
      options.warmup = parse_size(take("--warmup"), "--warmup");
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: msm_route_bench --layout common|variable "
             "--route auto|1|2|3|4 --terms K --batch N "
             "--duration-secs S [--warmup N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.layout != "common" && options.layout != "variable") {
    throw std::runtime_error("layout must be common or variable");
  }
  if (options.route != "auto" && options.route != "1" &&
      options.route != "2" && options.route != "3" &&
      options.route != "4") {
    throw std::runtime_error("route must be auto or 1..4");
  }
  if (options.layout == "variable" &&
      (options.term_count < 1u || options.term_count > 3u)) {
    throw std::runtime_error("Route 1 variable benchmark supports K=1,2,3");
  }
  return options;
}

void select_route(const std::string &route) {
  if (route == "auto") {
    unsetenv("CUBBS_BBS_MSM_BENCH_ROUTE");
  } else {
    setenv("CUBBS_BBS_MSM_BENCH_ROUTE", route.c_str(), 1);
  }
}

uint8_t expected_route(const Options &options) {
  if (options.route != "auto") {
    return static_cast<uint8_t>(std::stoul(options.route));
  }
  if (options.layout == "variable") {
    return options.term_count == 3u && options.batch_size >= 128u
               ? CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM
               : CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL;
  }
  return options.term_count <= 258u
             ? CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS
             : CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE;
}

bool results_ok(const std::vector<CuBbsBbsGpuResult> &results,
                size_t count, uint8_t route) {
  return std::all_of(
      results.begin(), results.begin() + count,
      [route](const CuBbsBbsGpuResult &result) {
        return result.code == CUBBS_BBS_GPU_SUCCESS &&
               result.bytes_written == 48u && result.value == route;
      });
}

struct BenchResult {
  uint8_t observed_route{0u};
  size_t batches{0u};
  double elapsed_secs{0.0};
};

BenchResult run_common(const Options &options) {
  std::vector<PointBytes> points(options.term_count);
  std::vector<CuBbsBbsGpuByteSlice> point_slices(options.term_count);
  for (size_t term = 0u; term < options.term_count; ++term) {
    points[term] = point_multiple(term + 1u);
    point_slices[term] =
        CuBbsBbsGpuByteSlice{points[term].data(), points[term].size()};
  }

  const size_t scalar_count = options.batch_size * options.term_count;
  std::vector<uint8_t> scalar_rows(scalar_count * 32u);
  for (size_t index = 0u; index < scalar_count; ++index) {
    const auto scalar = scalar_from_seed(0x4f4d4e4942425300ULL + index);
    std::memcpy(scalar_rows.data() + index * 32u, scalar.data(), 32u);
  }
  std::vector<PointBytes> outputs(options.batch_size);
  std::vector<CuBbsBbsGpuByteBuffer> output_buffers(options.batch_size);
  std::vector<CuBbsBbsGpuResult> results(options.batch_size);
  for (size_t row = 0u; row < options.batch_size; ++row) {
    output_buffers[row] =
        CuBbsBbsGpuByteBuffer{outputs[row].data(), outputs[row].size(), 0u};
  }

  auto invoke = [&](size_t output_count) {
    CuBbsBbsGpuG1MsmCommonPointsBatchItem group{
        point_slices.data(), options.term_count, nullptr, scalar_rows.data(),
        options.term_count * 32u, output_count, output_buffers.data()};
    return cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
        CUBBS_BBS_GPU_ABI_VERSION, &group, 1u, results.data());
  };

  const size_t check_count = std::min<size_t>(2u, options.batch_size);
  select_route("6");
  if (invoke(check_count) != CUBBS_BBS_GPU_SUCCESS ||
      !results_ok(results, check_count,
                  CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS)) {
    throw std::runtime_error("BLST common-points correctness reference failed");
  }
  const std::vector<PointBytes> expected(outputs.begin(),
                                         outputs.begin() + check_count);

  select_route(options.route);
  const uint8_t wanted_route = expected_route(options);
  if (invoke(check_count) != CUBBS_BBS_GPU_SUCCESS ||
      !results_ok(results, check_count, wanted_route) ||
      !std::equal(expected.begin(), expected.end(), outputs.begin())) {
    throw std::runtime_error("common-points route correctness gate failed");
  }

  for (size_t iteration = 0u; iteration < options.warmup; ++iteration) {
    if (invoke(options.batch_size) != CUBBS_BBS_GPU_SUCCESS ||
        !results_ok(results, options.batch_size, wanted_route)) {
      throw std::runtime_error("common-points warmup failed");
    }
  }
  const auto started = Clock::now();
  size_t batches = 0u;
  double elapsed = 0.0;
  do {
    if (invoke(options.batch_size) != CUBBS_BBS_GPU_SUCCESS ||
        !results_ok(results, options.batch_size, wanted_route)) {
      throw std::runtime_error("common-points timed batch failed");
    }
    ++batches;
    elapsed = std::chrono::duration<double>(Clock::now() - started).count();
  } while (elapsed < options.duration_secs);
  return BenchResult{wanted_route, batches, elapsed};
}

BenchResult run_variable(const Options &options) {
  const size_t point_count = options.batch_size * options.term_count;
  std::vector<PointBytes> points(point_count);
  std::vector<CuBbsBbsGpuByteSlice> point_slices(point_count);
  std::vector<ScalarBytes> scalars(point_count);
  std::vector<CuBbsBbsGpuByteSlice> scalar_slices(point_count);
  for (size_t item = 0u; item < options.batch_size; ++item) {
    for (size_t term = 0u; term < options.term_count; ++term) {
      const size_t index = item * options.term_count + term;
      points[index] = point_multiple(index + 1u);
      point_slices[index] =
          CuBbsBbsGpuByteSlice{points[index].data(), points[index].size()};
      scalars[index] =
          options.term_count == 2u && term == 0u
              ? scalar_one()
              : scalar_from_seed(0x5641524941424c45ULL + index);
      scalar_slices[index] =
          CuBbsBbsGpuByteSlice{scalars[index].data(), scalars[index].size()};
    }
  }
  std::vector<PointBytes> outputs(options.batch_size);
  std::vector<CuBbsBbsGpuG1MsmItem> items(options.batch_size);
  std::vector<CuBbsBbsGpuResult> results(options.batch_size);
  for (size_t item = 0u; item < options.batch_size; ++item) {
    items[item] = CuBbsBbsGpuG1MsmItem{
        point_slices.data() + item * options.term_count,
        scalar_slices.data() + item * options.term_count, options.term_count,
        CuBbsBbsGpuByteBuffer{outputs[item].data(), outputs[item].size(), 0u}};
  }
  auto invoke = [&](size_t item_count) {
    return cubbs_bbs_gpu_g1_msm_batch_v2(
        CUBBS_BBS_GPU_ABI_VERSION, items.data(), item_count, results.data());
  };

  const size_t check_count = std::min<size_t>(8u, options.batch_size);
  select_route("5");
  if (invoke(check_count) != CUBBS_BBS_GPU_SUCCESS ||
      !results_ok(results, check_count, CUBBS_BBS_MSM_ROUTE_BLST_ITEM)) {
    throw std::runtime_error("BLST variable-point correctness reference failed");
  }
  const std::vector<PointBytes> expected(outputs.begin(),
                                         outputs.begin() + check_count);

  select_route(options.route);
  const uint8_t wanted_route = expected_route(options);
  // Automatic Route 1 selection depends on the submitted batch size. Check
  // the full batch so this gate observes the same route as the timed calls,
  // while comparing the sampled outputs already computed by BLST above.
  const size_t route_check_count =
      options.route == "auto" ? options.batch_size : check_count;
  if (invoke(route_check_count) != CUBBS_BBS_GPU_SUCCESS ||
      !results_ok(results, route_check_count, wanted_route) ||
      !std::equal(expected.begin(), expected.end(), outputs.begin())) {
    throw std::runtime_error("variable-point route correctness gate failed");
  }

  for (size_t iteration = 0u; iteration < options.warmup; ++iteration) {
    if (invoke(options.batch_size) != CUBBS_BBS_GPU_SUCCESS ||
        !results_ok(results, options.batch_size, wanted_route)) {
      throw std::runtime_error("variable-point warmup failed");
    }
  }
  const auto started = Clock::now();
  size_t batches = 0u;
  double elapsed = 0.0;
  do {
    if (invoke(options.batch_size) != CUBBS_BBS_GPU_SUCCESS ||
        !results_ok(results, options.batch_size, wanted_route)) {
      throw std::runtime_error("variable-point timed batch failed");
    }
    ++batches;
    elapsed = std::chrono::duration<double>(Clock::now() - started).count();
  } while (elapsed < options.duration_secs);
  return BenchResult{wanted_route, batches, elapsed};
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const BenchResult result =
        options.layout == "common" ? run_common(options) : run_variable(options);
    const double rows =
        static_cast<double>(result.batches * options.batch_size);
    const double rows_per_sec = rows / result.elapsed_secs;
    std::cout << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"tool\": \"msm-route-bench\",\n"
              << "  \"abiVersion\": " << CUBBS_BBS_GPU_ABI_VERSION << ",\n"
              << "  \"layout\": \"" << options.layout << "\",\n"
              << "  \"routeRequested\": \"" << options.route << "\",\n"
              << "  \"routeObserved\": "
              << static_cast<unsigned>(result.observed_route) << ",\n"
              << "  \"termCount\": " << options.term_count << ",\n"
              << "  \"batchSize\": " << options.batch_size << ",\n"
              << "  \"durationRequestedSecs\": " << options.duration_secs
              << ",\n"
              << "  \"elapsedSecs\": " << result.elapsed_secs << ",\n"
              << "  \"batches\": " << result.batches << ",\n"
              << "  \"rows\": " << static_cast<uint64_t>(rows) << ",\n"
              << "  \"rowsPerSec\": " << rows_per_sec << ",\n"
              << "  \"correctnessPassed\": true,\n"
              << "  \"failed\": 0\n"
              << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "msm_route_bench: " << error.what() << "\n";
    return 1;
  }
}
