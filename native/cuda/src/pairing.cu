#include "prepare_b_msm.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)
#define FEATURE_BLS12_381
#include <ff/bls12-381-fp2.hpp>

static size_t env_size_or_default(const char *name, size_t default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || parsed == 0ul) return default_value;
  return static_cast<size_t>(parsed);
}

static bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "no") != 0 &&
         std::strcmp(value, "NO") != 0;
}

#include "kernels/pairing.cuh"
#endif
