#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

// Build the fixed-base comb table on the GPU. Host-side construction with
// sppark field objects is prohibitively expensive because those types are
// tuned for device arithmetic. One block owns one BBS generator; its 16
// threads materialize all 4-bit digits while thread 0 advances the generator
// by four doublings for the next scalar window.
__global__ static void cubbs_bbs_g1_fixed_comb4_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *comb_table) {
  const size_t term = blockIdx.x;
  const uint32_t digit = threadIdx.x;
  if (term >= term_count || digit >= 16u) {
    return;
  }

  __shared__ __align__(16) uint8_t power_storage[4u * sizeof(cubbs_sppark_bucket_t)];
  auto *powers = reinterpret_cast<cubbs_sppark_bucket_t *>(power_storage);
  if (digit == 0u) {
    powers[0] = points[term];
  }
  __syncthreads();

  for (uint32_t window = 0u; window < 64u; ++window) {
    if (digit == 0u) {
      powers[1] = powers[0];
      powers[1].add(powers[1]);
      powers[2] = powers[1];
      powers[2].add(powers[2]);
      powers[3] = powers[2];
      powers[3].add(powers[3]);
    }
    __syncthreads();

    cubbs_sppark_bucket_t multiple;
    multiple.inf();
#pragma unroll
    for (uint32_t bit = 0u; bit < 4u; ++bit) {
      if ((digit & (1u << bit)) != 0u) {
        if (multiple.is_inf()) {
          multiple = powers[bit];
        } else {
          multiple.add(powers[bit]);
        }
      }
    }
    comb_table[((term * 64u + window) * 16u) + digit] = multiple;
    __syncthreads();
    if (digit == 0u) {
      powers[0] = powers[3];
      powers[0].add(powers[0]);
    }
    __syncthreads();
  }
}

__global__ static void cubbs_bbs_g1_fixed_signed_comb5_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *comb_table) {
  const size_t term = blockIdx.x;
  const uint32_t digit = threadIdx.x;
  if (term >= term_count || digit >= 17u) {
    return;
  }

  __shared__ __align__(16) uint8_t
      power_storage[5u * sizeof(cubbs_sppark_bucket_t)];
  auto *powers = reinterpret_cast<cubbs_sppark_bucket_t *>(power_storage);
  if (digit == 0u) {
    powers[0] = points[term];
  }
  __syncthreads();

  for (uint32_t window = 0u; window < 52u; ++window) {
    if (digit == 0u) {
      powers[1] = powers[0];
      powers[1].add(powers[1]);
      powers[2] = powers[1];
      powers[2].add(powers[2]);
      powers[3] = powers[2];
      powers[3].add(powers[3]);
      powers[4] = powers[3];
      powers[4].add(powers[4]);
    }
    __syncthreads();

    cubbs_sppark_bucket_t multiple;
    multiple.inf();
#pragma unroll
    for (uint32_t bit = 0u; bit < 5u; ++bit) {
      if ((digit & (1u << bit)) != 0u) {
        if (multiple.is_inf()) {
          multiple = powers[bit];
        } else {
          multiple.add(powers[bit]);
        }
      }
    }
    comb_table[((term * 52u + window) * 17u) + digit] = multiple;
    __syncthreads();
    if (digit == 0u) {
      powers[0] = powers[4];
      powers[0].add(powers[0]);
    }
    __syncthreads();
  }
}

__global__ static void cubbs_bbs_g1_fixed_signed_comb6_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *comb_table) {
  const size_t term = blockIdx.x;
  const uint32_t digit = threadIdx.x;
  if (term >= term_count || digit >= 33u) return;

  __shared__ __align__(16) uint8_t
      power_storage[6u * sizeof(cubbs_sppark_bucket_t)];
  auto *powers = reinterpret_cast<cubbs_sppark_bucket_t *>(power_storage);
  if (digit == 0u) powers[0] = points[term];
  __syncthreads();
  for (uint32_t window = 0u; window < 43u; ++window) {
    if (digit == 0u) {
#pragma unroll
      for (uint32_t bit = 1u; bit < 6u; ++bit) {
        powers[bit] = powers[bit - 1u];
        powers[bit].add(powers[bit]);
      }
    }
    __syncthreads();
    cubbs_sppark_bucket_t multiple;
    multiple.inf();
#pragma unroll
    for (uint32_t bit = 0u; bit < 6u; ++bit) {
      if ((digit & (1u << bit)) != 0u) {
        if (multiple.is_inf()) multiple = powers[bit];
        else multiple.add(powers[bit]);
      }
    }
    comb_table[((term * 43u + window) * 33u) + digit] = multiple;
    __syncthreads();
    if (digit == 0u) {
      powers[0] = powers[5];
      powers[0].add(powers[0]);
    }
    __syncthreads();
  }
}

__global__ static void cubbs_bbs_g1_fixed_signed_comb8_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *comb_table) {
  const size_t term = blockIdx.x;
  const uint32_t digit = threadIdx.x;
  if (term >= term_count || digit >= 129u) return;
  __shared__ __align__(16) uint8_t
      power_storage[8u * sizeof(cubbs_sppark_bucket_t)];
  auto *powers = reinterpret_cast<cubbs_sppark_bucket_t *>(power_storage);
  if (digit == 0u) powers[0] = points[term];
  __syncthreads();
  for (uint32_t window = 0u; window < 32u; ++window) {
    if (digit == 0u) {
#pragma unroll
      for (uint32_t bit = 1u; bit < 8u; ++bit) {
        powers[bit] = powers[bit - 1u];
        powers[bit].add(powers[bit]);
      }
    }
    __syncthreads();
    cubbs_sppark_bucket_t multiple;
    multiple.inf();
#pragma unroll
    for (uint32_t bit = 0u; bit < 8u; ++bit) {
      if ((digit & (1u << bit)) != 0u) {
        if (multiple.is_inf()) multiple = powers[bit];
        else multiple.add(powers[bit]);
      }
    }
    comb_table[((term * 32u + window) * 129u) + digit] = multiple;
    __syncthreads();
    if (digit == 0u) {
      powers[0] = powers[7];
      powers[0].add(powers[0]);
    }
    __syncthreads();
  }
}

__global__ static void cubbs_bbs_g1_fixed_signed_comb10_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *comb_table) {
  const size_t term = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (term >= term_count) return;
  __shared__ __align__(16) uint8_t
      power_storage[10u * sizeof(cubbs_sppark_bucket_t)];
  auto *powers = reinterpret_cast<cubbs_sppark_bucket_t *>(power_storage);
  if (tid == 0u) powers[0] = points[term];
  __syncthreads();
  for (uint32_t window = 0u; window < 26u; ++window) {
    if (tid == 0u) {
#pragma unroll
      for (uint32_t bit = 1u; bit < 10u; ++bit) {
        powers[bit] = powers[bit - 1u];
        powers[bit].add(powers[bit]);
      }
    }
    __syncthreads();
    for (uint32_t digit = tid; digit < 513u; digit += blockDim.x) {
      cubbs_sppark_bucket_t multiple;
      multiple.inf();
#pragma unroll
      for (uint32_t bit = 0u; bit < 10u; ++bit) {
        if ((digit & (1u << bit)) != 0u) {
          if (multiple.is_inf()) multiple = powers[bit];
          else multiple.add(powers[bit]);
        }
      }
      comb_table[((term * 26u + window) * 513u) + digit] = multiple;
    }
    __syncthreads();
    if (tid == 0u) {
      powers[0] = powers[9];
      powers[0].add(powers[0]);
    }
    __syncthreads();
  }
}

__global__ static void cubbs_bbs_g1_fixed_signed_comb9_precompute_kernel(
    const cubbs_sppark_plain_affine_t *points, size_t term_count,
    cubbs_sppark_bucket_t *comb_table) {
  const size_t term = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (term >= term_count) return;
  __shared__ __align__(16) uint8_t
      power_storage[9u * sizeof(cubbs_sppark_bucket_t)];
  auto *powers = reinterpret_cast<cubbs_sppark_bucket_t *>(power_storage);
  if (tid == 0u) powers[0] = points[term];
  __syncthreads();
  for (uint32_t window = 0u; window < 29u; ++window) {
    if (tid == 0u) {
#pragma unroll
      for (uint32_t bit = 1u; bit < 9u; ++bit) {
        powers[bit] = powers[bit - 1u];
        powers[bit].add(powers[bit]);
      }
    }
    __syncthreads();
    for (uint32_t digit = tid; digit < 257u; digit += blockDim.x) {
      cubbs_sppark_bucket_t multiple;
      multiple.inf();
#pragma unroll
      for (uint32_t bit = 0u; bit < 9u; ++bit) {
        if ((digit & (1u << bit)) != 0u) {
          if (multiple.is_inf()) multiple = powers[bit];
          else multiple.add(powers[bit]);
        }
      }
      comb_table[((term * 29u + window) * 257u) + digit] = multiple;
    }
    __syncthreads();
    if (tid == 0u) {
      powers[0] = powers[8];
      powers[0].add(powers[0]);
    }
    __syncthreads();
  }
}

static uint32_t fused_common_points_threads_per_row() {
  // Default 128 threads/row: ~1 term/thread for the largest mixed VC shape
  // (130 terms), and it roughly doubles active warps per SM vs the 32-thread
  // scaffold, lifting occupancy. Overridable via env for sweeps.
  const size_t configured = env_size_or_default(
      "CUBBS_BBS_GPU_FUSED_COMMON_POINTS_THREADS", 128u);
  if (configured <= 32u) {
    return 32u;
  }
  if (configured <= 64u) {
    return 64u;
  }
  if (configured >= 256u) {
    return 256u;
  }
  return 128u;
}

// Adaptive thread count: pick the smallest power-of-two >= term_count (so
// every thread owns at least one term and no warp is mostly idle), clamped to
// [32, 128]. A full warp (32) is the floor for good scheduling; 128 is the
// occupancy sweet spot for the largest BBS shapes. This matters most for the
// small proof-T2 common shapes (8 terms): a fixed 128 wastes 7/8 of the
// threads there, while 32 gives ~1 term/thread and triples resident blocks
// per SM. Overridable by CUBBS_BBS_GPU_FUSED_COMMON_POINTS_THREADS (0 = auto).
static uint32_t fused_common_points_adaptive_threads(size_t term_count,
                                                     size_t output_count) {
  const long configured = static_cast<long>(
      env_size_or_default("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_THREADS", 0u));
  if (configured > 0u) {
    return fused_common_points_threads_per_row();
  }
  // Large output batches already expose enough row-level parallelism. One
  // warp per row maximizes residency and avoids the saturation loss from
  // assigning four warps to every independent VC.
  if (output_count >= 512u) {
    return 32u;
  }
  if (output_count >= 128u) {
    if (term_count < 34u) {
      return 32u;
    }
    if (term_count <= 160u) {
      return 64u;
    }
  }
  // Tiny proof-stage MSMs retain one warp. Fixed-comb BBS rows need four
  // warps to hide the larger comb-table memory latency on current GPUs.
  if (term_count < 18u) {
    return 32u;
  }
  if (term_count <= 160u) {
    return 128u;
  }
  if (term_count <= 256u) {
    return 64u;
  }
  return 128u;
}


#endif
