#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

__global__ static void cubbs_bbs_g1_msm_common_points_bucket4_kernel(
    const cubbs_sppark_plain_affine_t *points,
    const uint8_t *scalars_le, size_t term_count,
    size_t output_count, cubbs_sppark_bucket_t *outputs) {
  constexpr uint32_t kThreadsPerRow = 128u;
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerRow * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);

  const size_t row = blockIdx.x;
  if (row >= output_count) {
    return;
  }

  cubbs_sppark_bucket_t acc;
  acc.inf();
  const uint32_t tid = threadIdx.x;
  const uint8_t *row_scalars = scalars_le + (row * term_count * 32u);

  for (int window = 63; window >= 0; --window) {
    if (tid < 15u) {
      const uint8_t bucket_digit = static_cast<uint8_t>(tid + 1u);
      cubbs_sppark_bucket_t bucket;
      bucket.inf();
      for (size_t term = 0; term < term_count; ++term) {
        const uint8_t digit = cubbs_bbs_scalar_le_window4(
            row_scalars + (term * 32u), static_cast<uint32_t>(window));
        if (digit == bucket_digit) {
          bucket.add(points[term]);
        }
      }
      partials[tid] = bucket;
    }
    __syncthreads();

    if (tid == 0u) {
      if (window != 63) {
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
          acc.add(acc);
        }
      }

      cubbs_sppark_bucket_t running;
      cubbs_sppark_bucket_t window_sum;
      running.inf();
      window_sum.inf();
      for (int bucket = 14; bucket >= 0; --bucket) {
        running.add(partials[bucket]);
        window_sum.add(running);
      }
      acc.add(window_sum);
    }
    __syncthreads();
  }

  if (tid == 0u) {
    outputs[row] = acc;
  }
}

#endif
