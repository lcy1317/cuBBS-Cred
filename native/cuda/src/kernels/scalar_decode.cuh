#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

__device__ __forceinline__ static bool cubbs_bbs_scalar_be_is_canonical(
    const uint8_t *scalar_be) {
  static constexpr uint8_t fr_modulus_be[32] = {
      0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48,
      0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
      0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe,
      0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01};
#pragma unroll
  for (uint32_t byte = 0u; byte < 32u; ++byte) {
    if (scalar_be[byte] < fr_modulus_be[byte]) return true;
    if (scalar_be[byte] > fr_modulus_be[byte]) return false;
  }
  return false;
}

// Convert a strided row-major matrix of canonical big-endian Fr scalars into
// the compact little-endian layout consumed by the cuBBS-Cred MSM kernels.
// Invalid values are replaced with zero and reported through invalid_out, so
// no non-canonical field element reaches device arithmetic.
__global__ static void cubbs_bbs_decode_packed_scalar_rows_be_kernel(
    const uint8_t *scalar_rows_be, size_t row_count, size_t term_count,
    size_t scalar_row_stride, size_t scalar_row_offset,
    uint8_t *scalars_le, uint32_t *invalid_out) {
  const size_t scalar =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t scalar_count = row_count * term_count;
  if (scalar >= scalar_count) return;

  const size_t row = scalar / term_count;
  const size_t term = scalar - row * term_count;
  const uint8_t *input = scalar_rows_be + row * scalar_row_stride +
                         scalar_row_offset + term * 32u;
  uint8_t *output = scalars_le + scalar * 32u;
  const bool canonical = cubbs_bbs_scalar_be_is_canonical(input);
#pragma unroll
  for (uint32_t byte = 0u; byte < 32u; ++byte) {
    output[byte] = canonical ? input[31u - byte] : 0u;
  }
  if (!canonical) atomicExch(invalid_out, 1u);
}

static cudaError_t cubbs_bbs_launch_packed_scalar_decode(
    const uint8_t *device_scalar_rows_be, size_t row_count,
    size_t term_count, size_t scalar_row_stride, size_t scalar_row_offset,
    uint8_t *device_scalars_le, uint32_t *device_invalid,
    cudaStream_t stream) {
  if (row_count == 0u || term_count == 0u) return cudaSuccess;
  constexpr uint32_t threads = 256u;
  const size_t scalar_count = row_count * term_count;
  const uint32_t blocks =
      static_cast<uint32_t>((scalar_count + threads - 1u) / threads);
  cubbs_bbs_decode_packed_scalar_rows_be_kernel
      <<<blocks, threads, 0, stream>>>(
          device_scalar_rows_be, row_count, term_count, scalar_row_stride,
          scalar_row_offset, device_scalars_le, device_invalid);
  return cudaGetLastError();
}

// The packed sign path can append one row multiplier in front of each scalar
// row. Scaling on the device removes output_count * term_count Fr
// multiplications from the host without changing the MSM kernels' input
// layout. Scalars enter and leave this kernel in canonical little-endian form.
__global__ static void cubbs_bbs_scale_scalar_rows_kernel(
    uint8_t *scalars_le, const uint8_t *row_scales_le, size_t term_count,
    size_t scalar_count) {
  const size_t scalar = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        static_cast<size_t>(threadIdx.x);
  if (scalar >= scalar_count) return;

  const size_t row = scalar / term_count;
  static_assert(sizeof(cubbs_sppark_scalar_t) == 32u,
                "BLS12-381 Fr storage must remain 32 bytes");
  cubbs_sppark_scalar_t value;
  cubbs_sppark_scalar_t scale;
  memcpy(&value, scalars_le + scalar * 32u, sizeof(value));
  memcpy(&scale, row_scales_le + row * 32u, sizeof(scale));
  value.to();
  scale.to();
  value *= scale;
  value.from();
  memcpy(scalars_le + scalar * 32u, &value, sizeof(value));
}

#endif
