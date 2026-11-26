#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_COMMON_BASE_CUH
#define CUBBS_BBS_COMMON_BASE_CUH

// ============================================================================
// Core G1 / GLV / Radix-16 / Affine Normalization Device Primitives
// ============================================================================

// BLS12-381 G1 GLV endomorphism -sigma^2(P) = (beta^2 * X, -Y, ZZZ, ZZ).
__device__ __forceinline__ static void cubbs_bbs_neg_sigma2_in_place(
    cubbs_sppark_bucket_t *point) {
  fp_t *f = reinterpret_cast<fp_t *>(point);
  fp_t beta_sq;
  const uint64_t beta_sq_limbs[6] = {
      0x30f1361b798a64e8, 0xf3b8ddab7ece5a2a, 0x16a8ca3ac61577f7,
      0xc26a2ff874fd029b, 0x3636b76660701c6e, 0x051ba4ab241b6160};
  memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
  f[0] = f[0] * beta_sq;
  point->cneg(true);
}

// Exact GLV scalar split s = remainder + quotient * z^2 (mod r) where both
// remainder and quotient fit in 128 bits for every canonical Fr scalar.
__device__ __forceinline__ static void cubbs_bbs_scalar_div_by_z2_device(
    const uint8_t scalar_le[32], uint8_t remainder_le[16],
    uint8_t quotient_le[16]) {
  constexpr uint64_t divisor_lo = 0x0000000100000000ULL;
  constexpr uint64_t divisor_hi = 0xac45a4010001a402ULL;
  uint64_t remainder_lo = 0u;
  uint64_t remainder_hi = 0u;
  memcpy(&remainder_lo, scalar_le + 16u, sizeof(remainder_lo));
  memcpy(&remainder_hi, scalar_le + 24u, sizeof(remainder_hi));
  uint64_t remainder_top = 0u;
  uint64_t quotient_lo = 0u;
  uint64_t quotient_hi = 0u;

  #pragma unroll 1
  for (int bit = 127; bit >= 0; --bit) {
    remainder_top = (remainder_top << 1) | (remainder_hi >> 63);
    remainder_hi = (remainder_hi << 1) | (remainder_lo >> 63);
    remainder_lo = (remainder_lo << 1) |
                   ((scalar_le[bit >> 3] >> (bit & 7)) & 1u);
    const bool ge = remainder_top != 0u || remainder_hi > divisor_hi ||
                    (remainder_hi == divisor_hi && remainder_lo >= divisor_lo);
    if (!ge) continue;
    const uint64_t old_lo = remainder_lo;
    remainder_lo -= divisor_lo;
    const uint64_t borrow_lo = old_lo < divisor_lo;
    const uint64_t sub_hi = divisor_hi + borrow_lo;
    const uint64_t borrow_hi = remainder_hi < sub_hi || sub_hi < divisor_hi;
    remainder_hi -= sub_hi;
    remainder_top -= borrow_hi;
    if (bit < 64) quotient_lo |= 1ULL << bit;
    else quotient_hi |= 1ULL << (bit - 64);
  }
  memcpy(remainder_le, &remainder_lo, sizeof(remainder_lo));
  memcpy(remainder_le + 8, &remainder_hi, sizeof(remainder_hi));
  memcpy(quotient_le, &quotient_lo, sizeof(quotient_lo));
  memcpy(quotient_le + 8, &quotient_hi, sizeof(quotient_hi));
}

// Signed Radix-16 recoding for 128-bit scalars (33 signed digits in [-8, +8]).
__device__ __forceinline__ static void cubbs_bbs_pack_signed_radix16(
    const uint8_t scalar_le[16], uint64_t packed[3]) {
  packed[0] = 0u;
  packed[1] = 0u;
  packed[2] = 0u;
  uint32_t carry = 0u;
  #pragma unroll
  for (uint32_t window = 0u; window < 32u; ++window) {
    const uint32_t byte = window >> 1u;
    const uint32_t shift = (window & 1u) * 4u;
    const uint32_t raw = ((scalar_le[byte] >> shift) & 0x0fu) + carry;
    const int32_t digit = raw > 8u ? static_cast<int32_t>(raw) - 16
                                   : static_cast<int32_t>(raw);
    carry = raw > 8u ? 1u : 0u;
    const uint64_t code = static_cast<uint64_t>(digit + 8);
    const uint32_t bit = window * 5u;
    const uint32_t word = bit >> 6u;
    const uint32_t offset = bit & 63u;
    packed[word] |= code << offset;
    if (offset > 59u) packed[word + 1u] |= code >> (64u - offset);
  }
  packed[2] |= static_cast<uint64_t>(carry + 8u) << 32u;
}

__device__ __forceinline__ static int32_t cubbs_bbs_signed_radix16_digit(
    const uint64_t packed[3], uint32_t window) {
  const uint32_t bit = window * 5u;
  const uint32_t word = bit >> 6u;
  const uint32_t offset = bit & 63u;
  uint64_t code = packed[word] >> offset;
  if (offset > 59u && word < 2u) {
    code |= packed[word + 1u] << (64u - offset);
  }
  return static_cast<int32_t>(code & 0x1fu) - 8;
}

__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_scalar_mul_glv_signed_radix16_shared(
    const cubbs_sppark_bucket_t *multiples,
    const uint8_t remainder_le[16], const uint8_t quotient_le[16]) {
  uint64_t remainder_digits[3];
  uint64_t quotient_digits[3];
  cubbs_bbs_pack_signed_radix16(remainder_le, remainder_digits);
  cubbs_bbs_pack_signed_radix16(quotient_le, quotient_digits);
  cubbs_sppark_bucket_t acc;
  acc.inf();
  bool started = false;
  #pragma unroll 1
  for (int window = 32; window >= 0; --window) {
    if (started) {
      #pragma unroll
      for (uint32_t doubling = 0u; doubling < 4u; ++doubling) acc.add(acc);
    }
    const int32_t d1 = cubbs_bbs_signed_radix16_digit(
        remainder_digits, static_cast<uint32_t>(window));
    const int32_t d2 = cubbs_bbs_signed_radix16_digit(
        quotient_digits, static_cast<uint32_t>(window));
    if (d1 != 0) {
      auto term = multiples[d1 < 0 ? -d1 : d1];
      if (d1 < 0) term.cneg(true);
      if (!started) { acc = term; started = true; }
      else acc.add(term);
    }
    if (d2 != 0) {
      auto term = multiples[d2 < 0 ? -d2 : d2];
      if (d2 < 0) term.cneg(true);
      cubbs_bbs_neg_sigma2_in_place(&term);
      if (!started) { acc = term; started = true; }
      else acc.add(term);
    }
  }
  return acc;
}

__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_scalar_mul_glv_device(cubbs_sppark_bucket_t base,
                                    const uint8_t remainder_le[16],
                                    const uint8_t quotient_le[16]) {
  cubbs_sppark_bucket_t neg_sigma2 = base;
  cubbs_bbs_neg_sigma2_in_place(&neg_sigma2);

  cubbs_sppark_bucket_t acc;
  acc.inf();
  bool started = false;
  for (int byte = 15; byte >= 0; --byte) {
    const uint8_t b1 = remainder_le[byte];
    const uint8_t b2 = quotient_le[byte];
    #pragma unroll
    for (int bit = 7; bit >= 0; --bit) {
      if (started) acc.add(acc);
      const bool d1 = ((b1 >> bit) & 1u) != 0u;
      const bool d2 = ((b2 >> bit) & 1u) != 0u;
      if (d1) {
        if (!started) { acc = base; started = true; }
        else acc.add(base);
      }
      if (d2) {
        if (!started) { acc = neg_sigma2; started = true; }
        else acc.add(neg_sigma2);
      }
    }
  }
  return acc;
}

// Build [0, 1*P, 2*P, ..., 8*P] in XYZZ for small-B Verify and Dual-MSM.
__global__ static void cubbs_bbs_proof_verify_radix16_tables_kernel(
    const cubbs_sppark_plain_affine_t *proof_bases, size_t proof_point_count,
    cubbs_sppark_bucket_t *tables) {
  constexpr uint32_t table_size = 9u;
  const size_t point =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= proof_point_count) return;
  auto *table = tables + point * table_size;
  table[0].inf();
  table[1] = proof_bases[point];
  #pragma unroll
  for (uint32_t digit = 2u; digit < table_size; ++digit) {
    table[digit] = table[digit - 1u];
    table[digit].add(table[1]);
  }
}

// 256-thread Montgomery batch XYZZ -> plain affine normalizer (used during Lim-Lee table build).
__global__ static void cubbs_bbs_g1_comb_xyzz_to_affine_kernel(
    const cubbs_sppark_bucket_t *input,
    cubbs_sppark_plain_affine_t *output, size_t count) {
  constexpr uint32_t kThreads = 256u;
  __shared__ __align__(16) uint8_t scan_storage[2u * kThreads * sizeof(fp_t)];
  __shared__ __align__(16) uint8_t total_storage[sizeof(fp_t)];
  auto *scan_a = reinterpret_cast<fp_t *>(scan_storage);
  auto *scan_b = scan_a + kThreads;
  auto *total_product = reinterpret_cast<fp_t *>(total_storage);
  const uint32_t tid = threadIdx.x;
  const size_t index = static_cast<size_t>(blockIdx.x) * kThreads + tid;
  const bool valid = index < count;
  cubbs_sppark_bucket_t point;
  if (valid) point = input[index];
  const bool finite = valid && !point.is_inf();
  const fp_t *coordinates = reinterpret_cast<const fp_t *>(&point);
  const fp_t one = fp_t::one();
  const fp_t z = finite ? coordinates[2] : one;

  scan_b[tid] = z;
  __syncthreads();
  if (tid == 0u) {
    fp_t product = one;
    for (uint32_t item = 0u; item < kThreads; ++item) {
      scan_a[item] = product;
      product *= scan_b[item];
    }
    *total_product = product;
  }
  __syncthreads();
  if (tid == 0u) {
    static constexpr uint64_t modulus_minus_two[6] = {
        0xb9feffffffffaaa9ull, 0x1eabfffeb153ffffull,
        0x6730d2a0f6b0f624ull, 0x64774b84f38512bfull,
        0x4b1ba7b6434bacd7ull, 0x1a0111ea397fe69aull};
    fp_t inverse = one;
    const fp_t base = *total_product;
    for (int bit = 380; bit >= 0; --bit) {
      inverse *= inverse;
      if (((modulus_minus_two[bit / 64] >> (bit % 64)) & 1ull) != 0ull) {
        inverse *= base;
      }
    }
    for (int item = static_cast<int>(kThreads) - 1; item >= 0; --item) {
      const fp_t inverse_item = inverse * scan_a[item];
      inverse *= scan_b[item];
      scan_a[item] = inverse_item;
    }
  }
  __syncthreads();

  if (valid) {
    if (!finite) {
      fp_t zero;
      zero.zero();
      output[index] = cubbs_sppark_plain_affine_t{zero, zero};
    } else {
      const fp_t inverse_zzz = scan_a[tid];
      fp_t affine_y = inverse_zzz * coordinates[1];
      fp_t inverse_z = inverse_zzz * coordinates[3];
      fp_t affine_x = inverse_z * inverse_z;
      affine_x *= coordinates[0];
      output[index] = cubbs_sppark_plain_affine_t{affine_x, affine_y};
    }
  }
}

// Generic variable-base GPU MSM fallback (one block per independent MSM row).
__global__ static void cubbs_bbs_g1_msm_generic_rows_glv_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    const size_t *row_offsets, size_t item_count,
    cubbs_sppark_bucket_t *outputs) {
  constexpr uint32_t kThreads = 128u;
  __shared__ __align__(16) uint8_t
      partial_storage[kThreads * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t item = blockIdx.x;
  if (item >= item_count) return;

  const uint32_t tid = threadIdx.x;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  const size_t begin = row_offsets[item];
  const size_t end = row_offsets[item + 1u];
  for (size_t term = begin + tid; term < end; term += blockDim.x) {
    cubbs_sppark_bucket_t base;
    base = bases[term];
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    cubbs_bbs_scalar_div_by_z2_device(
        scalars_le + term * 32u, remainder_le, quotient_le);
    acc.add(cubbs_bbs_g1_scalar_mul_glv_device(base, remainder_le, quotient_le));
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) outputs[item] = partials[0];
}

#endif // CUBBS_BBS_COMMON_BASE_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
