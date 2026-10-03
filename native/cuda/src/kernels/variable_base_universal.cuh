#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

// Device-side G1 scalar multiplication via double-and-add over a 256-bit
// little-endian scalar. Uses sppark's xyzz_t arithmetic (add/doubling), which
// is device-compatible. The scalar is processed MSB-first (bit 255 down to 0)
// so each step doubles the accumulator and conditionally adds the base point.
// ~255 doublings + ~128 additions per mul. For the 6 fixed BBS proof-gen muls
// this runs in parallel across proofs (one block per proof), so per-proof
// latency is hidden by throughput. The base point is accepted in projective
// (xyzz) form so intermediate results (d, abar) can be re-multiplied without
// round-tripping through affine.
// BLS12-381 G1 GLV endomorphism beta (cube root of unity in Fp, beta^3=1).
// Derived from BLST src/e1.c (Copyright Supranational LLC, Apache-2.0; see
// NOTICE and THIRD_PARTY.md). Used by GLV scalar multiplication to halve the
// scalar length via the endomorphism sigma(P) = (X, Y, beta*Z).
// fp_t (mont_t<381>) stores 6x uint64 Montgomery limbs. We store beta in the
// same Montgomery form as fp_t so the device can multiply directly.
// This constant is beta in Montgomery form, matching blst's limb layout.
// For now we use the blst beta value; the device kernel will apply it to the
// ZZ field (Z^2) of xyzz_t to produce sigma(P) = (X, Y, beta^2*ZZ, ...).
// NOTE: GLV decomposition requires scalar splitting (k -> k1 + lambda*k2)
// which needs lattice reduction done on the host. The device kernel receives
// the two 128-bit half-scalars and does a joint 2-point double-and-add.

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

template <uint32_t WINDOW_BITS>
__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_scalar_mul_glv_window_shared(
    const cubbs_sppark_bucket_t *multiples,
    const uint8_t remainder_le[16], const uint8_t quotient_le[16]) {
  constexpr uint32_t window_count = (128u + WINDOW_BITS - 1u) / WINDOW_BITS;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  bool started = false;
  #pragma unroll 1
  for (int window = static_cast<int>(window_count) - 1; window >= 0; --window) {
    if (started) {
      #pragma unroll
      for (uint32_t doubling = 0u; doubling < WINDOW_BITS; ++doubling) {
        acc.add(acc);
      }
    }
    uint8_t d1 = 0u;
    uint8_t d2 = 0u;
    #pragma unroll
    for (uint32_t offset = 0u; offset < WINDOW_BITS; ++offset) {
      const uint32_t bit = static_cast<uint32_t>(window) * WINDOW_BITS + offset;
      if (bit < 128u) {
        d1 |= ((remainder_le[bit >> 3u] >> (bit & 7u)) & 1u) << offset;
        d2 |= ((quotient_le[bit >> 3u] >> (bit & 7u)) & 1u) << offset;
      }
    }
    if (d1 != 0u) {
      auto term = multiples[d1];
      if (!started) { acc = term; started = true; }
      else acc.add(term);
    }
    if (d2 != 0u) {
      auto term = multiples[d2];
      cubbs_bbs_neg_sigma2_in_place(&term);
      if (!started) { acc = term; started = true; }
      else acc.add(term);
    }
  }
  return acc;
}

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

// Exact host-side form of BLST's G1 GLV split. Keeping the 32-byte
// (remainder || quotient) layout identical to the device consumer lets batch
// paths move the integer division off the GPU without changing the MSM math.
static void cubbs_bbs_scalar_div_by_z2_host(const uint8_t scalar_le[32],
                                             uint8_t split_le[32]) {
  static constexpr uint64_t divisor[2] = {
      0x0000000100000000ULL, 0xac45a4010001a402ULL};
  uint64_t value[4];
  std::memcpy(value, scalar_le, sizeof(value));
  for (size_t limb = 2u; limb-- > 0u;) {
    const uint64_t quotient =
        div_3_limbs(value + limb + 1u, divisor[0], divisor[1]);
    (void)quot_rem_128(value + limb, divisor, quotient);
  }
  std::memcpy(split_le, value, sizeof(value));
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

__device__ __forceinline__ static void cubbs_bbs_pack_signed_radix256(
    const uint8_t scalar_le[16], uint64_t packed[3]) {
  packed[0] = 0u;
  packed[1] = 0u;
  packed[2] = 0u;
  uint32_t carry = 0u;
  #pragma unroll
  for (uint32_t window = 0u; window < 16u; ++window) {
    const uint32_t raw = static_cast<uint32_t>(scalar_le[window]) + carry;
    const int32_t digit = raw > 128u ? static_cast<int32_t>(raw) - 256
                                     : static_cast<int32_t>(raw);
    carry = raw > 128u ? 1u : 0u;
    const uint64_t code = static_cast<uint64_t>(digit + 128);
    const uint32_t bit = window * 9u;
    const uint32_t word = bit >> 6u;
    const uint32_t offset = bit & 63u;
    packed[word] |= code << offset;
    if (offset > 55u) packed[word + 1u] |= code >> (64u - offset);
  }
  packed[2] |= static_cast<uint64_t>(carry + 128u) << 16u;
}

__device__ __forceinline__ static int32_t cubbs_bbs_signed_radix256_digit(
    const uint64_t packed[3], uint32_t window) {
  const uint32_t bit = window * 9u;
  const uint32_t word = bit >> 6u;
  const uint32_t offset = bit & 63u;
  uint64_t code = packed[word] >> offset;
  if (offset > 55u && word < 2u) {
    code |= packed[word + 1u] << (64u - offset);
  }
  return static_cast<int32_t>(code & 0x1ffu) - 128;
}

__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_scalar_mul_glv_signed_radix256_shared(
    const cubbs_sppark_bucket_t *multiples,
    const uint8_t remainder_le[16], const uint8_t quotient_le[16]) {
  uint64_t remainder_digits[3];
  uint64_t quotient_digits[3];
  cubbs_bbs_pack_signed_radix256(remainder_le, remainder_digits);
  cubbs_bbs_pack_signed_radix256(quotient_le, quotient_digits);
  cubbs_sppark_bucket_t acc;
  acc.inf();
  bool started = false;
  #pragma unroll 1
  for (int window = 16; window >= 0; --window) {
    if (started) {
      #pragma unroll
      for (uint32_t doubling = 0u; doubling < 8u; ++doubling) acc.add(acc);
    }
    const int32_t d1 = cubbs_bbs_signed_radix256_digit(
        remainder_digits, static_cast<uint32_t>(window));
    const int32_t d2 = cubbs_bbs_signed_radix256_digit(
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

// blst's BLS12-381 G1 GLV split is exact and especially simple: divide k by
// z², where z² = 0xac45a4010001a4020000000100000000. If k=q*z²+r,
// the curve endomorphism relation gives k*P = r*P-q*sigma²(P). Both q and r
// fit in 128 bits for every scalar in Fr.
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
  // Every canonical Fr scalar is below 2^255 and its high 128-bit half is
  // smaller than z^2. Seed the remainder with that half and divide only the
  // low half, removing 128 iterations that can never emit a quotient bit.
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

__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_scalar_mul_glv_device(cubbs_sppark_bucket_t base,
                                    const uint8_t remainder_le[16],
                                    const uint8_t quotient_le[16]) {
  // sigma²(P) has affine x=beta²*x and unchanged y. Multiplying projective X
  // preserves a valid XYZZ representation for subsequent additions.
  cubbs_sppark_bucket_t neg_sigma2 = base;
  {
    fp_t *f = reinterpret_cast<fp_t *>(&neg_sigma2);
    fp_t beta_sq;
    const uint64_t beta_sq_limbs[6] = {
        0x30f1361b798a64e8, 0xf3b8ddab7ece5a2a, 0x16a8ca3ac61577f7,
        0xc26a2ff874fd029b, 0x3636b76660701c6e, 0x051ba4ab241b6160};
    memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
    f[0] = f[0] * beta_sq;
  }
  neg_sigma2.cneg(true);

  // Joint double-and-add over the two exact 128-bit components.
  cubbs_sppark_bucket_t acc;
  acc.inf();
  bool started = false;
  for (int byte = 15; byte >= 0; --byte) {
    const uint8_t b1 = remainder_le[byte];
    const uint8_t b2 = quotient_le[byte];
    #pragma unroll
    for (int bit = 7; bit >= 0; --bit) {
      if (started) {
        acc.add(acc);
      }
      const bool d1 = (b1 >> bit) & 1u;
      const bool d2 = (b2 >> bit) & 1u;
      if (d1) {
        if (!started) { acc = base; started = true; }
        else { acc.add(base); }
      }
      if (d2) {
        if (!started) { acc = neg_sigma2; started = true; }
        else { acc.add(neg_sigma2); }
      }
    }
  }
  return acc;
}

// Universal variable-base MSM fallback. One block owns one independent MSM
// row, and its threads distribute that row's arbitrary number of terms. This
// is intentionally less specialized than the one/three-term proof kernels but
// keeps every GPU-selected shape on an cuBBS-Cred executable kernel.
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
    auto product = cubbs_bbs_g1_scalar_mul_glv_device(
        base, remainder_le, quotient_le);
    acc.add(product);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) outputs[item] = partials[0];
}

// Route 4 owns arbitrary and heterogeneous rows, but a single CTA must not own
// every term of a very large row. This kernel adds a second grid dimension for
// term tiles; the companion reduction kernel combines the per-row partials.
// Row offsets retain the fully general variable-base layout and the old
// one-CTA kernel remains the bounded-memory fallback.
template <uint32_t kThreadsPerTile>
__global__ static void cubbs_bbs_g1_msm_generic_rows_glv_tiled_kernel(
    const cubbs_sppark_plain_affine_t *bases, const uint8_t *scalars_le,
    const size_t *row_offsets, size_t item_count, size_t tile_terms,
    size_t max_tile_count, cubbs_sppark_bucket_t *tile_partials) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreadsPerTile * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t item = blockIdx.x;
  const size_t tile = blockIdx.y;
  if (item >= item_count || tile >= max_tile_count) return;

  const size_t row_begin = row_offsets[item];
  const size_t row_end = row_offsets[item + 1u];
  const size_t begin = row_begin + tile * tile_terms;
  if (begin >= row_end) return;
  const size_t end = min(row_end, begin + tile_terms);
  const uint32_t tid = threadIdx.x;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t term = begin + tid; term < end; term += kThreadsPerTile) {
    cubbs_sppark_bucket_t base = bases[term];
    uint8_t remainder_le[16];
    uint8_t quotient_le[16];
    cubbs_bbs_scalar_div_by_z2_device(
        scalars_le + term * 32u, remainder_le, quotient_le);
    acc.add(cubbs_bbs_g1_scalar_mul_glv_device(
        base, remainder_le, quotient_le));
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreadsPerTile >> 1u; stride > 0u;
       stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) {
    tile_partials[item * max_tile_count + tile] = partials[0];
  }
}

template <uint32_t kThreads>
__global__ static void cubbs_bbs_g1_msm_generic_rows_glv_reduce_kernel(
    const cubbs_sppark_bucket_t *tile_partials, const size_t *row_offsets,
    size_t item_count, size_t tile_terms, size_t max_tile_count,
    cubbs_sppark_bucket_t *outputs) {
  __shared__ __align__(16) uint8_t
      partial_storage[kThreads * sizeof(cubbs_sppark_bucket_t)];
  auto *partials = reinterpret_cast<cubbs_sppark_bucket_t *>(partial_storage);
  const size_t item = blockIdx.x;
  if (item >= item_count) return;
  const size_t row_terms = row_offsets[item + 1u] - row_offsets[item];
  const size_t tile_count =
      row_terms == 0u ? 0u : (row_terms + tile_terms - 1u) / tile_terms;
  const uint32_t tid = threadIdx.x;
  cubbs_sppark_bucket_t acc;
  acc.inf();
  for (size_t tile = tid; tile < tile_count && tile < max_tile_count;
       tile += kThreads) {
    acc.add(tile_partials[item * max_tile_count + tile]);
  }
  partials[tid] = acc;
  __syncthreads();
  for (uint32_t stride = kThreads >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partials[tid].add(partials[tid + stride]);
    __syncthreads();
  }
  if (tid == 0u) outputs[item] = partials[0];
}

static int32_t cubbs_bbs_cuda_g1_variable_base_generic_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  const size_t byte_budget = env_size_or_default(
      "CUBBS_BBS_GPU_GENERIC_ITEM_BUDGET_BYTES", 256u * 1024u * 1024u);
  const size_t bytes_per_term =
      sizeof(cubbs_sppark_plain_affine_t) + 32u;

  for (size_t item_begin = 0u; item_begin < item_count;) {
    size_t item_end = item_begin;
    size_t term_count = 0u;
    while (item_end < item_count &&
           item_end - item_begin < static_cast<size_t>(UINT32_MAX)) {
      const size_t next_terms = items[item_end].term_count;
      if (next_terms > static_cast<size_t>(-1) - term_count) {
        return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
      }
      const size_t combined_terms = term_count + next_terms;
      const bool over_budget = bytes_per_term > 0u &&
          combined_terms > std::max<size_t>(1u, byte_budget) / bytes_per_term;
      if (item_end > item_begin && over_budget) break;
      term_count = combined_terms;
      ++item_end;
      if (over_budget) break;
    }
    const size_t rows = item_end - item_begin;
    if (rows == 0u || term_count > static_cast<size_t>(-1) / 32u) {
      return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }

    size_t max_row_terms = 0u;
    for (size_t row = item_begin; row < item_end; ++row) {
      max_row_terms = std::max(max_row_terms, items[row].term_count);
    }
    const size_t universal_tile_terms = bbs_universal_item_tile_terms();
    const size_t max_tile_count = max_row_terms == 0u
        ? 0u
        : (max_row_terms + universal_tile_terms - 1u) /
              universal_tile_terms;
    bool use_tiled =
        bbs_universal_tiled_kernel_enabled() &&
        max_row_terms >= bbs_universal_item_tiled_min_terms() &&
        max_tile_count > 1u && max_tile_count <= 65535u &&
        rows <= static_cast<size_t>(UINT32_MAX) &&
        max_tile_count <= static_cast<size_t>(-1) / rows;
    const size_t partial_count = use_tiled ? rows * max_tile_count : 0u;
    if (use_tiled &&
        partial_count > bbs_universal_item_partial_budget_bytes() /
                            sizeof(cubbs_sppark_bucket_t)) {
      use_tiled = false;
    }

    std::vector<size_t> host_offsets(rows + 1u, 0u);
    std::vector<CuBbsBbsGpuByteSlice> point_slices(term_count);
    std::vector<CuBbsBbsGpuByteSlice> scalar_slices(term_count);
    size_t cursor = 0u;
    for (size_t local_row = 0u; local_row < rows; ++local_row) {
      const auto &item = items[item_begin + local_row];
      host_offsets[local_row] = cursor;
      for (size_t term = 0u; term < item.term_count; ++term) {
        point_slices[cursor] = item.points[term];
        scalar_slices[cursor] = item.scalars[term];
        ++cursor;
      }
    }
    host_offsets[rows] = cursor;

    std::vector<cubbs_sppark_plain_affine_t> host_bases(term_count);
    std::vector<uint8_t> host_scalars(term_count * 32u);
    int32_t status = CUBBS_BBS_GPU_SUCCESS;
    if (term_count > 0u) {
      status = cubbs_bbs_blst_decode_g1_points(
          point_slices.data(), term_count,
          reinterpret_cast<uint8_t *>(host_bases.data()),
          sizeof(cubbs_sppark_plain_affine_t));
      if (status == CUBBS_BBS_GPU_SUCCESS) {
        status = cubbs_fused_decode_scalars_parallel(
            scalar_slices.data(), term_count, host_scalars.data());
      }
    }
    if (status != CUBBS_BBS_GPU_SUCCESS) {
      for (size_t row = item_begin; row < item_end; ++row) {
        set_result(results, row, status, 0u);
      }
      item_begin = item_end;
      continue;
    }

    cubbs_sppark_plain_affine_t *device_bases = nullptr;
    uint8_t *device_scalars = nullptr;
    size_t *device_offsets = nullptr;
    cubbs_sppark_bucket_t *device_outputs = nullptr;
    cubbs_sppark_bucket_t *device_tile_partials = nullptr;
    const size_t base_bytes =
        term_count * sizeof(cubbs_sppark_plain_affine_t);
    cudaError_t error = cudaMalloc(
        &device_bases, std::max<size_t>(1u, base_bytes));
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_scalars,
                         std::max<size_t>(1u, term_count * 32u));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_offsets,
                         (rows + 1u) * sizeof(*device_offsets));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_outputs, rows * sizeof(*device_outputs));
    }
    if (error == cudaSuccess && use_tiled) {
      error = cudaMalloc(
          &device_tile_partials,
          partial_count * sizeof(*device_tile_partials));
      // A temporary partial-buffer allocation must not revoke Route 4's
      // universal behavior. Fall back to its original one-CTA implementation
      // when only this optimization workspace is unavailable.
      if (error == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        error = cudaSuccess;
        use_tiled = false;
      }
    }
    if (error == cudaSuccess && base_bytes > 0u) {
      error = cudaMemcpy(device_bases, host_bases.data(), base_bytes,
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess && term_count > 0u) {
      error = cudaMemcpy(device_scalars, host_scalars.data(), term_count * 32u,
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_offsets, host_offsets.data(),
                         (rows + 1u) * sizeof(*device_offsets),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      if (use_tiled) {
        constexpr uint32_t tile_threads = 128u;
        const dim3 grid(static_cast<uint32_t>(rows),
                        static_cast<uint32_t>(max_tile_count));
        cubbs_bbs_g1_msm_generic_rows_glv_tiled_kernel<tile_threads>
            <<<grid, tile_threads>>>(
                device_bases, device_scalars, device_offsets, rows,
                universal_tile_terms, max_tile_count, device_tile_partials);
        error = cudaGetLastError();
        if (error == cudaSuccess) {
          constexpr uint32_t reduce_threads = 128u;
          cubbs_bbs_g1_msm_generic_rows_glv_reduce_kernel<reduce_threads>
              <<<static_cast<uint32_t>(rows), reduce_threads>>>(
                  device_tile_partials, device_offsets, rows,
                  universal_tile_terms, max_tile_count, device_outputs);
          error = cudaGetLastError();
        }
      } else {
        cubbs_bbs_g1_msm_generic_rows_glv_kernel<<<
            static_cast<uint32_t>(rows), 128u>>>(
            device_bases, device_scalars, device_offsets, rows, device_outputs);
        error = cudaGetLastError();
      }
    }
    std::vector<cubbs_sppark_bucket_t> host_outputs(rows);
    if (error == cudaSuccess) {
      error = cudaMemcpy(host_outputs.data(), device_outputs,
                         rows * sizeof(host_outputs[0]),
                         cudaMemcpyDeviceToHost);
    }
    if (device_outputs != nullptr) (void)cudaFree(device_outputs);
    if (device_tile_partials != nullptr)
      (void)cudaFree(device_tile_partials);
    if (device_offsets != nullptr) (void)cudaFree(device_offsets);
    if (device_scalars != nullptr) (void)cudaFree(device_scalars);
    if (device_bases != nullptr) (void)cudaFree(device_bases);

    if (error != cudaSuccess) {
      for (size_t row = item_begin; row < item_end; ++row) {
        set_result(results, row, CUBBS_BBS_GPU_ERR_INTERNAL, 0u);
      }
      item_begin = item_end;
      continue;
    }
    for (size_t local_row = 0u; local_row < rows; ++local_row) {
      const size_t row = item_begin + local_row;
      const cubbs_sppark_point_t projective =
          static_cast<cubbs_sppark_point_t>(host_outputs[local_row]);
      const int32_t compress_status = cubbs_bbs_blst_compress_g1_projective(
          &projective, items[row].compressed_point_out);
      if (compress_status == CUBBS_BBS_GPU_SUCCESS) {
        set_result_route(results, row, CUBBS_BBS_GPU_SUCCESS, 48u,
                         CUBBS_BBS_MSM_ROUTE_CUBBS_GENERIC);
      } else {
        set_result(results, row, compress_status, 0u);
      }
    }
    item_begin = item_end;
  }
  return CUBBS_BBS_GPU_SUCCESS;
}


#endif
