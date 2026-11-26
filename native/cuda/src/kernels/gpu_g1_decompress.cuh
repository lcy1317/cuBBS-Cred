#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_GPU_G1_DECOMPRESS_CUH
#define CUBBS_BBS_GPU_G1_DECOMPRESS_CUH

// ============================================================================
// 100% GPU BLS12-381 G1 Uncompress + Scott (2021/1130) Subgroup Check
// ============================================================================
// Bit-exact match to BLST blst_p1_uncompress() + blst_p1_affine_in_g1() with
// optional non-infinity enforcement (required by RFC BBS for A, Abar, Bbar, D).
// Eliminates all CPU G1 decoding for unique signatures and unique proofs.

__device__ __noinline__ static fp_t cubbs_bbs_fp_pow_p_plus_1_div_4(const fp_t &a) {
  // (p + 1) / 4 in little-endian 64-bit limbs (379 bits, MSB at bit 378).
  static constexpr uint64_t EXP[6] = {
      0xee7fbfffffffeaabULL, 0x07aaffffac54ffffULL,
      0xd9cc34a83dac3d89ULL, 0xd91dd2e13ce144afULL,
      0x92c6e9ed90d2eb35ULL, 0x0680447a8e5ff9a6ULL};
  const fp_t t1 = a;
  const fp_t t2 = t1 * t1;
  const fp_t t3 = t2 * t1;

  fp_t res = t1; // pair 189 (bits 379..378) == 1
  #pragma unroll 1
  for (int pair_idx = 188; pair_idx >= 0; --pair_idx) {
    res = res * res;
    res = res * res;
    const uint32_t bit = static_cast<uint32_t>(pair_idx) * 2u;
    const uint32_t d = static_cast<uint32_t>((EXP[bit >> 6u] >> (bit & 63u)) & 3ull);
    if (d != 0u) {
      fp_t term = t1;
      if (d == 2u) term = t2;
      else if (d == 3u) term = t3;
      res = res * term;
    }
  }
  return res;
}

// Computes [-z]P = [0xd201000000010000]P from an affine point P using exact
// BLST addition chain (63 doublings + 5 mixed-affine additions).
__device__ __noinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_times_minus_z_affine(const cubbs_sppark_plain_affine_t &p) {
  cubbs_sppark_bucket_t acc;
  acc = p;
  cubbs_bbs_xyzz_dbl_u(acc); // 0x2
  static constexpr uint32_t DBLS[5] = {2u, 3u, 9u, 32u, 16u};
  #pragma unroll 1
  for (uint32_t step = 0u; step < 5u; ++step) {
    acc.uadd(p, false);
    const uint32_t n = DBLS[step];
    #pragma unroll 1
    for (uint32_t d = 0u; d < n; ++d) {
      cubbs_bbs_xyzz_dbl_u(acc);
    }
  }
  return acc;
}

// Computes [-z]Q = [0xd201000000010000]Q from an XYZZ point Q using exact
// BLST addition chain (63 doublings + 5 XYZZ additions).
__device__ __noinline__ static cubbs_sppark_bucket_t
cubbs_bbs_g1_times_minus_z_xyzz(const cubbs_sppark_bucket_t &q) {
  cubbs_sppark_bucket_t acc = q;
  cubbs_bbs_xyzz_dbl_u(acc); // 0x2
  static constexpr uint32_t DBLS[5] = {2u, 3u, 9u, 32u, 16u};
  #pragma unroll 1
  for (uint32_t step = 0u; step < 5u; ++step) {
    acc.uadd(q);
    const uint32_t n = DBLS[step];
    #pragma unroll 1
    for (uint32_t d = 0u; d < n; ++d) {
      cubbs_bbs_xyzz_dbl_u(acc);
    }
  }
  return acc;
}

// Uncompresses a 48-byte BLS12-381 G1 point and verifies on-curve + Scott G1 subgroup.
// Returns true iff valid; writes Montgomery plain affine (x, y) to *out.
__device__ static bool cubbs_bbs_g1_uncompress_and_check_device(
    const uint8_t in48[48], bool reject_infinity,
    cubbs_sppark_plain_affine_t *out) {
  const uint8_t b0 = in48[0];
  if ((b0 & 0x80u) == 0u) return false; // compressed bit required

  if ((b0 & 0x40u) != 0u) {
    // Infinity point: all remaining 382 bits must be zero.
    uint8_t acc_or = static_cast<uint8_t>(b0 & 0x3fu);
    #pragma unroll
    for (int i = 1; i < 48; ++i) acc_or |= in48[i];
    if (acc_or != 0u) return false;
    if (reject_infinity) return false;
    fp_t zero;
    zero.zero();
    *out = cubbs_sppark_plain_affine_t{zero, zero};
    return true;
  }

  const bool sign_flag = (b0 & 0x20u) != 0u;

  // Parse 48-byte big-endian X (with top 3 flag bits cleared) into 6 LE u64 limbs.
  uint64_t x_limbs[6];
  #pragma unroll
  for (int limb = 0; limb < 6; ++limb) {
    const int base = (5 - limb) * 8;
    const uint8_t first = (base == 0) ? static_cast<uint8_t>(b0 & 0x1fu) : in48[base];
    x_limbs[limb] =
        (static_cast<uint64_t>(first) << 56) |
        (static_cast<uint64_t>(in48[base + 1]) << 48) |
        (static_cast<uint64_t>(in48[base + 2]) << 40) |
        (static_cast<uint64_t>(in48[base + 3]) << 32) |
        (static_cast<uint64_t>(in48[base + 4]) << 24) |
        (static_cast<uint64_t>(in48[base + 5]) << 16) |
        (static_cast<uint64_t>(in48[base + 6]) << 8) |
        static_cast<uint64_t>(in48[base + 7]);
  }

  // Canonical field check: x < p strictly.
  static constexpr uint64_t FP_MOD_LE[6] = {
      0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
      0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
      0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
  bool less_than_p = false;
  #pragma unroll
  for (int limb = 5; limb >= 0; --limb) {
    if (x_limbs[limb] != FP_MOD_LE[limb]) {
      less_than_p = x_limbs[limb] < FP_MOD_LE[limb];
      break;
    }
  }
  if (!less_than_p) return false;

  // (0, +-2) is not in G1 subgroup.
  if ((x_limbs[0] | x_limbs[1] | x_limbs[2] | x_limbs[3] | x_limbs[4] | x_limbs[5]) == 0ull) {
    return false;
  }

  fp_t x;
  memcpy(&x, x_limbs, 48u);
  x.to();

  // rhs = x^3 + 4 (in Montgomery form).
  fp_t four = fp_t::one();
  four = four + four;
  four = four + four;
  const fp_t rhs = (x * x * x) + four;

  fp_t y = cubbs_bbs_fp_pow_p_plus_1_div_4(rhs);
  if (!((y * y) - rhs).is_zero()) return false;

  // Enforce lexicographic sign flag (y > (p-1)/2).
  fp_t y_std = y;
  y_std.from();
  uint64_t y_limbs[6];
  memcpy(y_limbs, &y_std, 48u);
  static constexpr uint64_t FP_HALF_LE[6] = {
      0xdcff7fffffffd555ull, 0x0f55ffff58a9ffffull,
      0xb39869507b587b12ull, 0xb23ba5c279c2895full,
      0x258dd3db21a5d66bull, 0x0d0088f51cbff34dull};
  bool y_gt_half = false;
  #pragma unroll
  for (int limb = 5; limb >= 0; --limb) {
    if (y_limbs[limb] != FP_HALF_LE[limb]) {
      y_gt_half = y_limbs[limb] > FP_HALF_LE[limb];
      break;
    }
  }
  if (y_gt_half != sign_flag) {
    y.cneg(true);
  }

  const cubbs_sppark_plain_affine_t pt{x, y};

  // Scott (2021/1130) G1 subgroup check:
  //   [-z]([-z]P) == [z^2]P == (beta^2 * x, -y) in XYZZ coordinates.
  const cubbs_sppark_bucket_t t0 = cubbs_bbs_g1_times_minus_z_affine(pt);
  if (t0.is_inf()) return false;
  const cubbs_sppark_bucket_t t1 = cubbs_bbs_g1_times_minus_z_xyzz(t0);
  if (t1.is_inf()) return false;

  fp_t beta_sq;
  const uint64_t beta_sq_limbs[6] = {
      0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
      0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
  memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));

  const fp_t target_x = x * beta_sq;
  fp_t target_y = y;
  target_y.cneg(true);

  const fp_t *c = reinterpret_cast<const fp_t *>(&t1);
  // xyzz_t layout: c[0]=X, c[1]=Y, c[2]=ZZZ, c[3]=ZZ.
  if (!(c[0] - (target_x * c[3])).is_zero()) return false;
  if (!(c[1] - (target_y * c[2])).is_zero()) return false;

  *out = pt;
  return true;
}

// Kernel: decompress `count` compressed 48B G1 points spaced at `in_stride`
// (with `pts_per_item` consecutive 48B points per item, e.g., 1 for signatures
// at stride 80, or 3 for proofs [Abar||Bbar||D] at proof_stride).
__global__ static void cubbs_bbs_g1_uncompress_strided_kernel(
    const uint8_t *compressed_base,
    size_t item_count,
    size_t item_stride,
    uint32_t pts_per_item,
    bool reject_infinity,
    cubbs_sppark_plain_affine_t *affines_out,
    uint32_t *invalid_flag) {
  const size_t total = item_count * pts_per_item;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const size_t item = idx / pts_per_item;
  const uint32_t sub = static_cast<uint32_t>(idx - (item * pts_per_item));
  const uint8_t *src = compressed_base + (item * item_stride) + (sub * 48u);
  cubbs_sppark_plain_affine_t pt;
  if (!cubbs_bbs_g1_uncompress_and_check_device(src, reject_infinity, &pt)) {
    atomicExch(invalid_flag, 1u);
    fp_t zero;
    zero.zero();
    affines_out[idx] = cubbs_sppark_plain_affine_t{zero, zero};
    return;
  }
  affines_out[idx] = pt;
}

// Kernel: scale each affine G1 point P_i by a 128-bit LE scalar r_i (at 32B stride)
// using Non-Adjacent Form (NAF: signed binary {-1, 0, +1}, 0 SMEM, ~42.7 mixed
// affine additions instead of ~64 — 33% fewer additions than binary!).
// Outputs XYZZ points for fast 64-thread Montgomery batch normalization to affine.
__global__ static void cubbs_bbs_g1_scale_128bit_to_xyzz_kernel(
    const cubbs_sppark_plain_affine_t *points_in,
    size_t point_stride, // 1 for Verify A_i; 3 for ProofVerify [Abar,Bbar,D] (Abar at +0)
    const uint8_t *scalars_le_stride32,
    size_t count,
    cubbs_sppark_bucket_t *xyzz_out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  const cubbs_sppark_plain_affine_t p = points_in[idx * point_stride];
  const uint8_t *s_le = scalars_le_stride32 + (idx * 32u);

  uint64_t lo = 0ull, hi = 0ull;
  memcpy(&lo, s_le, 8u);
  memcpy(&hi, s_le + 8u, 8u);

  // Convert 128-bit unsigned (hi:lo) to 129-digit NAF bitmasks (pos, neg, top_pos).
  uint64_t pos_lo = 0ull, pos_hi = 0ull;
  uint64_t neg_lo = 0ull, neg_hi = 0ull;
  uint32_t carry = 0u;
  #pragma unroll 1
  for (uint32_t b = 0u; b < 128u; ++b) {
    if ((lo & 1ull) != 0ull) {
      if ((lo & 2ull) != 0ull) {
        // digit = -1, add 1 to (carry:hi:lo) before shifting right by 1.
        if (b < 64u) neg_lo |= (1ull << b);
        else neg_hi |= (1ull << (b - 64u));
        lo += 1ull;
        if (lo == 0ull) {
          hi += 1ull;
          if (hi == 0ull) carry += 1u;
        }
      } else {
        // digit = +1, subtract 1 (clears LSB, handled by >> 1).
        if (b < 64u) pos_lo |= (1ull << b);
        else pos_hi |= (1ull << (b - 64u));
      }
    }
    lo = (lo >> 1u) | (hi << 63u);
    hi = (hi >> 1u) | (static_cast<uint64_t>(carry) << 63u);
    carry >>= 1u;
  }
  const bool top_pos = (lo != 0ull);

  cubbs_sppark_bucket_t acc;
  if (top_pos) {
    acc = p;
  } else {
    acc.inf();
  }
  #pragma unroll 1
  for (int bit = 127; bit >= 0; --bit) {
    if (!acc.is_inf()) {
      cubbs_bbs_xyzz_dbl_u(acc);
    }
    const uint32_t shift = static_cast<uint32_t>(bit & 63);
    const uint64_t pw = (bit >= 64) ? pos_hi : pos_lo;
    const uint64_t nw = (bit >= 64) ? neg_hi : neg_lo;
    if (((pw >> shift) & 1ull) != 0ull) {
      acc.uadd(p, false);
    } else if (((nw >> shift) & 1ull) != 0ull) {
      acc.uadd(p, true);
    }
  }
  xyzz_out[idx] = acc;
}

// Normalize xyzz_in[0..count-1] -> affine_out[0..count-1] using 64-thread
// Montgomery batch inversion (SPPARK uniform warp reciprocal).
__global__ static void cubbs_bbs_xyzz_to_plain_affine_batch64_kernel(
    const cubbs_sppark_bucket_t *xyzz_in,
    cubbs_sppark_plain_affine_t *affine_out,
    size_t count) {
  constexpr uint32_t kThreads = 64u;
  __shared__ __align__(16) uint8_t scan_a_raw[kThreads * sizeof(fp_t)];
  __shared__ __align__(16) uint8_t scan_b_raw[kThreads * sizeof(fp_t)];
  auto *scan_a = reinterpret_cast<fp_t *>(scan_a_raw);
  auto *scan_b = reinterpret_cast<fp_t *>(scan_b_raw);

  const uint32_t tid = threadIdx.x;
  const size_t idx = static_cast<size_t>(blockIdx.x) * kThreads + tid;
  const bool valid = idx < count;

  cubbs_sppark_bucket_t pt;
  pt.inf();
  bool finite = false;
  fp_t zzz = fp_t::one();
  if (valid) {
    pt = xyzz_in[idx];
    finite = !pt.is_inf();
    if (finite) {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      zzz = coords[2];
    }
  }
  scan_b[tid] = zzz;
  __syncthreads();

  fp_t prod = fp_t::one();
  if (tid == 0u) {
    for (uint32_t i = 0u; i < kThreads; ++i) {
      scan_a[i] = prod;
      prod *= scan_b[i];
    }
  }
  __syncthreads();

  if (tid < 32u) {
    const fp_t inv_in = (tid == 0u) ? prod : fp_t::one();
    const fp_t inv_all = inv_in.reciprocal();
    if (tid == 0u) {
      fp_t inv = inv_all;
      for (int i = static_cast<int>(kThreads) - 1; i >= 0; --i) {
        const fp_t inv_i = inv * scan_a[i];
        inv *= scan_b[i];
        scan_a[i] = inv_i;
      }
    }
  }
  __syncthreads();

  if (valid) {
    if (!finite) {
      fp_t zero;
      zero.zero();
      affine_out[idx] = cubbs_sppark_plain_affine_t{zero, zero};
    } else {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      const fp_t inv_zzz = scan_a[tid];
      const fp_t affine_y = inv_zzz * coords[1];
      const fp_t inv_z = inv_zzz * coords[3];
      fp_t affine_x = inv_z * inv_z;
      affine_x *= coords[0];
      affine_out[idx] = cubbs_sppark_plain_affine_t{affine_x, affine_y};
    }
  }
}

#endif // CUBBS_BBS_GPU_G1_DECOMPRESS_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
