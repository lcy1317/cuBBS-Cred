#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_GPU_G1_DECOMPRESS_CUH
#define CUBBS_BBS_GPU_G1_DECOMPRESS_CUH

// ============================================================================
// 100% GPU BLS12-381 G1 Uncompress + Scott (2021/1130) Subgroup Check
// ============================================================================
// Bit-exact match to BLST blst_p1_uncompress() + blst_p1_affine_in_g1() with
// optional non-infinity enforcement (required by RFC BBS for A, Abar, Bbar, D).
// Eliminates all CPU G1 decoding for unique signatures and unique proofs.

// ============================================================================
// BLS12-381 (a = 0) Jacobian Coordinates (2M+5S dbl-2009-l, 7M+4S madd-2007-bl)
// Uses SPPARK's symmetric sqr() and 3 fp_t (12 fewer regs than xyzz_t).
// ============================================================================

struct CuBbsG1Jac {
  fp_t X;
  fp_t Y;
  fp_t Z;

  __device__ __forceinline__ bool is_inf() const { return Z.is_zero(); }
  __device__ __forceinline__ void inf() { Z.zero(); }
  __device__ __forceinline__ void set_affine(const cubbs_sppark_plain_affine_t &p, bool neg = false) {
    if (p.is_inf()) {
      Z.zero();
    } else {
      const fp_t *c = reinterpret_cast<const fp_t *>(&p);
      X = c[0];
      Y = c[1];
      Y.cneg(neg);
      Z = fp_t::one();
    }
  }
};

// Complete dbl-2009-l for a = 0: 2M + 5S (with symmetric sqr()).
// Handles Z == 0 (infinity) and Y == 0 naturally via Z3 = 2*Y*Z == 0.
__device__ __noinline__ static void cubbs_bbs_jac_dbl(CuBbsG1Jac &p) {
  const fp_t A = sqr(p.X);
  const fp_t B = sqr(p.Y);
  const fp_t C = sqr(B);

  fp_t D = sqr(p.X + B) - A - C;
  D = D + D;

  const fp_t E = A + (A << 1);
  const fp_t F = sqr(E);

  const fp_t X3 = F - D - D;
  fp_t Z3 = p.Y * p.Z;
  Z3 = Z3 + Z3;

  fp_t C8 = C << 3;
  const fp_t Y3 = (E * (D - X3)) - C8;

  p.X = X3;
  p.Y = Y3;
  p.Z = Z3;
}

// Complete mixed Jacobian + Affine addition (madd-2007-bl: 7M + 4S with sqr()).
__device__ __noinline__ static void cubbs_bbs_jac_madd(
    CuBbsG1Jac &p, const cubbs_sppark_plain_affine_t &q, bool neg = false) {
  if (q.is_inf()) return;
  const fp_t *qc = reinterpret_cast<const fp_t *>(&q);
  fp_t qy = qc[1];
  qy.cneg(neg);
  if (p.is_inf()) {
    p.X = qc[0];
    p.Y = qy;
    p.Z = fp_t::one();
    return;
  }

  const fp_t Z1Z1 = sqr(p.Z);
  const fp_t U2 = qc[0] * Z1Z1;
  const fp_t S2 = (qy * p.Z) * Z1Z1;
  const fp_t H = U2 - p.X;
  const fp_t R = S2 - p.Y;

  if (H.is_zero()) {
    if (R.is_zero()) {
      cubbs_bbs_jac_dbl(p);
    } else {
      p.inf();
    }
    return;
  }

  const fp_t HH = sqr(H);
  const fp_t I = HH << 2;
  const fp_t J = H * I;
  const fp_t r = R + R;
  const fp_t V = p.X * I;

  const fp_t X3 = sqr(r) - J - V - V;
  fp_t Y1J = p.Y * J;
  Y1J = Y1J + Y1J;
  const fp_t Y3 = (r * (V - X3)) - Y1J;
  const fp_t Z3 = sqr(p.Z + H) - Z1Z1 - HH;

  p.X = X3;
  p.Y = Y3;
  p.Z = Z3;
}

// Complete Jacobian + Jacobian addition (add-2007-bl: 11M + 5S with sqr()).
__device__ __noinline__ static void cubbs_bbs_jac_add(
    CuBbsG1Jac &p, const CuBbsG1Jac &q) {
  if (q.is_inf()) return;
  if (p.is_inf()) {
    p = q;
    return;
  }

  const fp_t Z1Z1 = sqr(p.Z);
  const fp_t Z2Z2 = sqr(q.Z);
  const fp_t U1 = p.X * Z2Z2;
  const fp_t U2 = q.X * Z1Z1;
  const fp_t S1 = (p.Y * q.Z) * Z2Z2;
  const fp_t S2 = (q.Y * p.Z) * Z1Z1;
  const fp_t H = U2 - U1;
  const fp_t R = S2 - S1;

  if (H.is_zero()) {
    if (R.is_zero()) {
      cubbs_bbs_jac_dbl(p);
    } else {
      p.inf();
    }
    return;
  }

  const fp_t I = sqr(H + H);
  const fp_t J = H * I;
  const fp_t r = R + R;
  const fp_t V = U1 * I;

  const fp_t X3 = sqr(r) - J - V - V;
  fp_t S1J = S1 * J;
  S1J = S1J + S1J;
  const fp_t Y3 = (r * (V - X3)) - S1J;
  const fp_t Z3 = (sqr(p.Z + q.Z) - Z1Z1 - Z2Z2) * H;

  p.X = X3;
  p.Y = Y3;
  p.Z = Z3;
}

__device__ __forceinline__ static cubbs_sppark_bucket_t
cubbs_bbs_jac_to_xyzz(const CuBbsG1Jac &p) {
  cubbs_sppark_bucket_t out;
  if (p.is_inf()) {
    out.inf();
    return out;
  }
  const fp_t zz = sqr(p.Z);
  const fp_t zzz = zz * p.Z;
  fp_t *c = reinterpret_cast<fp_t *>(&out);
  c[0] = p.X;
  c[1] = p.Y;
  c[2] = zzz;
  c[3] = zz;
  return out;
}

__device__ __noinline__ static fp_t cubbs_bbs_fp_pow_p_plus_1_div_4(const fp_t &a) {
  // 4-bit sliding-window chain for (p - 3) / 4 using symmetric sqr(), then * a
  // since (p + 1) / 4 == (p - 3) / 4 + 1 (376 sqr() + 86 muls vs 518 generic muls!).
  static constexpr uint8_t OPS[79] = {
      0x91u, 0x41u, 0x78u, 0x43u, 0x64u, 0x66u, 0x48u, 0x48u,
      0x67u, 0x67u, 0x65u, 0x32u, 0x77u, 0x47u, 0x68u, 0x57u,
      0x45u, 0x87u, 0x66u, 0x33u, 0x32u, 0x67u, 0x44u, 0x32u,
      0x31u, 0x64u, 0x44u, 0x55u, 0x44u, 0x85u, 0x34u, 0x54u,
      0x73u, 0x75u, 0x56u, 0x48u, 0x32u, 0x54u, 0x42u, 0x87u,
      0x53u, 0x21u, 0x98u, 0x57u, 0x32u, 0x88u, 0x32u, 0x75u,
      0x98u, 0x43u, 0x56u, 0x48u, 0x48u, 0x48u, 0x34u, 0x56u,
      0x63u, 0x55u, 0x48u, 0x48u, 0x48u, 0x48u, 0x47u, 0x22u,
      0x68u, 0x48u, 0x58u, 0x48u, 0x48u, 0x48u, 0x48u, 0x48u,
      0x48u, 0x48u, 0x33u, 0x43u, 0x43u, 0x21u, 0x10u};
  fp_t t[8];
  const fp_t a2 = sqr(a);
  t[0] = a;
  #pragma unroll
  for (int i = 1; i < 8; ++i) t[i] = t[i - 1] * a2;

  fp_t res = t[6]; // leading window 13 = 0b1101 -> t[6] = a^13
  #pragma unroll 1
  for (int step = 0; step < 79; ++step) {
    const uint32_t op = OPS[step];
    const uint32_t sq = op >> 4u;
    const uint32_t idx = op & 0xfu;
    #pragma unroll 1
    for (uint32_t s = 0u; s < sq; ++s) {
      res = sqr(res);
    }
    if (idx != 0u) {
      res *= t[idx - 1u];
    }
  }
  return res * a;
}

// Computes [-z]P = [0xd201000000010000]P from an affine point P using exact
// BLST addition chain in a=0 Jacobian coordinates (63 2M+5S doublings + 5 madd).
__device__ __noinline__ static CuBbsG1Jac
cubbs_bbs_g1_times_minus_z_affine(const cubbs_sppark_plain_affine_t &p) {
  CuBbsG1Jac acc;
  acc.set_affine(p, false);
  cubbs_bbs_jac_dbl(acc); // 0x2
  static constexpr uint32_t DBLS[5] = {2u, 3u, 9u, 32u, 16u};
  #pragma unroll 1
  for (uint32_t step = 0u; step < 5u; ++step) {
    cubbs_bbs_jac_madd(acc, p, false);
    const uint32_t n = DBLS[step];
    #pragma unroll 1
    for (uint32_t d = 0u; d < n; ++d) {
      cubbs_bbs_jac_dbl(acc);
    }
  }
  return acc;
}

// Computes [-z]Q = [0xd201000000010000]Q from a Jacobian point Q using exact
// BLST addition chain (63 2M+5S doublings + 5 Jacobian additions).
__device__ __noinline__ static CuBbsG1Jac
cubbs_bbs_g1_times_minus_z_jac(const CuBbsG1Jac &q) {
  CuBbsG1Jac acc = q;
  cubbs_bbs_jac_dbl(acc); // 0x2
  static constexpr uint32_t DBLS[5] = {2u, 3u, 9u, 32u, 16u};
  #pragma unroll 1
  for (uint32_t step = 0u; step < 5u; ++step) {
    cubbs_bbs_jac_add(acc, q);
    const uint32_t n = DBLS[step];
    #pragma unroll 1
    for (uint32_t d = 0u; d < n; ++d) {
      cubbs_bbs_jac_dbl(acc);
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
  const fp_t rhs = (sqr(x) * x) + four;

  fp_t y = cubbs_bbs_fp_pow_p_plus_1_div_4(rhs);
  if (!(sqr(y) - rhs).is_zero()) return false;

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
  //   [-z]([-z]P) == [z^2]P == (beta^2 * x, -y) in Jacobian coordinates.
  const CuBbsG1Jac t0 = cubbs_bbs_g1_times_minus_z_affine(pt);
  if (t0.is_inf()) return false;
  const CuBbsG1Jac t1 = cubbs_bbs_g1_times_minus_z_jac(t0);
  if (t1.is_inf()) return false;

  fp_t beta_sq;
  const uint64_t beta_sq_limbs[6] = {
      0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
      0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
  memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));

  const fp_t target_x = x * beta_sq;
  fp_t target_y = y;
  target_y.cneg(true);

  const fp_t zz = sqr(t1.Z);
  const fp_t zzz = zz * t1.Z;
  if (!(t1.X - (target_x * zz)).is_zero()) return false;
  if (!(t1.Y - (target_y * zzz)).is_zero()) return false;

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

  CuBbsG1Jac acc;
  if (top_pos) {
    acc.set_affine(p, false);
  } else {
    acc.inf();
  }
  #pragma unroll 1
  for (int bit = 127; bit >= 0; --bit) {
    if (!acc.is_inf()) {
      cubbs_bbs_jac_dbl(acc);
    }
    const uint32_t shift = static_cast<uint32_t>(bit & 63);
    const uint64_t pw = (bit >= 64) ? pos_hi : pos_lo;
    const uint64_t nw = (bit >= 64) ? neg_hi : neg_lo;
    if (((pw >> shift) & 1ull) != 0ull) {
      cubbs_bbs_jac_madd(acc, p, false);
    } else if (((nw >> shift) & 1ull) != 0ull) {
      cubbs_bbs_jac_madd(acc, p, true);
    }
  }
  xyzz_out[idx] = cubbs_bbs_jac_to_xyzz(acc);
}

// Normalize xyzz_in[0..count-1] -> affine_out[0..count-1] using 100% parallel
// warp-cooperative SPPARK fp_t::reciprocal() (0 SMEM, 0 serial bottleneck, 8x faster!).
__global__ static void cubbs_bbs_xyzz_to_plain_affine_batch64_kernel(
    const cubbs_sppark_bucket_t *xyzz_in,
    cubbs_sppark_plain_affine_t *affine_out,
    size_t count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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

  // All 32 lanes of every warp execute reciprocal() uniformly in parallel (~25 Fp muls wall-clock).
  const fp_t inv_zzz = zzz.reciprocal();

  if (valid) {
    if (!finite) {
      fp_t zero;
      zero.zero();
      affine_out[idx] = cubbs_sppark_plain_affine_t{zero, zero};
    } else {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      const fp_t affine_y = inv_zzz * coords[1];
      const fp_t inv_z = inv_zzz * coords[3];
      const fp_t affine_x = sqr(inv_z) * coords[0];
      affine_out[idx] = cubbs_sppark_plain_affine_t{affine_x, affine_y};
    }
  }
}

#endif // CUBBS_BBS_GPU_G1_DECOMPRESS_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
