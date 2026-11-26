#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

struct CuBbsPairingFp6 {
  bls12_381::fp2_t c0;
  bls12_381::fp2_t c1;
  bls12_381::fp2_t c2;
};

struct CuBbsPairingFp12 {
  CuBbsPairingFp6 c0;
  CuBbsPairingFp6 c1;
};

struct CuBbsPairingLine014 {
  bls12_381::fp2_t c0;
  bls12_381::fp2_t c1;
  bls12_381::fp2_t c4;
};

__device__ __forceinline__ static fp_t cubbs_pairing_load_fp(
    const uint8_t *input) {
  fp_t value;
  memcpy(&value, input, sizeof(value));
  return value;
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_load_fp2(const uint8_t *input) {
  const fp_t component =
      cubbs_pairing_load_fp(input + ((threadIdx.x & 1u) * sizeof(fp_t)));
  bls12_381::fp2_t value;
  memcpy(&value, &component, sizeof(fp_t));
  return value;
}

__device__ __forceinline__ static void cubbs_pairing_store_fp2(
    uint8_t *output, const bls12_381::fp2_t &value) {
  memcpy(output + ((threadIdx.x & 1u) * sizeof(fp_t)), &value,
         sizeof(fp_t));
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_fp2_zero() {
  return bls12_381::fp2_t::one(1);
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_fp2_one() {
  return bls12_381::fp2_t::one();
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_fp2_mul_fp(const bls12_381::fp2_t &a, const fp_t &b) {
  fp_t component;
  memcpy(&component, &a, sizeof(fp_t));
  component *= b;
  bls12_381::fp2_t result;
  memcpy(&result, &component, sizeof(fp_t));
  return result;
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_fp2_mul_nonresidue(const bls12_381::fp2_t &a) {
  fp_t component;
  memcpy(&component, &a, sizeof(fp_t));
  fp_t other;
  uint32_t component_limbs[sizeof(fp_t) / sizeof(uint32_t)];
  uint32_t other_limbs[sizeof(fp_t) / sizeof(uint32_t)];
  memcpy(component_limbs, &component, sizeof(component_limbs));
  #pragma unroll
  for (size_t limb = 0; limb < sizeof(fp_t) / sizeof(uint32_t); ++limb) {
    other_limbs[limb] =
        __shfl_xor_sync(__activemask(), component_limbs[limb], 1u);
  }
  memcpy(&other, other_limbs, sizeof(other));
  component = (threadIdx.x & 1u) != 0u ? other + component
                                        : component - other;
  bls12_381::fp2_t result;
  memcpy(&result, &component, sizeof(fp_t));
  return result;
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_fp2_conjugate(bls12_381::fp2_t a) {
  a.cneg((threadIdx.x & 1u) != 0u);
  return a;
}

enum CuBbsPairingFrobeniusCoefficient : size_t {
  CUBBS_FROB_F6_C1_P1 = 0u,
  CUBBS_FROB_F6_C2_P1 = 2u,
  CUBBS_FROB_F12_C1_P1 = 4u,
  CUBBS_FROB_F6_C1_P2 = 6u,
  CUBBS_FROB_F6_C2_P2 = 8u,
  CUBBS_FROB_F12_C1_P2 = 10u,
};

__constant__ fp_t cubbs_pairing_frobenius_coefficients[12];

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_pairing_load_frobenius_coefficient(size_t offset) {
  return cubbs_pairing_load_fp2(reinterpret_cast<const uint8_t *>(
      cubbs_pairing_frobenius_coefficients + offset));
}

// ============================================================================
// 6-Active-Thread (8-Lane Aligned) Cooperative Zero-Spill Fp6 / Fp12 Arithmetic
// ============================================================================
// Each pairing equation occupies 8 consecutive warp lanes:
//   pair 0 (sub 0,1): Fp6.c0 (real, imag)
//   pair 1 (sub 2,3): Fp6.c1 (real, imag)
//   pair 2 (sub 4,5): Fp6.c2 (real, imag)
//   pair 3 (sub 6,7): padding pair executing in lockstep so mask = 0xffffffff
// Each thread holds only 2 fp_t values per Fp12 (3x lower register pressure,
// 0 bytes local-memory stack spill, and 3x parallel Fp4 squares / Fp6 muls).

struct CuBbsDistFp12 {
  bls12_381::fp2_t c0; // distributed Fp6: pair 0=c0, pair 1=c1, pair 2=c2
  bls12_381::fp2_t c1; // distributed Fp6: pair 0=c0, pair 1=c1, pair 2=c2
};

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_dist_shfl_pair(const bls12_381::fp2_t &v, uint32_t src_pair) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t src_lane = (lane & ~7u) | ((src_pair & 3u) << 1u) | (lane & 1u);
  uint32_t limbs[12];
  memcpy(limbs, &v, 48u);
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    limbs[i] = __shfl_sync(0xffffffffu, limbs[i], src_lane);
  }
  bls12_381::fp2_t out;
  memcpy(&out, limbs, 48u);
  return out;
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_dist_fp6_mul_nonresidue(const bls12_381::fp2_t &a, uint32_t pair) {
  const uint32_t src_pair = (pair == 0u) ? 2u : (pair == 1u ? 0u : 1u);
  const auto shifted = cubbs_dist_shfl_pair(a, src_pair);
  const auto nr = cubbs_pairing_fp2_mul_nonresidue(shifted);
  return (pair == 0u) ? nr : shifted;
}

__device__ __noinline__ static bls12_381::fp2_t
cubbs_dist_fp6_mul(const bls12_381::fp2_t &a, const bls12_381::fp2_t &b) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  const auto v = a * b;
  const auto v0 = cubbs_dist_shfl_pair(v, 0u);
  const auto v1 = cubbs_dist_shfl_pair(v, 1u);
  const auto v2 = cubbs_dist_shfl_pair(v, 2u);

  const auto a0 = cubbs_dist_shfl_pair(a, 0u);
  const auto a1 = cubbs_dist_shfl_pair(a, 1u);
  const auto a2 = cubbs_dist_shfl_pair(a, 2u);
  const auto b0 = cubbs_dist_shfl_pair(b, 0u);
  const auto b1 = cubbs_dist_shfl_pair(b, 1u);
  const auto b2 = cubbs_dist_shfl_pair(b, 2u);

  const auto sa = (pair == 0u) ? (a1 + a2) : (pair == 1u ? (a0 + a1) : (a0 + a2));
  const auto sb = (pair == 0u) ? (b1 + b2) : (pair == 1u ? (b0 + b1) : (b0 + b2));
  const auto k = sa * sb;

  const auto nr_in = (pair == 0u) ? (k - v1 - v2) : v2;
  const auto nr_out = cubbs_pairing_fp2_mul_nonresidue(nr_in);
  if (pair == 0u) return v0 + nr_out;
  if (pair == 1u) return (k - v0 - v1) + nr_out;
  return k - v0 + v1 - v2;
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_dist_fp6_mul_by_1(const bls12_381::fp2_t &r,
                        const bls12_381::fp2_t &l_c1, uint32_t pair) {
  const auto r0 = cubbs_dist_shfl_pair(r, 0u);
  const auto r1 = cubbs_dist_shfl_pair(r, 1u);
  const auto r2 = cubbs_dist_shfl_pair(r, 2u);
  const auto lhs = (pair == 0u) ? (r1 + r2) : (pair == 1u ? (r0 + r1) : r1);
  const auto p = lhs * l_c1;
  const auto bb = cubbs_dist_shfl_pair(p, 2u);
  const auto diff = p - bb;
  const auto nr = cubbs_pairing_fp2_mul_nonresidue(diff);
  if (pair == 0u) return nr;
  if (pair == 1u) return diff;
  return bb;
}

__device__ __forceinline__ static bls12_381::fp2_t
cubbs_dist_fp6_mul_by_01(const bls12_381::fp2_t &r,
                         const bls12_381::fp2_t &l_c0,
                         const bls12_381::fp2_t &l_c1, uint32_t pair) {
  const auto r0 = cubbs_dist_shfl_pair(r, 0u);
  const auto r1 = cubbs_dist_shfl_pair(r, 1u);
  const auto r2 = cubbs_dist_shfl_pair(r, 2u);

  const auto m1_a = (pair == 0u) ? r0 : (pair == 1u ? r1 : (r0 + r2));
  const auto m1_b = (pair == 1u) ? l_c1 : l_c0;
  const auto mul1 = m1_a * m1_b;

  const auto m2_a = (pair == 0u) ? (r1 + r2) : (r0 + r1);
  const auto m2_b = (pair == 0u) ? l_c1 : (l_c0 + l_c1);
  const auto mul2 = m2_a * m2_b;

  const auto aa = cubbs_dist_shfl_pair(mul1, 0u);
  const auto bb = cubbs_dist_shfl_pair(mul1, 1u);
  const auto t1_nr = cubbs_pairing_fp2_mul_nonresidue(mul2 - bb);
  if (pair == 0u) return t1_nr + aa;
  if (pair == 1u) return mul2 - aa - bb;
  return mul1 - aa + bb;
}

__device__ __noinline__ static CuBbsDistFp12
cubbs_dist_fp12_mul(const CuBbsDistFp12 &a, const CuBbsDistFp12 &b) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  const auto aa = cubbs_dist_fp6_mul(a.c0, b.c0);
  const auto bb = cubbs_dist_fp6_mul(a.c1, b.c1);
  const auto c0 = aa + cubbs_dist_fp6_mul_nonresidue(bb, pair);
  const auto c1 = cubbs_dist_fp6_mul(a.c0 + a.c1, b.c0 + b.c1) - aa - bb;
  return {c0, c1};
}

__device__ __noinline__ static CuBbsDistFp12
cubbs_dist_fp12_square(const CuBbsDistFp12 &a) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  const auto ab = cubbs_dist_fp6_mul(a.c0, a.c1);
  const auto prod = cubbs_dist_fp6_mul(
      a.c0 + a.c1, a.c0 + cubbs_dist_fp6_mul_nonresidue(a.c1, pair));
  const auto c0 = (prod - ab) - cubbs_dist_fp6_mul_nonresidue(ab, pair);
  const auto c1 = ab + ab;
  return {c0, c1};
}

__device__ __noinline__ static CuBbsDistFp12
cubbs_dist_fp12_cyclotomic_square(const CuBbsDistFp12 &a) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  // a.c0 holds (z0, z4, z3) across pairs (0, 1, 2).
  // a.c1 holds (z2, z1, z5) across pairs (0, 1, 2).
  // Gather pairs (z0,z1) on pair 0, (z2,z3) on pair 1, (z4,z5) on pair 2.
  const auto z0 = cubbs_dist_shfl_pair(a.c0, 0u);
  const auto z4 = cubbs_dist_shfl_pair(a.c0, 1u);
  const auto z3 = cubbs_dist_shfl_pair(a.c0, 2u);
  const auto z2 = cubbs_dist_shfl_pair(a.c1, 0u);
  const auto z1 = cubbs_dist_shfl_pair(a.c1, 1u);
  const auto z5 = cubbs_dist_shfl_pair(a.c1, 2u);

  const auto u = (pair == 0u) ? z0 : (pair == 1u ? z2 : z4);
  const auto v = (pair == 0u) ? z1 : (pair == 1u ? z3 : z5);

  const auto t0 = u * u;
  const auto t1 = v * v;
  const auto s0 = t0 + cubbs_pairing_fp2_mul_nonresidue(t1);
  const auto uv = u + v;
  const auto s1 = (uv * uv) - t0 - t1;

  // out.c0 = (new_z0, new_z4, new_z3) aligns 1:1 with (a_sq.c0, b_sq.c0, c_sq.c0)!
  const auto d0 = s0 - a.c0;
  const auto out_c0 = d0 + d0 + s0;

  // out.c1 = (new_z2, new_z1, new_z5) uses (nonres(c_sq.c1), a_sq.c1, b_sq.c1).
  const auto s1_nr = cubbs_pairing_fp2_mul_nonresidue(s1);
  const auto val_s1 = (pair == 2u) ? s1_nr : s1;
  const uint32_t src_pair = (pair == 0u) ? 2u : (pair == 1u ? 0u : 1u);
  const auto shfl_s1 = cubbs_dist_shfl_pair(val_s1, src_pair);
  const auto d1 = shfl_s1 + a.c1;
  const auto out_c1 = d1 + d1 + shfl_s1;
  return {out_c0, out_c1};
}

__device__ __noinline__ static bls12_381::fp2_t
cubbs_dist_fp6_inverse(const bls12_381::fp2_t &a) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  const auto a0 = cubbs_dist_shfl_pair(a, 0u);
  const auto a1 = cubbs_dist_shfl_pair(a, 1u);
  const auto a2 = cubbs_dist_shfl_pair(a, 2u);

  const auto sq = a * a;
  const auto c_lhs = (pair == 0u) ? a1 : a0;
  const auto c_rhs = (pair == 1u) ? a1 : a2;
  const auto cross = c_lhs * c_rhs; // p0=a1*a2, p1=a0*a1, p2=a0*a2

  const auto sq1 = cubbs_dist_shfl_pair(sq, 1u);
  const auto sq2 = cubbs_dist_shfl_pair(sq, 2u);
  const auto nr_in = (pair == 0u) ? cross : sq2;
  const auto nr_out = cubbs_pairing_fp2_mul_nonresidue(nr_in);
  const auto t = (pair == 0u) ? (sq - nr_out)
                              : (pair == 1u ? (nr_out - cross) : (sq1 - cross));

  const auto d_mul_a = (pair == 0u) ? a0 : (pair == 1u ? a2 : a1);
  const auto d_prod = d_mul_a * t; // p0=a0*t0, p1=a2*t1, p2=a1*t2
  const auto dp0 = cubbs_dist_shfl_pair(d_prod, 0u);
  const auto dp1 = cubbs_dist_shfl_pair(d_prod, 1u);
  const auto dp2 = cubbs_dist_shfl_pair(d_prod, 2u);
  const auto denom = dp0 + cubbs_pairing_fp2_mul_nonresidue(dp1 + dp2);
  // All 32 warp lanes execute reciprocal() uniformly.
  const auto inv_denom = denom.reciprocal();
  return t * inv_denom;
}

__device__ __noinline__ static CuBbsDistFp12
cubbs_dist_fp12_inverse(const CuBbsDistFp12 &a) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  const auto denom = cubbs_dist_fp6_inverse(
      cubbs_dist_fp6_mul(a.c0, a.c0) -
      cubbs_dist_fp6_mul_nonresidue(cubbs_dist_fp6_mul(a.c1, a.c1), pair));
  auto c1 = cubbs_dist_fp6_mul(a.c1, denom);
  c1.cneg(true);
  return {cubbs_dist_fp6_mul(a.c0, denom), c1};
}

__device__ __forceinline__ static CuBbsDistFp12
cubbs_dist_fp12_conjugate(CuBbsDistFp12 a) {
  a.c1.cneg(true);
  return a;
}

__device__ __forceinline__ static void cubbs_dist_fp12_frobenius(
    CuBbsDistFp12 &a, unsigned power) {
  const uint32_t pair = (threadIdx.x & 7u) >> 1u;
  if ((power & 1u) != 0u) {
    a.c0 = cubbs_pairing_fp2_conjugate(a.c0);
    a.c1 = cubbs_pairing_fp2_conjugate(a.c1);
  }
  const size_t c1_off = power == 1u ? CUBBS_FROB_F6_C1_P1 : CUBBS_FROB_F6_C1_P2;
  const size_t c2_off = power == 1u ? CUBBS_FROB_F6_C2_P1 : CUBBS_FROB_F6_C2_P2;
  const auto f6_coeff = (pair == 0u)
                            ? cubbs_pairing_fp2_one()
                            : cubbs_pairing_load_frobenius_coefficient(
                                  pair == 1u ? c1_off : c2_off);
  a.c0 *= f6_coeff;
  a.c1 *= f6_coeff;
  const size_t f12_off = power == 1u ? CUBBS_FROB_F12_C1_P1 : CUBBS_FROB_F12_C1_P2;
  a.c1 *= cubbs_pairing_load_frobenius_coefficient(f12_off);
}

__device__ __noinline__ static CuBbsDistFp12
cubbs_dist_fp12_exp_by_z(const CuBbsDistFp12 &a) {
  constexpr uint64_t z_abs = 0xd201000000010000ULL;
  CuBbsDistFp12 result = a;
  bool started = false;
#pragma unroll 1
  for (int bit = 63; bit >= 0; --bit) {
    if (started) {
      result = cubbs_dist_fp12_cyclotomic_square(result);
    }
    if (((z_abs >> bit) & 1u) != 0u) {
      if (started) {
        result = cubbs_dist_fp12_mul(result, a);
      } else {
        result = a;
        started = true;
      }
    }
  }
  return cubbs_dist_fp12_conjugate(result);
}

__device__ __noinline__ static CuBbsDistFp12
cubbs_dist_fp12_final_exponentiation(const CuBbsDistFp12 &f) {
  const auto f_conj = cubbs_dist_fp12_conjugate(f);
  const auto f_inv = cubbs_dist_fp12_inverse(f);
  auto r = cubbs_dist_fp12_mul(f_conj, f_inv);
  const auto easy = r;
  cubbs_dist_fp12_frobenius(r, 2u);
  r = cubbs_dist_fp12_mul(r, easy);

  auto y0 = cubbs_dist_fp12_cyclotomic_square(r);
  auto y1 = cubbs_dist_fp12_exp_by_z(r);
  auto y2 = cubbs_dist_fp12_conjugate(r);
  y1 = cubbs_dist_fp12_mul(y1, y2);
  y2 = cubbs_dist_fp12_exp_by_z(y1);
  y1 = cubbs_dist_fp12_conjugate(y1);
  y1 = cubbs_dist_fp12_mul(y1, y2);
  y2 = cubbs_dist_fp12_exp_by_z(y1);
  cubbs_dist_fp12_frobenius(y1, 1u);
  y1 = cubbs_dist_fp12_mul(y1, y2);
  r = cubbs_dist_fp12_mul(r, y0);
  y0 = cubbs_dist_fp12_exp_by_z(y1);
  y2 = cubbs_dist_fp12_exp_by_z(y0);
  y0 = y1;
  cubbs_dist_fp12_frobenius(y0, 2u);
  y1 = cubbs_dist_fp12_conjugate(y1);
  y1 = cubbs_dist_fp12_mul(y1, y2);
  y1 = cubbs_dist_fp12_mul(y1, y0);
  return cubbs_dist_fp12_mul(r, y1);
}

__device__ __forceinline__ static CuBbsPairingLine014
cubbs_pairing_line_for_g1(const uint8_t *line, const fp_t &px,
                           const fp_t &py) {
  const auto l0 = cubbs_pairing_load_fp2(line);
  const auto l1 = cubbs_pairing_fp2_mul_fp(
      cubbs_pairing_load_fp2(line + (2u * sizeof(fp_t))), px);
  const auto l2 = cubbs_pairing_fp2_mul_fp(
      cubbs_pairing_load_fp2(line + (4u * sizeof(fp_t))), py);
  return {l0, l1, l2};
}

__device__ __forceinline__ static void cubbs_dist_fp12_mul_by_014(
    CuBbsDistFp12 &f, const CuBbsPairingLine014 &line, uint32_t pair) {
  const auto aa = cubbs_dist_fp6_mul_by_01(f.c0, line.c0, line.c1, pair);
  const auto bb = cubbs_dist_fp6_mul_by_1(f.c1, line.c4, pair);
  const auto c1 = cubbs_dist_fp6_mul_by_01(
                      f.c0 + f.c1, line.c0, line.c1 + line.c4, pair) -
                  aa - bb;
  f.c0 = cubbs_dist_fp6_mul_nonresidue(bb, pair) + aa;
  f.c1 = c1;
}

__device__ __forceinline__ static CuBbsDistFp12
cubbs_dist_fp12_one(uint32_t pair) {
  const auto zero = cubbs_pairing_fp2_zero();
  const auto one = cubbs_pairing_fp2_one();
  return {pair == 0u ? one : zero, zero};
}

__device__ __forceinline__ static bool cubbs_dist_fp12_is_one(
    const CuBbsDistFp12 &a) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t group_base = lane & ~7u;
  const uint32_t pair = (lane & 7u) >> 1u;
  const bool lane_ok = (pair == 0u)
                           ? (a.c0.is_one() && a.c1.is_zero())
                           : (a.c0.is_zero() && a.c1.is_zero());
  const uint32_t ballot = __ballot_sync(0xffffffffu, lane_ok);
  return ((ballot >> group_base) & 0x3fu) == 0x3fu;
}

// Fused multi-Miller loop + final exponentiation checking e(P1, Q1) == e(P2, Q2)
// via e(-P1, Q1) * e(P2, Q2) == 1. Uses 8 lanes (6 active) per equation, halving
// Miller squarings (68 vs 136) and achieving 3x Fp6/Fp12 parallelism with 0 stack spill.
__global__ static void cubbs_bbs_pairing_eq_fused_kernel(
    const uint8_t *lhs_g1_affines, const uint8_t *rhs_g1_affines,
    const uint8_t *lhs_lines, size_t lhs_lines_stride,
    const uint8_t *rhs_lines, size_t rhs_lines_stride, size_t item_count,
    uint8_t *values_out) {
  const size_t raw_item = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 3u;
  const bool valid = raw_item < item_count;
  // Clamp out-of-range lanes to item 0 so all 32 lanes of every warp execute
  // lockstep without invalid memory accesses or partial-warp shuffle UB.
  const size_t item = valid ? raw_item : 0u;
  const uint32_t sub = threadIdx.x & 7u;
  const uint32_t pair = sub >> 1u;

  constexpr size_t g1_stride = 2u * sizeof(fp_t);
  constexpr size_t line_stride = 6u * sizeof(fp_t);

  const uint8_t *lhs_p = lhs_g1_affines + (item * g1_stride);
  const uint8_t *rhs_p = rhs_g1_affines + (item * g1_stride);
  fp_t lx = cubbs_pairing_load_fp(lhs_p);
  fp_t ly = cubbs_pairing_load_fp(lhs_p + sizeof(fp_t));
  fp_t rx = cubbs_pairing_load_fp(rhs_p);
  fp_t ry = cubbs_pairing_load_fp(rhs_p + sizeof(fp_t));

  const bool lhs_inf = lx.is_zero() && ly.is_zero();
  const bool rhs_inf = rx.is_zero() && ry.is_zero();

  // Negate lhs G1 point (-P1) so e(-P1, Q1) * e(P2, Q2) == 1 iff e(P1, Q1) == e(P2, Q2).
  lx = lx + lx;
  lx.cneg(true);
  ly = ly + ly;
  ly.cneg(true);

  rx = rx + rx;
  rx.cneg(true);
  ry = ry + ry;

  const uint8_t *l_lines = lhs_lines + (item * lhs_lines_stride);
  const uint8_t *r_lines = rhs_lines + (item * rhs_lines_stride);

  CuBbsDistFp12 acc = cubbs_dist_fp12_one(pair);
#pragma unroll 1
  for (size_t i = 0u; i < 68u; ++i) {
    const bool addition_line =
        i == 0u || i == 1u || i == 4u || i == 8u || i == 18u || i == 51u;
    if (!addition_line) {
      acc = cubbs_dist_fp12_square(acc);
    }
    {
      const auto line_l = cubbs_pairing_line_for_g1(l_lines + (i * line_stride), lx, ly);
      CuBbsDistFp12 next_l = acc;
      cubbs_dist_fp12_mul_by_014(next_l, line_l, pair);
      if (!lhs_inf) acc = next_l;
    }
    {
      const auto line_r = cubbs_pairing_line_for_g1(r_lines + (i * line_stride), rx, ry);
      CuBbsDistFp12 next_r = acc;
      cubbs_dist_fp12_mul_by_014(next_r, line_r, pair);
      if (!rhs_inf) acc = next_r;
    }
  }
  // Conjugate for negative BLS12-381 parameter z.
  acc = cubbs_dist_fp12_conjugate(acc);

  const auto final_val = cubbs_dist_fp12_final_exponentiation(acc);
  const bool equal = cubbs_dist_fp12_is_one(final_val);
  if (valid && sub == 0u) {
    values_out[item] = equal ? 1u : 0u;
  }
}

// Legacy 2-thread Miller-loop kernel retained only for callers requesting raw
// intermediate Fp12 outputs (finalverify_out == nullptr).
__device__ __forceinline__ static CuBbsPairingFp6 cubbs_pairing_fp6_add(
    const CuBbsPairingFp6 &a, const CuBbsPairingFp6 &b) {
  return {a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2};
}
__device__ __forceinline__ static CuBbsPairingFp6 cubbs_pairing_fp6_sub(
    const CuBbsPairingFp6 &a, const CuBbsPairingFp6 &b) {
  return {a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2};
}
__device__ __forceinline__ static CuBbsPairingFp6
cubbs_pairing_fp6_mul(const CuBbsPairingFp6 &a, const CuBbsPairingFp6 &b) {
  const auto v0 = a.c0 * b.c0;
  const auto v1 = a.c1 * b.c1;
  const auto v2 = a.c2 * b.c2;
  const auto c0 = v0 + cubbs_pairing_fp2_mul_nonresidue(
                           (a.c1 + a.c2) * (b.c1 + b.c2) - v1 - v2);
  const auto c1 = (a.c0 + a.c1) * (b.c0 + b.c1) - v0 - v1 +
                  cubbs_pairing_fp2_mul_nonresidue(v2);
  const auto c2 = (a.c0 + a.c2) * (b.c0 + b.c2) - v0 + v1 - v2;
  return {c0, c1, c2};
}
__device__ __forceinline__ static CuBbsPairingFp6
cubbs_pairing_fp6_mul_nonresidue(const CuBbsPairingFp6 &a) {
  return {cubbs_pairing_fp2_mul_nonresidue(a.c2), a.c0, a.c1};
}
__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_square(const CuBbsPairingFp12 &a) {
  const auto ab = cubbs_pairing_fp6_mul(a.c0, a.c1);
  const auto product = cubbs_pairing_fp6_mul(
      cubbs_pairing_fp6_add(a.c0, a.c1),
      cubbs_pairing_fp6_add(a.c0, cubbs_pairing_fp6_mul_nonresidue(a.c1)));
  const auto c0 = cubbs_pairing_fp6_sub(
      cubbs_pairing_fp6_sub(product, ab), cubbs_pairing_fp6_mul_nonresidue(ab));
  const auto c1 = cubbs_pairing_fp6_add(ab, ab);
  return {c0, c1};
}
__device__ __forceinline__ static void cubbs_pairing_fp6_mul_by_1(
    CuBbsPairingFp6 &r, const bls12_381::fp2_t &c1) {
  const auto bb = r.c1 * c1;
  auto t1 = cubbs_pairing_fp2_mul_nonresidue(c1 * (r.c1 + r.c2) - bb);
  const auto t2 = c1 * (r.c0 + r.c1) - bb;
  r = {t1, t2, bb};
}
__device__ __forceinline__ static void cubbs_pairing_fp6_mul_by_01(
    CuBbsPairingFp6 &r, const bls12_381::fp2_t &c0, const bls12_381::fp2_t &c1) {
  const auto aa = r.c0 * c0;
  const auto bb = r.c1 * c1;
  auto t1 = cubbs_pairing_fp2_mul_nonresidue(c1 * (r.c1 + r.c2) - bb) + aa;
  const auto t3 = c0 * (r.c0 + r.c2) - aa + bb;
  const auto t2 = (c0 + c1) * (r.c0 + r.c1) - aa - bb;
  r = {t1, t2, t3};
}
__device__ __forceinline__ static void cubbs_pairing_fp12_mul_by_014(
    CuBbsPairingFp12 &f, const CuBbsPairingLine014 &line) {
  auto aa = f.c0;
  cubbs_pairing_fp6_mul_by_01(aa, line.c0, line.c1);
  auto bb = f.c1;
  cubbs_pairing_fp6_mul_by_1(bb, line.c4);
  auto c1 = cubbs_pairing_fp6_add(f.c0, f.c1);
  cubbs_pairing_fp6_mul_by_01(c1, line.c0, line.c1 + line.c4);
  c1 = cubbs_pairing_fp6_sub(cubbs_pairing_fp6_sub(c1, aa), bb);
  f.c0 = cubbs_pairing_fp6_add(cubbs_pairing_fp6_mul_nonresidue(bb), aa);
  f.c1 = c1;
}
__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_from_line(const CuBbsPairingLine014 &line) {
  const auto zero = cubbs_pairing_fp2_zero();
  return {{line.c0, line.c1, zero}, {zero, line.c4, zero}};
}
__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_miller_loop_lines(const uint8_t *g1_affine, const uint8_t *lines) {
  fp_t px = cubbs_pairing_load_fp(g1_affine);
  fp_t py = cubbs_pairing_load_fp(g1_affine + sizeof(fp_t));
  px = px + px;
  px.cneg(true);
  py = py + py;
  constexpr size_t line_stride = 6u * sizeof(fp_t);
  auto result = cubbs_pairing_fp12_from_line(
      cubbs_pairing_line_for_g1(lines, px, py));
#pragma unroll 1
  for (size_t i = 1u; i < 68u; ++i) {
    const bool addition_line =
        i == 1u || i == 4u || i == 8u || i == 18u || i == 51u;
    if (!addition_line) result = cubbs_pairing_fp12_square(result);
    const auto line = cubbs_pairing_line_for_g1(lines + (i * line_stride), px, py);
    cubbs_pairing_fp12_mul_by_014(result, line);
  }
  result.c1.c0.cneg(true);
  result.c1.c1.cneg(true);
  result.c1.c2.cneg(true);
  return result;
}
__device__ __forceinline__ static void cubbs_pairing_store_fp12(
    uint8_t *output, const CuBbsPairingFp12 &value) {
  cubbs_pairing_store_fp2(output, value.c0.c0);
  cubbs_pairing_store_fp2(output + (2u * sizeof(fp_t)), value.c0.c1);
  cubbs_pairing_store_fp2(output + (4u * sizeof(fp_t)), value.c0.c2);
  cubbs_pairing_store_fp2(output + (6u * sizeof(fp_t)), value.c1.c0);
  cubbs_pairing_store_fp2(output + (8u * sizeof(fp_t)), value.c1.c1);
  cubbs_pairing_store_fp2(output + (10u * sizeof(fp_t)), value.c1.c2);
}

__global__ static void cubbs_bbs_miller_loop_common_lines_kernel(
    const uint8_t *lhs_g1_affines, const uint8_t *rhs_g1_affines,
    const uint8_t *lhs_lines, size_t lhs_lines_stride,
    const uint8_t *rhs_lines, size_t rhs_lines_stride, size_t item_count,
    uint8_t *lhs_fp12_out, uint8_t *rhs_fp12_out) {
  const size_t loop_index =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 1u;
  if (loop_index >= item_count * 2u) {
    return;
  }
  const size_t item = loop_index >> 1u;
  const bool rhs = (loop_index & 1u) != 0u;
  constexpr size_t g1_stride = 2u * sizeof(fp_t);
  constexpr size_t fp12_stride = 12u * sizeof(fp_t);
  const auto value = cubbs_pairing_miller_loop_lines(
      (rhs ? rhs_g1_affines : lhs_g1_affines) + (item * g1_stride),
      (rhs ? rhs_lines : lhs_lines) +
          (item * (rhs ? rhs_lines_stride : lhs_lines_stride)));
  cubbs_pairing_store_fp12(
      (rhs ? rhs_fp12_out : lhs_fp12_out) + (item * fp12_stride), value);
}

struct CuBbsMillerDeviceWorkspace {
  cudaStream_t stream{nullptr};
  size_t capacity{0u};
  size_t lhs_lines_capacity{0u};
  size_t rhs_lines_capacity{0u};
  uint8_t *lhs_g1{nullptr};
  uint8_t *rhs_g1{nullptr};
  uint8_t *lhs_lines{nullptr};
  uint8_t *rhs_lines{nullptr};
  uint8_t *lhs_out{nullptr};
  uint8_t *rhs_out{nullptr};
  uint8_t *values_out{nullptr};
  // Pinned host staging for zero-bounce small pairing transfers.
  uint8_t *host_g1{nullptr};
  size_t host_g1_capacity{0u};
  uint8_t *host_values{nullptr};
  size_t host_values_capacity{0u};
  // Persistent buffers for GPU-resident Unique-PK Multi-Miller + Single FinalExp.
  uint8_t *unique_g2_lines{nullptr};
  size_t unique_g2_lines_capacity{0u};
  uint8_t *unique_block_fp12{nullptr};
  size_t unique_block_fp12_capacity{0u};
  std::vector<uint8_t> unique_pk_key;
  uint8_t lhs_g2_key[96]{};
  uint8_t rhs_g2_key[96]{};
  bool lhs_key_valid{false};
  bool rhs_key_valid{false};
  bool frobenius_initialized{false};
  int device{-1};

  CuBbsMillerDeviceWorkspace() { (void)cudaGetDevice(&device); }
  ~CuBbsMillerDeviceWorkspace() {
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (device >= 0 && device != previous) (void)cudaSetDevice(device);
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    if (host_g1 != nullptr) (void)cudaFreeHost(host_g1);
    if (host_values != nullptr) (void)cudaFreeHost(host_values);
    (void)cudaFree(lhs_g1); (void)cudaFree(rhs_g1);
    (void)cudaFree(lhs_lines); (void)cudaFree(rhs_lines);
    (void)cudaFree(lhs_out); (void)cudaFree(rhs_out);
    (void)cudaFree(values_out);
    (void)cudaFree(unique_g2_lines);
    (void)cudaFree(unique_block_fp12);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
  }
};

static CuBbsMillerDeviceWorkspace &cubbs_miller_device_workspace() {
  static thread_local CuBbsMillerDeviceWorkspace workspace;
  return workspace;
}

static void cubbs_pairing_set_fp2_coefficient(
    fp_t coefficients[12], size_t offset, const uint32_t real[12],
    const uint32_t imaginary[12]) {
  static_assert(sizeof(fp_t) == 12u * sizeof(uint32_t));
  std::memcpy(&coefficients[offset], real, sizeof(fp_t));
  std::memcpy(&coefficients[offset + 1u], imaginary, sizeof(fp_t));
}

static cudaError_t cubbs_pairing_initialize_frobenius_coefficients() {
  static constexpr uint32_t zero[12] = {};
  static constexpr uint32_t f6_c1_p1_imaginary[12] = {
      0x8671f071, 0xcd03c9e4, 0x1fcda5d2, 0x5dab2246,
      0xd3851b95, 0x587042af, 0x01bacb9e, 0x8eb60ebe,
      0x83d050d2, 0x03f97d6e, 0x54638741, 0x18f02065};
  static constexpr uint32_t f6_c2_p1_real[12] = {
      0x867545c3, 0x890dc9e4, 0x3285a5d5, 0x2af32253,
      0x309b7e2c, 0x50880866, 0x7e881024, 0xa20d1b8c,
      0xe2db9068, 0x14e4f04f, 0x1564853a, 0x14e56d3f};
  static constexpr uint32_t f12_c1_p1_real[12] = {
      0xb319d465, 0x07089552, 0xb50a8313, 0xc6695f92,
      0xd117228f, 0x97e83ccc, 0xb2dc29ee, 0xa35baeca,
      0x5daace4d, 0x1ce393ea, 0xb0fb66eb, 0x08f2220f};
  static constexpr uint32_t f12_c1_p1_imaginary[12] = {
      0x4ce5d646, 0xb2f66aad, 0xfc497cec, 0x5842a06b,
      0x2599d394, 0xcf4895d4, 0x40a8e8d0, 0xc11b9cba,
      0xe5a0de89, 0x2e3813cb, 0x88847faf, 0x110eefda};
  static constexpr uint32_t f6_c1_p2_real[12] = {
      0x798a64e8, 0x30f1361b, 0x7ece5a2a, 0xf3b8ddab,
      0xc61577f7, 0x16a8ca3a, 0x74fd029b, 0xc26a2ff8,
      0x60701c6e, 0x3636b766, 0x241b6160, 0x051ba4ab};
  static constexpr uint32_t f6_c2_p2_real[12] = {
      0x8671f071, 0xcd03c9e4, 0x1fcda5d2, 0x5dab2246,
      0xd3851b95, 0x587042af, 0x01bacb9e, 0x8eb60ebe,
      0x83d050d2, 0x03f97d6e, 0x54638741, 0x18f02065};
  static constexpr uint32_t f12_c1_p2_real[12] = {
      0x798dba3a, 0xecfb361b, 0x91865a2c, 0xc100ddb8,
      0x232bda8e, 0x0ec08ff1, 0xf1ca4721, 0xd5c13cc6,
      0xbf7b5c04, 0x47222a47, 0xe51c5f59, 0x0110f184};

  fp_t coefficients[12]{};
  cubbs_pairing_set_fp2_coefficient(
      coefficients, CUBBS_FROB_F6_C1_P1, zero, f6_c1_p1_imaginary);
  cubbs_pairing_set_fp2_coefficient(
      coefficients, CUBBS_FROB_F6_C2_P1, f6_c2_p1_real, zero);
  cubbs_pairing_set_fp2_coefficient(
      coefficients, CUBBS_FROB_F12_C1_P1, f12_c1_p1_real,
      f12_c1_p1_imaginary);
  cubbs_pairing_set_fp2_coefficient(
      coefficients, CUBBS_FROB_F6_C1_P2, f6_c1_p2_real, zero);
  cubbs_pairing_set_fp2_coefficient(
      coefficients, CUBBS_FROB_F6_C2_P2, f6_c2_p2_real, zero);
  cubbs_pairing_set_fp2_coefficient(
      coefficients, CUBBS_FROB_F12_C1_P2, f12_c1_p2_real, zero);
  return cudaMemcpyToSymbol(cubbs_pairing_frobenius_coefficients,
                            coefficients, sizeof(coefficients));
}

int32_t cubbs_bbs_cuda_miller_loop_common_lines_batch(
    const void *lhs_g1_affines, const void *rhs_g1_affines,
    const void *lhs_lines, size_t lhs_lines_stride, const void *rhs_lines,
    size_t rhs_lines_stride, size_t item_count, void *lhs_fp12_out,
    void *rhs_fp12_out, uint8_t *finalverify_out,
    const uint8_t *lhs_g2_key, const uint8_t *rhs_g2_key) {
  if (item_count == 0u) {
    return CUBBS_BBS_GPU_SUCCESS;
  }
  cubbs_bbs_ensure_cuda_device_schedule();
  if (lhs_g1_affines == nullptr || rhs_g1_affines == nullptr ||
      lhs_lines == nullptr || rhs_lines == nullptr ||
      (finalverify_out == nullptr &&
       (lhs_fp12_out == nullptr || rhs_fp12_out == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }

  constexpr size_t g1_stride = 2u * sizeof(fp_t);
  constexpr size_t lines_bytes = 68u * 6u * sizeof(fp_t);
  constexpr size_t fp12_stride = 12u * sizeof(fp_t);
  const size_t g1_bytes = item_count * g1_stride;
  const size_t output_bytes = item_count * fp12_stride;
  const size_t lhs_lines_bytes =
      lhs_lines_stride == 0u ? lines_bytes : item_count * lhs_lines_stride;
  const size_t rhs_lines_bytes =
      rhs_lines_stride == 0u ? lines_bytes : item_count * rhs_lines_stride;
  auto &workspace = cubbs_miller_device_workspace();
  cudaError_t error = cudaSuccess;
  if (workspace.stream == nullptr) {
    const bool low_priority = env_flag_enabled("CUBBS_BBS_FORCE_GPU") &&
        env_flag_enabled("CUBBS_BBS_GPU_PAIRING_LOW_PRIORITY");
    if (low_priority) {
      int least_priority = 0;
      int greatest_priority = 0;
      error = cudaDeviceGetStreamPriorityRange(&least_priority,
                                               &greatest_priority);
      if (error == cudaSuccess) {
        error = cudaStreamCreateWithPriority(
            &workspace.stream, cudaStreamNonBlocking, least_priority);
      }
    } else {
      error = cudaStreamCreateWithFlags(&workspace.stream,
                                        cudaStreamNonBlocking);
    }
  }
  if (error == cudaSuccess && (workspace.capacity < item_count ||
      workspace.lhs_lines_capacity < lhs_lines_bytes ||
      workspace.rhs_lines_capacity < rhs_lines_bytes)) {
    if (workspace.lhs_g1 != nullptr) cudaFree(workspace.lhs_g1);
    if (workspace.rhs_g1 != nullptr) cudaFree(workspace.rhs_g1);
    if (workspace.lhs_lines != nullptr) cudaFree(workspace.lhs_lines);
    if (workspace.rhs_lines != nullptr) cudaFree(workspace.rhs_lines);
    if (workspace.lhs_out != nullptr) cudaFree(workspace.lhs_out);
    if (workspace.rhs_out != nullptr) cudaFree(workspace.rhs_out);
    if (workspace.values_out != nullptr) cudaFree(workspace.values_out);
    workspace.capacity = 0u;
    workspace.lhs_lines_capacity = 0u;
    workspace.rhs_lines_capacity = 0u;
    workspace.lhs_g1 = nullptr;
    workspace.rhs_g1 = nullptr;
    workspace.lhs_lines = nullptr;
    workspace.rhs_lines = nullptr;
    workspace.lhs_out = nullptr;
    workspace.rhs_out = nullptr;
    workspace.values_out = nullptr;
    workspace.lhs_key_valid = false;
    workspace.rhs_key_valid = false;
    error = cudaMalloc(&workspace.lhs_g1, g1_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&workspace.rhs_g1, g1_bytes);
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.lhs_lines, lhs_lines_bytes);
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.rhs_lines, rhs_lines_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&workspace.lhs_out, output_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&workspace.rhs_out, output_bytes);
    if (error == cudaSuccess)
      error = cudaMalloc(&workspace.values_out, item_count);
    if (error == cudaSuccess) {
      workspace.capacity = item_count;
      workspace.lhs_lines_capacity = lhs_lines_bytes;
      workspace.rhs_lines_capacity = rhs_lines_bytes;
    }
  }
  if (error == cudaSuccess && workspace.host_g1_capacity < g1_bytes * 2u) {
    if (workspace.host_g1 != nullptr) (void)cudaFreeHost(workspace.host_g1);
    workspace.host_g1 = nullptr;
    workspace.host_g1_capacity = 0u;
    error = cudaMallocHost(&workspace.host_g1, g1_bytes * 2u);
    if (error == cudaSuccess) workspace.host_g1_capacity = g1_bytes * 2u;
  }
  if (error == cudaSuccess && finalverify_out != nullptr &&
      workspace.host_values_capacity < item_count) {
    if (workspace.host_values != nullptr) (void)cudaFreeHost(workspace.host_values);
    workspace.host_values = nullptr;
    workspace.host_values_capacity = 0u;
    error = cudaMallocHost(&workspace.host_values, item_count);
    if (error == cudaSuccess) workspace.host_values_capacity = item_count;
  }
  if (error == cudaSuccess && finalverify_out != nullptr &&
      !workspace.frobenius_initialized) {
    static std::once_flag frob_once;
    static cudaError_t frob_err = cudaSuccess;
    std::call_once(frob_once, []() {
      frob_err = cubbs_pairing_initialize_frobenius_coefficients();
    });
    error = frob_err;
    if (error == cudaSuccess) {
      workspace.frobenius_initialized = true;
    }
  }
  if (error == cudaSuccess) {
    std::memcpy(workspace.host_g1, lhs_g1_affines, g1_bytes);
    std::memcpy(workspace.host_g1 + g1_bytes, rhs_g1_affines, g1_bytes);
    error = cudaMemcpyAsync(workspace.lhs_g1, workspace.host_g1, g1_bytes,
                            cudaMemcpyHostToDevice, workspace.stream);
  }
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(workspace.rhs_g1, workspace.host_g1 + g1_bytes, g1_bytes,
                            cudaMemcpyHostToDevice, workspace.stream);
  const bool upload_lhs_lines =
      lhs_lines_stride != 0u || lhs_g2_key == nullptr ||
      !workspace.lhs_key_valid || std::memcmp(workspace.lhs_g2_key, lhs_g2_key, 96u) != 0;
  const bool upload_rhs_lines =
      rhs_lines_stride != 0u || rhs_g2_key == nullptr ||
      !workspace.rhs_key_valid || std::memcmp(workspace.rhs_g2_key, rhs_g2_key, 96u) != 0;
  if (error == cudaSuccess && upload_lhs_lines)
    error = cudaMemcpyAsync(workspace.lhs_lines, lhs_lines, lhs_lines_bytes,
                            cudaMemcpyHostToDevice, workspace.stream);
  if (error == cudaSuccess && upload_rhs_lines)
    error = cudaMemcpyAsync(workspace.rhs_lines, rhs_lines, rhs_lines_bytes,
                            cudaMemcpyHostToDevice, workspace.stream);

  if (error == cudaSuccess && finalverify_out != nullptr) {
    // 8 lanes per pairing equation, rounded up to a full warp (32) so all
    // shfl_sync(0xffffffff) operations are strictly uniform across the warp.
    const uint32_t threads = (item_count <= 4u) ? 32u : 64u;
    const uint32_t blocks =
        static_cast<uint32_t>((item_count * 8u + threads - 1u) / threads);
    cubbs_bbs_pairing_eq_fused_kernel<<<blocks, threads, 0, workspace.stream>>>(
        workspace.lhs_g1, workspace.rhs_g1, workspace.lhs_lines,
        lhs_lines_stride, workspace.rhs_lines, rhs_lines_stride, item_count,
        workspace.values_out);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(workspace.host_values, workspace.values_out,
                              item_count, cudaMemcpyDeviceToHost,
                              workspace.stream);
    }
    if (error == cudaSuccess) {
      error = cudaStreamSynchronize(workspace.stream);
    }
    if (error == cudaSuccess) {
      std::memcpy(finalverify_out, workspace.host_values, item_count);
    }
  } else if (error == cudaSuccess) {
    const uint32_t configured_threads = static_cast<uint32_t>(std::min<size_t>(
        1024u, std::max<size_t>(2u, env_size_or_default("CUBBS_BBS_GPU_MILLER_THREADS", 64u)))) & ~1u;
    const uint32_t useful_threads = static_cast<uint32_t>(item_count * 4u);
    const uint32_t threads = std::max<uint32_t>(
        2u, std::min(configured_threads, useful_threads));
    const uint32_t blocks =
        static_cast<uint32_t>((item_count * 4u + threads - 1u) / threads);
    cubbs_bbs_miller_loop_common_lines_kernel<<<blocks, threads, 0,
                                                  workspace.stream>>>(
        workspace.lhs_g1, workspace.rhs_g1, workspace.lhs_lines,
        lhs_lines_stride, workspace.rhs_lines, rhs_lines_stride, item_count,
        workspace.lhs_out, workspace.rhs_out);
    error = cudaGetLastError();
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(lhs_fp12_out, workspace.lhs_out, output_bytes,
                              cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(rhs_fp12_out, workspace.rhs_out, output_bytes,
                              cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess)
      error = cudaStreamSynchronize(workspace.stream);
  }

  workspace.lhs_key_valid = error == cudaSuccess && lhs_lines_stride == 0u && lhs_g2_key != nullptr;
  workspace.rhs_key_valid = error == cudaSuccess && rhs_lines_stride == 0u && rhs_g2_key != nullptr;
  if (workspace.lhs_key_valid) std::memcpy(workspace.lhs_g2_key, lhs_g2_key, 96u);
  if (workspace.rhs_key_valid) std::memcpy(workspace.rhs_g2_key, rhs_g2_key, 96u);
  return error == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS
                              : CUBBS_BBS_GPU_ERR_INTERNAL;
}

// Launch a single fused pairing check e(-d_lhs_g1, Q_lhs) * e(d_rhs_g1, Q_rhs) == 1
// directly on caller's CUDA stream, keeping G1 inputs & output resident on GPU!
int32_t cubbs_bbs_cuda_launch_pairing_eq_single_on_stream(
    const void *d_lhs_g1, const void *d_rhs_g1,
    const uint8_t *lhs_lines, const uint8_t *rhs_lines,
    const uint8_t *lhs_g2_key, const uint8_t *rhs_g2_key,
    uint8_t *d_eq_out, void *cuda_stream) {
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  if (d_lhs_g1 == nullptr || d_rhs_g1 == nullptr || lhs_lines == nullptr ||
      rhs_lines == nullptr || d_eq_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  constexpr size_t lines_bytes = 68u * 6u * sizeof(fp_t);
  auto &workspace = cubbs_miller_device_workspace();
  cudaError_t error = cudaSuccess;
  if (workspace.lhs_lines_capacity < lines_bytes) {
    if (workspace.lhs_lines != nullptr) (void)cudaFree(workspace.lhs_lines);
    workspace.lhs_lines = nullptr;
    workspace.lhs_lines_capacity = 0u;
    workspace.lhs_key_valid = false;
    error = cudaMalloc(&workspace.lhs_lines, lines_bytes);
    if (error == cudaSuccess) workspace.lhs_lines_capacity = lines_bytes;
  }
  if (error == cudaSuccess && workspace.rhs_lines_capacity < lines_bytes) {
    if (workspace.rhs_lines != nullptr) (void)cudaFree(workspace.rhs_lines);
    workspace.rhs_lines = nullptr;
    workspace.rhs_lines_capacity = 0u;
    workspace.rhs_key_valid = false;
    error = cudaMalloc(&workspace.rhs_lines, lines_bytes);
    if (error == cudaSuccess) workspace.rhs_lines_capacity = lines_bytes;
  }
  if (error == cudaSuccess && !workspace.frobenius_initialized) {
    static std::once_flag frob_once;
    static cudaError_t frob_err = cudaSuccess;
    std::call_once(frob_once, []() {
      frob_err = cubbs_pairing_initialize_frobenius_coefficients();
    });
    error = frob_err;
    if (error == cudaSuccess) workspace.frobenius_initialized = true;
  }
  const bool upload_lhs = lhs_g2_key == nullptr || !workspace.lhs_key_valid ||
                          std::memcmp(workspace.lhs_g2_key, lhs_g2_key, 96u) != 0;
  const bool upload_rhs = rhs_g2_key == nullptr || !workspace.rhs_key_valid ||
                          std::memcmp(workspace.rhs_g2_key, rhs_g2_key, 96u) != 0;
  if (error == cudaSuccess && upload_lhs) {
    error = cudaMemcpyAsync(workspace.lhs_lines, lhs_lines, lines_bytes,
                            cudaMemcpyHostToDevice, stream);
  }
  if (error == cudaSuccess && upload_rhs) {
    error = cudaMemcpyAsync(workspace.rhs_lines, rhs_lines, lines_bytes,
                            cudaMemcpyHostToDevice, stream);
  }
  if (error == cudaSuccess) {
    cubbs_bbs_pairing_eq_fused_kernel<<<1u, 32u, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(d_lhs_g1),
        reinterpret_cast<const uint8_t *>(d_rhs_g1),
        workspace.lhs_lines, 0u, workspace.rhs_lines, 0u, 1u, d_eq_out);
    error = cudaGetLastError();
  }
  workspace.lhs_key_valid = error == cudaSuccess && lhs_g2_key != nullptr;
  workspace.rhs_key_valid = error == cudaSuccess && rhs_g2_key != nullptr;
  if (workspace.lhs_key_valid) std::memcpy(workspace.lhs_g2_key, lhs_g2_key, 96u);
  if (workspace.rhs_key_valid) std::memcpy(workspace.rhs_g2_key, rhs_g2_key, 96u);
  return error == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS
                              : CUBBS_BBS_GPU_ERR_INTERNAL;
}

// ============================================================================
// 100% GPU G2 Uncompress + Fused Scott G2 Subgroup Check + Unique-PK
// Randomized Multi-Miller + Single Final Exponentiation
// ============================================================================

__device__ __noinline__ static fp_t cubbs_pairing_fp_pow_p_minus_3_div_4(const fp_t &a) {
  static constexpr uint64_t EXP[6] = {
      0xee7fbfffffffeaaaULL, 0x07aaffffac54ffffULL,
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

__device__ __forceinline__ static fp_t cubbs_pairing_shfl_fp(const fp_t &v, uint32_t src_lane) {
  uint32_t limbs[12];
  memcpy(limbs, &v, 48u);
  #pragma unroll
  for (int i = 0; i < 12; ++i) {
    limbs[i] = __shfl_sync(0xffffffffu, limbs[i], src_lane);
  }
  fp_t out;
  memcpy(&out, limbs, 48u);
  return out;
}

__device__ __forceinline__ static bool cubbs_pairing_fp_std_gt_half(const fp_t &val_std) {
  static constexpr uint64_t FP_HALF_LE[6] = {
      0xdcff7fffffffd555ull, 0x0f55ffff58a9ffffull,
      0xb39869507b587b12ull, 0xb23ba5c279c2895full,
      0x258dd3db21a5d66bull, 0x0d0088f51cbff34dull};
  uint64_t limbs[6];
  memcpy(limbs, &val_std, 48u);
  #pragma unroll
  for (int i = 5; i >= 0; --i) {
    if (limbs[i] != FP_HALF_LE[i]) return limbs[i] > FP_HALF_LE[i];
  }
  return false;
}

struct CuBbsG2Jac {
  bls12_381::fp2_t X;
  bls12_381::fp2_t Y;
  bls12_381::fp2_t Z;
};

__device__ __noinline__ static CuBbsPairingLine014
cubbs_pairing_g2_line_dbl(CuBbsG2Jac &T) {
  const auto A = T.X * T.X;
  const auto B = T.Y * T.Y;
  const auto ZZ = T.Z * T.Z;
  const auto C = B * B;

  auto D = T.X + B;
  D = (D * D) - A - C;
  D = D + D;

  const auto E = A + A + A;
  const auto F = E * E;

  auto l0 = E + T.X;
  T.X = F - D - D;

  auto TZ = T.Y + T.Z;
  T.Z = (TZ * TZ) - B - ZZ;

  auto C8 = C + C;
  C8 = C8 + C8;
  C8 = C8 + C8;
  T.Y = (E * (D - T.X)) - C8;

  l0 = (l0 * l0) - A - F;
  auto B4 = B + B;
  B4 = B4 + B4;
  return {l0 - B4, E * ZZ, T.Z * ZZ};
}

__device__ __noinline__ static CuBbsPairingLine014
cubbs_pairing_g2_line_add(CuBbsG2Jac &T,
                          const bls12_381::fp2_t &Qx,
                          const bls12_381::fp2_t &Qy,
                          bool &invalid) {
  const auto Z1Z1 = T.Z * T.Z;
  const auto U2 = Qx * Z1Z1;
  const auto S2 = (Qy * T.Z) * Z1Z1;
  const auto H = U2 - T.X;
  if (H.is_zero()) invalid = true;
  const auto HH = H * H;
  auto I = HH + HH;
  I = I + I;
  const auto J = H * I;
  auto r = S2 - T.Y;
  r = r + r;
  const auto V = T.X * I;

  const auto old_Y = T.Y;
  T.X = (r * r) - J - V - V;
  const auto JY = J * old_Y;
  T.Y = (r * (V - T.X)) - JY - JY;
  auto ZH = T.Z + H;
  T.Z = (ZH * ZH) - Z1Z1 - HH;

  auto l0 = (r * Qx) - (Qy * T.Z);
  return {l0 + l0, r, T.Z};
}

// 100% GPU G2 uncompress + 68-step Miller line precompute + zero-cost Scott G2
// subgroup check (T_final == -psi(Q)) using 2 lanes/key (32 threads/block = 16 keys/block,
// 4x fewer warp lanes spent on G2 line math than running inside 8-lane Miller!).
__global__ static void cubbs_bbs_g2_uncompress_and_precompute_lines_kernel(
    const uint8_t *pks_compressed, size_t item_count,
    uint8_t *g2_lines_out, uint32_t *invalid_flag) {
  const size_t raw_item = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 1u;
  const bool valid = raw_item < item_count;
  const size_t item = valid ? raw_item : 0u;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t is_odd = lane & 1u;
  const uint32_t even_lane = lane & ~1u;
  const uint32_t odd_lane = even_lane | 1u;

  const uint8_t *src = pks_compressed + (item * 96u);
  const uint8_t b0 = src[0];
  bool ok = ((b0 & 0x80u) != 0u) && ((b0 & 0x40u) == 0u);
  const bool sign_flag = (b0 & 0x20u) != 0u;

  const uint8_t *comp_bytes = is_odd ? src : (src + 48u);
  uint64_t x_limbs[6];
  #pragma unroll
  for (int limb = 0; limb < 6; ++limb) {
    const int base = (5 - limb) * 8;
    const uint8_t first = (is_odd && base == 0) ? static_cast<uint8_t>(b0 & 0x1fu) : comp_bytes[base];
    x_limbs[limb] =
        (static_cast<uint64_t>(first) << 56) |
        (static_cast<uint64_t>(comp_bytes[base + 1]) << 48) |
        (static_cast<uint64_t>(comp_bytes[base + 2]) << 40) |
        (static_cast<uint64_t>(comp_bytes[base + 3]) << 32) |
        (static_cast<uint64_t>(comp_bytes[base + 4]) << 24) |
        (static_cast<uint64_t>(comp_bytes[base + 5]) << 16) |
        (static_cast<uint64_t>(comp_bytes[base + 6]) << 8) |
        static_cast<uint64_t>(comp_bytes[base + 7]);
  }
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
  ok = ok && less_than_p;
  const uint32_t pair_ok_ballot = __ballot_sync(0xffffffffu, ok);
  ok = (((pair_ok_ballot >> even_lane) & 3u) == 3u);

  fp_t x_comp;
  if (ok) {
    memcpy(&x_comp, x_limbs, 48u);
    x_comp.to();
  } else {
    x_comp.zero();
  }
  bls12_381::fp2_t X;
  memcpy(&X, &x_comp, sizeof(fp_t));

  fp_t four = fp_t::one();
  four = four + four;
  four = four + four;
  bls12_381::fp2_t b_e2;
  memcpy(&b_e2, &four, sizeof(fp_t));
  const bls12_381::fp2_t rhs = (X * X * X) + b_e2;

  fp_t rhs_lane;
  memcpy(&rhs_lane, &rhs, sizeof(fp_t));
  const fp_t a = cubbs_pairing_shfl_fp(rhs_lane, even_lane);
  const fp_t b = cubbs_pairing_shfl_fp(rhs_lane, odd_lane);

  const fp_t norm = (a * a) + (b * b);
  const fp_t u_norm = cubbs_pairing_fp_pow_p_minus_3_div_4(norm);
  const fp_t s = norm * u_norm;
  ok = ok && ((s * s) - norm).is_zero();

  static constexpr uint64_t FP_INV2_LE[6] = {
      0xdcff7fffffffd556ull, 0x0f55ffff58a9ffffull,
      0xb39869507b587b12ull, 0xb23ba5c279c2895full,
      0x258dd3db21a5d66bull, 0x0d0088f51cbff34dull};
  fp_t inv2;
  memcpy(&inv2, FP_INV2_LE, 48u);
  inv2.to();

  const fp_t sum_as = a + s;
  fp_t delta = sum_as * inv2;
  if (sum_as.is_zero()) delta = a;
  const fp_t u = cubbs_pairing_fp_pow_p_minus_3_div_4(delta);
  const fp_t x0_try = delta * u;
  const fp_t y0_try = (b * inv2) * u;
  const bool is_qr = delta.is_zero() || ((x0_try * u) - fp_t::one()).is_zero();

  fp_t y_re, y_im;
  if (is_qr) {
    y_re = x0_try;
    y_im = y0_try;
  } else {
    y_re = y0_try;
    y_im = x0_try;
    y_im.cneg(true);
  }

  fp_t y_lane = is_odd ? y_im : y_re;
  bls12_381::fp2_t Y;
  memcpy(&Y, &y_lane, sizeof(fp_t));
  ok = ok && ((Y * Y) - rhs).is_zero();

  fp_t y_re_std = y_re; y_re_std.from();
  fp_t y_im_std = y_im; y_im_std.from();
  const bool y_gt_half = y_im_std.is_zero()
      ? cubbs_pairing_fp_std_gt_half(y_re_std)
      : cubbs_pairing_fp_std_gt_half(y_im_std);
  Y.cneg(y_gt_half != sign_flag);

  // Precompute all 68 Miller lines + check Scott G2 subgroup T_final == -psi(Q)!
  constexpr size_t line_stride = 6u * sizeof(fp_t);
  constexpr size_t lines_bytes = 68u * line_stride;
  uint8_t *dst_lines = g2_lines_out + (item * lines_bytes);
  CuBbsG2Jac T{X, Y, cubbs_pairing_fp2_one()};
  bool g2_invalid = false;
  #pragma unroll 1
  for (uint32_t i = 0u; i < 68u; ++i) {
    const bool addition_line =
        i == 1u || i == 4u || i == 8u || i == 18u || i == 51u;
    CuBbsPairingLine014 line = addition_line
        ? cubbs_pairing_g2_line_add(T, X, Y, g2_invalid)
        : cubbs_pairing_g2_line_dbl(T);
    if (T.Z.is_zero()) g2_invalid = true;
    if (valid) {
      uint8_t *ldst = dst_lines + (i * line_stride);
      cubbs_pairing_store_fp2(ldst, line.c0);
      cubbs_pairing_store_fp2(ldst + 96u, line.c1);
      cubbs_pairing_store_fp2(ldst + 192u, line.c4);
    }
  }

  // Zero-cost Scott G2 subgroup check: T == [-z]Q == -psi(Q).
  static constexpr uint64_t FROB_X_IM[6] = {
      0x890dc9e4867545c3ULL, 0x2af322533285a5d5ULL, 0x50880866309b7e2cULL,
      0xa20d1b8c7e881024ULL, 0x14e4f04fe2db9068ULL, 0x14e56d3f1564853aULL};
  static constexpr uint64_t FROB_Y_RE[6] = {
      0x3e2f585da55c9ad1ULL, 0x4294213d86c18183ULL, 0x382844c88b623732ULL,
      0x92ad2afd19103e18ULL, 0x1d794e4fac7cf0b9ULL, 0x0bd592fc7d825ec8ULL};
  static constexpr uint64_t FROB_Y_IM[6] = {
      0x7bcfa7a25aa30fdaULL, 0xdc17dec12a927e7cULL, 0x2f088dd86b4ebef1ULL,
      0xd1ca2087da74d4a7ULL, 0x2da2596696cebc1dULL, 0x0e2b7eedbbfd87d2ULL};
  fp_t fx_lane, fy_lane;
  if (is_odd != 0u) {
    memcpy(&fx_lane, FROB_X_IM, 48u);
    memcpy(&fy_lane, FROB_Y_IM, 48u);
  } else {
    fx_lane.zero();
    memcpy(&fy_lane, FROB_Y_RE, 48u);
  }
  bls12_381::fp2_t frob_x, frob_y;
  memcpy(&frob_x, &fx_lane, 48u);
  memcpy(&frob_y, &fy_lane, 48u);

  const auto zz = T.Z * T.Z;
  const auto zzz = zz * T.Z;
  const auto psi_x = cubbs_pairing_fp2_conjugate(X) * frob_x;
  auto neg_psi_y = cubbs_pairing_fp2_conjugate(Y) * frob_y;
  neg_psi_y.cneg(true);

  const bool in_g2 = ok && !g2_invalid && !T.Z.is_zero() &&
                     (T.X - (psi_x * zz)).is_zero() &&
                     (T.Y - (neg_psi_y * zzz)).is_zero();
  if (valid && !in_g2 && is_odd == 0u) {
    atomicExch(invalid_flag, 1u);
  }
}

__device__ __forceinline__ static void cubbs_dist_fp12_store(
    uint8_t *dst576, const CuBbsDistFp12 &v, uint32_t pair) {
  if (pair < 3u) {
    cubbs_pairing_store_fp2(dst576 + (pair * 96u), v.c0);
    cubbs_pairing_store_fp2(dst576 + 288u + (pair * 96u), v.c1);
  }
}

__device__ __forceinline__ static CuBbsDistFp12 cubbs_dist_fp12_load(
    const uint8_t *src576, uint32_t pair) {
  if (pair < 3u) {
    return {cubbs_pairing_load_fp2(src576 + (pair * 96u)),
            cubbs_pairing_load_fp2(src576 + 288u + (pair * 96u))};
  }
  return {cubbs_pairing_fp2_zero(), cubbs_pairing_fp2_zero()};
}

// Fast shared-squaring 1-warp (4-group x kItemsPerGroup) Miller loop reading
// GPU-precomputed validated G2 lines + in-warp 4->1 Fp12 product reduction.
// Sharing ONE Fp12 squaring across kItemsPerGroup items cuts Miller Fp12 squarings
// by 2x (K=2) to 4x (K=4) with zero extra register pressure!
template <uint32_t kItemsPerGroup>
__global__ static void cubbs_bbs_unique_pk_miller_lines_block_reduce_kernel(
    const uint8_t *lhs_g1_affines,  // B * 96B plain affine P_i
    const uint8_t *g2_lines_all,    // B * (68 * 288B) validated GPU G2 lines
    size_t item_count,
    uint8_t *block_fp12_out) {      // num_blocks * 576B
  __shared__ __align__(16) uint8_t s_fp12[4u * 576u];
  __shared__ __align__(16) uint8_t s_g1[4u * kItemsPerGroup * 96u];
  __shared__ uint8_t s_valid[4u * kItemsPerGroup];

  const uint32_t g_in_blk = threadIdx.x >> 3u; // 0..3
  const uint32_t sub = threadIdx.x & 7u;
  const uint32_t pair = sub >> 1u;
  const size_t base_item =
      (static_cast<size_t>(blockIdx.x) * 4u + g_in_blk) * kItemsPerGroup;

  // Pre-transform and cache kItemsPerGroup G1 affine points in shared memory.
  if (sub == 0u) {
    #pragma unroll
    for (uint32_t k = 0u; k < kItemsPerGroup; ++k) {
      const size_t raw_item = base_item + k;
      const bool valid = raw_item < item_count;
      const size_t item = valid ? raw_item : 0u;
      const uint8_t *p_ptr = lhs_g1_affines + (item * 96u);
      fp_t px = cubbs_pairing_load_fp(p_ptr);
      fp_t py = cubbs_pairing_load_fp(p_ptr + sizeof(fp_t));
      const bool p_inf = px.is_zero() && py.is_zero();
      px = px + px;
      px.cneg(true);
      py = py + py;
      uint8_t *dst = s_g1 + ((g_in_blk * kItemsPerGroup + k) * 96u);
      memcpy(dst, &px, sizeof(fp_t));
      memcpy(dst + sizeof(fp_t), &py, sizeof(fp_t));
      s_valid[g_in_blk * kItemsPerGroup + k] = (valid && !p_inf) ? 1u : 0u;
    }
  }
  __syncwarp();

  constexpr size_t line_stride = 6u * sizeof(fp_t);
  constexpr size_t lines_bytes = 68u * line_stride;

  CuBbsDistFp12 acc = cubbs_dist_fp12_one(pair);
  #pragma unroll 1
  for (uint32_t i = 0u; i < 68u; ++i) {
    const bool addition_line =
        i == 0u || i == 1u || i == 4u || i == 8u || i == 18u || i == 51u;
    if (!addition_line) {
      acc = cubbs_dist_fp12_square(acc);
    }
    #pragma unroll
    for (uint32_t k = 0u; k < kItemsPerGroup; ++k) {
      const size_t raw_item = base_item + k;
      const size_t item = (raw_item < item_count) ? raw_item : 0u;
      const uint8_t *g1_ptr = s_g1 + ((g_in_blk * kItemsPerGroup + k) * 96u);
      const fp_t px = cubbs_pairing_load_fp(g1_ptr);
      const fp_t py = cubbs_pairing_load_fp(g1_ptr + sizeof(fp_t));
      const uint8_t *lines = g2_lines_all + (item * lines_bytes);
      const auto line = cubbs_pairing_line_for_g1(lines + (i * line_stride), px, py);
      CuBbsDistFp12 next = acc;
      cubbs_dist_fp12_mul_by_014(next, line, pair);
      if (s_valid[g_in_blk * kItemsPerGroup + k] != 0u) acc = next;
    }
  }
  acc = cubbs_dist_fp12_conjugate(acc);

  cubbs_dist_fp12_store(s_fp12 + (g_in_blk * 576u), acc, pair);
  __syncwarp();
  #pragma unroll
  for (uint32_t stride = 2u; stride > 0u; stride >>= 1u) {
    const uint32_t partner = (g_in_blk < stride) ? (g_in_blk + stride) : g_in_blk;
    const auto rhs = cubbs_dist_fp12_load(s_fp12 + (partner * 576u), pair);
    const auto prod = cubbs_dist_fp12_mul(acc, rhs);
    if (g_in_blk < stride) {
      acc = prod;
      cubbs_dist_fp12_store(s_fp12 + (g_in_blk * 576u), acc, pair);
    }
    __syncwarp();
  }

  if (g_in_blk == 0u) {
    cubbs_dist_fp12_store(block_fp12_out + (static_cast<size_t>(blockIdx.x) * 576u), acc, pair);
  }
}

// Single-block finalizer: multiplies all block Fp12 partials + Miller(-S_g2, G2),
// runs ONE Final Exponentiation, and writes *eq_out = (final == 1).
__global__ static void cubbs_bbs_unique_pk_finalize_pairing_kernel(
    const uint8_t *block_fp12_in, size_t num_blocks,
    const uint8_t *rhs_g1_agg,      // 1 plain affine point S_g2 (negated here -> -S_g2)
    const uint8_t *g2_lines,        // precomputed 68-line G2 generator table
    uint8_t *eq_out) {
  __shared__ __align__(16) uint8_t s_fp12[8u * 576u];

  const uint32_t g_id = threadIdx.x >> 3u; // 0..7
  const uint32_t sub = threadIdx.x & 7u;
  const uint32_t pair = sub >> 1u;

  const auto fp12_one = cubbs_dist_fp12_one(pair);
  CuBbsDistFp12 acc = fp12_one;
  #pragma unroll 1
  for (size_t base = 0u; base < num_blocks; base += 8u) {
    const size_t idx = base + g_id;
    const auto val = (idx < num_blocks)
        ? cubbs_dist_fp12_load(block_fp12_in + (idx * 576u), pair)
        : fp12_one;
    acc = cubbs_dist_fp12_mul(acc, val);
  }

  // Evaluate Miller(-S_g2, G2) uniformly across all 8 groups, multiply into group 0.
  fp_t gx = cubbs_pairing_load_fp(rhs_g1_agg);
  fp_t gy = cubbs_pairing_load_fp(rhs_g1_agg + sizeof(fp_t));
  const bool g_inf = gx.is_zero() && gy.is_zero();
  // Negate S_g2: (-2*Gx, -2*Gy)
  gx = gx + gx;
  gx.cneg(true);
  gy = gy + gy;
  gy.cneg(true);

  constexpr size_t line_stride = 6u * sizeof(fp_t);
  CuBbsDistFp12 g2_acc = fp12_one;
  #pragma unroll 1
  for (size_t i = 0u; i < 68u; ++i) {
    const bool addition_line =
        i == 0u || i == 1u || i == 4u || i == 8u || i == 18u || i == 51u;
    if (!addition_line) {
      g2_acc = cubbs_dist_fp12_square(g2_acc);
    }
    const auto line = cubbs_pairing_line_for_g1(g2_lines + (i * line_stride), gx, gy);
    CuBbsDistFp12 next = g2_acc;
    cubbs_dist_fp12_mul_by_014(next, line, pair);
    if (!g_inf) g2_acc = next;
  }
  g2_acc = cubbs_dist_fp12_conjugate(g2_acc);

  const auto g2_term = (g_id == 0u) ? g2_acc : fp12_one;
  acc = cubbs_dist_fp12_mul(acc, g2_term);

  cubbs_dist_fp12_store(s_fp12 + (g_id * 576u), acc, pair);
  __syncthreads();
  #pragma unroll
  for (uint32_t stride = 4u; stride > 0u; stride >>= 1u) {
    const uint32_t partner = (g_id < stride) ? (g_id + stride) : g_id;
    const auto rhs = cubbs_dist_fp12_load(s_fp12 + (partner * 576u), pair);
    const auto prod = cubbs_dist_fp12_mul(acc, rhs);
    if (g_id < stride) {
      acc = prod;
      cubbs_dist_fp12_store(s_fp12 + (g_id * 576u), acc, pair);
    }
    __syncthreads();
  }

  // Uniform Final Exponentiation across warp 0 (group 0 holds the reduced product).
  const auto final_val = cubbs_dist_fp12_final_exponentiation(acc);
  const bool equal = cubbs_dist_fp12_is_one(final_val);
  if (threadIdx.x == 0u) {
    *eq_out = equal ? 1u : 0u;
  }
}

int32_t cubbs_bbs_cuda_launch_pairing_unique_pk_on_stream(
    const void *d_lhs_g1_affines, const void *d_rhs_g1_agg,
    const uint8_t *d_pks_compressed, const uint8_t *h_pks_compressed,
    size_t item_count, const uint8_t *g2_lines, const uint8_t *g2_key,
    uint32_t *d_invalid_flag, uint8_t *d_eq_out, void *cuda_stream) {
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  if (d_lhs_g1_affines == nullptr || d_rhs_g1_agg == nullptr ||
      d_pks_compressed == nullptr || g2_lines == nullptr ||
      d_invalid_flag == nullptr || d_eq_out == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  constexpr size_t lines_bytes = 68u * 6u * sizeof(fp_t);
  const size_t all_lines_bytes = item_count * lines_bytes;
  const uint32_t items_per_group =
      (item_count >= 512u) ? 4u : ((item_count >= 256u) ? 2u : 1u);
  const size_t items_per_block = 4u * static_cast<size_t>(items_per_group);
  const size_t num_blocks = (item_count + items_per_block - 1u) / items_per_block;
  const size_t blk_fp12_bytes = num_blocks * 576u;

  auto &workspace = cubbs_miller_device_workspace();
  cudaError_t error = cudaSuccess;

  if (workspace.rhs_lines_capacity < lines_bytes) {
    if (workspace.rhs_lines != nullptr) (void)cudaFree(workspace.rhs_lines);
    workspace.rhs_lines = nullptr;
    workspace.rhs_lines_capacity = 0u;
    workspace.rhs_key_valid = false;
    error = cudaMalloc(&workspace.rhs_lines, lines_bytes);
    if (error == cudaSuccess) workspace.rhs_lines_capacity = lines_bytes;
  }
  if (error == cudaSuccess && workspace.unique_g2_lines_capacity < all_lines_bytes) {
    if (workspace.unique_g2_lines != nullptr) (void)cudaFree(workspace.unique_g2_lines);
    workspace.unique_g2_lines = nullptr;
    workspace.unique_g2_lines_capacity = 0u;
    workspace.unique_pk_key.clear();
    error = cudaMalloc(&workspace.unique_g2_lines, all_lines_bytes);
    if (error == cudaSuccess) workspace.unique_g2_lines_capacity = all_lines_bytes;
  }
  if (error == cudaSuccess && workspace.unique_block_fp12_capacity < blk_fp12_bytes) {
    if (workspace.unique_block_fp12 != nullptr) (void)cudaFree(workspace.unique_block_fp12);
    workspace.unique_block_fp12 = nullptr;
    workspace.unique_block_fp12_capacity = 0u;
    error = cudaMalloc(&workspace.unique_block_fp12, blk_fp12_bytes);
    if (error == cudaSuccess) workspace.unique_block_fp12_capacity = blk_fp12_bytes;
  }
  if (error == cudaSuccess && !workspace.frobenius_initialized) {
    static std::once_flag frob_once;
    static cudaError_t frob_err = cudaSuccess;
    std::call_once(frob_once, []() {
      frob_err = cubbs_pairing_initialize_frobenius_coefficients();
    });
    error = frob_err;
    if (error == cudaSuccess) workspace.frobenius_initialized = true;
  }
  const bool upload_rhs = g2_key == nullptr || !workspace.rhs_key_valid ||
                          std::memcmp(workspace.rhs_g2_key, g2_key, 96u) != 0;
  if (error == cudaSuccess && upload_rhs) {
    error = cudaMemcpyAsync(workspace.rhs_lines, g2_lines, lines_bytes,
                            cudaMemcpyHostToDevice, stream);
  }
  const bool pk_cache_hit =
      h_pks_compressed != nullptr &&
      workspace.unique_pk_key.size() == item_count * 96u &&
      std::memcmp(workspace.unique_pk_key.data(), h_pks_compressed, item_count * 96u) == 0;
  if (error == cudaSuccess && !pk_cache_hit) {
    // 32 threads/block (16 keys/block -> 32 blocks for B=512, fills RTX 5060 Ti SMs!).
    const uint32_t dec_blocks = static_cast<uint32_t>((item_count * 2u + 31u) / 32u);
    cubbs_bbs_g2_uncompress_and_precompute_lines_kernel<<<dec_blocks, 32u, 0, stream>>>(
        d_pks_compressed, item_count, workspace.unique_g2_lines, d_invalid_flag);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    if (items_per_group == 4u) {
      cubbs_bbs_unique_pk_miller_lines_block_reduce_kernel<4u>
          <<<static_cast<uint32_t>(num_blocks), 32u, 0, stream>>>(
              reinterpret_cast<const uint8_t *>(d_lhs_g1_affines),
              workspace.unique_g2_lines, item_count, workspace.unique_block_fp12);
    } else if (items_per_group == 2u) {
      cubbs_bbs_unique_pk_miller_lines_block_reduce_kernel<2u>
          <<<static_cast<uint32_t>(num_blocks), 32u, 0, stream>>>(
              reinterpret_cast<const uint8_t *>(d_lhs_g1_affines),
              workspace.unique_g2_lines, item_count, workspace.unique_block_fp12);
    } else {
      cubbs_bbs_unique_pk_miller_lines_block_reduce_kernel<1u>
          <<<static_cast<uint32_t>(num_blocks), 32u, 0, stream>>>(
              reinterpret_cast<const uint8_t *>(d_lhs_g1_affines),
              workspace.unique_g2_lines, item_count, workspace.unique_block_fp12);
    }
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    cubbs_bbs_unique_pk_finalize_pairing_kernel<<<1u, 64u, 0, stream>>>(
        workspace.unique_block_fp12, num_blocks,
        reinterpret_cast<const uint8_t *>(d_rhs_g1_agg),
        workspace.rhs_lines, d_eq_out);
    error = cudaGetLastError();
  }
  workspace.rhs_key_valid = error == cudaSuccess && g2_key != nullptr;
  if (workspace.rhs_key_valid) std::memcpy(workspace.rhs_g2_key, g2_key, 96u);
  return error == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS
                              : CUBBS_BBS_GPU_ERR_INTERNAL;
}

void cubbs_bbs_cuda_confirm_unique_pk_lines_cache(
    const uint8_t *h_pks_compressed, size_t item_count) {
  if (h_pks_compressed == nullptr || item_count == 0u) return;
  auto &workspace = cubbs_miller_device_workspace();
  const size_t bytes = item_count * 96u;
  if (workspace.unique_pk_key.size() == bytes &&
      std::memcmp(workspace.unique_pk_key.data(), h_pks_compressed, bytes) == 0) {
    return;
  }
  workspace.unique_pk_key.resize(bytes);
  std::memcpy(workspace.unique_pk_key.data(), h_pks_compressed, bytes);
}

#endif
