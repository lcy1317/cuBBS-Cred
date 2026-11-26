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
    if (!lhs_inf) {
      const auto line_l = cubbs_pairing_line_for_g1(l_lines + (i * line_stride), lx, ly);
      cubbs_dist_fp12_mul_by_014(acc, line_l, pair);
    }
    if (!rhs_inf) {
      const auto line_r = cubbs_pairing_line_for_g1(r_lines + (i * line_stride), rx, ry);
      cubbs_dist_fp12_mul_by_014(acc, line_r, pair);
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

#endif
