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

__device__ __forceinline__ static CuBbsPairingFp6 cubbs_pairing_fp6_add(
    const CuBbsPairingFp6 &a, const CuBbsPairingFp6 &b) {
  return {a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2};
}

__device__ __forceinline__ static CuBbsPairingFp6 cubbs_pairing_fp6_sub(
    const CuBbsPairingFp6 &a, const CuBbsPairingFp6 &b) {
  return {a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2};
}

__device__ __forceinline__ static CuBbsPairingFp6
cubbs_pairing_fp6_mul(const CuBbsPairingFp6 &a,
                       const CuBbsPairingFp6 &b) {
  const auto v0 = a.c0 * b.c0;
  const auto v1 = a.c1 * b.c1;
  const auto v2 = a.c2 * b.c2;
  const auto c0 =
      v0 + cubbs_pairing_fp2_mul_nonresidue(
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
cubbs_pairing_fp12_mul(const CuBbsPairingFp12 &a,
                        const CuBbsPairingFp12 &b) {
  const auto aa = cubbs_pairing_fp6_mul(a.c0, b.c0);
  const auto bb = cubbs_pairing_fp6_mul(a.c1, b.c1);
  const auto c0 = cubbs_pairing_fp6_add(
      aa, cubbs_pairing_fp6_mul_nonresidue(bb));
  const auto c1 = cubbs_pairing_fp6_sub(
      cubbs_pairing_fp6_sub(
          cubbs_pairing_fp6_mul(cubbs_pairing_fp6_add(a.c0, a.c1),
                                 cubbs_pairing_fp6_add(b.c0, b.c1)),
          aa),
      bb);
  return {c0, c1};
}

__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_square(const CuBbsPairingFp12 &a) {
  const auto ab = cubbs_pairing_fp6_mul(a.c0, a.c1);
  const auto product = cubbs_pairing_fp6_mul(
      cubbs_pairing_fp6_add(a.c0, a.c1),
      cubbs_pairing_fp6_add(a.c0,
                             cubbs_pairing_fp6_mul_nonresidue(a.c1)));
  const auto c0 = cubbs_pairing_fp6_sub(
      cubbs_pairing_fp6_sub(product, ab),
      cubbs_pairing_fp6_mul_nonresidue(ab));
  const auto c1 = cubbs_pairing_fp6_add(ab, ab);
  return {c0, c1};
}

struct CuBbsPairingFp4 {
  bls12_381::fp2_t c0;
  bls12_381::fp2_t c1;
};

__device__ __forceinline__ static CuBbsPairingFp4
cubbs_pairing_fp4_square(const bls12_381::fp2_t &a,
                          const bls12_381::fp2_t &b) {
  const auto t0 = a * a;
  const auto t1 = b * b;
  const auto c0 = t0 + cubbs_pairing_fp2_mul_nonresidue(t1);
  const auto c1 = (a + b) * (a + b) - t0 - t1;
  return {c0, c1};
}

// Granger-Scott squaring in the degree-six cyclotomic subgroup. Final
// exponentiation spends most of its time here, so this avoids full Fp12
// squaring's Fp6 products and uses three independent Fp4 squares instead.
__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_cyclotomic_square(const CuBbsPairingFp12 &a) {
  auto z0 = a.c0.c0;
  auto z4 = a.c0.c1;
  auto z3 = a.c0.c2;
  auto z2 = a.c1.c0;
  auto z1 = a.c1.c1;
  auto z5 = a.c1.c2;

  const auto a_square = cubbs_pairing_fp4_square(z0, z1);
  z0 = (a_square.c0 - z0) + (a_square.c0 - z0) + a_square.c0;
  z1 = (a_square.c1 + z1) + (a_square.c1 + z1) + a_square.c1;

  const auto b_square = cubbs_pairing_fp4_square(z2, z3);
  const auto c_square = cubbs_pairing_fp4_square(z4, z5);
  z4 = (b_square.c0 - z4) + (b_square.c0 - z4) + b_square.c0;
  z5 = (b_square.c1 + z5) + (b_square.c1 + z5) + b_square.c1;
  const auto t = cubbs_pairing_fp2_mul_nonresidue(c_square.c1);
  z2 = (t + z2) + (t + z2) + t;
  z3 = (c_square.c0 - z3) + (c_square.c0 - z3) + c_square.c0;

  return {{z0, z4, z3}, {z2, z1, z5}};
}

__device__ __forceinline__ static CuBbsPairingFp6
cubbs_pairing_fp6_inverse(const CuBbsPairingFp6 &a) {
  const auto t0 = a.c0 * a.c0 -
                  cubbs_pairing_fp2_mul_nonresidue(a.c1 * a.c2);
  const auto t1 = cubbs_pairing_fp2_mul_nonresidue(a.c2 * a.c2) -
                  a.c0 * a.c1;
  const auto t2 = a.c1 * a.c1 - a.c0 * a.c2;
  const auto denominator =
      (a.c0 * t0 + cubbs_pairing_fp2_mul_nonresidue(a.c2 * t1 + a.c1 * t2))
          .reciprocal();
  return {t0 * denominator, t1 * denominator, t2 * denominator};
}

__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_inverse(const CuBbsPairingFp12 &a) {
  const auto denominator = cubbs_pairing_fp6_inverse(
      cubbs_pairing_fp6_sub(
          cubbs_pairing_fp6_mul(a.c0, a.c0),
          cubbs_pairing_fp6_mul_nonresidue(
              cubbs_pairing_fp6_mul(a.c1, a.c1))));
  auto c1 = cubbs_pairing_fp6_mul(a.c1, denominator);
  c1.c0.cneg(true);
  c1.c1.cneg(true);
  c1.c2.cneg(true);
  return {cubbs_pairing_fp6_mul(a.c0, denominator), c1};
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

__device__ __forceinline__ static void cubbs_pairing_fp6_frobenius(
    CuBbsPairingFp6 &a, unsigned power) {
  if ((power & 1u) != 0u) {
    a.c0 = cubbs_pairing_fp2_conjugate(a.c0);
    a.c1 = cubbs_pairing_fp2_conjugate(a.c1);
    a.c2 = cubbs_pairing_fp2_conjugate(a.c2);
  }
  const size_t c1_offset = power == 1u ? CUBBS_FROB_F6_C1_P1
                                      : CUBBS_FROB_F6_C1_P2;
  const size_t c2_offset = power == 1u ? CUBBS_FROB_F6_C2_P1
                                      : CUBBS_FROB_F6_C2_P2;
  a.c1 *= cubbs_pairing_load_frobenius_coefficient(c1_offset);
  a.c2 *= cubbs_pairing_load_frobenius_coefficient(c2_offset);
}

__device__ __forceinline__ static void cubbs_pairing_fp12_frobenius(
    CuBbsPairingFp12 &a, unsigned power) {
  cubbs_pairing_fp6_frobenius(a.c0, power);
  cubbs_pairing_fp6_frobenius(a.c1, power);
  const size_t offset = power == 1u ? CUBBS_FROB_F12_C1_P1
                                   : CUBBS_FROB_F12_C1_P2;
  const auto coefficient = cubbs_pairing_load_frobenius_coefficient(offset);
  a.c1.c0 *= coefficient;
  a.c1.c1 *= coefficient;
  a.c1.c2 *= coefficient;
}

__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_conjugate(CuBbsPairingFp12 a) {
  a.c1.c0.cneg(true);
  a.c1.c1.cneg(true);
  a.c1.c2.cneg(true);
  return a;
}

// Exponentiation by the negative BLS12-381 parameter
// z = -0xd201000000010000. The input is cyclotomic after the easy part, so
// conjugation is inversion.
__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_exp_by_z(const CuBbsPairingFp12 &a) {
  constexpr uint64_t z_abs = 0xd201000000010000ULL;
  CuBbsPairingFp12 result = a;
  bool started = false;
  #pragma unroll 1
  for (int bit = 63; bit >= 0; --bit) {
    if (started) {
      result = cubbs_pairing_fp12_cyclotomic_square(result);
    }
    if (((z_abs >> bit) & 1u) != 0u) {
      if (started) {
        result = cubbs_pairing_fp12_mul(result, a);
      } else {
        result = a;
        started = true;
      }
    }
  }
  return cubbs_pairing_fp12_conjugate(result);
}

// Fuentes-Castaneda et al. BLS12 final exponent decomposition. This is the
// same algebraic decomposition used by current pairing implementations, but
// expressed directly over cuBBS-Cred's distributed Fp2 CUDA representation.
__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_final_exponentiation(const CuBbsPairingFp12 &f) {
  const auto f_conjugate = cubbs_pairing_fp12_conjugate(f);
  const auto f_inverse = cubbs_pairing_fp12_inverse(f);
  auto r = cubbs_pairing_fp12_mul(f_conjugate, f_inverse);
  const auto easy = r;
  cubbs_pairing_fp12_frobenius(r, 2u);
  r = cubbs_pairing_fp12_mul(r, easy);

  auto y0 = cubbs_pairing_fp12_cyclotomic_square(r);
  auto y1 = cubbs_pairing_fp12_exp_by_z(r);
  auto y2 = cubbs_pairing_fp12_conjugate(r);
  y1 = cubbs_pairing_fp12_mul(y1, y2);
  y2 = cubbs_pairing_fp12_exp_by_z(y1);
  y1 = cubbs_pairing_fp12_conjugate(y1);
  y1 = cubbs_pairing_fp12_mul(y1, y2);
  y2 = cubbs_pairing_fp12_exp_by_z(y1);
  cubbs_pairing_fp12_frobenius(y1, 1u);
  y1 = cubbs_pairing_fp12_mul(y1, y2);
  r = cubbs_pairing_fp12_mul(r, y0);
  y0 = cubbs_pairing_fp12_exp_by_z(y1);
  y2 = cubbs_pairing_fp12_exp_by_z(y0);
  y0 = y1;
  cubbs_pairing_fp12_frobenius(y0, 2u);
  y1 = cubbs_pairing_fp12_conjugate(y1);
  y1 = cubbs_pairing_fp12_mul(y1, y2);
  y1 = cubbs_pairing_fp12_mul(y1, y0);
  return cubbs_pairing_fp12_mul(r, y1);
}

__device__ __forceinline__ static void cubbs_pairing_fp6_mul_by_1(
    CuBbsPairingFp6 &r, const bls12_381::fp2_t &c1) {
  const auto bb = r.c1 * c1;
  auto t1 = c1 * (r.c1 + r.c2) - bb;
  t1 = cubbs_pairing_fp2_mul_nonresidue(t1);
  const auto t2 = c1 * (r.c0 + r.c1) - bb;
  r = {t1, t2, bb};
}

__device__ __forceinline__ static void cubbs_pairing_fp6_mul_by_01(
    CuBbsPairingFp6 &r, const bls12_381::fp2_t &c0,
    const bls12_381::fp2_t &c1) {
  const auto aa = r.c0 * c0;
  const auto bb = r.c1 * c1;
  auto t1 = c1 * (r.c1 + r.c2) - bb;
  t1 = cubbs_pairing_fp2_mul_nonresidue(t1) + aa;
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
  f.c0 = cubbs_pairing_fp6_add(
      cubbs_pairing_fp6_mul_nonresidue(bb), aa);
  f.c1 = c1;
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

__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_fp12_from_line(const CuBbsPairingLine014 &line) {
  const auto zero = cubbs_pairing_fp2_zero();
  return {{line.c0, line.c1, zero}, {zero, line.c4, zero}};
}

__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_miller_loop_lines(const uint8_t *g1_affine,
                                  const uint8_t *lines) {
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
    if (!addition_line) {
      result = cubbs_pairing_fp12_square(result);
    }
    const auto line =
        cubbs_pairing_line_for_g1(lines + (i * line_stride), px, py);
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

__device__ __forceinline__ static CuBbsPairingFp12
cubbs_pairing_load_fp12(const uint8_t *input) {
  return {{cubbs_pairing_load_fp2(input),
           cubbs_pairing_load_fp2(input + (2u * sizeof(fp_t))),
           cubbs_pairing_load_fp2(input + (4u * sizeof(fp_t)))},
          {cubbs_pairing_load_fp2(input + (6u * sizeof(fp_t))),
           cubbs_pairing_load_fp2(input + (8u * sizeof(fp_t))),
           cubbs_pairing_load_fp2(input + (10u * sizeof(fp_t)))}};
}

__device__ __forceinline__ static bool cubbs_pairing_fp12_is_one(
    const CuBbsPairingFp12 &a) {
  bool result = a.c0.c0.is_one();
  result = result & a.c0.c1.is_zero();
  result = result & a.c0.c2.is_zero();
  result = result & a.c1.c0.is_zero();
  result = result & a.c1.c1.is_zero();
  result = result & a.c1.c2.is_zero();
  return result;
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


__global__ static void cubbs_bbs_pairing_finalverify_kernel(
    const uint8_t *lhs_fp12, const uint8_t *rhs_fp12, size_t item_count,
    uint8_t *values_out) {
  const size_t item = (blockIdx.x * blockDim.x + threadIdx.x) >> 1u;
  if (item >= item_count) {
    return;
  }
  constexpr size_t fp12_stride = 12u * sizeof(fp_t);
  auto lhs = cubbs_pairing_load_fp12(lhs_fp12 + (item * fp12_stride));
  const auto rhs = cubbs_pairing_load_fp12(rhs_fp12 + (item * fp12_stride));
  lhs = cubbs_pairing_fp12_conjugate(lhs);
  const auto quotient = cubbs_pairing_fp12_mul(lhs, rhs);
  const auto final_value =
      cubbs_pairing_fp12_final_exponentiation(quotient);
  const bool equal = cubbs_pairing_fp12_is_one(final_value);
  if ((threadIdx.x & 1u) == 0u) {
    values_out[item] = equal ? 1u : 0u;
  }
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
  // Montgomery-domain coefficients. Keeping these in the same raw 12x32-bit
  // layout used by device fp_t avoids a host/device field-class ABI dependency.
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
  if (error == cudaSuccess && finalverify_out != nullptr &&
      !workspace.frobenius_initialized) {
    error = cubbs_pairing_initialize_frobenius_coefficients();
    if (error == cudaSuccess) {
      workspace.frobenius_initialized = true;
    }
  }
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(workspace.lhs_g1, lhs_g1_affines, g1_bytes,
                            cudaMemcpyHostToDevice, workspace.stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(workspace.rhs_g1, rhs_g1_affines, g1_bytes,
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
  if (error == cudaSuccess) {
    const uint32_t configured_threads = static_cast<uint32_t>(std::min<size_t>(
        1024u, std::max<size_t>(2u, env_size_or_default("CUBBS_BBS_GPU_MILLER_THREADS", 64u)))) & ~1u;
    // Each Miller-loop instance is represented by two cooperating lanes.  A
    // reduced aggregate verification contains only two instances (lhs/rhs),
    // so launching a full 64-thread block reserves registers for lanes which
    // immediately return.  Cap small launches at their useful lane count;
    // larger batches retain the configured block size.
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
  }
  if (error == cudaSuccess && finalverify_out != nullptr) {
    const uint32_t configured_threads = static_cast<uint32_t>(std::min<size_t>(
        1024u, std::max<size_t>(2u, env_size_or_default("CUBBS_BBS_GPU_FINAL_EXP_THREADS", 64u)))) & ~1u;
    const uint32_t useful_threads = static_cast<uint32_t>(item_count * 2u);
    const uint32_t threads = std::max<uint32_t>(
        2u, std::min(configured_threads, useful_threads));
    const uint32_t blocks =
        static_cast<uint32_t>((item_count * 2u + threads - 1u) / threads);
    cubbs_bbs_pairing_finalverify_kernel
        <<<blocks, threads, 0, workspace.stream>>>(
            workspace.lhs_out, workspace.rhs_out, item_count,
            workspace.values_out);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && finalverify_out != nullptr)
    error = cudaMemcpyAsync(finalverify_out, workspace.values_out, item_count,
                            cudaMemcpyDeviceToHost, workspace.stream);
  if (error == cudaSuccess && finalverify_out == nullptr)
    error = cudaMemcpyAsync(lhs_fp12_out, workspace.lhs_out, output_bytes,
                            cudaMemcpyDeviceToHost, workspace.stream);
  if (error == cudaSuccess && finalverify_out == nullptr)
    error = cudaMemcpyAsync(rhs_fp12_out, workspace.rhs_out, output_bytes,
                            cudaMemcpyDeviceToHost, workspace.stream);
  if (error == cudaSuccess)
    error = cudaStreamSynchronize(workspace.stream);
  workspace.lhs_key_valid = error == cudaSuccess && lhs_lines_stride == 0u && lhs_g2_key != nullptr;
  workspace.rhs_key_valid = error == cudaSuccess && rhs_lines_stride == 0u && rhs_g2_key != nullptr;
  if (workspace.lhs_key_valid) std::memcpy(workspace.lhs_g2_key, lhs_g2_key, 96u);
  if (workspace.rhs_key_valid) std::memcpy(workspace.rhs_g2_key, rhs_g2_key, 96u);
  return error == cudaSuccess ? CUBBS_BBS_GPU_SUCCESS
                              : CUBBS_BBS_GPU_ERR_INTERNAL;
}

#endif
