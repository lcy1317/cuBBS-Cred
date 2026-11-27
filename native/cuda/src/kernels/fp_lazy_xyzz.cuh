#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_FP_LAZY_XYZZ_CUH
#define CUBBS_BBS_FP_LAZY_XYZZ_CUH

// ============================================================================
// 381-bit BLS12-381 Fp Lazy-Reduction Arithmetic ([0, 2p) / [0, 4p)) &
// Complete XYZZ Mixed Addition (0 final_sub on all muls + 78-mul symmetric sqr)
// ============================================================================
// Mathematical basis:
//   p < 2^381, R = 2^384 => 8*p = 0.813 * R < R (3 spare top bits!).
//   1. For ANY a in [0, 4p) and b in [0, 2p):
//        T = (a*b + q*p) / R < (8*p^2 + R*p) / R < 2*p
//      so CIOS Montgomery multiplication WITHOUT final_sub strictly stays in [0, 2p)!
//   2. For ANY a in [0, p):
//        a^2 < p^2 => high half s1 < p, low half redc(s0) <= p => s0_redc + s1 < 2*p
//      using 78-mul symmetric squaring (vs 144-mul generic mul) WITHOUT final_sub!
//   3. Predicated sub_2p / final_sub_once matches Blackwell's dual-issue ALU pipeline.
//   4. Correctness invariant:
//      - P = U2 - X1 and R = S2 - Y1 are reduced canonically to [0, p), so
//        P == 0 and R == 0 are 100% exact (never fooled by x == p).
//      - On the fast path P in (0, p) is nonzero mod p, so ZZ3, ZZZ3 are never
//        0 or p; acc.is_inf() remains 100% exact throughout the loop.
//      - Before any canonical SPPARK op (dbl_u, xyzz+xyzz, or output store),
//        cubbs_bbs_xyzz_canonicalize(acc) reduces all 4 limbs [0, 2p) -> [0, p).

static constexpr uint32_t CUBBS_LAZY_FP_M0 = 0xfffcfffdu;

__device__ __constant__ static const uint32_t CUBBS_LAZY_FP_MOD[12] = {
    0xffffaaabu, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau};

__device__ __constant__ static const uint32_t CUBBS_LAZY_FP_MOD2[12] = {
    0xffff5556u, 0x73fdffffu, 0x62a7ffffu, 0x3d57fffdu,
    0xed61ec48u, 0xce61a541u, 0xe70a257eu, 0xc8ee9709u,
    0x869759aeu, 0x96374f6cu, 0x72ffcd34u, 0x340223d4u};

__device__ __forceinline__ static void cubbs_lfp_mul_n(
    uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = 12u) {
  #pragma unroll
  for (size_t j = 0u; j < n; j += 2u) {
    asm volatile("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
                 : "=r"(acc[j]), "=r"(acc[j + 1u])
                 : "r"(a[j]), "r"(bi));
  }
}

__device__ __forceinline__ static void cubbs_lfp_cmad_n(
    uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = 12u) {
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
               : "+r"(acc[0]), "+r"(acc[1])
               : "r"(a[0]), "r"(bi));
  #pragma unroll
  for (size_t j = 2u; j < n; j += 2u) {
    asm volatile("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                 : "+r"(acc[j]), "+r"(acc[j + 1u])
                 : "r"(a[j]), "r"(bi));
  }
}

__device__ __forceinline__ static void cubbs_lfp_cadd_n(
    uint32_t *acc, const uint32_t *a, size_t n = 12u) {
  asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(acc[0]) : "r"(a[0]));
  #pragma unroll
  for (size_t i = 1u; i < n; ++i) {
    asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(acc[i]) : "r"(a[i]));
  }
}

__device__ __forceinline__ static void cubbs_lfp_madc_n_rshift(
    uint32_t *odd, const uint32_t *a, uint32_t bi) {
  #pragma unroll
  for (size_t j = 0u; j < 10u; j += 2u) {
    asm volatile("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
                 : "=r"(odd[j]), "=r"(odd[j + 1u])
                 : "r"(a[j]), "r"(bi), "r"(odd[j + 2u]), "r"(odd[j + 3u]));
  }
  asm volatile("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
               : "=r"(odd[10]), "=r"(odd[11])
               : "r"(a[10]), "r"(bi));
}

__device__ __forceinline__ static void cubbs_lfp_mad_n_redc(
    uint32_t *even, uint32_t *odd, const uint32_t *a, uint32_t bi, bool first) {
  if (first) {
    cubbs_lfp_mul_n(odd, a + 1u, bi, 12u);
    cubbs_lfp_mul_n(even, a, bi, 12u);
  } else {
    asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
    cubbs_lfp_madc_n_rshift(odd, a + 1u, bi);
    cubbs_lfp_cmad_n(even, a, bi, 12u);
    asm volatile("addc.u32 %0, %0, 0;" : "+r"(odd[11]));
  }
  uint32_t mi;
  asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(mi) : "r"(even[0]), "r"(CUBBS_LAZY_FP_M0));
  cubbs_lfp_cmad_n(odd, CUBBS_LAZY_FP_MOD + 1u, mi, 12u);
  cubbs_lfp_cmad_n(even, CUBBS_LAZY_FP_MOD, mi, 12u);
  asm volatile("addc.u32 %0, %0, 0;" : "+r"(odd[11]));
}

// Lazy CIOS Montgomery multiply: a in [0, 4p), b in [0, 2p) -> out in [0, 2p).
__device__ __forceinline__ static void cubbs_lfp_mul_2p(
    const uint32_t a[12], const uint32_t b[12], uint32_t out[12]) {
  uint32_t even[12], odd[12];
  #pragma unroll
  for (size_t i = 0u; i < 12u; i += 2u) {
    cubbs_lfp_mad_n_redc(even, odd, a, b[i], i == 0u);
    cubbs_lfp_mad_n_redc(odd, even, a, b[i + 1u], false);
  }
  cubbs_lfp_cadd_n(even, odd + 1u, 11u);
  asm volatile("addc.u32 %0, %0, 0;" : "+r"(even[11]));
  #pragma unroll
  for (size_t i = 0u; i < 12u; ++i) out[i] = even[i];
}

__device__ __forceinline__ static void cubbs_lfp_mad_row(
    uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n) {
  cubbs_lfp_cmad_n(odd, a + 1u, bi, n - 2u);
  asm volatile("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
               : "=r"(odd[n - 2u]), "=r"(odd[n - 1u])
               : "r"(a[n - 1u]), "r"(bi));
  cubbs_lfp_cmad_n(even, a, bi, n);
  asm volatile("addc.u32 %0, %0, 0;" : "+r"(odd[n - 1u]));
}

__device__ __forceinline__ static void cubbs_lfp_qad_row(
    uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n) {
  cubbs_lfp_cmad_n(odd, a, bi, n - 2u);
  asm volatile("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
               : "=r"(odd[n - 2u]), "=r"(odd[n - 1u])
               : "r"(a[n - 2u]), "r"(bi));
  cubbs_lfp_cmad_n(even, a + 1u, bi, n - 2u);
  asm volatile("addc.u32 %0, %0, 0;" : "+r"(odd[n - 1u]));
}

__device__ __forceinline__ static void cubbs_lfp_mul_by_1_row(
    uint32_t *even, uint32_t *odd, bool first) {
  uint32_t mi;
  if (first) {
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(mi) : "r"(even[0]), "r"(CUBBS_LAZY_FP_M0));
    cubbs_lfp_mul_n(odd, CUBBS_LAZY_FP_MOD + 1u, mi, 12u);
    cubbs_lfp_cmad_n(even, CUBBS_LAZY_FP_MOD, mi, 12u);
    asm volatile("addc.u32 %0, %0, 0;" : "+r"(odd[11]));
  } else {
    asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(mi) : "r"(even[0]), "r"(CUBBS_LAZY_FP_M0));
    cubbs_lfp_madc_n_rshift(odd, CUBBS_LAZY_FP_MOD + 1u, mi);
    cubbs_lfp_cmad_n(even, CUBBS_LAZY_FP_MOD, mi, 12u);
    asm volatile("addc.u32 %0, %0, 0;" : "+r"(odd[11]));
  }
}

// Lazy Symmetric Montgomery Squaring: a in [0, p) -> out in [0, 2p).
// Uses 78 32x32 muls (vs 144) and skips final_sub completely!
__device__ __forceinline__ static void cubbs_lfp_sqr_2p(
    const uint32_t a[12], uint32_t out[12]) {
  constexpr size_t n = 12u;
  uint32_t even[2u * n];
  uint32_t odd[2u * n - 2u];

  cubbs_lfp_mul_n(even + 2u, a + 2u, a[0], n - 2u);
  cubbs_lfp_mul_n(odd, a + 1u, a[0], n);

  size_t i = 0u;
  #pragma unroll
  while (i < n - 4u) {
    ++i;
    cubbs_lfp_mad_row(even + (2u * i + 2u), odd + (2u * i), a + (i + 1u), a[i], n - i - 1u);
    ++i;
    cubbs_lfp_qad_row(odd + (2u * i), even + (2u * i + 2u), a + (i + 1u), a[i], n - i);
  }

  asm volatile("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
               : "=r"(even[2u * n - 4u]), "=r"(even[2u * n - 3u])
               : "r"(a[n - 1u]), "r"(a[n - 3u]));
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
               : "+r"(odd[2u * n - 6u]), "+r"(odd[2u * n - 5u])
               : "r"(a[n - 2u]), "r"(a[n - 3u]));
  asm volatile("addc.u32 %0, %0, 0;" : "+r"(even[2u * n - 3u]));

  asm volatile("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
               : "=r"(odd[2u * n - 4u]), "=r"(odd[2u * n - 3u])
               : "r"(a[n - 1u]), "r"(a[n - 2u]));

  cubbs_lfp_cadd_n(even + 2u, odd + 1u, 2u * n - 4u);
  asm volatile("addc.u32 %0, %1, 0;" : "=r"(even[2u * n - 2u]) : "r"(odd[2u * n - 3u]));

  even[0] = 0u;
  asm volatile("add.cc.u32 %0, %1, %1;" : "=r"(even[1]) : "r"(odd[0]));
  #pragma unroll
  for (size_t j = 2u; j < 2u * n - 1u; ++j) {
    asm volatile("addc.cc.u32 %0, %0, %0;" : "+r"(even[j]));
  }
  asm volatile("addc.u32 %0, 0, 0;" : "=r"(even[2u * n - 1u]));

  asm volatile("mad.lo.cc.u32 %0, %2, %2, %0; madc.hi.cc.u32 %1, %2, %2, %1;"
               : "+r"(even[0]), "+r"(even[1])
               : "r"(a[0]));
  #pragma unroll
  for (size_t k = 1u; k < n; ++k) {
    asm volatile("madc.lo.cc.u32 %0, %2, %2, %0; madc.hi.cc.u32 %1, %2, %2, %1;"
                 : "+r"(even[2u * k]), "+r"(even[2u * k + 1u])
                 : "r"(a[k]));
  }

  uint32_t rodd[12];
  #pragma unroll
  for (size_t k = 0u; k < 12u; k += 2u) {
    cubbs_lfp_mul_by_1_row(even, rodd, k == 0u);
    cubbs_lfp_mul_by_1_row(rodd, even, false);
  }
  cubbs_lfp_cadd_n(even, rodd + 1u, 11u);
  asm volatile("addc.u32 %0, %0, 0;" : "+r"(even[11]));
  cubbs_lfp_cadd_n(even, even + 12u, 12u);
  #pragma unroll
  for (size_t k = 0u; k < 12u; ++k) out[k] = even[k];
}

// Predicated reduction [0, 2p) -> [0, p) (optimal on Blackwell dual-issue ALU).
__device__ __forceinline__ static void cubbs_lfp_final_sub_once(uint32_t x[12]) {
  uint32_t tmp[12];
  uint32_t borrow;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(x[0]), "r"(CUBBS_LAZY_FP_MOD[0]));
  #pragma unroll
  for (size_t i = 1u; i < 12u; ++i) {
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(x[i]), "r"(CUBBS_LAZY_FP_MOD[i]));
  }
  asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
  asm volatile("{ .reg.pred %ge; setp.eq.u32 %ge, %0, 0;" :: "r"(borrow));
  #pragma unroll
  for (size_t i = 0u; i < 12u; ++i) {
    asm volatile("@%ge mov.b32 %0, %1;" : "+r"(x[i]) : "r"(tmp[i]));
  }
  asm volatile("}");
}

// Predicated lazy subtraction: a, b in [0, 2p) -> out in [0, 2p).
__device__ __forceinline__ static void cubbs_lfp_sub_2p(
    const uint32_t a[12], const uint32_t b[12], uint32_t out[12]) {
  uint32_t tmp[12];
  uint32_t borrow;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(out[0]) : "r"(a[0]), "r"(b[0]));
  #pragma unroll
  for (size_t i = 1u; i < 12u; ++i) {
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(out[i]) : "r"(a[i]), "r"(b[i]));
  }
  asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));

  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(out[0]), "r"(CUBBS_LAZY_FP_MOD2[0]));
  #pragma unroll
  for (size_t i = 1u; i < 11u; ++i) {
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(out[i]), "r"(CUBBS_LAZY_FP_MOD2[i]));
  }
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(tmp[11]) : "r"(out[11]), "r"(CUBBS_LAZY_FP_MOD2[11]));

  asm volatile("{ .reg.pred %neg; setp.ne.u32 %neg, %0, 0;" :: "r"(borrow));
  #pragma unroll
  for (size_t i = 0u; i < 12u; ++i) {
    asm volatile("@%neg mov.b32 %0, %1;" : "+r"(out[i]) : "r"(tmp[i]));
  }
  asm volatile("}");
}

// Canonical subtraction: a, b in [0, 2p) -> out in [0, p) (exact is_zero!).
__device__ __forceinline__ static void cubbs_lfp_sub_canonical(
    const uint32_t a[12], const uint32_t b[12], uint32_t out[12]) {
  cubbs_lfp_sub_2p(a, b, out);
  cubbs_lfp_final_sub_once(out);
}

// Unreduced branchless difference: a, b in [0, 2p) -> out = a + 2p - b in (0, 4p).
__device__ __forceinline__ static void cubbs_lfp_add_2p_sub_4p(
    const uint32_t a[12], const uint32_t b[12], uint32_t out[12]) {
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(out[0]) : "r"(a[0]), "r"(CUBBS_LAZY_FP_MOD2[0]));
  #pragma unroll
  for (size_t i = 1u; i < 11u; ++i) {
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(out[i]) : "r"(a[i]), "r"(CUBBS_LAZY_FP_MOD2[i]));
  }
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(out[11]) : "r"(a[11]), "r"(CUBBS_LAZY_FP_MOD2[11]));

  asm volatile("sub.cc.u32 %0, %0, %1;" : "+r"(out[0]) : "r"(b[0]));
  #pragma unroll
  for (size_t i = 1u; i < 11u; ++i) {
    asm volatile("subc.cc.u32 %0, %0, %1;" : "+r"(out[i]) : "r"(b[i]));
  }
  asm volatile("subc.u32 %0, %0, %1;" : "+r"(out[11]) : "r"(b[11]));
}

__device__ __forceinline__ static bool cubbs_lfp_is_zero(const uint32_t x[12]) {
  uint32_t acc = x[0] | x[1];
  #pragma unroll
  for (size_t i = 2u; i < 12u; i += 2u) acc |= (x[i] | x[i + 1u]);
  return acc == 0u;
}

// Canonicalize all 4 XYZZ coordinates from [0, 2p) -> [0, p).
__device__ __forceinline__ static void cubbs_bbs_xyzz_canonicalize(
    cubbs_sppark_bucket_t &acc) {
  auto *limbs = reinterpret_cast<uint32_t *>(&acc);
  cubbs_lfp_final_sub_once(limbs + 0u);
  cubbs_lfp_final_sub_once(limbs + 12u);
  cubbs_lfp_final_sub_once(limbs + 24u);
  cubbs_lfp_final_sub_once(limbs + 36u);
}

// Rare edge-case fallback (P1 == P2 doubling or P1 == -P2 infinity).
__device__ __noinline__ static void cubbs_bbs_xyzz_madd_rare_fallback(
    cubbs_sppark_bucket_t &acc,
    const cubbs_sppark_plain_affine_t &p2,
    bool subtract,
    bool is_double) {
  if (is_double) {
    cubbs_bbs_xyzz_canonicalize(acc);
    acc.uadd(p2, subtract);
  } else {
    acc.inf();
  }
}

// Variant A (optimal for limlee_direct M=64, 256): inlined 1-mul-site lazy state machine.
__device__ __forceinline__ static void cubbs_bbs_xyzz_madd_lazy_inline(
    cubbs_sppark_bucket_t &acc,
    const cubbs_sppark_plain_affine_t &p2,
    bool subtract) {
  if (p2.is_inf()) return;
  if (acc.is_inf()) {
    acc = p2;
    acc.cneg(subtract);
    return;
  }

  fp_t p2_x, p2_y;
  {
    const fp_t *pc = reinterpret_cast<const fp_t *>(&p2);
    p2_x = pc[0];
    p2_y = pc[1];
    p2_y.cneg(subtract);
  }

  uint32_t x1[12], y1[12], zzz1[12], zz1[12];
  uint32_t x2[12], A[12], B[12], R[12], PP[12], PPP[12];
  memcpy(x1,   reinterpret_cast<const uint8_t *>(&acc) + 0u,   48u);
  memcpy(y1,   reinterpret_cast<const uint8_t *>(&acc) + 48u,  48u);
  memcpy(zzz1, reinterpret_cast<const uint8_t *>(&acc) + 96u,  48u);
  memcpy(zz1,  reinterpret_cast<const uint8_t *>(&acc) + 144u, 48u);
  memcpy(x2, &p2_x, 48u);
  memcpy(A,  &p2_y, 48u);
  #pragma unroll
  for (size_t i = 0u; i < 12u; ++i) B[i] = zzz1[i];

  int pc = -1;
  #pragma unroll 1
  do {
    cubbs_lfp_mul_2p(A, B, A);
    switch (++pc) {
    case 0:
      cubbs_lfp_sub_canonical(A, y1, R);
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) { A[i] = x2[i]; B[i] = zz1[i]; }
      break;
    case 1:
      cubbs_lfp_sub_canonical(A, x1, A);
      if (cubbs_lfp_is_zero(A)) {
        cubbs_bbs_xyzz_madd_rare_fallback(acc, p2, subtract, cubbs_lfp_is_zero(R));
        return;
      }
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) B[i] = A[i];
      break;
    case 2:
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) PP[i] = A[i];
      break;
    case 3:
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) { PPP[i] = A[i]; B[i] = zzz1[i]; }
      break;
    case 4:
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) { zzz1[i] = A[i]; A[i] = y1[i]; B[i] = PPP[i]; }
      break;
    case 5:
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) { y1[i] = A[i]; A[i] = zz1[i]; B[i] = PP[i]; }
      break;
    case 6:
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) { zz1[i] = A[i]; A[i] = x1[i]; }
      break;
    case 7:
      #pragma unroll
      for (size_t i = 0u; i < 12u; ++i) { PP[i] = A[i]; A[i] = R[i]; B[i] = R[i]; }
      break;
    case 8:
      cubbs_lfp_sub_2p(A, PPP, x1);
      cubbs_lfp_sub_2p(x1, PP, x1);
      cubbs_lfp_sub_2p(x1, PP, x1);
      cubbs_lfp_add_2p_sub_4p(PP, x1, A);
      break;
    case 9:
      cubbs_lfp_sub_2p(A, y1, y1);
      break;
    }
  } while (pc < 9);

  memcpy(reinterpret_cast<uint8_t *>(&acc) + 0u,   x1,   48u);
  memcpy(reinterpret_cast<uint8_t *>(&acc) + 48u,  y1,   48u);
  memcpy(reinterpret_cast<uint8_t *>(&acc) + 96u,  zzz1, 48u);
  memcpy(reinterpret_cast<uint8_t *>(&acc) + 144u, zz1,  48u);
}

// Variant B (optimal for limlee_split M=2048 & Pippenger Pass 4 M=65,536):
// Direct __noinline__ 8M + 2S using 78-mul symmetric sqr_2p + 0 final_sub!
__device__ __noinline__ static void cubbs_bbs_xyzz_madd_lazy_8m2s(
    cubbs_sppark_bucket_t &acc,
    const cubbs_sppark_plain_affine_t &p2,
    bool subtract) {
  if (p2.is_inf()) return;
  if (acc.is_inf()) {
    acc = p2;
    acc.cneg(subtract);
    return;
  }

  fp_t p2_x, p2_y;
  {
    const fp_t *pc = reinterpret_cast<const fp_t *>(&p2);
    p2_x = pc[0];
    p2_y = pc[1];
    p2_y.cneg(subtract);
  }

  uint32_t x1[12], y1[12], zzz1[12], zz1[12];
  uint32_t x2[12], y2[12];
  memcpy(x1,   reinterpret_cast<const uint8_t *>(&acc) + 0u,   48u);
  memcpy(y1,   reinterpret_cast<const uint8_t *>(&acc) + 48u,  48u);
  memcpy(zzz1, reinterpret_cast<const uint8_t *>(&acc) + 96u,  48u);
  memcpy(zz1,  reinterpret_cast<const uint8_t *>(&acc) + 144u, 48u);
  memcpy(x2, &p2_x, 48u);
  memcpy(y2, &p2_y, 48u);

  uint32_t s2[12], u2[12], P[12], R[12];
  cubbs_lfp_mul_2p(y2, zzz1, s2);
  cubbs_lfp_sub_canonical(s2, y1, R);
  cubbs_lfp_mul_2p(x2, zz1, u2);
  cubbs_lfp_sub_canonical(u2, x1, P);

  if (cubbs_lfp_is_zero(P)) {
    if (cubbs_lfp_is_zero(R)) {
      cubbs_bbs_xyzz_canonicalize(acc);
      acc.uadd(p2, subtract);
    } else {
      acc.inf();
    }
    return;
  }

  uint32_t PP[12], PPP[12], zzz3[12], zz3[12], Q[12], y1_ppp[12], RR[12];
  cubbs_lfp_sqr_2p(P, PP);
  cubbs_lfp_mul_2p(P, PP, PPP);
  cubbs_lfp_mul_2p(zzz1, PPP, zzz3);
  cubbs_lfp_mul_2p(zz1, PP, zz3);
  cubbs_lfp_mul_2p(x1, PP, Q);
  cubbs_lfp_mul_2p(y1, PPP, y1_ppp);
  cubbs_lfp_sqr_2p(R, RR);

  uint32_t x3[12], q_minus_x3[12], r_qx3[12], y3[12];
  cubbs_lfp_sub_2p(RR, PPP, x3);
  cubbs_lfp_sub_2p(x3, Q, x3);
  cubbs_lfp_sub_2p(x3, Q, x3);
  cubbs_lfp_add_2p_sub_4p(Q, x3, q_minus_x3);
  cubbs_lfp_mul_2p(q_minus_x3, R, r_qx3);
  cubbs_lfp_sub_2p(r_qx3, y1_ppp, y3);

  memcpy(reinterpret_cast<uint8_t *>(&acc) + 0u,   x3,   48u);
  memcpy(reinterpret_cast<uint8_t *>(&acc) + 48u,  y3,   48u);
  memcpy(reinterpret_cast<uint8_t *>(&acc) + 96u,  zzz3, 48u);
  memcpy(reinterpret_cast<uint8_t *>(&acc) + 144u, zz3,  48u);
}

#endif // CUBBS_BBS_FP_LAZY_XYZZ_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
