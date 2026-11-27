#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_FUSED_PROOF_CUH
#define CUBBS_BBS_FUSED_PROOF_CUH

// ============================================================================
// GPU-Resident Fused BBS ProofGen & ProofVerify Pipelines (ABI v2)
// ============================================================================
// Eliminates CPU crypto (H2S packing, Scalar::random, T2 term expansion,
// BLST point compression, SHA-256 challenge, scalar inversion/finalization)
// by executing the entire proof generation and proof verification pipelines
// end-to-end on the GPU in a single CUDA stream synchronization.

// ---- Device helpers: RFC 8439 ChaCha20 CSPRNG -> uniform BLS12-381 Fr ----

__device__ __forceinline__ static uint32_t cubbs_rotl32(uint32_t x, uint32_t n) {
  return (x << n) | (x >> (32u - n));
}

__device__ __forceinline__ static void cubbs_chacha20_qr(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  a += b; d ^= a; d = cubbs_rotl32(d, 16u);
  c += d; b ^= c; b = cubbs_rotl32(b, 12u);
  a += b; d ^= a; d = cubbs_rotl32(d, 8u);
  c += d; b ^= c; b = cubbs_rotl32(b, 7u);
}

// Generates a uniform BLS12-381 Fr scalar (in little-endian 32 bytes) from a
// 32-byte OS-seeded key, unique (row, scalar_idx) nonce, using 384 bits (12
// ChaCha20 words) reduced mod r (statistical distance < 2^-128).
__device__ static void cubbs_bbs_chacha20_fr_le_device(
    const uint32_t seed_words[8], uint64_t row, uint32_t scalar_idx,
    bool force_nonzero, uint8_t out_le[32]) {
  uint32_t st[16];
  st[0] = 0x61707865u;
  st[1] = 0x3320646eu;
  st[2] = 0x79622d32u;
  st[3] = 0x6b206574u;
  #pragma unroll
  for (int i = 0; i < 8; ++i) st[4 + i] = seed_words[i];
  st[12] = 0u;
  st[13] = static_cast<uint32_t>(row & 0xffffffffull);
  st[14] = static_cast<uint32_t>(row >> 32u);
  st[15] = scalar_idx;

  uint32_t x[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) x[i] = st[i];

  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    cubbs_chacha20_qr(x[0], x[4], x[8],  x[12]);
    cubbs_chacha20_qr(x[1], x[5], x[9],  x[13]);
    cubbs_chacha20_qr(x[2], x[6], x[10], x[14]);
    cubbs_chacha20_qr(x[3], x[7], x[11], x[15]);
    cubbs_chacha20_qr(x[0], x[5], x[10], x[15]);
    cubbs_chacha20_qr(x[1], x[6], x[11], x[12]);
    cubbs_chacha20_qr(x[2], x[7], x[8],  x[13]);
    cubbs_chacha20_qr(x[3], x[4], x[9],  x[14]);
  }

  uint32_t words12[12];
  #pragma unroll
  for (int i = 0; i < 12; ++i) words12[i] = x[i] + st[i];

  uint64_t limbs[4];
  bls12_381_fr_reduce_words12_to_le64(words12, limbs);
  if (force_nonzero && (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0ull) {
    limbs[0] = 1ull;
  }
  memcpy(out_le, limbs, 32u);
}

__device__ __forceinline__ static void cubbs_bbs_store_fr_be(
    cubbs_sppark_scalar_t val, uint8_t out_be[32]) {
  val.from();
  uint8_t le[32];
  memcpy(le, &val, 32u);
  #pragma unroll
  for (int i = 0; i < 32; ++i) out_be[i] = le[31 - i];
}

__device__ __forceinline__ static void cubbs_bbs_sha256_push_u64_be(
    CuBbsSha256Ctx &ctx, uint64_t value) {
  ctx.push_word(static_cast<uint32_t>(value >> 32u));
  ctx.push_word(static_cast<uint32_t>(value & 0xffffffffull));
}

// Computes RFC draft-irtf-cfrg-bbs-signatures proof challenge c = H2S(input, dst)
// directly into a canonical LE 32-byte scalar with zero temporary buffers.
__device__ static void cubbs_bbs_compute_proof_challenge_le_device(
    const uint32_t *disclosed_indexes, size_t disclosed_count,
    const uint8_t *disclosed_scalars_le_base, size_t disclosed_scalar_stride,
    bool disclosed_uses_full_b_layout,
    const uint8_t abar48[48], const uint8_t bbar48[48], const uint8_t d48[48],
    const uint8_t t1_48[48], const uint8_t t2_48[48],
    const uint8_t domain_be[32],
    const uint8_t *ph, size_t ph_len,
    const uint8_t *challenge_dst, size_t challenge_dst_len,
    uint8_t out_le[32]) {
  CuBbsSha256Ctx ctx;
  uint32_t b0[8];
  uint32_t b1[8];
  uint32_t b2[8];

  ctx.init_after_zpad();
  cubbs_bbs_sha256_push_u64_be(ctx, static_cast<uint64_t>(disclosed_count));
  size_t k = 0u;
  // Direct 8-item block-unrolled loop (block_len == 8 -> 8 * 40B = 5 exact SHA-256 blocks):
  // Zero switch(), zero block_len branches, direct 128-bit coalesced uint4 loads!
  auto load_scalar_be = [&](size_t kk, uint32_t out_be[8]) {
    const uint32_t idx_k = disclosed_indexes[kk];
    const uint8_t *s_le = disclosed_uses_full_b_layout
                              ? (disclosed_scalars_le_base + ((2u + idx_k) * 32u))
                              : (disclosed_scalars_le_base + (kk * disclosed_scalar_stride));
    const uint4 *v = reinterpret_cast<const uint4 *>(s_le);
    const uint4 lo = v[0];
    const uint4 hi = v[1];
    out_be[0] = hi.w; out_be[1] = hi.z; out_be[2] = hi.y; out_be[3] = hi.x;
    out_be[4] = lo.w; out_be[5] = lo.z; out_be[6] = lo.y; out_be[7] = lo.x;
    return idx_k;
  };
  for (; k + 8u <= disclosed_count; k += 8u) {
    uint32_t sw[8];
    // Item 0 (word 2 -> 12)
    ctx.w2 = 0u; ctx.w3 = load_scalar_be(k + 0u, sw);
    ctx.w4 = sw[0]; ctx.w5 = sw[1]; ctx.w6 = sw[2]; ctx.w7 = sw[3];
    ctx.w8 = sw[4]; ctx.w9 = sw[5]; ctx.w10 = sw[6]; ctx.w11 = sw[7];
    // Item 1 (word 12 -> compress -> word 6)
    ctx.w12 = 0u; ctx.w13 = load_scalar_be(k + 1u, sw);
    ctx.w14 = sw[0]; ctx.w15 = sw[1];
    ctx.compress();
    ctx.w0 = sw[2]; ctx.w1 = sw[3]; ctx.w2 = sw[4]; ctx.w3 = sw[5]; ctx.w4 = sw[6]; ctx.w5 = sw[7];
    // Item 2 (word 6 -> compress -> word 0)
    ctx.w6 = 0u; ctx.w7 = load_scalar_be(k + 2u, sw);
    ctx.w8 = sw[0]; ctx.w9 = sw[1]; ctx.w10 = sw[2]; ctx.w11 = sw[3];
    ctx.w12 = sw[4]; ctx.w13 = sw[5]; ctx.w14 = sw[6]; ctx.w15 = sw[7];
    ctx.compress();
    // Item 3 (word 0 -> word 10)
    ctx.w0 = 0u; ctx.w1 = load_scalar_be(k + 3u, sw);
    ctx.w2 = sw[0]; ctx.w3 = sw[1]; ctx.w4 = sw[2]; ctx.w5 = sw[3];
    ctx.w6 = sw[4]; ctx.w7 = sw[5]; ctx.w8 = sw[6]; ctx.w9 = sw[7];
    // Item 4 (word 10 -> compress -> word 4)
    ctx.w10 = 0u; ctx.w11 = load_scalar_be(k + 4u, sw);
    ctx.w12 = sw[0]; ctx.w13 = sw[1]; ctx.w14 = sw[2]; ctx.w15 = sw[3];
    ctx.compress();
    ctx.w0 = sw[4]; ctx.w1 = sw[5]; ctx.w2 = sw[6]; ctx.w3 = sw[7];
    // Item 5 (word 4 -> word 14)
    ctx.w4 = 0u; ctx.w5 = load_scalar_be(k + 5u, sw);
    ctx.w6 = sw[0]; ctx.w7 = sw[1]; ctx.w8 = sw[2]; ctx.w9 = sw[3];
    ctx.w10 = sw[4]; ctx.w11 = sw[5]; ctx.w12 = sw[6]; ctx.w13 = sw[7];
    // Item 6 (word 14 -> compress -> word 8)
    ctx.w14 = 0u; ctx.w15 = load_scalar_be(k + 6u, sw);
    ctx.compress();
    ctx.w0 = sw[0]; ctx.w1 = sw[1]; ctx.w2 = sw[2]; ctx.w3 = sw[3];
    ctx.w4 = sw[4]; ctx.w5 = sw[5]; ctx.w6 = sw[6]; ctx.w7 = sw[7];
    // Item 7 (word 8 -> compress -> word 2)
    ctx.w8 = 0u; ctx.w9 = load_scalar_be(k + 7u, sw);
    ctx.w10 = sw[0]; ctx.w11 = sw[1]; ctx.w12 = sw[2]; ctx.w13 = sw[3];
    ctx.w14 = sw[4]; ctx.w15 = sw[5];
    ctx.compress();
    ctx.w0 = sw[6]; ctx.w1 = sw[7];
    ctx.total_len += 320ull;
  }
  for (; k < disclosed_count; ++k) {
    const uint32_t idx = disclosed_indexes[k];
    cubbs_bbs_sha256_push_u64_be(ctx, static_cast<uint64_t>(idx));
    const uint8_t *s_le = disclosed_uses_full_b_layout
                              ? (disclosed_scalars_le_base + ((2u + idx) * 32u))
                              : (disclosed_scalars_le_base + (k * disclosed_scalar_stride));
    ctx.update_scalar_le_as_be(s_le);
  }
  ctx.update(abar48, 48u);
  ctx.update(bbar48, 48u);
  ctx.update(d48, 48u);
  ctx.update(t1_48, 48u);
  ctx.update(t2_48, 48u);
  ctx.update(domain_be, 32u);
  cubbs_bbs_sha256_push_u64_be(ctx, static_cast<uint64_t>(ph_len));
  if (ph_len > 0u) ctx.update(ph, ph_len);
  ctx.push_byte(0u);
  ctx.push_byte(48u);
  ctx.push_byte(0u);
  ctx.update(challenge_dst, challenge_dst_len);
  ctx.push_byte(static_cast<uint8_t>(challenge_dst_len));
  ctx.finalize_words(b0);

  ctx.init();
  ctx.w0 = b0[0]; ctx.w1 = b0[1]; ctx.w2 = b0[2]; ctx.w3 = b0[3];
  ctx.w4 = b0[4]; ctx.w5 = b0[5]; ctx.w6 = b0[6]; ctx.w7 = b0[7];
  ctx.block_len = 32u;
  ctx.total_len = 32ull;
  ctx.push_byte(1u);
  ctx.update(challenge_dst, challenge_dst_len);
  ctx.push_byte(static_cast<uint8_t>(challenge_dst_len));
  ctx.finalize_words(b1);

  ctx.init();
  ctx.w0 = b0[0] ^ b1[0]; ctx.w1 = b0[1] ^ b1[1];
  ctx.w2 = b0[2] ^ b1[2]; ctx.w3 = b0[3] ^ b1[3];
  ctx.w4 = b0[4] ^ b1[4]; ctx.w5 = b0[5] ^ b1[5];
  ctx.w6 = b0[6] ^ b1[6]; ctx.w7 = b0[7] ^ b1[7];
  ctx.block_len = 32u;
  ctx.total_len = 32ull;
  ctx.push_byte(2u);
  ctx.update(challenge_dst, challenge_dst_len);
  ctx.push_byte(static_cast<uint8_t>(challenge_dst_len));
  ctx.finalize_words(b2);

  uint32_t words12[12];
  #pragma unroll
  for (int i = 0; i < 8; ++i) words12[i] = b1[i];
  #pragma unroll
  for (int i = 0; i < 4; ++i) words12[8 + i] = b2[i];

  uint64_t limbs[4];
  bls12_381_fr_reduce_words12_to_le64(words12, limbs);
  memcpy(out_le, limbs, 32u);
}

// ============================================================================
// GPU-Resident Fused ProofGen Kernels
// ============================================================================

// Fill slots 0 (1) and 1 (domain_le) of b_scalars_le, AND fold r2 into
// d_scalars_le slots 0 (r2) and 1 (r2 * domain) so Lim-Lee computes D = r2*B directly!
__global__ static void cubbs_bbs_proof_gen_init_bd_header_scalars_kernel(
    const uint8_t *domains_be, size_t domain_stride,
    const uint8_t *row_state_le,
    size_t item_count, size_t term_count,
    uint8_t *b_scalars_le, uint8_t *d_scalars_le, uint32_t *invalid_out) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= item_count) return;
  uint8_t *row_b = b_scalars_le + (row * term_count * 32u);
  uint8_t *row_d = d_scalars_le + (row * term_count * 32u);
  memset(row_b, 0, 32u);
  row_b[0] = 1u;

  const uint8_t *r2_le = row_state_le + ((row * 6u + 1u) * 32u);
  memcpy(row_d, r2_le, 32u);

  const uint8_t *dom_be = domains_be + (domain_stride == 0u ? 0u : row * domain_stride);
  if (!cubbs_bbs_scalar_be_is_canonical(dom_be)) {
    atomicExch(invalid_out, 1u);
    memset(row_b + 32u, 0, 32u);
    memset(row_d + 32u, 0, 32u);
    return;
  }
  uint8_t dom_le[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) dom_le[i] = dom_be[31 - i];
  memcpy(row_b + 32u, dom_le, 32u);

  cubbs_sppark_scalar_t fr_dom, fr_r2;
  memcpy(&fr_dom, dom_le, 32u); fr_dom.to();
  memcpy(&fr_r2, r2_le, 32u);   fr_r2.to();
  cubbs_sppark_scalar_t fr_r2_dom = fr_dom * fr_r2;
  fr_r2_dom.from();
  memcpy(row_d + 32u, &fr_r2_dom, 32u);
}

// Fused H2S + r2-scaling: writes unscaled m_j to b_scalars_le (for Stage 5 challenge
// and m_cap[u]) and r2*m_j to d_scalars_le (so Lim-Lee computes D = r2*B directly!).
__global__ static void cubbs_bbs_proof_gen_h2s_and_scale_kernel(
    const uint8_t *messages, const uint64_t *offsets, const uint64_t *lengths,
    size_t uniform_len, const uint8_t *row_state_le,
    size_t item_count, size_t message_count, size_t term_count,
    const uint8_t *message_dst, size_t message_dst_len,
    uint8_t *b_scalars_le, uint8_t *d_scalars_le) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = item_count * message_count;
  if (idx >= total) return;

  size_t msg_off = 0u;
  size_t msg_len = uniform_len;
  if (lengths != nullptr) {
    msg_off = static_cast<size_t>(offsets[idx]);
    msg_len = static_cast<size_t>(lengths[idx]);
  } else if (offsets != nullptr) {
    const auto *offs32 = reinterpret_cast<const uint32_t *>(offsets);
    msg_off = static_cast<size_t>(offs32[idx]);
    msg_len = static_cast<size_t>(offs32[idx + 1u] - offs32[idx]);
  } else {
    msg_off = idx * uniform_len;
  }

  const size_t row = idx / message_count;
  const size_t msg = idx - (row * message_count);
  uint32_t uniform_words[12];
  expand_message_xmd_sha256_48_words_device(
      messages + msg_off, msg_len,
      message_dst, message_dst_len, uniform_words);
  uint64_t le[4];
  bls12_381_fr_reduce_words12_to_le64(uniform_words, le);
  const size_t slot_off = (row * term_count + 2u + msg) * 32u;
  bls12_381_fr_store_le(le, b_scalars_le + slot_off);

  cubbs_sppark_scalar_t fr_m, fr_r2;
  memcpy(&fr_m, le, 32u);
  memcpy(&fr_r2, row_state_le + ((row * 6u + 1u) * 32u), 32u);
  fr_m.to();
  fr_r2.to();
  cubbs_sppark_scalar_t fr_scaled = fr_m * fr_r2;
  fr_scaled.from();
  memcpy(d_scalars_le + slot_off, &fr_scaled, 32u);
}

// Normalize D = r2*B from xyzz -> plain affine AND simultaneously compress D
// directly into pts5_compressed[row * 240 + 96..144] (zero duplicate inversion!).
__global__ static void cubbs_bbs_xyzz_to_plain_affine_and_compress_d_kernel(
    const cubbs_sppark_bucket_t *d_xyzz_in,
    cubbs_sppark_plain_affine_t *d_affine_out,
    uint8_t *pts5_compressed_out,
    size_t count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool valid = idx < count;

  cubbs_sppark_bucket_t pt;
  pt.inf();
  bool finite = false;
  fp_t zzz = fp_t::one();
  if (valid) {
    pt = d_xyzz_in[idx];
    finite = !pt.is_inf();
    if (finite) {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      zzz = coords[2];
    }
  }
  const fp_t inv_zzz = zzz.reciprocal();

  if (valid) {
    fp_t affine_x, affine_y;
    if (!finite) {
      affine_x.zero();
      affine_y.zero();
    } else {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      affine_y = inv_zzz * coords[1];
      const fp_t inv_z = inv_zzz * coords[3];
      affine_x = sqr(inv_z) * coords[0];
    }
    d_affine_out[idx] = cubbs_sppark_plain_affine_t{affine_x, affine_y};
    cubbs_bbs_compress_affine_fp_to_be48(
        affine_x, affine_y, finite, pts5_compressed_out + (idx * 240u) + 96u);
  }
}

// Compress 4 proof points [Abar, Bbar, T1, T2] directly into pts5_compressed
// at offsets [0, 48, 144, 192] (slot 96 already holds D!).
__global__ static void cubbs_bbs_g1_xyzz_to_compressed_pts4_into_pts5_kernel(
    const cubbs_sppark_bucket_t *pts4_in,
    size_t item_count,
    uint8_t *pts5_compressed_out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = item_count * 4u;
  const bool valid = idx < total;

  cubbs_sppark_bucket_t pt;
  pt.inf();
  bool finite = false;
  fp_t zzz = fp_t::one();
  if (valid) {
    pt = pts4_in[idx];
    finite = !pt.is_inf();
    if (finite) {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      zzz = coords[2];
    }
  }
  const fp_t inv_zzz = zzz.reciprocal();

  if (valid) {
    fp_t affine_x, affine_y;
    if (finite) {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pt);
      affine_y = inv_zzz * coords[1];
      const fp_t inv_z = inv_zzz * coords[3];
      affine_x = sqr(inv_z) * coords[0];
    }
    const size_t row = idx >> 2u;
    const uint32_t sub = static_cast<uint32_t>(idx & 3u);
    const uint32_t dst_slot = (sub < 2u) ? sub : (sub + 1u); // 0->0, 1->1, 2->3, 3->4
    cubbs_bbs_compress_affine_fp_to_be48(
        affine_x, affine_y, finite,
        pts5_compressed_out + (row * 240u) + (static_cast<size_t>(dst_slot) * 48u));
  }
}

// Per-proof GPU ChaCha20 CSPRNG + 6 GLV-split multiplier preparation (D = r2*B folded!):
//   row_state[row * 6 + 0..5] = [r1, r2, e_tilde, r1_tilde, r3_tilde, e] (LE)
//   glv_splits[row * 6 + 0..5] = GLV(rem||quot) of:
//     [0] r1*r2           (for Abar = (r1*r2)*A)
//     [1] r1              (for Bbar_pos = r1*D)
//     [2] r1*r2*e         (for Bbar_neg = -(r1*r2*e)*A)
//     [3] r1*r2*e_tilde   (for T1_a = (r1*r2*e_tilde)*A)
//     [4] r1_tilde        (for T1_b = r1_tilde*D)
//     [5] r3_tilde        (for T2_var = r3_tilde*D)
__global__ static void cubbs_bbs_proof_gen_prep_scalars_kernel(
    const uint8_t *signatures_e_be,
    const uint8_t *csprng_seed,
    size_t item_count,
    uint8_t *row_state_le,    // item_count * 6 * 32
    uint8_t *glv_splits_le,   // item_count * 6 * 32
    uint32_t *invalid_out) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= item_count) return;

  const uint8_t *e_be = signatures_e_be + (row * 32u);
  if (!cubbs_bbs_scalar_be_is_canonical(e_be)) {
    atomicExch(invalid_out, 1u);
    memset(row_state_le + (row * 6u * 32u), 0, 6u * 32u);
    memset(glv_splits_le + (row * 6u * 32u), 0, 6u * 32u);
    return;
  }

  uint32_t seed_w[8];
  memcpy(seed_w, csprng_seed, 32u);

  uint8_t r1_le[32], r2_le[32], et_le[32], r1t_le[32], r3t_le[32], e_le[32];
  cubbs_bbs_chacha20_fr_le_device(seed_w, row, 0u, true, r1_le);
  cubbs_bbs_chacha20_fr_le_device(seed_w, row, 1u, true, r2_le);
  cubbs_bbs_chacha20_fr_le_device(seed_w, row, 2u, false, et_le);
  cubbs_bbs_chacha20_fr_le_device(seed_w, row, 3u, false, r1t_le);
  cubbs_bbs_chacha20_fr_le_device(seed_w, row, 4u, false, r3t_le);
  #pragma unroll
  for (int i = 0; i < 32; ++i) e_le[i] = e_be[31 - i];

  uint8_t *st = row_state_le + (row * 6u * 32u);
  memcpy(st + 0u * 32u, r1_le, 32u);
  memcpy(st + 1u * 32u, r2_le, 32u);
  memcpy(st + 2u * 32u, et_le, 32u);
  memcpy(st + 3u * 32u, r1t_le, 32u);
  memcpy(st + 4u * 32u, r3t_le, 32u);
  memcpy(st + 5u * 32u, e_le, 32u);

  cubbs_sppark_scalar_t fr_r1, fr_r2, fr_et, fr_e;
  memcpy(&fr_r1, r1_le, 32u);   fr_r1.to();
  memcpy(&fr_r2, r2_le, 32u);   fr_r2.to();
  memcpy(&fr_et, et_le, 32u);   fr_et.to();
  memcpy(&fr_e, e_le, 32u);     fr_e.to();

  const cubbs_sppark_scalar_t fr_r1r2 = fr_r1 * fr_r2;
  const cubbs_sppark_scalar_t fr_m2 = fr_r1r2 * fr_e;
  const cubbs_sppark_scalar_t fr_m3 = fr_r1r2 * fr_et;

  auto store_split_fr = [&](uint32_t slot, cubbs_sppark_scalar_t val) {
    val.from();
    uint8_t le[32];
    memcpy(le, &val, 32u);
    uint8_t *dst = glv_splits_le + ((row * 6u + slot) * 32u);
    cubbs_bbs_scalar_div_by_z2_device(le, dst, dst + 16u);
  };
  auto store_split_le = [&](uint32_t slot, const uint8_t le[32]) {
    uint8_t *dst = glv_splits_le + ((row * 6u + slot) * 32u);
    cubbs_bbs_scalar_div_by_z2_device(le, dst, dst + 16u);
  };
  store_split_fr(0u, fr_r1r2);
  store_split_le(1u, r1_le);
  store_split_fr(2u, fr_m2);
  store_split_fr(3u, fr_m3);
  store_split_le(4u, r1t_le);
  store_split_le(5u, r3t_le);
}

// Parallel ChaCha20 generator for m_tilde[row * U + u] (all B*U scalars in parallel!).
__global__ static void cubbs_bbs_proof_gen_m_tilde_kernel(
    const uint8_t *csprng_seed, size_t item_count, size_t undisclosed_count,
    uint8_t *m_tilde_le) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = item_count * undisclosed_count;
  if (idx >= total) return;
  const size_t row = idx / undisclosed_count;
  const uint32_t u = static_cast<uint32_t>(idx - (row * undisclosed_count));
  uint32_t seed_w[8];
  memcpy(seed_w, csprng_seed, 32u);
  cubbs_bbs_chacha20_fr_le_device(seed_w, row, 5u + u, false, m_tilde_le + (idx * 32u));
}

// Build [1*P, 2*P, ..., 8*P] in XYZZ for each affine base point using
// single-mul dbl_u + mixed-affine uadd (0 stack spill).
__global__ static void cubbs_bbs_build_radix16_tables_xyzz_kernel(
    const cubbs_sppark_plain_affine_t *bases_in,
    cubbs_sppark_bucket_t *tables_xyzz_out,
    size_t point_count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= point_count) return;
  const cubbs_sppark_plain_affine_t base = bases_in[idx];
  cubbs_sppark_bucket_t *out = tables_xyzz_out + (idx * 8u);
  cubbs_sppark_bucket_t acc;
  acc = base;
  out[0] = acc;
  if (!acc.is_inf()) {
    cubbs_bbs_xyzz_dbl_u(acc);
  }
  out[1] = acc;
  #pragma unroll 1
  for (uint32_t d = 2u; d < 8u; ++d) {
    acc.uadd(base, false);
    out[d] = acc;
  }
}

static cudaError_t cubbs_bbs_launch_build_radix16_affine_tables(
    const cubbs_sppark_plain_affine_t *d_bases,
    size_t point_count,
    cubbs_sppark_plain_affine_t *d_tables_aff_out,
    cudaStream_t stream) {
  if (point_count == 0u) return cudaSuccess;
  const size_t total_entries = point_count * 8u;
  auto &msm_arena = cubbs_bbs_thread_msm_arena();
  cudaError_t err = msm_arena.reserve_and_reset(
      CuBbsDeviceArena::align_up(total_entries * sizeof(cubbs_sppark_bucket_t)));
  if (err != cudaSuccess) return err;
  cubbs_sppark_bucket_t *d_tmp_xyzz = msm_arena.alloc<cubbs_sppark_bucket_t>(total_entries);
  {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((point_count + t - 1u) / t);
    cubbs_bbs_build_radix16_tables_xyzz_kernel<<<b, t, 0, stream>>>(
        d_bases, d_tmp_xyzz, point_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    constexpr uint32_t t = 64u;
    const uint32_t b = static_cast<uint32_t>((total_entries + t - 1u) / t);
    cubbs_bbs_xyzz_to_plain_affine_batch64_kernel<<<b, t, 0, stream>>>(
        d_tmp_xyzz, d_tables_aff_out, total_entries);
    err = cudaGetLastError();
  }
  return err;
}

// 4-way Straus simultaneous multi-base Signed Radix-16 GLV kernel for ProofGen
// in a=0 Jacobian coordinates (2M+5S dbl-2009-l + 7M+4S madd-2007-bl).
// Role 0 (D = r2*B) is eliminated by folding r2 directly into the Lim-Lee B-MSM!
__global__ static void __launch_bounds__(64, 5)
cubbs_bbs_proof_gen_straus4_jac_kernel(
    const cubbs_sppark_plain_affine_t *d_tables_aff,
    const cubbs_sppark_plain_affine_t *a_tables_aff,
    const uint8_t *glv_splits_le,
    cubbs_sppark_bucket_t *pts4_xyzz,
    size_t item_count) {
  __shared__ uint64_t s_digs[64u][4u][3u];

  const uint32_t tid = threadIdx.x;
  const size_t proof = static_cast<size_t>(blockIdx.x) * 64u + tid;
  if (proof >= item_count) return;
  const uint32_t role = blockIdx.y;

  uint32_t slot0 = 0u, slot1 = 0u;
  uint32_t n_passes = 2u;
  bool base0_is_d = true;
  bool base1_is_d = false;
  bool base1_neg = false;
  uint32_t out_slot = 0u;

  if (role == 0u) {
    slot0 = 0u; n_passes = 2u; base0_is_d = false; out_slot = 0u; // Abar = (r1*r2)*A
  } else if (role == 1u) {
    slot0 = 1u; slot1 = 2u; n_passes = 4u; base0_is_d = true; base1_is_d = false; base1_neg = true; out_slot = 1u; // Bbar = r1*D - (r1*r2*e)*A
  } else if (role == 2u) {
    slot0 = 3u; slot1 = 4u; n_passes = 4u; base0_is_d = false; base1_is_d = true; base1_neg = false; out_slot = 2u; // T1 = (r1*r2*e_tilde)*A + r1_tilde*D
  } else {
    slot0 = 5u; n_passes = 2u; base0_is_d = true; out_slot = 3u; // T2_var = r3_tilde*D
  }

  {
    const uint8_t *c0 = glv_splits_le + ((proof * 6u + slot0) * 32u);
    uint64_t r[3], q[3];
    cubbs_bbs_pack_signed_radix16(c0, r);
    cubbs_bbs_pack_signed_radix16(c0 + 16u, q);
    #pragma unroll
    for (int k = 0; k < 3; ++k) { s_digs[tid][0][k] = r[k]; s_digs[tid][1][k] = q[k]; }
  }
  if (n_passes == 4u) {
    const uint8_t *c1 = glv_splits_le + ((proof * 6u + slot1) * 32u);
    uint64_t r[3], q[3];
    cubbs_bbs_pack_signed_radix16(c1, r);
    cubbs_bbs_pack_signed_radix16(c1 + 16u, q);
    #pragma unroll
    for (int k = 0; k < 3; ++k) { s_digs[tid][2][k] = r[k]; s_digs[tid][3][k] = q[k]; }
  }

  const cubbs_sppark_plain_affine_t *tbl0 =
      (base0_is_d ? d_tables_aff : a_tables_aff) + (proof * 8u);
  const cubbs_sppark_plain_affine_t *tbl1 =
      (base1_is_d ? d_tables_aff : a_tables_aff) + (proof * 8u);

  CuBbsG1Jac acc;
  acc.inf();
  #pragma unroll 1
  for (int window = 32; window >= 0; --window) {
    if (!acc.is_inf()) {
      #pragma unroll 1
      for (uint32_t doubling = 0u; doubling < 4u; ++doubling) {
        cubbs_bbs_jac_dbl(acc);
      }
    }
    #pragma unroll 1
    for (uint32_t pass = 0u; pass < n_passes; ++pass) {
      const int32_t d = cubbs_bbs_signed_radix16_digit(
          s_digs[tid][pass], static_cast<uint32_t>(window));
      if (d != 0) {
        const uint32_t mag = static_cast<uint32_t>(d < 0 ? -d : d);
        const cubbs_sppark_plain_affine_t *tbl = (pass < 2u) ? tbl0 : tbl1;
        cubbs_sppark_plain_affine_t p = tbl[mag - 1u];
        bool neg = (d < 0) ^ ((pass >= 2u) && base1_neg);
        if ((pass & 1u) != 0u) {
          if (!p.is_inf()) {
            fp_t beta_sq;
            const uint64_t beta_sq_limbs[6] = {
                0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
                0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
            memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
            fp_t *px = reinterpret_cast<fp_t *>(&p);
            px[0] = px[0] * beta_sq;
          }
          neg = !neg;
        }
        cubbs_bbs_jac_madd(acc, p, neg);
      }
    }
  }
  pts4_xyzz[proof * 4u + out_slot] = cubbs_bbs_jac_to_xyzz(acc);
}

// 2-way Straus simultaneous multi-base Signed Radix-16 GLV kernel for ProofVerify
// in a=0 Jacobian coordinates (2M+5S dbl-2009-l + 7M+4S madd-2007-bl).
__global__ static void __launch_bounds__(64, 5)
cubbs_bbs_proof_verify_straus2_affine_u_kernel(
    const cubbs_sppark_plain_affine_t *pts_tables_aff,
    const uint8_t *proof_glv_splits_le,
    size_t item_count,
    cubbs_sppark_bucket_t *t1_xyzz,
    cubbs_sppark_bucket_t *t2_xyzz) {
  __shared__ uint64_t s_digs[64u][6u][3u];

  const uint32_t tid = threadIdx.x;
  const size_t proof = static_cast<size_t>(blockIdx.x) * 64u + tid;
  if (proof >= item_count) return;
  const bool is_t1 = (blockIdx.y == 0u);
  const uint32_t n_passes = is_t1 ? 6u : 2u;

  if (is_t1) {
    #pragma unroll 1
    for (uint32_t s = 0u; s < 3u; ++s) {
      const uint8_t *comp = proof_glv_splits_le + ((proof * 4u + s) * 32u);
      uint64_t r[3], q[3];
      cubbs_bbs_pack_signed_radix16(comp, r);
      cubbs_bbs_pack_signed_radix16(comp + 16u, q);
      #pragma unroll
      for (int k = 0; k < 3; ++k) {
        s_digs[tid][s * 2u + 0u][k] = r[k];
        s_digs[tid][s * 2u + 1u][k] = q[k];
      }
    }
  } else {
    const uint8_t *comp = proof_glv_splits_le + ((proof * 4u + 3u) * 32u);
    uint64_t r[3], q[3];
    cubbs_bbs_pack_signed_radix16(comp, r);
    cubbs_bbs_pack_signed_radix16(comp + 16u, q);
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      s_digs[tid][0u][k] = r[k];
      s_digs[tid][1u][k] = q[k];
    }
  }

  const cubbs_sppark_plain_affine_t *proof_tbl = pts_tables_aff + (proof * 3u * 8u);
  CuBbsG1Jac acc;
  acc.inf();
  #pragma unroll 1
  for (int window = 32; window >= 0; --window) {
    if (!acc.is_inf()) {
      #pragma unroll 1
      for (uint32_t doubling = 0u; doubling < 4u; ++doubling) {
        cubbs_bbs_jac_dbl(acc);
      }
    }
    #pragma unroll 1
    for (uint32_t pass = 0u; pass < n_passes; ++pass) {
      const int32_t d = cubbs_bbs_signed_radix16_digit(
          s_digs[tid][pass], static_cast<uint32_t>(window));
      if (d != 0) {
        const uint32_t mag = static_cast<uint32_t>(d < 0 ? -d : d);
        const uint32_t base_idx = is_t1 ? (pass >> 1u) : 2u;
        cubbs_sppark_plain_affine_t p = proof_tbl[base_idx * 8u + (mag - 1u)];
        bool neg = (d < 0);
        if ((pass & 1u) != 0u) {
          if (!p.is_inf()) {
            fp_t beta_sq;
            const uint64_t beta_sq_limbs[6] = {
                0x30f1361b798a64e8ull, 0xf3b8ddab7ece5a2aull, 0x16a8ca3ac61577f7ull,
                0xc26a2ff874fd029bull, 0x3636b76660701c6eull, 0x051ba4ab241b6160ull};
            memcpy(&beta_sq, beta_sq_limbs, sizeof(beta_sq));
            fp_t *px = reinterpret_cast<fp_t *>(&p);
            px[0] = px[0] * beta_sq;
          }
          neg = !neg;
        }
        cubbs_bbs_jac_madd(acc, p, neg);
      }
    }
  }

  const cubbs_sppark_bucket_t acc_xyzz = cubbs_bbs_jac_to_xyzz(acc);
  if (is_t1) {
    t1_xyzz[proof] = acc_xyzz;
  } else {
    cubbs_sppark_bucket_t t2 = t2_xyzz[proof];
    cubbs_bbs_limlee_xyzz_add_noinline(t2, acc_xyzz);
    t2_xyzz[proof] = t2;
  }
}

__global__ static void cubbs_bbs_proof_gen_add_t2_common_kernel(
    cubbs_sppark_bucket_t *pts4_xyzz,
    const cubbs_sppark_bucket_t *t2_common_xyzz,
    size_t item_count) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < item_count) {
    pts4_xyzz[row * 4u + 3u].uadd(t2_common_xyzz[row]);
  }
}

// Compute challenge c, r3 = r2^-1, finalize all response scalars, and assemble
// the complete RFC wire proof directly into proofs_out[row * proof_stride].
__global__ static void cubbs_bbs_proof_gen_challenge_finalize_kernel(
    const uint32_t *disclosed_indexes, size_t disclosed_count,
    const uint32_t *undisclosed_indexes, size_t undisclosed_count,
    const uint8_t *b_scalars_le, size_t term_count,
    const uint8_t *row_state_le,
    const uint8_t *m_tilde_le,
    const uint8_t *pts5_compressed, // item_count * 240: [Abar||Bbar||D||T1||T2]
    const uint8_t *domains_be, size_t domain_stride,
    const uint8_t *packed_ph, const uint64_t *ph_offsets, const uint64_t *ph_lengths,
    const uint8_t *challenge_dst, size_t challenge_dst_len,
    size_t item_count, size_t proof_stride,
    uint8_t *proofs_out) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool valid = row < item_count;

  // All 32 lanes of every warp MUST execute fr_t::reciprocal() uniformly!
  cubbs_sppark_scalar_t fr_r2 = cubbs_sppark_scalar_t::one();
  if (valid) {
    memcpy(&fr_r2, row_state_le + ((row * 6u + 1u) * 32u), 32u);
    fr_r2.to();
  }
  const cubbs_sppark_scalar_t fr_r3 = fr_r2.reciprocal();
  if (!valid) return;

  const uint8_t *pts5 = pts5_compressed + (row * 240u);
  const uint8_t *dom_be = domains_be + (domain_stride == 0u ? 0u : row * domain_stride);
  const uint8_t *ph = (ph_lengths[row] > 0u) ? (packed_ph + ph_offsets[row]) : nullptr;

  uint8_t c_le[32];
  cubbs_bbs_compute_proof_challenge_le_device(
      disclosed_indexes, disclosed_count,
      b_scalars_le + (row * term_count * 32u), 32u, true,
      pts5 + 0u, pts5 + 48u, pts5 + 96u, pts5 + 144u, pts5 + 192u,
      dom_be, ph, static_cast<size_t>(ph_lengths[row]),
      challenge_dst, challenge_dst_len, c_le);

  cubbs_sppark_scalar_t fr_c;
  memcpy(&fr_c, c_le, 32u);
  fr_c.to();

  const uint8_t *st = row_state_le + (row * 6u * 32u);
  cubbs_sppark_scalar_t fr_r1, fr_et, fr_r1t, fr_r3t, fr_e;
  memcpy(&fr_r1, st + 0u * 32u, 32u);  fr_r1.to();
  memcpy(&fr_et, st + 2u * 32u, 32u);  fr_et.to();
  memcpy(&fr_r1t, st + 3u * 32u, 32u); fr_r1t.to();
  memcpy(&fr_r3t, st + 4u * 32u, 32u); fr_r3t.to();
  memcpy(&fr_e, st + 5u * 32u, 32u);   fr_e.to();

  uint8_t *out = proofs_out + (row * proof_stride);
  // [0..144]: Abar || Bbar || D
  memcpy(out, pts5, 144u);
  // [144..176]: e_cap = e_tilde + e * c
  cubbs_bbs_store_fr_be(fr_et + (fr_e * fr_c), out + 144u);
  // [176..208]: r1_cap = r1_tilde - r1 * c
  cubbs_bbs_store_fr_be(fr_r1t - (fr_r1 * fr_c), out + 176u);
  // [208..240]: r3_cap = r3_tilde - r3 * c
  cubbs_bbs_store_fr_be(fr_r3t - (fr_r3 * fr_c), out + 208u);
  // [240 .. 240 + U*32]: m_cap[u] = m_tilde[u] + m[undisclosed[u]] * c
  const uint8_t *row_b_scalars = b_scalars_le + (row * term_count * 32u);
  const uint8_t *row_mt = m_tilde_le + (row * undisclosed_count * 32u);
  for (size_t u = 0u; u < undisclosed_count; ++u) {
    const uint32_t msg_idx = undisclosed_indexes[u];
    cubbs_sppark_scalar_t fr_mt, fr_m;
    memcpy(&fr_mt, row_mt + (u * 32u), 32u);
    memcpy(&fr_m, row_b_scalars + ((2u + msg_idx) * 32u), 32u);
    fr_mt.to();
    fr_m.to();
    cubbs_bbs_store_fr_be(fr_mt + (fr_m * fr_c), out + 240u + (u * 32u));
  }
  // [240 + U*32 .. +32]: c (BE)
  uint8_t *c_out = out + 240u + (undisclosed_count * 32u);
  #pragma unroll
  for (int i = 0; i < 32; ++i) c_out[i] = c_le[31 - i];
}

// Thread-local GPU cache for ProofGen A points and ProofVerify (Abar, Bbar, D) points
// along with their prebuilt 8-entry affine radix-16 tables [1*P..8*P].
struct CuBbsFusedProofPointsCache {
  cubbs_sppark_plain_affine_t *d_points{nullptr};
  cubbs_sppark_plain_affine_t *d_radix16_aff{nullptr};
  size_t capacity{0u};
  std::string key;

  cudaError_t ensure(size_t count) {
    if (capacity >= count) return cudaSuccess;
    if (d_points != nullptr) (void)cudaFree(d_points);
    if (d_radix16_aff != nullptr) (void)cudaFree(d_radix16_aff);
    d_points = nullptr;
    d_radix16_aff = nullptr;
    capacity = 0u;
    key.clear();
    const size_t req = std::max<size_t>(1u, count);
    cudaError_t err = cudaMalloc(&d_points, req * sizeof(cubbs_sppark_plain_affine_t));
    if (err == cudaSuccess) {
      err = cudaMalloc(&d_radix16_aff, req * 8u * sizeof(cubbs_sppark_plain_affine_t));
    }
    if (err == cudaSuccess) {
      capacity = req;
    } else {
      if (d_points != nullptr) (void)cudaFree(d_points);
      if (d_radix16_aff != nullptr) (void)cudaFree(d_radix16_aff);
      d_points = nullptr;
      d_radix16_aff = nullptr;
    }
    return err;
  }

  ~CuBbsFusedProofPointsCache() {
    if (d_points != nullptr) (void)cudaFree(d_points);
    if (d_radix16_aff != nullptr) (void)cudaFree(d_radix16_aff);
  }
};

static CuBbsFusedProofPointsCache &cubbs_proof_gen_a_cache() {
  static thread_local CuBbsFusedProofPointsCache cache;
  return cache;
}

static CuBbsFusedProofPointsCache &cubbs_proof_verify_pts_cache() {
  static thread_local CuBbsFusedProofPointsCache cache;
  return cache;
}

int32_t cubbs_bbs_cuda_proof_gen_fused_common_batch(
    const CuBbsBbsGpuProofGenFusedCommonBatchItem *item,
    CuBbsBbsGpuResult *results) {
  if (item == nullptr) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  const size_t item_count = item->item_count;
  if (item_count == 0u) return CUBBS_BBS_GPU_SUCCESS;
  cubbs_bbs_ensure_cuda_device_schedule();

  const size_t term_count = item->common_term_count;
  const size_t R = item->disclosed_count;
  const size_t U = item->undisclosed_count;
  if (term_count < 2u || R + U != term_count - 2u ||
      item->common_points == nullptr ||
      (U > 0u && item->undisclosed_points == nullptr) ||
      (R > 0u && item->disclosed_indexes == nullptr) ||
      (U > 0u && item->undisclosed_indexes == nullptr) ||
      item->signatures == nullptr || item->domains_be == nullptr ||
      (item->domain_stride != 0u && item->domain_stride != 32u) ||
      item->ph_offsets == nullptr || item->ph_lengths == nullptr ||
      item->message_dst.data == nullptr || item->challenge_dst.data == nullptr ||
      item->csprng_seed == nullptr || item->proofs_out == nullptr ||
      item->proof_stride != 272u + (U * 32u) || results == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t message_count = term_count - 2u;
  const size_t total_messages = item_count * message_count;
  const bool meta_uniform =
      total_messages > 0u && item->message_offsets == nullptr && item->message_lengths == nullptr;
  const bool meta_u32 =
      total_messages > 0u && item->message_offsets != nullptr && item->message_lengths == nullptr;
  const bool meta_u64 =
      total_messages > 0u && item->message_offsets != nullptr && item->message_lengths != nullptr;
  if (total_messages > 0u &&
      ((!meta_uniform && !meta_u32 && !meta_u64) ||
       (meta_uniform && (item->packed_messages_len % total_messages) != 0u) ||
       (meta_u32 && item->packed_messages_len > 0xFFFFFFFFull) ||
       (item->packed_messages_len > 0u && item->packed_messages == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t uniform_msg_len = meta_uniform ? (item->packed_messages_len / total_messages) : 0u;

  // Acquire cached common-points table for B (M+2) and optionally T2 undisclosed (U).
  std::shared_ptr<CuBbsFusedCommonPointsCache> cached_b;
  int32_t status = get_fused_common_points_cache(
      item->common_points, term_count, &cached_b);
  if (status != CUBBS_BBS_GPU_SUCCESS || cached_b == nullptr) return status;

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached_u;
  if (U > 0u) {
    status = get_fused_common_points_cache(
        item->undisclosed_points, U, &cached_u);
    if (status != CUBBS_BBS_GPU_SUCCESS || cached_u == nullptr) return status;
  }

  const cudaStream_t stream = cubbs_fused_pinned_host_workspace().execution_stream(
      cached_b->stream != nullptr ? cached_b->stream : static_cast<cudaStream_t>(0));

  auto &pipe_ws = cubbs_fused_pipeline_pinned_workspace();
  auto &a_cache = cubbs_proof_gen_a_cache();

  // Fast zero-alloc check whether signature A_0..A_{B-1} match GPU-cached points+radix32 tables.
  bool a_matches = (a_cache.key.size() == item_count * 48u && a_cache.capacity >= item_count);
  if (a_matches) {
    const char *ck = a_cache.key.data();
    for (size_t i = 0u; i < item_count; ++i) {
      if (std::memcmp(ck + (i * 48u), item->signatures + (i * 80u), 48u) != 0) {
        a_matches = false;
        break;
      }
    }
  }
  const bool need_gpu_decode_a = !a_matches;

  const size_t dom_bytes = item->domain_stride == 0u ? 32u : item_count * 32u;
  const size_t sig_e_bytes = item_count * 32u;
  const size_t a_comp_bytes = need_gpu_decode_a ? (item_count * 48u) : 0u;
  const size_t disc_idx_bytes = R * sizeof(uint32_t);
  const size_t undisc_idx_bytes = U * sizeof(uint32_t);
  const size_t ph_meta_bytes = item_count * 2u * sizeof(uint64_t);
  const size_t ph_bytes = item->packed_ph_len;
  const size_t aux_in_bytes =
      dom_bytes + sig_e_bytes + a_comp_bytes + 32u + disc_idx_bytes + undisc_idx_bytes +
      ph_meta_bytes + ph_bytes + item->message_dst.len + item->challenge_dst.len + 96u;
  const size_t proofs_out_bytes = item_count * item->proof_stride;

  cudaError_t err = pipe_ws.ensure_messages(item->packed_messages_len);
  if (err == cudaSuccess && meta_u64)
    err = pipe_ws.ensure_msg_meta(total_messages);
  else if (err == cudaSuccess && meta_u32)
    err = pipe_ws.ensure_msg_meta((total_messages + 2u) / 2u);
  if (err == cudaSuccess)
    err = pipe_ws.ensure_aux(aux_in_bytes, proofs_out_bytes);
  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  if (item->packed_messages_len > 0u && pipe_ws.messages != item->packed_messages)
    std::memcpy(pipe_ws.messages, item->packed_messages, item->packed_messages_len);
  if (meta_u64) {
    if (pipe_ws.offsets != item->message_offsets)
      std::memcpy(pipe_ws.offsets, item->message_offsets, total_messages * sizeof(uint64_t));
    if (pipe_ws.lengths != item->message_lengths)
      std::memcpy(pipe_ws.lengths, item->message_lengths, total_messages * sizeof(uint64_t));
  } else if (meta_u32) {
    if (pipe_ws.offsets != item->message_offsets)
      std::memcpy(pipe_ws.offsets, item->message_offsets, (total_messages + 1u) * sizeof(uint32_t));
  }

  // Pack aux_in with 8-byte aligned offsets.
  size_t off = 0u;
  auto push_aligned = [&](size_t bytes, size_t align) -> size_t {
    off = (off + align - 1u) & ~(align - 1u);
    const size_t start = off;
    off += bytes;
    return start;
  };
  const size_t off_dom = push_aligned(dom_bytes, 8u);
  const size_t off_sig_e = push_aligned(sig_e_bytes, 8u);
  const size_t off_a_comp = push_aligned(a_comp_bytes, 8u);
  const size_t off_seed = push_aligned(32u, 8u);
  const size_t off_disc = push_aligned(disc_idx_bytes, 4u);
  const size_t off_undisc = push_aligned(undisc_idx_bytes, 4u);
  const size_t off_ph_off = push_aligned(item_count * sizeof(uint64_t), 8u);
  const size_t off_ph_len = push_aligned(item_count * sizeof(uint64_t), 8u);
  const size_t off_ph_data = push_aligned(ph_bytes, 1u);
  const size_t off_msg_dst = push_aligned(item->message_dst.len, 1u);
  const size_t off_chal_dst = push_aligned(item->challenge_dst.len, 1u);

  std::memcpy(pipe_ws.aux_in + off_dom, item->domains_be, dom_bytes);
  for (size_t i = 0u; i < item_count; ++i) {
    const uint8_t *sig = item->signatures + (i * 80u);
    std::memcpy(pipe_ws.aux_in + off_sig_e + (i * 32u), sig + 48u, 32u);
    if (need_gpu_decode_a) {
      std::memcpy(pipe_ws.aux_in + off_a_comp + (i * 48u), sig, 48u);
    }
  }
  std::memcpy(pipe_ws.aux_in + off_seed, item->csprng_seed, 32u);
  if (R > 0u) std::memcpy(pipe_ws.aux_in + off_disc, item->disclosed_indexes, disc_idx_bytes);
  if (U > 0u) std::memcpy(pipe_ws.aux_in + off_undisc, item->undisclosed_indexes, undisc_idx_bytes);
  std::memcpy(pipe_ws.aux_in + off_ph_off, item->ph_offsets, item_count * sizeof(uint64_t));
  std::memcpy(pipe_ws.aux_in + off_ph_len, item->ph_lengths, item_count * sizeof(uint64_t));
  if (ph_bytes > 0u) std::memcpy(pipe_ws.aux_in + off_ph_data, item->packed_ph, ph_bytes);
  if (item->message_dst.len > 0u)
    std::memcpy(pipe_ws.aux_in + off_msg_dst, item->message_dst.data, item->message_dst.len);
  if (item->challenge_dst.len > 0u)
    std::memcpy(pipe_ws.aux_in + off_chal_dst, item->challenge_dst.data, item->challenge_dst.len);

  // Allocate all transient device buffers from persistent thread-local arena (0 cudaMallocAsync!).
  const size_t d_off_bytes = meta_u64
      ? (total_messages * sizeof(uint64_t))
      : (meta_u32 ? ((total_messages + 1u) * sizeof(uint32_t)) : 0u);
  const size_t d_len_bytes = meta_u64 ? (total_messages * sizeof(uint64_t)) : 0u;
  auto &arena = cubbs_bbs_thread_device_arena();
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(item->packed_messages_len) +
      CuBbsDeviceArena::align_up(d_off_bytes) +
      CuBbsDeviceArena::align_up(d_len_bytes) +
      CuBbsDeviceArena::align_up(std::max<size_t>(1u, off)) +
      CuBbsDeviceArena::align_up(item_count * term_count * 32u) +
      CuBbsDeviceArena::align_up(item_count * term_count * 32u) +
      CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_plain_affine_t)) +
      CuBbsDeviceArena::align_up(item_count * 8u * sizeof(cubbs_sppark_plain_affine_t)) +
      CuBbsDeviceArena::align_up(item_count * 6u * 32u) +
      CuBbsDeviceArena::align_up(item_count * 6u * 32u) +
      (U > 0u ? (CuBbsDeviceArena::align_up(item_count * U * 32u) +
                 CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_bucket_t)))
              : 0u) +
      CuBbsDeviceArena::align_up(item_count * 4u * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(item_count * 240u) +
      CuBbsDeviceArena::align_up(proofs_out_bytes) +
      CuBbsDeviceArena::align_up(sizeof(uint32_t));
  if (err == cudaSuccess) err = arena.reserve_and_reset(arena_bytes);

  uint8_t *d_messages = nullptr;
  uint64_t *d_offsets = nullptr;
  uint64_t *d_lengths = nullptr;
  uint8_t *d_aux_in = nullptr;
  uint8_t *d_b_scalars_le = nullptr;
  uint8_t *d_d_scalars_le = nullptr;
  cubbs_sppark_bucket_t *d_d_xyzz = nullptr;
  cubbs_sppark_plain_affine_t *d_d_affines = nullptr;
  cubbs_sppark_plain_affine_t *d_d_tbl_aff = nullptr;
  uint8_t *d_row_state_le = nullptr;
  uint8_t *d_glv_splits_le = nullptr;
  uint8_t *d_m_tilde_le = nullptr;
  cubbs_sppark_bucket_t *d_pts4_xyzz = nullptr;
  cubbs_sppark_bucket_t *d_t2_common_xyzz = nullptr;
  uint8_t *d_pts5_compressed = nullptr;
  uint8_t *d_proofs_out = nullptr;
  uint32_t *d_invalid = nullptr;

  if (err == cudaSuccess) {
    d_messages = arena.alloc<uint8_t>(item->packed_messages_len);
    d_offsets = d_off_bytes > 0u ? reinterpret_cast<uint64_t *>(arena.alloc<uint8_t>(d_off_bytes)) : nullptr;
    d_lengths = d_len_bytes > 0u ? arena.alloc<uint64_t>(total_messages) : nullptr;
    d_aux_in = arena.alloc<uint8_t>(std::max<size_t>(1u, off));
    d_b_scalars_le = arena.alloc<uint8_t>(item_count * term_count * 32u);
    d_d_scalars_le = arena.alloc<uint8_t>(item_count * term_count * 32u);
    d_d_xyzz = arena.alloc<cubbs_sppark_bucket_t>(item_count);
    d_d_affines = arena.alloc<cubbs_sppark_plain_affine_t>(item_count);
    d_d_tbl_aff = arena.alloc<cubbs_sppark_plain_affine_t>(item_count * 8u);
    d_row_state_le = arena.alloc<uint8_t>(item_count * 6u * 32u);
    d_glv_splits_le = arena.alloc<uint8_t>(item_count * 6u * 32u);
    if (U > 0u) {
      d_m_tilde_le = arena.alloc<uint8_t>(item_count * U * 32u);
      d_t2_common_xyzz = arena.alloc<cubbs_sppark_bucket_t>(item_count);
    }
    d_pts4_xyzz = arena.alloc<cubbs_sppark_bucket_t>(item_count * 4u);
    d_pts5_compressed = arena.alloc<uint8_t>(item_count * 240u);
    d_proofs_out = arena.alloc<uint8_t>(proofs_out_bytes);
    d_invalid = arena.alloc<uint32_t>(1u);
  }
  if (err == cudaSuccess && need_gpu_decode_a) {
    err = a_cache.ensure(item_count);
  }

  if (err == cudaSuccess && item->packed_messages_len > 0u)
    err = cudaMemcpyAsync(d_messages, pipe_ws.messages, item->packed_messages_len,
                          cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess && d_off_bytes > 0u) {
    err = cudaMemcpyAsync(d_offsets, pipe_ws.offsets, d_off_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess && d_len_bytes > 0u) {
    err = cudaMemcpyAsync(d_lengths, pipe_ws.lengths, d_len_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(d_aux_in, pipe_ws.aux_in, off, cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(d_invalid, 0, sizeof(uint32_t), stream);

  const uint8_t *d_dom = d_aux_in + off_dom;
  const uint8_t *d_sig_e = d_aux_in + off_sig_e;
  const uint8_t *d_a_comp = d_aux_in + off_a_comp;
  const uint8_t *d_seed = d_aux_in + off_seed;
  const uint32_t *d_disc = reinterpret_cast<const uint32_t *>(d_aux_in + off_disc);
  const uint32_t *d_undisc = reinterpret_cast<const uint32_t *>(d_aux_in + off_undisc);
  const uint64_t *d_ph_off = reinterpret_cast<const uint64_t *>(d_aux_in + off_ph_off);
  const uint64_t *d_ph_len = reinterpret_cast<const uint64_t *>(d_aux_in + off_ph_len);
  const uint8_t *d_ph_data = d_aux_in + off_ph_data;
  const uint8_t *d_msg_dst = d_aux_in + off_msg_dst;
  const uint8_t *d_chal_dst = d_aux_in + off_chal_dst;

  // Stage 1: ChaCha20 CSPRNG prep (gives r2) + Fill [1, domain] & [r2, r2*domain] +
  // Fused H2S + r2-scaling so Lim-Lee computes D = r2*B directly!
  if (err == cudaSuccess) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_proof_gen_prep_scalars_kernel<<<b, t, 0, stream>>>(
        d_sig_e, d_seed, item_count, d_row_state_le, d_glv_splits_le, d_invalid);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_proof_gen_init_bd_header_scalars_kernel<<<b, t, 0, stream>>>(
        d_dom, item->domain_stride, d_row_state_le, item_count, term_count,
        d_b_scalars_le, d_d_scalars_le, d_invalid);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && total_messages > 0u) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((total_messages + t - 1u) / t);
    cubbs_bbs_proof_gen_h2s_and_scale_kernel<<<b, t, 0, stream>>>(
        d_messages, d_offsets, d_lengths, uniform_msg_len, d_row_state_le,
        item_count, message_count, term_count,
        d_msg_dst, item->message_dst.len, d_b_scalars_le, d_d_scalars_le);
    err = cudaGetLastError();
  }
  // Stage 2: Lim-Lee/Pippenger D-MSM (D = r2*B directly!) + D affine normalize & compress + Radix-16 table.
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_cached_common_msm_range(
        *cached_b, d_d_scalars_le, term_count, item_count, d_d_xyzz, stream);
  }
  if (err == cudaSuccess) {
    const uint32_t norm_blocks = static_cast<uint32_t>((item_count + 63u) / 64u);
    cubbs_bbs_xyzz_to_plain_affine_and_compress_d_kernel<<<norm_blocks, 64u, 0, stream>>>(
        d_d_xyzz, d_d_affines, d_pts5_compressed, item_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_build_radix16_affine_tables(
        d_d_affines, item_count, d_d_tbl_aff, stream);
  }
  if (err == cudaSuccess && U > 0u) {
    constexpr uint32_t t = 128u;
    const size_t total_u = item_count * U;
    const uint32_t b = static_cast<uint32_t>((total_u + t - 1u) / t);
    cubbs_bbs_proof_gen_m_tilde_kernel<<<b, t, 0, stream>>>(
        d_seed, item_count, U, d_m_tilde_le);
    err = cudaGetLastError();
  }

  // Stage 2b: 100% GPU G1 decompression + Scott subgroup check + Radix-16 tables for A on cache miss!
  if (err == cudaSuccess && need_gpu_decode_a) {
    constexpr uint32_t t = 64u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_g1_uncompress_strided_kernel<<<b, t, 0, stream>>>(
        d_a_comp, item_count, 48u, 1u, true, a_cache.d_points, d_invalid);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      err = cubbs_bbs_launch_build_radix16_affine_tables(
          a_cache.d_points, item_count, a_cache.d_radix16_aff, stream);
    }
  }

  // Stage 3: 4-way Jacobian Straus multi-base Signed Radix-16 GLV kernel + T2 undisclosed MSM.
  if (err == cudaSuccess) {
    const dim3 f_grid(static_cast<uint32_t>((item_count + 63u) / 64u), 4u);
    cubbs_bbs_proof_gen_straus4_jac_kernel<<<f_grid, 64u, 0, stream>>>(
        d_d_tbl_aff, a_cache.d_radix16_aff, d_glv_splits_le, d_pts4_xyzz, item_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && U > 0u) {
    err = cubbs_bbs_launch_cached_common_msm_range(
        *cached_u, d_m_tilde_le, U, item_count, d_t2_common_xyzz, stream);
    if (err == cudaSuccess) {
      constexpr uint32_t t = 128u;
      const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
      cubbs_bbs_proof_gen_add_t2_common_kernel<<<b, t, 0, stream>>>(
          d_pts4_xyzz, d_t2_common_xyzz, item_count);
      err = cudaGetLastError();
    }
  }

  // Stage 4: GPU parallel compress [Abar, Bbar, T1, T2] into slots {0,1,3,4} (slot 2 already holds D!).
  if (err == cudaSuccess) {
    constexpr uint32_t t = 64u;
    const uint32_t b = static_cast<uint32_t>(((item_count * 4u) + t - 1u) / t);
    cubbs_bbs_g1_xyzz_to_compressed_pts4_into_pts5_kernel<<<b, t, 0, stream>>>(
        d_pts4_xyzz, item_count, d_pts5_compressed);
    err = cudaGetLastError();
  }

  // Stage 5: GPU zero-spill challenge + r3 = r2^-1 + scalar finalize + wire assembly!
  if (err == cudaSuccess) {
    constexpr uint32_t t = 64u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_proof_gen_challenge_finalize_kernel<<<b, t, 0, stream>>>(
        d_disc, R, d_undisc, U, d_b_scalars_le, term_count,
        d_row_state_le, d_m_tilde_le, d_pts5_compressed,
        d_dom, item->domain_stride, d_ph_data, d_ph_off, d_ph_len,
        d_chal_dst, item->challenge_dst.len,
        item_count, item->proof_stride, d_proofs_out);
    err = cudaGetLastError();
  }

  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.aux_out, d_proofs_out, proofs_out_bytes,
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.invalid_flag, d_invalid, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }

  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;
  if (*pipe_ws.invalid_flag != 0u) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  if (need_gpu_decode_a) {
    a_cache.key.resize(item_count * 48u);
    std::memcpy(&a_cache.key[0], pipe_ws.aux_in + off_a_comp, item_count * 48u);
  }

  std::memcpy(item->proofs_out, pipe_ws.aux_out, proofs_out_bytes);
  for (size_t i = 0u; i < item_count; ++i) {
    set_result_route(results, i, CUBBS_BBS_GPU_SUCCESS, item->proof_stride,
                     CUBBS_BBS_MSM_ROUTE_FUSED_COMMON_POINTS);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

// ============================================================================
// GPU-Resident Fused ProofVerify Kernels
// ============================================================================

// Parse raw proof scalars, build common_scalars_le (B * (M+2) * 32) and
// proof_scalars_le (B * 4 * 32 = [e_cap, c, r1_cap, r3_cap]), plus recode
// 128-bit random pairing coefficients c_i into signed 8-bit digits (17 windows).
__global__ static void cubbs_bbs_proof_verify_parse_and_build_kernel(
    const uint8_t *proofs, size_t proof_stride,
    const uint32_t *disclosed_indexes, size_t disclosed_count,
    const uint32_t *undisclosed_indexes, size_t undisclosed_count,
    const uint8_t *disclosed_scalars_le,
    const uint8_t *domains_be, size_t domain_stride,
    const uint8_t *coefficients_le,
    size_t item_count, size_t term_count,
    uint8_t *common_scalars_le,
    uint8_t *proof_scalars_le,
    int16_t *coef_digits_out,
    uint32_t *invalid_out) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= item_count) return;

  const uint8_t *pr = proofs + (row * proof_stride);
  const uint8_t *dom_be = domains_be + (domain_stride == 0u ? 0u : row * domain_stride);
  const uint8_t *coef_le = coefficients_le + (row * 32u);
  const uint8_t *e_cap_be = pr + 144u;
  const uint8_t *r1_cap_be = pr + 176u;
  const uint8_t *r3_cap_be = pr + 208u;
  const uint8_t *c_be = pr + 240u + (undisclosed_count * 32u);

  uint64_t coef_hi0 = 0ull, coef_hi1 = 0ull;
  memcpy(&coef_hi0, coef_le + 16u, 8u);
  memcpy(&coef_hi1, coef_le + 24u, 8u);

  // Spec check: Abar != Identity_G1 (bit 6 of byte 0 == 0) + canonical scalars.
  bool ok = ((pr[0] & 0x40u) == 0u) &&
            cubbs_bbs_scalar_be_is_canonical(dom_be) &&
            cubbs_bbs_scalar_be_is_canonical(e_cap_be) &&
            cubbs_bbs_scalar_be_is_canonical(r1_cap_be) &&
            cubbs_bbs_scalar_be_is_canonical(r3_cap_be) &&
            cubbs_bbs_scalar_be_is_canonical(c_be) &&
            ((coef_hi0 | coef_hi1) == 0ull);

  auto be_to_le = [](const uint8_t src[32], uint8_t dst[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = src[31 - i];
  };

  uint8_t *row_common = common_scalars_le + (row * term_count * 32u);
  uint8_t *row_ps = proof_scalars_le + (row * 4u * 32u);

  if (!ok) {
    atomicExch(invalid_out, 1u);
    memset(row_common, 0, term_count * 32u);
    memset(row_ps, 0, 4u * 32u);
    if (coef_digits_out != nullptr) {
      const uint8_t zero16[16] = {};
      cubbs_bbs_recode_s8_128_store(zero16, row, item_count, coef_digits_out);
    }
    return;
  }

  uint8_t c_le[32], dom_le[32], e_le[32], r1_le[32], r3_le[32];
  be_to_le(c_be, c_le);
  be_to_le(dom_be, dom_le);
  be_to_le(e_cap_be, e_le);
  be_to_le(r1_cap_be, r1_le);
  be_to_le(r3_cap_be, r3_le);

  // Proof variable GLV-split scalars (rem || quot): [0]=e_cap (Abar), [1]=c (Bbar), [2]=r1_cap (D), [3]=r3_cap (D)
  cubbs_bbs_scalar_div_by_z2_device(e_le,  row_ps + 0u * 32u, row_ps + 0u * 32u + 16u);
  cubbs_bbs_scalar_div_by_z2_device(c_le,  row_ps + 1u * 32u, row_ps + 1u * 32u + 16u);
  cubbs_bbs_scalar_div_by_z2_device(r1_le, row_ps + 2u * 32u, row_ps + 2u * 32u + 16u);
  cubbs_bbs_scalar_div_by_z2_device(r3_le, row_ps + 3u * 32u, row_ps + 3u * 32u + 16u);

  cubbs_sppark_scalar_t fr_c, fr_dom;
  memcpy(&fr_c, c_le, 32u);     fr_c.to();
  memcpy(&fr_dom, dom_le, 32u); fr_dom.to();

  // Common slot 0 = c (for P1), slot 1 = domain * c (for Q1).
  memcpy(row_common + 0u * 32u, c_le, 32u);
  cubbs_sppark_scalar_t fr_dc = fr_dom * fr_c;
  fr_dc.from();
  memcpy(row_common + 1u * 32u, &fr_dc, 32u);

  // Disclosed slots: H[disclosed[k]] gets disclosed_scalar[k] * c.
  const uint8_t *row_disc_s = disclosed_scalars_le + (row * disclosed_count * 32u);
  for (size_t k = 0u; k < disclosed_count; ++k) {
    cubbs_sppark_scalar_t fr_m;
    memcpy(&fr_m, row_disc_s + (k * 32u), 32u);
    fr_m.to();
    cubbs_sppark_scalar_t prod = fr_m * fr_c;
    prod.from();
    memcpy(row_common + ((2u + disclosed_indexes[k]) * 32u), &prod, 32u);
  }

  // Undisclosed slots: H[undisclosed[u]] gets m_cap[u].
  for (size_t u = 0u; u < undisclosed_count; ++u) {
    const uint8_t *m_be = pr + 240u + (u * 32u);
    if (!cubbs_bbs_scalar_be_is_canonical(m_be)) {
      atomicExch(invalid_out, 1u);
      memset(row_common + ((2u + undisclosed_indexes[u]) * 32u), 0, 32u);
    } else {
      be_to_le(m_be, row_common + ((2u + undisclosed_indexes[u]) * 32u));
    }
  }

  if (coef_digits_out != nullptr) {
    cubbs_bbs_recode_s8_128_store(coef_le, row, item_count, coef_digits_out);
  }
}

// Extract [Abar_0..Abar_{B-1}, Bbar_0..Bbar_{B-1}] (Same-PK) OR ONLY [Bbar_0..Bbar_{B-1}]
// (Unique-PK, 50% less work!) from interleaved [Abar,Bbar,D].
__global__ static void cubbs_bbs_proof_verify_extract_ab_kernel(
    const cubbs_sppark_plain_affine_t *proof_bases_3,
    cubbs_sppark_plain_affine_t *ab_streams_out,
    size_t item_count,
    bool only_bbar) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= item_count) return;
  if (only_bbar) {
    ab_streams_out[row] = proof_bases_3[row * 3u + 1u];
  } else {
    ab_streams_out[row] = proof_bases_3[row * 3u + 0u];
    ab_streams_out[item_count + row] = proof_bases_3[row * 3u + 1u];
  }
}

// Fast 128-bit scalar mul r_i * Abar_i reusing the ALREADY-BUILT Radix-16 affine
// table [1*Abar_i .. 8*Abar_i] from Stage 1b (~30 mixed adds instead of ~64!).
__global__ static void cubbs_bbs_g1_scale_128bit_from_radix16_aff_kernel(
    const cubbs_sppark_plain_affine_t *radix16_tables_3, // stride 3 * 8 per proof; Abar at +0
    const uint8_t *scalars_le_stride32,
    size_t item_count,
    cubbs_sppark_bucket_t *xyzz_out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= item_count) return;

  const uint8_t *s_le = scalars_le_stride32 + (idx * 32u);
  uint64_t digs[3];
  cubbs_bbs_pack_signed_radix16(s_le, digs);

  const cubbs_sppark_plain_affine_t *tbl = radix16_tables_3 + (idx * 24u);
  CuBbsG1Jac acc;
  acc.inf();
  #pragma unroll 1
  for (int window = 32; window >= 0; --window) {
    if (!acc.is_inf()) {
      #pragma unroll 1
      for (uint32_t doubling = 0u; doubling < 4u; ++doubling) {
        cubbs_bbs_jac_dbl(acc);
      }
    }
    const int32_t d = cubbs_bbs_signed_radix16_digit(digs, static_cast<uint32_t>(window));
    if (d != 0) {
      const uint32_t mag = static_cast<uint32_t>(d < 0 ? -d : d);
      cubbs_bbs_jac_madd(acc, tbl[mag - 1u], d < 0);
    }
  }
  xyzz_out[idx] = cubbs_bbs_jac_to_xyzz(acc);
}

// Compress T1[row] and T2[row] by packing into one contiguous 2*B array first?
// Even simpler: normalize & compress T1 and T2 directly in one 64-thread kernel!
__global__ static void cubbs_bbs_proof_verify_check_challenge_kernel(
    const uint8_t *proofs, size_t proof_stride,
    const uint32_t *disclosed_indexes, size_t disclosed_count,
    size_t undisclosed_count,
    const uint8_t *disclosed_scalars_le,
    const uint8_t *t1_compressed, // item_count * 48
    const uint8_t *t2_compressed, // item_count * 48
    const uint8_t *domains_be, size_t domain_stride,
    const uint8_t *packed_ph, const uint64_t *ph_offsets, const uint64_t *ph_lengths,
    const uint8_t *challenge_dst, size_t challenge_dst_len,
    size_t item_count,
    uint32_t *challenge_fail_out) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= item_count) return;

  const uint8_t *pr = proofs + (row * proof_stride);
  const uint8_t *dom_be = domains_be + (domain_stride == 0u ? 0u : row * domain_stride);
  const uint8_t *ph = (ph_lengths[row] > 0u) ? (packed_ph + ph_offsets[row]) : nullptr;

  uint8_t computed_c_le[32];
  cubbs_bbs_compute_proof_challenge_le_device(
      disclosed_indexes, disclosed_count,
      disclosed_scalars_le + (row * disclosed_count * 32u), 32u, false,
      pr + 0u, pr + 48u, pr + 96u,
      t1_compressed + (row * 48u), t2_compressed + (row * 48u),
      dom_be, ph, static_cast<size_t>(ph_lengths[row]),
      challenge_dst, challenge_dst_len, computed_c_le);

  const uint8_t *expected_c_be = pr + 240u + (undisclosed_count * 32u);
  uint32_t diff = 0u;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    diff |= static_cast<uint32_t>(computed_c_le[31 - i] ^ expected_c_be[i]);
  }
  if (diff != 0u) {
    atomicExch(challenge_fail_out, 1u);
  }
}

// Normalize aggregate (lhs_agg, rhs_agg) to affine for inline GPU pairing check.
// When unique_pk == true, skips unused lhs_agg (agg_outputs[0]).
__global__ static void cubbs_bbs_proof_verify_finalize_pair_affines_kernel(
    const cubbs_sppark_bucket_t *agg_outputs,
    bool unique_pk,
    cubbs_sppark_plain_affine_t *pair_affines_out,
    uint32_t *inf_flag_out) {
  const uint32_t tid = threadIdx.x;
  __shared__ __align__(16) uint8_t pts_smem[2u * sizeof(cubbs_sppark_bucket_t)];
  auto *pts = reinterpret_cast<cubbs_sppark_bucket_t *>(pts_smem);
  if (tid < 2u) {
    if (unique_pk && tid == 0u) {
      pts[0].inf();
    } else {
      pts[tid] = agg_outputs[tid];
    }
  }
  __syncwarp();
  if (tid == 0u) {
    const bool any_inf = unique_pk ? pts[1].is_inf() : (pts[0].is_inf() || pts[1].is_inf());
    *inf_flag_out = any_inf ? 1u : 0u;
  }
  __syncwarp();

  const fp_t one = fp_t::one();
  fp_t zzz = one;
  if (tid < 2u && !pts[tid].is_inf()) {
    const fp_t *coords = reinterpret_cast<const fp_t *>(&pts[tid]);
    zzz = coords[2];
  }
  const fp_t inv_zzz = zzz.reciprocal();
  if (tid < 2u) {
    if (pts[tid].is_inf()) {
      fp_t zero;
      zero.zero();
      pair_affines_out[tid] = cubbs_sppark_plain_affine_t{zero, zero};
    } else {
      const fp_t *coords = reinterpret_cast<const fp_t *>(&pts[tid]);
      const fp_t affine_y = inv_zzz * coords[1];
      const fp_t inv_z = inv_zzz * coords[3];
      fp_t affine_x = inv_z * inv_z;
      affine_x *= coords[0];
      pair_affines_out[tid] = cubbs_sppark_plain_affine_t{affine_x, affine_y};
    }
  }
}

int32_t cubbs_bbs_cuda_proof_verify_fused_common_batch(
    const CuBbsBbsGpuProofVerifyFusedCommonBatchItem *item,
    CuBbsBbsGpuResult *result) {
  if (item == nullptr || result == nullptr) return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  const size_t item_count = item->item_count;
  if (item_count == 0u) {
    result->code = CUBBS_BBS_GPU_SUCCESS;
    result->value = 1u;
    result->bytes_written = 0u;
    return CUBBS_BBS_GPU_SUCCESS;
  }
  cubbs_bbs_ensure_cuda_device_schedule();

  const size_t term_count = item->common_term_count;
  const size_t R = item->disclosed_count;
  const size_t U = item->undisclosed_count;
  const bool skip_pairing = (item->public_key.len == 0u);
  const bool unique_pk = (!skip_pairing && item_count > 1u &&
                          item->public_key.len == item_count * 96u);
  if (term_count < 2u || R + U != term_count - 2u ||
      item->common_points == nullptr ||
      (R > 0u && item->disclosed_indexes == nullptr) ||
      (U > 0u && item->undisclosed_indexes == nullptr) ||
      item->proofs == nullptr || item->proof_stride != 272u + (U * 32u) ||
      (!skip_pairing && (item->public_key.data == nullptr ||
                         (!unique_pk && item->public_key.len != 96u) ||
                         item->g2_generator.data == nullptr || item->g2_generator.len != 96u)) ||
      item->coefficients_le == nullptr || item->domains_be == nullptr ||
      (item->domain_stride != 0u && item->domain_stride != 32u) ||
      item->ph_offsets == nullptr || item->ph_lengths == nullptr ||
      item->message_dst.data == nullptr || item->challenge_dst.data == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t total_disclosed = item_count * R;
  const bool meta_uniform =
      total_disclosed > 0u && item->message_offsets == nullptr && item->message_lengths == nullptr;
  const bool meta_u32 =
      total_disclosed > 0u && item->message_offsets != nullptr && item->message_lengths == nullptr;
  const bool meta_u64 =
      total_disclosed > 0u && item->message_offsets != nullptr && item->message_lengths != nullptr;
  if (total_disclosed > 0u &&
      ((!meta_uniform && !meta_u32 && !meta_u64) ||
       (meta_uniform && (item->packed_disclosed_messages_len % total_disclosed) != 0u) ||
       (meta_u32 && item->packed_disclosed_messages_len > 0xFFFFFFFFull) ||
       (item->packed_disclosed_messages_len > 0u && item->packed_disclosed_messages == nullptr))) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  const size_t uniform_msg_len =
      meta_uniform ? (item->packed_disclosed_messages_len / total_disclosed) : 0u;

  // Prepare/cache G2 lines: for Same-PK cache pk_lines + g2_lines; for Unique-PK only g2_lines!
  auto &pipe_ws = cubbs_fused_pipeline_pinned_workspace();
  if (!skip_pairing) {
    if (!unique_pk) {
      if (!pipe_ws.pk_valid ||
          std::memcmp(pipe_ws.pk_key, item->public_key.data, 96u) != 0) {
        const int32_t st =
            cubbs_bbs_blst_get_prepared_g2_lines(item->public_key, pipe_ws.pk_lines);
        if (st != CUBBS_BBS_GPU_SUCCESS) return st;
        std::memcpy(pipe_ws.pk_key, item->public_key.data, 96u);
        pipe_ws.pk_valid = true;
      }
    }
    if (!pipe_ws.g2_valid ||
        std::memcmp(pipe_ws.g2_key, item->g2_generator.data, 96u) != 0) {
      const int32_t st =
          cubbs_bbs_blst_get_prepared_g2_lines(item->g2_generator, pipe_ws.g2_lines);
      if (st != CUBBS_BBS_GPU_SUCCESS) return st;
      std::memcpy(pipe_ws.g2_key, item->g2_generator.data, 96u);
      pipe_ws.g2_valid = true;
    }
  }

  std::shared_ptr<CuBbsFusedCommonPointsCache> cached;
  int32_t status = get_fused_common_points_cache(
      item->common_points, term_count, &cached);
  if (status != CUBBS_BBS_GPU_SUCCESS || cached == nullptr) return status;

  const cudaStream_t stream = cubbs_fused_pinned_host_workspace().execution_stream(
      cached->stream != nullptr ? cached->stream : static_cast<cudaStream_t>(0));

  auto &pts_cache = cubbs_proof_verify_pts_cache();
  const size_t total_pts = item_count * 3u;
  bool pts_matches = (pts_cache.key.size() == item_count * 144u &&
                      pts_cache.capacity >= total_pts);
  if (pts_matches) {
    const char *ck = pts_cache.key.data();
    for (size_t i = 0u; i < item_count; ++i) {
      if (std::memcmp(ck + (i * 144u), item->proofs + (i * item->proof_stride), 144u) != 0) {
        pts_matches = false;
        break;
      }
    }
  }
  const bool need_gpu_decode_pts = !pts_matches;

  const size_t proofs_bytes = item_count * item->proof_stride;
  const size_t dom_bytes = item->domain_stride == 0u ? 32u : item_count * 32u;
  const size_t coef_bytes = item_count * 32u;
  const size_t pks_comp_bytes = unique_pk ? (item_count * 96u) : 0u;
  const size_t disc_idx_bytes = R * sizeof(uint32_t);
  const size_t undisc_idx_bytes = U * sizeof(uint32_t);
  const size_t ph_bytes = item->packed_ph_len;
  const size_t aux_in_bytes =
      proofs_bytes + dom_bytes + coef_bytes + pks_comp_bytes +
      disc_idx_bytes + undisc_idx_bytes +
      (item_count * 2u * sizeof(uint64_t)) + ph_bytes +
      item->message_dst.len + item->challenge_dst.len + 96u;
  const size_t aux_out_bytes = 32u;

  cudaError_t err = pipe_ws.ensure_messages(item->packed_disclosed_messages_len);
  if (err == cudaSuccess && meta_u64)
    err = pipe_ws.ensure_msg_meta(total_disclosed);
  else if (err == cudaSuccess && meta_u32)
    err = pipe_ws.ensure_msg_meta((total_disclosed + 2u) / 2u);
  if (err == cudaSuccess)
    err = pipe_ws.ensure_aux(aux_in_bytes, aux_out_bytes);
  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  if (item->packed_disclosed_messages_len > 0u &&
      pipe_ws.messages != item->packed_disclosed_messages) {
    std::memcpy(pipe_ws.messages, item->packed_disclosed_messages,
                item->packed_disclosed_messages_len);
  }
  if (meta_u64) {
    if (pipe_ws.offsets != item->message_offsets)
      std::memcpy(pipe_ws.offsets, item->message_offsets, total_disclosed * sizeof(uint64_t));
    if (pipe_ws.lengths != item->message_lengths)
      std::memcpy(pipe_ws.lengths, item->message_lengths, total_disclosed * sizeof(uint64_t));
  } else if (meta_u32) {
    if (pipe_ws.offsets != item->message_offsets)
      std::memcpy(pipe_ws.offsets, item->message_offsets, (total_disclosed + 1u) * sizeof(uint32_t));
  }

  size_t off = 0u;
  auto push_aligned = [&](size_t bytes, size_t align) -> size_t {
    off = (off + align - 1u) & ~(align - 1u);
    const size_t start = off;
    off += bytes;
    return start;
  };
  const size_t off_proofs = push_aligned(proofs_bytes, 8u);
  const size_t off_dom = push_aligned(dom_bytes, 8u);
  const size_t off_coef = push_aligned(coef_bytes, 8u);
  const size_t off_pks = push_aligned(pks_comp_bytes, 8u);
  const size_t off_disc = push_aligned(disc_idx_bytes, 4u);
  const size_t off_undisc = push_aligned(undisc_idx_bytes, 4u);
  const size_t off_ph_off = push_aligned(item_count * sizeof(uint64_t), 8u);
  const size_t off_ph_len = push_aligned(item_count * sizeof(uint64_t), 8u);
  const size_t off_ph_data = push_aligned(ph_bytes, 1u);
  const size_t off_msg_dst = push_aligned(item->message_dst.len, 1u);
  const size_t off_chal_dst = push_aligned(item->challenge_dst.len, 1u);

  std::memcpy(pipe_ws.aux_in + off_proofs, item->proofs, proofs_bytes);
  std::memcpy(pipe_ws.aux_in + off_dom, item->domains_be, dom_bytes);
  std::memcpy(pipe_ws.aux_in + off_coef, item->coefficients_le, coef_bytes);
  if (unique_pk) std::memcpy(pipe_ws.aux_in + off_pks, item->public_key.data, pks_comp_bytes);
  if (R > 0u) std::memcpy(pipe_ws.aux_in + off_disc, item->disclosed_indexes, disc_idx_bytes);
  if (U > 0u) std::memcpy(pipe_ws.aux_in + off_undisc, item->undisclosed_indexes, undisc_idx_bytes);
  std::memcpy(pipe_ws.aux_in + off_ph_off, item->ph_offsets, item_count * sizeof(uint64_t));
  std::memcpy(pipe_ws.aux_in + off_ph_len, item->ph_lengths, item_count * sizeof(uint64_t));
  if (ph_bytes > 0u) std::memcpy(pipe_ws.aux_in + off_ph_data, item->packed_ph, ph_bytes);
  if (item->message_dst.len > 0u)
    std::memcpy(pipe_ws.aux_in + off_msg_dst, item->message_dst.data, item->message_dst.len);
  if (item->challenge_dst.len > 0u)
    std::memcpy(pipe_ws.aux_in + off_chal_dst, item->challenge_dst.data, item->challenge_dst.len);

  const uint32_t chunks = static_cast<uint32_t>((item_count + 2047u) / 2048u);

  // Allocate all transient device buffers from persistent thread-local arena (0 cudaMallocAsync!).
  // d_t12_xyzz holds contiguous [T1_0..T1_{B-1}, T2_0..T2_{B-1}] -> 1 single compress kernel!
  const size_t d_off_bytes = meta_u64
      ? (total_disclosed * sizeof(uint64_t))
      : (meta_u32 ? ((total_disclosed + 1u) * sizeof(uint32_t)) : 0u);
  const size_t d_len_bytes = meta_u64 ? (total_disclosed * sizeof(uint64_t)) : 0u;
  auto &arena = cubbs_bbs_thread_device_arena();
  const uint32_t pair_streams = unique_pk ? 1u : 2u;
  const size_t arena_bytes =
      CuBbsDeviceArena::align_up(item->packed_disclosed_messages_len) +
      CuBbsDeviceArena::align_up(d_off_bytes) +
      CuBbsDeviceArena::align_up(d_len_bytes) +
      CuBbsDeviceArena::align_up(total_disclosed * 32u) +
      CuBbsDeviceArena::align_up(std::max<size_t>(1u, off)) +
      CuBbsDeviceArena::align_up(item_count * term_count * 32u) +
      CuBbsDeviceArena::align_up(item_count * 4u * 32u) +
      (!skip_pairing ? CuBbsDeviceArena::align_up(17u * item_count * sizeof(int16_t)) : 0u) +
      CuBbsDeviceArena::align_up(item_count * 2u * sizeof(cubbs_sppark_bucket_t)) +
      CuBbsDeviceArena::align_up(item_count * 2u * 48u) +
      (!skip_pairing
           ? (CuBbsDeviceArena::align_up(item_count * static_cast<size_t>(pair_streams) * sizeof(cubbs_sppark_plain_affine_t)) +
              CuBbsDeviceArena::align_up(static_cast<size_t>(pair_streams) * 17u * chunks * sizeof(cubbs_sppark_bucket_t)) +
              CuBbsDeviceArena::align_up(2u * sizeof(cubbs_sppark_bucket_t)) +
              CuBbsDeviceArena::align_up(2u * sizeof(cubbs_sppark_plain_affine_t)) +
              CuBbsDeviceArena::align_up(sizeof(uint8_t)))
           : 0u) +
      (unique_pk
           ? (CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_bucket_t)) +
              CuBbsDeviceArena::align_up(item_count * sizeof(cubbs_sppark_plain_affine_t)))
           : 0u) +
      CuBbsDeviceArena::align_up(3u * sizeof(uint32_t));
  if (err == cudaSuccess) err = arena.reserve_and_reset(arena_bytes);

  uint8_t *d_messages = nullptr;
  uint64_t *d_offsets = nullptr;
  uint64_t *d_lengths = nullptr;
  uint8_t *d_disc_scalars_le = nullptr;
  uint8_t *d_aux_in = nullptr;
  uint8_t *d_common_scalars_le = nullptr;
  uint8_t *d_proof_scalars_le = nullptr;
  int16_t *d_coef_digits = nullptr;
  cubbs_sppark_bucket_t *d_t12_xyzz = nullptr;
  uint8_t *d_t12_compressed = nullptr;
  cubbs_sppark_plain_affine_t *d_ab_streams = nullptr;
  cubbs_sppark_bucket_t *d_win_chunk_sums = nullptr;
  cubbs_sppark_bucket_t *d_agg_outputs = nullptr;
  cubbs_sppark_plain_affine_t *d_pair_affines = nullptr;
  cubbs_sppark_bucket_t *d_scaled_abar_xyzz = nullptr;
  cubbs_sppark_plain_affine_t *d_scaled_abar_affines = nullptr;
  uint32_t *d_flags = nullptr; // [0]=invalid, [1]=challenge_fail, [2]=inf_flag
  uint8_t *d_eq_out = nullptr;

  if (err == cudaSuccess) {
    d_messages = arena.alloc<uint8_t>(item->packed_disclosed_messages_len);
    d_offsets = d_off_bytes > 0u ? reinterpret_cast<uint64_t *>(arena.alloc<uint8_t>(d_off_bytes)) : nullptr;
    d_lengths = d_len_bytes > 0u ? arena.alloc<uint64_t>(total_disclosed) : nullptr;
    d_disc_scalars_le = arena.alloc<uint8_t>(total_disclosed * 32u);
    d_aux_in = arena.alloc<uint8_t>(std::max<size_t>(1u, off));
    d_common_scalars_le = arena.alloc<uint8_t>(item_count * term_count * 32u);
    d_proof_scalars_le = arena.alloc<uint8_t>(item_count * 4u * 32u);
    if (!skip_pairing) {
      d_coef_digits = arena.alloc<int16_t>(17u * item_count);
    }
    d_t12_xyzz = arena.alloc<cubbs_sppark_bucket_t>(item_count * 2u);
    d_t12_compressed = arena.alloc<uint8_t>(item_count * 2u * 48u);
    if (!skip_pairing) {
      d_ab_streams = arena.alloc<cubbs_sppark_plain_affine_t>(item_count * static_cast<size_t>(pair_streams));
      d_win_chunk_sums = arena.alloc<cubbs_sppark_bucket_t>(static_cast<size_t>(pair_streams) * 17u * chunks);
      d_agg_outputs = arena.alloc<cubbs_sppark_bucket_t>(2u);
      d_pair_affines = arena.alloc<cubbs_sppark_plain_affine_t>(2u);
      d_eq_out = arena.alloc<uint8_t>(1u);
    }
    if (unique_pk) {
      d_scaled_abar_xyzz = arena.alloc<cubbs_sppark_bucket_t>(item_count);
      d_scaled_abar_affines = arena.alloc<cubbs_sppark_plain_affine_t>(item_count);
    }
    d_flags = arena.alloc<uint32_t>(3u);
  }
  if (err == cudaSuccess && need_gpu_decode_pts) {
    err = pts_cache.ensure(total_pts);
  }

  cubbs_sppark_bucket_t *d_t1_xyzz = d_t12_xyzz;
  cubbs_sppark_bucket_t *d_t2_xyzz = d_t12_xyzz + item_count;
  uint8_t *d_t1_compressed = d_t12_compressed;
  uint8_t *d_t2_compressed = d_t12_compressed + (item_count * 48u);

  if (err == cudaSuccess && item->packed_disclosed_messages_len > 0u)
    err = cudaMemcpyAsync(d_messages, pipe_ws.messages, item->packed_disclosed_messages_len,
                          cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess && d_off_bytes > 0u) {
    err = cudaMemcpyAsync(d_offsets, pipe_ws.offsets, d_off_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess && d_len_bytes > 0u) {
    err = cudaMemcpyAsync(d_lengths, pipe_ws.lengths, d_len_bytes,
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(d_aux_in, pipe_ws.aux_in, off, cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(d_flags, 0, 3u * sizeof(uint32_t), stream);

  const uint8_t *d_proofs = d_aux_in + off_proofs;
  const uint8_t *d_dom = d_aux_in + off_dom;
  const uint8_t *d_coef = d_aux_in + off_coef;
  const uint8_t *d_pks_comp = d_aux_in + off_pks;
  const uint32_t *d_disc = reinterpret_cast<const uint32_t *>(d_aux_in + off_disc);
  const uint32_t *d_undisc = reinterpret_cast<const uint32_t *>(d_aux_in + off_undisc);
  const uint64_t *d_ph_off = reinterpret_cast<const uint64_t *>(d_aux_in + off_ph_off);
  const uint64_t *d_ph_len = reinterpret_cast<const uint64_t *>(d_aux_in + off_ph_len);
  const uint8_t *d_ph_data = d_aux_in + off_ph_data;
  const uint8_t *d_msg_dst = d_aux_in + off_msg_dst;
  const uint8_t *d_chal_dst = d_aux_in + off_chal_dst;

  // Stage 1: H2S on disclosed messages + parse proof scalars + Lim-Lee/Pippenger T2 Common MSM.
  if (err == cudaSuccess && total_disclosed > 0u) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((total_disclosed + t - 1u) / t);
    cubbs_bbs_verify_h2s_flat_to_le_kernel<<<b, t, 0, stream>>>(
        d_messages, d_offsets, d_lengths, uniform_msg_len, total_disclosed,
        d_msg_dst, item->message_dst.len, d_disc_scalars_le);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_proof_verify_parse_and_build_kernel<<<b, t, 0, stream>>>(
        d_proofs, item->proof_stride, d_disc, R, d_undisc, U,
        d_disc_scalars_le, d_dom, item->domain_stride, d_coef,
        item_count, term_count, d_common_scalars_le, d_proof_scalars_le,
        d_coef_digits, &d_flags[0]);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_cached_common_msm_range(
        *cached, d_common_scalars_le, term_count, item_count, d_t2_xyzz, stream);
  }

  // Stage 1b: 100% GPU G1 decompression + Scott subgroup check + Radix-16 tables on cache miss!
  if (err == cudaSuccess && need_gpu_decode_pts) {
    constexpr uint32_t t = 64u;
    const uint32_t b = static_cast<uint32_t>((total_pts + t - 1u) / t);
    cubbs_bbs_g1_uncompress_strided_kernel<<<b, t, 0, stream>>>(
        d_proofs, item_count, item->proof_stride, 3u, true,
        pts_cache.d_points, &d_flags[0]);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      err = cubbs_bbs_launch_build_radix16_affine_tables(
          pts_cache.d_points, total_pts, pts_cache.d_radix16_aff, stream);
    }
  }

  // Stage 2: 2-way Straus Signed Radix-16 GLV kernel (inline T2 += r3_cap*D!).
  if (err == cudaSuccess) {
    const dim3 v_grid(static_cast<uint32_t>((item_count + 63u) / 64u), 2u);
    cubbs_bbs_proof_verify_straus2_affine_u_kernel<<<v_grid, 64u, 0, stream>>>(
        pts_cache.d_radix16_aff, d_proof_scalars_le, item_count,
        d_t1_xyzz, d_t2_xyzz);
    err = cudaGetLastError();
  }

  // Stage 3: Single merged GPU Montgomery batch-compress for BOTH T1 & T2!
  if (err == cudaSuccess) {
    err = cubbs_bbs_launch_xyzz_to_compressed(
        d_t12_xyzz, item_count * 2u, d_t12_compressed, 48u, 0u, stream);
  }

  // Stage 4: GPU zero-spill challenge verification (computed_c == proof_c for all B items).
  if (err == cudaSuccess) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_proof_verify_check_challenge_kernel<<<b, t, 0, stream>>>(
        d_proofs, item->proof_stride, d_disc, R, U, d_disc_scalars_le,
        d_t1_compressed, d_t2_compressed, d_dom, item->domain_stride,
        d_ph_data, d_ph_off, d_ph_len, d_chal_dst, item->challenge_dst.len,
        item_count, &d_flags[1]);
    err = cudaGetLastError();
  }

  // Stage 5: Aggregate Pairing (Same-PK single pairing OR Unique-PK randomized multi-pairing!).
  // Unique-PK skips unused sum(r_i * Abar_i) (50% less Pippenger!) and reuses the already-built
  // Radix-16 affine table in pts_cache.d_radix16_aff to compute r_i * Abar_i!
  if (err == cudaSuccess && !skip_pairing) {
    constexpr uint32_t t = 128u;
    const uint32_t b = static_cast<uint32_t>((item_count + t - 1u) / t);
    cubbs_bbs_proof_verify_extract_ab_kernel<<<b, t, 0, stream>>>(
        pts_cache.d_points, d_ab_streams, item_count, unique_pk);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && !skip_pairing) {
    const dim3 bucket_grid(17u, pair_streams, chunks);
    cubbs_bbs_verify_pippenger_buckets_s8_kernel<<<bucket_grid, 128u, 0, stream>>>(
        d_ab_streams, item_count, d_coef_digits, 0u, item_count, chunks,
        d_win_chunk_sums);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && !skip_pairing) {
    cubbs_bbs_verify_pippenger_combine_s8_kernel<<<1u, 32u, 0, stream>>>(
        d_win_chunk_sums, chunks, pair_streams, 0u,
        unique_pk ? &d_agg_outputs[1] : d_agg_outputs);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && !skip_pairing) {
    cubbs_bbs_proof_verify_finalize_pair_affines_kernel<<<1u, 32u, 0, stream>>>(
        d_agg_outputs, unique_pk, d_pair_affines, &d_flags[2]);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && !skip_pairing) {
    if (!unique_pk) {
      const int32_t pair_launch = cubbs_bbs_cuda_launch_pairing_eq_single_on_stream(
          &d_pair_affines[0], &d_pair_affines[1],
          pipe_ws.pk_lines, pipe_ws.g2_lines,
          item->public_key.data, item->g2_generator.data,
          d_eq_out, stream);
      if (pair_launch != CUBBS_BBS_GPU_SUCCESS) err = cudaErrorUnknown;
    } else {
      // Unique-PK GPU-resident randomized multi-pairing:
      //   prod_i e(r_i * Abar_i, pk_i) * e(-sum r_i * Bbar_i, G_2) == 1 (ONE FinalExp!).
      constexpr uint32_t sc_threads = 64u;
      const uint32_t sc_blocks = static_cast<uint32_t>((item_count + sc_threads - 1u) / sc_threads);
      cubbs_bbs_g1_scale_128bit_from_radix16_aff_kernel<<<sc_blocks, sc_threads, 0, stream>>>(
          pts_cache.d_radix16_aff, d_coef, item_count, d_scaled_abar_xyzz);
      err = cudaGetLastError();
      if (err == cudaSuccess) {
        cubbs_bbs_xyzz_to_plain_affine_batch64_kernel<<<sc_blocks, 64u, 0, stream>>>(
            d_scaled_abar_xyzz, d_scaled_abar_affines, item_count);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) {
        const int32_t pair_launch = cubbs_bbs_cuda_launch_pairing_unique_pk_on_stream(
            d_scaled_abar_affines, &d_pair_affines[1], d_pks_comp,
            item->public_key.data, item_count,
            pipe_ws.g2_lines, item->g2_generator.data, &d_flags[0], d_eq_out, stream);
        if (pair_launch != CUBBS_BBS_GPU_SUCCESS) err = cudaErrorUnknown;
      }
    }
  }

  // Stage 6: Single 16-byte D2H + ONE cudaStreamSynchronize!
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(pipe_ws.aux_out, d_flags, 3u * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess && !skip_pairing) {
    err = cudaMemcpyAsync(pipe_ws.aux_out + (3u * sizeof(uint32_t)), d_eq_out,
                          sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }

  if (err != cudaSuccess) return CUBBS_BBS_GPU_ERR_INTERNAL;

  uint32_t flags[3] = {0u, 0u, 0u};
  std::memcpy(flags, pipe_ws.aux_out, sizeof(flags));
  if (flags[0] != 0u || flags[2] != 0u) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  if (unique_pk) {
    cubbs_bbs_cuda_confirm_unique_pk_lines_cache(item->public_key.data, item_count);
  }
  if (need_gpu_decode_pts) {
    pts_cache.key.resize(item_count * 144u);
    for (size_t i = 0u; i < item_count; ++i) {
      std::memcpy(&pts_cache.key[i * 144u], item->proofs + (i * item->proof_stride), 144u);
    }
  }
  const uint8_t eq_value = skip_pairing ? 1u : pipe_ws.aux_out[sizeof(flags)];
  const uint8_t all_valid = (flags[1] == 0u && eq_value == 1u) ? 1u : 0u;
  result->code = CUBBS_BBS_GPU_SUCCESS;
  result->value = all_valid;
  result->bytes_written = 0u;
  return CUBBS_BBS_GPU_SUCCESS;
}

#endif // CUBBS_BBS_FUSED_PROOF_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
