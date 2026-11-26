#ifndef CUBBS_BBS_CUDA_SHA256_CUH
#define CUBBS_BBS_CUDA_SHA256_CUH

#include <cuda_runtime.h>

#include <stdint.h>

static __device__ __forceinline__ uint32_t rotr32(uint32_t value, uint32_t bits) {
  return (value >> bits) | (value << (32u - bits));
}

static __device__ __forceinline__ uint32_t load_be32(const uint8_t *bytes) {
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         static_cast<uint32_t>(bytes[3]);
}

static __device__ __forceinline__ void store_be32(uint8_t *bytes, uint32_t value) {
  bytes[0] = static_cast<uint8_t>(value >> 24);
  bytes[1] = static_cast<uint8_t>(value >> 16);
  bytes[2] = static_cast<uint8_t>(value >> 8);
  bytes[3] = static_cast<uint8_t>(value);
}

#define CUBBS_SHA256_STEP(A, B, C, D, E, F, G, H, KVAL, WVAL)                  \
  do {                                                                         \
    const uint32_t _s1 = rotr32(E, 6) ^ rotr32(E, 11) ^ rotr32(E, 25);         \
    const uint32_t _ch = (E & F) ^ ((~E) & G);                                 \
    const uint32_t _t1 = H + _s1 + _ch + (KVAL) + (WVAL);                      \
    const uint32_t _s0 = rotr32(A, 2) ^ rotr32(A, 13) ^ rotr32(A, 22);         \
    const uint32_t _maj = (A & B) ^ (A & C) ^ (B & C);                         \
    D += _t1;                                                                  \
    H = _t1 + _s0 + _maj;                                                      \
  } while (0)

#define CUBBS_SHA256_EXPAND(W0, W1, W9, W14)                                   \
  do {                                                                         \
    const uint32_t _ws0 = rotr32(W1, 7) ^ rotr32(W1, 18) ^ (W1 >> 3);          \
    const uint32_t _ws1 = rotr32(W14, 17) ^ rotr32(W14, 19) ^ (W14 >> 10);     \
    W0 = W0 + _ws0 + W9 + _ws1;                                                \
  } while (0)

// Fully register-resident SHA-256 compression (0 bytes local-memory stack spill).
static __device__ __forceinline__ void sha256_transform_regs(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
    uint32_t w8, uint32_t w9, uint32_t w10, uint32_t w11,
    uint32_t w12, uint32_t w13, uint32_t w14, uint32_t w15,
    uint32_t &s0, uint32_t &s1, uint32_t &s2, uint32_t &s3,
    uint32_t &s4, uint32_t &s5, uint32_t &s6, uint32_t &s7) {
  uint32_t a = s0, b = s1, c = s2, d = s3;
  uint32_t e = s4, f = s5, g = s6, h = s7;

  // Rounds 0..15
  CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0x428a2f98u, w0);
  CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0x71374491u, w1);
  CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0xb5c0fbcfu, w2);
  CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0xe9b5dba5u, w3);
  CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0x3956c25bu, w4);
  CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0x59f111f1u, w5);
  CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0x923f82a4u, w6);
  CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0xab1c5ed5u, w7);
  CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0xd807aa98u, w8);
  CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0x12835b01u, w9);
  CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0x243185beu, w10);
  CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0x550c7dc3u, w11);
  CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0x72be5d74u, w12);
  CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0x80deb1feu, w13);
  CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0x9bdc06a7u, w14);
  CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0xc19bf174u, w15);

  // Rounds 16..31
  CUBBS_SHA256_EXPAND(w0, w1, w9, w14);   CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0xe49b69c1u, w0);
  CUBBS_SHA256_EXPAND(w1, w2, w10, w15);  CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0xefbe4786u, w1);
  CUBBS_SHA256_EXPAND(w2, w3, w11, w0);   CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0x0fc19dc6u, w2);
  CUBBS_SHA256_EXPAND(w3, w4, w12, w1);   CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0x240ca1ccu, w3);
  CUBBS_SHA256_EXPAND(w4, w5, w13, w2);   CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0x2de92c6fu, w4);
  CUBBS_SHA256_EXPAND(w5, w6, w14, w3);   CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0x4a7484aau, w5);
  CUBBS_SHA256_EXPAND(w6, w7, w15, w4);   CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0x5cb0a9dcu, w6);
  CUBBS_SHA256_EXPAND(w7, w8, w0, w5);    CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0x76f988dau, w7);
  CUBBS_SHA256_EXPAND(w8, w9, w1, w6);    CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0x983e5152u, w8);
  CUBBS_SHA256_EXPAND(w9, w10, w2, w7);   CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0xa831c66du, w9);
  CUBBS_SHA256_EXPAND(w10, w11, w3, w8);  CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0xb00327c8u, w10);
  CUBBS_SHA256_EXPAND(w11, w12, w4, w9);  CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0xbf597fc7u, w11);
  CUBBS_SHA256_EXPAND(w12, w13, w5, w10); CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0xc6e00bf3u, w12);
  CUBBS_SHA256_EXPAND(w13, w14, w6, w11); CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0xd5a79147u, w13);
  CUBBS_SHA256_EXPAND(w14, w15, w7, w12); CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0x06ca6351u, w14);
  CUBBS_SHA256_EXPAND(w15, w0, w8, w13);  CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0x14292967u, w15);

  // Rounds 32..47
  CUBBS_SHA256_EXPAND(w0, w1, w9, w14);   CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0x27b70a85u, w0);
  CUBBS_SHA256_EXPAND(w1, w2, w10, w15);  CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0x2e1b2138u, w1);
  CUBBS_SHA256_EXPAND(w2, w3, w11, w0);   CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0x4d2c6dfcu, w2);
  CUBBS_SHA256_EXPAND(w3, w4, w12, w1);   CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0x53380d13u, w3);
  CUBBS_SHA256_EXPAND(w4, w5, w13, w2);   CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0x650a7354u, w4);
  CUBBS_SHA256_EXPAND(w5, w6, w14, w3);   CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0x766a0abbu, w5);
  CUBBS_SHA256_EXPAND(w6, w7, w15, w4);   CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0x81c2c92eu, w6);
  CUBBS_SHA256_EXPAND(w7, w8, w0, w5);    CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0x92722c85u, w7);
  CUBBS_SHA256_EXPAND(w8, w9, w1, w6);    CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0xa2bfe8a1u, w8);
  CUBBS_SHA256_EXPAND(w9, w10, w2, w7);   CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0xa81a664bu, w9);
  CUBBS_SHA256_EXPAND(w10, w11, w3, w8);  CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0xc24b8b70u, w10);
  CUBBS_SHA256_EXPAND(w11, w12, w4, w9);  CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0xc76c51a3u, w11);
  CUBBS_SHA256_EXPAND(w12, w13, w5, w10); CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0xd192e819u, w12);
  CUBBS_SHA256_EXPAND(w13, w14, w6, w11); CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0xd6990624u, w13);
  CUBBS_SHA256_EXPAND(w14, w15, w7, w12); CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0xf40e3585u, w14);
  CUBBS_SHA256_EXPAND(w15, w0, w8, w13);  CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0x106aa070u, w15);

  // Rounds 48..63
  CUBBS_SHA256_EXPAND(w0, w1, w9, w14);   CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0x19a4c116u, w0);
  CUBBS_SHA256_EXPAND(w1, w2, w10, w15);  CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0x1e376c08u, w1);
  CUBBS_SHA256_EXPAND(w2, w3, w11, w0);   CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0x2748774cu, w2);
  CUBBS_SHA256_EXPAND(w3, w4, w12, w1);   CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0x34b0bcb5u, w3);
  CUBBS_SHA256_EXPAND(w4, w5, w13, w2);   CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0x391c0cb3u, w4);
  CUBBS_SHA256_EXPAND(w5, w6, w14, w3);   CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0x4ed8aa4au, w5);
  CUBBS_SHA256_EXPAND(w6, w7, w15, w4);   CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0x5b9cca4fu, w6);
  CUBBS_SHA256_EXPAND(w7, w8, w0, w5);    CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0x682e6ff3u, w7);
  CUBBS_SHA256_EXPAND(w8, w9, w1, w6);    CUBBS_SHA256_STEP(a, b, c, d, e, f, g, h, 0x748f82eeu, w8);
  CUBBS_SHA256_EXPAND(w9, w10, w2, w7);   CUBBS_SHA256_STEP(h, a, b, c, d, e, f, g, 0x78a5636fu, w9);
  CUBBS_SHA256_EXPAND(w10, w11, w3, w8);  CUBBS_SHA256_STEP(g, h, a, b, c, d, e, f, 0x84c87814u, w10);
  CUBBS_SHA256_EXPAND(w11, w12, w4, w9);  CUBBS_SHA256_STEP(f, g, h, a, b, c, d, e, 0x8cc70208u, w11);
  CUBBS_SHA256_EXPAND(w12, w13, w5, w10); CUBBS_SHA256_STEP(e, f, g, h, a, b, c, d, 0x90befffau, w12);
  CUBBS_SHA256_EXPAND(w13, w14, w6, w11); CUBBS_SHA256_STEP(d, e, f, g, h, a, b, c, 0xa4506cebu, w13);
  CUBBS_SHA256_EXPAND(w14, w15, w7, w12); CUBBS_SHA256_STEP(c, d, e, f, g, h, a, b, 0xbef9a3f7u, w14);
  CUBBS_SHA256_EXPAND(w15, w0, w8, w13);  CUBBS_SHA256_STEP(b, c, d, e, f, g, h, a, 0xc67178f2u, w15);

  s0 += a; s1 += b; s2 += c; s3 += d;
  s4 += e; s5 += f; s6 += g; s7 += h;
}

#undef CUBBS_SHA256_STEP
#undef CUBBS_SHA256_EXPAND

// Zero-spill streaming SHA-256 context holding state and block words in scalar registers.
struct CuBbsSha256Ctx {
  uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
  uint32_t cur_word;
  uint32_t block_len;
  uint64_t total_len;

  __device__ __forceinline__ void set_w(uint32_t idx, uint32_t val) {
    switch (idx) {
      case 0:  w0  = val; break; case 1:  w1  = val; break;
      case 2:  w2  = val; break; case 3:  w3  = val; break;
      case 4:  w4  = val; break; case 5:  w5  = val; break;
      case 6:  w6  = val; break; case 7:  w7  = val; break;
      case 8:  w8  = val; break; case 9:  w9  = val; break;
      case 10: w10 = val; break; case 11: w11 = val; break;
      case 12: w12 = val; break; case 13: w13 = val; break;
      case 14: w14 = val; break; default: w15 = val; break;
    }
  }

  __device__ __forceinline__ void compress() {
    sha256_transform_regs(w0, w1, w2, w3, w4, w5, w6, w7,
                          w8, w9, w10, w11, w12, w13, w14, w15,
                          s0, s1, s2, s3, s4, s5, s6, s7);
  }

  __device__ __forceinline__ void init() {
    s0 = 0x6a09e667u; s1 = 0xbb67ae85u; s2 = 0x3c6ef372u; s3 = 0xa54ff53au;
    s4 = 0x510e527fu; s5 = 0x9b05688cu; s6 = 0x1f83d9abu; s7 = 0x5be0cd19u;
    cur_word = 0u;
    block_len = 0u;
    total_len = 0ull;
  }

  // Precomputed state after compressing 64 zero bytes (RFC 9380 Z_pad).
  __device__ __forceinline__ void init_after_zpad() {
    s0 = 0xda5698beu; s1 = 0x17b9b469u; s2 = 0x62335799u; s3 = 0x779fbecau;
    s4 = 0x8ce5d491u; s5 = 0xc0d26243u; s6 = 0xbafef9eau; s7 = 0x1837a9d8u;
    cur_word = 0u;
    block_len = 0u;
    total_len = 64ull;
  }

  __device__ __forceinline__ void push_word(uint32_t word) {
    set_w(block_len >> 2, word);
    block_len += 4u;
    total_len += 4ull;
    if (block_len == 64u) {
      compress();
      block_len = 0u;
    }
  }

  __device__ __forceinline__ void push_byte(uint8_t byte) {
    const uint32_t lane = block_len & 3u;
    cur_word = (lane == 0u) ? (static_cast<uint32_t>(byte) << 24)
                            : (cur_word | (static_cast<uint32_t>(byte) << (24u - (lane * 8u))));
    block_len += 1u;
    total_len += 1ull;
    if ((block_len & 3u) == 0u) {
      set_w((block_len >> 2) - 1u, cur_word);
      if (block_len == 64u) {
        compress();
        block_len = 0u;
      }
    }
  }

  __device__ __forceinline__ void update(const uint8_t *data, size_t len) {
    size_t i = 0u;
    while (i < len && (block_len & 3u) != 0u) {
      push_byte(data[i++]);
    }
    while (len - i >= 64u && block_len == 0u) {
      w0  = load_be32(data + i + 0);  w1  = load_be32(data + i + 4);
      w2  = load_be32(data + i + 8);  w3  = load_be32(data + i + 12);
      w4  = load_be32(data + i + 16); w5  = load_be32(data + i + 20);
      w6  = load_be32(data + i + 24); w7  = load_be32(data + i + 28);
      w8  = load_be32(data + i + 32); w9  = load_be32(data + i + 36);
      w10 = load_be32(data + i + 40); w11 = load_be32(data + i + 44);
      w12 = load_be32(data + i + 48); w13 = load_be32(data + i + 52);
      w14 = load_be32(data + i + 56); w15 = load_be32(data + i + 60);
      compress();
      i += 64u;
      total_len += 64ull;
    }
    while (len - i >= 4u) {
      push_word(load_be32(data + i));
      i += 4u;
    }
    while (i < len) {
      push_byte(data[i++]);
    }
  }

  __device__ __forceinline__ void update_words_be(const uint32_t *words,
                                                  size_t word_count) {
    if ((block_len & 3u) == 0u) {
      for (size_t i = 0u; i < word_count; ++i) {
        push_word(words[i]);
      }
    } else {
      for (size_t i = 0u; i < word_count; ++i) {
        const uint32_t v = words[i];
        push_byte(static_cast<uint8_t>(v >> 24));
        push_byte(static_cast<uint8_t>(v >> 16));
        push_byte(static_cast<uint8_t>(v >> 8));
        push_byte(static_cast<uint8_t>(v));
      }
    }
  }

  __device__ __forceinline__ void update_scalar_le_as_be(const uint8_t *le32) {
    if ((block_len & 3u) == 0u) {
      // On little-endian CUDA, u32[word] already places the most-significant
      // byte of each 4-byte LE limb in bits 31..24 — the exact SHA-256 BE word.
      const uint32_t *u32 = reinterpret_cast<const uint32_t *>(le32);
#pragma unroll
      for (int word = 7; word >= 0; --word) {
        push_word(u32[word]);
      }
    } else {
      for (int b = 31; b >= 0; --b) {
        push_byte(le32[b]);
      }
    }
  }

  __device__ __forceinline__ void finalize_words(uint32_t out[8]) {
    const uint64_t bit_len = total_len * 8ull;
    const uint32_t lane = block_len & 3u;
    const uint32_t pad_word =
        (lane == 0u) ? 0x80000000u : (cur_word | (0x80u << (24u - (lane * 8u))));
    uint32_t word_idx = block_len >> 2;
    set_w(word_idx++, pad_word);

    if (word_idx > 14u) {
      if (word_idx == 15u) w15 = 0u;
      compress();
      w0 = 0u; w1 = 0u; w2 = 0u; w3 = 0u;
      w4 = 0u; w5 = 0u; w6 = 0u; w7 = 0u;
      w8 = 0u; w9 = 0u; w10 = 0u; w11 = 0u;
      w12 = 0u; w13 = 0u;
    } else {
      while (word_idx < 14u) {
        set_w(word_idx++, 0u);
      }
    }
    w14 = static_cast<uint32_t>(bit_len >> 32);
    w15 = static_cast<uint32_t>(bit_len & 0xffffffffu);
    compress();

    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
    out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7;
  }

  __device__ __forceinline__ void finalize_bytes(uint8_t out[32]) {
    uint32_t words[8];
    finalize_words(words);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      store_be32(out + (i * 4), words[i]);
    }
  }
};

static __device__ void sha256_digest_device(const uint8_t *message, size_t len, uint8_t out[32]) {
  CuBbsSha256Ctx ctx;
  ctx.init();
  ctx.update(message, len);
  ctx.finalize_bytes(out);
}

static __device__ __forceinline__ void expand_message_xmd_sha256_48_words_device(
    const uint8_t *message, size_t len, const uint8_t *dst, size_t dst_len,
    uint32_t out_words[12]) {
  CuBbsSha256Ctx ctx;
  uint32_t b0[8];
  uint32_t b1[8];
  uint32_t b2[8];

  // b0 = SHA256(Z_pad || message || I2OSP(48, 2) || I2OSP(0, 1) || dst || I2OSP(dst_len, 1))
  ctx.init_after_zpad();
  ctx.update(message, len);
  ctx.push_byte(0u);
  ctx.push_byte(48u);
  ctx.push_byte(0u);
  ctx.update(dst, dst_len);
  ctx.push_byte(static_cast<uint8_t>(dst_len));
  ctx.finalize_words(b0);

  // b1 = SHA256(b0 || I2OSP(1, 1) || dst || I2OSP(dst_len, 1))
  ctx.init();
  ctx.w0 = b0[0]; ctx.w1 = b0[1]; ctx.w2 = b0[2]; ctx.w3 = b0[3];
  ctx.w4 = b0[4]; ctx.w5 = b0[5]; ctx.w6 = b0[6]; ctx.w7 = b0[7];
  ctx.block_len = 32u;
  ctx.total_len = 32ull;
  ctx.push_byte(1u);
  ctx.update(dst, dst_len);
  ctx.push_byte(static_cast<uint8_t>(dst_len));
  ctx.finalize_words(b1);

  // b2 = SHA256((b0 ^ b1) || I2OSP(2, 1) || dst || I2OSP(dst_len, 1))
  ctx.init();
  ctx.w0 = b0[0] ^ b1[0]; ctx.w1 = b0[1] ^ b1[1];
  ctx.w2 = b0[2] ^ b1[2]; ctx.w3 = b0[3] ^ b1[3];
  ctx.w4 = b0[4] ^ b1[4]; ctx.w5 = b0[5] ^ b1[5];
  ctx.w6 = b0[6] ^ b1[6]; ctx.w7 = b0[7] ^ b1[7];
  ctx.block_len = 32u;
  ctx.total_len = 32ull;
  ctx.push_byte(2u);
  ctx.update(dst, dst_len);
  ctx.push_byte(static_cast<uint8_t>(dst_len));
  ctx.finalize_words(b2);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out_words[i] = b1[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out_words[8 + i] = b2[i];
  }
}

static __device__ void expand_message_xmd_sha256_48_device(const uint8_t *message, size_t len,
                                                           const uint8_t *dst, size_t dst_len,
                                                           uint8_t out[48]) {
  uint32_t words[12];
  expand_message_xmd_sha256_48_words_device(message, len, dst, dst_len, words);
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    store_be32(out + (i * 4), words[i]);
  }
}

// Fast exact 384-bit -> 255-bit modular reduction mod BLS12-381 Fr order r.
static __device__ __forceinline__ void bls12_381_fr_reduce_words12_to_le64(
    const uint32_t w[12], uint64_t out_le[4]) {
  static constexpr uint64_t ORDER_LE[4] = {
      0xffffffff00000001ull, 0x53bda402fffe5bfeull,
      0x3339d80809a1d805ull, 0x73eda753299d7d48ull};
  static constexpr uint64_t R2_LE[4] = {
      0xc999e990f3f29c6dull, 0x2b6cedcb87925c23ull,
      0x05d314967254398full, 0x0748d9d99f59ff11ull};
  static constexpr uint64_t R_INV = 0xfffffffeffffffffull;

  const uint64_t hi[2] = {
      (static_cast<uint64_t>(w[2]) << 32) | static_cast<uint64_t>(w[3]),
      (static_cast<uint64_t>(w[0]) << 32) | static_cast<uint64_t>(w[1])};

  uint64_t lo[4] = {
      (static_cast<uint64_t>(w[10]) << 32) | static_cast<uint64_t>(w[11]),
      (static_cast<uint64_t>(w[8]) << 32) | static_cast<uint64_t>(w[9]),
      (static_cast<uint64_t>(w[6]) << 32) | static_cast<uint64_t>(w[7]),
      (static_cast<uint64_t>(w[4]) << 32) | static_cast<uint64_t>(w[5])};

  auto sub_order_if_ge = [](uint64_t v[4]) {
    uint64_t d0, d1, d2, d3;
    uint64_t borrow = 0ull;
    {
      const uint64_t a = v[0], b = ORDER_LE[0];
      d0 = a - b;
      borrow = (a < b) ? 1ull : 0ull;
    }
    {
      const uint64_t a = v[1], b = ORDER_LE[1];
      const uint64_t nb = (a < b) || (borrow && a == b);
      d1 = a - b - borrow;
      borrow = nb ? 1ull : 0ull;
    }
    {
      const uint64_t a = v[2], b = ORDER_LE[2];
      const uint64_t nb = (a < b) || (borrow && a == b);
      d2 = a - b - borrow;
      borrow = nb ? 1ull : 0ull;
    }
    {
      const uint64_t a = v[3], b = ORDER_LE[3];
      const uint64_t nb = (a < b) || (borrow && a == b);
      d3 = a - b - borrow;
      borrow = nb ? 1ull : 0ull;
    }
    if (!borrow) {
      v[0] = d0;
      v[1] = d1;
      v[2] = d2;
      v[3] = d3;
    }
  };

  sub_order_if_ge(lo);
  sub_order_if_ge(lo);

  uint64_t t[5] = {0ull, 0ull, 0ull, 0ull, 0ull};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint64_t c2 = 0ull;
    if (i < 2) {
      const uint64_t ai = hi[i];
      uint64_t c = 0ull;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned __int128 prod =
            static_cast<unsigned __int128>(t[j]) +
            static_cast<unsigned __int128>(ai) * static_cast<unsigned __int128>(R2_LE[j]) +
            static_cast<unsigned __int128>(c);
        t[j] = static_cast<uint64_t>(prod);
        c = static_cast<uint64_t>(prod >> 64);
      }
      const unsigned __int128 sum4 =
          static_cast<unsigned __int128>(t[4]) + static_cast<unsigned __int128>(c);
      t[4] = static_cast<uint64_t>(sum4);
      c2 = static_cast<uint64_t>(sum4 >> 64);
    }
    const uint64_t m = t[0] * R_INV;
    unsigned __int128 red =
        static_cast<unsigned __int128>(t[0]) +
        static_cast<unsigned __int128>(m) * static_cast<unsigned __int128>(ORDER_LE[0]);
    uint64_t c = static_cast<uint64_t>(red >> 64);
#pragma unroll
    for (int j = 1; j < 4; ++j) {
      red = static_cast<unsigned __int128>(t[j]) +
            static_cast<unsigned __int128>(m) * static_cast<unsigned __int128>(ORDER_LE[j]) +
            static_cast<unsigned __int128>(c);
      t[j - 1] = static_cast<uint64_t>(red);
      c = static_cast<uint64_t>(red >> 64);
    }
    const unsigned __int128 top =
        static_cast<unsigned __int128>(t[4]) + static_cast<unsigned __int128>(c);
    t[3] = static_cast<uint64_t>(top);
    t[4] = c2 + static_cast<uint64_t>(top >> 64);
  }

  sub_order_if_ge(t);

  uint64_t carry = 0ull;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned __int128 s =
        static_cast<unsigned __int128>(t[j]) +
        static_cast<unsigned __int128>(lo[j]) +
        static_cast<unsigned __int128>(carry);
    out_le[j] = static_cast<uint64_t>(s);
    carry = static_cast<uint64_t>(s >> 64);
  }
  sub_order_if_ge(out_le);
}

static __device__ __forceinline__ void bls12_381_fr_store_be(const uint64_t le[4],
                                                             uint8_t scalar_be[32]) {
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) {
    const uint64_t v = le[3 - limb];
    const int base = limb * 8;
    scalar_be[base + 0] = static_cast<uint8_t>(v >> 56);
    scalar_be[base + 1] = static_cast<uint8_t>(v >> 48);
    scalar_be[base + 2] = static_cast<uint8_t>(v >> 40);
    scalar_be[base + 3] = static_cast<uint8_t>(v >> 32);
    scalar_be[base + 4] = static_cast<uint8_t>(v >> 24);
    scalar_be[base + 5] = static_cast<uint8_t>(v >> 16);
    scalar_be[base + 6] = static_cast<uint8_t>(v >> 8);
    scalar_be[base + 7] = static_cast<uint8_t>(v);
  }
}

static __device__ __forceinline__ void bls12_381_fr_store_le(const uint64_t le[4],
                                                             uint8_t scalar_le[32]) {
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) {
    const uint64_t v = le[limb];
    const int base = limb * 8;
    scalar_le[base + 0] = static_cast<uint8_t>(v);
    scalar_le[base + 1] = static_cast<uint8_t>(v >> 8);
    scalar_le[base + 2] = static_cast<uint8_t>(v >> 16);
    scalar_le[base + 3] = static_cast<uint8_t>(v >> 24);
    scalar_le[base + 4] = static_cast<uint8_t>(v >> 32);
    scalar_le[base + 5] = static_cast<uint8_t>(v >> 40);
    scalar_le[base + 6] = static_cast<uint8_t>(v >> 48);
    scalar_le[base + 7] = static_cast<uint8_t>(v >> 56);
  }
}

static __device__ void bls12_381_scalar_from_okm_48_device(const uint8_t uniform[48],
                                                           uint8_t scalar[32]) {
  uint32_t words[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    words[i] = load_be32(uniform + (i * 4));
  }
  uint64_t le[4];
  bls12_381_fr_reduce_words12_to_le64(words, le);
  bls12_381_fr_store_be(le, scalar);
}

#ifndef CUBBS_BBS_SHA256_DEVICE_ONLY

__global__ void cubbs_bbs_expand_xmd_sha256_48_batch_kernel(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *uniform_bytes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= item_count) {
    return;
  }
  expand_message_xmd_sha256_48_device(messages + offsets[index], lengths[index], dst, dst_len,
                                      uniform_bytes + (index * 48));
}

__global__ void cubbs_bbs_hash_to_scalar_sha256_batch_kernel(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *scalars) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= item_count) {
    return;
  }
  uint32_t uniform_words[12];
  expand_message_xmd_sha256_48_words_device(messages + offsets[index], lengths[index], dst,
                                            dst_len, uniform_words);
  uint64_t le[4];
  bls12_381_fr_reduce_words12_to_le64(uniform_words, le);
  bls12_381_fr_store_be(le, scalars + (index * 32));
}

__global__ void cubbs_bbs_sha256_batch_kernel(const uint8_t *messages,
                                               const size_t *offsets,
                                               const size_t *lengths,
                                               size_t item_count,
                                               uint8_t *digests) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= item_count) {
    return;
  }
  sha256_digest_device(messages + offsets[index], lengths[index], digests + (index * 32));
}

static inline cudaError_t cubbs_bbs_sha256_batch_host(const uint8_t *messages,
                                                       const size_t *offsets,
                                                       const size_t *lengths,
                                                       size_t item_count,
                                                       uint8_t *digests) {
  if (item_count == 0) {
    return cudaSuccess;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((item_count + threads - 1) / threads);
  cubbs_bbs_sha256_batch_kernel<<<blocks, threads>>>(messages, offsets, lengths, item_count,
                                                      digests);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceSynchronize();
}

static inline cudaError_t cubbs_bbs_expand_xmd_sha256_48_batch_host(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *uniform_bytes) {
  if (item_count == 0) {
    return cudaSuccess;
  }
  if (dst_len > 255) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((item_count + threads - 1) / threads);
  cubbs_bbs_expand_xmd_sha256_48_batch_kernel<<<blocks, threads>>>(
      messages, offsets, lengths, item_count, dst, dst_len, uniform_bytes);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceSynchronize();
}

static inline cudaError_t cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *scalars, cudaStream_t stream) {
  if (item_count == 0) {
    return cudaSuccess;
  }
  if (dst_len > 255) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  const int blocks = static_cast<int>((item_count + threads - 1) / threads);
  cubbs_bbs_hash_to_scalar_sha256_batch_kernel<<<blocks, threads, 0, stream>>>(
      messages, offsets, lengths, item_count, dst, dst_len, scalars);
  return cudaGetLastError();
}

static inline cudaError_t cubbs_bbs_hash_to_scalar_sha256_batch_host(
    const uint8_t *messages, const size_t *offsets, const size_t *lengths, size_t item_count,
    const uint8_t *dst, size_t dst_len, uint8_t *scalars) {
  cudaError_t status = cubbs_bbs_hash_to_scalar_sha256_batch_host_stream(
      messages, offsets, lengths, item_count, dst, dst_len, scalars, nullptr);
  return status == cudaSuccess ? cudaDeviceSynchronize() : status;
}

#endif // CUBBS_BBS_SHA256_DEVICE_ONLY

#endif
