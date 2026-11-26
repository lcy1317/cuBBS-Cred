#ifndef CUBBS_BBS_GPU_ABI_H
#define CUBBS_BBS_GPU_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUBBS_BBS_GPU_ABI_VERSION 2u

/* ABI v2 route registry. Routes 1-4 are cuBBS-Cred production routes; 5-6
 * retain BLST CPU fallbacks; 7-8 are explicit SPPARK benchmark/reference
 * entries and MUST NOT be selected automatically. */
#define CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM 1u
#define CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS 2u
#define CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE 3u
#define CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL 4u
#define CUBBS_BBS_MSM_ROUTE_BLST_ITEM 5u
#define CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS 6u
#define CUBBS_BBS_MSM_ROUTE_SPPARK_ITEM_REFERENCE 7u
#define CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE 8u

#define CUBBS_BBS_GPU_SUCCESS 0
#define CUBBS_BBS_GPU_ERR_INVALID_INPUT 1
#define CUBBS_BBS_GPU_ERR_INSUFFICIENT_OUTPUT 2
#define CUBBS_BBS_GPU_ERR_INTERNAL 3

#define CUBBS_BBS_GPU_BACKEND_KIND_UNKNOWN 0u
#define CUBBS_BBS_GPU_BACKEND_KIND_CUDA 1u
#define CUBBS_BBS_GPU_BACKEND_KIND_CPU_ABI_TEST 2u

#define CUBBS_BBS_SECRET_KEY_LEN 32u
#define CUBBS_BBS_PUBLIC_KEY_LEN 96u
#define CUBBS_BBS_SIGNATURE_LEN 80u
#define CUBBS_BBS_2023_HEADER_LEN 64u

typedef struct CuBbsBbsGpuByteSlice {
  const uint8_t *data;
  size_t len;
} CuBbsBbsGpuByteSlice;

typedef struct CuBbsBbsGpuByteBuffer {
  uint8_t *data;
  size_t capacity;
  size_t len;
} CuBbsBbsGpuByteBuffer;

typedef struct CuBbsBbsGpuIndexSlice {
  const size_t *data;
  size_t len;
} CuBbsBbsGpuIndexSlice;

typedef struct CuBbsBbsGpuResult {
  int32_t code;
  uint8_t value;
  size_t bytes_written;
} CuBbsBbsGpuResult;

typedef struct CuBbsBbsGpuBackendInfo {
  uint32_t backend_kind;
  uint8_t hardware_acceleration;
} CuBbsBbsGpuBackendInfo;

typedef struct CuBbsBbsGpuSignItem {
  CuBbsBbsGpuByteSlice secret_key;
  CuBbsBbsGpuByteSlice public_key;
  CuBbsBbsGpuByteSlice header;
  const CuBbsBbsGpuByteSlice *messages;
  size_t message_count;
  CuBbsBbsGpuByteBuffer signature_out;
} CuBbsBbsGpuSignItem;

typedef struct CuBbsBbsGpuVerifyItem {
  CuBbsBbsGpuByteSlice public_key;
  CuBbsBbsGpuByteSlice signature;
  CuBbsBbsGpuByteSlice header;
  const CuBbsBbsGpuByteSlice *messages;
  size_t message_count;
} CuBbsBbsGpuVerifyItem;

typedef struct CuBbsBbsGpuProofGenItem {
  CuBbsBbsGpuByteSlice public_key;
  CuBbsBbsGpuByteSlice signature;
  CuBbsBbsGpuByteSlice header;
  CuBbsBbsGpuByteSlice presentation_header;
  const CuBbsBbsGpuByteSlice *messages;
  size_t message_count;
  CuBbsBbsGpuIndexSlice disclosed_indexes;
  CuBbsBbsGpuByteBuffer proof_out;
} CuBbsBbsGpuProofGenItem;

typedef struct CuBbsBbsGpuProofVerifyItem {
  CuBbsBbsGpuByteSlice public_key;
  CuBbsBbsGpuByteSlice proof;
  CuBbsBbsGpuByteSlice header;
  CuBbsBbsGpuByteSlice presentation_header;
  const CuBbsBbsGpuByteSlice *disclosed_messages;
  size_t disclosed_count;
  CuBbsBbsGpuIndexSlice disclosed_indexes;
  size_t total_message_count;
} CuBbsBbsGpuProofVerifyItem;

typedef struct CuBbsBbsGpuMessagesToScalarItem {
  CuBbsBbsGpuByteSlice message;
  CuBbsBbsGpuByteSlice dst;
  CuBbsBbsGpuByteBuffer scalar_out;
} CuBbsBbsGpuMessagesToScalarItem;

typedef struct CuBbsBbsGpuSignEFromResidentScalarsItem {
  CuBbsBbsGpuByteSlice secret_key;
  CuBbsBbsGpuByteSlice domain;
  const CuBbsBbsGpuByteSlice *messages;
  size_t message_scalar_offset;
  size_t message_count;
  CuBbsBbsGpuByteBuffer message_scalars_out;
  CuBbsBbsGpuByteBuffer scalar_out;
} CuBbsBbsGpuSignEFromResidentScalarsItem;

typedef struct CuBbsBbsGpuPrepareBItem {
  CuBbsBbsGpuByteSlice public_key;
  CuBbsBbsGpuByteSlice header;
  const CuBbsBbsGpuByteSlice *message_scalars;
  size_t message_count;
  CuBbsBbsGpuByteBuffer b_out;
} CuBbsBbsGpuPrepareBItem;

typedef struct CuBbsBbsGpuG1MsmItem {
  const CuBbsBbsGpuByteSlice *points;
  const CuBbsBbsGpuByteSlice *scalars;
  size_t term_count;
  CuBbsBbsGpuByteBuffer point_out;
} CuBbsBbsGpuG1MsmItem;

typedef struct CuBbsBbsGpuG1MsmCommonPointsBatchItem {
  const CuBbsBbsGpuByteSlice *points;
  size_t term_count;
  const CuBbsBbsGpuByteSlice *scalar_rows;
  const uint8_t *scalar_rows_be;
  size_t scalar_row_stride;
  size_t output_count;
  CuBbsBbsGpuByteBuffer *point_outs;
} CuBbsBbsGpuG1MsmCommonPointsBatchItem;

typedef struct CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem {
  const CuBbsBbsGpuByteSlice *points;
  size_t term_count;
  const CuBbsBbsGpuByteSlice *scalar_rows;
  const CuBbsBbsGpuByteSlice *variable_points;
  const CuBbsBbsGpuByteSlice *variable_scalars;
  size_t output_count;
  CuBbsBbsGpuByteBuffer *point_outs;
} CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem;

/* BBS proof verification pair:
 *   T1 = challenge*Bbar + e_cap*Abar + r1_cap*D
 *   T2 = MSM(common_points, common_scalars) + r3_cap*D
 * proof_points are row-major [Bbar, Abar, D], proof_scalars are row-major
 * [challenge, e_cap, r1_cap, r3_cap]. */
typedef struct CuBbsBbsGpuProofVerifyMsmPairItem {
  const CuBbsBbsGpuByteSlice *common_points;
  size_t common_term_count;
  const CuBbsBbsGpuByteSlice *common_scalar_rows;
  const CuBbsBbsGpuByteSlice *proof_points;
  const CuBbsBbsGpuByteSlice *proof_scalars;
  size_t output_count;
  CuBbsBbsGpuByteBuffer *t1_outs;
  CuBbsBbsGpuByteBuffer *t2_outs;
} CuBbsBbsGpuProofVerifyMsmPairItem;

typedef struct CuBbsBbsGpuG1DualMsmSamePointsItem {
  const CuBbsBbsGpuByteSlice *points;
  const CuBbsBbsGpuByteSlice *lhs_scalars;
  const CuBbsBbsGpuByteSlice *rhs_scalars;
  size_t term_count;
  CuBbsBbsGpuByteBuffer lhs_out;
  CuBbsBbsGpuByteBuffer rhs_out;
} CuBbsBbsGpuG1DualMsmSamePointsItem;

typedef struct CuBbsBbsGpuG1DualMsmSamePointsLanesItem {
  const CuBbsBbsGpuByteSlice *points;
  const CuBbsBbsGpuByteSlice *lhs_scalars;
  const CuBbsBbsGpuByteSlice *rhs_scalars;
  size_t term_count;
  size_t lane_count;
  CuBbsBbsGpuByteBuffer lhs_outs;
  CuBbsBbsGpuByteBuffer rhs_outs;
} CuBbsBbsGpuG1DualMsmSamePointsLanesItem;

typedef struct CuBbsBbsGpuG1AddItem {
  CuBbsBbsGpuByteSlice lhs;
  CuBbsBbsGpuByteSlice rhs;
  CuBbsBbsGpuByteBuffer point_out;
} CuBbsBbsGpuG1AddItem;

typedef struct CuBbsBbsGpuPairingEqItem {
  CuBbsBbsGpuByteSlice lhs_g1;
  CuBbsBbsGpuByteSlice lhs_g2;
  CuBbsBbsGpuByteSlice rhs_g1;
  CuBbsBbsGpuByteSlice rhs_g2;
} CuBbsBbsGpuPairingEqItem;

typedef struct CuBbsBbsGpuPairingEqScaledItem {
  CuBbsBbsGpuByteSlice lhs_g1;
  CuBbsBbsGpuByteSlice lhs_g2;
  CuBbsBbsGpuByteSlice rhs_g1;
  CuBbsBbsGpuByteSlice rhs_g2;
  CuBbsBbsGpuByteSlice coefficient;
} CuBbsBbsGpuPairingEqScaledItem;

/* GPU proof-gen t2Prepare: computes the 6 fixed BBS G1 scalar multiplications
 * (d, abar, bbar, t1) on the device. The caller precomputes the 7 scalar
 * multipliers (including r1*r2) on the host and generates random scalars on
 * the host; the device only does G1 scalar multiplication. */
typedef struct CuBbsBbsGpuProofGenT2PrepareItem {
  CuBbsBbsGpuByteSlice compressed_b;   /* 48-byte compressed G1, from B-MSM */
  CuBbsBbsGpuByteSlice signature_a;    /* 48-byte compressed G1 */
  CuBbsBbsGpuByteSlice scalars_be[7];  /* r2,r1r2,r1r2,r1r2e,r1r2et,r2r1t,r3t */
  CuBbsBbsGpuByteSlice seed;           /* 32-byte CSPRNG seed (reserved) */
  CuBbsBbsGpuByteBuffer d_out;         /* 48-byte compressed d = B*r2 */
  CuBbsBbsGpuByteBuffer abar_out;      /* 48-byte compressed abar = A*(r1*r2) */
  CuBbsBbsGpuByteBuffer bbar_out;      /* 48-byte compressed bbar = d*r1 - abar*e */
  CuBbsBbsGpuByteBuffer t1_out;        /* 48-byte compressed t1 = abar*e_tilde + d*r1_tilde */
} CuBbsBbsGpuProofGenT2PrepareItem;

/* Fully GPU-resident BBS Sign for a common-generator batch (same message count M).
 * Computes message H2S, signature e, (sk + e)^-1, scaled B-MSM, and G1 compression
 * on one CUDA stream with zero intermediate host scalar round-trips. */
typedef struct CuBbsBbsGpuSignFusedCommonBatchItem {
  const CuBbsBbsGpuByteSlice *common_points; /* M + 2 compressed G1 points */
  size_t common_term_count;                  /* M + 2 */
  size_t item_count;                         /* B */
  const uint8_t *secret_keys_be;             /* B * 32 bytes BE (or 32 if stride == 0) */
  size_t secret_key_stride;                  /* 0 or 32 */
  const uint8_t *domains_be;                 /* B * 32 bytes BE (or 32 if stride == 0) */
  size_t domain_stride;                      /* 0 or 32 */
  const uint8_t *packed_messages;            /* contiguous raw message bytes */
  size_t packed_messages_len;
  const uint64_t *message_offsets;           /* B * M byte offsets */
  const uint64_t *message_lengths;           /* B * M byte lengths */
  CuBbsBbsGpuByteSlice message_dst;
  CuBbsBbsGpuByteSlice e_dst;
  uint8_t *signatures_out;                   /* B * 80 bytes: [A_compressed(48) || e_be(32)] */
} CuBbsBbsGpuSignFusedCommonBatchItem;

/* Fully GPU-resident BBS pre-B aggregate Verify for a same-public-key batch
 * (uniform or mixed message counts <= max_M). Computes message H2S, random-coefficient
 * B-scalar reduction, 1-row B-MSM, dual A-MSM, G1 addition, and aggregate pairing on GPU. */
typedef struct CuBbsBbsGpuVerifyPreBAggregateFusedItem {
  const CuBbsBbsGpuByteSlice *common_points; /* max_M + 2 compressed G1 points */
  size_t common_term_count;                  /* max_M + 2 */
  size_t item_count;                         /* B */
  const uint8_t *signatures;                 /* B * 80 bytes: [A_compressed(48) || e_be(32)] */
  const uint8_t *coefficients_le;            /* B * 32 bytes LE random nonzero scalars */
  const uint8_t *domains_be;                 /* B * 32 bytes BE (or 32 if stride == 0) */
  size_t domain_stride;                      /* 0 or 32 */
  const uint32_t *item_message_counts;       /* NULL if uniform M == common_term_count - 2 */
  const uint64_t *item_message_starts;       /* NULL if uniform; else B start offsets */
  const uint8_t *packed_messages;            /* contiguous raw message bytes */
  size_t packed_messages_len;
  const uint64_t *message_offsets;           /* total_message_count byte offsets */
  const uint64_t *message_lengths;           /* total_message_count byte lengths */
  size_t total_message_count;
  CuBbsBbsGpuByteSlice message_dst;
  CuBbsBbsGpuByteSlice public_key;           /* 96 bytes compressed G2 */
  CuBbsBbsGpuByteSlice g2_generator;         /* 96 bytes compressed G2 */
} CuBbsBbsGpuVerifyPreBAggregateFusedItem;

typedef struct CuBbsBbsGpuProofGenFusedCommonBatchItem {
  const CuBbsBbsGpuByteSlice *common_points;      /* M + 2 compressed G1 points */
  size_t common_term_count;                       /* M + 2 */
  const CuBbsBbsGpuByteSlice *undisclosed_points; /* U compressed G1 points (NULL if U == 0) */
  const uint32_t *disclosed_indexes;              /* R sorted ascending indexes */
  size_t disclosed_count;                         /* R */
  const uint32_t *undisclosed_indexes;            /* U sorted ascending indexes */
  size_t undisclosed_count;                       /* U = M - R */
  size_t item_count;                              /* B */
  const uint8_t *signatures;                      /* B * 80 bytes: [A(48) || e_be(32)] */
  const uint8_t *domains_be;                      /* B * 32 or 32 bytes if stride == 0 */
  size_t domain_stride;                           /* 0 or 32 */
  const uint8_t *packed_messages;                 /* contiguous raw message bytes */
  size_t packed_messages_len;
  const uint64_t *message_offsets;                /* B * M byte offsets */
  const uint64_t *message_lengths;                /* B * M byte lengths */
  const uint8_t *packed_ph;                       /* contiguous presentation headers */
  size_t packed_ph_len;
  const uint64_t *ph_offsets;                     /* B byte offsets */
  const uint64_t *ph_lengths;                     /* B byte lengths */
  CuBbsBbsGpuByteSlice message_dst;
  CuBbsBbsGpuByteSlice challenge_dst;
  const uint8_t *csprng_seed;                     /* 32-byte fresh OS entropy seed */
  uint8_t *proofs_out;                            /* B * proof_stride contiguous buffer */
  size_t proof_stride;                            /* 272 + U * 32 */
} CuBbsBbsGpuProofGenFusedCommonBatchItem;

typedef struct CuBbsBbsGpuProofVerifyFusedCommonBatchItem {
  const CuBbsBbsGpuByteSlice *common_points; /* M + 2 compressed G1 points */
  size_t common_term_count;                  /* M + 2 */
  const uint32_t *disclosed_indexes;         /* R sorted ascending indexes */
  size_t disclosed_count;                    /* R */
  const uint32_t *undisclosed_indexes;       /* U sorted ascending indexes */
  size_t undisclosed_count;                  /* U = M - R */
  size_t item_count;                         /* B */
  const uint8_t *proofs;                     /* B * proof_stride raw proof bytes */
  size_t proof_stride;                       /* 272 + U * 32 */
  const uint8_t *coefficients_le;            /* B * 32 bytes LE 128-bit random nonzero */
  const uint8_t *domains_be;                 /* B * 32 or 32 bytes if stride == 0 */
  size_t domain_stride;                      /* 0 or 32 */
  const uint8_t *packed_disclosed_messages;  /* contiguous raw disclosed messages */
  size_t packed_disclosed_messages_len;
  const uint64_t *message_offsets;           /* B * R byte offsets */
  const uint64_t *message_lengths;           /* B * R byte lengths */
  const uint8_t *packed_ph;                  /* contiguous presentation headers */
  size_t packed_ph_len;
  const uint64_t *ph_offsets;                /* B byte offsets */
  const uint64_t *ph_lengths;                /* B byte lengths */
  CuBbsBbsGpuByteSlice message_dst;
  CuBbsBbsGpuByteSlice challenge_dst;
  CuBbsBbsGpuByteSlice public_key;           /* 96 bytes compressed G2 */
  CuBbsBbsGpuByteSlice g2_generator;         /* 96 bytes compressed G2 */
} CuBbsBbsGpuProofVerifyFusedCommonBatchItem;

int32_t cubbs_bbs_gpu_backend_info_v2(uint32_t abi_version,
                                       CuBbsBbsGpuBackendInfo *info_out);

int32_t cubbs_bbs_gpu_sign_batch_v2(uint32_t abi_version,
                                     CuBbsBbsGpuSignItem *items,
                                     size_t item_count,
                                     CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_verify_batch_v2(uint32_t abi_version,
                                       const CuBbsBbsGpuVerifyItem *items,
                                       size_t item_count,
                                       CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_proof_gen_batch_v2(uint32_t abi_version,
                                          CuBbsBbsGpuProofGenItem *items,
                                          size_t item_count,
                                          CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_proof_verify_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuProofVerifyItem *items,
    size_t item_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_messages_to_scalar_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuMessagesToScalarItem *items,
    size_t item_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_sign_e_from_resident_scalars_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuSignEFromResidentScalarsItem *items,
    size_t item_count, CuBbsBbsGpuByteSlice message_dst,
    CuBbsBbsGpuByteSlice e_dst, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_prepare_b_batch_v2(uint32_t abi_version,
                                          CuBbsBbsGpuPrepareBItem *items,
                                          size_t item_count,
                                          CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_msm_batch_v2(uint32_t abi_version,
                                       CuBbsBbsGpuG1MsmItem *items,
                                       size_t item_count,
                                       CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmCommonPointsBatchItem *items,
    size_t group_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmCommonPointsBatchItem *items,
    size_t group_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_msm_common_points_plus_variable_batch_v2(
    uint32_t abi_version,
    CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem *items,
    size_t group_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_proof_verify_msm_pair_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuProofVerifyMsmPairItem *items,
    size_t group_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_dual_msm_same_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1DualMsmSamePointsItem *items,
    size_t item_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_dual_msm_same_points_lanes_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1DualMsmSamePointsLanesItem *items,
    size_t item_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_g1_add_batch_v2(uint32_t abi_version,
                                       CuBbsBbsGpuG1AddItem *items,
                                       size_t item_count,
                                       CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_pairing_eq_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuPairingEqItem *items,
    size_t item_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_pairing_eq_aggregate_v2(
    uint32_t abi_version, const CuBbsBbsGpuPairingEqScaledItem *items,
    size_t item_count, CuBbsBbsGpuResult *result);

int32_t cubbs_bbs_gpu_proof_gen_t2_prepare_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuProofGenT2PrepareItem *items,
    size_t item_count, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_sign_fused_common_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuSignFusedCommonBatchItem *item,
    CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_verify_pre_b_aggregate_fused_v2(
    uint32_t abi_version, const CuBbsBbsGpuVerifyPreBAggregateFusedItem *item,
    CuBbsBbsGpuResult *result);

int32_t cubbs_bbs_gpu_pipeline_pinned_scratch_v2(
    uint32_t abi_version, size_t msg_bytes, size_t msg_count,
    uint8_t **messages_out, uint64_t **offsets_out, uint64_t **lengths_out);

int32_t cubbs_bbs_gpu_proof_gen_fused_common_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuProofGenFusedCommonBatchItem *item,
    CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_gpu_proof_verify_fused_common_batch_v2(
    uint32_t abi_version, const CuBbsBbsGpuProofVerifyFusedCommonBatchItem *item,
    CuBbsBbsGpuResult *result);

#ifdef __cplusplus
}
#endif

#endif
