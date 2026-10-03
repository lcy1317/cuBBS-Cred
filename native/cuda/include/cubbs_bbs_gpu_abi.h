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

#ifdef __cplusplus
}
#endif

#endif
