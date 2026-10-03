#ifndef CUBBS_PREPARE_B_MSM_CUH
#define CUBBS_PREPARE_B_MSM_CUH

#include "cubbs_bbs_gpu_abi.h"

#include <functional>
#include <stddef.h>
#include <stdint.h>

// Shared bounded Host helper pool used by CUDA-side preparation paths. Keeping
// this entry point in one translation unit prevents each primitive from
// creating a fresh set of std::threads for every submitted batch.
void cubbs_bbs_run_host_tasks(
    size_t task_count, const std::function<void(size_t)> &task);

struct CuBbsPrepareBInput {
  const uint8_t *public_key;
  const uint8_t *header;
  const CuBbsBbsGpuByteSlice *message_scalars;
  size_t message_count;
  uint8_t *compressed_b_out;
};

struct CuBbsG1MsmInput {
  const CuBbsBbsGpuByteSlice *points;
  const CuBbsBbsGpuByteSlice *scalars;
  size_t term_count;
  uint8_t *compressed_point_out;
};

int32_t cubbs_bbs_cuda_prepare_b_msm_batch(const CuBbsPrepareBInput *items,
                                     size_t item_count,
                                     CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_g1_msm_batch(const CuBbsG1MsmInput *items,
                                     size_t item_count,
                                     CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_g1_msm_common_points_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_g1_msm_fused_common_points_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const uint8_t *scalar_rows_be, size_t scalar_row_stride,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_g1_msm_fused_common_points_plus_variable_matrix(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const CuBbsBbsGpuByteSlice *scalar_rows, size_t output_count,
    const CuBbsBbsGpuByteSlice *variable_points,
    const CuBbsBbsGpuByteSlice *variable_scalars,
    CuBbsBbsGpuByteBuffer *point_outs, CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_proof_verify_msm_pair_matrix(
    const CuBbsBbsGpuByteSlice *common_points, size_t common_term_count,
    const CuBbsBbsGpuByteSlice *common_scalar_rows,
    const CuBbsBbsGpuByteSlice *proof_points,
    const CuBbsBbsGpuByteSlice *proof_scalars, size_t output_count,
    CuBbsBbsGpuByteBuffer *t1_outs, CuBbsBbsGpuByteBuffer *t2_outs,
    CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_cuda_g1_dual_msm_same_scalars_lanes(
    const CuBbsBbsGpuByteSlice *lhs_points,
    const CuBbsBbsGpuByteSlice *rhs_points,
    const CuBbsBbsGpuByteSlice *scalars, size_t item_count,
    size_t aggregate_lane_count, uint8_t *lhs_compressed_outs,
    uint8_t *rhs_compressed_outs);

int32_t cubbs_bbs_cuda_g1_dual_msm_same_scalars(
    const CuBbsBbsGpuByteSlice *lhs_points,
    const CuBbsBbsGpuByteSlice *rhs_points,
    const CuBbsBbsGpuByteSlice *scalars, size_t item_count,
    uint8_t *lhs_compressed_out, uint8_t *rhs_compressed_out);

int32_t cubbs_bbs_cuda_g1_dual_msm_same_points(
    const CuBbsBbsGpuByteSlice *points,
    const CuBbsBbsGpuByteSlice *lhs_scalars,
    const CuBbsBbsGpuByteSlice *rhs_scalars, size_t term_count,
    uint8_t *lhs_compressed_out, uint8_t *rhs_compressed_out);

int32_t cubbs_bbs_cuda_g1_dual_msm_same_points_lanes(
    const CuBbsBbsGpuByteSlice *points,
    const CuBbsBbsGpuByteSlice *lhs_scalars,
    const CuBbsBbsGpuByteSlice *rhs_scalars, size_t term_count,
    size_t aggregate_lane_count, uint8_t *lhs_compressed_outs,
    uint8_t *rhs_compressed_outs);

int32_t cubbs_bbs_blst_g1_msm_batch(const CuBbsG1MsmInput *items,
                                     size_t item_count,
                                     CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_blst_g1_msm_common_points_batch(
    const CuBbsG1MsmInput *items, size_t item_count,
    CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_blst_g1_msm_common_points_matrix_packed(
    const CuBbsBbsGpuByteSlice *points, size_t term_count,
    const uint8_t *scalar_rows_be, size_t output_count,
    size_t scalar_row_stride, CuBbsBbsGpuByteBuffer *point_outs,
    CuBbsBbsGpuResult *results);

int32_t cubbs_bbs_blst_decode_g1_msm_terms(const CuBbsG1MsmInput *item,
                                            uint8_t *affine_out,
                                            size_t affine_stride,
                                            uint8_t *scalar_out,
                                            size_t scalar_stride);

int32_t cubbs_bbs_blst_decode_g1_points(const CuBbsBbsGpuByteSlice *points,
                                         size_t point_count,
                                         uint8_t *affine_out,
                                         size_t affine_stride);

int32_t cubbs_bbs_blst_decode_scalars(const CuBbsBbsGpuByteSlice *scalars,
                                       size_t scalar_count,
                                       uint8_t *scalar_out,
                                       size_t scalar_stride);

int32_t cubbs_bbs_blst_compress_g1_projective(const void *projective,
                                               uint8_t *compressed_out);

int32_t cubbs_bbs_blst_compress_g1_projective_batch(
    const void *const *projectives, uint8_t *const *compressed_out,
    size_t point_count);

int32_t cubbs_bbs_cuda_miller_loop_common_lines_batch(
    const void *lhs_g1_affines, const void *rhs_g1_affines,
    const void *lhs_lines, size_t lhs_lines_stride, const void *rhs_lines,
    size_t rhs_lines_stride, size_t item_count, void *lhs_fp12_out,
    void *rhs_fp12_out, uint8_t *finalverify_out,
    const uint8_t *lhs_g2_key, const uint8_t *rhs_g2_key);

int32_t cubbs_bbs_cuda_proof_gen_t2_prepare(
    struct CuBbsBbsGpuProofGenT2PrepareItem *items, size_t item_count,
    CuBbsBbsGpuResult *results);

#endif
