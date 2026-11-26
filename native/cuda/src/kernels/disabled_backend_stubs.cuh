#if !defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

int32_t cubbs_bbs_cuda_proof_gen_t2_prepare(
    CuBbsBbsGpuProofGenT2PrepareItem *items, size_t item_count,
    CuBbsBbsGpuResult *results) {
  (void)items;
  if (item_count > 0u && results == nullptr) {
    return CUBBS_BBS_GPU_ERR_INVALID_INPUT;
  }
  for (size_t i = 0; i < item_count; ++i) {
    set_result(results, i, CUBBS_BBS_GPU_ERR_INTERNAL, 0);
  }
  return CUBBS_BBS_GPU_SUCCESS;
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_scalars_lanes(
    const CuBbsBbsGpuByteSlice *, const CuBbsBbsGpuByteSlice *,
    const CuBbsBbsGpuByteSlice *, size_t, size_t, uint8_t *, uint8_t *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_scalars(
    const CuBbsBbsGpuByteSlice *, const CuBbsBbsGpuByteSlice *,
    const CuBbsBbsGpuByteSlice *, size_t, uint8_t *, uint8_t *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_points(
    const CuBbsBbsGpuByteSlice *, const CuBbsBbsGpuByteSlice *,
    const CuBbsBbsGpuByteSlice *, size_t, uint8_t *, uint8_t *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_g1_dual_msm_same_points_lanes(
    const CuBbsBbsGpuByteSlice *, const CuBbsBbsGpuByteSlice *,
    const CuBbsBbsGpuByteSlice *, size_t, size_t, uint8_t *, uint8_t *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_sign_fused_common_batch(
    const CuBbsBbsGpuSignFusedCommonBatchItem *, CuBbsBbsGpuResult *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_verify_pre_b_aggregate_fused(
    const CuBbsBbsGpuVerifyPreBAggregateFusedItem *, CuBbsBbsGpuResult *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_pipeline_pinned_scratch(
    size_t, size_t, uint8_t **, uint64_t **, uint64_t **) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_proof_gen_fused_common_batch(
    const CuBbsBbsGpuProofGenFusedCommonBatchItem *, CuBbsBbsGpuResult *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}

int32_t cubbs_bbs_cuda_proof_verify_fused_common_batch(
    const CuBbsBbsGpuProofVerifyFusedCommonBatchItem *, CuBbsBbsGpuResult *) {
  return CUBBS_BBS_GPU_ERR_INTERNAL;
}
#endif
