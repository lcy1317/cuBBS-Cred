//! C ABI contract for a future hardware GPU BBS backend.
//!
//! The Rust W3C layer talks to `BbsEngine`; a CUDA shared library can implement
//! this ABI and be called from `GpuBbsEngine` without changing the W3C API or
//! proofValue formats.

use crate::{BBS2023_HEADER_LEN, PUBLIC_KEY_LEN, SECRET_KEY_LEN, SIGNATURE_LEN};

pub const CUBBS_BBS_GPU_ABI_VERSION: u32 = 2;

pub const CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM: u8 = 1;
pub const CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS: u8 = 2;
pub const CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE: u8 = 3;
pub const CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL: u8 = 4;
pub const CUBBS_BBS_MSM_ROUTE_BLST_ITEM: u8 = 5;
pub const CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS: u8 = 6;
pub const CUBBS_BBS_MSM_ROUTE_SPPARK_ITEM_REFERENCE: u8 = 7;
pub const CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE: u8 = 8;

pub const CUBBS_BBS_GPU_SIGN_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_sign_batch_v2\0";
pub const CUBBS_BBS_GPU_VERIFY_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_verify_batch_v2\0";
pub const CUBBS_BBS_GPU_PROOF_GEN_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_proof_gen_batch_v2\0";
pub const CUBBS_BBS_GPU_PROOF_VERIFY_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_proof_verify_batch_v2\0";
pub const CUBBS_BBS_GPU_BACKEND_INFO_SYMBOL: &[u8] = b"cubbs_bbs_gpu_backend_info_v2\0";
pub const CUBBS_BBS_GPU_MESSAGES_TO_SCALAR_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_messages_to_scalar_batch_v2\0";
pub const CUBBS_BBS_GPU_SIGN_E_FROM_RESIDENT_SCALARS_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_sign_e_from_resident_scalars_batch_v2\0";
pub const CUBBS_BBS_GPU_PREPARE_B_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_prepare_b_batch_v2\0";
pub const CUBBS_BBS_GPU_G1_MSM_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_g1_msm_batch_v2\0";
pub const CUBBS_BBS_GPU_G1_MSM_COMMON_POINTS_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_g1_msm_common_points_batch_v2\0";
pub const CUBBS_BBS_GPU_G1_MSM_COMMON_POINTS_PLUS_VARIABLE_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_g1_msm_common_points_plus_variable_batch_v2\0";
pub const CUBBS_BBS_GPU_PROOF_VERIFY_MSM_PAIR_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_proof_verify_msm_pair_batch_v2\0";
pub const CUBBS_BBS_GPU_G1_DUAL_MSM_SAME_POINTS_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_g1_dual_msm_same_points_batch_v2\0";
pub const CUBBS_BBS_GPU_G1_DUAL_MSM_SAME_POINTS_LANES_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_g1_dual_msm_same_points_lanes_batch_v2\0";
pub const CUBBS_BBS_GPU_G1_ADD_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_g1_add_batch_v2\0";
pub const CUBBS_BBS_GPU_PAIRING_EQ_BATCH_SYMBOL: &[u8] = b"cubbs_bbs_gpu_pairing_eq_batch_v2\0";
pub const CUBBS_BBS_GPU_PAIRING_EQ_AGGREGATE_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_pairing_eq_aggregate_v2\0";
pub const CUBBS_BBS_GPU_PROOF_GEN_T2_PREPARE_BATCH_SYMBOL: &[u8] =
    b"cubbs_bbs_gpu_proof_gen_t2_prepare_batch_v2\0";

pub const CUBBS_BBS_GPU_SUCCESS: i32 = 0;
pub const CUBBS_BBS_GPU_ERR_INVALID_INPUT: i32 = 1;
pub const CUBBS_BBS_GPU_ERR_INSUFFICIENT_OUTPUT: i32 = 2;
pub const CUBBS_BBS_GPU_ERR_INTERNAL: i32 = 3;

pub const CUBBS_BBS_GPU_BACKEND_KIND_UNKNOWN: u32 = 0;
pub const CUBBS_BBS_GPU_BACKEND_KIND_CUDA: u32 = 1;
pub const CUBBS_BBS_GPU_BACKEND_KIND_CPU_ABI_TEST: u32 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuByteSlice {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuByteBuffer {
    pub data: *mut u8,
    pub capacity: usize,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuIndexSlice {
    pub data: *const usize,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuResult {
    /// `CUBBS_BBS_GPU_SUCCESS` on success, otherwise an `CUBBS_BBS_GPU_ERR_*`.
    pub code: i32,
    /// Boolean result for verify operations; ignored for sign/proof generation.
    pub value: u8,
    /// Number of bytes written to the operation output buffer.
    pub bytes_written: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuBackendInfo {
    /// One of `CUBBS_BBS_GPU_BACKEND_KIND_*`.
    pub backend_kind: u32,
    /// Non-zero only for a real hardware GPU implementation.
    pub hardware_acceleration: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuSignItem {
    /// Must be exactly `SECRET_KEY_LEN` bytes.
    pub secret_key: CuBbsBbsGpuByteSlice,
    /// Must be exactly `PUBLIC_KEY_LEN` bytes.
    pub public_key: CuBbsBbsGpuByteSlice,
    /// Must be exactly `BBS2023_HEADER_LEN` bytes.
    pub header: CuBbsBbsGpuByteSlice,
    pub messages: *const CuBbsBbsGpuByteSlice,
    pub message_count: usize,
    /// Caller-provided output buffer. Must fit `SIGNATURE_LEN` bytes.
    pub signature_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuVerifyItem {
    /// Must be exactly `PUBLIC_KEY_LEN` bytes.
    pub public_key: CuBbsBbsGpuByteSlice,
    /// Must be exactly `SIGNATURE_LEN` bytes.
    pub signature: CuBbsBbsGpuByteSlice,
    /// Must be exactly `BBS2023_HEADER_LEN` bytes.
    pub header: CuBbsBbsGpuByteSlice,
    pub messages: *const CuBbsBbsGpuByteSlice,
    pub message_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuProofGenItem {
    /// Must be exactly `PUBLIC_KEY_LEN` bytes.
    pub public_key: CuBbsBbsGpuByteSlice,
    /// Must be exactly `SIGNATURE_LEN` bytes.
    pub signature: CuBbsBbsGpuByteSlice,
    /// Must be exactly `BBS2023_HEADER_LEN` bytes.
    pub header: CuBbsBbsGpuByteSlice,
    pub presentation_header: CuBbsBbsGpuByteSlice,
    pub messages: *const CuBbsBbsGpuByteSlice,
    pub message_count: usize,
    pub disclosed_indexes: CuBbsBbsGpuIndexSlice,
    /// Caller-provided proof output buffer.
    pub proof_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuProofVerifyItem {
    /// Must be exactly `PUBLIC_KEY_LEN` bytes.
    pub public_key: CuBbsBbsGpuByteSlice,
    pub proof: CuBbsBbsGpuByteSlice,
    /// Must be exactly `BBS2023_HEADER_LEN` bytes.
    pub header: CuBbsBbsGpuByteSlice,
    pub presentation_header: CuBbsBbsGpuByteSlice,
    pub disclosed_messages: *const CuBbsBbsGpuByteSlice,
    pub disclosed_count: usize,
    pub disclosed_indexes: CuBbsBbsGpuIndexSlice,
    pub total_message_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuMessagesToScalarItem {
    pub message: CuBbsBbsGpuByteSlice,
    pub dst: CuBbsBbsGpuByteSlice,
    /// Caller-provided scalar output buffer. Must fit 32 bytes.
    pub scalar_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuSignEFromResidentScalarsItem {
    pub secret_key: CuBbsBbsGpuByteSlice,
    pub domain: CuBbsBbsGpuByteSlice,
    pub messages: *const CuBbsBbsGpuByteSlice,
    pub message_scalar_offset: usize,
    pub message_count: usize,
    pub message_scalars_out: CuBbsBbsGpuByteBuffer,
    pub scalar_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuPrepareBItem {
    /// Must be exactly `PUBLIC_KEY_LEN` bytes.
    pub public_key: CuBbsBbsGpuByteSlice,
    /// Must be exactly `BBS2023_HEADER_LEN` bytes.
    pub header: CuBbsBbsGpuByteSlice,
    /// Big-endian 32-byte BLS12-381 scalars.
    pub message_scalars: *const CuBbsBbsGpuByteSlice,
    pub message_count: usize,
    /// Caller-provided compressed G1 output buffer. Must fit 48 bytes.
    pub b_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuG1MsmItem {
    /// Compressed G1 points, 48 bytes each.
    pub points: *const CuBbsBbsGpuByteSlice,
    /// Big-endian 32-byte scalars.
    pub scalars: *const CuBbsBbsGpuByteSlice,
    pub term_count: usize,
    /// Caller-provided compressed G1 output buffer. Must fit 48 bytes.
    pub point_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuG1MsmCommonPointsBatchItem {
    /// Compressed G1 points, 48 bytes each, shared by every output row.
    pub points: *const CuBbsBbsGpuByteSlice,
    pub term_count: usize,
    /// Row-major big-endian 32-byte scalars: output_count * term_count slices.
    pub scalar_rows: *const CuBbsBbsGpuByteSlice,
    /// Optional row-major packed big-endian scalar matrix.
    pub scalar_rows_be: *const u8,
    /// Distance in bytes between two packed scalar rows.
    pub scalar_row_stride: usize,
    pub output_count: usize,
    /// Caller-provided compressed G1 output buffers. Each must fit 48 bytes.
    pub point_outs: *mut CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem {
    pub points: *const CuBbsBbsGpuByteSlice,
    pub term_count: usize,
    pub scalar_rows: *const CuBbsBbsGpuByteSlice,
    pub variable_points: *const CuBbsBbsGpuByteSlice,
    pub variable_scalars: *const CuBbsBbsGpuByteSlice,
    pub output_count: usize,
    pub point_outs: *mut CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuProofVerifyMsmPairItem {
    pub common_points: *const CuBbsBbsGpuByteSlice,
    pub common_term_count: usize,
    pub common_scalar_rows: *const CuBbsBbsGpuByteSlice,
    pub proof_points: *const CuBbsBbsGpuByteSlice,
    pub proof_scalars: *const CuBbsBbsGpuByteSlice,
    pub output_count: usize,
    pub t1_outs: *mut CuBbsBbsGpuByteBuffer,
    pub t2_outs: *mut CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuG1DualMsmSamePointsItem {
    pub points: *const CuBbsBbsGpuByteSlice,
    pub lhs_scalars: *const CuBbsBbsGpuByteSlice,
    pub rhs_scalars: *const CuBbsBbsGpuByteSlice,
    pub term_count: usize,
    pub lhs_out: CuBbsBbsGpuByteBuffer,
    pub rhs_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuG1DualMsmSamePointsLanesItem {
    pub points: *const CuBbsBbsGpuByteSlice,
    pub lhs_scalars: *const CuBbsBbsGpuByteSlice,
    pub rhs_scalars: *const CuBbsBbsGpuByteSlice,
    pub term_count: usize,
    pub lane_count: usize,
    pub lhs_outs: CuBbsBbsGpuByteBuffer,
    pub rhs_outs: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuG1AddItem {
    /// Compressed G1 point, 48 bytes.
    pub lhs: CuBbsBbsGpuByteSlice,
    /// Compressed G1 point, 48 bytes.
    pub rhs: CuBbsBbsGpuByteSlice,
    /// Caller-provided compressed G1 output buffer. Must fit 48 bytes.
    pub point_out: CuBbsBbsGpuByteBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuPairingEqItem {
    /// Compressed G1 point, 48 bytes.
    pub lhs_g1: CuBbsBbsGpuByteSlice,
    /// Compressed G2 point, 96 bytes.
    pub lhs_g2: CuBbsBbsGpuByteSlice,
    /// Compressed G1 point, 48 bytes.
    pub rhs_g1: CuBbsBbsGpuByteSlice,
    /// Compressed G2 point, 96 bytes.
    pub rhs_g2: CuBbsBbsGpuByteSlice,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuPairingEqScaledItem {
    /// Compressed G1 point, 48 bytes.
    pub lhs_g1: CuBbsBbsGpuByteSlice,
    /// Compressed G2 point, 96 bytes.
    pub lhs_g2: CuBbsBbsGpuByteSlice,
    /// Compressed G1 point, 48 bytes.
    pub rhs_g1: CuBbsBbsGpuByteSlice,
    /// Compressed G2 point, 96 bytes.
    pub rhs_g2: CuBbsBbsGpuByteSlice,
    /// Big-endian 32-byte scalar coefficient applied to both G1 points.
    pub coefficient: CuBbsBbsGpuByteSlice,
}

/// GPU proof-gen t2Prepare item. Computes the 6 fixed BBS G1 scalar
/// multiplications (d, abar, bbar, t1) on the device, eliminating the host
/// t2Prepare phase (dominated by 6 G1 scalar muls at ~215us each on CPU) that
/// sits between the B-MSM and T2-MSM and forces the GPU idle.
///
/// The Rust caller precomputes the 7 host-side scalar multipliers (including
/// composite products, since the device has no scalar-field arithmetic) as big-endian
/// 32-byte blobs, and generates the random scalars on the host (cost ~3us).
/// The device only does G1 scalar multiplication + point addition.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuBbsBbsGpuProofGenT2PrepareItem {
    /// Compressed G1 point B, 48 bytes (GPU B-MSM output).
    pub compressed_b: CuBbsBbsGpuByteSlice,
    /// Compressed G1 point signature.A, 48 bytes.
    pub signature_a: CuBbsBbsGpuByteSlice,
    /// 7 host-precomputed big-endian 32-byte scalar multipliers:
    /// [0]=r2, [1]=r1*r2, [2]=r1*r2, [3]=r1*r2*e,
    /// [4]=r1*r2*e_tilde, [5]=r2*r1_tilde, [6]=r3_tilde.
    pub scalars_be: [CuBbsBbsGpuByteSlice; 7],
    /// 32-byte CSPRNG seed (reserved for future device-side derivation).
    pub seed: CuBbsBbsGpuByteSlice,
    /// Output: compressed G1 point d = B * r2, 48 bytes.
    pub d_out: CuBbsBbsGpuByteBuffer,
    /// Output: compressed G1 point abar = A * (r1 * r2), 48 bytes.
    pub abar_out: CuBbsBbsGpuByteBuffer,
    /// Output: compressed G1 point bbar = d*r1 - abar*e, 48 bytes.
    pub bbar_out: CuBbsBbsGpuByteBuffer,
    /// Output: compressed G1 point t1 = abar*e_tilde + d*r1_tilde, 48 bytes.
    pub t1_out: CuBbsBbsGpuByteBuffer,
}

pub type CuBbsBbsGpuSignBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuSignItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuVerifyBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *const CuBbsBbsGpuVerifyItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuProofGenBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuProofGenItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuProofVerifyBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *const CuBbsBbsGpuProofVerifyItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuBackendInfoV2 =
    unsafe extern "C" fn(abi_version: u32, info_out: *mut CuBbsBbsGpuBackendInfo) -> i32;

pub type CuBbsBbsGpuMessagesToScalarBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuMessagesToScalarItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuSignEFromResidentScalarsBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuSignEFromResidentScalarsItem,
    item_count: usize,
    message_dst: CuBbsBbsGpuByteSlice,
    e_dst: CuBbsBbsGpuByteSlice,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuPrepareBBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuPrepareBItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuG1MsmBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuG1MsmItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuG1MsmCommonPointsBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuG1MsmCommonPointsBatchItem,
    group_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuG1MsmCommonPointsPlusVariableBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem,
    group_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuProofVerifyMsmPairBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuProofVerifyMsmPairItem,
    group_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuG1DualMsmSamePointsBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuG1DualMsmSamePointsItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuG1DualMsmSamePointsLanesBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuG1DualMsmSamePointsLanesItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuG1AddBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuG1AddItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuPairingEqBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *const CuBbsBbsGpuPairingEqItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuPairingEqAggregateV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *const CuBbsBbsGpuPairingEqScaledItem,
    item_count: usize,
    result: *mut CuBbsBbsGpuResult,
) -> i32;

pub type CuBbsBbsGpuProofGenT2PrepareBatchV2 = unsafe extern "C" fn(
    abi_version: u32,
    items: *mut CuBbsBbsGpuProofGenT2PrepareItem,
    item_count: usize,
    results: *mut CuBbsBbsGpuResult,
) -> i32;

pub const fn fixed_lengths() -> (usize, usize, usize, usize) {
    (
        SECRET_KEY_LEN,
        PUBLIC_KEY_LEN,
        SIGNATURE_LEN,
        BBS2023_HEADER_LEN,
    )
}
