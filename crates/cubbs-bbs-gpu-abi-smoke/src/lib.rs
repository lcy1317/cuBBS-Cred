//! CPU-backed shared library used to smoke-test the cuBBS-Cred GPU BBS ABI.
//!
//! This crate intentionally does not claim hardware acceleration. It proves that
//! cuBBS-Cred can cross the dynamic-library ABI boundary before real CUDA kernels
//! are linked.

use std::{panic, ptr, slice};

use cubbs_bbs::{
    gpu_abi, ProofGenRequest, ProofVerifyRequest, SignRequest, VerifyRequest, ZkryptiumBbsEngine,
};
use zkryptium::{
    bbsplus::ciphersuites::Bls12381Sha256,
    utils::{message::bbsplus_message::BBSplusMessage, util::bbsplus_utils::hash_to_scalar},
};

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_backend_info_v2(
    abi_version: u32,
    info_out: *mut gpu_abi::CuBbsBbsGpuBackendInfo,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION || info_out.is_null() {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }

    ptr::write(
        info_out,
        gpu_abi::CuBbsBbsGpuBackendInfo {
            backend_kind: gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CPU_ABI_TEST,
            hardware_acceleration: 0,
        },
    );
    gpu_abi::CUBBS_BBS_GPU_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_sign_batch_v2(
    abi_version: u32,
    items: *mut gpu_abi::CuBbsBbsGpuSignItem,
    item_count: usize,
    results: *mut gpu_abi::CuBbsBbsGpuResult,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    ffi_guard(|| {
        let items = mut_slice(items, item_count)?;
        let results = mut_slice(results, item_count)?;
        let engine = ZkryptiumBbsEngine::new();

        for (item, result) in items.iter_mut().zip(results.iter_mut()) {
            let request = SignRequest {
                secret_key: bytes(item.secret_key)?.to_vec(),
                public_key: bytes(item.public_key)?.to_vec(),
                header: bytes(item.header)?.to_vec(),
                messages: byte_slice_array(item.messages, item.message_count)?,
            };
            match engine.sign(request) {
                Ok(signature) => write_bytes(item.signature_out, &signature, result),
                Err(_) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL, 0, 0),
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_verify_batch_v2(
    abi_version: u32,
    items: *const gpu_abi::CuBbsBbsGpuVerifyItem,
    item_count: usize,
    results: *mut gpu_abi::CuBbsBbsGpuResult,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    ffi_guard(|| {
        let items = const_slice(items, item_count)?;
        let results = mut_slice(results, item_count)?;
        let engine = ZkryptiumBbsEngine::new();

        for (item, result) in items.iter().zip(results.iter_mut()) {
            let request = VerifyRequest {
                public_key: bytes(item.public_key)?.to_vec(),
                signature: bytes(item.signature)?.to_vec(),
                header: bytes(item.header)?.to_vec(),
                messages: byte_slice_array(item.messages, item.message_count)?,
            };
            match engine.verify(request) {
                Ok(ok) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_SUCCESS, ok as u8, 0),
                Err(_) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL, 0, 0),
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_proof_gen_batch_v2(
    abi_version: u32,
    items: *mut gpu_abi::CuBbsBbsGpuProofGenItem,
    item_count: usize,
    results: *mut gpu_abi::CuBbsBbsGpuResult,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    ffi_guard(|| {
        let items = mut_slice(items, item_count)?;
        let results = mut_slice(results, item_count)?;
        let engine = ZkryptiumBbsEngine::new();

        for (item, result) in items.iter_mut().zip(results.iter_mut()) {
            let request = ProofGenRequest {
                public_key: bytes(item.public_key)?.to_vec(),
                signature: bytes(item.signature)?.to_vec(),
                header: bytes(item.header)?.to_vec(),
                presentation_header: bytes(item.presentation_header)?.to_vec(),
                messages: byte_slice_array(item.messages, item.message_count)?,
                disclosed_indexes: indexes(item.disclosed_indexes)?.to_vec(),
            };
            match engine.proof_gen(request) {
                Ok(proof) => write_bytes(item.proof_out, &proof, result),
                Err(_) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL, 0, 0),
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_proof_verify_batch_v2(
    abi_version: u32,
    items: *const gpu_abi::CuBbsBbsGpuProofVerifyItem,
    item_count: usize,
    results: *mut gpu_abi::CuBbsBbsGpuResult,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    ffi_guard(|| {
        let items = const_slice(items, item_count)?;
        let results = mut_slice(results, item_count)?;
        let engine = ZkryptiumBbsEngine::new();

        for (item, result) in items.iter().zip(results.iter_mut()) {
            let request = ProofVerifyRequest {
                public_key: bytes(item.public_key)?.to_vec(),
                proof: bytes(item.proof)?.to_vec(),
                header: bytes(item.header)?.to_vec(),
                presentation_header: bytes(item.presentation_header)?.to_vec(),
                disclosed_messages: byte_slice_array(
                    item.disclosed_messages,
                    item.disclosed_count,
                )?,
                disclosed_indexes: indexes(item.disclosed_indexes)?.to_vec(),
                total_message_count: item.total_message_count,
            };
            match engine.proof_verify(request) {
                Ok(ok) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_SUCCESS, ok as u8, 0),
                Err(_) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL, 0, 0),
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_messages_to_scalar_batch_v2(
    abi_version: u32,
    items: *mut gpu_abi::CuBbsBbsGpuMessagesToScalarItem,
    item_count: usize,
    results: *mut gpu_abi::CuBbsBbsGpuResult,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    ffi_guard(|| {
        let items = mut_slice(items, item_count)?;
        let results = mut_slice(results, item_count)?;

        for (item, result) in items.iter_mut().zip(results.iter_mut()) {
            let message = bytes(item.message)?;
            let dst = bytes(item.dst)?;
            match hash_to_scalar::<Bls12381Sha256>(message, dst)
                .map(BBSplusMessage::new)
                .map(|scalar| scalar.to_bytes_be().to_vec())
            {
                Ok(scalar) => write_bytes(item.scalar_out, &scalar, result),
                Err(_) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL, 0, 0),
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn cubbs_bbs_gpu_prepare_b_batch_v2(
    abi_version: u32,
    items: *mut gpu_abi::CuBbsBbsGpuPrepareBItem,
    item_count: usize,
    results: *mut gpu_abi::CuBbsBbsGpuResult,
) -> i32 {
    if abi_version != gpu_abi::CUBBS_BBS_GPU_ABI_VERSION {
        return gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT;
    }
    ffi_guard(|| {
        let items = mut_slice(items, item_count)?;
        let results = mut_slice(results, item_count)?;
        let engine = ZkryptiumBbsEngine::new();

        for (item, result) in items.iter_mut().zip(results.iter_mut()) {
            let public_key = bytes(item.public_key)?;
            let header = bytes(item.header)?;
            let message_scalars = byte_slice_array(item.message_scalars, item.message_count)?;
            match engine.prepare_b_for_message_scalars(public_key, header, &message_scalars) {
                Ok(prepared_b) => write_bytes(item.b_out, &prepared_b.compressed_b, result),
                Err(_) => *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL, 0, 0),
            }
        }

        Ok(())
    })
}

fn ffi_guard(f: impl FnOnce() -> Result<(), ()>) -> i32 {
    match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => gpu_abi::CUBBS_BBS_GPU_SUCCESS,
        Ok(Err(())) => gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT,
        Err(_) => gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL,
    }
}

unsafe fn const_slice<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T], ()> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(());
    }
    Ok(slice::from_raw_parts(ptr, len))
}

unsafe fn mut_slice<'a, T>(ptr: *mut T, len: usize) -> Result<&'a mut [T], ()> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(());
    }
    Ok(slice::from_raw_parts_mut(ptr, len))
}

unsafe fn bytes<'a>(value: gpu_abi::CuBbsBbsGpuByteSlice) -> Result<&'a [u8], ()> {
    const_slice(value.data, value.len)
}

unsafe fn indexes<'a>(value: gpu_abi::CuBbsBbsGpuIndexSlice) -> Result<&'a [usize], ()> {
    const_slice(value.data, value.len)
}

unsafe fn byte_slice_array(
    ptr: *const gpu_abi::CuBbsBbsGpuByteSlice,
    len: usize,
) -> Result<Vec<Vec<u8>>, ()> {
    const_slice(ptr, len)?
        .iter()
        .map(|value| bytes(*value).map(<[u8]>::to_vec))
        .collect()
}

fn write_bytes(
    out: gpu_abi::CuBbsBbsGpuByteBuffer,
    bytes: &[u8],
    result: &mut gpu_abi::CuBbsBbsGpuResult,
) {
    if out.data.is_null() || out.capacity < bytes.len() {
        *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_ERR_INSUFFICIENT_OUTPUT, 0, 0);
        return;
    }

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out.data, bytes.len());
    }
    *result = gpu_result(gpu_abi::CUBBS_BBS_GPU_SUCCESS, 0, bytes.len());
}

fn gpu_result(code: i32, value: u8, bytes_written: usize) -> gpu_abi::CuBbsBbsGpuResult {
    gpu_abi::CuBbsBbsGpuResult {
        code,
        value,
        bytes_written,
    }
}
