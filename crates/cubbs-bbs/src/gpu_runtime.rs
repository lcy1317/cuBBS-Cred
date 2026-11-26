//! Native library lifetime, ABI calls, and prepared MSM buffers.

use super::*;
use std::cell::RefCell;

thread_local! {
    // Reusable scratch for FFI point-slice arrays (avoids 1 MB malloc/free per M=65,536 call).
    static POINT_SLICES_SCRATCH: RefCell<Vec<gpu_abi::CuBbsBbsGpuByteSlice>> =
        RefCell::new(Vec::new());
}

#[derive(Clone, Copy)]
pub(super) enum CompactMsgMeta<'a> {
    Uniform,
    U32Offsets(&'a [u32]),
    U64OffsetsLengths(&'a [u64], &'a [u64]),
}

#[derive(Clone)]
pub(super) struct GpuKernelLibrary {
    pub(super) _library: Arc<Library>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) messages_to_scalar_batch: Option<gpu_abi::CuBbsBbsGpuMessagesToScalarBatchV2>,
    pub(super) sign_e_from_resident_scalars_batch:
        Option<gpu_abi::CuBbsBbsGpuSignEFromResidentScalarsBatchV2>,
    pub(super) prepare_b_batch: Option<gpu_abi::CuBbsBbsGpuPrepareBBatchV2>,
    pub(super) g1_msm_batch: Option<gpu_abi::CuBbsBbsGpuG1MsmBatchV2>,
    pub(super) g1_msm_common_points_batch: Option<gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchV2>,
    pub(super) g1_dual_msm_same_points_batch:
        Option<gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsBatchV2>,
    pub(super) g1_dual_msm_same_points_lanes_batch:
        Option<gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsLanesBatchV2>,
    pub(super) g1_add_batch: Option<gpu_abi::CuBbsBbsGpuG1AddBatchV2>,
    pub(super) pairing_eq_batch: Option<gpu_abi::CuBbsBbsGpuPairingEqBatchV2>,
    pub(super) pairing_eq_aggregate: Option<gpu_abi::CuBbsBbsGpuPairingEqAggregateV2>,
    pub(super) sign_fused_common_batch: Option<gpu_abi::CuBbsBbsGpuSignFusedCommonBatchV2>,
    pub(super) verify_pre_b_aggregate_fused:
        Option<gpu_abi::CuBbsBbsGpuVerifyPreBAggregateFusedV2>,
    pub(super) pipeline_pinned_scratch: Option<gpu_abi::CuBbsBbsGpuPipelinePinnedScratchV2>,
    pub(super) proof_gen_fused_common_batch:
        Option<gpu_abi::CuBbsBbsGpuProofGenFusedCommonBatchV2>,
    pub(super) proof_verify_fused_common_batch:
        Option<gpu_abi::CuBbsBbsGpuProofVerifyFusedCommonBatchV2>,
    pub(super) backend_kind: u32,
    pub(super) hardware_acceleration: bool,
}

pub(super) struct PreparedGpuMsmBenchGroup {
    // These owned buffers are never resized after construction. The ABI
    // descriptors below therefore keep stable pointers for the plan lifetime.
    pub(super) _compressed_points: Vec<Vec<u8>>,
    pub(super) point_slices: Vec<gpu_abi::CuBbsBbsGpuByteSlice>,
    pub(super) scalar_rows_be: Vec<u8>,
    pub(super) outputs: Vec<Vec<u8>>,
    pub(super) point_outs: Vec<gpu_abi::CuBbsBbsGpuByteBuffer>,
}

/// Allocation-free Rust-side execution plan for common-points GPU MSM.
///
/// The plan deliberately requires `&mut self`: one plan owns one set of output
/// buffers and may have only one in-flight native call. Create one plan per
/// benchmark stream. Preparation and validation happen once; [`Self::run`]
/// reuses all Rust-owned input descriptors, result slots, and output buffers.
pub struct GpuMsmBenchPlan {
    pub(super) kernel: Arc<GpuKernelLibrary>,
    pub(super) groups: Vec<PreparedGpuMsmBenchGroup>,
    pub(super) items: Vec<gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchItem>,
    pub(super) results: Vec<gpu_abi::CuBbsBbsGpuResult>,
    pub(super) total_outputs: usize,
}

impl fmt::Debug for GpuMsmBenchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuMsmBenchPlan")
            .field("kernel", &self.kernel)
            .field("group_count", &self.groups.len())
            .field("total_outputs", &self.total_outputs)
            .finish()
    }
}

impl GpuMsmBenchPlan {
    pub(super) fn prepare(
        kernel: Arc<GpuKernelLibrary>,
        inputs: Vec<GpuMsmBenchGroupInput>,
    ) -> Result<Self> {
        if kernel.g1_msm_common_points_batch.is_none() {
            return Err(Error::Gpu(
                "GPU common-points MSM symbol is unavailable for prepared bench".to_owned(),
            ));
        }
        if inputs.is_empty() {
            return Err(Error::Gpu(
                "prepared GPU MSM bench requires at least one group".to_owned(),
            ));
        }

        let mut groups = Vec::with_capacity(inputs.len());
        let mut total_outputs = 0usize;
        for input in inputs {
            if input.compressed_points.is_empty() || input.output_count == 0 {
                return Err(Error::Gpu(
                    "prepared GPU MSM bench groups must have points and outputs".to_owned(),
                ));
            }
            if input
                .compressed_points
                .iter()
                .any(|point| point.len() != 48)
            {
                return Err(Error::Gpu(
                    "prepared GPU MSM bench points must be compressed 48-byte G1 points".to_owned(),
                ));
            }
            let scalar_row_stride = input
                .compressed_points
                .len()
                .checked_mul(32)
                .ok_or_else(|| Error::Gpu("prepared GPU MSM scalar stride overflow".to_owned()))?;
            let expected_scalar_bytes = input
                .output_count
                .checked_mul(scalar_row_stride)
                .ok_or_else(|| Error::Gpu("prepared GPU MSM scalar matrix overflow".to_owned()))?;
            if input.scalar_rows_be.len() != expected_scalar_bytes {
                return Err(Error::Gpu(format!(
                    "prepared GPU MSM scalar matrix mismatch: expected {expected_scalar_bytes} bytes, got {}",
                    input.scalar_rows_be.len()
                )));
            }
            total_outputs = total_outputs
                .checked_add(input.output_count)
                .ok_or_else(|| Error::Gpu("prepared GPU MSM output count overflow".to_owned()))?;

            let point_slices = input
                .compressed_points
                .iter()
                .map(|point| gpu_abi::CuBbsBbsGpuByteSlice {
                    data: point.as_ptr(),
                    len: point.len(),
                })
                .collect::<Vec<_>>();
            let mut outputs = (0..input.output_count)
                .map(|_| vec![0u8; 48])
                .collect::<Vec<_>>();
            let point_outs = outputs
                .iter_mut()
                .map(|point| gpu_abi::CuBbsBbsGpuByteBuffer {
                    data: point.as_mut_ptr(),
                    capacity: point.len(),
                    len: 0,
                })
                .collect::<Vec<_>>();
            groups.push(PreparedGpuMsmBenchGroup {
                _compressed_points: input.compressed_points,
                point_slices,
                scalar_rows_be: input.scalar_rows_be,
                outputs,
                point_outs,
            });
        }

        // Every pointer below targets a heap allocation owned by `groups`.
        // None of those vectors are resized after this point.
        let items = groups
            .iter_mut()
            .map(|group| gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchItem {
                points: group.point_slices.as_ptr(),
                term_count: group.point_slices.len(),
                scalar_rows: std::ptr::null(),
                scalar_rows_be: group.scalar_rows_be.as_ptr(),
                scalar_row_stride: group.point_slices.len() * 32,
                output_count: group.outputs.len(),
                point_outs: group.point_outs.as_mut_ptr(),
            })
            .collect::<Vec<_>>();

        Ok(Self {
            kernel,
            groups,
            items,
            results: empty_gpu_results(total_outputs),
            total_outputs,
        })
    }

    /// Executes one prepared native batch without allocating or copying input
    /// data on the Rust side.
    pub fn run(&mut self) -> Result<GpuMsmBenchRun> {
        let call = self
            .kernel
            .g1_msm_common_points_batch
            .ok_or_else(|| Error::Gpu("prepared GPU MSM symbol disappeared".to_owned()))?;
        for group in &mut self.groups {
            for output in &mut group.point_outs {
                output.len = 0;
            }
        }
        for result in &mut self.results {
            result.code = gpu_abi::CUBBS_BBS_GPU_SUCCESS;
            result.value = 0;
            result.bytes_written = 0;
        }
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                self.items.as_mut_ptr(),
                self.items.len(),
                self.results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Err(Error::Gpu(gpu_status_message(status)));
        }
        if let Some((index, result)) = self.results.iter().enumerate().find(|(_, result)| {
            result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS || result.bytes_written != 48
        }) {
            return Err(Error::Gpu(format!(
                "prepared GPU MSM output {index} failed: code={}, bytes={}",
                result.code, result.bytes_written
            )));
        }
        if self
            .groups
            .iter()
            .flat_map(|group| group.point_outs.iter())
            .any(|output| output.len != 48)
        {
            return Err(Error::Gpu(
                "prepared GPU MSM returned an invalid compressed point length".to_owned(),
            ));
        }
        Ok(GpuMsmBenchRun {
            group_count: self.groups.len(),
            output_count: self.total_outputs,
        })
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn output_count(&self) -> usize {
        self.total_outputs
    }

    pub fn output(&self, group_index: usize, row_index: usize) -> Option<&[u8]> {
        let group = self.groups.get(group_index)?;
        let len = group.point_outs.get(row_index)?.len;
        group
            .outputs
            .get(row_index)
            .and_then(|output| output.get(..len))
    }
}

impl fmt::Debug for GpuKernelLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuKernelLibrary")
            .field("abi_version", &gpu_abi::CUBBS_BBS_GPU_ABI_VERSION)
            .field("backend_kind", &self.backend_kind)
            .field("hardware_acceleration", &self.hardware_acceleration)
            .finish_non_exhaustive()
    }
}

impl Default for BbsEngine {
    fn default() -> Self {
        Self::new(BbsBackend::ZkryptiumCpu)
    }
}

impl GpuKernelLibrary {
    /// Loads the CUDA ABI library once for each configured library path and
    /// retains a strong reference for the process lifetime.
    ///
    /// A W3C service constructs short-lived `Bbs2023Core` values for individual
    /// batches.  Reusing the resolved library avoids repeated dynamic-loader
    /// reference-count and symbol-resolution work on that hot path, while the
    /// strong reference makes the intended native entrypoint lifetime explicit.
    /// The configured library path is immutable for a service process, while
    /// keying the cache preserves isolated test and embedding configurations
    /// that use another path.
    pub(super) fn load_shared_from_env() -> Result<Option<Arc<Self>>> {
        let Some(path) = std::env::var_os("CUBBS_BBS_GPU_LIBRARY") else {
            return Ok(None);
        };
        if path.is_empty() {
            return Ok(None);
        }

        static LIBRARIES: OnceLock<Mutex<HashMap<std::ffi::OsString, Arc<GpuKernelLibrary>>>> =
            OnceLock::new();
        let libraries = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(kernel) = libraries.lock().unwrap().get(&path).cloned() {
            return Ok(Some(kernel));
        }

        // Do not hold the cache lock while loading the shared object: CUDA
        // initialization can be comparatively slow.  If another thread wins
        // the race, its persistent handle is returned and this temporary
        // handle is safely dropped while the library remains loaded.
        let loaded = Arc::new(Self::load_from_path(&path)?);
        let mut libraries = libraries.lock().unwrap();
        let kernel = match libraries.entry(path) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&loaded));
                loaded
            }
        };
        Ok(Some(kernel))
    }

    pub(super) fn load_from_path(path: &std::ffi::OsStr) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .map_err(|error| Error::Gpu(format!("failed to load GPU library: {error}")))?;
        let messages_to_scalar_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuMessagesToScalarBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_MESSAGES_TO_SCALAR_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let sign_e_from_resident_scalars_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuSignEFromResidentScalarsBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_SIGN_E_FROM_RESIDENT_SCALARS_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let prepare_b_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuPrepareBBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PREPARE_B_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let g1_msm_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuG1MsmBatchV2>(gpu_abi::CUBBS_BBS_GPU_G1_MSM_BATCH_SYMBOL)
                .ok()
                .map(|symbol| *symbol)
        };
        let g1_msm_common_points_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_G1_MSM_COMMON_POINTS_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let g1_dual_msm_same_points_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_G1_DUAL_MSM_SAME_POINTS_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let g1_dual_msm_same_points_lanes_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsLanesBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_G1_DUAL_MSM_SAME_POINTS_LANES_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let g1_add_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuG1AddBatchV2>(gpu_abi::CUBBS_BBS_GPU_G1_ADD_BATCH_SYMBOL)
                .ok()
                .map(|symbol| *symbol)
        };
        let pairing_eq_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuPairingEqBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PAIRING_EQ_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let pairing_eq_aggregate = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuPairingEqAggregateV2>(
                    gpu_abi::CUBBS_BBS_GPU_PAIRING_EQ_AGGREGATE_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let sign_fused_common_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuSignFusedCommonBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_SIGN_FUSED_COMMON_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let verify_pre_b_aggregate_fused = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuVerifyPreBAggregateFusedV2>(
                    gpu_abi::CUBBS_BBS_GPU_VERIFY_PRE_B_AGGREGATE_FUSED_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let pipeline_pinned_scratch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuPipelinePinnedScratchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PIPELINE_PINNED_SCRATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let proof_gen_fused_common_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuProofGenFusedCommonBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PROOF_GEN_FUSED_COMMON_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let proof_verify_fused_common_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuProofVerifyFusedCommonBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PROOF_VERIFY_FUSED_COMMON_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let backend_info = unsafe {
            match library.get::<gpu_abi::CuBbsBbsGpuBackendInfoV2>(
                gpu_abi::CUBBS_BBS_GPU_BACKEND_INFO_SYMBOL,
            ) {
                Ok(symbol) => {
                    let mut info = gpu_abi::CuBbsBbsGpuBackendInfo {
                        backend_kind: gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_UNKNOWN,
                        hardware_acceleration: 0,
                    };
                    let status = symbol(gpu_abi::CUBBS_BBS_GPU_ABI_VERSION, &mut info);
                    if status == gpu_abi::CUBBS_BBS_GPU_SUCCESS {
                        info
                    } else {
                        gpu_abi::CuBbsBbsGpuBackendInfo {
                            backend_kind: gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_UNKNOWN,
                            hardware_acceleration: 0,
                        }
                    }
                }
                Err(_) => gpu_abi::CuBbsBbsGpuBackendInfo {
                    backend_kind: gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_UNKNOWN,
                    hardware_acceleration: 0,
                },
            }
        };

        Ok(Self {
            _library: Arc::new(library),
            messages_to_scalar_batch,
            sign_e_from_resident_scalars_batch,
            prepare_b_batch,
            g1_msm_batch,
            g1_msm_common_points_batch,
            g1_dual_msm_same_points_batch,
            g1_dual_msm_same_points_lanes_batch,
            g1_add_batch,
            pairing_eq_batch,
            pairing_eq_aggregate,
            sign_fused_common_batch,
            verify_pre_b_aggregate_fused,
            pipeline_pinned_scratch,
            proof_gen_fused_common_batch,
            proof_verify_fused_common_batch,
            backend_kind: backend_info.backend_kind,
            hardware_acceleration: backend_info.hardware_acceleration != 0,
        })
    }

    pub(super) fn can_precompute_message_scalars(&self) -> bool {
        self.messages_to_scalar_batch.is_some()
    }

    pub(super) fn uses_hardware_acceleration(&self) -> bool {
        self.hardware_acceleration
            || (self.backend_kind == gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CUDA
                && (self.messages_to_scalar_batch.is_some()
                    || self.prepare_b_batch.is_some()
                    || self.g1_msm_batch.is_some()
                    || self.g1_msm_common_points_batch.is_some()
                    || self.g1_dual_msm_same_points_batch.is_some()
                    || self.g1_dual_msm_same_points_lanes_batch.is_some()
                    || self.g1_add_batch.is_some()
                    || self.pairing_eq_batch.is_some()
                    || self.pairing_eq_aggregate.is_some()))
    }

    pub(super) fn can_prepare_b(&self) -> bool {
        // ABI v2 CUDA libraries retain the prepare-B symbol for binary
        // compatibility, but the CUDA implementation is only a validation
        // stub. CUDA prepare-B is performed through the generic G1 MSM paths.
        self.prepare_b_batch.is_some()
            && self.backend_kind != gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CUDA
    }

    pub(super) fn can_g1_msm(&self) -> bool {
        self.g1_msm_batch.is_some()
    }

    pub(super) fn can_g1_msm_common_points(&self) -> bool {
        self.g1_msm_common_points_batch.is_some()
    }

    pub(super) fn can_g1_dual_msm_same_points(&self) -> bool {
        self.g1_dual_msm_same_points_batch.is_some()
    }

    pub(super) fn can_g1_dual_msm_same_points_lanes(&self) -> bool {
        self.g1_dual_msm_same_points_lanes_batch.is_some()
    }

    pub(super) fn can_g1_add(&self) -> bool {
        self.g1_add_batch.is_some()
    }

    pub(super) fn can_pairing_eq(&self) -> bool {
        self.pairing_eq_batch.is_some()
    }

    pub(super) fn can_pairing_eq_aggregate(&self) -> bool {
        self.pairing_eq_aggregate.is_some()
    }

    pub(super) fn can_sign_fused_common_batch(&self) -> bool {
        self.sign_fused_common_batch.is_some()
            && !std::env::var("CUBBS_BBS_GPU_SIGN_FUSED_RESIDENT")
                .map(|v| matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
                .unwrap_or(false)
    }

    pub(super) fn can_verify_pre_b_aggregate_fused(&self) -> bool {
        self.verify_pre_b_aggregate_fused.is_some()
            && !std::env::var("CUBBS_BBS_GPU_VERIFY_FUSED_RESIDENT")
                .map(|v| matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
                .unwrap_or(false)
    }

    pub(super) fn can_proof_gen_fused_common_batch(&self) -> bool {
        self.proof_gen_fused_common_batch.is_some()
            && !std::env::var("CUBBS_BBS_GPU_PROOF_GEN_FUSED_RESIDENT")
                .map(|v| matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
                .unwrap_or(false)
    }

    pub(super) fn can_proof_verify_fused_common_batch(&self) -> bool {
        self.proof_verify_fused_common_batch.is_some()
            && !std::env::var("CUBBS_BBS_GPU_PROOF_VERIFY_FUSED_RESIDENT")
                .map(|v| matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
                .unwrap_or(false)
    }

    /// Returns thread-local CUDA pinned host slices `(messages, offsets, lengths)`
    /// owned by the native pipeline workspace for zero-copy single-pass packing.
    pub(super) unsafe fn pipeline_pinned_scratch_slices(
        &self,
        msg_bytes: usize,
        msg_count: usize,
    ) -> Option<(&mut [u8], &mut [u64], &mut [u64])> {
        let call = self.pipeline_pinned_scratch?;
        let mut msg_ptr: *mut u8 = std::ptr::null_mut();
        let mut off_ptr: *mut u64 = std::ptr::null_mut();
        let mut len_ptr: *mut u64 = std::ptr::null_mut();
        let status = call(
            gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
            msg_bytes,
            msg_count,
            &mut msg_ptr,
            &mut off_ptr,
            &mut len_ptr,
        );
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return None;
        }
        let msgs = if msg_bytes == 0 {
            &mut [][..]
        } else {
            if msg_ptr.is_null() {
                return None;
            }
            std::slice::from_raw_parts_mut(msg_ptr, msg_bytes)
        };
        let offs = if msg_count == 0 {
            &mut [][..]
        } else {
            if off_ptr.is_null() || len_ptr.is_null() {
                return None;
            }
            std::slice::from_raw_parts_mut(off_ptr, msg_count)
        };
        let lens = if msg_count == 0 {
            &mut [][..]
        } else {
            std::slice::from_raw_parts_mut(len_ptr, msg_count)
        };
        Some((msgs, offs, lens))
    }

    /// Compact message metadata encoding passed over the existing v2 ABI:
    /// - `Uniform`: both pointers NULL (0 B/msg metadata when all messages have equal length)
    /// - `U32Offsets`: `message_offsets` points to `u32[N + 1]`, `message_lengths` is NULL (4 B/msg)
    /// - `U64OffsetsLengths`: legacy `u64[N]` offsets + `u64[N]` lengths (16 B/msg)
    pub(super) fn compact_meta_ptrs(meta: CompactMsgMeta<'_>) -> (*const u64, *const u64) {
        match meta {
            CompactMsgMeta::Uniform => (std::ptr::null(), std::ptr::null()),
            CompactMsgMeta::U32Offsets(offs32) => {
                if offs32.is_empty() {
                    (std::ptr::null(), std::ptr::null())
                } else {
                    (offs32.as_ptr() as *const u64, std::ptr::null())
                }
            }
            CompactMsgMeta::U64OffsetsLengths(offs, lens) => {
                if offs.is_empty() {
                    (std::ptr::null(), std::ptr::null())
                } else {
                    (offs.as_ptr(), lens.as_ptr())
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sign_fused_common_batch(
        &self,
        common_points: &[Vec<u8>],
        item_count: usize,
        secret_keys_be: &[u8],
        secret_key_stride: usize,
        domains_be: &[u8],
        domain_stride: usize,
        packed_messages: &[u8],
        msg_meta: CompactMsgMeta<'_>,
    ) -> Result<Vec<u8>> {
        let call = self.sign_fused_common_batch.ok_or_else(|| {
            Error::Gpu("GPU fused resident Sign symbol is unavailable".to_owned())
        })?;
        let point_slices = common_points
            .iter()
            .map(|p| byte_slice(p))
            .collect::<Vec<_>>();
        let message_dst = bbs_message_scalar_dst();
        let e_dst = bbs_signature_e_dst();
        let mut signatures_out = vec![0u8; item_count * SIGNATURE_LEN];
        let mut gpu_results = empty_gpu_results(item_count);
        let (message_offsets, message_lengths) = Self::compact_meta_ptrs(msg_meta);
        let item = gpu_abi::CuBbsBbsGpuSignFusedCommonBatchItem {
            common_points: point_slices.as_ptr(),
            common_term_count: point_slices.len(),
            item_count,
            secret_keys_be: secret_keys_be.as_ptr(),
            secret_key_stride,
            domains_be: domains_be.as_ptr(),
            domain_stride,
            packed_messages: if packed_messages.is_empty() {
                std::ptr::null()
            } else {
                packed_messages.as_ptr()
            },
            packed_messages_len: packed_messages.len(),
            message_offsets,
            message_lengths,
            message_dst: byte_slice(&message_dst),
            e_dst: byte_slice(&e_dst),
            signatures_out: signatures_out.as_mut_ptr(),
        };
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &item,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Err(Error::Gpu(gpu_status_message(status)));
        }
        if gpu_results.iter().any(|r| {
            r.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS || r.bytes_written != SIGNATURE_LEN
        }) {
            return Err(Error::Gpu(
                "GPU fused resident Sign returned per-item failure".to_owned(),
            ));
        }
        Ok(signatures_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_pre_b_aggregate_fused(
        &self,
        common_points: &[Vec<u8>],
        item_count: usize,
        signatures: &[u8],
        coefficients_le: &[u8],
        domains_be: &[u8],
        domain_stride: usize,
        item_message_counts: Option<&[u32]>,
        item_message_starts: Option<&[u64]>,
        packed_messages: &[u8],
        total_message_count: usize,
        msg_meta: CompactMsgMeta<'_>,
        public_key: &[u8],
        g2_generator: &[u8],
    ) -> Result<bool> {
        let call = self.verify_pre_b_aggregate_fused.ok_or_else(|| {
            Error::Gpu("GPU fused resident Verify symbol is unavailable".to_owned())
        })?;
        let message_dst = bbs_message_scalar_dst();
        let mut gpu_result = gpu_abi::CuBbsBbsGpuResult {
            code: gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL,
            value: 0,
            bytes_written: 0,
        };
        let (message_offsets, message_lengths) = Self::compact_meta_ptrs(msg_meta);
        let status = POINT_SLICES_SCRATCH.with(|cell| {
            let mut point_slices = cell.borrow_mut();
            point_slices.clear();
            point_slices.reserve(common_points.len());
            for p in common_points {
                point_slices.push(byte_slice(p));
            }
            let item = gpu_abi::CuBbsBbsGpuVerifyPreBAggregateFusedItem {
                common_points: point_slices.as_ptr(),
                common_term_count: point_slices.len(),
                item_count,
                signatures: signatures.as_ptr(),
                coefficients_le: coefficients_le.as_ptr(),
                domains_be: domains_be.as_ptr(),
                domain_stride,
                item_message_counts: item_message_counts
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                item_message_starts: item_message_starts
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                packed_messages: if packed_messages.is_empty() {
                    std::ptr::null()
                } else {
                    packed_messages.as_ptr()
                },
                packed_messages_len: packed_messages.len(),
                message_offsets,
                message_lengths,
                total_message_count,
                message_dst: byte_slice(&message_dst),
                public_key: byte_slice(public_key),
                g2_generator: byte_slice(g2_generator),
            };
            unsafe {
                call(
                    gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                    &item,
                    &mut gpu_result,
                )
            }
        });
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || gpu_result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS
        {
            return Err(Error::Gpu(gpu_status_message(if status != 0 {
                status
            } else {
                gpu_result.code
            })));
        }
        Ok(gpu_result.value != 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn proof_gen_fused_common_batch(
        &self,
        common_points: &[Vec<u8>],
        undisclosed_points: &[Vec<u8>],
        disclosed_indexes: &[u32],
        undisclosed_indexes: &[u32],
        item_count: usize,
        signatures: &[u8],
        domains_be: &[u8],
        domain_stride: usize,
        packed_messages: &[u8],
        msg_meta: CompactMsgMeta<'_>,
        packed_ph: &[u8],
        ph_offsets: &[u64],
        ph_lengths: &[u64],
        csprng_seed: &[u8; 32],
    ) -> Result<(Vec<u8>, usize)> {
        let call = self.proof_gen_fused_common_batch.ok_or_else(|| {
            Error::Gpu("GPU fused ProofGen symbol is unavailable".to_owned())
        })?;
        let point_slices = common_points
            .iter()
            .map(|p| byte_slice(p))
            .collect::<Vec<_>>();
        let undisc_point_slices = undisclosed_points
            .iter()
            .map(|p| byte_slice(p))
            .collect::<Vec<_>>();
        let proof_stride = 272usize + (undisclosed_indexes.len() * 32usize);
        let mut proofs_out = vec![0u8; item_count * proof_stride];
        let mut gpu_results = empty_gpu_results(item_count);
        let message_dst = bbs_message_scalar_dst();
        let challenge_dst = bbs_signature_e_dst();
        let (message_offsets, message_lengths) = Self::compact_meta_ptrs(msg_meta);
        let item = gpu_abi::CuBbsBbsGpuProofGenFusedCommonBatchItem {
            common_points: point_slices.as_ptr(),
            common_term_count: point_slices.len(),
            undisclosed_points: if undisc_point_slices.is_empty() {
                std::ptr::null()
            } else {
                undisc_point_slices.as_ptr()
            },
            disclosed_indexes: if disclosed_indexes.is_empty() {
                std::ptr::null()
            } else {
                disclosed_indexes.as_ptr()
            },
            disclosed_count: disclosed_indexes.len(),
            undisclosed_indexes: if undisclosed_indexes.is_empty() {
                std::ptr::null()
            } else {
                undisclosed_indexes.as_ptr()
            },
            undisclosed_count: undisclosed_indexes.len(),
            item_count,
            signatures: signatures.as_ptr(),
            domains_be: domains_be.as_ptr(),
            domain_stride,
            packed_messages: if packed_messages.is_empty() {
                std::ptr::null()
            } else {
                packed_messages.as_ptr()
            },
            packed_messages_len: packed_messages.len(),
            message_offsets,
            message_lengths,
            packed_ph: if packed_ph.is_empty() {
                std::ptr::null()
            } else {
                packed_ph.as_ptr()
            },
            packed_ph_len: packed_ph.len(),
            ph_offsets: ph_offsets.as_ptr(),
            ph_lengths: ph_lengths.as_ptr(),
            message_dst: byte_slice(&message_dst),
            challenge_dst: byte_slice(&challenge_dst),
            csprng_seed: csprng_seed.as_ptr(),
            proofs_out: proofs_out.as_mut_ptr(),
            proof_stride,
        };
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &item,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Err(Error::Gpu(gpu_status_message(status)));
        }
        if gpu_results.iter().any(|r| {
            r.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS || r.bytes_written != proof_stride
        }) {
            return Err(Error::Gpu("GPU fused ProofGen returned an item error".to_owned()));
        }
        Ok((proofs_out, proof_stride))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn proof_verify_fused_common_batch(
        &self,
        common_points: &[Vec<u8>],
        disclosed_indexes: &[u32],
        undisclosed_indexes: &[u32],
        item_count: usize,
        proofs: &[u8],
        proof_stride: usize,
        coefficients_le: &[u8],
        domains_be: &[u8],
        domain_stride: usize,
        packed_disclosed_messages: &[u8],
        msg_meta: CompactMsgMeta<'_>,
        packed_ph: &[u8],
        ph_offsets: &[u64],
        ph_lengths: &[u64],
        public_key: &[u8],
        g2_generator: &[u8],
    ) -> Result<bool> {
        let call = self.proof_verify_fused_common_batch.ok_or_else(|| {
            Error::Gpu("GPU fused ProofVerify symbol is unavailable".to_owned())
        })?;
        let point_slices = common_points
            .iter()
            .map(|p| byte_slice(p))
            .collect::<Vec<_>>();
        let mut gpu_result = gpu_abi::CuBbsBbsGpuResult {
            code: gpu_abi::CUBBS_BBS_GPU_SUCCESS,
            value: 0,
            bytes_written: 0,
        };
        let message_dst = bbs_message_scalar_dst();
        let challenge_dst = bbs_signature_e_dst();
        let (message_offsets, message_lengths) = Self::compact_meta_ptrs(msg_meta);
        let item = gpu_abi::CuBbsBbsGpuProofVerifyFusedCommonBatchItem {
            common_points: point_slices.as_ptr(),
            common_term_count: point_slices.len(),
            disclosed_indexes: if disclosed_indexes.is_empty() {
                std::ptr::null()
            } else {
                disclosed_indexes.as_ptr()
            },
            disclosed_count: disclosed_indexes.len(),
            undisclosed_indexes: if undisclosed_indexes.is_empty() {
                std::ptr::null()
            } else {
                undisclosed_indexes.as_ptr()
            },
            undisclosed_count: undisclosed_indexes.len(),
            item_count,
            proofs: proofs.as_ptr(),
            proof_stride,
            coefficients_le: coefficients_le.as_ptr(),
            domains_be: domains_be.as_ptr(),
            domain_stride,
            packed_disclosed_messages: if packed_disclosed_messages.is_empty() {
                std::ptr::null()
            } else {
                packed_disclosed_messages.as_ptr()
            },
            packed_disclosed_messages_len: packed_disclosed_messages.len(),
            message_offsets,
            message_lengths,
            packed_ph: if packed_ph.is_empty() {
                std::ptr::null()
            } else {
                packed_ph.as_ptr()
            },
            packed_ph_len: packed_ph.len(),
            ph_offsets: ph_offsets.as_ptr(),
            ph_lengths: ph_lengths.as_ptr(),
            message_dst: byte_slice(&message_dst),
            challenge_dst: byte_slice(&challenge_dst),
            public_key: byte_slice(public_key),
            g2_generator: byte_slice(g2_generator),
        };
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &item,
                &mut gpu_result,
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || gpu_result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS
        {
            return Err(Error::Gpu(gpu_status_message(if status != 0 {
                status
            } else {
                gpu_result.code
            })));
        }
        Ok(gpu_result.value != 0)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn messages_to_scalar_batch(
        &self,
        messages: Vec<Vec<u8>>,
        dst: Vec<u8>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(messages_to_scalar_batch) = self.messages_to_scalar_batch else {
            return batch_call_error(
                messages.len(),
                Error::Gpu("GPU messages_to_scalar symbol is unavailable".to_owned()),
                fail_fast,
            );
        };

        let attempted = messages.len();
        let mut scalars = (0..attempted).map(|_| vec![0u8; 32]).collect::<Vec<_>>();
        let mut items = messages
            .iter()
            .enumerate()
            .map(
                |(index, message)| gpu_abi::CuBbsBbsGpuMessagesToScalarItem {
                    message: byte_slice(message),
                    dst: byte_slice(&dst),
                    scalar_out: byte_buffer(&mut scalars[index]),
                },
            )
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            messages_to_scalar_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_generated_batch(attempted, fail_fast, gpu_results, scalars, |index| {
            items[index].scalar_out.len
        })
    }

    pub(super) fn sign_e_from_resident_message_scalars(
        &self,
        secret_keys: &[&[u8]],
        message_counts: &[usize],
        domain: &[u8],
    ) -> Result<Vec<BBSplusMessage>> {
        if std::env::var("CUBBS_BBS_GPU_SIGN_RESIDENT_E")
            .map(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
        {
            return Err(Error::Gpu(
                "GPU resident-scalar Sign-e path is disabled".to_owned(),
            ));
        }
        let call = self.sign_e_from_resident_scalars_batch.ok_or_else(|| {
            Error::Gpu("GPU resident-scalar Sign-e symbol is unavailable".to_owned())
        })?;
        if secret_keys.len() != message_counts.len() || domain.len() != 32 {
            return Err(Error::Gpu(
                "GPU resident-scalar Sign-e descriptor mismatch".to_owned(),
            ));
        }
        let item_count = secret_keys.len();
        let mut packed_scalars = vec![0u8; item_count * 32];
        let mut scalar_offset = 0usize;
        let mut items = secret_keys
            .iter()
            .zip(message_counts)
            .enumerate()
            .map(|(index, (secret_key, message_count))| {
                let item = gpu_abi::CuBbsBbsGpuSignEFromResidentScalarsItem {
                    secret_key: byte_slice(secret_key),
                    domain: byte_slice(domain),
                    messages: std::ptr::null(),
                    message_scalar_offset: scalar_offset,
                    message_count: *message_count,
                    message_scalars_out: gpu_abi::CuBbsBbsGpuByteBuffer {
                        data: std::ptr::null_mut(),
                        capacity: 0,
                        len: 0,
                    },
                    scalar_out: gpu_abi::CuBbsBbsGpuByteBuffer {
                        data: packed_scalars[index * 32..].as_mut_ptr(),
                        capacity: 32,
                        len: 0,
                    },
                };
                scalar_offset += *message_count;
                item
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(item_count);
        let message_dst = bbs_message_scalar_dst();
        let e_dst = bbs_signature_e_dst();
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                byte_slice(&message_dst),
                byte_slice(&e_dst),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Err(Error::Gpu(gpu_status_message(status)));
        }
        if gpu_results.iter().any(|result| {
            result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS || result.bytes_written != 32
        }) {
            return Err(Error::Gpu(
                "GPU resident-scalar Sign-e job failed".to_owned(),
            ));
        }
        packed_scalars
            .chunks_exact(32)
            .map(parse_message_scalar_bytes)
            .collect()
    }

    pub(super) fn sign_h2s_pipeline(
        &self,
        reqs: &[SignRequest],
        domain: &[u8],
    ) -> Result<(Vec<BBSplusMessage>, Vec<BBSplusMessage>)> {
        if std::env::var("CUBBS_BBS_GPU_SIGN_RESIDENT_E")
            .map(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
        {
            return Err(Error::Gpu("GPU Sign H2S pipeline is disabled".to_owned()));
        }
        let call = self
            .sign_e_from_resident_scalars_batch
            .ok_or_else(|| Error::Gpu("GPU Sign H2S pipeline symbol is unavailable".to_owned()))?;
        if domain.len() != 32 || reqs.is_empty() {
            return Err(Error::Gpu(
                "GPU Sign H2S pipeline descriptor mismatch".to_owned(),
            ));
        }
        let message_slices = reqs
            .iter()
            .map(|req| {
                req.messages
                    .iter()
                    .map(|message| byte_slice(message))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let total_message_count = reqs.iter().map(|req| req.messages.len()).sum::<usize>();
        let mut message_scalar_bytes = vec![0u8; total_message_count * 32];
        let mut e_scalar_bytes = vec![0u8; reqs.len() * 32];
        let mut scalar_offset = 0usize;
        let mut items = reqs
            .iter()
            .enumerate()
            .map(|(index, req)| {
                let message_count = req.messages.len();
                let item = gpu_abi::CuBbsBbsGpuSignEFromResidentScalarsItem {
                    secret_key: byte_slice(&req.secret_key),
                    domain: byte_slice(domain),
                    messages: message_slices[index].as_ptr(),
                    message_scalar_offset: scalar_offset,
                    message_count,
                    message_scalars_out: gpu_abi::CuBbsBbsGpuByteBuffer {
                        data: message_scalar_bytes[scalar_offset * 32..].as_mut_ptr(),
                        capacity: message_count * 32,
                        len: 0,
                    },
                    scalar_out: gpu_abi::CuBbsBbsGpuByteBuffer {
                        data: e_scalar_bytes[index * 32..].as_mut_ptr(),
                        capacity: 32,
                        len: 0,
                    },
                };
                scalar_offset += message_count;
                item
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(reqs.len());
        let message_dst = bbs_message_scalar_dst();
        let e_dst = bbs_signature_e_dst();
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                byte_slice(&message_dst),
                byte_slice(&e_dst),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Err(Error::Gpu(gpu_status_message(status)));
        }
        if gpu_results.iter().any(|result| {
            result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS || result.bytes_written != 32
        }) {
            return Err(Error::Gpu("GPU Sign H2S pipeline failed".to_owned()));
        }
        let message_scalars = message_scalar_bytes
            .chunks_exact(32)
            .map(parse_message_scalar_bytes)
            .collect::<Result<Vec<_>>>()?;
        let e_scalars = e_scalar_bytes
            .chunks_exact(32)
            .map(parse_message_scalar_bytes)
            .collect::<Result<Vec<_>>>()?;
        Ok((message_scalars, e_scalars))
    }

    pub(super) fn messages_to_scalar_packed_borrowed(
        &self,
        messages: &[&[u8]],
        dst: &[u8],
    ) -> Result<Vec<u8>> {
        let messages_to_scalar_batch = self
            .messages_to_scalar_batch
            .ok_or_else(|| Error::Gpu("GPU messages_to_scalar symbol is unavailable".to_owned()))?;
        let mut scalars = vec![0u8; messages.len() * 32];
        let mut items = messages
            .iter()
            .enumerate()
            .map(
                |(index, message)| gpu_abi::CuBbsBbsGpuMessagesToScalarItem {
                    message: byte_slice(message),
                    dst: byte_slice(dst),
                    scalar_out: gpu_abi::CuBbsBbsGpuByteBuffer {
                        data: scalars[index * 32..].as_mut_ptr(),
                        capacity: 32,
                        len: 0,
                    },
                },
            )
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(messages.len());
        let status = unsafe {
            messages_to_scalar_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Err(Error::Gpu(gpu_status_message(status)));
        }
        for (item, gpu_result) in items.iter().zip(gpu_results.iter()) {
            if gpu_result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS
                || gpu_result.bytes_written != 32
                || item.scalar_out.len != 32
            {
                return Err(gpu_item_error(*gpu_result));
            }
        }
        Ok(scalars)
    }

    pub(super) fn prepare_b_batch(
        &self,
        reqs: &[SignRequest],
        message_scalar_batches: &[Vec<BBSplusMessage>],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(prepare_b_batch) = self.prepare_b_batch else {
            return batch_call_error(
                reqs.len(),
                Error::Gpu("GPU prepare_b symbol is unavailable".to_owned()),
                fail_fast,
            );
        };

        let attempted = reqs.len();
        if message_scalar_batches.len() != attempted {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU prepare_b scalar batch count mismatch".to_owned()),
                fail_fast,
            );
        }

        let scalar_bytes = message_scalar_batches
            .iter()
            .map(|scalars| {
                scalars
                    .iter()
                    .map(|scalar| scalar.to_bytes_be().to_vec())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scalar_slices = scalar_bytes
            .iter()
            .map(|scalars| {
                scalars
                    .iter()
                    .map(|scalar| byte_slice(scalar))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut prepared_b = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut items = reqs
            .iter()
            .enumerate()
            .map(|(index, req)| gpu_abi::CuBbsBbsGpuPrepareBItem {
                public_key: byte_slice(&req.public_key),
                header: byte_slice(&req.header),
                message_scalars: scalar_slices[index].as_ptr(),
                message_count: scalar_slices[index].len(),
                b_out: byte_buffer(&mut prepared_b[index]),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            prepare_b_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_generated_batch(attempted, fail_fast, gpu_results, prepared_b, |index| {
            items[index].b_out.len
        })
    }

    pub(super) fn g1_msm_batch(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(g1_msm_batch) = self.g1_msm_batch else {
            return batch_call_error(
                terms.len(),
                Error::Gpu("GPU g1_msm symbol is unavailable".to_owned()),
                fail_fast,
            );
        };

        let attempted = terms.len();
        for term in terms {
            if term.compressed_points.len() != term.scalar_be.len() {
                return batch_call_error(
                    attempted,
                    Error::Gpu("GPU g1_msm point/scalar count mismatch".to_owned()),
                    fail_fast,
                );
            }
        }

        let point_slices = terms
            .iter()
            .map(|term| {
                term.compressed_points
                    .iter()
                    .map(|point| byte_slice(point))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scalar_slices = terms
            .iter()
            .map(|term| {
                term.scalar_be
                    .iter()
                    .map(|scalar| byte_slice(scalar))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut points = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut items = terms
            .iter()
            .enumerate()
            .map(|(index, _)| gpu_abi::CuBbsBbsGpuG1MsmItem {
                points: point_slices[index].as_ptr(),
                scalars: scalar_slices[index].as_ptr(),
                term_count: point_slices[index].len(),
                point_out: byte_buffer(&mut points[index]),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            g1_msm_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_generated_batch(attempted, fail_fast, gpu_results, points, |index| {
            items[index].point_out.len
        })
    }

    pub(super) fn g1_dual_msm_same_points_borrowed(
        &self,
        points: &[&[u8]],
        lhs_scalars: &[[u8; 32]],
        rhs_scalars: &[[u8; 32]],
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        let call = self.g1_dual_msm_same_points_batch?;
        if points.is_empty()
            || points.len() != lhs_scalars.len()
            || points.len() != rhs_scalars.len()
            || points.iter().any(|point| point.len() != 48)
        {
            return None;
        }
        let point_slices = points
            .iter()
            .map(|point| byte_slice(point))
            .collect::<Vec<_>>();
        let lhs_scalar_slices = lhs_scalars
            .iter()
            .map(|scalar| byte_slice(scalar))
            .collect::<Vec<_>>();
        let rhs_scalar_slices = rhs_scalars
            .iter()
            .map(|scalar| byte_slice(scalar))
            .collect::<Vec<_>>();
        let mut lhs = vec![0u8; 48];
        let mut rhs = vec![0u8; 48];
        let mut item = gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsItem {
            points: point_slices.as_ptr(),
            lhs_scalars: lhs_scalar_slices.as_ptr(),
            rhs_scalars: rhs_scalar_slices.as_ptr(),
            term_count: points.len(),
            lhs_out: byte_buffer(&mut lhs),
            rhs_out: byte_buffer(&mut rhs),
        };
        let mut result = empty_gpu_results(1);
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                result.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || result[0].code != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || item.lhs_out.len != 48
            || item.rhs_out.len != 48
        {
            return None;
        }
        Some((lhs, rhs))
    }

    pub(super) fn g1_dual_msm_same_points_lanes_borrowed(
        &self,
        points: &[&[u8]],
        lhs_scalars: &[[u8; 32]],
        rhs_scalars: &[[u8; 32]],
        lane_count: usize,
    ) -> Option<(Vec<Vec<u8>>, Vec<Vec<u8>>)> {
        let call = self.g1_dual_msm_same_points_lanes_batch?;
        if points.is_empty()
            || lane_count == 0
            || lane_count > points.len()
            || points.len() != lhs_scalars.len()
            || points.len() != rhs_scalars.len()
            || points.iter().any(|point| point.len() != 48)
        {
            return None;
        }
        let output_bytes = lane_count.checked_mul(48)?;
        let point_slices = points
            .iter()
            .map(|point| byte_slice(point))
            .collect::<Vec<_>>();
        let lhs_scalar_slices = lhs_scalars
            .iter()
            .map(|scalar| byte_slice(scalar))
            .collect::<Vec<_>>();
        let rhs_scalar_slices = rhs_scalars
            .iter()
            .map(|scalar| byte_slice(scalar))
            .collect::<Vec<_>>();
        let mut lhs = vec![0u8; output_bytes];
        let mut rhs = vec![0u8; output_bytes];
        let mut item = gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsLanesItem {
            points: point_slices.as_ptr(),
            lhs_scalars: lhs_scalar_slices.as_ptr(),
            rhs_scalars: rhs_scalar_slices.as_ptr(),
            term_count: points.len(),
            lane_count,
            lhs_outs: byte_buffer(&mut lhs),
            rhs_outs: byte_buffer(&mut rhs),
        };
        let mut result = empty_gpu_results(1);
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                result.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || result[0].code != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || item.lhs_outs.len != output_bytes
            || item.rhs_outs.len != output_bytes
        {
            return None;
        }
        Some((
            lhs.chunks_exact(48).map(<[u8]>::to_vec).collect(),
            rhs.chunks_exact(48).map(<[u8]>::to_vec).collect(),
        ))
    }

    pub(super) fn g1_msm_common_points_batch(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(g1_msm_common_points_batch) = self.g1_msm_common_points_batch else {
            return self.g1_msm_batch(terms, fail_fast);
        };
        if terms.is_empty() {
            return batch_run_result(0, Vec::new());
        }
        if !all_msm_terms_have_common_points(terms) {
            return self.g1_msm_batch(terms, fail_fast);
        }

        let attempted = terms.len();
        let first = &terms[0];
        if first.compressed_points.len() != first.scalar_be.len()
            || terms
                .iter()
                .any(|term| term.scalar_be.len() != first.compressed_points.len())
            || terms
                .iter()
                .flat_map(|term| term.scalar_be.iter())
                .any(|scalar| scalar.len() != 32)
        {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU common-points MSM point/scalar count mismatch".to_owned()),
                fail_fast,
            );
        }

        let point_slices = first
            .compressed_points
            .iter()
            .map(|point| byte_slice(point))
            .collect::<Vec<_>>();
        let scalar_row_stride = first.compressed_points.len() * 32;
        let mut packed_scalar_rows = Vec::with_capacity(attempted * scalar_row_stride);
        for term in terms {
            for scalar in &term.scalar_be {
                packed_scalar_rows.extend_from_slice(scalar);
            }
        }
        let mut points = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut point_outs = points
            .iter_mut()
            .map(|point| byte_buffer(point))
            .collect::<Vec<_>>();
        let mut item = gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchItem {
            points: point_slices.as_ptr(),
            term_count: point_slices.len(),
            scalar_rows: std::ptr::null(),
            scalar_rows_be: packed_scalar_rows.as_ptr(),
            scalar_row_stride,
            output_count: attempted,
            point_outs: point_outs.as_mut_ptr(),
        };
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            g1_msm_common_points_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        let point_lens = point_outs.iter().map(|out| out.len).collect::<Vec<_>>();
        drop(point_outs);
        collect_generated_batch(attempted, fail_fast, gpu_results, points, |index| {
            point_lens[index]
        })
    }

    pub(super) fn g1_msm_common_points_packed(
        &self,
        compressed_points: &[Vec<u8>],
        scalar_rows_be: &[u8],
        output_count: usize,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(g1_msm_common_points_batch) = self.g1_msm_common_points_batch else {
            return batch_call_error(
                output_count,
                Error::Gpu("GPU packed common-points MSM symbol is unavailable".to_owned()),
                fail_fast,
            );
        };
        let base_scalar_row_stride = compressed_points.len() * 32;
        let scalar_row_stride = if output_count == 0 {
            base_scalar_row_stride
        } else if scalar_rows_be.len() % output_count == 0 {
            scalar_rows_be.len() / output_count
        } else {
            0
        };
        if compressed_points.iter().any(|point| point.len() != 48)
            || scalar_rows_be.len() != output_count * scalar_row_stride
            || !matches!(
                scalar_row_stride,
                stride if stride == base_scalar_row_stride
                    || stride == base_scalar_row_stride + 32
            )
        {
            return batch_call_error(
                output_count,
                Error::Gpu("GPU packed common-points MSM shape mismatch".to_owned()),
                fail_fast,
            );
        }
        let point_slices = compressed_points
            .iter()
            .map(|point| byte_slice(point))
            .collect::<Vec<_>>();
        let mut points = (0..output_count).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut point_outs = points
            .iter_mut()
            .map(|point| byte_buffer(point))
            .collect::<Vec<_>>();
        let mut item = gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchItem {
            points: point_slices.as_ptr(),
            term_count: point_slices.len(),
            scalar_rows: std::ptr::null(),
            scalar_rows_be: scalar_rows_be.as_ptr(),
            scalar_row_stride,
            output_count,
            point_outs: point_outs.as_mut_ptr(),
        };
        let mut gpu_results = empty_gpu_results(output_count);
        let status = unsafe {
            g1_msm_common_points_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(
                output_count,
                Error::Gpu(gpu_status_message(status)),
                fail_fast,
            );
        }
        let point_lens = point_outs.iter().map(|out| out.len).collect::<Vec<_>>();
        drop(point_outs);
        collect_generated_batch(output_count, fail_fast, gpu_results, points, |index| {
            point_lens[index]
        })
    }

    pub(super) fn g1_msm_common_point_groups_batch(
        &self,
        groups: &[Vec<BbsMsmTerms>],
        fail_fast: bool,
    ) -> Vec<BbsBatchRunResult<Vec<u8>>> {
        let Some(g1_msm_common_points_batch) = self.g1_msm_common_points_batch else {
            return groups
                .iter()
                .map(|group| self.g1_msm_batch(group, fail_fast))
                .collect();
        };
        if groups.is_empty() {
            return Vec::new();
        }
        if groups
            .iter()
            .any(|group| group.is_empty() || !all_msm_terms_have_common_points(group))
        {
            return groups
                .iter()
                .map(|group| self.g1_msm_batch(group, fail_fast))
                .collect();
        }

        let point_slices = groups
            .iter()
            .map(|group| {
                group[0]
                    .compressed_points
                    .iter()
                    .map(|point| byte_slice(point))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let packed_scalar_rows = groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let term_count = point_slices[group_index].len();
                let mut packed = Vec::with_capacity(group.len() * term_count * 32);
                for term in group {
                    for scalar in &term.scalar_be {
                        packed.extend_from_slice(scalar);
                    }
                }
                packed
            })
            .collect::<Vec<_>>();
        if groups.iter().enumerate().any(|(group_index, group)| {
            let term_count = point_slices[group_index].len();
            group.iter().any(|term| {
                term.scalar_be.len() != term_count
                    || term.scalar_be.iter().any(|scalar| scalar.len() != 32)
            })
        }) {
            return groups
                .iter()
                .map(|group| {
                    batch_call_error(
                        group.len(),
                        Error::Gpu("GPU common-points MSM group scalar count mismatch".to_owned()),
                        fail_fast,
                    )
                })
                .collect();
        }

        let mut outputs = groups
            .iter()
            .map(|group| (0..group.len()).map(|_| vec![0u8; 48]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut output_buffers = outputs
            .iter_mut()
            .map(|group_outputs| {
                group_outputs
                    .iter_mut()
                    .map(|point| byte_buffer(point))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut items = groups
            .iter()
            .enumerate()
            .map(
                |(group_index, group)| gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchItem {
                    points: point_slices[group_index].as_ptr(),
                    term_count: point_slices[group_index].len(),
                    scalar_rows: std::ptr::null(),
                    scalar_rows_be: packed_scalar_rows[group_index].as_ptr(),
                    scalar_row_stride: point_slices[group_index].len() * 32,
                    output_count: group.len(),
                    point_outs: output_buffers[group_index].as_mut_ptr(),
                },
            )
            .collect::<Vec<_>>();
        let total_outputs = groups.iter().map(Vec::len).sum::<usize>();
        let mut gpu_results = empty_gpu_results(total_outputs);
        let status = unsafe {
            g1_msm_common_points_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return groups
                .iter()
                .map(|group| {
                    batch_call_error(
                        group.len(),
                        Error::Gpu(gpu_status_message(status)),
                        fail_fast,
                    )
                })
                .collect();
        }

        let output_lens = output_buffers
            .iter()
            .map(|group| group.iter().map(|buffer| buffer.len).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        drop(output_buffers);

        let mut cursor = 0usize;
        groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let group_len = group.len();
                let group_results = gpu_results[cursor..cursor + group_len].to_vec();
                cursor += group_len;
                collect_generated_batch(
                    group_len,
                    fail_fast,
                    group_results,
                    outputs[group_index].clone(),
                    |index| output_lens[group_index][index],
                )
            })
            .collect()
    }

    pub(super) fn g1_msm_batch_routes(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<u8> {
        let Some(g1_msm_batch) = self.g1_msm_batch else {
            return batch_call_error(
                terms.len(),
                Error::Gpu("GPU g1_msm symbol is unavailable".to_owned()),
                fail_fast,
            );
        };

        let attempted = terms.len();
        for term in terms {
            if term.compressed_points.len() != term.scalar_be.len() {
                return batch_call_error(
                    attempted,
                    Error::Gpu("GPU g1_msm point/scalar count mismatch".to_owned()),
                    fail_fast,
                );
            }
        }

        let point_slices = terms
            .iter()
            .map(|term| {
                term.compressed_points
                    .iter()
                    .map(|point| byte_slice(point))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scalar_slices = terms
            .iter()
            .map(|term| {
                term.scalar_be
                    .iter()
                    .map(|scalar| byte_slice(scalar))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut points = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut items = terms
            .iter()
            .enumerate()
            .map(|(index, _)| gpu_abi::CuBbsBbsGpuG1MsmItem {
                points: point_slices[index].as_ptr(),
                scalars: scalar_slices[index].as_ptr(),
                term_count: point_slices[index].len(),
                point_out: byte_buffer(&mut points[index]),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            g1_msm_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        let mut results = Vec::with_capacity(attempted);
        for (index, gpu_result) in gpu_results.into_iter().enumerate() {
            if gpu_result.code == gpu_abi::CUBBS_BBS_GPU_SUCCESS {
                results.push(BbsBatchItemResult {
                    index,
                    result: Ok(gpu_result.value),
                });
            } else {
                results.push(BbsBatchItemResult {
                    index,
                    result: Err(gpu_item_error(gpu_result)),
                });
                if fail_fast {
                    break;
                }
            }
        }
        batch_run_result(attempted, results)
    }

    pub(super) fn g1_msm_common_points_batch_routes(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<u8> {
        let Some(g1_msm_common_points_batch) = self.g1_msm_common_points_batch else {
            return self.g1_msm_batch_routes(terms, fail_fast);
        };
        if terms.is_empty() {
            return batch_run_result(0, Vec::new());
        }
        if !all_msm_terms_have_common_points(terms) {
            return self.g1_msm_batch_routes(terms, fail_fast);
        }

        let attempted = terms.len();
        let first = &terms[0];
        if first.compressed_points.len() != first.scalar_be.len()
            || terms
                .iter()
                .any(|term| term.scalar_be.len() != first.compressed_points.len())
            || terms
                .iter()
                .flat_map(|term| term.scalar_be.iter())
                .any(|scalar| scalar.len() != 32)
        {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU common-points MSM route point/scalar count mismatch".to_owned()),
                fail_fast,
            );
        }

        let point_slices = first
            .compressed_points
            .iter()
            .map(|point| byte_slice(point))
            .collect::<Vec<_>>();
        let scalar_row_stride = first.compressed_points.len() * 32;
        let mut packed_scalar_rows = Vec::with_capacity(attempted * scalar_row_stride);
        for term in terms {
            for scalar in &term.scalar_be {
                packed_scalar_rows.extend_from_slice(scalar);
            }
        }
        let mut points = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut point_outs = points
            .iter_mut()
            .map(|point| byte_buffer(point))
            .collect::<Vec<_>>();
        let mut item = gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchItem {
            points: point_slices.as_ptr(),
            term_count: point_slices.len(),
            scalar_rows: std::ptr::null(),
            scalar_rows_be: packed_scalar_rows.as_ptr(),
            scalar_row_stride,
            output_count: attempted,
            point_outs: point_outs.as_mut_ptr(),
        };
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            g1_msm_common_points_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        let mut results = Vec::with_capacity(attempted);
        for (index, gpu_result) in gpu_results.into_iter().enumerate() {
            if gpu_result.code == gpu_abi::CUBBS_BBS_GPU_SUCCESS {
                results.push(BbsBatchItemResult {
                    index,
                    result: Ok(gpu_result.value),
                });
            } else {
                results.push(BbsBatchItemResult {
                    index,
                    result: Err(gpu_item_error(gpu_result)),
                });
                if fail_fast {
                    break;
                }
            }
        }
        batch_run_result(attempted, results)
    }

    pub(super) fn g1_add_batch(
        &self,
        pairs: &[(Vec<u8>, Vec<u8>)],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(g1_add_batch) = self.g1_add_batch else {
            return batch_call_error(
                pairs.len(),
                Error::Gpu("GPU g1_add symbol is unavailable".to_owned()),
                fail_fast,
            );
        };

        let attempted = pairs.len();
        let mut points = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut items = pairs
            .iter()
            .enumerate()
            .map(|(index, (lhs, rhs))| gpu_abi::CuBbsBbsGpuG1AddItem {
                lhs: byte_slice(lhs),
                rhs: byte_slice(rhs),
                point_out: byte_buffer(&mut points[index]),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            g1_add_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_generated_batch(attempted, fail_fast, gpu_results, points, |index| {
            items[index].point_out.len
        })
    }

    pub(super) fn pairing_eq_batch(
        &self,
        items: &[BbsPairingEqInput],
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        let Some(pairing_eq_batch) = self.pairing_eq_batch else {
            return batch_call_error(
                items.len(),
                Error::Gpu("GPU pairing_eq symbol is unavailable".to_owned()),
                fail_fast,
            );
        };

        let attempted = items.len();
        let gpu_items = items
            .iter()
            .map(|item| gpu_abi::CuBbsBbsGpuPairingEqItem {
                lhs_g1: byte_slice(&item.lhs_g1),
                lhs_g2: byte_slice(&item.lhs_g2),
                rhs_g1: byte_slice(&item.rhs_g1),
                rhs_g2: byte_slice(&item.rhs_g2),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            pairing_eq_batch(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                gpu_items.as_ptr(),
                gpu_items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_boolean_batch(attempted, fail_fast, gpu_results)
    }

    pub(super) fn pairing_eq_aggregate(
        &self,
        items: &[BbsPairingEqInput],
        fail_fast: bool,
    ) -> Option<BbsBatchRunResult<bool>> {
        let pairing_eq_aggregate = self.pairing_eq_aggregate?;
        let coefficients = pairing_aggregate_coefficients_for_items(items).ok()?;
        let gpu_items = items
            .iter()
            .zip(coefficients.iter())
            .map(
                |(item, coefficient)| gpu_abi::CuBbsBbsGpuPairingEqScaledItem {
                    lhs_g1: byte_slice(&item.lhs_g1),
                    lhs_g2: byte_slice(&item.lhs_g2),
                    rhs_g1: byte_slice(&item.rhs_g1),
                    rhs_g2: byte_slice(&item.rhs_g2),
                    coefficient: byte_slice(coefficient),
                },
            )
            .collect::<Vec<_>>();
        let mut gpu_result = gpu_abi::CuBbsBbsGpuResult {
            code: gpu_abi::CUBBS_BBS_GPU_SUCCESS,
            value: 0,
            bytes_written: 0,
        };
        let status = unsafe {
            pairing_eq_aggregate(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                gpu_items.as_ptr(),
                gpu_items.len(),
                &mut gpu_result,
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS
            || gpu_result.code != gpu_abi::CUBBS_BBS_GPU_SUCCESS
        {
            return None;
        }
        if gpu_result.value != 1 {
            return None;
        }

        let results = (0..items.len())
            .map(|index| BbsBatchItemResult {
                index,
                result: Ok(true),
            })
            .collect::<Vec<_>>();
        let _ = fail_fast;
        Some(batch_run_result(items.len(), results))
    }
}
