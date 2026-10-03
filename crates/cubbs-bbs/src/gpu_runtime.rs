//! Native library lifetime, ABI calls, and prepared MSM buffers.

use super::*;

#[derive(Clone)]
pub(super) struct GpuKernelLibrary {
    pub(super) _library: Arc<Library>,
    pub(super) sign_batch: gpu_abi::CuBbsBbsGpuSignBatchV2,
    pub(super) verify_batch: gpu_abi::CuBbsBbsGpuVerifyBatchV2,
    pub(super) proof_gen_batch: gpu_abi::CuBbsBbsGpuProofGenBatchV2,
    pub(super) proof_verify_batch: gpu_abi::CuBbsBbsGpuProofVerifyBatchV2,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) messages_to_scalar_batch: Option<gpu_abi::CuBbsBbsGpuMessagesToScalarBatchV2>,
    pub(super) sign_e_from_resident_scalars_batch:
        Option<gpu_abi::CuBbsBbsGpuSignEFromResidentScalarsBatchV2>,
    pub(super) prepare_b_batch: Option<gpu_abi::CuBbsBbsGpuPrepareBBatchV2>,
    pub(super) g1_msm_batch: Option<gpu_abi::CuBbsBbsGpuG1MsmBatchV2>,
    pub(super) g1_msm_common_points_batch: Option<gpu_abi::CuBbsBbsGpuG1MsmCommonPointsBatchV2>,
    pub(super) g1_msm_common_points_plus_variable_batch:
        Option<gpu_abi::CuBbsBbsGpuG1MsmCommonPointsPlusVariableBatchV2>,
    pub(super) proof_verify_msm_pair_batch: Option<gpu_abi::CuBbsBbsGpuProofVerifyMsmPairBatchV2>,
    pub(super) g1_dual_msm_same_points_batch:
        Option<gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsBatchV2>,
    pub(super) g1_dual_msm_same_points_lanes_batch:
        Option<gpu_abi::CuBbsBbsGpuG1DualMsmSamePointsLanesBatchV2>,
    pub(super) g1_add_batch: Option<gpu_abi::CuBbsBbsGpuG1AddBatchV2>,
    pub(super) pairing_eq_batch: Option<gpu_abi::CuBbsBbsGpuPairingEqBatchV2>,
    pub(super) pairing_eq_aggregate: Option<gpu_abi::CuBbsBbsGpuPairingEqAggregateV2>,
    pub(super) proof_gen_t2_prepare_batch: Option<gpu_abi::CuBbsBbsGpuProofGenT2PrepareBatchV2>,
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
        let sign_batch = unsafe {
            *library
                .get::<gpu_abi::CuBbsBbsGpuSignBatchV2>(gpu_abi::CUBBS_BBS_GPU_SIGN_BATCH_SYMBOL)
                .map_err(|error| Error::Gpu(format!("missing GPU sign symbol: {error}")))?
        };
        let verify_batch = unsafe {
            *library
                .get::<gpu_abi::CuBbsBbsGpuVerifyBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_VERIFY_BATCH_SYMBOL,
                )
                .map_err(|error| Error::Gpu(format!("missing GPU verify symbol: {error}")))?
        };
        let proof_gen_batch = unsafe {
            *library
                .get::<gpu_abi::CuBbsBbsGpuProofGenBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PROOF_GEN_BATCH_SYMBOL,
                )
                .map_err(|error| Error::Gpu(format!("missing GPU proof-gen symbol: {error}")))?
        };
        let proof_verify_batch = unsafe {
            *library
                .get::<gpu_abi::CuBbsBbsGpuProofVerifyBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PROOF_VERIFY_BATCH_SYMBOL,
                )
                .map_err(|error| Error::Gpu(format!("missing GPU proof-verify symbol: {error}")))?
        };
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
        let g1_msm_common_points_plus_variable_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuG1MsmCommonPointsPlusVariableBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_G1_MSM_COMMON_POINTS_PLUS_VARIABLE_BATCH_SYMBOL,
                )
                .ok()
                .map(|symbol| *symbol)
        };
        let proof_verify_msm_pair_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuProofVerifyMsmPairBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PROOF_VERIFY_MSM_PAIR_BATCH_SYMBOL,
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
        let proof_gen_t2_prepare_batch = unsafe {
            library
                .get::<gpu_abi::CuBbsBbsGpuProofGenT2PrepareBatchV2>(
                    gpu_abi::CUBBS_BBS_GPU_PROOF_GEN_T2_PREPARE_BATCH_SYMBOL,
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
            sign_batch,
            verify_batch,
            proof_gen_batch,
            proof_verify_batch,
            messages_to_scalar_batch,
            sign_e_from_resident_scalars_batch,
            prepare_b_batch,
            g1_msm_batch,
            g1_msm_common_points_batch,
            g1_msm_common_points_plus_variable_batch,
            proof_verify_msm_pair_batch,
            g1_dual_msm_same_points_batch,
            g1_dual_msm_same_points_lanes_batch,
            g1_add_batch,
            pairing_eq_batch,
            pairing_eq_aggregate,
            proof_gen_t2_prepare_batch,
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
                    || self.g1_msm_common_points_plus_variable_batch.is_some()
                    || self.proof_verify_msm_pair_batch.is_some()
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

    pub(super) fn can_g1_msm_common_points_plus_variable(&self) -> bool {
        self.g1_msm_common_points_plus_variable_batch.is_some()
    }

    pub(super) fn can_proof_verify_msm_pair(&self) -> bool {
        self.proof_verify_msm_pair_batch.is_some()
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

    pub(super) fn can_proof_gen_t2_prepare(&self) -> bool {
        self.proof_gen_t2_prepare_batch.is_some()
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

    pub(super) fn g1_msm_common_points_plus_variable_batch(
        &self,
        terms: &[BbsMsmTerms],
        variable_index: usize,
        fail_fast: bool,
    ) -> Option<BbsBatchRunResult<Vec<u8>>> {
        let call = self.g1_msm_common_points_plus_variable_batch?;
        let first = terms.first()?;
        let term_count = first.compressed_points.len();
        if terms.len() < 2
            || variable_index >= term_count
            || term_count <= 1
            || term_count < gpu_split_variable_msm_min_terms()
            || first.scalar_be.len() != term_count
            || terms.iter().any(|term| {
                term.compressed_points.len() != term_count
                    || term.scalar_be.len() != term_count
                    || term
                        .compressed_points
                        .iter()
                        .enumerate()
                        .any(|(index, point)| {
                            index != variable_index && *point != first.compressed_points[index]
                        })
            })
        {
            return None;
        }
        let attempted = terms.len();
        let point_slices = first
            .compressed_points
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != variable_index)
            .map(|(_, point)| byte_slice(point))
            .collect::<Vec<_>>();
        let scalar_rows = terms
            .iter()
            .flat_map(|term| {
                term.scalar_be
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != variable_index)
                    .map(|(_, scalar)| byte_slice(scalar))
            })
            .collect::<Vec<_>>();
        let variable_points = terms
            .iter()
            .map(|term| byte_slice(&term.compressed_points[variable_index]))
            .collect::<Vec<_>>();
        let variable_scalars = terms
            .iter()
            .map(|term| byte_slice(&term.scalar_be[variable_index]))
            .collect::<Vec<_>>();
        let mut points = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut point_outs = points
            .iter_mut()
            .map(|point| byte_buffer(point))
            .collect::<Vec<_>>();
        let mut item = gpu_abi::CuBbsBbsGpuG1MsmCommonPointsPlusVariableItem {
            points: point_slices.as_ptr(),
            term_count: point_slices.len(),
            scalar_rows: scalar_rows.as_ptr(),
            variable_points: variable_points.as_ptr(),
            variable_scalars: variable_scalars.as_ptr(),
            output_count: attempted,
            point_outs: point_outs.as_mut_ptr(),
        };
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return Some(batch_call_error(
                attempted,
                Error::Gpu(gpu_status_message(status)),
                fail_fast,
            ));
        }
        let point_lens = point_outs.iter().map(|out| out.len).collect::<Vec<_>>();
        drop(point_outs);
        Some(collect_generated_batch(
            attempted,
            fail_fast,
            gpu_results,
            points,
            |index| point_lens[index],
        ))
    }

    pub(super) fn proof_verify_msm_pair_batch(
        &self,
        all_terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> Option<BbsBatchRunResult<Vec<u8>>> {
        let call = self.proof_verify_msm_pair_batch?;
        if all_terms.len() < 2 || !all_terms.len().is_multiple_of(2) {
            return None;
        }
        let t1_terms = all_terms.iter().step_by(2).collect::<Vec<_>>();
        let t2_terms = all_terms.iter().skip(1).step_by(2).collect::<Vec<_>>();
        if t1_terms
            .iter()
            .any(|term| term.compressed_points.len() != 3 || term.scalar_be.len() != 3)
        {
            return None;
        }
        let first_t2 = *t2_terms.first()?;
        let term_count = first_t2.compressed_points.len();
        if term_count <= 1
            || first_t2.scalar_be.len() != term_count
            || term_count < gpu_split_variable_msm_min_terms()
            || t2_terms.iter().any(|term| {
                term.compressed_points.len() != term_count || term.scalar_be.len() != term_count
            })
        {
            return None;
        }
        let variable_indexes = if t2_terms.len() == 1 {
            let d = &t1_terms[0].compressed_points[2];
            first_t2
                .compressed_points
                .iter()
                .enumerate()
                .filter_map(|(index, point)| (point == d).then_some(index))
                .collect::<Vec<_>>()
        } else {
            (0..term_count)
                .filter(|&point_index| {
                    let base = &first_t2.compressed_points[point_index];
                    t2_terms
                        .iter()
                        .skip(1)
                        .any(|term| term.compressed_points[point_index] != *base)
                })
                .collect::<Vec<_>>()
        };
        let [variable_index] = variable_indexes.as_slice() else {
            return None;
        };
        let variable_index = *variable_index;
        if t2_terms.iter().any(|term| {
            term.compressed_points
                .iter()
                .enumerate()
                .any(|(index, point)| {
                    index != variable_index && *point != first_t2.compressed_points[index]
                })
        }) || t1_terms
            .iter()
            .zip(t2_terms.iter())
            .any(|(t1, t2)| t1.compressed_points[2] != t2.compressed_points[variable_index])
        {
            return None;
        }

        let output_count = t1_terms.len();
        let common_points = first_t2
            .compressed_points
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != variable_index)
            .map(|(_, point)| byte_slice(point))
            .collect::<Vec<_>>();
        let common_scalar_rows = t2_terms
            .iter()
            .flat_map(|term| {
                term.scalar_be
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != variable_index)
                    .map(|(_, scalar)| byte_slice(scalar))
            })
            .collect::<Vec<_>>();
        let proof_points = t1_terms
            .iter()
            .flat_map(|term| term.compressed_points.iter().map(|point| byte_slice(point)))
            .collect::<Vec<_>>();
        let proof_scalars = t1_terms
            .iter()
            .zip(t2_terms.iter())
            .flat_map(|(t1, t2)| {
                t1.scalar_be
                    .iter()
                    .chain(std::iter::once(&t2.scalar_be[variable_index]))
                    .map(|scalar| byte_slice(scalar))
            })
            .collect::<Vec<_>>();
        let mut t1_outputs = (0..output_count).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut t2_outputs = (0..output_count).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut t1_buffers = t1_outputs
            .iter_mut()
            .map(|point| byte_buffer(point))
            .collect::<Vec<_>>();
        let mut t2_buffers = t2_outputs
            .iter_mut()
            .map(|point| byte_buffer(point))
            .collect::<Vec<_>>();
        let mut item = gpu_abi::CuBbsBbsGpuProofVerifyMsmPairItem {
            common_points: common_points.as_ptr(),
            common_term_count: common_points.len(),
            common_scalar_rows: common_scalar_rows.as_ptr(),
            proof_points: proof_points.as_ptr(),
            proof_scalars: proof_scalars.as_ptr(),
            output_count,
            t1_outs: t1_buffers.as_mut_ptr(),
            t2_outs: t2_buffers.as_mut_ptr(),
        };
        let mut gpu_results = empty_gpu_results(output_count);
        let status = unsafe {
            call(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                &mut item,
                1,
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return None;
        }
        drop(t1_buffers);
        drop(t2_buffers);

        let mut results = Vec::with_capacity(all_terms.len());
        for (row, gpu_result) in gpu_results.into_iter().enumerate() {
            let row_result = if gpu_result.code == gpu_abi::CUBBS_BBS_GPU_SUCCESS {
                Ok(())
            } else {
                Err(Error::Gpu(gpu_status_message(gpu_result.code)))
            };
            results.push(BbsBatchItemResult {
                index: row * 2,
                result: row_result.clone().map(|_| t1_outputs[row].clone()),
            });
            results.push(BbsBatchItemResult {
                index: row * 2 + 1,
                result: row_result.map(|_| t2_outputs[row].clone()),
            });
            if fail_fast && results.last().is_some_and(|item| item.result.is_err()) {
                break;
            }
        }
        Some(batch_run_result(all_terms.len(), results))
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

    /// GPU proof-gen t2Prepare: computes the 6 fixed G1 scalar multiplications
    /// (d, abar, bbar, t1) on the device. Caller precomputes the 7 scalar
    /// multipliers on the host. Returns 4 compressed G1 points per proof.
    pub(super) fn proof_gen_t2_prepare_batch(
        &self,
        inputs: &[BbsT2PrepareGpuInput<'_>],
        fail_fast: bool,
    ) -> BbsBatchRunResult<BbsT2PrepareGpuOutput> {
        let Some(proof_gen_t2_prepare_batch) = self.proof_gen_t2_prepare_batch else {
            return batch_call_error(
                inputs.len(),
                Error::Gpu("GPU proof_gen_t2_prepare symbol is unavailable".to_owned()),
                fail_fast,
            );
        };
        let attempted = inputs.len();
        let mut d_bufs = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut abar_bufs = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut bbar_bufs = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut t1_bufs = (0..attempted).map(|_| vec![0u8; 48]).collect::<Vec<_>>();
        let mut items = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let mut scalars_be = [gpu_abi::CuBbsBbsGpuByteSlice {
                    data: core::ptr::null(),
                    len: 0,
                }; 7];
                for (s, scalar) in input.scalars_be.iter().enumerate() {
                    scalars_be[s] = byte_slice(scalar);
                }
                gpu_abi::CuBbsBbsGpuProofGenT2PrepareItem {
                    compressed_b: byte_slice(input.compressed_b),
                    signature_a: byte_slice(input.signature_a),
                    scalars_be,
                    seed: byte_slice(input.seed),
                    d_out: byte_buffer(&mut d_bufs[index]),
                    abar_out: byte_buffer(&mut abar_bufs[index]),
                    bbar_out: byte_buffer(&mut bbar_bufs[index]),
                    t1_out: byte_buffer(&mut t1_bufs[index]),
                }
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            proof_gen_t2_prepare_batch(
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
        for index in 0..attempted {
            let r = &gpu_results[index];
            if r.code == gpu_abi::CUBBS_BBS_GPU_SUCCESS && r.bytes_written == 48 {
                results.push(BbsBatchItemResult {
                    index,
                    result: Ok(BbsT2PrepareGpuOutput {
                        d: std::mem::take(&mut d_bufs[index]),
                        abar: std::mem::take(&mut abar_bufs[index]),
                        bbar: std::mem::take(&mut bbar_bufs[index]),
                        t1: std::mem::take(&mut t1_bufs[index]),
                    }),
                });
            } else {
                results.push(BbsBatchItemResult {
                    index,
                    result: Err(gpu_item_error(*r)),
                });
                if fail_fast {
                    break;
                }
            }
        }
        batch_run_result(attempted, results)
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

    pub(super) fn sign_batch(
        &self,
        reqs: Vec<SignRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let attempted = reqs.len();
        let message_slices = reqs
            .iter()
            .map(|req| {
                req.messages
                    .iter()
                    .map(|msg| byte_slice(msg))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut signatures = (0..attempted)
            .map(|_| vec![0u8; SIGNATURE_LEN])
            .collect::<Vec<_>>();
        let mut items = reqs
            .iter()
            .enumerate()
            .map(|(index, req)| gpu_abi::CuBbsBbsGpuSignItem {
                secret_key: byte_slice(&req.secret_key),
                public_key: byte_slice(&req.public_key),
                header: byte_slice(&req.header),
                messages: message_slices[index].as_ptr(),
                message_count: message_slices[index].len(),
                signature_out: byte_buffer(&mut signatures[index]),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            (self.sign_batch)(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_generated_batch(attempted, fail_fast, gpu_results, signatures, |index| {
            items[index].signature_out.len
        })
    }

    pub(super) fn verify_batch(
        &self,
        reqs: Vec<VerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        let attempted = reqs.len();
        let message_slices = reqs
            .iter()
            .map(|req| {
                req.messages
                    .iter()
                    .map(|msg| byte_slice(msg))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let items = reqs
            .iter()
            .enumerate()
            .map(|(index, req)| gpu_abi::CuBbsBbsGpuVerifyItem {
                public_key: byte_slice(&req.public_key),
                signature: byte_slice(&req.signature),
                header: byte_slice(&req.header),
                messages: message_slices[index].as_ptr(),
                message_count: message_slices[index].len(),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            (self.verify_batch)(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_boolean_batch(attempted, fail_fast, gpu_results)
    }

    pub(super) fn proof_gen_batch(
        &self,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let attempted = reqs.len();
        let message_slices = reqs
            .iter()
            .map(|req| {
                req.messages
                    .iter()
                    .map(|msg| byte_slice(msg))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut proofs = reqs
            .iter()
            .map(|req| vec![0u8; proof_capacity(req.messages.len(), req.disclosed_indexes.len())])
            .collect::<Vec<_>>();
        let mut items = reqs
            .iter()
            .enumerate()
            .map(|(index, req)| gpu_abi::CuBbsBbsGpuProofGenItem {
                public_key: byte_slice(&req.public_key),
                signature: byte_slice(&req.signature),
                header: byte_slice(&req.header),
                presentation_header: byte_slice(&req.presentation_header),
                messages: message_slices[index].as_ptr(),
                message_count: message_slices[index].len(),
                disclosed_indexes: index_slice(&req.disclosed_indexes),
                proof_out: byte_buffer(&mut proofs[index]),
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            (self.proof_gen_batch)(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_mut_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_generated_batch(attempted, fail_fast, gpu_results, proofs, |index| {
            items[index].proof_out.len
        })
    }

    pub(super) fn proof_verify_batch(
        &self,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        let attempted = reqs.len();
        let disclosed_message_slices = reqs
            .iter()
            .map(|req| {
                req.disclosed_messages
                    .iter()
                    .map(|msg| byte_slice(msg))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let items = reqs
            .iter()
            .enumerate()
            .map(|(index, req)| gpu_abi::CuBbsBbsGpuProofVerifyItem {
                public_key: byte_slice(&req.public_key),
                proof: byte_slice(&req.proof),
                header: byte_slice(&req.header),
                presentation_header: byte_slice(&req.presentation_header),
                disclosed_messages: disclosed_message_slices[index].as_ptr(),
                disclosed_count: disclosed_message_slices[index].len(),
                disclosed_indexes: index_slice(&req.disclosed_indexes),
                total_message_count: req.total_message_count,
            })
            .collect::<Vec<_>>();
        let mut gpu_results = empty_gpu_results(attempted);
        let status = unsafe {
            (self.proof_verify_batch)(
                gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                items.as_ptr(),
                items.len(),
                gpu_results.as_mut_ptr(),
            )
        };
        if status != gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            return batch_call_error(attempted, Error::Gpu(gpu_status_message(status)), fail_fast);
        }

        collect_boolean_batch(attempted, fail_fast, gpu_results)
    }
}
