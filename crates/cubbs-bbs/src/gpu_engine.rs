//! GPU operation orchestration and fallback policy.

use super::*;

// FIFO is sufficient here: hits need no bookkeeping and retained templates
// cannot grow without bound as new issuer/header/shape combinations arrive.
#[derive(Debug)]
pub(super) struct MsmTemplateCache {
    entries: HashMap<BbsMsmTermTemplateKey, Arc<BbsMsmTerms>>,
    order: VecDeque<(BbsMsmTermTemplateKey, usize)>,
    retained_bytes: usize,
    budget: usize,
}

impl MsmTemplateCache {
    fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            retained_bytes: 0,
            budget,
        }
    }

    fn insert(&mut self, key: BbsMsmTermTemplateKey, value: Arc<BbsMsmTerms>) {
        if self.entries.contains_key(&key) {
            return;
        }
        // Account for owned payload and Vec headers; shared points are
        // conservatively charged per entry. This is not an RSS limit.
        let bytes = key.public_key.len()
            + key.header.len()
            + value
                .compressed_points
                .iter()
                .chain(&value.scalar_be)
                .map(|v| v.capacity() + std::mem::size_of::<Vec<u8>>())
                .sum::<usize>()
            + std::mem::size_of::<BbsMsmTerms>()
            + 2 * std::mem::size_of::<BbsMsmTermTemplateKey>();
        if bytes > self.budget {
            return;
        }
        while self.retained_bytes > self.budget - bytes {
            let Some((oldest, size)) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
            self.retained_bytes -= size;
        }
        self.retained_bytes += bytes;
        self.order.push_back((key.clone(), bytes));
        self.entries.insert(key, value);
    }
}

impl GpuBbsEngine {
    pub fn new() -> Self {
        Self {
            protocol: CuBbsProtocolEngine::new(),
            fallback: reference::ZkryptiumReference::from_env().unwrap_or_default(),
            kernel: GpuKernelLibrary::load_shared_from_env().ok().flatten(),
            b_msm_term_templates: Arc::new(Mutex::new(MsmTemplateCache::new(64 * 1024 * 1024))),
        }
    }

    pub(super) fn reference_fallback<T>(
        &self,
        operation: &'static str,
        run: impl FnOnce(&reference::ZkryptiumReference) -> Result<T>,
    ) -> Result<T> {
        if force_gpu_execution() {
            return Err(Error::HardwareGpuUnavailable(format!(
                "CUBBS_BBS_FORCE_GPU forbids CPU fallback for {operation}"
            )));
        }
        run(&self.fallback)
    }

    pub(super) fn prepare_b_msm_terms_for_prehashed_message_scalars_cached(
        &self,
        req: &SignRequest,
        scalars: &[BBSplusMessage],
    ) -> Result<BbsMsmTerms> {
        let key = BbsMsmTermTemplateKey {
            public_key: req.public_key.clone(),
            header: req.header.clone(),
            message_count: scalars.len(),
        };

        let template = self
            .b_msm_term_templates
            .lock()
            .unwrap()
            .entries
            .get(&key)
            .cloned();
        if let Some(template) = template {
            return bbs_msm_terms_from_template(&template, scalars);
        }

        let terms = self
            .protocol
            .prepare_b_msm_terms_for_prehashed_message_scalars(req, scalars)?;
        self.b_msm_term_templates
            .lock()
            .unwrap()
            .insert(key, Arc::new(terms.clone()));
        Ok(terms)
    }

    pub fn backend_info(&self) -> BbsBackendInfo {
        match &self.kernel {
            Some(kernel) if kernel.hardware_acceleration => BbsBackendInfo {
                requested: "gpu",
                active: "cuda-abi-v2",
                hardware_acceleration: true,
                fallback: false,
                gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                detail: "CUDA BBS kernel loaded through cuBBS-Cred GPU ABI v2",
            },
            Some(kernel)
                if kernel.backend_kind == gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CPU_ABI_TEST =>
            {
                BbsBackendInfo {
                    requested: "gpu",
                    active: "gpu-abi-v2-cpu-test",
                    hardware_acceleration: false,
                    fallback: false,
                    gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                    detail: "GPU ABI test library loaded; calls cross the shared-library boundary without hardware acceleration",
                }
            }
            Some(kernel)
                if kernel.backend_kind == gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CUDA
                    && kernel.can_precompute_message_scalars() =>
            {
                BbsBackendInfo {
                    requested: "gpu",
                    active: if force_gpu_execution() && kernel.can_g1_msm() {
                        "cuda-abi-v2-force-gpu"
                    } else if kernel.can_g1_msm() {
                        "cuda-abi-v2-auto-hybrid"
                    } else {
                        "cuda-abi-v2-hybrid-message-scalar"
                    },
                    hardware_acceleration: kernel.uses_hardware_acceleration(),
                    fallback: false,
                    gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                    detail: if kernel.can_g1_msm() && gpu_precompute_signature_e_enabled() {
                        "CUDA ABI library loaded; all GPU-selected G1 MSM shapes stay on cuBBS-Cred kernels: signed-comb8/9/10 and GLV tiled-window4 cover tuned common-point shapes, while bounded common-point window4 and variable-base GLV row kernels provide the universal fallback; SPPARK supplies field/curve primitives; its Pippenger kernel is linked only in an explicit route 7/8 reference benchmark build; proof T2 can fuse common generators with one variable point, proof-gen uses compressed GPU state and packed homogeneous B input, large pairing batches use cuBBS-Cred CUDA Miller loops and random-coefficient aggregate verification when G2 inputs are common, sign also precomputes e scalars, and experimental proof challenge scalar GPU precompute is disabled by default"
                    } else if kernel.can_g1_msm() {
                        "CUDA ABI library loaded; all GPU-selected G1 MSM shapes stay on cuBBS-Cred kernels: signed-comb8/9/10 and GLV tiled-window4 cover tuned common-point shapes, while bounded common-point window4 and variable-base GLV row kernels provide the universal fallback; SPPARK supplies field/curve primitives; its Pippenger kernel is linked only in an explicit route 7/8 reference benchmark build; proof T2 can fuse common generators with one variable point, proof-gen uses compressed GPU state and packed homogeneous B input, large pairing batches use cuBBS-Cred CUDA Miller loops and random-coefficient aggregate verification when G2 inputs are common, optional sign e scalar GPU precompute is disabled, and experimental proof challenge scalar GPU precompute is disabled by default"
                    } else if gpu_precompute_signature_e_enabled() {
                        "CUDA ABI library loaded; BBS sign/verify/proof paths precompute message scalars on GPU, sign also precomputes e scalars on GPU, while generator/MSM/pairing kernels are still CPU-backed"
                    } else {
                        "CUDA ABI library loaded; BBS sign/verify/proof paths precompute message scalars on GPU, optional sign e scalar GPU precompute is disabled, and generator/MSM/pairing kernels are still CPU-backed"
                    },
                }
            }
            Some(kernel) if kernel.backend_kind == gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CUDA => {
                BbsBackendInfo {
                    requested: "gpu",
                    active: "cuda-abi-v2-incomplete",
                    hardware_acceleration: false,
                    fallback: false,
                    gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                    detail: "CUDA ABI library loaded and device probe passed, but BBS kernels have not asserted hardware acceleration",
                }
            }
            Some(_) => BbsBackendInfo {
                requested: "gpu",
                active: "gpu-abi-v2",
                hardware_acceleration: false,
                fallback: false,
                gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                detail: "GPU ABI library loaded, but hardware acceleration capability was not asserted",
            },
            None => BbsBackendInfo {
                requested: "gpu",
                active: "zkryptium-cpu",
                hardware_acceleration: false,
                fallback: true,
                gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                detail: "GPU backend requested, but CUBBS_BBS_GPU_LIBRARY is not loaded; using CPU fallback",
            },
        }
    }

    pub fn generate_keypair(&self, rng: &mut impl RngCore) -> Result<KeyPair> {
        self.protocol.generate_keypair(rng)
    }

    pub fn public_key_from_secret(&self, secret_key: &[u8]) -> Result<Vec<u8>> {
        self.protocol.public_key_from_secret(secret_key)
    }

    #[cfg(test)]
    pub(super) fn messages_to_scalar_batch_for_test(
        &self,
        messages: Vec<Vec<u8>>,
        dst: Vec<u8>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match &self.kernel {
            Some(kernel) => kernel.messages_to_scalar_batch(messages, dst, fail_fast),
            None => batch_call_error(
                messages.len(),
                Error::Gpu("GPU kernel is not loaded".to_owned()),
                fail_fast,
            ),
        }
    }

    pub fn sign(&self, req: SignRequest) -> Result<Vec<u8>> {
        let Some(kernel) = &self.kernel else {
            return self.reference_fallback("Sign", |fallback| fallback.sign(&req));
        };
        if !kernel.can_precompute_message_scalars() {
            return self.reference_fallback("Sign", |fallback| fallback.sign(&req));
        }

        if force_gpu_execution() {
            let requests = vec![req];
            let plan = planner::CuBbsPlanner::plan_sign(&requests);
            if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                return Err(error);
            }
            let result = self.sign_batch_with_gpu_message_scalars(kernel, requests, true);
            return result
                .results
                .into_iter()
                .next()
                .ok_or_else(|| Error::Gpu("GPU Sign returned no result".to_owned()))?
                .result;
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.messages)?;
        if kernel.can_g1_msm() {
            let terms =
                self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(&req, &scalars)?;
            let gpu_prepared = kernel.g1_msm_batch(std::slice::from_ref(&terms), true);
            if gpu_prepared.failed == 0 {
                if let Some(prepared_b) = gpu_prepared.results.into_iter().next() {
                    let e_input = signature_e_input_from_msm_terms(&req, &scalars, &terms)?;
                    let e = if gpu_precompute_signature_e_enabled() {
                        let mut e_scalars =
                            gpu_precompute_signature_e_scalars(kernel, vec![e_input], true)?;
                        e_scalars.pop().ok_or_else(|| {
                            Error::Gpu("GPU signature e scalar is missing".to_owned())
                        })?
                    } else {
                        cpu_signature_e_scalar(&e_input)?
                    };
                    return prepared_b.result.and_then(|compressed_b| {
                        self.protocol.sign_prehashed_message_scalars_e_and_b(
                            req,
                            e,
                            BbsPreparedB { compressed_b },
                        )
                    });
                }
            }
        }

        if !gpu_precompute_signature_e_enabled() {
            return self.protocol.sign_prehashed_message_scalars(req, scalars);
        }

        let e_input = self
            .protocol
            .signature_e_input_for_prehashed_message_scalars(&req, &scalars)?;
        let mut e_scalars = gpu_precompute_signature_e_scalars(kernel, vec![e_input], true)?;
        let e = e_scalars
            .pop()
            .ok_or_else(|| Error::Gpu("GPU signature e scalar is missing".to_owned()))?;
        self.protocol
            .sign_prehashed_message_scalars_and_e(req, scalars, e)
    }

    pub fn profile_sign(&self, req: SignRequest) -> Result<BbsSignProfile> {
        if force_gpu_execution() {
            return Err(Error::HardwareGpuUnavailable(
                "CUBBS_BBS_FORCE_GPU forbids the CPU-only Sign profiling helper".to_owned(),
            ));
        }
        self.protocol.profile_sign(req)
    }

    pub fn prepare_b_for_sign(&self, req: &SignRequest) -> Result<BbsPreparedB> {
        let Some(kernel) = &self.kernel else {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU PrepareB".to_owned(),
                ));
            }
            return self.protocol.prepare_b_for_sign(req);
        };
        if !kernel.can_precompute_message_scalars() {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S for PrepareB".to_owned(),
                ));
            }
            return self.protocol.prepare_b_for_sign(req);
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.messages)?;
        if kernel.can_g1_msm() {
            let terms =
                self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(req, &scalars)?;
            let gpu_prepared = kernel.g1_msm_batch(&[terms], true);
            if gpu_prepared.failed == 0 {
                if let Some(item) = gpu_prepared.results.into_iter().next() {
                    return item
                        .result
                        .map(|compressed_b| BbsPreparedB { compressed_b });
                }
            }
        }
        if kernel.can_prepare_b() {
            let gpu_prepared =
                kernel.prepare_b_batch(std::slice::from_ref(req), &[scalars.clone()], true);
            if gpu_prepared.failed == 0 {
                if let Some(item) = gpu_prepared.results.into_iter().next() {
                    return item
                        .result
                        .map(|compressed_b| BbsPreparedB { compressed_b });
                }
            }
        }
        if force_gpu_execution() {
            return Err(Error::Gpu(
                "GPU PrepareB MSM failed; CPU cryptographic fallback is disabled by CUBBS_BBS_FORCE_GPU"
                    .to_owned(),
            ));
        }
        self.protocol
            .prepare_b_for_prehashed_message_scalars(req, &scalars)
    }

    pub fn prepare_b_msm_terms_for_sign(&self, req: &SignRequest) -> Result<BbsMsmTerms> {
        let Some(kernel) = &self.kernel else {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S for MSM term preparation".to_owned(),
                ));
            }
            return self.protocol.prepare_b_msm_terms_for_sign(req);
        };
        if !kernel.can_precompute_message_scalars() {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S for MSM term preparation".to_owned(),
                ));
            }
            return self.protocol.prepare_b_msm_terms_for_sign(req);
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.messages)?;
        self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(req, &scalars)
    }

    pub fn prepare_g1_msm_bench_plan(
        &self,
        groups: Vec<GpuMsmBenchGroupInput>,
    ) -> Result<GpuMsmBenchPlan> {
        let kernel = self.kernel.as_ref().ok_or_else(|| {
            Error::Gpu("GPU kernel is unavailable for prepared G1 MSM bench".to_owned())
        })?;
        if !kernel.uses_hardware_acceleration() {
            return Err(Error::HardwareGpuUnavailable(
                "prepared G1 MSM bench requires a hardware GPU backend".to_owned(),
            ));
        }
        GpuMsmBenchPlan::prepare(Arc::clone(kernel), groups)
    }

    pub fn g1_msm_batch_for_bench(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(kernel) = &self.kernel else {
            return batch_call_error(
                terms.len(),
                Error::Gpu("GPU kernel is unavailable for raw G1 MSM bench".to_owned()),
                fail_fast,
            );
        };
        if !kernel.can_g1_msm() {
            return batch_call_error(
                terms.len(),
                Error::Gpu("GPU G1 MSM symbol is unavailable for raw bench".to_owned()),
                fail_fast,
            );
        }
        g1_msm_batch_grouped_by_points(kernel, terms, fail_fast)
    }

    pub fn g1_msm_batch_split_for_bench(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(kernel) = &self.kernel else {
            return batch_call_error(
                terms.len(),
                Error::Gpu("GPU kernel is unavailable for split G1 MSM bench".to_owned()),
                fail_fast,
            );
        };
        if !kernel.can_g1_msm() {
            return batch_call_error(
                terms.len(),
                Error::Gpu("GPU G1 MSM symbol is unavailable for split bench".to_owned()),
                fail_fast,
            );
        }
        g1_msm_batch_split_single_variable_term_grouped(kernel, terms, fail_fast)
            .unwrap_or_else(|| g1_msm_batch_grouped_by_points(kernel, terms, fail_fast))
    }

    pub fn g1_msm_route_summary_for_bench(
        &self,
        terms: &[BbsMsmTerms],
        split: bool,
        fail_fast: bool,
    ) -> Result<BbsMsmRouteSummary> {
        let Some(kernel) = &self.kernel else {
            return Err(Error::Gpu(
                "GPU kernel is unavailable for G1 MSM route bench".to_owned(),
            ));
        };
        if !kernel.can_g1_msm() {
            return Err(Error::Gpu(
                "GPU G1 MSM symbol is unavailable for route bench".to_owned(),
            ));
        }
        if split {
            Ok(g1_msm_route_summary_split_for_bench(
                kernel, terms, fail_fast,
            ))
        } else {
            Ok(g1_msm_route_summary_grouped_by_points(
                kernel, terms, fail_fast,
            ))
        }
    }

    pub fn proof_msm_shapes_for_bench(
        &self,
        proof_gen_reqs: Vec<ProofGenRequest>,
        proof_verify_reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> Result<Vec<BbsMsmShapeSummary>> {
        self.proof_msm_terms_for_bench(proof_gen_reqs, proof_verify_reqs, fail_fast)
            .map(|batches| {
                batches
                    .iter()
                    .map(|batch| summarize_msm_terms(batch.label, &batch.terms))
                    .collect()
            })
    }

    pub fn proof_msm_terms_for_bench(
        &self,
        proof_gen_reqs: Vec<ProofGenRequest>,
        proof_verify_reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> Result<Vec<BbsMsmTermsBatch>> {
        let Some(kernel) = &self.kernel else {
            return Err(Error::Gpu(
                "GPU kernel is unavailable for proof MSM terms bench".to_owned(),
            ));
        };
        if !kernel.can_precompute_message_scalars() || !kernel.can_g1_msm() {
            return Err(Error::Gpu(
                "GPU scalar/MSM symbols are unavailable for proof MSM terms bench".to_owned(),
            ));
        }

        let attempted = proof_gen_reqs.len();
        let flat_messages = proof_gen_reqs
            .iter()
            .flat_map(|req| req.messages.iter().cloned())
            .collect::<Vec<_>>();
        let flat_scalars = gpu_precompute_flat_message_scalars(kernel, flat_messages, fail_fast)
            .map_err(|error| Error::Gpu(format!("proof-gen scalar precompute failed: {error}")))?;
        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in proof_gen_reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.messages.len();
            let scalars = flat_scalars
                .get(scalar_cursor..scalar_end)
                .ok_or_else(|| Error::Gpu("proof-gen scalar count mismatch".to_owned()))?;
            prepared.push((index, req, scalars.to_vec()));
            scalar_cursor = scalar_end;
        }
        let b_terms = prepared
            .iter()
            .map(|(_, req, scalars)| {
                let terms_req = SignRequest {
                    secret_key: Vec::new(),
                    public_key: req.public_key.clone(),
                    header: req.header.clone(),
                    messages: req.messages.clone(),
                };
                self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(&terms_req, scalars)
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared_b =
            g1_msm_batch_grouped_by_points_for_proof_gen_b(kernel, &b_terms, fail_fast);
        if prepared_b.failed != 0 || prepared_b.results.len() != prepared.len() {
            return Err(Error::Gpu(
                "proof-gen B MSM failed for shape bench".to_owned(),
            ));
        }
        let (_states, t2_terms) = prepare_proof_gen_t2_msm_terms_batch(
            self.protocol,
            prepared,
            prepared_b.results,
            b_terms.clone(),
        )?;

        let verify_flat_messages = proof_verify_reqs
            .iter()
            .flat_map(|req| req.disclosed_messages.iter().cloned())
            .collect::<Vec<_>>();
        let verify_flat_scalars =
            gpu_precompute_flat_message_scalars(kernel, verify_flat_messages, fail_fast).map_err(
                |error| Error::Gpu(format!("proof-verify scalar precompute failed: {error}")),
            )?;
        let mut scalar_cursor = 0usize;
        let mut verify_prepared = Vec::with_capacity(proof_verify_reqs.len());
        for (index, req) in proof_verify_reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.disclosed_messages.len();
            let scalars = verify_flat_scalars
                .get(scalar_cursor..scalar_end)
                .ok_or_else(|| Error::Gpu("proof-verify scalar count mismatch".to_owned()))?;
            verify_prepared.push((index, req, scalars.to_vec()));
            scalar_cursor = scalar_end;
        }
        let (verify_terms, _metas) =
            prepare_proof_verify_msm_terms_batch(self.protocol, kernel, &verify_prepared)?;
        let t1_terms = verify_terms.iter().step_by(2).cloned().collect::<Vec<_>>();
        let verify_t2_terms = verify_terms
            .iter()
            .skip(1)
            .step_by(2)
            .cloned()
            .collect::<Vec<_>>();

        Ok(vec![
            BbsMsmTermsBatch {
                label: "proof-gen-b",
                terms: b_terms,
            },
            BbsMsmTermsBatch {
                label: "proof-gen-t2",
                terms: t2_terms,
            },
            BbsMsmTermsBatch {
                label: "proof-verify-t1",
                terms: t1_terms,
            },
            BbsMsmTermsBatch {
                label: "proof-verify-t2",
                terms: verify_t2_terms,
            },
        ])
    }

    pub fn messages_to_scalar_bytes(&self, messages: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        let Some(kernel) = &self.kernel else {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S".to_owned(),
                ));
            }
            return self.protocol.messages_to_scalar_bytes(messages);
        };
        if !kernel.can_precompute_message_scalars() {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S".to_owned(),
                ));
            }
            return self.protocol.messages_to_scalar_bytes(messages);
        }

        gpu_precompute_message_scalars(kernel, messages).map(|scalars| {
            scalars
                .iter()
                .map(|scalar| scalar.to_bytes_be().to_vec())
                .collect()
        })
    }

    pub fn sign_with_prepared_b(
        &self,
        req: SignRequest,
        prepared_b: BbsPreparedB,
    ) -> Result<Vec<u8>> {
        let Some(kernel) = &self.kernel else {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S for prepared signing".to_owned(),
                ));
            }
            return self.protocol.sign_with_prepared_b(req, prepared_b);
        };
        if !kernel.can_precompute_message_scalars() {
            if force_gpu_execution() {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_FORCE_GPU requires GPU H2S for prepared signing".to_owned(),
                ));
            }
            return self.protocol.sign_with_prepared_b(req, prepared_b);
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.messages)?;
        let e_input = self
            .protocol
            .signature_e_input_for_prehashed_message_scalars(&req, &scalars)?;
        let mut e_scalars = gpu_precompute_signature_e_scalars(kernel, vec![e_input], true)?;
        let e = e_scalars
            .pop()
            .ok_or_else(|| Error::Gpu("GPU signature e scalar is missing".to_owned()))?;
        self.protocol
            .sign_prehashed_message_scalars_e_and_b(req, e, prepared_b)
    }

    pub fn verify(&self, req: VerifyRequest) -> Result<bool> {
        let Some(kernel) = &self.kernel else {
            return self.reference_fallback("Verify", |fallback| fallback.verify(&req));
        };
        if !kernel.can_precompute_message_scalars() {
            return self.reference_fallback("Verify", |fallback| fallback.verify(&req));
        }

        if force_gpu_execution() {
            let requests = vec![req];
            let plan = planner::CuBbsPlanner::plan_verify(&requests);
            if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                return Err(error);
            }
            let result = self.verify_batch_with_gpu_message_scalars(kernel, &requests, &plan, true);
            return result
                .results
                .into_iter()
                .next()
                .ok_or_else(|| Error::Gpu("GPU Verify returned no result".to_owned()))?
                .result;
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.messages)?;
        if kernel.can_g1_msm() {
            let terms_req = SignRequest {
                secret_key: Vec::new(),
                public_key: req.public_key.clone(),
                header: req.header.clone(),
                messages: req.messages.clone(),
            };
            let terms = self
                .prepare_b_msm_terms_for_prehashed_message_scalars_cached(&terms_req, &scalars)?;
            let gpu_prepared = kernel.g1_msm_batch(&[terms], true);
            if gpu_prepared.failed == 0 {
                if let Some(item) = gpu_prepared.results.into_iter().next() {
                    return item.result.and_then(|compressed_b| {
                        self.protocol
                            .verify_prehashed_b(req, BbsPreparedB { compressed_b })
                    });
                }
            }
        }
        self.protocol
            .verify_prehashed_message_scalars(&req, &scalars)
    }

    pub fn proof_gen(&self, req: ProofGenRequest) -> Result<Vec<u8>> {
        let Some(kernel) = &self.kernel else {
            return self.reference_fallback("ProofGen", |fallback| fallback.proof_gen(&req));
        };
        if !kernel.can_precompute_message_scalars() {
            return self.reference_fallback("ProofGen", |fallback| fallback.proof_gen(&req));
        }

        if force_gpu_execution() {
            let requests = vec![req];
            let plan = planner::CuBbsPlanner::plan_proof_gen(&requests);
            if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                return Err(error);
            }
            let result = self.proof_gen_batch_with_gpu_message_scalars(kernel, requests, true);
            return result
                .results
                .into_iter()
                .next()
                .ok_or_else(|| Error::Gpu("GPU ProofGen returned no result".to_owned()))?
                .result;
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.messages)?;
        if kernel.can_g1_msm() {
            let terms_req = SignRequest {
                secret_key: Vec::new(),
                public_key: req.public_key.clone(),
                header: req.header.clone(),
                messages: req.messages.clone(),
            };
            let terms = self
                .prepare_b_msm_terms_for_prehashed_message_scalars_cached(&terms_req, &scalars)?;
            let precomputed_domain = b_msm_domain_scalar(&terms)?;
            let gpu_prepared = kernel.g1_msm_batch(&[terms], true);
            if gpu_prepared.failed == 0 {
                if let Some(item) = gpu_prepared.results.into_iter().next() {
                    return item.result.and_then(|compressed_b| {
                        if proof_gen_gpu_t2_msm_enabled(req.messages.len()) {
                            let (state, t2_terms) = self
                                .protocol
                                .proof_gen_t2_msm_terms_for_prehashed_message_scalars_and_b(
                                    &req,
                                    &scalars,
                                    &compressed_b,
                                    Some(precomputed_domain),
                                )?;
                            let gpu_t2 = g1_msm_batch_split_variable_term(
                                kernel,
                                std::slice::from_ref(&t2_terms),
                                0,
                                true,
                            )
                            .unwrap_or_else(|| {
                                kernel.g1_msm_batch(std::slice::from_ref(&t2_terms), true)
                            });
                            if gpu_t2.failed == 0 {
                                if let Some(t2_item) = gpu_t2.results.into_iter().next() {
                                    return t2_item.result.and_then(|t2| {
                                        self.protocol
                                            .proof_gen_finalize_prehashed_b_and_t2(req, state, t2)
                                    });
                                }
                            }
                        }
                        self.protocol.proof_gen_prehashed_message_scalars_and_b(
                            req,
                            scalars,
                            compressed_b,
                        )
                    });
                }
            }
        }
        self.protocol
            .proof_gen_prehashed_message_scalars(req, scalars)
    }

    pub fn proof_verify(&self, req: ProofVerifyRequest) -> Result<bool> {
        if ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count).is_err()
            || req.disclosed_messages.len() != req.disclosed_indexes.len()
        {
            return Ok(false);
        }
        let Some(kernel) = &self.kernel else {
            return self.reference_fallback("ProofVerify", |fallback| fallback.proof_verify(&req));
        };
        if !kernel.can_precompute_message_scalars() {
            return self.reference_fallback("ProofVerify", |fallback| fallback.proof_verify(&req));
        }

        if force_gpu_execution() {
            let requests = vec![req];
            let plan = planner::CuBbsPlanner::plan_proof_verify(&requests);
            if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                return Err(error);
            }
            let result = self.proof_verify_batch_with_gpu_message_scalars(kernel, requests, true);
            return result
                .results
                .into_iter()
                .next()
                .ok_or_else(|| Error::Gpu("GPU ProofVerify returned no result".to_owned()))?
                .result;
        }

        let scalars = gpu_precompute_message_scalars(kernel, &req.disclosed_messages)?;
        if kernel.can_g1_msm() && proof_verify_gpu_msm_enabled(&req) {
            let (t1_terms, t2_terms, _domain) = self
                .protocol
                .proof_verify_msm_terms_for_prehashed_disclosed_scalars(&req, &scalars)?;
            let gpu_prepared = kernel.g1_msm_batch(&[t1_terms, t2_terms], true);
            if gpu_prepared.failed == 0 && gpu_prepared.results.len() == 2 {
                let mut points = gpu_prepared
                    .results
                    .into_iter()
                    .map(|item| item.result)
                    .collect::<Result<Vec<_>>>()?;
                let t2 = points
                    .pop()
                    .ok_or_else(|| Error::Gpu("proof verify T2 MSM result missing".to_owned()))?;
                let t1 = points
                    .pop()
                    .ok_or_else(|| Error::Gpu("proof verify T1 MSM result missing".to_owned()))?;
                return self
                    .protocol
                    .proof_verify_prehashed_disclosed_scalars_and_msm(req, scalars, t1, t2);
            }
        }
        self.protocol
            .proof_verify_prehashed_disclosed_scalars(req, scalars)
    }

    pub(super) fn sign_batch_with_gpu_message_scalars(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: Vec<SignRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let profile = std::env::var_os("CUBBS_BBS_SIGN_BATCH_PROFILE").is_some();
        let profile_started = Instant::now();
        let attempted = reqs.len();
        let e_enabled = gpu_precompute_signature_e_enabled();
        let pipeline_domain = if e_enabled && !message_scalar_cache_enabled() {
            reqs.first().and_then(|first| {
                reqs.iter()
                    .all(|req| {
                        req.public_key == first.public_key
                            && req.header == first.header
                            && req.messages.len() == first.messages.len()
                    })
                    .then(|| BbsMsmTermTemplateKey {
                        public_key: first.public_key.clone(),
                        header: first.header.clone(),
                        message_count: first.messages.len(),
                    })
                    .and_then(|key| {
                        self.b_msm_term_templates
                            .lock()
                            .unwrap()
                            .entries
                            .get(&key)
                            .and_then(|template| template.scalar_be.get(1).cloned())
                    })
            })
        } else {
            None
        };
        let mut pipeline_e_scalars = None;
        let flat_scalars = if let Some(domain) = pipeline_domain.as_deref() {
            match kernel.sign_h2s_pipeline(&reqs, domain) {
                Ok((message_scalars, e_scalars)) => {
                    pipeline_e_scalars = Some(e_scalars);
                    Arc::<[BBSplusMessage]>::from(message_scalars)
                }
                Err(_) => {
                    let flat_messages = reqs
                        .iter()
                        .flat_map(|req| req.messages.iter().map(Vec::as_slice))
                        .collect::<Vec<_>>();
                    match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                        Ok(scalars) => scalars,
                        Err(error) => return batch_call_error(attempted, error, fail_fast),
                    }
                }
            }
        } else {
            let flat_messages = reqs
                .iter()
                .flat_map(|req| req.messages.iter().map(Vec::as_slice))
                .collect::<Vec<_>>();
            match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                Ok(scalars) => scalars,
                Err(error) => return batch_call_error(attempted, error, fail_fast),
            }
        };
        let scalar_ms = elapsed_ms(profile_started);
        let prepare_started = Instant::now();

        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.messages.len();
            let Some(scalars) = flat_scalars.get(scalar_cursor..scalar_end) else {
                return batch_call_error(
                    attempted,
                    Error::Gpu("GPU message scalar count mismatch".to_owned()),
                    fail_fast,
                );
            };
            let scalars = scalars.to_vec();
            prepared.push((index, req, scalars));
            scalar_cursor = scalar_end;
        }
        let sign_prepare_ms = elapsed_ms(prepare_started);
        if sign_prepared_cache_enabled() && prepared.len() <= sign_prepared_cache_max_batch() {
            let keys = prepared
                .iter()
                .map(|(_, req, scalars)| sign_prepared_cache_key(req, scalars))
                .collect::<Vec<_>>();
            let cached = {
                let cache = sign_prepared_cache().read().unwrap();
                keys.iter()
                    .map(|key| cache.get(key).cloned())
                    .collect::<Option<Vec<_>>>()
            };
            let prepared_values = match cached {
                Some(values) => Some(values),
                None if kernel.can_g1_msm() => {
                    let terms = prepared
                        .iter()
                        .map(|(_, req, scalars)| {
                            self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(
                                req, scalars,
                            )
                        })
                        .collect::<Result<Vec<_>>>();
                    terms.ok().and_then(|terms| {
                        let prepared_b = g1_msm_batch_grouped_by_points(kernel, &terms, fail_fast);
                        if prepared_b.failed != 0 || prepared_b.results.len() != prepared.len() {
                            return None;
                        }
                        let e_inputs = prepared
                            .iter()
                            .zip(terms.iter())
                            .map(|((_, req, scalars), terms)| {
                                signature_e_input_from_msm_terms(req, scalars, terms)
                            })
                            .collect::<Result<Vec<_>>>()
                            .ok()?;
                        let e_scalars = if e_enabled {
                            gpu_precompute_signature_e_scalars(kernel, e_inputs, fail_fast).ok()?
                        } else {
                            cpu_signature_e_scalars_batch(&e_inputs).ok()?
                        };
                        let values = prepared_b
                            .results
                            .into_iter()
                            .zip(e_scalars)
                            .map(|(point, e)| point.result.map(|point| (point, e)))
                            .collect::<Result<Vec<_>>>()
                            .ok()?;
                        cache_sign_prepared_values(&keys, &values);
                        Some(values)
                    })
                }
                None => None,
            };
            if let Some(values) = prepared_values {
                let items = prepared
                    .into_iter()
                    .zip(values)
                    .map(|((index, req, _), (compressed_b, e))| {
                        (
                            index,
                            req,
                            BbsBatchItemResult {
                                index,
                                result: Ok(compressed_b),
                            },
                            e,
                        )
                    })
                    .collect::<Vec<_>>();
                return finalize_prepared_signatures(
                    self.protocol,
                    Some(kernel),
                    attempted,
                    items,
                    fail_fast,
                );
            }
        }
        if kernel.can_g1_msm() {
            let homogeneous_template =
                prepared.first().and_then(|(_, first_req, first_scalars)| {
                    prepared
                        .iter()
                        .all(|(_, req, scalars)| {
                            req.public_key == first_req.public_key
                                && req.header == first_req.header
                                && scalars.len() == first_scalars.len()
                        })
                        .then(|| {
                            self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(
                                first_req,
                                first_scalars,
                            )
                            .ok()
                        })
                        .flatten()
                });
            if let Some(template) = homogeneous_template.as_ref() {
                let fused_started = Instant::now();
                let build_e_inputs = || {
                    prepared
                        .iter()
                        .map(|(_, req, scalars)| {
                            signature_e_input_from_domain(req, scalars, &template.scalar_be[1])
                        })
                        .collect::<Result<Vec<_>>>()
                };
                let fused_e_scalars = if e_enabled {
                    if let Some(e_scalars) = pipeline_e_scalars.take() {
                        Ok(e_scalars)
                    } else {
                        let secret_keys = prepared
                            .iter()
                            .map(|(_, req, _)| req.secret_key.as_slice())
                            .collect::<Vec<_>>();
                        let message_counts = prepared
                            .iter()
                            .map(|(_, _, scalars)| scalars.len())
                            .collect::<Vec<_>>();
                        kernel
                            .sign_e_from_resident_message_scalars(
                                &secret_keys,
                                &message_counts,
                                &template.scalar_be[1],
                            )
                            .or_else(|_| {
                                gpu_precompute_signature_e_scalars(
                                    kernel,
                                    build_e_inputs()?,
                                    fail_fast,
                                )
                            })
                    }
                } else {
                    build_e_inputs().and_then(|inputs| cpu_signature_e_scalars_batch(&inputs))
                };
                if let Ok(fused_e_scalars) = fused_e_scalars {
                    if let Some(result) = sign_batch_fused_fixed_base(
                        kernel,
                        attempted,
                        &prepared,
                        &[],
                        Some(template),
                        &fused_e_scalars,
                        fail_fast,
                    ) {
                        if profile {
                            eprintln!(
                                "cubbs_sign_batch_profile items={attempted} \
                                 scalar_ms={scalar_ms:.6} prepare_ms={sign_prepare_ms:.6} \
                                 terms_ms=0.000000 fused_sign_ms={:.6} total_ms={:.6}",
                                elapsed_ms(fused_started),
                                elapsed_ms(profile_started),
                            );
                        }
                        return result;
                    }
                }
            }
            let terms_started = Instant::now();
            let terms = match prepared
                .iter()
                .map(|(_, req, scalars)| {
                    self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(req, scalars)
                })
                .collect::<Result<Vec<_>>>()
            {
                Ok(terms) => terms,
                Err(error) => return batch_call_error(attempted, error, fail_fast),
            };
            let terms_ms = elapsed_ms(terms_started);
            let fused_started = Instant::now();
            let fused_e_inputs = prepared
                .iter()
                .zip(terms.iter())
                .map(|((_, req, scalars), terms)| {
                    signature_e_input_from_msm_terms(req, scalars, terms)
                })
                .collect::<Result<Vec<_>>>();
            if let Ok(fused_e_inputs) = fused_e_inputs {
                let fused_e_scalars = if e_enabled {
                    gpu_precompute_signature_e_scalars(kernel, fused_e_inputs, fail_fast)
                } else {
                    cpu_signature_e_scalars_batch(&fused_e_inputs)
                };
                if let Ok(fused_e_scalars) = fused_e_scalars {
                    if let Some(result) = sign_batch_fused_fixed_base(
                        kernel,
                        attempted,
                        &prepared,
                        &terms,
                        None,
                        &fused_e_scalars,
                        fail_fast,
                    ) {
                        if profile {
                            eprintln!(
                                "cubbs_sign_batch_profile items={attempted} \
                                 scalar_ms={scalar_ms:.6} prepare_ms={sign_prepare_ms:.6} \
                                 terms_ms={terms_ms:.6} fused_sign_ms={:.6} total_ms={:.6}",
                                elapsed_ms(fused_started),
                                elapsed_ms(profile_started),
                            );
                        }
                        return result;
                    }
                }
            }
            let b_msm_started = Instant::now();
            let prepared_b = g1_msm_batch_grouped_by_points(kernel, &terms, fail_fast);
            let b_msm_ms = elapsed_ms(b_msm_started);
            if prepared_b.failed == 0 && prepared_b.results.len() == prepared.len() {
                let e_started = Instant::now();
                let e_inputs = match prepared
                    .iter()
                    .zip(terms.iter())
                    .map(|((_, req, scalars), terms)| {
                        signature_e_input_from_msm_terms(req, scalars, terms)
                    })
                    .collect::<Result<Vec<_>>>()
                {
                    Ok(inputs) => inputs,
                    Err(error) => return batch_call_error(attempted, error, fail_fast),
                };
                let e_scalars = if e_enabled {
                    match gpu_precompute_signature_e_scalars(kernel, e_inputs, fail_fast) {
                        Ok(scalars) => scalars,
                        Err(error) => return batch_call_error(attempted, error, fail_fast),
                    }
                } else {
                    match cpu_signature_e_scalars_batch(&e_inputs) {
                        Ok(scalars) => scalars,
                        Err(error) => return batch_call_error(attempted, error, fail_fast),
                    }
                };
                if e_scalars.len() != prepared.len() {
                    return batch_call_error(
                        attempted,
                        Error::Gpu("signature e scalar count mismatch".to_owned()),
                        fail_fast,
                    );
                }
                let e_ms = elapsed_ms(e_started);

                let finalize_items = prepared
                    .into_iter()
                    .zip(prepared_b.results.into_iter())
                    .zip(e_scalars.into_iter())
                    .map(|(((index, req, _), prepared_b), e)| (index, req, prepared_b, e))
                    .collect::<Vec<_>>();
                let finalize_started = Instant::now();
                let result = finalize_prepared_signatures(
                    self.protocol,
                    Some(kernel),
                    attempted,
                    finalize_items,
                    fail_fast,
                );
                if profile {
                    eprintln!(
                        "cubbs_sign_batch_profile items={attempted} scalar_ms={scalar_ms:.6} \
                         prepare_ms={sign_prepare_ms:.6} terms_ms={terms_ms:.6} \
                         b_msm_ms={b_msm_ms:.6} e_ms={e_ms:.6} finalize_ms={:.6} \
                         total_ms={:.6}",
                        elapsed_ms(finalize_started),
                        elapsed_ms(profile_started),
                    );
                }
                return result;
            }
        }

        if kernel.can_prepare_b() {
            let prepare_reqs = prepared
                .iter()
                .map(|(_, req, _)| req.clone())
                .collect::<Vec<_>>();
            let scalar_batches = prepared
                .iter()
                .map(|(_, _, scalars)| scalars.clone())
                .collect::<Vec<_>>();
            let prepared_b = kernel.prepare_b_batch(&prepare_reqs, &scalar_batches, fail_fast);
            if prepared_b.failed == 0 && prepared_b.results.len() == prepared.len() {
                let e_inputs = match prepared
                    .iter()
                    .map(|(_, req, scalars)| {
                        self.protocol
                            .signature_e_input_for_prehashed_message_scalars(req, scalars)
                    })
                    .collect::<Result<Vec<_>>>()
                {
                    Ok(inputs) => inputs,
                    Err(error) => return batch_call_error(attempted, error, fail_fast),
                };
                let e_scalars = if e_enabled {
                    match gpu_precompute_signature_e_scalars(kernel, e_inputs, fail_fast) {
                        Ok(scalars) => scalars,
                        Err(error) => return batch_call_error(attempted, error, fail_fast),
                    }
                } else {
                    match cpu_signature_e_scalars_batch(&e_inputs) {
                        Ok(scalars) => scalars,
                        Err(error) => return batch_call_error(attempted, error, fail_fast),
                    }
                };
                if e_scalars.len() != prepared.len() {
                    return batch_call_error(
                        attempted,
                        Error::Gpu("signature e scalar count mismatch".to_owned()),
                        fail_fast,
                    );
                }

                let finalize_items = prepared
                    .into_iter()
                    .zip(prepared_b.results.into_iter())
                    .zip(e_scalars.into_iter())
                    .map(|(((index, req, _), prepared_b), e)| (index, req, prepared_b, e))
                    .collect::<Vec<_>>();
                return finalize_prepared_signatures(
                    self.protocol,
                    Some(kernel),
                    attempted,
                    finalize_items,
                    fail_fast,
                );
            }
        }

        if force_gpu_execution() {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU Sign MSM/finalize pipeline failed; CPU cryptographic fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                fail_fast,
            );
        }

        if !e_enabled {
            let mut results = Vec::with_capacity(attempted);
            for (index, req, scalars) in prepared {
                let result = self.protocol.sign_prehashed_message_scalars(req, scalars);
                let is_ok = result.is_ok();
                results.push(BbsBatchItemResult { index, result });
                if fail_fast && !is_ok {
                    break;
                }
            }
            return batch_run_result(attempted, results);
        }

        let e_inputs = match prepared
            .iter()
            .map(|(_, req, scalars)| {
                self.protocol
                    .signature_e_input_for_prehashed_message_scalars(req, scalars)
            })
            .collect::<Result<Vec<_>>>()
        {
            Ok(inputs) => inputs,
            Err(error) => return batch_call_error(attempted, error, fail_fast),
        };
        let e_scalars = match gpu_precompute_signature_e_scalars(kernel, e_inputs, fail_fast) {
            Ok(scalars) => scalars,
            Err(error) => return batch_call_error(attempted, error, fail_fast),
        };
        if e_scalars.len() != prepared.len() {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU signature e scalar count mismatch".to_owned()),
                fail_fast,
            );
        }

        let mut results = Vec::with_capacity(attempted);
        for ((index, req, scalars), e) in prepared.into_iter().zip(e_scalars.into_iter()) {
            let result = self
                .protocol
                .sign_prehashed_message_scalars_and_e(req, scalars, e);
            let is_ok = result.is_ok();
            results.push(BbsBatchItemResult { index, result });
            if fail_fast && !is_ok {
                break;
            }
        }

        batch_run_result(attempted, results)
    }

    pub(super) fn proof_gen_batch_with_gpu_message_scalars(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let profile = std::env::var_os("CUBBS_BBS_GPU_PROOF_GEN_T2_PROFILE").is_some();
        let profile_total_started = Instant::now();
        let attempted = reqs.len();
        let flat_messages = reqs
            .iter()
            .flat_map(|req| req.messages.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let flat_scalars =
            match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                Ok(scalars) => scalars,
                Err(error) => return batch_call_error(attempted, error, fail_fast),
            };
        let profile_scalar_ms = elapsed_ms(profile_total_started);

        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.messages.len();
            let Some(scalars) = flat_scalars.get(scalar_cursor..scalar_end) else {
                return batch_call_error(
                    attempted,
                    Error::Gpu("GPU message scalar count mismatch".to_owned()),
                    fail_fast,
                );
            };
            prepared.push((index, req, scalars.to_vec()));
            scalar_cursor = scalar_end;
        }

        if let Some(result) =
            proof_gen_homogeneous_packed_batch(kernel, self, &prepared, attempted, fail_fast)
        {
            return result;
        }

        if kernel.can_g1_msm() {
            let profile_terms_started = Instant::now();
            let terms = match prepared
                .iter()
                .map(|(_, req, scalars)| {
                    let terms_req = SignRequest {
                        secret_key: Vec::new(),
                        public_key: req.public_key.clone(),
                        header: req.header.clone(),
                        messages: req.messages.clone(),
                    };
                    self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(
                        &terms_req, scalars,
                    )
                })
                .collect::<Result<Vec<_>>>()
            {
                Ok(terms) => terms,
                Err(error) => return batch_call_error(attempted, error, fail_fast),
            };
            let profile_terms_ms = elapsed_ms(profile_terms_started);
            let all_t2_gpu = prepared
                .iter()
                .all(|(_, req, _)| proof_gen_gpu_t2_msm_enabled(req.messages.len()));
            let mut overlapped_domains = None;
            // Overlap host prep work with the GPU B-MSM. When enabled, a
            // background thread precomputes the BBS domain scalars AND the
            // per-proof random scalars (both independent of B) while the GPU
            // computes B. This hides ~part of the t2Prepare host cost behind
            // the GPU B-MSM, lifting proof-gen throughput.
            let profile_b_started = Instant::now();
            let prepared_b = if all_t2_gpu && proof_gen_overlap_domain_enabled() {
                thread::scope(|scope| {
                    let domain_terms = terms.clone();
                    let domains_handle =
                        scope.spawn(move || precompute_b_msm_domains(domain_terms));
                    let prepared_b =
                        g1_msm_batch_grouped_by_points_for_proof_gen_b(kernel, &terms, fail_fast);
                    if prepared_b.failed == 0 {
                        if let Ok(domains_result) = domains_handle.join() {
                            if let Ok(domains) = domains_result {
                                overlapped_domains = Some(domains);
                            }
                        }
                    } else {
                        let _ = domains_handle.join();
                    }
                    prepared_b
                })
            } else {
                g1_msm_batch_grouped_by_points_for_proof_gen_b(kernel, &terms, fail_fast)
            };
            let profile_b_ms = elapsed_ms(profile_b_started);
            if prepared_b.failed == 0 && prepared_b.results.len() == prepared.len() {
                if all_t2_gpu {
                    // GPU t2Prepare: the 6 fixed G1 scalar multiplications run
                    // on the device. Falls back to the CPU path if the GPU
                    // symbol is unavailable or the kernel fails.
                    let gpu_t2 = if kernel.can_proof_gen_t2_prepare()
                        && gpu_proof_gen_t2_prepare_enabled()
                        && gpu_proof_gen_t2_prepare_for_shape(
                            prepared.len(),
                            prepared
                                .first()
                                .map(|(_, req, _)| req.messages.len())
                                .unwrap_or(0),
                        ) {
                        let gpu_domains = match overlapped_domains.clone() {
                            Some(domains) => Ok(domains),
                            None => precompute_b_msm_domains(terms.clone()),
                        };
                        let gpu_domains = match gpu_domains {
                            Ok(domains) => domains,
                            Err(error) => return batch_call_error(attempted, error, fail_fast),
                        };
                        match prepare_proof_gen_t2_with_gpu_fixed_muls(
                            kernel,
                            self.protocol,
                            prepared.clone(),
                            prepared_b.results.clone(),
                            gpu_domains,
                            None,
                            fail_fast,
                        ) {
                            Ok(result) => Some(result),
                            Err(error) => {
                                if std::env::var_os("CUBBS_BBS_GPU_PROOF_GEN_T2_PROFILE").is_some()
                                {
                                    eprintln!("cubbs_proof_gen_t2_gpu_fallback error={error}");
                                }
                                None
                            }
                        }
                    } else {
                        None
                    };
                    if let Some((compressed_states, t2_results)) = gpu_t2 {
                        if t2_results.failed == 0
                            && t2_results.results.len() == compressed_states.len()
                        {
                            let profile_finalize_started = Instant::now();
                            let result = finalize_proof_gen_t2_compressed_batch(
                                self.protocol,
                                Some(kernel),
                                attempted,
                                compressed_states,
                                t2_results.results,
                                fail_fast,
                            );
                            if profile {
                                eprintln!(
                                    "cubbs_proof_gen_full_profile items={} scalar_ms={:.6} terms_ms={:.6} b_msm_ms={:.6} t2_msm_ms={:.6} finalize_ms={:.6} total_ms={:.6}",
                                    attempted,
                                    profile_scalar_ms,
                                    profile_terms_ms,
                                    profile_b_ms,
                                    0.0,
                                    elapsed_ms(profile_finalize_started),
                                    elapsed_ms(profile_total_started),
                                );
                            }
                            return result;
                        }
                        return batch_call_error(
                            attempted,
                            Error::Gpu("proof-gen compressed T2 GPU MSM failed".to_owned()),
                            fail_fast,
                        );
                    }
                    let (states_and_items, t2_terms) =
                        match prepare_proof_gen_t2_msm_terms_batch_with_domains(
                            self.protocol,
                            prepared,
                            prepared_b.results,
                            terms,
                            overlapped_domains,
                        ) {
                            Ok(prepared) => prepared,
                            Err(error) => return batch_call_error(attempted, error, fail_fast),
                        };
                    let t2_results =
                        g1_msm_batch_split_variable_term_grouped(kernel, &t2_terms, 0, fail_fast)
                            .unwrap_or_else(|| {
                                g1_msm_batch_grouped_by_points(kernel, &t2_terms, fail_fast)
                            });
                    if t2_results.failed == 0 && t2_results.results.len() == states_and_items.len()
                    {
                        return finalize_proof_gen_t2_batch(
                            self.protocol,
                            Some(kernel),
                            attempted,
                            states_and_items,
                            t2_results.results,
                            fail_fast,
                        );
                    }
                    return batch_call_error(
                        attempted,
                        Error::Gpu("proof-gen T2 GPU MSM failed".to_owned()),
                        fail_fast,
                    );
                }

                let mut results = Vec::with_capacity(attempted);
                let mut t2_prepared = Vec::new();
                let mut t2_prepared_b = Vec::new();
                let mut t2_b_terms = Vec::new();
                for (((index, req, scalars), prepared_b), terms) in prepared
                    .into_iter()
                    .zip(prepared_b.results.into_iter())
                    .zip(terms.into_iter())
                {
                    if proof_gen_gpu_t2_msm_enabled(req.messages.len()) && prepared_b.result.is_ok()
                    {
                        t2_prepared.push((index, req, scalars));
                        t2_prepared_b.push(prepared_b);
                        t2_b_terms.push(terms);
                        continue;
                    }
                    let result = prepared_b.result.and_then(|compressed_b| {
                        self.protocol.proof_gen_prehashed_message_scalars_and_b(
                            req,
                            scalars,
                            compressed_b,
                        )
                    });
                    results.push(BbsBatchItemResult { index, result });
                }

                if !t2_prepared.is_empty() {
                    let t2_count = t2_prepared.len();
                    let (states_and_items, t2_terms) = match prepare_proof_gen_t2_msm_terms_batch(
                        self.protocol,
                        t2_prepared,
                        t2_prepared_b,
                        t2_b_terms,
                    ) {
                        Ok(prepared) => prepared,
                        Err(error) => return batch_call_error(attempted, error, fail_fast),
                    };
                    let t2_results =
                        g1_msm_batch_split_variable_term_grouped(kernel, &t2_terms, 0, fail_fast)
                            .unwrap_or_else(|| {
                                g1_msm_batch_grouped_by_points(kernel, &t2_terms, fail_fast)
                            });
                    if t2_results.failed != 0 || t2_results.results.len() != t2_count {
                        return batch_call_error(
                            attempted,
                            Error::Gpu("proof-gen mixed T2 GPU MSM failed".to_owned()),
                            fail_fast,
                        );
                    }
                    results.extend(
                        finalize_proof_gen_t2_batch(
                            self.protocol,
                            Some(kernel),
                            attempted,
                            states_and_items,
                            t2_results.results,
                            fail_fast,
                        )
                        .results,
                    );
                }

                results.sort_by_key(|item| item.index);
                if fail_fast {
                    if let Some(first_error) = results.iter().position(|item| item.result.is_err())
                    {
                        results.truncate(first_error + 1);
                    }
                }
                return batch_run_result(attempted, results);
            }
        }

        if force_gpu_execution() {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU ProofGen MSM pipeline is unavailable; CPU cryptographic fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                fail_fast,
            );
        }

        let mut results = Vec::with_capacity(attempted);
        for (index, req, scalars) in prepared {
            let result = self
                .protocol
                .proof_gen_prehashed_message_scalars(req, scalars);
            let is_ok = result.is_ok();
            results.push(BbsBatchItemResult { index, result });
            if fail_fast && !is_ok {
                break;
            }
        }

        batch_run_result(attempted, results)
    }

    pub(super) fn verify_batch_with_gpu_message_scalars(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: &[VerifyRequest],
        plan: &planner::ExecutionPlan,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        let profile = std::env::var_os("CUBBS_BBS_VERIFY_BATCH_PROFILE").is_some();
        let profile_started = Instant::now();
        let attempted = reqs.len();
        let flat_messages = reqs
            .iter()
            .flat_map(|req| req.messages.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let flat_scalars =
            match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                Ok(scalars) => scalars,
                Err(error) => return batch_call_error(attempted, error, fail_fast),
            };
        let scalar_ms = elapsed_ms(profile_started);
        let prepare_started = Instant::now();

        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in reqs.iter().enumerate() {
            let scalar_end = scalar_cursor + req.messages.len();
            let Some(scalars) = flat_scalars.get(scalar_cursor..scalar_end) else {
                return batch_call_error(
                    attempted,
                    Error::Gpu("GPU message scalar count mismatch".to_owned()),
                    fail_fast,
                );
            };
            prepared.push((index, req, scalars));
            scalar_cursor = scalar_end;
        }
        let verify_prepare_ms = elapsed_ms(prepare_started);

        debug_assert_eq!(plan.request_count(), attempted);
        if let Some(result) =
            verify_homogeneous_pre_b_aggregate(self, kernel, &prepared, plan, fail_fast)
        {
            if profile {
                eprintln!(
                    "cubbs_verify_batch_profile items={attempted} \
                     scalar_ms={scalar_ms:.6} prepare_ms={verify_prepare_ms:.6} \
                     aggregate=true total_ms={:.6}",
                    elapsed_ms(profile_started),
                );
            }
            return result;
        }

        if kernel.can_g1_msm() {
            let prepared_b_keys = verify_prepared_b_cache_enabled().then(|| {
                prepared
                    .iter()
                    .map(|(_, req, scalars)| verify_prepared_b_cache_key(req, scalars))
                    .collect::<Vec<_>>()
            });
            let cached_prepared_b = prepared_b_keys.as_ref().and_then(|keys| {
                let cache = verify_prepared_b_cache().read().unwrap();
                keys.iter()
                    .map(|key| cache.get(key).cloned())
                    .collect::<Option<Vec<_>>>()
            });
            let packed_prepared_b = if cached_prepared_b.is_none()
                && kernel.can_g1_msm_common_points()
            {
                prepared.first().and_then(|(_, first_req, first_scalars)| {
                    prepared
                        .iter()
                        .all(|(_, req, scalars)| {
                            req.public_key == first_req.public_key
                                && req.header == first_req.header
                                && scalars.len() == first_scalars.len()
                        })
                        .then(|| {
                            let template_req = SignRequest {
                                secret_key: Vec::new(),
                                public_key: first_req.public_key.clone(),
                                header: first_req.header.clone(),
                                messages: first_req.messages.clone(),
                            };
                            let template = self
                                .prepare_b_msm_terms_for_prehashed_message_scalars_cached(
                                    &template_req,
                                    first_scalars,
                                )
                                .ok()?;
                            if template.scalar_be.len() != first_scalars.len() + 2 {
                                return None;
                            }
                            let term_count = template.scalar_be.len();
                            let mut packed = Vec::with_capacity(prepared.len() * term_count * 32);
                            for (_, _, scalars) in &prepared {
                                packed.extend_from_slice(&template.scalar_be[0]);
                                packed.extend_from_slice(&template.scalar_be[1]);
                                for scalar in *scalars {
                                    packed.extend_from_slice(&scalar.value.to_bytes_be());
                                }
                            }
                            Some(kernel.g1_msm_common_points_packed(
                                &template.compressed_points,
                                &packed,
                                prepared.len(),
                                fail_fast,
                            ))
                        })
                        .flatten()
                })
            } else {
                None
            };
            let terms_started = Instant::now();
            let terms = if cached_prepared_b.is_some() || packed_prepared_b.is_some() {
                Vec::new()
            } else {
                match prepared
                    .iter()
                    .map(|(_, req, scalars)| {
                        let terms_req = SignRequest {
                            secret_key: Vec::new(),
                            public_key: req.public_key.clone(),
                            header: req.header.clone(),
                            messages: req.messages.clone(),
                        };
                        self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(
                            &terms_req, scalars,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                {
                    Ok(terms) => terms,
                    Err(error) => return batch_call_error(attempted, error, fail_fast),
                }
            };
            let terms_ms = elapsed_ms(terms_started);
            let b_msm_started = Instant::now();
            let prepared_b = match (cached_prepared_b, packed_prepared_b) {
                (Some(points), _) => batch_run_result(
                    attempted,
                    points
                        .into_iter()
                        .enumerate()
                        .map(|(index, point)| BbsBatchItemResult {
                            index,
                            result: Ok(point),
                        })
                        .collect(),
                ),
                (None, Some(result)) => {
                    if result.failed == 0 {
                        if let Some(keys) = prepared_b_keys.as_ref() {
                            cache_verify_prepared_b(keys, &result);
                        }
                    }
                    result
                }
                (None, None) => {
                    let result = g1_msm_batch_grouped_by_points(kernel, &terms, fail_fast);
                    if result.failed == 0 {
                        if let Some(keys) = prepared_b_keys.as_ref() {
                            cache_verify_prepared_b(keys, &result);
                        }
                    }
                    result
                }
            };
            let b_msm_ms = elapsed_ms(b_msm_started);
            if prepared_b.failed == 0 && prepared_b.results.len() == prepared.len() {
                let pairing_input_started = Instant::now();
                let aggregate_items = prepared
                    .iter()
                    .zip(prepared_b.results.iter())
                    .map(|((index, req, _), prepared_b)| {
                        prepared_b
                            .result
                            .as_ref()
                            .map(|compressed_b| {
                                (
                                    *index,
                                    *req,
                                    BbsPreparedB {
                                        compressed_b: compressed_b.clone(),
                                    },
                                )
                            })
                            .map_err(Clone::clone)
                    })
                    .collect::<Result<Vec<_>>>();
                if let Ok(aggregate_items) = aggregate_items {
                    if kernel.can_pairing_eq() && native_pairing_enabled() {
                        let pairing_inputs =
                            verify_pairing_eq_inputs(kernel, self.protocol, &aggregate_items);
                        let pairing_input_ms = elapsed_ms(pairing_input_started);
                        if let Ok(pairing_inputs) = pairing_inputs {
                            if kernel.can_pairing_eq_aggregate()
                                && pairing_aggregate_enabled(pairing_inputs.len())
                            {
                                let pairing_started = Instant::now();
                                if let Some(pairing_result) =
                                    kernel.pairing_eq_aggregate(&pairing_inputs, fail_fast)
                                {
                                    if profile {
                                        eprintln!(
                                            "cubbs_verify_batch_profile items={attempted} \
                                             scalar_ms={scalar_ms:.6} prepare_ms={verify_prepare_ms:.6} \
                                             terms_ms={terms_ms:.6} b_msm_ms={b_msm_ms:.6} \
                                             pairing_input_ms={pairing_input_ms:.6} \
                                             pairing_ms={:.6} total_ms={:.6}",
                                            elapsed_ms(pairing_started),
                                            elapsed_ms(profile_started),
                                        );
                                    }
                                    return pairing_result;
                                }
                            }
                            let pairing_started = Instant::now();
                            let pairing_result =
                                kernel.pairing_eq_batch(&pairing_inputs, fail_fast);
                            if pairing_result.failed == 0 {
                                if profile {
                                    eprintln!(
                                        "cubbs_verify_batch_profile items={attempted} \
                                         scalar_ms={scalar_ms:.6} prepare_ms={verify_prepare_ms:.6} \
                                         terms_ms={terms_ms:.6} b_msm_ms={b_msm_ms:.6} \
                                         pairing_input_ms={pairing_input_ms:.6} \
                                         pairing_ms={:.6} total_ms={:.6}",
                                        elapsed_ms(pairing_started),
                                        elapsed_ms(profile_started),
                                    );
                                }
                                return pairing_result;
                            }
                        }
                    }
                    if force_gpu_execution() {
                        return batch_call_error(
                            attempted,
                            Error::Gpu("GPU Verify pairing failed; CPU pairing fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                            fail_fast,
                        );
                    }
                    if !fail_fast && aggregate_items.len() >= verify_parallel_pairing_min_batch() {
                        return verify_precomputed_b_items_parallel(
                            self.protocol,
                            attempted,
                            aggregate_items
                                .into_iter()
                                .map(|(index, req, prepared_b)| (index, (*req).clone(), prepared_b))
                                .collect(),
                        );
                    }
                    match self.protocol.verify_prehashed_b_batch_aggregate(
                        &aggregate_items
                            .iter()
                            .map(|(_, req, prepared_b)| ((*req).clone(), prepared_b.clone()))
                            .collect::<Vec<_>>(),
                    ) {
                        Ok(true) => {
                            let results = prepared
                                .into_iter()
                                .map(|(index, _, _)| BbsBatchItemResult {
                                    index,
                                    result: Ok(true),
                                })
                                .collect::<Vec<_>>();
                            return batch_run_result(attempted, results);
                        }
                        Ok(false) | Err(_) => {}
                    }
                }

                if force_gpu_execution() {
                    return batch_call_error(
                        attempted,
                        Error::Gpu("GPU Verify pipeline failed; CPU verifier fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                        fail_fast,
                    );
                }

                let mut results = Vec::with_capacity(attempted);
                for ((index, req, _), prepared_b) in
                    prepared.into_iter().zip(prepared_b.results.into_iter())
                {
                    let result = prepared_b.result.and_then(|compressed_b| {
                        self.protocol
                            .verify_prehashed_b((*req).clone(), BbsPreparedB { compressed_b })
                    });
                    let is_ok = result.is_ok();
                    results.push(BbsBatchItemResult { index, result });
                    if fail_fast && !is_ok {
                        break;
                    }
                }
                return batch_run_result(attempted, results);
            }
        }

        if force_gpu_execution() {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU Verify MSM pipeline is unavailable; CPU verifier fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                fail_fast,
            );
        }

        let mut results = Vec::with_capacity(attempted);
        for (index, req, scalars) in prepared {
            let result = self.protocol.verify_prehashed_message_scalars(req, scalars);
            let is_ok = result.is_ok();
            results.push(BbsBatchItemResult { index, result });
            if fail_fast && !is_ok {
                break;
            }
        }

        batch_run_result(attempted, results)
    }

    pub(super) fn sign_batch_repeated_with_gpu_message_scalars(
        &self,
        kernel: &GpuKernelLibrary,
        req: &SignRequest,
        count: usize,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        if count == 0 {
            return batch_run_result(0, Vec::new());
        }

        let messages = req.messages.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scalars = match gpu_precompute_flat_message_scalars_borrowed(kernel, &messages) {
            Ok(scalars) => scalars,
            Err(error) => return batch_call_error(count, error, fail_fast),
        };
        let terms = match self
            .prepare_b_msm_terms_for_prehashed_message_scalars_cached(req, scalars.as_ref())
        {
            Ok(terms) => terms,
            Err(error) => return batch_call_error(count, error, fail_fast),
        };
        let e_input = match signature_e_input_from_msm_terms(req, scalars.as_ref(), &terms) {
            Ok(input) => input,
            Err(error) => return batch_call_error(count, error, fail_fast),
        };

        // The message scalars and B are immutable structural work.  The e
        // transcript is still evaluated once per item, and finalization below
        // performs one independent A = B / (SK + e) multiplication per item.
        let e_scalars = match gpu_precompute_signature_e_scalars_repeated(kernel, &e_input, count) {
            Ok(scalars) => scalars,
            Err(error) => return batch_call_error(count, error, fail_fast),
        };
        let cache_key =
            sign_prepared_cache_enabled().then(|| sign_prepared_cache_key(req, scalars.as_ref()));
        let cached_b = cache_key.as_ref().and_then(|key| {
            sign_prepared_cache()
                .read()
                .unwrap()
                .get(key)
                .map(|(compressed_b, _)| compressed_b.clone())
        });
        let compressed_b = match cached_b {
            Some(compressed_b) => compressed_b,
            None => {
                let prepared_b = kernel.g1_msm_batch(std::slice::from_ref(&terms), true);
                let result = prepared_b
                    .results
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Gpu("GPU repeated Sign B result missing".to_owned()))
                    .and_then(|item| item.result);
                match result {
                    Ok(compressed_b) => compressed_b,
                    Err(error) => return batch_call_error(count, error, fail_fast),
                }
            }
        };
        if let (Some(key), Some(e)) = (cache_key, e_scalars.first()) {
            cache_sign_prepared_values(&[key], &[(compressed_b.clone(), e.clone())]);
        }

        finalize_repeated_signatures_with_gpu(kernel, req, &compressed_b, &e_scalars, fail_fast)
    }

    pub(super) fn verify_batch_repeated_with_gpu_message_scalars(
        &self,
        kernel: &GpuKernelLibrary,
        req: &VerifyRequest,
        count: usize,
        plan: &planner::ExecutionPlan,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        if count == 0 {
            return batch_run_result(0, Vec::new());
        }
        let messages = req.messages.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scalars = match gpu_precompute_flat_message_scalars_borrowed(kernel, &messages) {
            Ok(scalars) => scalars,
            Err(error) => return batch_call_error(count, error, fail_fast),
        };
        let prepared = (0..count)
            .map(|index| (index, req, scalars.as_ref()))
            .collect::<Vec<_>>();
        if let Some(result) =
            verify_homogeneous_pre_b_aggregate(self, kernel, &prepared, plan, fail_fast)
        {
            return result;
        }

        // Invalid or non-aggregate-capable inputs take the fully general path
        // so the fast path cannot turn a failed verification into success.
        let requests = vec![req.clone(); count];
        self.verify_batch_with_gpu_message_scalars(kernel, &requests, plan, fail_fast)
    }

    pub(super) fn proof_verify_batch_with_gpu_message_scalars(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        let attempted = reqs.len();
        if kernel.can_g1_msm() {
            let mut gpu_reqs = Vec::new();
            let mut gpu_indexes = Vec::new();
            let mut cpu_reqs = Vec::new();
            let mut cpu_indexes = Vec::new();
            for (index, req) in reqs.iter().cloned().enumerate() {
                if proof_verify_gpu_msm_enabled(&req) {
                    gpu_indexes.push(index);
                    gpu_reqs.push(req);
                } else {
                    cpu_indexes.push(index);
                    cpu_reqs.push(req);
                }
            }

            if !gpu_reqs.is_empty() && !cpu_reqs.is_empty() {
                let mut results = Vec::with_capacity(attempted);
                let gpu_results =
                    self.proof_verify_batch_with_gpu_message_scalars(kernel, gpu_reqs, fail_fast);
                for item in gpu_results.results {
                    if let Some(&index) = gpu_indexes.get(item.index) {
                        results.push(BbsBatchItemResult {
                            index,
                            result: item.result,
                        });
                    }
                }
                let cpu_results = run_bbs_batch(cpu_reqs, fail_fast, |req| {
                    self.reference_fallback("ProofVerify", |fallback| fallback.proof_verify(&req))
                });
                for item in cpu_results.results {
                    if let Some(&index) = cpu_indexes.get(item.index) {
                        results.push(BbsBatchItemResult {
                            index,
                            result: item.result,
                        });
                    }
                }

                results.sort_by_key(|item| item.index);
                if fail_fast {
                    if let Some(first_error) = results.iter().position(|item| item.result.is_err())
                    {
                        results.truncate(first_error + 1);
                    }
                }
                return batch_run_result(attempted, results);
            }
        }

        let flat_messages = reqs
            .iter()
            .flat_map(|req| req.disclosed_messages.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let flat_scalars =
            match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                Ok(scalars) => scalars,
                Err(error) => return batch_call_error(attempted, error, fail_fast),
            };

        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.disclosed_messages.len();
            let Some(scalars) = flat_scalars.get(scalar_cursor..scalar_end) else {
                return batch_call_error(
                    attempted,
                    Error::Gpu("GPU disclosed message scalar count mismatch".to_owned()),
                    fail_fast,
                );
            };
            prepared.push((index, req, scalars.to_vec()));
            scalar_cursor = scalar_end;
        }

        if kernel.can_g1_msm()
            && prepared
                .iter()
                .all(|(_, req, _)| proof_verify_gpu_msm_enabled(req))
        {
            let (all_terms, metas) =
                match prepare_proof_verify_msm_terms_batch(self.protocol, kernel, &prepared) {
                    Ok(prepared_terms) => prepared_terms,
                    Err(error) => return batch_call_error(attempted, error, fail_fast),
                };
            let precompute_pairing = proof_verify_pairing_overlap_enabled(prepared.len())
                && kernel.can_pairing_eq_aggregate()
                && pairing_aggregate_enabled(prepared.len());
            let (msm, precomputed_pairing) = if precompute_pairing {
                let pre_pairing_inputs = metas
                    .iter()
                    .map(|(_, meta)| meta.pairing_input.clone())
                    .collect::<Vec<_>>();
                thread::scope(|scope| {
                    let pairing_handle =
                        scope.spawn(|| kernel.pairing_eq_aggregate(&pre_pairing_inputs, fail_fast));
                    let msm = g1_msm_batch_proof_verify_pairs(kernel, &all_terms, fail_fast);
                    let pairing = pairing_handle
                        .join()
                        .expect("proof-verify aggregate pairing worker panicked");
                    (msm, pairing)
                })
            } else {
                (
                    g1_msm_batch_proof_verify_pairs(kernel, &all_terms, fail_fast),
                    None,
                )
            };
            if msm.failed == 0 && msm.results.len() == prepared.len() * 2 {
                let aggregate_items = prepared
                    .iter()
                    .zip(msm.results.chunks_exact(2))
                    .zip(metas.iter())
                    .map(|(((index, req, scalars), pair), (domain, meta))| {
                        let t1 = pair[0].result.as_ref().map_err(Clone::clone)?;
                        let t2 = pair[1].result.as_ref().map_err(Clone::clone)?;
                        Ok((
                            *index,
                            req.clone(),
                            scalars.clone(),
                            t1.clone(),
                            t2.clone(),
                            *domain,
                            meta.clone(),
                        ))
                    })
                    .collect::<Result<Vec<_>>>();
                if let Ok(aggregate_items) = aggregate_items {
                    if force_gpu_execution() {
                        return proof_verify_precomputed_items_strict_gpu(
                            kernel,
                            attempted,
                            &aggregate_items,
                            fail_fast,
                        );
                    }
                    if kernel.can_pairing_eq() && native_pairing_enabled() {
                        if kernel.can_precompute_message_scalars() && gpu_challenge_scalar_enabled()
                        {
                            let challenge_and_pairing_inputs = aggregate_items
                                .iter()
                                .map(|(_, req, scalars, t1, t2, domain, meta)| {
                                    let challenge_input =
                                        proof_verify_challenge_input_from_prepared(
                                            req, scalars, t1, t2, *domain, meta,
                                        )?;
                                    let pairing_input = meta.pairing_input.clone();
                                    Ok((req, challenge_input, pairing_input))
                                })
                                .collect::<Result<Vec<_>>>();
                            if let Ok(challenge_and_pairing_inputs) = challenge_and_pairing_inputs {
                                let challenge_inputs = challenge_and_pairing_inputs
                                    .iter()
                                    .map(|(_, challenge_input, _)| challenge_input.clone())
                                    .collect::<Vec<_>>();
                                if let Ok(challenges) = gpu_precompute_challenge_scalars(
                                    kernel,
                                    challenge_inputs,
                                    fail_fast,
                                ) {
                                    if challenges.len() == challenge_and_pairing_inputs.len()
                                        && challenge_and_pairing_inputs
                                            .iter()
                                            .zip(challenges.iter())
                                            .all(|((req, _, _), challenge)| {
                                                self.protocol
                                                    .proof_verify_prehashed_disclosed_scalars_and_msm_challenge_scalar(
                                                        req, challenge.clone(),
                                                    )
                                                    .is_ok()
                                            })
                                    {
                                        let pairing_inputs = challenge_and_pairing_inputs
                                            .into_iter()
                                            .map(|(_, _, pairing_input)| pairing_input)
                                            .collect::<Vec<_>>();
                                        if let Some(pairing_result) = &precomputed_pairing {
                                            return pairing_result.clone();
                                        }
                                        if kernel.can_pairing_eq_aggregate()
                                            && pairing_aggregate_enabled(pairing_inputs.len())
                                        {
                                            if let Some(pairing_result) = kernel
                                                .pairing_eq_aggregate(&pairing_inputs, fail_fast)
                                            {
                                                return pairing_result;
                                            }
                                        }
                                        let pairing_result =
                                            kernel.pairing_eq_batch(&pairing_inputs, fail_fast);
                                        if pairing_result.failed == 0 {
                                            return pairing_result;
                                        }
                                    }
                                }
                            }
                        }
                        let pairing_inputs = aggregate_items
                            .iter()
                            .map(|(_, req, scalars, t1, t2, domain, meta)| {
                                proof_verify_challenge_and_pairing_eq_input_from_prepared(
                                    req, scalars, t1, t2, *domain, meta,
                                )
                            })
                            .collect::<Result<Vec<_>>>();
                        if let Ok(pairing_inputs) = pairing_inputs {
                            if let Some(pairing_result) = &precomputed_pairing {
                                return pairing_result.clone();
                            }
                            if kernel.can_pairing_eq_aggregate()
                                && pairing_aggregate_enabled(pairing_inputs.len())
                            {
                                if let Some(pairing_result) =
                                    kernel.pairing_eq_aggregate(&pairing_inputs, fail_fast)
                                {
                                    return pairing_result;
                                }
                            }
                            let pairing_result =
                                kernel.pairing_eq_batch(&pairing_inputs, fail_fast);
                            if pairing_result.failed == 0 {
                                return pairing_result;
                            }
                        }
                    }
                    if force_gpu_execution() {
                        return batch_call_error(
                            attempted,
                            Error::Gpu("GPU ProofVerify pairing failed; CPU pairing fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                            fail_fast,
                        );
                    }
                    if !fail_fast
                        && aggregate_items.len() >= proof_verify_parallel_pairing_min_batch()
                    {
                        return proof_verify_precomputed_msm_items_parallel(
                            self.protocol,
                            attempted,
                            aggregate_items,
                        );
                    }
                    match self
                        .protocol
                        .proof_verify_prehashed_disclosed_scalars_and_msm_batch_aggregate(
                            &aggregate_items
                                .iter()
                                .map(|(_, req, scalars, t1, t2, domain, _meta)| {
                                    (
                                        req.clone(),
                                        scalars.clone(),
                                        t1.clone(),
                                        t2.clone(),
                                        *domain,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        ) {
                        Ok(true) => {
                            let results = prepared
                                .into_iter()
                                .map(|(index, _, _)| BbsBatchItemResult {
                                    index,
                                    result: Ok(true),
                                })
                                .collect::<Vec<_>>();
                            return batch_run_result(attempted, results);
                        }
                        Ok(false) | Err(_) => {}
                    }
                }

                if force_gpu_execution() {
                    return batch_call_error(
                        attempted,
                        Error::Gpu("GPU ProofVerify pipeline failed; CPU verifier fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                        fail_fast,
                    );
                }

                let mut results = Vec::with_capacity(attempted);
                let mut msm_results = msm.results.into_iter();
                for (index, req, scalars) in prepared.into_iter() {
                    let t1 = msm_results
                        .next()
                        .ok_or_else(|| Error::Gpu("proof verify T1 MSM result missing".to_owned()));
                    let t2 = msm_results
                        .next()
                        .ok_or_else(|| Error::Gpu("proof verify T2 MSM result missing".to_owned()));
                    let result = t1
                        .and_then(|item| item.result)
                        .and_then(|t1| t2.and_then(|item| item.result).map(|t2| (t1, t2)))
                        .and_then(|(t1, t2)| {
                            self.protocol
                                .proof_verify_prehashed_disclosed_scalars_and_msm(
                                    req, scalars, t1, t2,
                                )
                        });
                    let is_ok = result.is_ok();
                    results.push(BbsBatchItemResult { index, result });
                    if fail_fast && !is_ok {
                        break;
                    }
                }
                return batch_run_result(attempted, results);
            }
        }

        if force_gpu_execution() {
            return batch_call_error(
                attempted,
                Error::Gpu("GPU ProofVerify MSM pipeline is unavailable; CPU verifier fallback is disabled by CUBBS_BBS_FORCE_GPU".to_owned()),
                fail_fast,
            );
        }

        let mut results = Vec::with_capacity(attempted);
        for (index, req, scalars) in prepared {
            let result = self
                .protocol
                .proof_verify_prehashed_disclosed_scalars(req, scalars);
            let is_ok = result.is_ok();
            results.push(BbsBatchItemResult { index, result });
            if fail_fast && !is_ok {
                break;
            }
        }

        batch_run_result(attempted, results)
    }

    pub fn sign_batch(
        &self,
        reqs: Vec<SignRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match &self.kernel {
            Some(kernel) if kernel.can_precompute_message_scalars() => {
                let plan = planner::CuBbsPlanner::plan_sign(&reqs);
                if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                    return batch_call_error(reqs.len(), error, fail_fast);
                }
                if sign_shape_batching_enabled() {
                    if let Some(result) = self.sign_batch_with_gpu_shape_batching(
                        kernel,
                        reqs.clone(),
                        &plan,
                        fail_fast,
                    ) {
                        return result;
                    }
                }
                self.sign_batch_with_gpu_message_scalars(kernel, reqs, fail_fast)
            }
            None => run_bbs_batch(reqs, fail_fast, |req| self.sign(req)),
            Some(_) => run_bbs_batch(reqs, fail_fast, |req| self.sign(req)),
        }
    }

    pub(super) fn sign_batch_repeated(
        &self,
        req: &SignRequest,
        count: usize,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match &self.kernel {
            Some(kernel) if kernel.can_precompute_message_scalars() && kernel.can_g1_msm() => {
                let plan = planner::CuBbsPlanner::plan_sign_repeated(req, count);
                if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                    return batch_call_error(count, error, fail_fast);
                }
                self.sign_batch_repeated_with_gpu_message_scalars(kernel, req, count, fail_fast)
            }
            _ => run_bbs_batch(vec![req.clone(); count], fail_fast, |req| self.sign(req)),
        }
    }

    pub(super) fn sign_batch_with_gpu_shape_batching(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: Vec<SignRequest>,
        plan: &planner::ExecutionPlan,
        fail_fast: bool,
    ) -> Option<BbsBatchRunResult<Vec<u8>>> {
        let attempted = reqs.len();
        if attempted < proof_bbs_mixed_shape_min_batch() {
            return None;
        }
        if plan.batches.len() < 2 || plan.batches.len() > 3 {
            return None;
        }
        if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
            return Some(batch_call_error(attempted, error, fail_fast));
        }

        let mut requests = reqs.into_iter().map(Some).collect::<Vec<_>>();
        let buckets = plan
            .batches
            .iter()
            .cloned()
            .map(|planned| {
                let indexes = planned.request_indices;
                let batch = indexes
                    .iter()
                    .map(|&index| {
                        requests[index]
                            .take()
                            .expect("planner emitted each sign request exactly once")
                    })
                    .collect::<Vec<_>>();
                (indexes, batch)
            })
            .collect::<Vec<_>>();
        let results = thread::scope(|scope| {
            buckets
                .into_iter()
                .map(|(indexes, batch)| {
                    scope.spawn(move || {
                        let bucket =
                            self.sign_batch_with_gpu_message_scalars(kernel, batch, fail_fast);
                        bucket
                            .results
                            .into_iter()
                            .filter_map(|item| {
                                indexes
                                    .get(item.index)
                                    .copied()
                                    .map(|index| BbsBatchItemResult {
                                        index,
                                        result: item.result,
                                    })
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|worker| worker.join().expect("sign shape worker panicked"))
                .collect::<Vec<_>>()
        });
        Some(merge_batch_results_with_fail_fast(
            attempted, results, fail_fast, None,
        ))
    }

    pub fn verify_batch(
        &self,
        reqs: Vec<VerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        self.verify_batch_borrowed(&reqs, fail_fast)
    }

    pub fn verify_batch_borrowed(
        &self,
        reqs: &[VerifyRequest],
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        match &self.kernel {
            Some(kernel) if kernel.can_precompute_message_scalars() => {
                let plan = planner::CuBbsPlanner::plan_verify(reqs);
                if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                    return batch_call_error(reqs.len(), error, fail_fast);
                }
                self.verify_batch_with_gpu_message_scalars(kernel, reqs, &plan, fail_fast)
            }
            _ => run_bbs_batch(reqs.to_vec(), fail_fast, |req| self.verify(req)),
        }
    }

    pub(super) fn verify_batch_repeated(
        &self,
        req: &VerifyRequest,
        count: usize,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        match &self.kernel {
            Some(kernel) if kernel.can_precompute_message_scalars() => {
                let plan = planner::CuBbsPlanner::plan_verify_repeated(req, count);
                if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                    return batch_call_error(count, error, fail_fast);
                }
                self.verify_batch_repeated_with_gpu_message_scalars(
                    kernel, req, count, &plan, fail_fast,
                )
            }
            _ => run_bbs_batch(vec![req.clone(); count], fail_fast, |req| self.verify(req)),
        }
    }

    pub fn proof_gen_batch(
        &self,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match &self.kernel {
            Some(kernel) if kernel.can_precompute_message_scalars() => {
                let plan = planner::CuBbsPlanner::plan_proof_gen(&reqs);
                if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                    return batch_call_error(reqs.len(), error, fail_fast);
                }
                if proof_bbs_mixed_shape_batching_enabled() {
                    if proof_gen_shape_batching_eligible(&reqs) {
                        return self.proof_gen_batch_with_gpu_shape_batching(
                            kernel, reqs, plan, fail_fast,
                        );
                    }
                }
                self.proof_gen_batch_with_gpu_message_scalars(kernel, reqs, fail_fast)
            }
            _ => run_bbs_batch(reqs, fail_fast, |req| self.proof_gen(req)),
        }
    }

    pub fn proof_verify_batch(
        &self,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        match &self.kernel {
            Some(kernel) if kernel.can_precompute_message_scalars() => {
                let plan = planner::CuBbsPlanner::plan_proof_verify(&reqs);
                if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
                    return batch_call_error(reqs.len(), error, fail_fast);
                }
                if kernel.can_precompute_message_scalars() && kernel.can_g1_msm() {
                    return self.profile_proof_verify_batch(reqs, fail_fast).result;
                }
                if proof_bbs_mixed_shape_batching_enabled() {
                    if let Some(result) =
                        self.proof_verify_batch_with_shape_batching(kernel, reqs.clone(), fail_fast)
                    {
                        return result;
                    }
                }
                self.proof_verify_batch_with_gpu_message_scalars(kernel, reqs, fail_fast)
            }
            _ => run_bbs_batch(reqs, fail_fast, |req| self.proof_verify(req)),
        }
    }

    pub(super) fn proof_gen_batch_with_gpu_shape_batching(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: Vec<ProofGenRequest>,
        plan: planner::ExecutionPlan,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let attempted = reqs.len();
        let shared_public_key = reqs
            .first()
            .is_none_or(|first| reqs.iter().all(|req| req.public_key == first.public_key));

        debug_assert!(plan.batches.len() >= proof_bbs_mixed_shape_min_groups());
        if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
            return batch_call_error(attempted, error, fail_fast);
        }

        let mut requests = reqs.into_iter().map(Some).collect::<Vec<_>>();
        let mut buckets = plan
            .batches
            .into_iter()
            .map(|planned| {
                let indexes = planned.request_indices;
                let batch = indexes
                    .iter()
                    .map(|&index| {
                        requests[index]
                            .take()
                            .expect("planner emitted each proof-gen request exactly once")
                    })
                    .collect::<Vec<_>>();
                (indexes, batch)
            })
            .collect::<Vec<(Vec<usize>, Vec<ProofGenRequest>)>>();

        buckets.sort_by_key(|(indexes, _)| indexes.iter().copied().min().unwrap_or_default());

        let parallel_streams =
            proof_gen_mixed_shape_parallel_streams(if shared_public_key { 3 } else { 4 });
        if parallel_streams <= 1 {
            let mut results = Vec::with_capacity(attempted);
            let mut first_error_index = None;
            for (indexes, batch) in buckets {
                let bucket_result =
                    self.proof_gen_batch_with_gpu_message_scalars(kernel, batch, fail_fast);
                for item in bucket_result.results {
                    if let Some(&original_index) = indexes.get(item.index) {
                        let is_error = item.result.is_err();
                        results.push(BbsBatchItemResult {
                            index: original_index,
                            result: item.result,
                        });
                        if is_error {
                            first_error_index =
                                Some(original_index.min(first_error_index.unwrap_or(attempted)));
                        }
                    }
                }
                if fail_fast && first_error_index.is_some() {
                    break;
                }
            }

            return merge_batch_results_with_fail_fast(
                attempted,
                results,
                fail_fast && first_error_index.is_some(),
                first_error_index,
            );
        }

        let stream_count = parallel_streams.min(buckets.len());
        let queue = Arc::new(Mutex::new(VecDeque::from(buckets)));
        let first_error_index = Arc::new(AtomicUsize::new(attempted + 1));

        let results = thread::scope(|scope| {
            (0..stream_count)
                .map(|_| {
                    let queue = Arc::clone(&queue);
                    let first_error_index = Arc::clone(&first_error_index);
                    scope.spawn(move || {
                        let mut bucket_results = Vec::new();
                        loop {
                            let job = {
                                let mut queue_guard = queue.lock().unwrap();
                                queue_guard.pop_front()
                            };
                            let Some((indexes, batch)) = job else {
                                break;
                            };

                            let bucket_result = self
                                .proof_gen_batch_with_gpu_message_scalars(kernel, batch, fail_fast);
                            let mut local_first_error = None;
                            for item in bucket_result.results {
                                if let Some(&original_index) = indexes.get(item.index) {
                                    let is_error = item.result.is_err();
                                    bucket_results.push(BbsBatchItemResult {
                                        index: original_index,
                                        result: item.result,
                                    });
                                    if is_error {
                                        local_first_error = Some(original_index);
                                    }
                                }
                            }

                            if let Some(error_index) = local_first_error {
                                let mut observed = first_error_index.load(Ordering::Acquire);
                                while error_index < observed {
                                    match first_error_index.compare_exchange_weak(
                                        observed,
                                        error_index,
                                        Ordering::Release,
                                        Ordering::Acquire,
                                    ) {
                                        Ok(_) => break,
                                        Err(next) => observed = next,
                                    }
                                }
                            }
                        }
                        bucket_results
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .expect("proof-gen mixed-shape worker panicked")
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        });

        let first_error_index = {
            let observed = first_error_index.load(Ordering::Acquire);
            (observed != attempted + 1).then_some(observed)
        };

        merge_batch_results_with_fail_fast(
            attempted,
            results,
            fail_fast && first_error_index.is_some(),
            first_error_index,
        )
    }

    pub(super) fn proof_verify_batch_with_shape_batching(
        &self,
        kernel: &GpuKernelLibrary,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> Option<BbsBatchRunResult<bool>> {
        let attempted = reqs.len();
        let shared_public_key = reqs
            .first()
            .is_none_or(|first| reqs.iter().all(|req| req.public_key == first.public_key));
        if attempted < proof_bbs_mixed_shape_min_batch() {
            return None;
        }
        if !proof_verify_requests_are_mixed(&reqs) {
            return None;
        }

        let plan = planner::CuBbsPlanner::plan_proof_verify(&reqs);
        if plan.batches.len() < proof_bbs_mixed_shape_min_groups() {
            return None;
        }
        if let Some(error) = gpu_runtime_plan_error(kernel, &plan) {
            return Some(batch_call_error(attempted, error, fail_fast));
        }

        let mut requests = reqs.into_iter().map(Some).collect::<Vec<_>>();
        let mut buckets = plan
            .batches
            .into_iter()
            .map(|planned| {
                let indexes = planned.request_indices;
                let batch = indexes
                    .iter()
                    .map(|&index| {
                        requests[index]
                            .take()
                            .expect("planner emitted each proof-verify request exactly once")
                    })
                    .collect::<Vec<_>>();
                (indexes, batch)
            })
            .collect::<Vec<(Vec<usize>, Vec<ProofVerifyRequest>)>>();

        buckets.sort_by_key(|(indexes, _)| indexes.iter().copied().min().unwrap_or_default());

        let parallel_streams =
            proof_verify_mixed_shape_parallel_streams(if shared_public_key { 3 } else { 4 });
        if parallel_streams <= 1 {
            let mut results = Vec::with_capacity(attempted);
            let mut first_error_index = None;
            for (indexes, batch) in buckets {
                let bucket_result =
                    self.proof_verify_batch_with_gpu_message_scalars(kernel, batch, fail_fast);
                for item in &bucket_result.results {
                    if let Some(&original_index) = indexes.get(item.index) {
                        results.push(BbsBatchItemResult {
                            index: original_index,
                            result: item.result.clone(),
                        });
                    }
                }

                if fail_fast {
                    if let Some((index, _)) = indexes
                        .iter()
                        .cloned()
                        .zip(bucket_result.results.iter())
                        .find_map(|(index, item)| item.result.is_err().then_some((index, ())))
                    {
                        first_error_index = Some(index.min(first_error_index.unwrap_or(attempted)));
                        break;
                    }
                }
            }

            return Some(merge_batch_results_with_fail_fast(
                attempted,
                results,
                fail_fast && first_error_index.is_some(),
                first_error_index,
            ));
        }

        let stream_count = parallel_streams.min(buckets.len());
        let queue = Arc::new(Mutex::new(VecDeque::from(buckets)));
        let first_error_index = Arc::new(AtomicUsize::new(attempted + 1));

        let results = thread::scope(|scope| {
            (0..stream_count)
                .map(|_| {
                    let queue = Arc::clone(&queue);
                    let first_error_index = Arc::clone(&first_error_index);
                    scope.spawn(move || {
                        let mut bucket_results = Vec::new();
                        loop {
                            let job = {
                                let mut queue_guard = queue.lock().unwrap();
                                queue_guard.pop_front()
                            };
                            let Some((indexes, batch)) = job else {
                                break;
                            };

                            let bucket_result = self.proof_verify_batch_with_gpu_message_scalars(
                                kernel, batch, fail_fast,
                            );
                            let mut local_first_error = None;
                            for item in &bucket_result.results {
                                if let Some(&original_index) = indexes.get(item.index) {
                                    bucket_results.push(BbsBatchItemResult {
                                        index: original_index,
                                        result: item.result.clone(),
                                    });
                                    if item.result.is_err() {
                                        local_first_error = Some(original_index);
                                    }
                                }
                            }

                            if let Some(error_index) = local_first_error {
                                let mut observed = first_error_index.load(Ordering::Acquire);
                                while error_index < observed {
                                    match first_error_index.compare_exchange_weak(
                                        observed,
                                        error_index,
                                        Ordering::Release,
                                        Ordering::Acquire,
                                    ) {
                                        Ok(_) => break,
                                        Err(next) => observed = next,
                                    }
                                }
                            }
                        }
                        bucket_results
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .expect("proof-verify mixed-shape worker panicked")
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        });

        let first_error_index = {
            let observed = first_error_index.load(Ordering::Acquire);
            (observed != attempted + 1).then_some(observed)
        };

        Some(merge_batch_results_with_fail_fast(
            attempted,
            results,
            fail_fast && first_error_index.is_some(),
            first_error_index,
        ))
    }

    pub fn profile_proof_gen_batch(
        &self,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsProofGenBatchProfile {
        let total_started = Instant::now();
        let attempted = reqs.len();
        let Some(kernel) = &self.kernel else {
            let result = if force_gpu_execution() {
                batch_call_error(
                    attempted,
                    Error::HardwareGpuUnavailable(
                        "CUBBS_BBS_FORCE_GPU requires the GPU ProofGen pipeline".to_owned(),
                    ),
                    fail_fast,
                )
            } else {
                run_bbs_batch(reqs, fail_fast, |req| self.fallback.proof_gen(&req))
            };
            return BbsProofGenBatchProfile {
                result,
                message_scalar_ms: 0.0,
                b_terms_ms: 0.0,
                b_msm_ms: 0.0,
                t2_prepare_ms: 0.0,
                t2_msm_ms: 0.0,
                finalize_ms: elapsed_ms(total_started),
                total_ms: elapsed_ms(total_started),
            };
        };
        if !kernel.can_precompute_message_scalars() {
            let result = self.proof_gen_batch(reqs, fail_fast);
            return BbsProofGenBatchProfile {
                result,
                message_scalar_ms: 0.0,
                b_terms_ms: 0.0,
                b_msm_ms: 0.0,
                t2_prepare_ms: 0.0,
                t2_msm_ms: 0.0,
                finalize_ms: elapsed_ms(total_started),
                total_ms: elapsed_ms(total_started),
            };
        }

        let scalar_started = Instant::now();
        let flat_messages = reqs
            .iter()
            .flat_map(|req| req.messages.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let flat_scalars =
            match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                Ok(scalars) => scalars,
                Err(error) => {
                    let result = batch_call_error(attempted, error, fail_fast);
                    return BbsProofGenBatchProfile {
                        result,
                        message_scalar_ms: elapsed_ms(scalar_started),
                        b_terms_ms: 0.0,
                        b_msm_ms: 0.0,
                        t2_prepare_ms: 0.0,
                        t2_msm_ms: 0.0,
                        finalize_ms: 0.0,
                        total_ms: elapsed_ms(total_started),
                    };
                }
            };
        let message_scalar_ms = elapsed_ms(scalar_started);

        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.messages.len();
            let Some(scalars) = flat_scalars.get(scalar_cursor..scalar_end) else {
                let result = batch_call_error(
                    attempted,
                    Error::Gpu("GPU message scalar count mismatch".to_owned()),
                    fail_fast,
                );
                return BbsProofGenBatchProfile {
                    result,
                    message_scalar_ms,
                    b_terms_ms: 0.0,
                    b_msm_ms: 0.0,
                    t2_prepare_ms: 0.0,
                    t2_msm_ms: 0.0,
                    finalize_ms: 0.0,
                    total_ms: elapsed_ms(total_started),
                };
            };
            prepared.push((index, req, scalars.to_vec()));
            scalar_cursor = scalar_end;
        }

        if !kernel.can_g1_msm() {
            let finalize_started = Instant::now();
            let result = run_bbs_batch(
                prepared
                    .into_iter()
                    .map(|(_, req, scalars)| (req, scalars))
                    .collect::<Vec<_>>(),
                fail_fast,
                |(req, scalars)| {
                    self.protocol
                        .proof_gen_prehashed_message_scalars(req, scalars)
                },
            );
            return BbsProofGenBatchProfile {
                result,
                message_scalar_ms,
                b_terms_ms: 0.0,
                b_msm_ms: 0.0,
                t2_prepare_ms: 0.0,
                t2_msm_ms: 0.0,
                finalize_ms: elapsed_ms(finalize_started),
                total_ms: elapsed_ms(total_started),
            };
        }

        let b_terms_started = Instant::now();
        let terms = match prepared
            .iter()
            .map(|(_, req, scalars)| {
                let terms_req = SignRequest {
                    secret_key: Vec::new(),
                    public_key: req.public_key.clone(),
                    header: req.header.clone(),
                    messages: req.messages.clone(),
                };
                self.prepare_b_msm_terms_for_prehashed_message_scalars_cached(&terms_req, scalars)
            })
            .collect::<Result<Vec<_>>>()
        {
            Ok(terms) => terms,
            Err(error) => {
                let result = batch_call_error(attempted, error, fail_fast);
                return BbsProofGenBatchProfile {
                    result,
                    message_scalar_ms,
                    b_terms_ms: elapsed_ms(b_terms_started),
                    b_msm_ms: 0.0,
                    t2_prepare_ms: 0.0,
                    t2_msm_ms: 0.0,
                    finalize_ms: 0.0,
                    total_ms: elapsed_ms(total_started),
                };
            }
        };
        let b_terms_ms = elapsed_ms(b_terms_started);

        let all_t2_gpu = prepared
            .iter()
            .all(|(_, req, _)| proof_gen_gpu_t2_msm_enabled(req.messages.len()));
        let mut overlapped_domains = None;
        let b_msm_started = Instant::now();
        let prepared_b = if all_t2_gpu && proof_gen_overlap_domain_enabled() {
            thread::scope(|scope| {
                let domain_terms = terms.clone();
                let domains_handle = scope.spawn(move || precompute_b_msm_domains(domain_terms));
                let prepared_b =
                    g1_msm_batch_grouped_by_points_for_proof_gen_b(kernel, &terms, fail_fast);
                if prepared_b.failed == 0 {
                    if let Ok(Ok(domains)) = domains_handle.join() {
                        overlapped_domains = Some(domains);
                    }
                } else {
                    let _ = domains_handle.join();
                }
                prepared_b
            })
        } else {
            g1_msm_batch_grouped_by_points_for_proof_gen_b(kernel, &terms, fail_fast)
        };
        let b_msm_ms = elapsed_ms(b_msm_started);
        if prepared_b.failed != 0 || prepared_b.results.len() != prepared.len() {
            let result = batch_call_error(
                attempted,
                Error::Gpu("proof-gen B GPU MSM failed".to_owned()),
                fail_fast,
            );
            return BbsProofGenBatchProfile {
                result,
                message_scalar_ms,
                b_terms_ms,
                b_msm_ms,
                t2_prepare_ms: 0.0,
                t2_msm_ms: 0.0,
                finalize_ms: 0.0,
                total_ms: elapsed_ms(total_started),
            };
        }

        if all_t2_gpu {
            let t2_prepare_started = Instant::now();
            let (states_and_items, t2_terms) =
                match prepare_proof_gen_t2_msm_terms_batch_with_domains(
                    self.protocol,
                    prepared,
                    prepared_b.results,
                    terms,
                    overlapped_domains,
                ) {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        let result = batch_call_error(attempted, error, fail_fast);
                        return BbsProofGenBatchProfile {
                            result,
                            message_scalar_ms,
                            b_terms_ms,
                            b_msm_ms,
                            t2_prepare_ms: elapsed_ms(t2_prepare_started),
                            t2_msm_ms: 0.0,
                            finalize_ms: 0.0,
                            total_ms: elapsed_ms(total_started),
                        };
                    }
                };
            let t2_prepare_ms = elapsed_ms(t2_prepare_started);

            let t2_msm_started = Instant::now();
            let t2_results =
                g1_msm_batch_split_variable_term_grouped(kernel, &t2_terms, 0, fail_fast)
                    .unwrap_or_else(|| {
                        g1_msm_batch_grouped_by_points(kernel, &t2_terms, fail_fast)
                    });
            let t2_msm_ms = elapsed_ms(t2_msm_started);
            if t2_results.failed == 0 && t2_results.results.len() == states_and_items.len() {
                let finalize_started = Instant::now();
                let result = finalize_proof_gen_t2_batch(
                    self.protocol,
                    Some(kernel),
                    attempted,
                    states_and_items,
                    t2_results.results,
                    fail_fast,
                );
                let finalize_ms = elapsed_ms(finalize_started);
                return BbsProofGenBatchProfile {
                    result,
                    message_scalar_ms,
                    b_terms_ms,
                    b_msm_ms,
                    t2_prepare_ms,
                    t2_msm_ms,
                    finalize_ms,
                    total_ms: elapsed_ms(total_started),
                };
            }
            let result = batch_call_error(
                attempted,
                Error::Gpu("proof-gen T2 GPU MSM failed".to_owned()),
                fail_fast,
            );
            return BbsProofGenBatchProfile {
                result,
                message_scalar_ms,
                b_terms_ms,
                b_msm_ms,
                t2_prepare_ms,
                t2_msm_ms,
                finalize_ms: 0.0,
                total_ms: elapsed_ms(total_started),
            };
        }

        let finalize_started = Instant::now();
        let mut results = Vec::with_capacity(attempted);
        for ((index, req, scalars), prepared_b) in prepared.into_iter().zip(prepared_b.results) {
            let result = prepared_b.result.and_then(|compressed_b| {
                self.protocol
                    .proof_gen_prehashed_message_scalars_and_b(req, scalars, compressed_b)
            });
            let is_ok = result.is_ok();
            results.push(BbsBatchItemResult { index, result });
            if fail_fast && !is_ok {
                break;
            }
        }
        BbsProofGenBatchProfile {
            result: batch_run_result(attempted, results),
            message_scalar_ms,
            b_terms_ms,
            b_msm_ms,
            t2_prepare_ms: 0.0,
            t2_msm_ms: 0.0,
            finalize_ms: elapsed_ms(finalize_started),
            total_ms: elapsed_ms(total_started),
        }
    }

    pub fn profile_proof_verify_batch(
        &self,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsProofVerifyBatchProfile {
        let total_started = Instant::now();
        let attempted = reqs.len();
        let execution_plan = planner::CuBbsPlanner::plan_proof_verify(&reqs);
        debug_assert_eq!(execution_plan.request_count(), attempted);
        let Some(kernel) = &self.kernel else {
            let result = if force_gpu_execution() {
                batch_call_error(
                    attempted,
                    Error::HardwareGpuUnavailable(
                        "CUBBS_BBS_FORCE_GPU requires the GPU ProofVerify pipeline".to_owned(),
                    ),
                    fail_fast,
                )
            } else {
                run_bbs_batch(reqs, fail_fast, |req| self.fallback.proof_verify(&req))
            };
            return BbsProofVerifyBatchProfile {
                result,
                message_scalar_ms: 0.0,
                msm_terms_ms: 0.0,
                msm_ms: 0.0,
                challenge_pairing_input_ms: 0.0,
                pairing_ms: 0.0,
                fallback_finalize_ms: elapsed_ms(total_started),
                total_ms: elapsed_ms(total_started),
            };
        };
        if !kernel.can_precompute_message_scalars() {
            let result = self.proof_verify_batch(reqs, fail_fast);
            return BbsProofVerifyBatchProfile {
                result,
                message_scalar_ms: 0.0,
                msm_terms_ms: 0.0,
                msm_ms: 0.0,
                challenge_pairing_input_ms: 0.0,
                pairing_ms: 0.0,
                fallback_finalize_ms: elapsed_ms(total_started),
                total_ms: elapsed_ms(total_started),
            };
        }

        let scalar_started = Instant::now();
        let flat_messages = reqs
            .iter()
            .flat_map(|req| req.disclosed_messages.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let flat_scalars =
            match gpu_precompute_flat_message_scalars_borrowed(kernel, &flat_messages) {
                Ok(scalars) => scalars,
                Err(error) => {
                    let result = batch_call_error(attempted, error, fail_fast);
                    return BbsProofVerifyBatchProfile {
                        result,
                        message_scalar_ms: elapsed_ms(scalar_started),
                        msm_terms_ms: 0.0,
                        msm_ms: 0.0,
                        challenge_pairing_input_ms: 0.0,
                        pairing_ms: 0.0,
                        fallback_finalize_ms: 0.0,
                        total_ms: elapsed_ms(total_started),
                    };
                }
            };
        let message_scalar_ms = elapsed_ms(scalar_started);

        let mut scalar_cursor = 0usize;
        let mut prepared = Vec::with_capacity(attempted);
        for (index, req) in reqs.into_iter().enumerate() {
            let scalar_end = scalar_cursor + req.disclosed_messages.len();
            let Some(scalars) = flat_scalars.get(scalar_cursor..scalar_end) else {
                let result = batch_call_error(
                    attempted,
                    Error::Gpu("GPU disclosed message scalar count mismatch".to_owned()),
                    fail_fast,
                );
                return BbsProofVerifyBatchProfile {
                    result,
                    message_scalar_ms,
                    msm_terms_ms: 0.0,
                    msm_ms: 0.0,
                    challenge_pairing_input_ms: 0.0,
                    pairing_ms: 0.0,
                    fallback_finalize_ms: 0.0,
                    total_ms: elapsed_ms(total_started),
                };
            };
            prepared.push((index, req, scalars.to_vec()));
            scalar_cursor = scalar_end;
        }

        if !kernel.can_g1_msm()
            || !prepared
                .iter()
                .all(|(_, req, _)| proof_verify_gpu_msm_enabled(req))
        {
            let fallback_started = Instant::now();
            let result = if force_gpu_execution() {
                batch_call_error(
                    attempted,
                    Error::Gpu(
                        "GPU ProofVerify MSM pipeline is unavailable; CPU verifier fallback is disabled by CUBBS_BBS_FORCE_GPU"
                            .to_owned(),
                    ),
                    fail_fast,
                )
            } else {
                run_bbs_batch(prepared, fail_fast, |(_, req, scalars)| {
                    self.protocol
                        .proof_verify_prehashed_disclosed_scalars(req, scalars)
                })
            };
            return BbsProofVerifyBatchProfile {
                result,
                message_scalar_ms,
                msm_terms_ms: 0.0,
                msm_ms: 0.0,
                challenge_pairing_input_ms: 0.0,
                pairing_ms: 0.0,
                fallback_finalize_ms: elapsed_ms(fallback_started),
                total_ms: elapsed_ms(total_started),
            };
        }

        let terms_started = Instant::now();
        let (all_terms, metas) =
            match prepare_proof_verify_msm_terms_batch(self.protocol, kernel, &prepared) {
                Ok(prepared_terms) => prepared_terms,
                Err(error) => {
                    let result = batch_call_error(attempted, error, fail_fast);
                    return BbsProofVerifyBatchProfile {
                        result,
                        message_scalar_ms,
                        msm_terms_ms: elapsed_ms(terms_started),
                        msm_ms: 0.0,
                        challenge_pairing_input_ms: 0.0,
                        pairing_ms: 0.0,
                        fallback_finalize_ms: 0.0,
                        total_ms: elapsed_ms(total_started),
                    };
                }
            };
        let msm_terms_ms = elapsed_ms(terms_started);

        let msm_started = Instant::now();
        let msm = g1_msm_batch_proof_verify_pairs_inner(
            kernel,
            &all_terms,
            fail_fast,
            execution_plan.batches.len() > 1,
        );
        let msm_ms = elapsed_ms(msm_started);
        if msm.failed != 0 || msm.results.len() != prepared.len() * 2 {
            let result = batch_call_error(
                attempted,
                Error::Gpu("proof verify GPU MSM failed".to_owned()),
                fail_fast,
            );
            return BbsProofVerifyBatchProfile {
                result,
                message_scalar_ms,
                msm_terms_ms,
                msm_ms,
                challenge_pairing_input_ms: 0.0,
                pairing_ms: 0.0,
                fallback_finalize_ms: 0.0,
                total_ms: elapsed_ms(total_started),
            };
        }

        let aggregate_items = prepared
            .iter()
            .zip(msm.results.chunks_exact(2))
            .zip(metas.iter())
            .map(|(((index, req, scalars), pair), (domain, meta))| {
                let t1 = pair[0].result.as_ref().map_err(Clone::clone)?;
                let t2 = pair[1].result.as_ref().map_err(Clone::clone)?;
                Ok((
                    *index,
                    req.clone(),
                    scalars.clone(),
                    t1.clone(),
                    t2.clone(),
                    *domain,
                    meta.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>();
        let aggregate_items = match aggregate_items {
            Ok(items) => items,
            Err(error) => {
                let result = batch_call_error(attempted, error, fail_fast);
                return BbsProofVerifyBatchProfile {
                    result,
                    message_scalar_ms,
                    msm_terms_ms,
                    msm_ms,
                    challenge_pairing_input_ms: 0.0,
                    pairing_ms: 0.0,
                    fallback_finalize_ms: 0.0,
                    total_ms: elapsed_ms(total_started),
                };
            }
        };

        let input_started = Instant::now();
        let pairing_inputs = aggregate_items
            .iter()
            .map(|(_, req, scalars, t1, t2, domain, meta)| {
                proof_verify_challenge_and_pairing_eq_input_from_prepared(
                    req, scalars, t1, t2, *domain, meta,
                )
            })
            .collect::<Result<Vec<_>>>();
        let challenge_pairing_input_ms = elapsed_ms(input_started);
        if let Ok(pairing_inputs) = pairing_inputs {
            if kernel.can_pairing_eq() && native_pairing_enabled() {
                let pairing_started = Instant::now();
                let result = if kernel.can_pairing_eq_aggregate()
                    && pairing_aggregate_enabled(pairing_inputs.len())
                {
                    kernel
                        .pairing_eq_aggregate(&pairing_inputs, fail_fast)
                        .unwrap_or_else(|| kernel.pairing_eq_batch(&pairing_inputs, fail_fast))
                } else {
                    kernel.pairing_eq_batch(&pairing_inputs, fail_fast)
                };
                let pairing_ms = elapsed_ms(pairing_started);
                if result.failed == 0 {
                    return BbsProofVerifyBatchProfile {
                        result,
                        message_scalar_ms,
                        msm_terms_ms,
                        msm_ms,
                        challenge_pairing_input_ms,
                        pairing_ms,
                        fallback_finalize_ms: 0.0,
                        total_ms: elapsed_ms(total_started),
                    };
                }
            }
        }

        if force_gpu_execution() {
            let result = batch_call_error(
                attempted,
                Error::Gpu(
                    "GPU ProofVerify pairing failed; CPU pairing fallback is disabled by CUBBS_BBS_FORCE_GPU"
                        .to_owned(),
                ),
                fail_fast,
            );
            return BbsProofVerifyBatchProfile {
                result,
                message_scalar_ms,
                msm_terms_ms,
                msm_ms,
                challenge_pairing_input_ms,
                pairing_ms: 0.0,
                fallback_finalize_ms: 0.0,
                total_ms: elapsed_ms(total_started),
            };
        }

        let fallback_started = Instant::now();
        let result = if !fail_fast
            && aggregate_items.len() >= proof_verify_parallel_pairing_min_batch()
        {
            proof_verify_precomputed_msm_items_parallel(self.protocol, attempted, aggregate_items)
        } else {
            run_bbs_batch(
                aggregate_items,
                fail_fast,
                |(_, req, scalars, t1, t2, _, _)| {
                    self.protocol
                        .proof_verify_prehashed_disclosed_scalars_and_msm(req, scalars, t1, t2)
                },
            )
        };
        BbsProofVerifyBatchProfile {
            result,
            message_scalar_ms,
            msm_terms_ms,
            msm_ms,
            challenge_pairing_input_ms,
            pairing_ms: 0.0,
            fallback_finalize_ms: elapsed_ms(fallback_started),
            total_ms: elapsed_ms(total_started),
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn template_cache_is_bounded_and_does_not_invalidate_readers() {
        let value = Arc::new(BbsMsmTerms {
            compressed_points: Arc::new(vec![vec![1; 48]]),
            scalar_be: vec![vec![2; 32]],
        });
        let key = |n| BbsMsmTermTemplateKey {
            public_key: vec![n; 96],
            header: vec![0; 64],
            message_count: 1,
        };
        let mut cache = MsmTemplateCache::new(usize::MAX);
        cache.insert(key(1), value.clone());
        cache.budget = cache.retained_bytes;
        cache.insert(key(1), value.clone());
        assert_eq!(cache.order.len(), 1);
        cache.insert(key(2), value.clone());
        assert!(!cache.entries.contains_key(&key(1)));
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.retained_bytes <= cache.budget);
        assert_eq!(value.scalar_be[0], vec![2; 32]);
        let mut disabled = MsmTemplateCache::new(0);
        disabled.insert(key(3), value);
        assert!(disabled.entries.is_empty());
    }
}
