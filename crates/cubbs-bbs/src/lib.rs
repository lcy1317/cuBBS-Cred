//! cuBBS-Cred BBS primitive engine built on its modified ZKryptium-derived
//! protocol layer.
//!
//! This crate intentionally does not depend on SpruceID's `ssi-bbs` wrapper or
//! on any W3C/Data Integrity layer. Independent registry versions of ZKryptium
//! are isolated in `reference` for conformance and recoverable fallback.

mod execution;
mod gpu_engine;
mod gpu_runtime;
mod protocol;

use execution::*;
use gpu_runtime::GpuKernelLibrary;
pub use gpu_runtime::GpuMsmBenchPlan;

pub mod gpu_abi;
pub mod planner;
pub mod reference;

use std::{
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, RwLock,
    },
    thread,
    time::Instant,
};

use bls12_381_plus::{group::Curve, G1Affine, G1Projective, G2Projective, Scalar};
use cubbs_bbs_protocol::{
    bbsplus::{
        ciphersuites::{BbsCiphersuite as _, Bls12381Sha256},
        keys::{BBSplusPublicKey, BBSplusSecretKey},
        proof::{
            BBSplusProofGenT2CompressedState, BBSplusProofGenT2MsmTerms, BBSplusProofGenT2State,
            BBSplusProofVerifyMsmTerms, BBSplusProofVerifyPrecomputedItem,
        },
        signature::{BBSplusPrepareBMsmTerms, BBSplusPreparedB},
    },
    keys::pair::KeyPair as ZkKeyPair,
    schemes::{
        algorithms::BBSplus,
        generics::{PoKSignature, Signature as ZkSignature},
    },
    utils::message::bbsplus_message::BBSplusMessage,
    utils::util::bbsplus_utils::{hash_to_scalar, ScalarExt},
};
use libloading::Library;
use rand::RngCore;
use reference::CpuReferenceBackend as _;
use sha2::{Digest, Sha256};

pub const CIPHERSUITE_ID: &str = "BBS_BLS12381G1_XMD:SHA-256_SSWU_RO_";
pub const SECRET_KEY_LEN: usize = 32;
pub const PUBLIC_KEY_LEN: usize = 96;
pub const SIGNATURE_LEN: usize = 80;
pub const BBS2023_HEADER_LEN: usize = 64;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("invalid secret key")]
    InvalidSecretKey,

    #[error("invalid public key")]
    InvalidPublicKey,

    #[error("invalid signature")]
    InvalidSignature,

    #[error("invalid proof")]
    InvalidProof,

    #[error("invalid bbs-2023 header length: expected 64 bytes, got {0}")]
    InvalidHeaderLength(usize),

    #[error("invalid disclosed indexes")]
    InvalidDisclosedIndexes,

    #[error("unknown BBS backend: {0}")]
    UnknownBackend(String),

    #[error("hardware GPU BBS backend is unavailable: {0}")]
    HardwareGpuUnavailable(String),

    #[error("gpu bbs backend error: {0}")]
    Gpu(String),

    #[error("zkryptium error: {0}")]
    Zkryptium(String),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyPair {
    pub secret_key: Vec<u8>,
    pub public_key: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignRequest {
    pub secret_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub header: Vec<u8>,
    pub messages: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyRequest {
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
    pub header: Vec<u8>,
    pub messages: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofGenRequest {
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
    pub header: Vec<u8>,
    pub presentation_header: Vec<u8>,
    pub messages: Vec<Vec<u8>>,
    pub disclosed_indexes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofVerifyRequest {
    pub public_key: Vec<u8>,
    pub proof: Vec<u8>,
    pub header: Vec<u8>,
    pub presentation_header: Vec<u8>,
    pub disclosed_messages: Vec<Vec<u8>>,
    pub disclosed_indexes: Vec<usize>,
    pub total_message_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BbsBatchItemResult<T> {
    pub index: usize,
    pub result: Result<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BbsBatchRunResult<T> {
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub results: Vec<BbsBatchItemResult<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BbsSignProfile {
    pub message_scalar_ms: f64,
    pub generator_ms: f64,
    pub domain_ms: f64,
    pub e_scalar_ms: f64,
    pub msm_and_finalize_ms: f64,
    pub total_ms: f64,
    pub message_count: usize,
    pub signature_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BbsProofGenBatchProfile {
    pub result: BbsBatchRunResult<Vec<u8>>,
    pub message_scalar_ms: f64,
    pub b_terms_ms: f64,
    pub b_msm_ms: f64,
    pub t2_prepare_ms: f64,
    pub t2_msm_ms: f64,
    pub finalize_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BbsProofVerifyBatchProfile {
    pub result: BbsBatchRunResult<bool>,
    pub message_scalar_ms: f64,
    pub msm_terms_ms: f64,
    pub msm_ms: f64,
    pub challenge_pairing_input_ms: f64,
    pub pairing_ms: f64,
    pub fallback_finalize_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BbsMsmShapeSummary {
    pub label: &'static str,
    pub item_count: usize,
    pub min_term_count: usize,
    pub max_term_count: usize,
    pub common_point_groups: usize,
    pub largest_common_point_group: usize,
    pub single_variable_point_index: Option<usize>,
    pub term_count_groups: usize,
    pub split_candidate_groups: usize,
    pub largest_split_candidate_group: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BbsMsmRouteSummary {
    pub unknown: usize,
    pub blst_item: usize,
    pub blst_common_points: usize,
    pub sppark_item: usize,
    pub sppark_common_points: usize,
    pub fused_item: usize,
    pub fused_common_points: usize,
    pub cubbs_large: usize,
    pub cubbs_generic: usize,
}

impl BbsMsmRouteSummary {
    fn record_code(&mut self, code: u8) {
        match code {
            gpu_abi::CUBBS_BBS_MSM_ROUTE_CUBBS_ITEM => self.fused_item += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_CUBBS_COMMON_POINTS => self.fused_common_points += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_CUBBS_LARGE => self.cubbs_large += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL => self.cubbs_generic += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_BLST_ITEM => self.blst_item += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_BLST_COMMON_POINTS => self.blst_common_points += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_SPPARK_ITEM_REFERENCE => self.sppark_item += 1,
            gpu_abi::CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE => {
                self.sppark_common_points += 1
            }
            _ => self.unknown += 1,
        }
    }

    fn merge(&mut self, other: Self) {
        self.unknown += other.unknown;
        self.blst_item += other.blst_item;
        self.blst_common_points += other.blst_common_points;
        self.sppark_item += other.sppark_item;
        self.sppark_common_points += other.sppark_common_points;
        self.fused_item += other.fused_item;
        self.fused_common_points += other.fused_common_points;
        self.cubbs_large += other.cubbs_large;
        self.cubbs_generic += other.cubbs_generic;
    }

    pub fn total(&self) -> usize {
        self.unknown
            + self.blst_item
            + self.blst_common_points
            + self.sppark_item
            + self.sppark_common_points
            + self.fused_item
            + self.fused_common_points
            + self.cubbs_large
            + self.cubbs_generic
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BbsMsmTermsBatch {
    pub label: &'static str,
    pub terms: Vec<BbsMsmTerms>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BbsPreparedB {
    pub compressed_b: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BbsMsmTerms {
    pub compressed_points: Arc<Vec<Vec<u8>>>,
    pub scalar_be: Vec<Vec<u8>>,
}

/// One common-points MSM matrix prepared by a device-ceiling benchmark.
///
/// `scalar_rows_be` is a row-major matrix containing `output_count` rows and
/// `compressed_points.len()` canonical, big-endian 32-byte scalars per row.
/// Construction validates and pins the shape into a [`GpuMsmBenchPlan`], so
/// the timed loop does not clone requests, repack scalars, or allocate output
/// buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuMsmBenchGroupInput {
    pub compressed_points: Vec<Vec<u8>>,
    pub scalar_rows_be: Vec<u8>,
    pub output_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMsmBenchRun {
    pub group_count: usize,
    pub output_count: usize,
}

/// Input for the GPU proof-gen t2Prepare (6 fixed G1 scalar multiplications).
/// The 7 scalars are host-precomputed for six independent base-point muls:
/// [r2, r1*r2, r1*r2, r1*r2*e, r1*r2*e_tilde, r2*r1_tilde, r3_tilde].
#[derive(Debug, Clone)]
struct BbsT2PrepareGpuInput<'a> {
    compressed_b: &'a [u8],
    signature_a: &'a [u8],
    scalars_be: [&'a [u8]; 7],
    seed: &'a [u8],
}

/// Output of the GPU proof-gen t2Prepare: the 4 compressed G1 points.
#[derive(Debug, Clone)]
struct BbsT2PrepareGpuOutput {
    d: Vec<u8>,
    abar: Vec<u8>,
    bbar: Vec<u8>,
    t1: Vec<u8>,
}

/// Pre-computed B2 scalars/points for t2Prepare that are independent of B
/// (generated in a background thread overlapping the GPU B-MSM).
#[derive(Debug, Clone, PartialEq, Eq)]
struct BbsPairingEqInput {
    lhs_g1: Vec<u8>,
    lhs_g2: Vec<u8>,
    rhs_g1: Vec<u8>,
    rhs_g2: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BbsMsmTermTemplateKey {
    public_key: Vec<u8>,
    header: Vec<u8>,
    message_count: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CuBbsProtocolEngine;

/// Backward-compatible public name for callers that selected the old CPU engine.
pub type ZkryptiumBbsEngine = CuBbsProtocolEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BbsBackend {
    ZkryptiumCpu,
    GpuRequestedCpuFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BbsBackendInfo {
    pub requested: &'static str,
    pub active: &'static str,
    pub hardware_acceleration: bool,
    pub fallback: bool,
    pub gpu_abi_version: u32,
    pub detail: &'static str,
}

impl BbsBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" | "zkryptium" | "zkryptium-cpu" => Ok(Self::ZkryptiumCpu),
            "gpu" | "cuda" | "gpu-fallback" | "gpu-requested-cpu-fallback" => {
                Ok(Self::GpuRequestedCpuFallback)
            }
            other => Err(Error::UnknownBackend(other.to_owned())),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::ZkryptiumCpu => "zkryptium-cpu",
            Self::GpuRequestedCpuFallback => "gpu-requested-cpu-fallback",
        }
    }

    pub const fn is_gpu_requested(self) -> bool {
        matches!(self, Self::GpuRequestedCpuFallback)
    }

    pub const fn info(self) -> BbsBackendInfo {
        match self {
            Self::ZkryptiumCpu => BbsBackendInfo {
                requested: "cpu",
                active: "zkryptium-cpu",
                hardware_acceleration: false,
                fallback: false,
                gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                detail: "CPU path backed by the cuBBS-Cred ZKryptium-derived protocol engine",
            },
            Self::GpuRequestedCpuFallback => BbsBackendInfo {
                requested: "gpu",
                active: "zkryptium-cpu",
                hardware_acceleration: false,
                fallback: true,
                gpu_abi_version: gpu_abi::CUBBS_BBS_GPU_ABI_VERSION,
                detail: "GPU backend requested, but CUBBS_BBS_GPU_LIBRARY is not loaded; using CPU fallback",
            },
        }
    }

    pub fn ensure_hardware_available(self) -> Result<()> {
        if self.is_gpu_requested() {
            let kernel = GpuKernelLibrary::load_shared_from_env()?;
            let Some(kernel) = kernel else {
                return Err(Error::HardwareGpuUnavailable(
                    "CUBBS_BBS_GPU_LIBRARY is not set or empty".to_owned(),
                ));
            };
            if !kernel.uses_hardware_acceleration() {
                return Err(Error::HardwareGpuUnavailable(
                    "loaded GPU ABI library did not expose CUDA-backed BBS acceleration".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GpuBbsEngine {
    protocol: CuBbsProtocolEngine,
    fallback: reference::ZkryptiumReference,
    kernel: Option<Arc<GpuKernelLibrary>>,
    b_msm_term_templates: Arc<Mutex<gpu_engine::MsmTemplateCache>>,
}

#[derive(Debug, Clone)]
pub enum BbsEngine {
    ZkryptiumCpu(CuBbsProtocolEngine),
    GpuRequestedCpuFallback(GpuBbsEngine),
}

impl Default for GpuBbsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BbsEngine {
    pub fn new(backend: BbsBackend) -> Self {
        match backend {
            BbsBackend::ZkryptiumCpu => Self::ZkryptiumCpu(CuBbsProtocolEngine::new()),
            BbsBackend::GpuRequestedCpuFallback => {
                Self::GpuRequestedCpuFallback(GpuBbsEngine::new())
            }
        }
    }

    pub fn backend(&self) -> BbsBackend {
        match self {
            Self::ZkryptiumCpu(_) => BbsBackend::ZkryptiumCpu,
            Self::GpuRequestedCpuFallback(_) => BbsBackend::GpuRequestedCpuFallback,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend().name()
    }

    pub fn backend_info(&self) -> BbsBackendInfo {
        match self {
            Self::ZkryptiumCpu(_) => BbsBackend::ZkryptiumCpu.info(),
            Self::GpuRequestedCpuFallback(engine) => engine.backend_info(),
        }
    }

    pub fn generate_keypair(&self, rng: &mut impl RngCore) -> Result<KeyPair> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.generate_keypair(rng),
            Self::GpuRequestedCpuFallback(engine) => engine.generate_keypair(rng),
        }
    }

    pub fn public_key_from_secret(&self, secret_key: &[u8]) -> Result<Vec<u8>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.public_key_from_secret(secret_key),
            Self::GpuRequestedCpuFallback(engine) => engine.public_key_from_secret(secret_key),
        }
    }

    pub fn sign(&self, req: SignRequest) -> Result<Vec<u8>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.sign(req),
            Self::GpuRequestedCpuFallback(engine) => engine.sign(req),
        }
    }

    pub fn profile_sign(&self, req: SignRequest) -> Result<BbsSignProfile> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.profile_sign(req),
            Self::GpuRequestedCpuFallback(engine) => engine.profile_sign(req),
        }
    }

    pub fn prepare_b_for_sign(&self, req: &SignRequest) -> Result<BbsPreparedB> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.prepare_b_for_sign(req),
            Self::GpuRequestedCpuFallback(engine) => engine.prepare_b_for_sign(req),
        }
    }

    pub fn prepare_b_msm_terms_for_sign(&self, req: &SignRequest) -> Result<BbsMsmTerms> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.prepare_b_msm_terms_for_sign(req),
            Self::GpuRequestedCpuFallback(engine) => engine.prepare_b_msm_terms_for_sign(req),
        }
    }

    pub fn prepare_g1_msm_bench_plan(
        &self,
        groups: Vec<GpuMsmBenchGroupInput>,
    ) -> Result<GpuMsmBenchPlan> {
        match self {
            Self::GpuRequestedCpuFallback(engine) => engine.prepare_g1_msm_bench_plan(groups),
            Self::ZkryptiumCpu(_) => Err(Error::Gpu(
                "CPU backend does not expose prepared G1 MSM bench".to_owned(),
            )),
        }
    }

    pub fn g1_msm_batch_for_bench(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match self {
            Self::GpuRequestedCpuFallback(engine) => {
                engine.g1_msm_batch_for_bench(terms, fail_fast)
            }
            Self::ZkryptiumCpu(_) => batch_call_error(
                terms.len(),
                Error::Gpu("CPU backend does not expose raw G1 MSM bench".to_owned()),
                fail_fast,
            ),
        }
    }

    pub fn g1_msm_batch_split_for_bench(
        &self,
        terms: &[BbsMsmTerms],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match self {
            Self::GpuRequestedCpuFallback(engine) => {
                engine.g1_msm_batch_split_for_bench(terms, fail_fast)
            }
            Self::ZkryptiumCpu(_) => batch_call_error(
                terms.len(),
                Error::Gpu("CPU backend does not expose split raw G1 MSM bench".to_owned()),
                fail_fast,
            ),
        }
    }

    pub fn g1_msm_route_summary_for_bench(
        &self,
        terms: &[BbsMsmTerms],
        split: bool,
        fail_fast: bool,
    ) -> Result<BbsMsmRouteSummary> {
        match self {
            Self::GpuRequestedCpuFallback(engine) => {
                engine.g1_msm_route_summary_for_bench(terms, split, fail_fast)
            }
            Self::ZkryptiumCpu(_) => Err(Error::Gpu(
                "CPU backend does not expose G1 MSM route bench".to_owned(),
            )),
        }
    }

    pub fn proof_msm_shapes_for_bench(
        &self,
        proof_gen_reqs: Vec<ProofGenRequest>,
        proof_verify_reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> Result<Vec<BbsMsmShapeSummary>> {
        match self {
            Self::GpuRequestedCpuFallback(engine) => {
                engine.proof_msm_shapes_for_bench(proof_gen_reqs, proof_verify_reqs, fail_fast)
            }
            Self::ZkryptiumCpu(_) => Err(Error::Gpu(
                "CPU backend does not expose proof MSM shape bench".to_owned(),
            )),
        }
    }

    pub fn proof_msm_terms_for_bench(
        &self,
        proof_gen_reqs: Vec<ProofGenRequest>,
        proof_verify_reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> Result<Vec<BbsMsmTermsBatch>> {
        match self {
            Self::GpuRequestedCpuFallback(engine) => {
                engine.proof_msm_terms_for_bench(proof_gen_reqs, proof_verify_reqs, fail_fast)
            }
            Self::ZkryptiumCpu(_) => Err(Error::Gpu(
                "CPU backend does not expose proof MSM terms bench".to_owned(),
            )),
        }
    }

    pub fn messages_to_scalar_bytes(&self, messages: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.messages_to_scalar_bytes(messages),
            Self::GpuRequestedCpuFallback(engine) => engine.messages_to_scalar_bytes(messages),
        }
    }

    pub fn sign_with_prepared_b(
        &self,
        req: SignRequest,
        prepared_b: BbsPreparedB,
    ) -> Result<Vec<u8>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.sign_with_prepared_b(req, prepared_b),
            Self::GpuRequestedCpuFallback(engine) => engine.sign_with_prepared_b(req, prepared_b),
        }
    }

    pub fn verify(&self, req: VerifyRequest) -> Result<bool> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.verify(req),
            Self::GpuRequestedCpuFallback(engine) => engine.verify(req),
        }
    }

    pub fn proof_gen(&self, req: ProofGenRequest) -> Result<Vec<u8>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.proof_gen(req),
            Self::GpuRequestedCpuFallback(engine) => engine.proof_gen(req),
        }
    }

    pub fn proof_verify(&self, req: ProofVerifyRequest) -> Result<bool> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.proof_verify(req),
            Self::GpuRequestedCpuFallback(engine) => engine.proof_verify(req),
        }
    }

    pub fn sign_batch(
        &self,
        reqs: Vec<SignRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.sign_batch(reqs, fail_fast),
            Self::GpuRequestedCpuFallback(engine) => engine.sign_batch(reqs, fail_fast),
        }
    }

    /// Sign structurally cached requests without cloning them in the HTTP
    /// layer. Every entry still executes one BBS Sign operation.
    pub fn sign_batch_shared(
        &self,
        reqs: &[Arc<SignRequest>],
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        let Some(first) = reqs.first() else {
            return batch_run_result(0, Vec::new());
        };
        if reqs.iter().all(|req| Arc::ptr_eq(req, first)) {
            return match self {
                Self::ZkryptiumCpu(engine) => {
                    engine.sign_batch(vec![(**first).clone(); reqs.len()], fail_fast)
                }
                Self::GpuRequestedCpuFallback(engine) => {
                    engine.sign_batch_repeated(first, reqs.len(), fail_fast)
                }
            };
        }
        self.sign_batch(reqs.iter().map(|req| (**req).clone()).collect(), fail_fast)
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
        match self {
            Self::ZkryptiumCpu(engine) => engine.verify_batch(reqs.to_vec(), fail_fast),
            Self::GpuRequestedCpuFallback(engine) => engine.verify_batch_borrowed(reqs, fail_fast),
        }
    }

    /// Verify structurally cached requests without deep-cloning their message
    /// vectors. A homogeneous batch still contains one verification item per
    /// caller; only immutable input storage and message scalars are shared.
    pub fn verify_batch_shared(
        &self,
        reqs: &[Arc<VerifyRequest>],
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        let Some(first) = reqs.first() else {
            return batch_run_result(0, Vec::new());
        };
        if reqs.iter().all(|req| Arc::ptr_eq(req, first)) {
            return match self {
                Self::ZkryptiumCpu(engine) => {
                    engine.verify_batch(vec![(**first).clone(); reqs.len()], fail_fast)
                }
                Self::GpuRequestedCpuFallback(engine) => {
                    engine.verify_batch_repeated(first, reqs.len(), fail_fast)
                }
            };
        }
        self.verify_batch(reqs.iter().map(|req| (**req).clone()).collect(), fail_fast)
    }

    pub fn proof_gen_batch(
        &self,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.proof_gen_batch(reqs, fail_fast),
            Self::GpuRequestedCpuFallback(engine) => engine.proof_gen_batch(reqs, fail_fast),
        }
    }

    pub fn proof_verify_batch(
        &self,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        match self {
            Self::ZkryptiumCpu(engine) => engine.proof_verify_batch(reqs, fail_fast),
            Self::GpuRequestedCpuFallback(engine) => engine.proof_verify_batch(reqs, fail_fast),
        }
    }

    pub fn profile_proof_gen_batch(
        &self,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsProofGenBatchProfile {
        match self {
            Self::ZkryptiumCpu(engine) => {
                let started = Instant::now();
                let result = engine.proof_gen_batch(reqs, fail_fast);
                let total_ms = elapsed_ms(started);
                BbsProofGenBatchProfile {
                    result,
                    message_scalar_ms: 0.0,
                    b_terms_ms: 0.0,
                    b_msm_ms: 0.0,
                    t2_prepare_ms: 0.0,
                    t2_msm_ms: 0.0,
                    finalize_ms: total_ms,
                    total_ms,
                }
            }
            Self::GpuRequestedCpuFallback(engine) => {
                engine.profile_proof_gen_batch(reqs, fail_fast)
            }
        }
    }

    pub fn profile_proof_verify_batch(
        &self,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsProofVerifyBatchProfile {
        match self {
            Self::ZkryptiumCpu(engine) => {
                let started = Instant::now();
                let result = engine.proof_verify_batch(reqs, fail_fast);
                let total_ms = elapsed_ms(started);
                BbsProofVerifyBatchProfile {
                    result,
                    message_scalar_ms: 0.0,
                    msm_terms_ms: 0.0,
                    msm_ms: 0.0,
                    challenge_pairing_input_ms: 0.0,
                    pairing_ms: 0.0,
                    fallback_finalize_ms: total_ms,
                    total_ms,
                }
            }
            Self::GpuRequestedCpuFallback(engine) => {
                engine.profile_proof_verify_batch(reqs, fail_fast)
            }
        }
    }
}

#[cfg(test)]
mod tests;
