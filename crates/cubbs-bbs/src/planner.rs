//! cuBBS-Cred-owned lowering metadata between the protocol layer and GPU runtime.
//!
//! The planner intentionally contains no `zkryptium` types. It describes
//! protocol-compatible batches and the GPU engines they require; concrete
//! scalar, point, and pairing buffers are materialized by the runtime.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::{ProofGenRequest, ProofVerifyRequest, SignRequest, VerifyRequest};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BbsOperation {
    Sign,
    Verify,
    ProofGen,
    ProofVerify,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MsmPointLayout {
    CommonPoints,
    VariablePoints,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionJob {
    HashToScalar {
        message_inputs: usize,
        transcript_inputs: usize,
    },
    Msm {
        layout: MsmPointLayout,
        estimated_terms: usize,
    },
    Pairing {
        equations_per_item: usize,
        public_key_context_is_common: bool,
    },
}

/// Shape fields that must agree before requests share one lowering path.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BatchCompatibilityKey {
    pub operation: BbsOperation,
    pub message_count: usize,
    pub disclosed_indexes: Vec<usize>,
    /// Digest of a header-dependent lowering context; never protocol input.
    pub protocol_context: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchExecutionIr {
    pub key: BatchCompatibilityKey,
    pub request_indices: Vec<usize>,
    pub jobs: Vec<ExecutionJob>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionPlan {
    pub batches: Vec<BatchExecutionIr>,
}

impl ExecutionPlan {
    pub fn request_count(&self) -> usize {
        self.batches
            .iter()
            .map(|batch| batch.request_indices.len())
            .sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct CuBbsPlanner;

impl CuBbsPlanner {
    pub fn plan_sign(requests: &[SignRequest]) -> ExecutionPlan {
        Self::group(
            requests,
            |request| BatchCompatibilityKey {
                operation: BbsOperation::Sign,
                message_count: request.messages.len(),
                disclosed_indexes: Vec::new(),
                protocol_context: Some(context_digest(&request.header)),
            },
            |key, indices| BatchExecutionIr {
                jobs: vec![
                    ExecutionJob::HashToScalar {
                        message_inputs: key.message_count * indices.len(),
                        transcript_inputs: 2 * indices.len(),
                    },
                    ExecutionJob::Msm {
                        layout: MsmPointLayout::CommonPoints,
                        estimated_terms: key.message_count + 2,
                    },
                ],
                key,
                request_indices: indices,
            },
        )
    }

    /// Plan a homogeneous Sign batch without materializing `count` deep
    /// copies of the same structurally cached request.
    pub fn plan_sign_repeated(request: &SignRequest, count: usize) -> ExecutionPlan {
        if count == 0 {
            return ExecutionPlan::default();
        }
        let key = BatchCompatibilityKey {
            operation: BbsOperation::Sign,
            message_count: request.messages.len(),
            disclosed_indexes: Vec::new(),
            protocol_context: Some(context_digest(&request.header)),
        };
        ExecutionPlan {
            batches: vec![BatchExecutionIr {
                jobs: vec![
                    ExecutionJob::HashToScalar {
                        // Immutable messages are hashed once; every item still
                        // computes its own signature transcript scalar.
                        message_inputs: key.message_count,
                        transcript_inputs: 2 * count,
                    },
                    ExecutionJob::Msm {
                        layout: MsmPointLayout::CommonPoints,
                        estimated_terms: key.message_count + 2,
                    },
                ],
                key,
                request_indices: (0..count).collect(),
            }],
        }
    }

    pub fn plan_verify(requests: &[VerifyRequest]) -> ExecutionPlan {
        let shared_public_key = all_equal_by(requests, |request| request.public_key.as_slice());
        Self::group(
            requests,
            |request| BatchCompatibilityKey {
                operation: BbsOperation::Verify,
                message_count: request.messages.len(),
                disclosed_indexes: Vec::new(),
                protocol_context: Some(context_digest(&request.header)),
            },
            move |key, indices| BatchExecutionIr {
                jobs: vec![
                    ExecutionJob::HashToScalar {
                        message_inputs: key.message_count * indices.len(),
                        transcript_inputs: indices.len(),
                    },
                    ExecutionJob::Msm {
                        layout: MsmPointLayout::CommonPoints,
                        estimated_terms: key.message_count + 2,
                    },
                    ExecutionJob::Pairing {
                        equations_per_item: 1,
                        public_key_context_is_common: shared_public_key,
                    },
                ],
                key,
                request_indices: indices,
            },
        )
    }

    /// Build the same plan as `plan_verify` without materializing `count`
    /// deep copies of one structurally cached request.
    pub fn plan_verify_repeated(request: &VerifyRequest, count: usize) -> ExecutionPlan {
        if count == 0 {
            return ExecutionPlan::default();
        }
        let key = BatchCompatibilityKey {
            operation: BbsOperation::Verify,
            message_count: request.messages.len(),
            disclosed_indexes: Vec::new(),
            protocol_context: Some(context_digest(&request.header)),
        };
        ExecutionPlan {
            batches: vec![BatchExecutionIr {
                jobs: vec![
                    ExecutionJob::HashToScalar {
                        // Immutable messages are hashed once for a repeated
                        // request; the aggregate verifier still accounts for
                        // every item below.
                        message_inputs: key.message_count,
                        transcript_inputs: count,
                    },
                    ExecutionJob::Msm {
                        layout: MsmPointLayout::CommonPoints,
                        estimated_terms: key.message_count + 2,
                    },
                    ExecutionJob::Pairing {
                        equations_per_item: 1,
                        public_key_context_is_common: true,
                    },
                ],
                key,
                request_indices: (0..count).collect(),
            }],
        }
    }

    pub fn plan_proof_gen(requests: &[ProofGenRequest]) -> ExecutionPlan {
        Self::group(
            requests,
            |request| BatchCompatibilityKey {
                operation: BbsOperation::ProofGen,
                message_count: request.messages.len(),
                disclosed_indexes: request.disclosed_indexes.clone(),
                protocol_context: None,
            },
            |key, indices| {
                let hidden = key
                    .message_count
                    .saturating_sub(key.disclosed_indexes.len());
                BatchExecutionIr {
                    jobs: vec![
                        ExecutionJob::HashToScalar {
                            message_inputs: key.message_count * indices.len(),
                            transcript_inputs: 2 * indices.len(),
                        },
                        ExecutionJob::Msm {
                            layout: MsmPointLayout::CommonPoints,
                            estimated_terms: key.message_count + 2,
                        },
                        ExecutionJob::Msm {
                            layout: MsmPointLayout::VariablePoints,
                            // T2 = (r1~ * r3~) B + sum_j m_j~ H_j.
                            estimated_terms: hidden + 1,
                        },
                    ],
                    key,
                    request_indices: indices,
                }
            },
        )
    }

    pub fn plan_proof_verify(requests: &[ProofVerifyRequest]) -> ExecutionPlan {
        let shared_public_key = all_equal_by(requests, |request| request.public_key.as_slice());
        Self::group(
            requests,
            |request| BatchCompatibilityKey {
                operation: BbsOperation::ProofVerify,
                message_count: request.total_message_count,
                disclosed_indexes: request.disclosed_indexes.clone(),
                protocol_context: None,
            },
            move |key, indices| {
                BatchExecutionIr {
                    jobs: vec![
                        ExecutionJob::HashToScalar {
                            message_inputs: key.disclosed_indexes.len() * indices.len(),
                            transcript_inputs: 2 * indices.len(),
                        },
                        ExecutionJob::Msm {
                            // T1 = c Bbar + e~ Abar + r1~ D.
                            layout: MsmPointLayout::VariablePoints,
                            estimated_terms: 3,
                        },
                        ExecutionJob::Msm {
                            // The common-generator part of T2 contains
                            // G1, Q1, and one H_i for every message position.
                            layout: MsmPointLayout::CommonPoints,
                            estimated_terms: key.message_count + 2,
                        },
                        ExecutionJob::Msm {
                            // The proof-specific D term completes T2.
                            layout: MsmPointLayout::VariablePoints,
                            estimated_terms: 1,
                        },
                        ExecutionJob::Pairing {
                            equations_per_item: 1,
                            public_key_context_is_common: shared_public_key,
                        },
                    ],
                    key,
                    request_indices: indices,
                }
            },
        )
    }

    fn group<T, K, B>(requests: &[T], key_for: K, build: B) -> ExecutionPlan
    where
        K: Fn(&T) -> BatchCompatibilityKey,
        B: Fn(BatchCompatibilityKey, Vec<usize>) -> BatchExecutionIr,
    {
        let mut positions = HashMap::<BatchCompatibilityKey, usize>::new();
        let mut groups = Vec::<(BatchCompatibilityKey, Vec<usize>)>::new();
        for (index, request) in requests.iter().enumerate() {
            let key = key_for(request);
            let position = match positions.get(&key) {
                Some(&position) => position,
                None => {
                    let position = groups.len();
                    positions.insert(key.clone(), position);
                    groups.push((key, Vec::new()));
                    position
                }
            };
            groups[position].1.push(index);
        }
        ExecutionPlan {
            batches: groups
                .into_iter()
                .map(|(key, indices)| build(key, indices))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages(count: usize) -> Vec<Vec<u8>> {
        (0..count).map(|index| vec![index as u8]).collect()
    }

    #[test]
    fn sign_groups_by_message_shape_and_protocol_context() {
        let request = |count, header: u8| SignRequest {
            secret_key: vec![],
            public_key: vec![],
            header: vec![header],
            messages: messages(count),
        };
        let plan = CuBbsPlanner::plan_sign(&[
            request(16, 1),
            request(64, 1),
            request(16, 1),
            request(16, 2),
        ]);
        assert_eq!(plan.request_count(), 4);
        assert_eq!(plan.batches.len(), 3);
        assert_eq!(plan.batches[0].request_indices, vec![0, 2]);
        assert!(plan.batches[0].jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::Msm {
                layout: MsmPointLayout::CommonPoints,
                estimated_terms: 18
            }
        )));
    }

    #[test]
    fn verify_declares_h2s_common_msm_and_pairing() {
        let request = VerifyRequest {
            public_key: vec![1],
            signature: vec![],
            header: vec![2],
            messages: messages(8),
        };
        let plan = CuBbsPlanner::plan_verify(&[request]);
        let jobs = &plan.batches[0].jobs;
        assert!(jobs
            .iter()
            .any(|job| matches!(job, ExecutionJob::HashToScalar { .. })));
        assert!(jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::Msm {
                layout: MsmPointLayout::CommonPoints,
                ..
            }
        )));
        assert!(jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::Pairing {
                equations_per_item: 1,
                ..
            }
        )));
    }

    #[test]
    fn repeated_plans_share_message_hashing_without_dropping_items() {
        let sign = SignRequest {
            secret_key: vec![1],
            public_key: vec![2],
            header: vec![3],
            messages: messages(64),
        };
        let verify = VerifyRequest {
            public_key: sign.public_key.clone(),
            signature: vec![4],
            header: sign.header.clone(),
            messages: sign.messages.clone(),
        };

        let sign_plan = CuBbsPlanner::plan_sign_repeated(&sign, 2048);
        let verify_plan = CuBbsPlanner::plan_verify_repeated(&verify, 2048);
        assert_eq!(sign_plan.request_count(), 2048);
        assert_eq!(verify_plan.request_count(), 2048);
        assert!(sign_plan.batches[0].jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::HashToScalar {
                message_inputs: 64,
                transcript_inputs: 4096
            }
        )));
        assert!(verify_plan.batches[0].jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::HashToScalar {
                message_inputs: 64,
                transcript_inputs: 2048
            }
        )));
        assert!(CuBbsPlanner::plan_sign_repeated(&sign, 0)
            .batches
            .is_empty());
        assert!(CuBbsPlanner::plan_verify_repeated(&verify, 0)
            .batches
            .is_empty());
    }

    #[test]
    fn proof_shapes_include_disclosure_layout() {
        let request = |indexes: Vec<usize>| ProofGenRequest {
            public_key: vec![1],
            signature: vec![],
            header: vec![2],
            presentation_header: vec![3],
            messages: messages(8),
            disclosed_indexes: indexes,
        };
        let plan = CuBbsPlanner::plan_proof_gen(&[
            request(vec![0, 2]),
            request(vec![0, 2]),
            request(vec![1, 3]),
        ]);
        assert_eq!(plan.batches.len(), 2);
        assert_eq!(plan.batches[0].request_indices, vec![0, 1]);
        assert!(plan.batches[0].jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::Msm {
                layout: MsmPointLayout::VariablePoints,
                estimated_terms: 7
            }
        )));
    }

    #[test]
    fn proof_verify_declares_all_gpu_engines() {
        let request = ProofVerifyRequest {
            public_key: vec![1],
            proof: vec![],
            header: vec![2],
            presentation_header: vec![3],
            disclosed_messages: messages(2),
            disclosed_indexes: vec![0, 4],
            total_message_count: 8,
        };
        let plan = CuBbsPlanner::plan_proof_verify(&[request]);
        let jobs = &plan.batches[0].jobs;
        assert!(jobs
            .iter()
            .any(|job| matches!(job, ExecutionJob::HashToScalar { .. })));
        assert!(jobs.iter().any(|job| matches!(
            job,
            ExecutionJob::Msm {
                layout: MsmPointLayout::CommonPoints,
                estimated_terms: 10
            }
        )));
        let variable_terms = jobs
            .iter()
            .filter_map(|job| match job {
                ExecutionJob::Msm {
                    layout: MsmPointLayout::VariablePoints,
                    estimated_terms,
                } => Some(*estimated_terms),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(variable_terms, vec![3, 1]);
        assert!(jobs
            .iter()
            .any(|job| matches!(job, ExecutionJob::Pairing { .. })));
    }
}

fn context_digest(input: &[u8]) -> [u8; 32] {
    Sha256::digest(input).into()
}

fn all_equal_by<T, V: ?Sized + PartialEq>(items: &[T], value: impl Fn(&T) -> &V) -> bool {
    items
        .first()
        .is_none_or(|first| items.iter().all(|item| value(item) == value(first)))
}
