//! Shared lowering, MSM grouping, aggregation and operation finalization.

use super::*;

pub(super) fn run_bbs_batch<I, T, F>(
    reqs: Vec<I>,
    fail_fast: bool,
    mut f: F,
) -> BbsBatchRunResult<T>
where
    F: FnMut(I) -> Result<T>,
{
    let mut results = Vec::with_capacity(reqs.len());
    let mut succeeded = 0;
    let mut failed = 0;

    for (index, req) in reqs.into_iter().enumerate() {
        match f(req) {
            Ok(value) => {
                succeeded += 1;
                results.push(BbsBatchItemResult {
                    index,
                    result: Ok(value),
                });
            }
            Err(error) => {
                failed += 1;
                results.push(BbsBatchItemResult {
                    index,
                    result: Err(error),
                });
                if fail_fast {
                    break;
                }
            }
        }
    }

    BbsBatchRunResult {
        attempted: results.len(),
        succeeded,
        failed,
        results,
    }
}

pub(super) fn batch_run_result<T>(
    attempted: usize,
    results: Vec<BbsBatchItemResult<T>>,
) -> BbsBatchRunResult<T> {
    let succeeded = results.iter().filter(|item| item.result.is_ok()).count();
    let failed = results.len() - succeeded;
    BbsBatchRunResult {
        attempted,
        succeeded,
        failed,
        results,
    }
}

pub(super) fn merge_batch_results_with_fail_fast<T>(
    attempted: usize,
    mut results: Vec<BbsBatchItemResult<T>>,
    enforce_fail_fast: bool,
    explicit_first_error_index: Option<usize>,
) -> BbsBatchRunResult<T> {
    results.sort_by_key(|item| item.index);
    if enforce_fail_fast {
        let cut = explicit_first_error_index.or_else(|| {
            results
                .iter()
                .position(|item| item.result.is_err())
                .map(|index| results[index].index)
        });
        if let Some(index) = cut {
            results.retain(|item| item.index <= index);
        }
    }
    batch_run_result(attempted, results)
}

pub(super) fn proof_gen_requests_are_mixed(reqs: &[ProofGenRequest]) -> bool {
    let Some(first) = reqs.first() else {
        return false;
    };
    reqs.iter().skip(1).any(|req| {
        req.public_key != first.public_key
            || req.header != first.header
            || req.presentation_header != first.presentation_header
            || req.messages.len() != first.messages.len()
            || req.disclosed_indexes != first.disclosed_indexes
    })
}

pub(super) fn proof_gen_shape_batching_eligible(reqs: &[ProofGenRequest]) -> bool {
    if reqs.len() < proof_bbs_mixed_shape_min_batch() || !proof_gen_requests_are_mixed(reqs) {
        return false;
    }
    let min_groups = proof_bbs_mixed_shape_min_groups();
    let mut shapes = HashSet::with_capacity(min_groups);
    for req in reqs {
        shapes.insert((req.messages.len(), req.disclosed_indexes.as_slice()));
        if shapes.len() >= min_groups {
            return true;
        }
    }
    false
}

pub(super) fn proof_verify_requests_are_mixed(reqs: &[ProofVerifyRequest]) -> bool {
    let Some(first) = reqs.first() else {
        return false;
    };
    reqs.iter().skip(1).any(|req| {
        req.public_key != first.public_key
            || req.header != first.header
            || req.presentation_header != first.presentation_header
            || req.total_message_count != first.total_message_count
            || req.disclosed_indexes != first.disclosed_indexes
            || req.disclosed_messages.len() != first.disclosed_messages.len()
    })
}

pub(super) fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub(super) fn proof_host_worker_count(item_count: usize) -> usize {
    if item_count == 0 {
        return 0;
    }
    let hardware = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8);
    let tuned_default = if item_count >= 512 {
        6
    } else if item_count >= 128 {
        8
    } else {
        hardware.min(8)
    };
    let configured = std::env::var("CUBBS_BBS_PROOF_HOST_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(tuned_default);
    configured.max(1).min(item_count)
}

pub(super) fn proof_gen_host_worker_count(item_count: usize) -> usize {
    if item_count == 0 {
        return 0;
    }
    let hardware = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8);
    let tuned_default = if item_count >= 512 {
        4
    } else if item_count >= 128 {
        4
    } else {
        hardware.min(8)
    };
    let configured = std::env::var("CUBBS_BBS_PROOF_GEN_HOST_WORKERS")
        .or_else(|_| std::env::var("CUBBS_BBS_PROOF_HOST_WORKERS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(tuned_default);
    configured.max(1).min(item_count)
}

pub(super) fn worker_ranges(item_count: usize) -> Vec<std::ops::Range<usize>> {
    let worker_count = proof_host_worker_count(item_count);
    if worker_count == 0 {
        return Vec::new();
    }
    let chunk_size = item_count.div_ceil(worker_count);
    (0..item_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(item_count))
        .collect()
}

pub(super) fn worker_ranges_with_count(
    item_count: usize,
    worker_count: usize,
) -> Vec<std::ops::Range<usize>> {
    if item_count == 0 || worker_count == 0 {
        return Vec::new();
    }
    let worker_count = worker_count.min(item_count);
    let chunk_size = item_count.div_ceil(worker_count);
    (0..item_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(item_count))
        .collect()
}

pub(super) fn proof_verify_unique_key_worker_count(item_count: usize) -> usize {
    if item_count == 0 {
        return 0;
    }
    let hardware = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8);
    let configured = std::env::var("CUBBS_BBS_PROOF_VERIFY_UNIQUE_KEY_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(hardware.min(32));
    configured.max(1).min(item_count)
}

pub(super) fn into_worker_chunks<T>(items: Vec<T>) -> Vec<Vec<T>> {
    let item_count = items.len();
    let worker_count = proof_host_worker_count(item_count);
    if worker_count == 0 {
        return Vec::new();
    }
    let chunk_size = item_count.div_ceil(worker_count);
    let mut chunks = Vec::with_capacity(worker_count);
    let mut current = Vec::with_capacity(chunk_size);
    for item in items {
        current.push(item);
        if current.len() == chunk_size {
            chunks.push(current);
            current = Vec::with_capacity(chunk_size);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

pub(super) fn into_proof_gen_worker_chunks<T>(items: Vec<T>) -> Vec<Vec<T>> {
    let item_count = items.len();
    let worker_count = proof_gen_host_worker_count(item_count);
    if worker_count == 0 {
        return Vec::new();
    }
    let chunk_size = item_count.div_ceil(worker_count);
    let mut chunks = Vec::with_capacity(worker_count);
    let mut current = Vec::with_capacity(chunk_size);
    for item in items {
        current.push(item);
        if current.len() == chunk_size {
            chunks.push(current);
            current = Vec::with_capacity(chunk_size);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

pub(super) type ProofGenT2PreparedItem = (
    usize,
    ProofGenRequest,
    Vec<BBSplusMessage>,
    Vec<u8>,
    BBSplusProofGenT2State,
);

pub(super) type ProofGenT2CompressedPreparedItem = (
    usize,
    ProofGenRequest,
    Vec<BBSplusMessage>,
    Vec<u8>,
    BBSplusProofGenT2CompressedState,
);

pub(super) type ProofVerifyPrecomputedMsmItem = (
    usize,
    ProofVerifyRequest,
    Vec<BBSplusMessage>,
    Vec<u8>,
    Vec<u8>,
    Scalar,
    ProofVerifyPreparedMeta,
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProofVerifyPreparedMeta {
    pub(super) pairing_input: BbsPairingEqInput,
    pub(super) challenge_be: Vec<u8>,
}

pub(super) fn proof_verify_precomputed_items_strict_gpu(
    kernel: &GpuKernelLibrary,
    attempted: usize,
    items: &[ProofVerifyPrecomputedMsmItem],
    fail_fast: bool,
) -> BbsBatchRunResult<bool> {
    let mut results = Vec::with_capacity(items.len());
    let mut valid_positions = Vec::with_capacity(items.len());
    let mut pairing_inputs = Vec::with_capacity(items.len());

    for (position, (index, req, scalars, t1, t2, domain, meta)) in items.iter().enumerate() {
        match proof_verify_challenge_and_pairing_eq_input_from_prepared(
            req, scalars, t1, t2, *domain, meta,
        ) {
            Ok(input) => {
                valid_positions.push((position, *index));
                pairing_inputs.push(input);
                results.push(None);
            }
            Err(Error::InvalidProof) => results.push(Some(BbsBatchItemResult {
                index: *index,
                result: Ok(false),
            })),
            Err(error) => results.push(Some(BbsBatchItemResult {
                index: *index,
                result: Err(error),
            })),
        }
    }

    if !pairing_inputs.is_empty() {
        let pairing_result = if !kernel.can_pairing_eq() {
            batch_call_error(
                pairing_inputs.len(),
                Error::Gpu(
                    "GPU Pairing symbol is unavailable and CPU fallback is disabled by CUBBS_BBS_FORCE_GPU"
                        .to_owned(),
                ),
                fail_fast,
            )
        } else if kernel.can_pairing_eq_aggregate()
            && pairing_aggregate_enabled(pairing_inputs.len())
        {
            kernel
                .pairing_eq_aggregate(&pairing_inputs, fail_fast)
                .unwrap_or_else(|| kernel.pairing_eq_batch(&pairing_inputs, fail_fast))
        } else {
            kernel.pairing_eq_batch(&pairing_inputs, fail_fast)
        };

        for ((position, index), item) in valid_positions.into_iter().zip(pairing_result.results) {
            results[position] = Some(BbsBatchItemResult {
                index,
                result: item.result,
            });
        }
    }

    let mut flattened = results
        .into_iter()
        .enumerate()
        .map(|(position, item)| {
            item.unwrap_or_else(|| BbsBatchItemResult {
                index: items[position].0,
                result: Err(Error::Gpu(
                    "GPU Pairing returned an incomplete result set".to_owned(),
                )),
            })
        })
        .collect::<Vec<_>>();
    flattened.sort_by_key(|item| item.index);
    if fail_fast {
        if let Some(first_error) = flattened.iter().position(|item| item.result.is_err()) {
            flattened.truncate(first_error + 1);
        }
    }
    batch_run_result(attempted, flattened)
}

pub(super) fn verify_precomputed_b_items_parallel(
    cpu: CuBbsProtocolEngine,
    attempted: usize,
    items: Vec<(usize, VerifyRequest, BbsPreparedB)>,
) -> BbsBatchRunResult<bool> {
    let results = thread::scope(|scope| {
        items
            .into_iter()
            .map(|(index, req, prepared_b)| {
                scope.spawn(move || {
                    let result = cpu.verify_prehashed_b(req, prepared_b);
                    BbsBatchItemResult { index, result }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("verify pairing worker panicked"))
            .collect::<Vec<_>>()
    });

    batch_run_result(attempted, results)
}

pub(super) fn verify_pairing_eq_inputs(
    kernel: &GpuKernelLibrary,
    cpu: CuBbsProtocolEngine,
    items: &[(usize, &VerifyRequest, BbsPreparedB)],
) -> Result<Vec<BbsPairingEqInput>> {
    let fused_g1_enabled = std::env::var("CUBBS_BBS_GPU_VERIFY_FUSED_G1")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true);
    if fused_g1_enabled {
        let one = Scalar::from(1u64).to_bytes_be().to_vec();
        let terms = items
            .iter()
            .map(|(_, req, prepared_b)| {
                if req.signature.len() != SIGNATURE_LEN {
                    return Err(Error::InvalidSignature);
                }
                let e = parse_message_scalar_bytes(&req.signature[48..80])?.value;
                Ok(BbsMsmTerms {
                    compressed_points: vec![
                        prepared_b.compressed_b.clone(),
                        req.signature[0..48].to_vec(),
                    ]
                    .into(),
                    scalar_be: vec![one.clone(), (-e).to_bytes_be().to_vec()],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rhs_points = kernel.g1_msm_batch(&terms, false);
        if rhs_points.failed == 0 && rhs_points.results.len() == items.len() {
            let rhs_g2 = g2_generator_compressed();
            return items
                .iter()
                .zip(rhs_points.results)
                .map(|((_, req, _), rhs)| {
                    Ok(BbsPairingEqInput {
                        lhs_g1: req.signature[0..48].to_vec(),
                        lhs_g2: req.public_key.clone(),
                        rhs_g1: rhs.result?,
                        rhs_g2: rhs_g2.clone(),
                    })
                })
                .collect();
        }
    }
    if items.len() >= verify_parallel_pairing_min_batch() {
        let projective_items: Vec<(Vec<u8>, Vec<u8>, G1Projective)> = thread::scope(|scope| {
            worker_ranges(items.len())
                .into_iter()
                .map(|range| {
                    scope.spawn(move || {
                        items[range]
                            .iter()
                            .map(|(_, req, prepared_b)| {
                                cpu.verify_prehashed_b_pairing_projective(req, prepared_b)
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("verify pairing input worker panicked"))
                .collect::<Result<Vec<_>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        })?;
        let projectives = projective_items
            .iter()
            .map(|(_, _, rhs)| *rhs)
            .collect::<Vec<_>>();
        let mut affines = vec![G1Affine::identity(); projectives.len()];
        G1Projective::batch_normalize(&projectives, &mut affines);
        let rhs_g2 = g2_generator_compressed();
        Ok(projective_items
            .into_iter()
            .zip(affines)
            .map(|((lhs_g1, lhs_g2, _), rhs)| BbsPairingEqInput {
                lhs_g1,
                lhs_g2,
                rhs_g1: rhs.to_compressed().to_vec(),
                rhs_g2: rhs_g2.clone(),
            })
            .collect())
    } else {
        items
            .iter()
            .map(|(_, req, prepared_b)| cpu.verify_prehashed_b_pairing_eq_input(req, prepared_b))
            .collect::<Result<Vec<_>>>()
    }
}

pub(super) fn verify_aggregate_shape_scalar_lanes(
    prepared: &[(usize, &VerifyRequest, &[BBSplusMessage])],
    positions: &[usize],
    coefficients: &[Scalar],
    domain: Scalar,
    term_count: usize,
    lane_count: usize,
) -> Vec<Vec<Scalar>> {
    let mut lanes = vec![vec![Scalar::from(0u64); term_count]; lane_count];
    if let Some(&first_position) = positions.first() {
        let (_, first_request, first_messages) = &prepared[first_position];
        if positions.iter().all(|&position| {
            let (_, request, messages) = &prepared[position];
            std::ptr::eq(*request, *first_request)
                && messages.len() == first_messages.len()
                && std::ptr::eq(messages.as_ptr(), first_messages.as_ptr())
        }) {
            let mut coefficient_sums = vec![Scalar::from(0u64); lane_count];
            for &position in positions {
                coefficient_sums[position % lane_count] += coefficients[position];
            }
            for (lane, coefficient_sum) in lanes.iter_mut().zip(coefficient_sums) {
                lane[0] = coefficient_sum;
                lane[1] = domain * coefficient_sum;
                for (message_position, message) in first_messages.iter().enumerate() {
                    lane[message_position + 2] = message.value * coefficient_sum;
                }
            }
            return lanes;
        }
    }
    for &position in positions {
        let (_, _, messages) = &prepared[position];
        let coefficient = coefficients[position];
        let aggregate = &mut lanes[position % lane_count];
        aggregate[0] += coefficient;
        aggregate[1] += domain * coefficient;
        for (message_position, message) in messages.iter().enumerate() {
            aggregate[message_position + 2] += message.value * coefficient;
        }
    }
    lanes
}

pub(super) fn repeated_verify_a_scalar_lanes(
    prepared: &[(usize, &VerifyRequest, &[BBSplusMessage])],
    coefficients: &[Scalar],
    lane_count: usize,
) -> Option<Vec<(Scalar, Scalar)>> {
    if lane_count == 0 || prepared.len() != coefficients.len() {
        return None;
    }
    let (_, first_request, _) = prepared.first()?;
    if !prepared
        .iter()
        .all(|(_, request, _)| std::ptr::eq(*request, *first_request))
    {
        return None;
    }
    let e = parse_message_scalar_bytes(first_request.signature.get(48..SIGNATURE_LEN)?)
        .ok()?
        .value;
    let mut sums = vec![Scalar::from(0u64); lane_count];
    for (position, coefficient) in coefficients.iter().enumerate() {
        sums[position % lane_count] += *coefficient;
    }
    Some(sums.into_iter().map(|sum| (sum, -(e * sum))).collect())
}

pub(super) fn verify_homogeneous_pre_b_aggregate(
    engine: &GpuBbsEngine,
    kernel: &GpuKernelLibrary,
    prepared: &[(usize, &VerifyRequest, &[BBSplusMessage])],
    plan: &planner::ExecutionPlan,
    _fail_fast: bool,
) -> Option<BbsBatchRunResult<bool>> {
    let profile = std::env::var_os("CUBBS_BBS_VERIFY_AGGREGATE_PROFILE").is_some();
    let profile_started = Instant::now();
    if !verify_pre_b_aggregate_enabled(prepared.len())
        || !kernel.can_g1_msm()
        || !kernel.can_g1_add()
        || !kernel.can_pairing_eq()
    {
        return None;
    }
    let (_, first_req, _) = prepared.first()?;
    if prepared.iter().any(|(_, req, _)| {
        req.public_key != first_req.public_key
            || req.signature.len() != SIGNATURE_LEN
            || (!verify_mixed_aggregate_enabled() && plan.batches.len() > 1)
    }) {
        return None;
    }
    // The A and A·e rows have the same dynamic signature-A points.  Keep the
    // multi-lane form coupled to the dedicated dual ABI: when the diagnostic
    // switch disables that ABI, the safe fallback below must remain two
    // singleton MSMs rather than a two-row common-points call (which would
    // persist every transient HTTP batch in the common-points comb cache).
    let use_dual_same_points = gpu_verify_dual_same_points_enabled(prepared.len());
    let lane_count = if use_dual_same_points
        && kernel.can_g1_msm_common_points()
        && kernel.can_g1_dual_msm_same_points_lanes()
    {
        verify_aggregate_lane_count(prepared.len())
    } else {
        1
    };
    let coefficients = pairing_aggregate_scalars(prepared.len());

    let repeated_a_lanes = repeated_verify_a_scalar_lanes(prepared, &coefficients, lane_count);
    let repeated_a_compressed = repeated_a_lanes.is_some();
    let a_capacity = if repeated_a_compressed {
        lane_count
    } else {
        prepared.len()
    };
    let mut a_points = Vec::<&[u8]>::with_capacity(a_capacity);
    let mut a_scalars = Vec::<[u8; 32]>::with_capacity(a_capacity);
    let mut ae_scalars = Vec::<[u8; 32]>::with_capacity(a_capacity);
    if let Some(lanes) = repeated_a_lanes {
        for (a, ae) in lanes {
            a_points.push(&first_req.signature[..48]);
            a_scalars.push(a.to_bytes_be());
            ae_scalars.push(ae.to_bytes_be());
        }
    } else {
        for ((_, req, _), coefficient) in prepared.iter().zip(coefficients.iter()) {
            let e = parse_message_scalar_bytes(&req.signature[48..SIGNATURE_LEN])
                .ok()?
                .value;
            a_points.push(&req.signature[..48]);
            a_scalars.push(coefficient.to_bytes_be());
            ae_scalars.push((-(e * coefficient)).to_bytes_be());
        }
    }

    let group_positions = plan
        .batches
        .iter()
        .map(|batch| batch.request_indices.clone())
        .collect::<Vec<_>>();
    let b_terms = thread::scope(|scope| {
        let coefficients = &coefficients;
        let prepared = prepared;
        group_positions
            .into_iter()
            .map(|positions| {
                let coefficients = coefficients;
                let prepared = prepared;
                scope.spawn(move || {
                    let first_position = *positions.first()?;
                    let (_, req, scalars) = &prepared[first_position];
                    let template_req = SignRequest {
                        secret_key: Vec::new(),
                        public_key: req.public_key.clone(),
                        header: req.header.clone(),
                        messages: req.messages.clone(),
                    };
                    let template = engine
                        .prepare_b_msm_terms_for_prehashed_message_scalars_cached(
                            &template_req,
                            scalars,
                        )
                        .ok()?;
                    if template.compressed_points.len() != scalars.len() + 2 {
                        return None;
                    }
                    let domain = parse_message_scalar_bytes(template.scalar_be.get(1)?)
                        .ok()?
                        .value;
                    let aggregate_scalar_lanes = verify_aggregate_shape_scalar_lanes(
                        prepared,
                        &positions,
                        coefficients,
                        domain,
                        template.compressed_points.len(),
                        lane_count,
                    );
                    if aggregate_scalar_lanes.len() != lane_count {
                        return None;
                    }
                    Some((template.compressed_points, aggregate_scalar_lanes))
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .expect("verify aggregate shape worker panicked")
            })
            .collect::<Option<Vec<_>>>()
    })?;
    let largest_b_terms = b_terms.iter().max_by_key(|(points, _)| points.len())?;
    let mut merged_scalar_lanes =
        vec![vec![Scalar::from(0u64); largest_b_terms.0.len()]; lane_count];
    for (points, scalar_lanes) in &b_terms {
        if scalar_lanes.len() != lane_count
            || scalar_lanes
                .iter()
                .any(|scalars| scalars.len() != points.len())
            || largest_b_terms.0[..points.len()] != points[..]
        {
            return None;
        }
        for (merged, scalars) in merged_scalar_lanes.iter_mut().zip(scalar_lanes) {
            for (target, scalar) in merged.iter_mut().zip(scalars) {
                *target += scalar;
            }
        }
    }
    let b_points_template = largest_b_terms.0.clone();
    let packed_b_scalars = merged_scalar_lanes
        .iter()
        .flat_map(|row| row.iter().flat_map(|scalar| scalar.to_bytes_be()))
        .collect::<Vec<_>>();
    let setup_ms = elapsed_ms(profile_started);
    let b_started = Instant::now();
    let b_result = if kernel.can_g1_msm_common_points() {
        kernel.g1_msm_common_points_packed(&b_points_template, &packed_b_scalars, lane_count, true)
    } else {
        let b_aggregate_terms = BbsMsmTerms {
            compressed_points: b_points_template,
            scalar_be: merged_scalar_lanes
                .pop()?
                .into_iter()
                .map(|scalar| scalar.to_bytes_be().to_vec())
                .collect(),
        };
        kernel.g1_msm_batch(&[b_aggregate_terms], true)
    };
    let b_aggregates = b_result
        .results
        .into_iter()
        .map(|item| item.result)
        .collect::<Result<Vec<_>>>()
        .ok()?;
    if b_aggregates.len() != lane_count {
        return None;
    }
    let b_ms = elapsed_ms(b_started);
    let a_started = Instant::now();
    let dual_aggregate_lanes = if use_dual_same_points && lane_count > 1 {
        kernel.g1_dual_msm_same_points_lanes_borrowed(
            &a_points,
            &a_scalars,
            &ae_scalars,
            lane_count,
        )
    } else if use_dual_same_points && kernel.can_g1_dual_msm_same_points() {
        kernel
            .g1_dual_msm_same_points_borrowed(&a_points, &a_scalars, &ae_scalars)
            .map(|(lhs, rhs)| (vec![lhs], vec![rhs]))
    } else {
        None
    };
    let (a_aggregates, ae_aggregates, a_msm_path) = if let Some(aggregates) = dual_aggregate_lanes {
        let path = if repeated_a_compressed {
            "dual-repeated-compressed"
        } else {
            "dual-same-points"
        };
        (aggregates.0, aggregates.1, path)
    } else {
        if lane_count != 1 {
            return None;
        }
        let a_terms = BbsMsmTerms {
            compressed_points: Arc::new(a_points.iter().map(|point| point.to_vec()).collect()),
            scalar_be: a_scalars.iter().map(|scalar| scalar.to_vec()).collect(),
        };
        let ae_terms = BbsMsmTerms {
            compressed_points: Arc::new(a_points.iter().map(|point| point.to_vec()).collect()),
            scalar_be: ae_scalars.iter().map(|scalar| scalar.to_vec()).collect(),
        };
        // Do not submit [A, A·e] as one two-row MSM.  Their identical,
        // request-dependent points would satisfy the native common-points
        // detector and populate its process-lifetime comb cache once per
        // transient batch.  Singleton inputs deliberately select the bounded
        // Route 4 generic path instead.
        let a_aggregate = kernel
            .g1_msm_batch(std::slice::from_ref(&a_terms), true)
            .results
            .into_iter()
            .next()?
            .result
            .ok()?;
        let ae_aggregate = kernel
            .g1_msm_batch(std::slice::from_ref(&ae_terms), true)
            .results
            .into_iter()
            .next()?
            .result
            .ok()?;
        (vec![a_aggregate], vec![ae_aggregate], "route4-singletons")
    };
    if a_aggregates.len() != lane_count || ae_aggregates.len() != lane_count {
        return None;
    }
    let a_ms = elapsed_ms(a_started);
    let add_started = Instant::now();
    let rhs_aggregates = kernel
        .g1_add_batch(
            &b_aggregates
                .into_iter()
                .zip(ae_aggregates)
                .collect::<Vec<_>>(),
            true,
        )
        .results
        .into_iter()
        .map(|item| item.result)
        .collect::<Result<Vec<_>>>()
        .ok()?;
    if rhs_aggregates.len() != lane_count {
        return None;
    }
    let add_ms = elapsed_ms(add_started);
    let rhs_g2 = g2_generator_compressed();
    let pairing_inputs = a_aggregates
        .into_iter()
        .zip(rhs_aggregates)
        .map(|(lhs_g1, rhs_g1)| BbsPairingEqInput {
            lhs_g1,
            lhs_g2: first_req.public_key.clone(),
            rhs_g1,
            rhs_g2: rhs_g2.clone(),
        })
        .collect::<Vec<_>>();
    let pairing_started = Instant::now();
    let pairing = kernel.pairing_eq_batch(&pairing_inputs, true);
    let pairing_ms = elapsed_ms(pairing_started);
    if pairing.failed != 0
        || pairing.results.len() != lane_count
        || pairing
            .results
            .iter()
            .any(|item| item.result.as_ref().ok().copied() != Some(true))
    {
        return None;
    }
    if profile {
        eprintln!(
            "cubbs_verify_aggregate_profile items={} lanes={} a_msm_path={a_msm_path} setup_ms={setup_ms:.6} \
             b_msm_ms={b_ms:.6} a_msm_ms={a_ms:.6} add_ms={add_ms:.6} \
             pairing_ms={pairing_ms:.6} total_ms={:.6}",
            prepared.len(),
            lane_count,
            elapsed_ms(profile_started),
        );
    }
    let results = prepared
        .iter()
        .map(|(index, _, _)| BbsBatchItemResult {
            index: *index,
            result: Ok(true),
        })
        .collect::<Vec<_>>();
    Some(batch_run_result(prepared.len(), results))
}

pub(super) fn proof_verify_precomputed_msm_items_parallel(
    cpu: CuBbsProtocolEngine,
    attempted: usize,
    items: Vec<ProofVerifyPrecomputedMsmItem>,
) -> BbsBatchRunResult<bool> {
    let results = thread::scope(|scope| {
        items
            .into_iter()
            .map(|(index, req, scalars, t1, t2, _domain, _meta)| {
                scope.spawn(move || {
                    let result =
                        cpu.proof_verify_prehashed_disclosed_scalars_and_msm(req, scalars, t1, t2);
                    BbsBatchItemResult { index, result }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("proof-verify pairing worker panicked"))
            .collect::<Vec<_>>()
    });

    batch_run_result(attempted, results)
}

pub(super) fn prepare_proof_gen_t2_msm_terms_batch(
    cpu: CuBbsProtocolEngine,
    prepared: Vec<(usize, ProofGenRequest, Vec<BBSplusMessage>)>,
    prepared_b: Vec<BbsBatchItemResult<Vec<u8>>>,
    b_terms: Vec<BbsMsmTerms>,
) -> Result<(Vec<ProofGenT2PreparedItem>, Vec<BbsMsmTerms>)> {
    prepare_proof_gen_t2_msm_terms_batch_with_domains(cpu, prepared, prepared_b, b_terms, None)
}

pub(super) fn proof_gen_homogeneous_packed_batch(
    kernel: &GpuKernelLibrary,
    engine: &GpuBbsEngine,
    prepared: &[(usize, ProofGenRequest, Vec<BBSplusMessage>)],
    attempted: usize,
    fail_fast: bool,
) -> Option<BbsBatchRunResult<Vec<u8>>> {
    let profile = std::env::var_os("CUBBS_BBS_GPU_PROOF_GEN_PROFILE").is_some();
    let total_started = Instant::now();
    if !gpu_proof_gen_t2_prepare_for_shape(
        prepared.len(),
        prepared
            .first()
            .map(|(_, req, _)| req.messages.len())
            .unwrap_or(0),
    ) || !kernel.can_g1_msm_common_points()
        || !kernel.can_proof_gen_t2_prepare()
        || !gpu_proof_gen_t2_prepare_enabled()
    {
        return None;
    }
    let (_, first_req, first_scalars) = prepared.first()?;
    if !proof_gen_gpu_t2_msm_enabled(first_scalars.len())
        || prepared.iter().any(|(_, req, scalars)| {
            req.public_key != first_req.public_key
                || req.header != first_req.header
                || req.messages.len() != first_req.messages.len()
                || req.disclosed_indexes != first_req.disclosed_indexes
                || scalars.len() != first_scalars.len()
        })
    {
        return None;
    }
    let terms_req = SignRequest {
        secret_key: Vec::new(),
        public_key: first_req.public_key.clone(),
        header: first_req.header.clone(),
        messages: first_req.messages.clone(),
    };
    let template = match engine
        .prepare_b_msm_terms_for_prehashed_message_scalars_cached(&terms_req, first_scalars)
    {
        Ok(template) => template,
        Err(error) => return Some(batch_call_error(attempted, error, fail_fast)),
    };
    if template.compressed_points.len() != first_scalars.len() + 2 {
        return Some(batch_call_error(
            attempted,
            Error::Gpu("proof-gen packed B template shape mismatch".to_owned()),
            fail_fast,
        ));
    }
    let domain = match template.scalar_be.get(1) {
        Some(bytes) => match parse_message_scalar_bytes(bytes) {
            Ok(value) => value.value,
            Err(error) => return Some(batch_call_error(attempted, error, fail_fast)),
        },
        None => {
            return Some(batch_call_error(
                attempted,
                Error::Gpu("proof-gen packed B template missing domain".to_owned()),
                fail_fast,
            ))
        }
    };
    let one = Scalar::from(1u64).to_bytes_be();
    let domain_be = domain.to_bytes_be();
    let scalar_row_stride = template.compressed_points.len() * 32;
    let mut packed = Vec::with_capacity(prepared.len() * scalar_row_stride);
    for (_, _, scalars) in prepared {
        packed.extend_from_slice(&one);
        packed.extend_from_slice(&domain_be);
        for scalar in scalars {
            packed.extend_from_slice(&scalar.value.to_bytes_be());
        }
    }
    let pack_ms = elapsed_ms(total_started);
    let b_started = Instant::now();
    let prepared_b = kernel.g1_msm_common_points_packed(
        &template.compressed_points,
        &packed,
        prepared.len(),
        fail_fast,
    );
    if prepared_b.failed != 0 || prepared_b.results.len() != prepared.len() {
        return Some(batch_call_error(
            attempted,
            Error::Gpu("proof-gen packed B MSM failed".to_owned()),
            fail_fast,
        ));
    }
    let b_msm_ms = elapsed_ms(b_started);
    let t2_started = Instant::now();
    let domains = vec![domain; prepared.len()];
    let (states, t2_results) = match prepare_proof_gen_t2_with_gpu_fixed_muls(
        kernel,
        engine.protocol,
        prepared.to_vec(),
        prepared_b.results,
        domains,
        Some(&template.compressed_points),
        fail_fast,
    ) {
        Ok(value) => value,
        Err(error) => return Some(batch_call_error(attempted, error, fail_fast)),
    };
    let t2_prepare_ms = elapsed_ms(t2_started);
    if t2_results.failed != 0 || t2_results.results.len() != states.len() {
        return Some(batch_call_error(
            attempted,
            Error::Gpu("proof-gen packed T2 MSM failed".to_owned()),
            fail_fast,
        ));
    }
    let finalize_started = Instant::now();
    let result = finalize_proof_gen_t2_compressed_batch(
        engine.protocol,
        Some(kernel),
        attempted,
        states,
        t2_results.results,
        fail_fast,
    );
    if profile {
        eprintln!(
            "cubbs_proof_gen_gpu_profile items={} pack_ms={pack_ms:.6} \
             b_msm_ms={b_msm_ms:.6} t2_prepare_ms={t2_prepare_ms:.6} \
             finalize_ms={:.6} total_ms={:.6}",
            prepared.len(),
            elapsed_ms(finalize_started),
            elapsed_ms(total_started),
        );
    }
    Some(result)
}

pub(super) fn prepare_proof_gen_t2_msm_terms_batch_with_domains(
    cpu: CuBbsProtocolEngine,
    prepared: Vec<(usize, ProofGenRequest, Vec<BBSplusMessage>)>,
    prepared_b: Vec<BbsBatchItemResult<Vec<u8>>>,
    b_terms: Vec<BbsMsmTerms>,
    precomputed_domains: Option<Vec<Scalar>>,
) -> Result<(Vec<ProofGenT2PreparedItem>, Vec<BbsMsmTerms>)> {
    let domains = precomputed_domains.filter(|domains| domains.len() == b_terms.len());
    let items = prepared
        .into_iter()
        .zip(prepared_b)
        .zip(b_terms)
        .enumerate()
        .collect::<Vec<_>>();

    let results = if items.len() >= proof_gpu_prepare_parallel_min_batch() {
        let chunks = into_proof_gen_worker_chunks(items);
        thread::scope(|scope| {
            let results = chunks
                .into_iter()
                .map(|chunk| {
                    let domains = domains.as_ref();
                    scope.spawn(move || {
                        chunk
                            .into_iter()
                            .map(
                                |(position, (((index, req, scalars), prepared_b), b_terms))| {
                                    let precomputed_domain =
                                        domains.and_then(|values| values.get(position).copied());
                                    prepare_proof_gen_t2_msm_terms_item(
                                        cpu,
                                        index,
                                        req,
                                        scalars,
                                        prepared_b,
                                        b_terms,
                                        precomputed_domain,
                                    )
                                },
                            )
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("proof-gen prepare worker panicked"))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Ok(results)
        })?
    } else {
        items
            .into_iter()
            .map(
                |(position, (((index, req, scalars), prepared_b), b_terms))| {
                    let precomputed_domain = domains
                        .as_ref()
                        .and_then(|values| values.get(position).copied());
                    prepare_proof_gen_t2_msm_terms_item(
                        cpu,
                        index,
                        req,
                        scalars,
                        prepared_b,
                        b_terms,
                        precomputed_domain,
                    )
                },
            )
            .collect::<Result<Vec<_>>>()?
    };

    let mut states_and_items = Vec::with_capacity(results.len());
    let mut t2_terms = Vec::with_capacity(results.len());
    for (state_item, terms) in results {
        states_and_items.push(state_item);
        t2_terms.push(terms);
    }

    Ok((states_and_items, t2_terms))
}

/// GPU proof-gen t2Prepare: the 6 fixed G1 scalar multiplications run on the
/// device. Host precomputes random scalars + the 7 scalar multipliers (cheap,
/// ~3us/proof); the GPU does the 6 expensive G1 muls (~215us each on CPU);
/// then the state + t2_terms are assembled via the zkryptium helper.
pub(super) fn prepare_proof_gen_t2_with_gpu_fixed_muls(
    kernel: &GpuKernelLibrary,
    cpu: CuBbsProtocolEngine,
    prepared: Vec<(usize, ProofGenRequest, Vec<BBSplusMessage>)>,
    prepared_b: Vec<BbsBatchItemResult<Vec<u8>>>,
    domains: Vec<Scalar>,
    expanded_t2_points: Option<&[Vec<u8>]>,
    fail_fast: bool,
) -> Result<(
    Vec<ProofGenT2CompressedPreparedItem>,
    BbsBatchRunResult<Vec<u8>>,
)> {
    use bls12_381_plus::ff::Field;
    let profile = std::env::var_os("CUBBS_BBS_GPU_PROOF_GEN_T2_PROFILE").is_some();
    let total_started = Instant::now();
    let n = prepared.len();
    if domains.len() != n {
        return Err(Error::Gpu("proof-gen domain count mismatch".to_owned()));
    }

    // Phase 1 (host, parallel): per proof, generate random scalars + 7
    // multipliers, extract signature.A (compressed) and e (scalar) from the
    // raw 80-byte signature [A(48) || e(32)].
    struct Prep {
        index: usize,
        req: ProofGenRequest,
        scalars: Vec<BBSplusMessage>,
        compressed_b: Vec<u8>,
        scalars_be: [[u8; 32]; 7],
        domain: Scalar,
        random_scalars: Vec<Scalar>,
    }
    let items_vec: Vec<(
        (usize, ProofGenRequest, Vec<BBSplusMessage>),
        BbsBatchItemResult<Vec<u8>>,
    )> = prepared
        .into_iter()
        .zip(prepared_b)
        .map(|((index, req, scalars), pb)| ((index, req, scalars), pb))
        .collect();

    let build_prep = |(((index, req, scalars), pb), domain): (
        (
            (usize, ProofGenRequest, Vec<BBSplusMessage>),
            BbsBatchItemResult<Vec<u8>>,
        ),
        Scalar,
    )|
     -> Result<Prep> {
        let compressed_b = pb.result?;
        if req.signature.len() != SIGNATURE_LEN {
            return Err(Error::Gpu(
                "proof-gen t2 GPU: bad signature length".to_owned(),
            ));
        }
        let e_arr = <[u8; 32]>::try_from(&req.signature[48..80])
            .map_err(|_| Error::Gpu("proof-gen t2 GPU: bad signature e slice".to_owned()))?;
        let e_ct = Scalar::from_be_bytes(&e_arr);
        if bool::from(e_ct.is_none()) {
            return Err(Error::Gpu("proof-gen t2 GPU: bad signature e".to_owned()));
        }
        let e = e_ct.unwrap();
        let l = scalars.len();
        let r = req.disclosed_indexes.len();
        let u = l
            .checked_sub(r)
            .ok_or(Error::Gpu("proof-gen t2 GPU: R>L".to_owned()))?;
        let mut rng = rand::thread_rng();
        let random_scalars: Vec<Scalar> = (0..5 + u).map(|_| Scalar::random(&mut rng)).collect();
        let r1 = random_scalars[0];
        let r2 = random_scalars[1];
        let e_tilde = random_scalars[2];
        let r1_tilde = random_scalars[3];
        let r3_tilde = random_scalars[4];
        let r1r2 = r1 * r2;
        let scalars_be = [
            r2.to_be_bytes(),
            r1r2.to_be_bytes(),
            r1r2.to_be_bytes(),
            (r1r2 * e).to_be_bytes(),
            (r1r2 * e_tilde).to_be_bytes(),
            (r2 * r1_tilde).to_be_bytes(),
            r3_tilde.to_be_bytes(),
        ];
        Ok(Prep {
            index,
            req,
            scalars,
            compressed_b,
            scalars_be,
            domain,
            random_scalars,
        })
    };
    let prep_items = items_vec.into_iter().zip(domains).collect::<Vec<_>>();
    let preps = if prep_items.len() >= proof_gpu_prepare_parallel_min_batch() {
        let chunks = into_proof_gen_worker_chunks(prep_items);
        thread::scope(|scope| {
            chunks
                .into_iter()
                .map(|chunk| {
                    let build_prep = &build_prep;
                    scope.spawn(move || {
                        chunk
                            .into_iter()
                            .map(build_prep)
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("proof-gen prep worker panicked"))
                .collect::<Result<Vec<_>>>()
                .map(|chunks| chunks.into_iter().flatten().collect::<Vec<_>>())
        })?
    } else {
        prep_items
            .into_iter()
            .map(build_prep)
            .collect::<Result<Vec<_>>>()?
    };
    let host_prep_ms = elapsed_ms(total_started);

    // Phase 2 (GPU): dispatch the 6 fixed muls.
    let gpu_started = Instant::now();
    let proof_gen_seed = [0u8; 32];
    let gpu_inputs: Vec<BbsT2PrepareGpuInput<'_>> = preps
        .iter()
        .map(|p| BbsT2PrepareGpuInput {
            compressed_b: &p.compressed_b,
            signature_a: &p.req.signature[..48],
            scalars_be: std::array::from_fn(|index| p.scalars_be[index].as_slice()),
            seed: &proof_gen_seed,
        })
        .collect();
    let early_t2_terms = preps
        .iter()
        .map(|prep| {
            cpu.proof_gen_t2_terms_from_compressed_b(
                prep.scalars.len(),
                &prep.req.disclosed_indexes,
                &prep.random_scalars,
                &prep.compressed_b,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    // T2 contains one variable-base term, B*r3_tilde. For homogeneous
    // batches B is itself an MSM over the cached B generators. Distribute
    // r3_tilde through those coefficients and merge the remaining hidden
    // generator terms, turning T2 into one fixed/common-points MSM. This
    // removes a per-proof variable-base scalar multiplication and its
    // per-row radix-16 table build from the CUDA hot path.
    let expanded_t2 = expanded_t2_points
        .filter(|_| gpu_proof_gen_expanded_t2_enabled())
        .and_then(|common_points| {
            if common_points.len()
                != preps
                    .first()
                    .map(|prep| prep.scalars.len() + 2)
                    .unwrap_or_default()
            {
                return None;
            }
            let point_positions = common_points
                .iter()
                .enumerate()
                .map(|(index, point)| (point.as_slice(), index))
                .collect::<HashMap<_, _>>();
            let mut packed = Vec::with_capacity(preps.len() * common_points.len() * 32);
            for (prep, terms) in preps.iter().zip(&early_t2_terms) {
                if terms.compressed_points.first() != Some(&prep.compressed_b)
                    || terms.compressed_points.len() != terms.scalar_be.len()
                {
                    return None;
                }
                let multiplier = parse_message_scalar_bytes(terms.scalar_be.first()?)
                    .ok()?
                    .value;
                let mut scalars = Vec::with_capacity(common_points.len());
                scalars.push(multiplier);
                scalars.push(prep.domain * multiplier);
                scalars.extend(
                    prep.scalars
                        .iter()
                        .map(|message| message.value * multiplier),
                );
                for (point, scalar) in terms.compressed_points.iter().zip(&terms.scalar_be).skip(1)
                {
                    let &position = point_positions.get(point.as_slice())?;
                    let scalar = parse_message_scalar_bytes(scalar).ok()?.value;
                    scalars[position] += scalar;
                }
                for scalar in scalars {
                    packed.extend_from_slice(&scalar.to_bytes_be());
                }
            }
            Some((common_points, packed))
        });
    let (gpu_result, fixed_muls_ms, t2_results, t2_msm_ms) = thread::scope(|scope| {
        let gpu_handle = scope.spawn(|| {
            let started = Instant::now();
            let result = kernel.proof_gen_t2_prepare_batch(&gpu_inputs, true);
            (result, elapsed_ms(started))
        });
        let t2_started = Instant::now();
        let t2_results = match expanded_t2.as_ref() {
            Some((common_points, packed)) => {
                kernel.g1_msm_common_points_packed(common_points, packed, preps.len(), fail_fast)
            }
            None => g1_msm_batch_split_variable_term_grouped(kernel, &early_t2_terms, 0, fail_fast)
                .unwrap_or_else(|| {
                    g1_msm_batch_grouped_by_points(kernel, &early_t2_terms, fail_fast)
                }),
        };
        let t2_msm_ms = elapsed_ms(t2_started);
        let (gpu_result, fixed_muls_ms) = gpu_handle
            .join()
            .expect("proof-gen fixed-mul worker panicked");
        (gpu_result, fixed_muls_ms, t2_results, t2_msm_ms)
    });
    if gpu_result.failed > 0 {
        return Err(Error::Gpu(format!(
            "proof-gen t2 GPU fixed-muls failed: {}/{}",
            gpu_result.failed, gpu_result.attempted
        )));
    }
    let gpu_ms = elapsed_ms(gpu_started);

    // Phase 3 (host): assemble compressed state from the 4 GPU points.
    let assemble_started = Instant::now();
    let assemble_item = |(prep, item): (Prep, BbsBatchItemResult<BbsT2PrepareGpuOutput>)|
     -> Result<ProofGenT2CompressedPreparedItem> {
        let out = item.result?;
        let Prep {
            index,
            req,
            scalars,
            compressed_b,
            domain,
            random_scalars,
            ..
        } = prep;
        let state = cpu.proof_gen_state_from_precomputed_compressed_points(
            &req,
            &scalars,
            random_scalars,
            out.d,
            out.abar,
            out.bbar,
            out.t1,
            domain,
        )?;
        Ok((index, req, scalars, compressed_b, state))
    };
    let pairs = preps
        .into_iter()
        .zip(gpu_result.results)
        .collect::<Vec<_>>();
    let assembled = if pairs.len() >= proof_gpu_prepare_parallel_min_batch() {
        let chunks = into_proof_gen_worker_chunks(pairs);
        thread::scope(|scope| {
            chunks
                .into_iter()
                .map(|chunk| {
                    let assemble_item = &assemble_item;
                    scope.spawn(move || {
                        chunk
                            .into_iter()
                            .map(assemble_item)
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("proof-gen GPU T2 assemble worker panicked")
                })
                .collect::<Result<Vec<_>>>()
                .map(|chunks| chunks.into_iter().flatten().collect::<Vec<_>>())
        })?
    } else {
        pairs
            .into_iter()
            .map(assemble_item)
            .collect::<Result<Vec<_>>>()?
    };
    let states_and_items = assembled;
    if profile {
        eprintln!(
            "cubbs_proof_gen_t2_gpu_profile items={} host_prep_ms={:.6} fixed_muls_ms={:.6} t2_msm_ms={:.6} gpu_ms={:.6} assemble_ms={:.6} total_ms={:.6}",
            n,
            host_prep_ms,
            fixed_muls_ms,
            t2_msm_ms,
            gpu_ms,
            elapsed_ms(assemble_started),
            elapsed_ms(total_started),
        );
    }
    Ok((states_and_items, t2_results))
}

pub(super) fn prepare_proof_gen_t2_msm_terms_item(
    cpu: CuBbsProtocolEngine,
    index: usize,
    req: ProofGenRequest,
    scalars: Vec<BBSplusMessage>,
    prepared_b: BbsBatchItemResult<Vec<u8>>,
    b_terms: BbsMsmTerms,
    precomputed_domain: Option<Scalar>,
) -> Result<(ProofGenT2PreparedItem, BbsMsmTerms)> {
    let compressed_b = prepared_b.result?;
    let precomputed_domain = match precomputed_domain {
        Some(domain) => domain,
        None => b_msm_domain_scalar(&b_terms)?,
    };
    let (state, terms) = cpu.proof_gen_t2_msm_terms_for_prehashed_message_scalars_and_b(
        &req,
        &scalars,
        &compressed_b,
        Some(precomputed_domain),
    )?;
    Ok(((index, req, scalars, compressed_b, state), terms))
}

pub(super) fn prepare_proof_verify_msm_terms_batch(
    cpu: CuBbsProtocolEngine,
    kernel: &GpuKernelLibrary,
    prepared: &[(usize, ProofVerifyRequest, Vec<BBSplusMessage>)],
) -> Result<(Vec<BbsMsmTerms>, Vec<(Scalar, ProofVerifyPreparedMeta)>)> {
    let cache_keys = proof_verify_prepared_cache_enabled().then(|| {
        prepared
            .iter()
            .map(|(_, req, scalars)| proof_verify_prepared_cache_key(req, scalars))
            .collect::<Vec<_>>()
    });
    if let Some(keys) = cache_keys.as_ref() {
        let cache = proof_verify_prepared_cache().lock().unwrap();
        let cached = keys
            .iter()
            .map(|key| cache.get(key).cloned())
            .collect::<Option<Vec<_>>>();
        drop(cache);
        if let Some(paired_terms) = cached {
            return proof_verify_terms_and_meta_from_paired(paired_terms);
        }
    }
    if proof_verify_shared_domain_enabled() {
        if let Some(paired_terms) =
            prepare_proof_verify_msm_terms_with_grouped_domains(cpu, prepared)?
        {
            if let Some(keys) = cache_keys.as_ref() {
                cache_proof_verify_prepared_terms(keys, &paired_terms);
            }
            return proof_verify_terms_and_meta_from_paired(paired_terms);
        }
    }

    let gpu_domains = gpu_proof_domain_enabled()
        .then(|| prepare_unique_key_proof_domains(kernel, prepared).ok())
        .flatten();

    let paired_terms = if prepared.len() >= proof_gpu_prepare_parallel_min_batch() {
        // This path is selected after grouped-domain preparation rejects a
        // high-cardinality issuer set. Domain and generator preparation are
        // CPU-heavy and independent per item, so use all available cores
        // without changing the lower worker count tuned for shared issuers.
        let ranges = worker_ranges_with_count(
            prepared.len(),
            proof_verify_unique_key_worker_count(prepared.len()),
        );
        thread::scope(|scope| {
            let results = ranges
                .into_iter()
                .map(|range| {
                    let prepared = prepared;
                    let gpu_domains = gpu_domains.as_ref();
                    scope.spawn(move || {
                        prepared[range.clone()]
                            .iter()
                            .enumerate()
                            .map(|(offset, (_, req, scalars))| {
                                let position = range.start + offset;
                                match gpu_domains.and_then(|values| values.get(position)) {
                                    Some((public_key, domain)) => cpu
                                        .proof_verify_prepared_msm_with_public_key_and_domain(
                                            req,
                                            scalars,
                                            public_key,
                                            Some(*domain),
                                        ),
                                    None => cpu
                                        .proof_verify_prepared_msm_for_prehashed_disclosed_scalars(
                                            req, scalars,
                                        ),
                                }
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("proof-verify prepare worker panicked"))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Ok(results)
        })?
    } else {
        prepared
            .iter()
            .enumerate()
            .map(|(position, (_, req, scalars))| {
                match gpu_domains.as_ref().and_then(|values| values.get(position)) {
                    Some((public_key, domain)) => cpu
                        .proof_verify_prepared_msm_with_public_key_and_domain(
                            req,
                            scalars,
                            public_key,
                            Some(*domain),
                        ),
                    None => {
                        cpu.proof_verify_prepared_msm_for_prehashed_disclosed_scalars(req, scalars)
                    }
                }
            })
            .collect::<Result<Vec<_>>>()?
    };

    if let Some(keys) = cache_keys.as_ref() {
        cache_proof_verify_prepared_terms(keys, &paired_terms);
    }
    proof_verify_terms_and_meta_from_paired(paired_terms)
}

pub(super) fn prepare_unique_key_proof_domains(
    kernel: &GpuKernelLibrary,
    prepared: &[(usize, ProofVerifyRequest, Vec<BBSplusMessage>)],
) -> Result<Vec<(BBSplusPublicKey, Scalar)>> {
    let ranges = worker_ranges_with_count(
        prepared.len(),
        proof_verify_unique_key_worker_count(prepared.len()),
    );
    let mut inputs = thread::scope(|scope| {
        ranges
            .into_iter()
            .map(|range| {
                scope.spawn(move || {
                    prepared[range.clone()]
                        .iter()
                        .enumerate()
                        .map(|(offset, (_, req, _))| {
                            let public_key = parse_public_key(&req.public_key)?;
                            let input = PoKSignature::<BBSplus<Bls12381Sha256>>::
                                proof_verify_domain_input_for_total_message_count(
                                    &public_key,
                                    req.total_message_count,
                                    Some(&req.disclosed_indexes),
                                    Some(&req.header),
                                )
                                .map_err(|error| Error::Zkryptium(error.to_string()))?;
                            Ok((range.start + offset, public_key, input))
                        })
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("proof domain input worker panicked"))
            .collect::<Result<Vec<_>>>()
            .map(|chunks| chunks.into_iter().flatten().collect::<Vec<_>>())
    })?;
    inputs.sort_by_key(|(position, _, _)| *position);
    let domain_inputs = inputs
        .iter()
        .map(|(_, _, input)| input.clone())
        .collect::<Vec<_>>();
    let domain_scalars = gpu_precompute_signature_e_scalars(kernel, domain_inputs, true)?;
    if domain_scalars.len() != inputs.len() {
        return Err(Error::Gpu(
            "GPU proof domain scalar count mismatch".to_owned(),
        ));
    }
    Ok(inputs
        .into_iter()
        .zip(domain_scalars)
        .map(|((_, public_key, _), scalar)| (public_key, scalar.value))
        .collect())
}

pub(super) type ProofVerifyPreparedTuple =
    (BbsMsmTerms, BbsMsmTerms, Scalar, ProofVerifyPreparedMeta);

pub(super) fn proof_verify_prepared_cache(
) -> &'static Mutex<HashMap<[u8; 32], ProofVerifyPreparedTuple>> {
    static CACHE: OnceLock<Mutex<HashMap<[u8; 32], ProofVerifyPreparedTuple>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn verify_prepared_b_cache() -> &'static RwLock<HashMap<[u8; 32], Vec<u8>>> {
    static CACHE: OnceLock<RwLock<HashMap<[u8; 32], Vec<u8>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(super) fn sign_prepared_cache() -> &'static RwLock<HashMap<[u8; 32], (Vec<u8>, BBSplusMessage)>>
{
    static CACHE: OnceLock<RwLock<HashMap<[u8; 32], (Vec<u8>, BBSplusMessage)>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(super) fn sign_prepared_cache_key(req: &SignRequest, scalars: &[BBSplusMessage]) -> [u8; 32] {
    let mut key = Sha256::new();
    key.update(b"CUBBS_BBS_SIGN_PREPARED_CACHE_V1");
    for value in [&req.secret_key, &req.public_key, &req.header] {
        key.update((value.len() as u64).to_be_bytes());
        key.update(value);
    }
    key.update((scalars.len() as u64).to_be_bytes());
    for scalar in scalars {
        key.update(scalar.value.to_bytes_be());
    }
    key.finalize().into()
}

pub(super) fn cache_sign_prepared_values(keys: &[[u8; 32]], values: &[(Vec<u8>, BBSplusMessage)]) {
    if keys.len() != values.len() {
        return;
    }
    let mut cache = sign_prepared_cache().write().unwrap();
    const MAX_CACHED_PREPARED_SIGNS: usize = 4096;
    if cache.len() + keys.len() > MAX_CACHED_PREPARED_SIGNS {
        cache.clear();
    }
    for (key, value) in keys.iter().zip(values.iter()) {
        cache.insert(*key, value.clone());
    }
}

pub(super) fn verify_prepared_b_cache_key(
    req: &VerifyRequest,
    scalars: &[BBSplusMessage],
) -> [u8; 32] {
    let mut key = Sha256::new();
    key.update(b"CUBBS_BBS_VERIFY_PREPARED_B_CACHE_V1");
    for value in [&req.public_key, &req.header] {
        key.update((value.len() as u64).to_be_bytes());
        key.update(value);
    }
    key.update((scalars.len() as u64).to_be_bytes());
    for scalar in scalars {
        key.update(scalar.value.to_bytes_be());
    }
    key.finalize().into()
}

pub(super) fn cache_verify_prepared_b(keys: &[[u8; 32]], result: &BbsBatchRunResult<Vec<u8>>) {
    if keys.len() != result.results.len() {
        return;
    }
    let mut cache = verify_prepared_b_cache().write().unwrap();
    const MAX_CACHED_PREPARED_B: usize = 8192;
    if cache.len() + keys.len() > MAX_CACHED_PREPARED_B {
        cache.clear();
    }
    for (key, item) in keys.iter().zip(result.results.iter()) {
        if let Ok(point) = &item.result {
            cache.insert(*key, point.clone());
        }
    }
}

pub(super) fn proof_verify_prepared_cache_key(
    req: &ProofVerifyRequest,
    scalars: &[BBSplusMessage],
) -> [u8; 32] {
    let mut key = Sha256::new();
    key.update(b"CUBBS_BBS_PROOF_VERIFY_PREPARED_CACHE_V1");
    for value in [&req.public_key, &req.proof, &req.header] {
        key.update((value.len() as u64).to_be_bytes());
        key.update(value);
    }
    key.update((req.total_message_count as u64).to_be_bytes());
    key.update((req.disclosed_indexes.len() as u64).to_be_bytes());
    for index in &req.disclosed_indexes {
        key.update((*index as u64).to_be_bytes());
    }
    for scalar in scalars {
        key.update(scalar.value.to_bytes_be());
    }
    key.finalize().into()
}

pub(super) fn cache_proof_verify_prepared_terms(
    keys: &[[u8; 32]],
    paired_terms: &[ProofVerifyPreparedTuple],
) {
    if keys.len() != paired_terms.len() {
        return;
    }
    let mut cache = proof_verify_prepared_cache().lock().unwrap();
    const MAX_CACHED_PROOFS: usize = 8192;
    if cache.len() + keys.len() > MAX_CACHED_PROOFS {
        cache.clear();
    }
    for (key, terms) in keys.iter().zip(paired_terms.iter()) {
        cache.insert(key.clone(), terms.clone());
    }
}

pub(super) fn prepare_proof_verify_msm_terms_with_grouped_domains(
    cpu: CuBbsProtocolEngine,
    prepared: &[(usize, ProofVerifyRequest, Vec<BBSplusMessage>)],
) -> Result<Option<Vec<ProofVerifyPreparedTuple>>> {
    if prepared.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let mut order = Vec::<(Vec<u8>, Vec<u8>, usize, Vec<usize>)>::new();
    let mut groups = HashMap::<(Vec<u8>, Vec<u8>, usize, Vec<usize>), Vec<usize>>::new();
    for (position, (_, req, _)) in prepared.iter().enumerate() {
        let key = (
            req.public_key.clone(),
            req.header.clone(),
            req.total_message_count,
            req.disclosed_indexes.clone(),
        );
        match groups.entry(key.clone()) {
            Entry::Occupied(mut entry) => entry.get_mut().push(position),
            Entry::Vacant(entry) => {
                order.push(key);
                entry.insert(vec![position]);
            }
        }
    }

    // Grouping is useful when many proofs share one issuer/domain. With
    // multi-tenant batches, however, a mostly unique key set turns this loop
    // into serial domain generation. Let the caller's range-parallel path
    // handle that high-cardinality case instead.
    if groups.len() > 1 && groups.len().saturating_mul(2) > prepared.len() {
        return Ok(None);
    }

    let mut positioned = Vec::<(usize, ProofVerifyPreparedTuple)>::with_capacity(prepared.len());
    for key in order {
        let positions = groups.remove(&key).unwrap_or_default();
        let Some(&first_position) = positions.first() else {
            continue;
        };
        let first_req = &prepared[first_position].1;
        let shared_public_key = parse_public_key(&first_req.public_key)?;
        let shared_domain =
            PoKSignature::<BBSplus<Bls12381Sha256>>::proof_verify_domain_for_total_message_count(
                &shared_public_key,
                first_req.total_message_count,
                Some(&first_req.disclosed_indexes),
                Some(&first_req.header),
            )
            .map_err(|e| Error::Zkryptium(e.to_string()))?;
        let mut group_results = if positions.len() >= proof_gpu_prepare_parallel_min_batch() {
            let position_chunks = into_worker_chunks(positions);
            thread::scope(|scope| {
                let results = position_chunks
                    .into_iter()
                    .map(|chunk| {
                        let prepared = prepared;
                        let shared_public_key = shared_public_key.clone();
                        scope.spawn(move || {
                            chunk
                                .into_iter()
                                .map(|position| {
                                    let (_, req, scalars) = &prepared[position];
                                    cpu.proof_verify_prepared_msm_with_public_key_and_domain(
                                        req,
                                        scalars,
                                        &shared_public_key,
                                        Some(shared_domain),
                                    )
                                    .map(|terms| (position, terms))
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().expect("proof-verify prepare worker panicked"))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                Ok(results)
            })?
        } else {
            positions
                .iter()
                .map(|&position| {
                    let (_, req, scalars) = &prepared[position];
                    cpu.proof_verify_prepared_msm_with_public_key_and_domain(
                        req,
                        scalars,
                        &shared_public_key,
                        Some(shared_domain),
                    )
                    .map(|terms| (position, terms))
                })
                .collect::<Result<Vec<_>>>()?
        };
        positioned.append(&mut group_results);
    }

    positioned.sort_by_key(|(position, _)| *position);
    Ok(Some(
        positioned
            .into_iter()
            .map(|(_, paired_terms)| paired_terms)
            .collect(),
    ))
}

pub(super) fn proof_verify_terms_and_meta_from_paired(
    paired_terms: Vec<ProofVerifyPreparedTuple>,
) -> Result<(Vec<BbsMsmTerms>, Vec<(Scalar, ProofVerifyPreparedMeta)>)> {
    let mut all_terms = Vec::with_capacity(paired_terms.len() * 2);
    let mut metas = Vec::with_capacity(paired_terms.len());
    for (t1_terms, t2_terms, domain, meta) in paired_terms {
        all_terms.push(t1_terms);
        all_terms.push(t2_terms);
        metas.push((domain, meta));
    }
    Ok((all_terms, metas))
}

pub(super) fn summarize_msm_terms(
    label: &'static str,
    terms: &[BbsMsmTerms],
) -> BbsMsmShapeSummary {
    let min_term_count = terms
        .iter()
        .map(|term| term.compressed_points.len())
        .min()
        .unwrap_or_default();
    let max_term_count = terms
        .iter()
        .map(|term| term.compressed_points.len())
        .max()
        .unwrap_or_default();
    let mut groups = HashMap::<Arc<Vec<Vec<u8>>>, usize>::new();
    let mut term_count_groups = HashMap::<usize, Vec<BbsMsmTerms>>::new();
    for term in terms {
        *groups.entry(term.compressed_points.clone()).or_default() += 1;
        term_count_groups
            .entry(term.compressed_points.len())
            .or_default()
            .push(term.clone());
    }
    let largest_common_point_group = groups.values().copied().max().unwrap_or_default();
    let split_candidate_groups = term_count_groups
        .values()
        .filter(|group| group.len() >= 2 && single_variable_point_index(group).is_some())
        .count();
    let largest_split_candidate_group = term_count_groups
        .values()
        .filter(|group| group.len() >= 2 && single_variable_point_index(group).is_some())
        .map(Vec::len)
        .max()
        .unwrap_or_default();
    BbsMsmShapeSummary {
        label,
        item_count: terms.len(),
        min_term_count,
        max_term_count,
        common_point_groups: groups.len(),
        largest_common_point_group,
        single_variable_point_index: single_variable_point_index(terms),
        term_count_groups: term_count_groups.len(),
        split_candidate_groups,
        largest_split_candidate_group,
    }
}

pub(super) fn g1_msm_batch_proof_verify_pairs(
    kernel: &GpuKernelLibrary,
    all_terms: &[BbsMsmTerms],
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    g1_msm_batch_proof_verify_pairs_inner(kernel, all_terms, fail_fast, true)
}

pub(super) fn g1_msm_batch_proof_verify_pairs_inner(
    kernel: &GpuKernelLibrary,
    all_terms: &[BbsMsmTerms],
    fail_fast: bool,
    allow_shape_grouping: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    if all_terms.len() < 2 || all_terms.len() % 2 != 0 {
        return g1_msm_batch_grouped_by_points(kernel, all_terms, fail_fast);
    }

    if allow_shape_grouping && proof_verify_shape_grouping_enabled() {
        let mut groups = HashMap::<Vec<Vec<u8>>, Vec<usize>>::new();
        let mut grouping_valid = true;
        for item in 0..all_terms.len() / 2 {
            let t1 = &all_terms[item * 2];
            let t2 = &all_terms[item * 2 + 1];
            let Some(d) = t1.compressed_points.get(2) else {
                grouping_valid = false;
                break;
            };
            let Some(variable_index) = t2.compressed_points.iter().position(|point| point == d)
            else {
                grouping_valid = false;
                break;
            };
            let mut common_points = t2.compressed_points.as_ref().clone();
            common_points.remove(variable_index);
            groups.entry(common_points).or_default().push(item);
        }
        if grouping_valid {
            if (2..=3).contains(&groups.len()) {
                let jobs = groups
                    .into_values()
                    .map(|items| {
                        let terms = items
                            .iter()
                            .flat_map(|&item| {
                                [all_terms[item * 2].clone(), all_terms[item * 2 + 1].clone()]
                            })
                            .collect::<Vec<_>>();
                        (items, terms)
                    })
                    .collect::<Vec<_>>();
                let results = thread::scope(|scope| {
                    jobs.into_iter()
                        .map(|(items, terms)| {
                            scope.spawn(move || {
                                let result = g1_msm_batch_proof_verify_pairs_inner(
                                    kernel, &terms, fail_fast, false,
                                );
                                result
                                    .results
                                    .into_iter()
                                    .filter_map(|entry| {
                                        let item = items.get(entry.index / 2).copied()?;
                                        Some(BbsBatchItemResult {
                                            index: item * 2 + entry.index % 2,
                                            result: entry.result,
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .flat_map(|worker| {
                            worker
                                .join()
                                .expect("proof-verify shape MSM worker panicked")
                        })
                        .collect::<Vec<_>>()
                });
                return merge_batch_results_with_fail_fast(
                    all_terms.len(),
                    results,
                    fail_fast,
                    None,
                );
            }
        }
    }

    let has_large_t2 = all_terms
        .iter()
        .skip(1)
        .step_by(2)
        .any(|terms| terms.compressed_points.len() > 258);
    let item_count = all_terms.len() / 2;
    let max_t2_terms = all_terms
        .iter()
        .skip(1)
        .step_by(2)
        .map(|terms| terms.compressed_points.len())
        .max()
        .unwrap_or(0);
    if kernel.can_proof_verify_msm_pair()
        && (has_large_t2 || gpu_proof_verify_msm_pair_enabled(item_count, max_t2_terms))
    {
        if let Some(result) = kernel.proof_verify_msm_pair_batch(all_terms, fail_fast) {
            if result.failed == 0 {
                return result;
            }
        }
    }

    let t1_terms = all_terms.iter().step_by(2).cloned().collect::<Vec<_>>();
    let t2_terms = all_terms
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .collect::<Vec<_>>();

    let Some(variable_index) = single_variable_point_index(&t2_terms) else {
        let t1_results = g1_msm_batch_grouped_by_points(kernel, &t1_terms, fail_fast);
        if let Some(t2_results) =
            g1_msm_batch_split_single_variable_term_grouped(kernel, &t2_terms, fail_fast)
        {
            return interleave_g1_msm_pair_results(
                all_terms.len(),
                t1_results,
                t2_results,
                fail_fast,
            );
        }
        if gpu_proof_verify_split_msm_enabled() {
            let t2_results = g1_msm_batch_grouped_by_points(kernel, &t2_terms, fail_fast);
            return interleave_g1_msm_pair_results(
                all_terms.len(),
                t1_results,
                t2_results,
                fail_fast,
            );
        }
        return g1_msm_batch_grouped_by_points(kernel, all_terms, fail_fast);
    };

    let run_t2 = || {
        g1_msm_batch_split_variable_term_grouped(kernel, &t2_terms, variable_index, fail_fast)
            .unwrap_or_else(|| g1_msm_batch_grouped_by_points(kernel, &t2_terms, fail_fast))
    };
    let (t1_results, t2_results) = if t1_terms.len() >= proof_verify_pair_msm_parallel_min_batch() {
        thread::scope(|scope| {
            let t1_handle =
                scope.spawn(|| g1_msm_batch_grouped_by_points(kernel, &t1_terms, fail_fast));
            let t2_results = run_t2();
            (
                t1_handle
                    .join()
                    .expect("proof-verify T1 MSM worker panicked"),
                t2_results,
            )
        })
    } else {
        (
            g1_msm_batch_grouped_by_points(kernel, &t1_terms, fail_fast),
            run_t2(),
        )
    };
    interleave_g1_msm_pair_results(all_terms.len(), t1_results, t2_results, fail_fast)
}

pub(super) fn g1_msm_batch_grouped_by_points(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let use_common_points_matrix_abi = gpu_common_points_msm_abi_enabled();
    g1_msm_batch_grouped_by_points_with_common_points_abi(
        kernel,
        terms,
        fail_fast,
        use_common_points_matrix_abi,
        !use_common_points_matrix_abi && gpu_native_mixed_msm_grouping_enabled(),
    )
}

pub(super) fn g1_msm_batch_grouped_by_points_for_proof_gen_b(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let use_common_points_matrix_abi = gpu_proof_gen_b_common_points_msm_abi_enabled();
    g1_msm_batch_grouped_by_points_with_common_points_abi(
        kernel,
        terms,
        fail_fast,
        use_common_points_matrix_abi,
        !use_common_points_matrix_abi && gpu_native_mixed_msm_grouping_enabled(),
    )
}

pub(super) fn g1_msm_batch_grouped_by_points_with_common_points_abi(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    fail_fast: bool,
    use_common_points_matrix_abi: bool,
    use_native_mixed_grouping: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    if terms.len() < 2 || all_msm_terms_have_common_points(terms) {
        if kernel.can_g1_msm_common_points() && use_common_points_matrix_abi {
            return kernel.g1_msm_common_points_batch(terms, fail_fast);
        }
        return kernel.g1_msm_batch(terms, fail_fast);
    }
    if use_native_mixed_grouping {
        return kernel.g1_msm_batch(terms, fail_fast);
    }

    let mut order = Vec::<Arc<Vec<Vec<u8>>>>::new();
    let mut groups = HashMap::<Arc<Vec<Vec<u8>>>, Vec<usize>>::new();
    for (index, term) in terms.iter().enumerate() {
        let key = term.compressed_points.clone();
        match groups.entry(key.clone()) {
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
            Entry::Vacant(entry) => {
                order.push(key);
                entry.insert(vec![index]);
            }
        }
    }

    if groups.values().all(|indexes| indexes.len() == 1) {
        return kernel.g1_msm_batch(terms, fail_fast);
    }

    let mut group_batches = Vec::<(Vec<usize>, Vec<BbsMsmTerms>)>::new();
    let mut singleton_indexes = Vec::new();
    for key in order {
        let indexes = groups.remove(&key).unwrap_or_default();
        if indexes.len() <= 1 {
            singleton_indexes.extend(indexes);
            continue;
        }
        let grouped_terms = indexes
            .iter()
            .map(|&index| terms[index].clone())
            .collect::<Vec<_>>();
        group_batches.push((indexes, grouped_terms));
    }
    if !singleton_indexes.is_empty() {
        let singleton_terms = singleton_indexes
            .iter()
            .map(|&index| terms[index].clone())
            .collect::<Vec<_>>();
        group_batches.push((singleton_indexes, singleton_terms));
    }

    let group_results = run_g1_msm_point_groups(
        kernel,
        group_batches,
        fail_fast,
        use_common_points_matrix_abi,
    );
    let mut results = (0..terms.len())
        .map(|index| BbsBatchItemResult {
            index,
            result: Err(Error::Gpu("GPU MSM group was not executed".to_owned())),
        })
        .collect::<Vec<_>>();
    for (indexes, grouped_results) in group_results {
        scatter_g1_msm_group_results(&mut results, &indexes, grouped_results, fail_fast);
    }

    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }

    batch_run_result(terms.len(), results)
}

pub(super) fn run_g1_msm_point_groups(
    kernel: &GpuKernelLibrary,
    group_batches: Vec<(Vec<usize>, Vec<BbsMsmTerms>)>,
    fail_fast: bool,
    use_common_points_matrix_abi: bool,
) -> Vec<(Vec<usize>, BbsBatchRunResult<Vec<u8>>)> {
    if kernel.can_g1_msm_common_points()
        && use_common_points_matrix_abi
        && group_batches.len() >= 2
        && group_batches
            .iter()
            .all(|(_, grouped_terms)| all_msm_terms_have_common_points(grouped_terms))
    {
        let indexes = group_batches
            .iter()
            .map(|(indexes, _)| indexes.clone())
            .collect::<Vec<_>>();
        let grouped_terms = group_batches
            .into_iter()
            .map(|(_, grouped_terms)| grouped_terms)
            .collect::<Vec<_>>();
        return indexes
            .into_iter()
            .zip(kernel.g1_msm_common_point_groups_batch(&grouped_terms, fail_fast))
            .collect();
    }

    if group_batches.len() >= gpu_point_group_parallel_min_groups() {
        return thread::scope(|scope| {
            group_batches
                .into_iter()
                .map(|(indexes, grouped_terms)| {
                    scope.spawn(move || {
                        let grouped_results = kernel.g1_msm_batch(&grouped_terms, fail_fast);
                        (indexes, grouped_results)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("point MSM group worker panicked"))
                .collect::<Vec<_>>()
        });
    }

    group_batches
        .into_iter()
        .map(|(indexes, grouped_terms)| {
            let grouped_results = kernel.g1_msm_batch(&grouped_terms, fail_fast);
            (indexes, grouped_results)
        })
        .collect()
}

pub(super) fn g1_msm_route_summary_grouped_by_points(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    fail_fast: bool,
) -> BbsMsmRouteSummary {
    if terms.is_empty() {
        return BbsMsmRouteSummary::default();
    }
    if terms.len() < 2 || all_msm_terms_have_common_points(terms) {
        if gpu_common_points_msm_abi_enabled() && all_msm_terms_have_common_points(terms) {
            return summarize_msm_route_codes(
                kernel.g1_msm_common_points_batch_routes(terms, fail_fast),
            );
        }
        return summarize_msm_route_codes(kernel.g1_msm_batch_routes(terms, fail_fast));
    }
    if gpu_native_mixed_msm_grouping_enabled() && !gpu_common_points_msm_abi_enabled() {
        return summarize_msm_route_codes(kernel.g1_msm_batch_routes(terms, fail_fast));
    }

    let mut order = Vec::<Arc<Vec<Vec<u8>>>>::new();
    let mut groups = HashMap::<Arc<Vec<Vec<u8>>>, Vec<usize>>::new();
    for (index, term) in terms.iter().enumerate() {
        let key = term.compressed_points.clone();
        match groups.entry(key.clone()) {
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
            Entry::Vacant(entry) => {
                order.push(key);
                entry.insert(vec![index]);
            }
        }
    }

    if groups.values().all(|indexes| indexes.len() == 1) {
        return summarize_msm_route_codes(kernel.g1_msm_batch_routes(terms, fail_fast));
    }

    let mut summary = BbsMsmRouteSummary::default();
    let mut singleton_indexes = Vec::new();
    for key in order {
        let indexes = groups.remove(&key).unwrap_or_default();
        if indexes.len() <= 1 {
            singleton_indexes.extend(indexes);
            continue;
        }
        let grouped_terms = indexes
            .iter()
            .map(|&index| terms[index].clone())
            .collect::<Vec<_>>();
        if gpu_common_points_msm_abi_enabled() {
            summary.merge(summarize_msm_route_codes(
                kernel.g1_msm_common_points_batch_routes(&grouped_terms, fail_fast),
            ));
        } else {
            summary.merge(summarize_msm_route_codes(
                kernel.g1_msm_batch_routes(&grouped_terms, fail_fast),
            ));
        }
    }
    if !singleton_indexes.is_empty() {
        let singleton_terms = singleton_indexes
            .iter()
            .map(|&index| terms[index].clone())
            .collect::<Vec<_>>();
        summary.merge(summarize_msm_route_codes(
            kernel.g1_msm_batch_routes(&singleton_terms, fail_fast),
        ));
    }
    summary
}

pub(super) fn g1_msm_route_summary_split_for_bench(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    fail_fast: bool,
) -> BbsMsmRouteSummary {
    if terms.is_empty() {
        return BbsMsmRouteSummary::default();
    }
    if let Some(variable_index) = single_variable_point_index(terms) {
        return g1_msm_route_summary_split_variable_term_grouped(
            kernel,
            terms,
            variable_index,
            fail_fast,
        );
    }

    let mut order = Vec::<usize>::new();
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for (index, term) in terms.iter().enumerate() {
        let term_count = term.compressed_points.len();
        match groups.entry(term_count) {
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
            Entry::Vacant(entry) => {
                order.push(term_count);
                entry.insert(vec![index]);
            }
        }
    }

    if groups.values().all(|indexes| indexes.len() == 1) {
        return g1_msm_route_summary_grouped_by_points(kernel, terms, fail_fast);
    }
    if groups.len() == 1 {
        return g1_msm_route_summary_grouped_by_points(kernel, terms, fail_fast);
    }

    let mut summary = BbsMsmRouteSummary::default();
    for term_count in order {
        let indexes = groups.remove(&term_count).unwrap_or_default();
        let grouped_terms = indexes
            .iter()
            .map(|&index| terms[index].clone())
            .collect::<Vec<_>>();
        summary.merge(g1_msm_route_summary_split_for_bench(
            kernel,
            &grouped_terms,
            fail_fast,
        ));
    }
    summary
}

pub(super) fn g1_msm_route_summary_split_variable_term_grouped(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    variable_index: usize,
    fail_fast: bool,
) -> BbsMsmRouteSummary {
    if terms.len() < 2 {
        return g1_msm_route_summary_grouped_by_points(kernel, terms, fail_fast);
    }
    if let Some((common_terms, variable_terms)) = split_variable_msm_terms(terms, variable_index) {
        let mut summary = g1_msm_route_summary_grouped_by_points(kernel, &common_terms, fail_fast);
        summary.merge(g1_msm_route_summary_grouped_by_points(
            kernel,
            &variable_terms,
            fail_fast,
        ));
        return summary;
    }

    let mut order = Vec::<usize>::new();
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for (index, term) in terms.iter().enumerate() {
        let term_count = term.compressed_points.len();
        match groups.entry(term_count) {
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
            Entry::Vacant(entry) => {
                order.push(term_count);
                entry.insert(vec![index]);
            }
        }
    }

    if groups.values().all(|indexes| indexes.len() == 1) {
        return g1_msm_route_summary_grouped_by_points(kernel, terms, fail_fast);
    }
    if groups.len() == 1 {
        return g1_msm_route_summary_grouped_by_points(kernel, terms, fail_fast);
    }

    let mut summary = BbsMsmRouteSummary::default();
    for term_count in order {
        let indexes = groups.remove(&term_count).unwrap_or_default();
        let grouped_terms = indexes
            .iter()
            .map(|&index| terms[index].clone())
            .collect::<Vec<_>>();
        summary.merge(g1_msm_route_summary_split_variable_term_grouped(
            kernel,
            &grouped_terms,
            variable_index,
            fail_fast,
        ));
    }
    summary
}

pub(super) fn summarize_msm_route_codes(routes: BbsBatchRunResult<u8>) -> BbsMsmRouteSummary {
    let mut summary = BbsMsmRouteSummary::default();
    for item in routes.results {
        match item.result {
            Ok(code) => summary.record_code(code),
            Err(_) => summary.unknown += 1,
        }
    }
    summary
}

pub(super) fn scatter_g1_msm_group_results(
    results: &mut [BbsBatchItemResult<Vec<u8>>],
    original_indexes: &[usize],
    grouped_results: BbsBatchRunResult<Vec<u8>>,
    fail_fast: bool,
) {
    for grouped_item in grouped_results.results {
        let Some(&original_index) = original_indexes.get(grouped_item.index) else {
            continue;
        };
        if let Some(slot) = results.get_mut(original_index) {
            *slot = BbsBatchItemResult {
                index: original_index,
                result: grouped_item.result,
            };
        }
        if fail_fast && results[original_index].result.is_err() {
            break;
        }
    }
}

pub(super) fn g1_msm_batch_split_variable_term_grouped(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    variable_index: usize,
    fail_fast: bool,
) -> Option<BbsBatchRunResult<Vec<u8>>> {
    if !gpu_split_variable_msm_enabled() || terms.len() < 2 {
        return None;
    }
    if let Some(result) = g1_msm_batch_split_variable_term(kernel, terms, variable_index, fail_fast)
    {
        return Some(result);
    }

    let mut order = Vec::<usize>::new();
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for (index, term) in terms.iter().enumerate() {
        let term_count = term.compressed_points.len();
        match groups.entry(term_count) {
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
            Entry::Vacant(entry) => {
                order.push(term_count);
                entry.insert(vec![index]);
            }
        }
    }

    if groups.values().all(|indexes| indexes.len() == 1) {
        return None;
    }

    let group_batches = order
        .into_iter()
        .map(|term_count| {
            let indexes = groups.remove(&term_count).unwrap_or_default();
            let grouped_terms = indexes
                .iter()
                .map(|&index| terms[index].clone())
                .collect::<Vec<_>>();
            (indexes, grouped_terms)
        })
        .collect::<Vec<_>>();

    let group_results = run_split_variable_msm_groups(
        kernel,
        group_batches,
        variable_index,
        fail_fast,
        "GPU MSM split group was not executed",
    );
    let used_split = group_results.iter().any(|(_, _, used_split)| *used_split);
    let mut results = (0..terms.len())
        .map(|index| BbsBatchItemResult {
            index,
            result: Err(Error::Gpu(
                "GPU MSM split group was not executed".to_owned(),
            )),
        })
        .collect::<Vec<_>>();
    for (indexes, grouped_results, _) in group_results {
        scatter_g1_msm_group_results(&mut results, &indexes, grouped_results, fail_fast);
    }

    if !used_split {
        return None;
    }
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }

    Some(batch_run_result(terms.len(), results))
}

pub(super) fn g1_msm_batch_split_single_variable_term_grouped(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    fail_fast: bool,
) -> Option<BbsBatchRunResult<Vec<u8>>> {
    if !gpu_split_variable_msm_enabled() || terms.len() < 2 {
        return None;
    }
    if let Some(variable_index) = single_variable_point_index(terms) {
        return g1_msm_batch_split_variable_term_grouped(kernel, terms, variable_index, fail_fast);
    }

    let mut order = Vec::<usize>::new();
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for (index, term) in terms.iter().enumerate() {
        let term_count = term.compressed_points.len();
        match groups.entry(term_count) {
            Entry::Occupied(mut entry) => entry.get_mut().push(index),
            Entry::Vacant(entry) => {
                order.push(term_count);
                entry.insert(vec![index]);
            }
        }
    }

    if groups.values().all(|indexes| indexes.len() == 1) {
        return None;
    }

    let group_batches = order
        .into_iter()
        .map(|term_count| {
            let indexes = groups.remove(&term_count).unwrap_or_default();
            let grouped_terms = indexes
                .iter()
                .map(|&index| terms[index].clone())
                .collect::<Vec<_>>();
            (indexes, grouped_terms)
        })
        .collect::<Vec<_>>();
    let group_results = run_auto_split_msm_groups(
        kernel,
        group_batches,
        fail_fast,
        "GPU MSM auto split group was not executed",
    );
    let used_split = group_results.iter().any(|(_, _, used_split)| *used_split);
    let mut results = (0..terms.len())
        .map(|index| BbsBatchItemResult {
            index,
            result: Err(Error::Gpu(
                "GPU MSM auto split group was not executed".to_owned(),
            )),
        })
        .collect::<Vec<_>>();
    for (indexes, grouped_results, _) in group_results {
        scatter_g1_msm_group_results(&mut results, &indexes, grouped_results, fail_fast);
    }

    if !used_split {
        return None;
    }
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }

    Some(batch_run_result(terms.len(), results))
}

pub(super) fn run_split_variable_msm_groups(
    kernel: &GpuKernelLibrary,
    group_batches: Vec<(Vec<usize>, Vec<BbsMsmTerms>)>,
    variable_index: usize,
    fail_fast: bool,
    error_message: &'static str,
) -> Vec<(Vec<usize>, BbsBatchRunResult<Vec<u8>>, bool)> {
    if group_batches.len() >= gpu_mixed_shape_parallel_min_groups() {
        return thread::scope(|scope| {
            group_batches
                .into_iter()
                .map(|(indexes, grouped_terms)| {
                    scope.spawn(move || {
                        let (grouped_results, used_split) = run_split_variable_msm_group(
                            kernel,
                            &grouped_terms,
                            variable_index,
                            fail_fast,
                            error_message,
                        );
                        (indexes, grouped_results, used_split)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("split MSM group worker panicked"))
                .collect::<Vec<_>>()
        });
    }

    group_batches
        .into_iter()
        .map(|(indexes, grouped_terms)| {
            let (grouped_results, used_split) = run_split_variable_msm_group(
                kernel,
                &grouped_terms,
                variable_index,
                fail_fast,
                error_message,
            );
            (indexes, grouped_results, used_split)
        })
        .collect()
}

pub(super) fn run_split_variable_msm_group(
    kernel: &GpuKernelLibrary,
    grouped_terms: &[BbsMsmTerms],
    variable_index: usize,
    fail_fast: bool,
    error_message: &'static str,
) -> (BbsBatchRunResult<Vec<u8>>, bool) {
    if grouped_terms.is_empty() {
        return (
            batch_call_error(
                grouped_terms.len(),
                Error::Gpu(error_message.to_owned()),
                fail_fast,
            ),
            false,
        );
    }
    if grouped_terms.len() > 1 {
        if let Some(split_results) =
            g1_msm_batch_split_variable_term(kernel, grouped_terms, variable_index, fail_fast)
        {
            return (split_results, true);
        }
    }
    (
        g1_msm_batch_grouped_by_points(kernel, grouped_terms, fail_fast),
        false,
    )
}

pub(super) fn run_auto_split_msm_groups(
    kernel: &GpuKernelLibrary,
    group_batches: Vec<(Vec<usize>, Vec<BbsMsmTerms>)>,
    fail_fast: bool,
    error_message: &'static str,
) -> Vec<(Vec<usize>, BbsBatchRunResult<Vec<u8>>, bool)> {
    if group_batches.len() >= gpu_mixed_shape_parallel_min_groups() {
        return thread::scope(|scope| {
            group_batches
                .into_iter()
                .map(|(indexes, grouped_terms)| {
                    scope.spawn(move || {
                        let (grouped_results, used_split) = run_auto_split_msm_group(
                            kernel,
                            &grouped_terms,
                            fail_fast,
                            error_message,
                        );
                        (indexes, grouped_results, used_split)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("auto split MSM group worker panicked"))
                .collect::<Vec<_>>()
        });
    }

    group_batches
        .into_iter()
        .map(|(indexes, grouped_terms)| {
            let (grouped_results, used_split) =
                run_auto_split_msm_group(kernel, &grouped_terms, fail_fast, error_message);
            (indexes, grouped_results, used_split)
        })
        .collect()
}

pub(super) fn run_auto_split_msm_group(
    kernel: &GpuKernelLibrary,
    grouped_terms: &[BbsMsmTerms],
    fail_fast: bool,
    error_message: &'static str,
) -> (BbsBatchRunResult<Vec<u8>>, bool) {
    let Some(variable_index) = single_variable_point_index(grouped_terms) else {
        return (
            g1_msm_batch_grouped_by_points(kernel, grouped_terms, fail_fast),
            false,
        );
    };
    run_split_variable_msm_group(
        kernel,
        grouped_terms,
        variable_index,
        fail_fast,
        error_message,
    )
}

pub(super) fn interleave_g1_msm_pair_results(
    attempted: usize,
    t1_results: BbsBatchRunResult<Vec<u8>>,
    t2_results: BbsBatchRunResult<Vec<u8>>,
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let pair_count = attempted / 2;
    let mut results = (0..attempted)
        .map(|index| BbsBatchItemResult {
            index,
            result: Err(Error::Gpu("GPU MSM pair was not executed".to_owned())),
        })
        .collect::<Vec<_>>();
    for (pair_index, t1_item) in t1_results.results.into_iter().enumerate() {
        let index = pair_index * 2;
        if index < attempted {
            results[index] = BbsBatchItemResult {
                index,
                result: t1_item.result,
            };
        }
    }
    for (pair_index, t2_item) in t2_results.results.into_iter().enumerate() {
        let index = (pair_index * 2) + 1;
        if index < attempted {
            results[index] = BbsBatchItemResult {
                index,
                result: t2_item.result,
            };
        }
    }
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }

    if results.len() < attempted && fail_fast {
        results.truncate(results.len().min(pair_count * 2));
    }

    batch_run_result(attempted, results)
}

pub(super) fn g1_msm_batch_split_variable_term(
    kernel: &GpuKernelLibrary,
    terms: &[BbsMsmTerms],
    variable_index: usize,
    fail_fast: bool,
) -> Option<BbsBatchRunResult<Vec<u8>>> {
    if gpu_common_plus_variable_msm_enabled() && kernel.can_g1_msm_common_points_plus_variable() {
        if let Some(result) =
            kernel.g1_msm_common_points_plus_variable_batch(terms, variable_index, fail_fast)
        {
            return Some(result);
        }
    }
    let (common_terms, variable_terms) = split_variable_msm_terms(terms, variable_index)?;
    let (common_results, variable_results) =
        if terms.len() >= gpu_split_variable_parallel_min_batch() {
            thread::scope(|scope| {
                let common_handle = scope
                    .spawn(|| g1_msm_batch_grouped_by_points(kernel, &common_terms, fail_fast));
                let variable_results =
                    g1_msm_batch_grouped_by_points(kernel, &variable_terms, fail_fast);
                (
                    common_handle
                        .join()
                        .expect("split MSM common worker panicked"),
                    variable_results,
                )
            })
        } else {
            (
                g1_msm_batch_grouped_by_points(kernel, &common_terms, fail_fast),
                g1_msm_batch_grouped_by_points(kernel, &variable_terms, fail_fast),
            )
        };
    Some(combine_split_g1_msm_results(
        kernel,
        terms.len(),
        common_results,
        variable_results,
        fail_fast,
    ))
}

pub(super) fn split_variable_msm_terms(
    terms: &[BbsMsmTerms],
    variable_index: usize,
) -> Option<(Vec<BbsMsmTerms>, Vec<BbsMsmTerms>)> {
    if !gpu_split_variable_msm_enabled() || terms.len() < 2 {
        return None;
    }
    let first = terms.first()?;
    if variable_index >= first.compressed_points.len()
        || first.compressed_points.len() != first.scalar_be.len()
        || first.compressed_points.len() <= 1
        || first.compressed_points.len() < gpu_split_variable_msm_min_terms()
    {
        return None;
    }
    if terms.iter().any(|term| {
        term.compressed_points.len() != first.compressed_points.len()
            || term.scalar_be.len() != first.scalar_be.len()
    }) {
        return None;
    }

    let mut common_terms = Vec::with_capacity(terms.len());
    let mut variable_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let mut common_points = Vec::with_capacity(term.compressed_points.len() - 1);
        let mut common_scalars = Vec::with_capacity(term.scalar_be.len() - 1);
        for (index, (point, scalar)) in term
            .compressed_points
            .iter()
            .zip(term.scalar_be.iter())
            .enumerate()
        {
            if index == variable_index {
                continue;
            }
            common_points.push(point.clone());
            common_scalars.push(scalar.clone());
        }
        common_terms.push(BbsMsmTerms {
            compressed_points: common_points.into(),
            scalar_be: common_scalars,
        });
        variable_terms.push(BbsMsmTerms {
            compressed_points: vec![term.compressed_points[variable_index].clone()].into(),
            scalar_be: vec![term.scalar_be[variable_index].clone()],
        });
    }

    if !all_msm_terms_have_common_points(&common_terms) {
        return None;
    }

    Some((common_terms, variable_terms))
}

pub(super) fn combine_split_g1_msm_results(
    kernel: &GpuKernelLibrary,
    attempted: usize,
    common_results: BbsBatchRunResult<Vec<u8>>,
    variable_results: BbsBatchRunResult<Vec<u8>>,
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    if kernel.can_g1_add()
        && gpu_native_g1_add_enabled()
        && common_results.failed == 0
        && variable_results.failed == 0
        && common_results.results.len() == attempted
        && variable_results.results.len() == attempted
    {
        let pairs = common_results
            .results
            .iter()
            .zip(variable_results.results.iter())
            .map(|(common, variable)| {
                Ok((
                    common.result.as_ref().map_err(Clone::clone)?.clone(),
                    variable.result.as_ref().map_err(Clone::clone)?.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>();
        if let Ok(pairs) = pairs {
            let native_results = kernel.g1_add_batch(&pairs, fail_fast);
            if native_results.failed == 0 && native_results.results.len() == attempted {
                return native_results;
            }
        }
    }

    let mut results = Vec::with_capacity(attempted);
    for (index, (common, variable)) in common_results
        .results
        .into_iter()
        .zip(variable_results.results.into_iter())
        .enumerate()
    {
        let result = common
            .result
            .and_then(|common_point| {
                variable
                    .result
                    .map(|variable_point| (common_point, variable_point))
            })
            .and_then(|(common_point, variable_point)| {
                add_compressed_g1_points(&common_point, &variable_point)
            });
        let is_ok = result.is_ok();
        results.push(BbsBatchItemResult { index, result });
        if fail_fast && !is_ok {
            break;
        }
    }

    batch_run_result(attempted, results)
}

pub(super) fn single_variable_point_index(terms: &[BbsMsmTerms]) -> Option<usize> {
    let first = terms.first()?;
    let term_count = first.compressed_points.len();
    if terms.len() < 2 || term_count == 0 {
        return None;
    }
    if terms
        .iter()
        .any(|term| term.compressed_points.len() != term_count)
    {
        return None;
    }

    let variable_indexes = (0..term_count)
        .filter(|&point_index| {
            let base = &first.compressed_points[point_index];
            terms
                .iter()
                .skip(1)
                .any(|term| term.compressed_points[point_index] != *base)
        })
        .collect::<Vec<_>>();
    match variable_indexes.as_slice() {
        [index] => Some(*index),
        _ => None,
    }
}

pub(super) fn all_msm_terms_have_common_points(terms: &[BbsMsmTerms]) -> bool {
    let Some(first) = terms.first() else {
        return false;
    };
    terms.iter().skip(1).all(|term| {
        term.compressed_points.len() == first.compressed_points.len()
            && term
                .compressed_points
                .iter()
                .zip(first.compressed_points.iter())
                .all(|(candidate, base)| candidate == base)
    })
}

pub(super) fn add_compressed_g1_points(lhs: &[u8], rhs: &[u8]) -> Result<Vec<u8>> {
    let lhs = parse_compressed_g1(lhs)?;
    let rhs = parse_compressed_g1(rhs)?;
    Ok((lhs + rhs).to_affine().to_compressed().to_vec())
}

pub(super) fn parse_compressed_g1(bytes: &[u8]) -> Result<G1Projective> {
    let compressed = <[u8; 48]>::try_from(bytes).map_err(|_| {
        Error::Gpu(format!(
            "compressed G1 point has invalid length: {}",
            bytes.len()
        ))
    })?;
    let point = G1Affine::from_compressed(&compressed);
    Option::<G1Affine>::from(point)
        .map(G1Projective::from)
        .ok_or(Error::Gpu("compressed G1 point is invalid".to_owned()))
}

pub(super) fn finalize_proof_gen_t2_batch(
    cpu: CuBbsProtocolEngine,
    kernel: Option<&GpuKernelLibrary>,
    attempted: usize,
    states_and_items: Vec<ProofGenT2PreparedItem>,
    t2_results: Vec<BbsBatchItemResult<Vec<u8>>>,
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let items = states_and_items
        .into_iter()
        .zip(t2_results)
        .map(|((index, req, _scalars, _compressed_b, state), t2_item)| (index, req, state, t2_item))
        .collect::<Vec<_>>();

    if let Some(kernel) = kernel
        .filter(|kernel| kernel.can_precompute_message_scalars() && gpu_challenge_scalar_enabled())
    {
        let challenge_inputs = items
            .iter()
            .map(|(_, req, state, t2_item)| {
                let t2 = t2_item.result.as_ref().map_err(Clone::clone)?;
                cpu.proof_gen_challenge_input_prehashed_b_and_t2(req, state, t2)
            })
            .collect::<Result<Vec<_>>>();
        if let Ok(challenge_inputs) = challenge_inputs {
            if let Ok(challenges) =
                gpu_precompute_challenge_scalars(kernel, challenge_inputs, fail_fast)
            {
                if challenges.len() == items.len() {
                    let mut results = if items.len() >= proof_gen_finalize_parallel_min_batch() {
                        let chunks = into_proof_gen_worker_chunks(
                            items.into_iter().zip(challenges).collect::<Vec<_>>(),
                        );
                        thread::scope(|scope| {
                            chunks
                                .into_iter()
                                .map(|chunk| {
                                    scope.spawn(move || {
                                        chunk
                                            .into_iter()
                                            .map(|((index, req, state, t2_item), challenge)| {
                                                let result = t2_item.result.and_then(|t2| {
                                                    cpu.proof_gen_finalize_prehashed_b_and_t2_with_challenge(
                                                        req, state, t2, challenge,
                                                    )
                                                });
                                                BbsBatchItemResult { index, result }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                })
                                .collect::<Vec<_>>()
                                .into_iter()
                                .map(|handle| {
                                    handle.join().expect("proof-gen finalize worker panicked")
                                })
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                    } else {
                        items
                            .into_iter()
                            .zip(challenges)
                            .map(|((index, req, state, t2_item), challenge)| {
                                let result = t2_item.result.and_then(|t2| {
                                    cpu.proof_gen_finalize_prehashed_b_and_t2_with_challenge(
                                        req, state, t2, challenge,
                                    )
                                });
                                BbsBatchItemResult { index, result }
                            })
                            .collect::<Vec<_>>()
                    };

                    results.sort_by_key(|item| item.index);
                    if fail_fast {
                        if let Some(first_error) =
                            results.iter().position(|item| item.result.is_err())
                        {
                            results.truncate(first_error + 1);
                        }
                    }

                    return batch_run_result(attempted, results);
                }
            }
        }
    }

    let mut results = if items.len() >= proof_gen_finalize_parallel_min_batch() {
        let chunks = into_proof_gen_worker_chunks(items);
        thread::scope(|scope| {
            chunks
                .into_iter()
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .into_iter()
                            .map(|(index, req, state, t2_item)| {
                                let result = t2_item.result.and_then(|t2| {
                                    cpu.proof_gen_finalize_prehashed_b_and_t2(req, state, t2)
                                });
                                BbsBatchItemResult { index, result }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("proof-gen finalize worker panicked"))
                .flatten()
                .collect::<Vec<_>>()
        })
    } else {
        items
            .into_iter()
            .map(|(index, req, state, t2_item)| {
                let result = t2_item
                    .result
                    .and_then(|t2| cpu.proof_gen_finalize_prehashed_b_and_t2(req, state, t2));
                BbsBatchItemResult { index, result }
            })
            .collect::<Vec<_>>()
    };

    results.sort_by_key(|item| item.index);
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }

    batch_run_result(attempted, results)
}

pub(super) fn finalize_proof_gen_t2_compressed_batch(
    cpu: CuBbsProtocolEngine,
    kernel: Option<&GpuKernelLibrary>,
    attempted: usize,
    states_and_items: Vec<ProofGenT2CompressedPreparedItem>,
    t2_results: Vec<BbsBatchItemResult<Vec<u8>>>,
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let items = states_and_items
        .into_iter()
        .zip(t2_results)
        .map(|((index, req, _scalars, _compressed_b, state), t2_item)| (index, req, state, t2_item))
        .collect::<Vec<_>>();
    if let Some(kernel) = kernel
        .filter(|kernel| kernel.can_precompute_message_scalars() && gpu_challenge_scalar_enabled())
    {
        let challenge_inputs = items
            .iter()
            .map(|(_, req, state, t2_item)| {
                let t2 = t2_item.result.as_ref().map_err(Clone::clone)?;
                cpu.proof_gen_challenge_input_compressed_state_and_t2(req, state, t2)
            })
            .collect::<Result<Vec<_>>>();
        if let Ok(challenge_inputs) = challenge_inputs {
            if let Ok(challenges) =
                gpu_precompute_challenge_scalars(kernel, challenge_inputs, fail_fast)
            {
                if challenges.len() == items.len() {
                    let mut results = thread::scope(|scope| {
                        into_proof_gen_worker_chunks(
                            items.into_iter().zip(challenges).collect::<Vec<_>>(),
                        )
                        .into_iter()
                        .map(|chunk| {
                            scope.spawn(move || {
                                chunk
                                    .into_iter()
                                    .map(|((index, req, state, t2_item), challenge)| {
                                        let result = t2_item.result.and_then(|_| {
                                            cpu.proof_gen_finalize_compressed_state_with_challenge(
                                                req, state, challenge,
                                            )
                                        });
                                        BbsBatchItemResult { index, result }
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|handle| {
                            handle
                                .join()
                                .expect("compressed GPU challenge finalize worker panicked")
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                    });
                    results.sort_by_key(|item| item.index);
                    return batch_run_result(attempted, results);
                }
            }
        }
    }
    let mut results = if items.len() >= proof_gen_finalize_parallel_min_batch() {
        let chunks = into_proof_gen_worker_chunks(items);
        thread::scope(|scope| {
            chunks
                .into_iter()
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .into_iter()
                            .map(|(index, req, state, t2_item)| {
                                let result = t2_item.result.and_then(|t2| {
                                    cpu.proof_gen_finalize_compressed_state_and_t2(req, state, t2)
                                });
                                BbsBatchItemResult { index, result }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("compressed proof-gen finalize worker panicked")
                })
                .flatten()
                .collect::<Vec<_>>()
        })
    } else {
        items
            .into_iter()
            .map(|(index, req, state, t2_item)| {
                let result = t2_item
                    .result
                    .and_then(|t2| cpu.proof_gen_finalize_compressed_state_and_t2(req, state, t2));
                BbsBatchItemResult { index, result }
            })
            .collect::<Vec<_>>()
    };
    results.sort_by_key(|item| item.index);
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }
    batch_run_result(attempted, results)
}

pub(super) fn finalize_prepared_signatures(
    cpu: CuBbsProtocolEngine,
    kernel: Option<&GpuKernelLibrary>,
    attempted: usize,
    items: Vec<(
        usize,
        SignRequest,
        BbsBatchItemResult<Vec<u8>>,
        BBSplusMessage,
    )>,
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let common_secret_key = items.first().and_then(|(_, first_req, _, _)| {
        items
            .iter()
            .all(|(_, req, _, _)| req.secret_key == first_req.secret_key)
            .then(|| parse_secret_key(&first_req.secret_key))
    });
    let common_secret_key = match common_secret_key {
        Some(Ok(secret_key)) => Some(secret_key),
        Some(Err(error)) => return batch_call_error(attempted, error, fail_fast),
        None => None,
    };
    let common_secret_key_ref = common_secret_key.as_ref();

    if let Some(kernel) = kernel.filter(|kernel| kernel.can_g1_msm() && gpu_sign_finalize_enabled())
    {
        if let Some(result) =
            finalize_prepared_signatures_with_gpu(kernel, attempted, &items, common_secret_key_ref)
        {
            return result;
        }
    }

    let mut results = if items.len() < sign_finalize_parallel_min_batch() {
        items
            .into_iter()
            .map(|(index, req, prepared_b, e)| {
                let result = prepared_b.result.and_then(|compressed_b| {
                    let prepared_b = BbsPreparedB { compressed_b };
                    match common_secret_key_ref {
                        Some(secret_key) => {
                            cpu.sign_prehashed_e_and_b_with_secret(secret_key, e, prepared_b)
                        }
                        None => cpu.sign_prehashed_message_scalars_e_and_b(req, e, prepared_b),
                    }
                });
                BbsBatchItemResult { index, result }
            })
            .collect::<Vec<_>>()
    } else {
        thread::scope(|scope| {
            items
                .into_iter()
                .map(|(index, req, prepared_b, e)| {
                    scope.spawn(move || {
                        let result = prepared_b.result.and_then(|compressed_b| {
                            let prepared_b = BbsPreparedB { compressed_b };
                            match common_secret_key_ref {
                                Some(secret_key) => cpu
                                    .sign_prehashed_e_and_b_with_secret(secret_key, e, prepared_b),
                                None => {
                                    cpu.sign_prehashed_message_scalars_e_and_b(req, e, prepared_b)
                                }
                            }
                        });
                        BbsBatchItemResult { index, result }
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("BBS finalize worker panicked"))
                .collect::<Vec<_>>()
        })
    };

    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }

    batch_run_result(attempted, results)
}

pub(super) fn finalize_repeated_signatures_with_gpu(
    kernel: &GpuKernelLibrary,
    req: &SignRequest,
    compressed_b: &[u8],
    e_scalars: &[BBSplusMessage],
    fail_fast: bool,
) -> BbsBatchRunResult<Vec<u8>> {
    let attempted = e_scalars.len();
    if compressed_b.len() != 48 {
        return batch_call_error(
            attempted,
            Error::Gpu("repeated Sign B has invalid length".to_owned()),
            fail_fast,
        );
    }
    let secret_key = match parse_secret_key(&req.secret_key) {
        Ok(secret_key) => secret_key,
        Err(error) => return batch_call_error(attempted, error, fail_fast),
    };

    let denominators = e_scalars
        .iter()
        .map(|e| secret_key.0 + e.value)
        .collect::<Vec<_>>();
    let mut prefixes = Vec::with_capacity(attempted);
    let mut product = Scalar::from(1u64);
    for denominator in &denominators {
        prefixes.push(product);
        product *= denominator;
    }
    let Some(mut product_inverse) = Option::<Scalar>::from(product.invert()) else {
        return batch_call_error(
            attempted,
            Error::Gpu("repeated Sign denominator is zero".to_owned()),
            fail_fast,
        );
    };
    let mut inverses = vec![Scalar::from(0u64); attempted];
    for index in (0..attempted).rev() {
        inverses[index] = product_inverse * prefixes[index];
        product_inverse *= denominators[index];
    }

    let a_points = if kernel.can_g1_msm_common_points() {
        let packed_scalars = inverses
            .iter()
            .flat_map(|inverse| inverse.to_bytes_be())
            .collect::<Vec<_>>();
        kernel.g1_msm_common_points_packed(
            &[compressed_b.to_vec()],
            &packed_scalars,
            attempted,
            fail_fast,
        )
    } else {
        let terms = inverses
            .iter()
            .map(|inverse| BbsMsmTerms {
                compressed_points: vec![compressed_b.to_vec()].into(),
                scalar_be: vec![inverse.to_bytes_be().to_vec()],
            })
            .collect::<Vec<_>>();
        kernel.g1_msm_batch(&terms, fail_fast)
    };

    let mut results = a_points
        .results
        .into_iter()
        .map(|item| {
            let result = item.result.and_then(|compressed_a| {
                let e = e_scalars.get(item.index).ok_or_else(|| {
                    Error::Gpu("repeated Sign e scalar result missing".to_owned())
                })?;
                if compressed_a.len() != 48 {
                    return Err(Error::Gpu(
                        "repeated signature A has invalid length".to_owned(),
                    ));
                }
                let mut signature = Vec::with_capacity(SIGNATURE_LEN);
                signature.extend_from_slice(&compressed_a);
                signature.extend_from_slice(&e.to_bytes_be());
                Ok(signature)
            });
            BbsBatchItemResult {
                index: item.index,
                result,
            }
        })
        .collect::<Vec<_>>();
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }
    batch_run_result(attempted, results)
}

pub(super) fn sign_batch_fused_fixed_base(
    kernel: &GpuKernelLibrary,
    attempted: usize,
    prepared: &[(usize, SignRequest, Vec<BBSplusMessage>)],
    terms: &[BbsMsmTerms],
    common_template: Option<&BbsMsmTerms>,
    e_scalars: &[BBSplusMessage],
    fail_fast: bool,
) -> Option<BbsBatchRunResult<Vec<u8>>> {
    let profile = std::env::var_os("CUBBS_BBS_SIGN_BATCH_PROFILE").is_some();
    let device_scalar_scale = std::env::var("CUBBS_BBS_GPU_SIGN_DEVICE_SCALAR_SCALE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(prepared.len() >= 512);
    let profile_started = Instant::now();
    if std::env::var("CUBBS_BBS_GPU_SIGN_FUSED_FIXED_BASE")
        .map(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(false)
        || (common_template.is_none() && prepared.len() != terms.len())
        || prepared.len() != e_scalars.len()
    {
        return None;
    }

    let common_secret_key = prepared.first().and_then(|(_, first_req, _)| {
        prepared
            .iter()
            .all(|(_, req, _)| req.secret_key == first_req.secret_key)
            .then(|| parse_secret_key(&first_req.secret_key).ok())
            .flatten()
    });
    let mut denominators = Vec::with_capacity(prepared.len());
    for ((_, req, _), e) in prepared.iter().zip(e_scalars) {
        let secret_key = match &common_secret_key {
            Some(secret_key) => secret_key.clone(),
            None => parse_secret_key(&req.secret_key).ok()?,
        };
        denominators.push(secret_key.0 + e.value);
    }

    let mut prefixes = Vec::with_capacity(denominators.len());
    let mut product = Scalar::from(1u64);
    for denominator in &denominators {
        prefixes.push(product);
        product *= denominator;
    }
    let mut product_inverse: Scalar = Option::from(product.invert())?;
    let mut inverses = vec![Scalar::from(0u64); denominators.len()];
    for index in (0..denominators.len()).rev() {
        inverses[index] = product_inverse * prefixes[index];
        product_inverse *= denominators[index];
    }
    let inverse_ms = elapsed_ms(profile_started);

    let packed_template = common_template.or_else(|| {
        all_msm_terms_have_common_points(terms)
            .then(|| terms.first())
            .flatten()
    });
    let scalar_pack_started = Instant::now();
    let (a_points, scalar_pack_ms, msm_ms) = if let Some(template) = packed_template {
        let common_domain = common_template
            .map(|template| {
                template
                    .scalar_be
                    .get(1)
                    .ok_or_else(|| Error::Gpu("fused sign template missing domain".to_owned()))
                    .and_then(|bytes| parse_message_scalar_bytes(bytes).map(|value| value.value))
            })
            .transpose()
            .ok()?;
        let packed_chunks = thread::scope(|scope| {
            worker_ranges(prepared.len())
                .into_iter()
                .map(|range| {
                    let inverses = &inverses;
                    scope.spawn(move || {
                        let scalar_row_terms =
                            template.compressed_points.len() + usize::from(device_scalar_scale);
                        let mut packed = Vec::with_capacity(range.len() * scalar_row_terms * 32);
                        for position in range {
                            let message_scalars = &prepared[position].2;
                            let inverse = inverses[position];
                            if template.compressed_points.len() != message_scalars.len() + 2
                                || (common_domain.is_none()
                                    && terms[position].scalar_be.get(1).is_none())
                            {
                                return Err(Error::Gpu(
                                    "fused sign MSM scalar shape mismatch".to_owned(),
                                ));
                            }
                            let domain = match common_domain {
                                Some(domain) => domain,
                                None => {
                                    parse_message_scalar_bytes(&terms[position].scalar_be[1])?.value
                                }
                            };
                            if device_scalar_scale {
                                packed.extend_from_slice(&inverse.to_bytes_be());
                                packed.extend_from_slice(&Scalar::from(1u64).to_bytes_be());
                                packed.extend_from_slice(&domain.to_bytes_be());
                                for message in message_scalars {
                                    packed.extend_from_slice(&message.value.to_bytes_be());
                                }
                            } else {
                                packed.extend_from_slice(&inverse.to_bytes_be());
                                packed.extend_from_slice(&(domain * inverse).to_bytes_be());
                                for message in message_scalars {
                                    packed.extend_from_slice(
                                        &(message.value * inverse).to_bytes_be(),
                                    );
                                }
                            }
                        }
                        Ok(packed)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("fused sign scalar worker panicked"))
                .collect::<Result<Vec<_>>>()
        })
        .ok()?;
        let packed_scalars = packed_chunks.into_iter().flatten().collect::<Vec<_>>();
        let scalar_pack_ms = elapsed_ms(scalar_pack_started);
        let msm_started = Instant::now();
        let points = kernel.g1_msm_common_points_packed(
            &template.compressed_points,
            &packed_scalars,
            prepared.len(),
            fail_fast,
        );
        (points, scalar_pack_ms, elapsed_ms(msm_started))
    } else {
        let scaled_terms = thread::scope(|scope| {
            worker_ranges(terms.len())
                .into_iter()
                .map(|range| {
                    let inverses = &inverses;
                    scope.spawn(move || {
                        range
                            .map(|position| {
                                let item = &terms[position];
                                let inverse = inverses[position];
                                let scalar_be = item
                                    .scalar_be
                                    .iter()
                                    .map(|bytes| {
                                        parse_message_scalar_bytes(bytes).map(|scalar| {
                                            (scalar.value * inverse).to_bytes_be().to_vec()
                                        })
                                    })
                                    .collect::<Result<Vec<_>>>()?;
                                Ok(BbsMsmTerms {
                                    compressed_points: item.compressed_points.clone(),
                                    scalar_be,
                                })
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("fused sign scalar worker panicked"))
                .collect::<Result<Vec<_>>>()
        })
        .ok()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let scalar_pack_ms = elapsed_ms(scalar_pack_started);
        let msm_started = Instant::now();
        let points = g1_msm_batch_grouped_by_points(kernel, &scaled_terms, fail_fast);
        (points, scalar_pack_ms, elapsed_ms(msm_started))
    };
    if a_points.failed != 0 || a_points.results.len() != prepared.len() {
        return None;
    }

    let assemble_started = Instant::now();
    let mut results = prepared
        .iter()
        .zip(e_scalars)
        .zip(a_points.results)
        .map(|(((index, _, _), e), point)| {
            let result = point.result.and_then(|compressed_a| {
                if compressed_a.len() != 48 {
                    return Err(Error::Gpu(
                        "fused signature A has invalid length".to_owned(),
                    ));
                }
                let mut signature = Vec::with_capacity(SIGNATURE_LEN);
                signature.extend_from_slice(&compressed_a);
                signature.extend_from_slice(&e.to_bytes_be());
                Ok(signature)
            });
            BbsBatchItemResult {
                index: *index,
                result,
            }
        })
        .collect::<Vec<_>>();
    if fail_fast {
        if let Some(first_error) = results.iter().position(|item| item.result.is_err()) {
            results.truncate(first_error + 1);
        }
    }
    if profile {
        eprintln!(
            "cubbs_sign_fused_profile items={} inverse_ms={inverse_ms:.6} \
             scalar_pack_ms={scalar_pack_ms:.6} msm_ms={msm_ms:.6} \
             assemble_ms={:.6} total_ms={:.6}",
            prepared.len(),
            elapsed_ms(assemble_started),
            elapsed_ms(profile_started),
        );
    }
    Some(batch_run_result(attempted, results))
}

pub(super) fn finalize_prepared_signatures_with_gpu(
    kernel: &GpuKernelLibrary,
    attempted: usize,
    items: &[(
        usize,
        SignRequest,
        BbsBatchItemResult<Vec<u8>>,
        BBSplusMessage,
    )],
    common_secret_key: Option<&BBSplusSecretKey>,
) -> Option<BbsBatchRunResult<Vec<u8>>> {
    let mut prepared = Vec::with_capacity(items.len());
    let mut denominators = Vec::with_capacity(items.len());

    for (index, req, prepared_b, e) in items {
        let compressed_b = prepared_b.result.as_ref().ok()?.clone();
        let secret_key = match common_secret_key {
            Some(secret_key) => secret_key.clone(),
            None => parse_secret_key(&req.secret_key).ok()?,
        };
        denominators.push(secret_key.0 + e.value);
        prepared.push((*index, e.to_bytes_be().to_vec(), compressed_b));
    }

    // Montgomery batch inversion: one field inversion for the whole batch,
    // plus prefix/reverse multiplications. Every denominator must be non-zero,
    // matching the per-item inversion path's validity condition.
    let mut prefixes = Vec::with_capacity(denominators.len());
    let mut product = Scalar::from(1u64);
    for denominator in &denominators {
        prefixes.push(product);
        product *= denominator;
    }
    let mut product_inverse: Scalar = Option::from(product.invert())?;
    let mut inverses = vec![Scalar::from(0u64); denominators.len()];
    for index in (0..denominators.len()).rev() {
        inverses[index] = product_inverse * prefixes[index];
        product_inverse *= denominators[index];
    }

    let mut msm_terms = Vec::with_capacity(items.len());
    let mut output_items = Vec::with_capacity(items.len());
    for ((index, e_bytes, compressed_b), inv) in prepared.into_iter().zip(inverses) {
        msm_terms.push(BbsMsmTerms {
            compressed_points: vec![compressed_b].into(),
            scalar_be: vec![inv.to_bytes_be().to_vec()],
        });
        output_items.push((index, e_bytes));
    }

    let a_points = kernel.g1_msm_batch(&msm_terms, true);
    if a_points.failed != 0 || a_points.results.len() != output_items.len() {
        return None;
    }

    let mut results = Vec::with_capacity(output_items.len());
    for ((index, e_bytes), a_point) in output_items.into_iter().zip(a_points.results.into_iter()) {
        let result = a_point.result.and_then(|compressed_a| {
            if compressed_a.len() != 48 {
                return Err(Error::Gpu(format!(
                    "GPU signature A point has invalid length: {}",
                    compressed_a.len()
                )));
            }
            let mut signature = Vec::with_capacity(SIGNATURE_LEN);
            signature.extend_from_slice(&compressed_a);
            signature.extend_from_slice(&e_bytes);
            Ok(signature)
        });
        results.push(BbsBatchItemResult { index, result });
    }

    Some(batch_run_result(attempted, results))
}

pub(super) fn byte_slice(bytes: &[u8]) -> gpu_abi::CuBbsBbsGpuByteSlice {
    gpu_abi::CuBbsBbsGpuByteSlice {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

pub(super) fn byte_buffer(bytes: &mut [u8]) -> gpu_abi::CuBbsBbsGpuByteBuffer {
    gpu_abi::CuBbsBbsGpuByteBuffer {
        data: bytes.as_mut_ptr(),
        capacity: bytes.len(),
        len: 0,
    }
}

pub(super) fn index_slice(indexes: &[usize]) -> gpu_abi::CuBbsBbsGpuIndexSlice {
    gpu_abi::CuBbsBbsGpuIndexSlice {
        data: indexes.as_ptr(),
        len: indexes.len(),
    }
}

pub(super) fn empty_gpu_results(count: usize) -> Vec<gpu_abi::CuBbsBbsGpuResult> {
    (0..count)
        .map(|_| gpu_abi::CuBbsBbsGpuResult {
            code: gpu_abi::CUBBS_BBS_GPU_SUCCESS,
            value: 0,
            bytes_written: 0,
        })
        .collect()
}

pub(super) fn proof_capacity(message_count: usize, disclosed_count: usize) -> usize {
    512 + (message_count * 48) + (disclosed_count * 48)
}

pub(super) fn bbs_message_scalar_dst() -> Vec<u8> {
    [
        Bls12381Sha256::API_ID,
        <Bls12381Sha256 as cubbs_bbs_protocol::bbsplus::ciphersuites::BbsCiphersuite>::MAP_MSG_SCALAR,
    ]
    .concat()
}

pub(super) fn bbs_signature_e_dst() -> Vec<u8> {
    [
        Bls12381Sha256::API_ID,
        <Bls12381Sha256 as cubbs_bbs_protocol::bbsplus::ciphersuites::BbsCiphersuite>::H2S,
    ]
    .concat()
}

pub(super) fn force_gpu_execution() -> bool {
    std::env::var("CUBBS_BBS_FORCE_GPU")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(false)
}

pub(super) fn gpu_runtime_plan_error(
    kernel: &GpuKernelLibrary,
    plan: &planner::ExecutionPlan,
) -> Option<Error> {
    for batch in &plan.batches {
        for job in &batch.jobs {
            let missing = match job {
                planner::ExecutionJob::HashToScalar { .. } => {
                    !kernel.can_precompute_message_scalars()
                }
                planner::ExecutionJob::Msm { .. } => !kernel.can_g1_msm(),
                planner::ExecutionJob::Pairing { .. } => !kernel.can_pairing_eq(),
            };
            if missing && force_gpu_execution() {
                return Some(Error::HardwareGpuUnavailable(format!(
                    "FORCE_GPU cannot lower {:?} for {:?}",
                    job, batch.key.operation
                )));
            }
        }
    }
    None
}

pub(super) fn gpu_precompute_signature_e_enabled() -> bool {
    if force_gpu_execution() {
        return true;
    }
    std::env::var("CUBBS_BBS_GPU_PRECOMPUTE_E")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub(super) fn gpu_sign_finalize_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_SIGN_FINALIZE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_challenge_scalar_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_CHALLENGE_SCALAR")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub(super) fn pairing_aggregate_enabled(item_count: usize) -> bool {
    std::env::var("CUBBS_BBS_PAIRING_AGGREGATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(item_count >= 128)
}

pub(super) fn native_pairing_enabled() -> bool {
    if force_gpu_execution() {
        return true;
    }
    std::env::var("CUBBS_BBS_NATIVE_PAIRING")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_proof_verify_split_msm_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_PROOF_VERIFY_SPLIT_MSM")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

pub(super) fn gpu_proof_verify_msm_pair_enabled(item_count: usize, max_t2_terms: usize) -> bool {
    std::env::var("CUBBS_BBS_GPU_PROOF_VERIFY_MSM_PAIR")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        // A single short proof does not amortize the fused-pair launch. Longer
        // T2 rows do, while every multi-item batch benefits from the fused ABI.
        .unwrap_or(item_count >= 2 || max_t2_terms >= 32)
}

pub(super) fn gpu_verify_dual_same_points_enabled(item_count: usize) -> bool {
    gpu_verify_dual_same_points_enabled_from_env(
        std::env::var("CUBBS_BBS_GPU_VERIFY_DUAL_SAME_POINTS")
            .ok()
            .as_deref(),
        item_count,
    )
}

pub(super) fn gpu_verify_dual_same_points_enabled_from_env(
    value: Option<&str>,
    item_count: usize,
) -> bool {
    value
        .map(|value| !matches!(value, "0" | "false" | "FALSE" | "no" | "NO"))
        // A pre-B Verify aggregate has two MSM rows over exactly the same
        // signature-A points. The dedicated dual kernel keeps one bounded
        // per-thread workspace and avoids sending a changing two-row point
        // set through the persistent common-points comb cache. Keep the
        // singleton path on the smaller generic route, where the dual launch
        // overhead is not amortized.
        .unwrap_or(item_count >= 2)
}

pub(super) fn proof_verify_pairing_overlap_enabled(item_count: usize) -> bool {
    let _ = item_count;
    std::env::var("CUBBS_BBS_PROOF_VERIFY_PAIRING_OVERLAP")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(false)
}

pub(super) fn gpu_split_variable_msm_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_SPLIT_VARIABLE_MSM")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_common_plus_variable_msm_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_COMMON_PLUS_VARIABLE_MSM")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_split_variable_msm_min_terms() -> usize {
    std::env::var("CUBBS_BBS_GPU_SPLIT_VARIABLE_MSM_MIN_TERMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
}

pub(super) fn gpu_split_variable_parallel_min_batch() -> usize {
    std::env::var("CUBBS_BBS_GPU_SPLIT_VARIABLE_PARALLEL_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

pub(super) fn gpu_mixed_shape_parallel_min_groups() -> usize {
    std::env::var("CUBBS_BBS_GPU_MIXED_SHAPE_PARALLEL_MIN_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
}

pub(super) fn proof_mixed_shape_parallel_streams(primary: &str, tuned_default: usize) -> usize {
    std::env::var(primary)
        .or_else(|_| std::env::var("CUBBS_BBS_PROOF_MIXED_SHAPE_PARALLEL_STREAMS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(tuned_default)
}

pub(super) fn proof_gen_mixed_shape_parallel_streams(tuned_default: usize) -> usize {
    proof_mixed_shape_parallel_streams(
        "CUBBS_BBS_PROOF_GEN_MIXED_SHAPE_PARALLEL_STREAMS",
        tuned_default,
    )
}

pub(super) fn proof_verify_mixed_shape_parallel_streams(tuned_default: usize) -> usize {
    proof_mixed_shape_parallel_streams(
        "CUBBS_BBS_PROOF_VERIFY_MIXED_SHAPE_PARALLEL_STREAMS",
        tuned_default,
    )
}

pub(super) fn proof_bbs_mixed_shape_batching_enabled() -> bool {
    std::env::var("CUBBS_BBS_PROOF_MIXED_SHAPE_BATCHING")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn sign_shape_batching_enabled() -> bool {
    std::env::var("CUBBS_BBS_SIGN_SHAPE_BATCHING")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn proof_verify_shape_grouping_enabled() -> bool {
    std::env::var("CUBBS_BBS_PROOF_VERIFY_SHAPE_GROUPING")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn proof_verify_prepared_cache_enabled() -> bool {
    std::env::var("CUBBS_BBS_PROOF_VERIFY_PREPARED_CACHE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn gpu_proof_domain_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_PROOF_DOMAIN")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn verify_mixed_aggregate_enabled() -> bool {
    std::env::var("CUBBS_BBS_VERIFY_MIXED_AGGREGATE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn verify_prepared_b_cache_enabled() -> bool {
    std::env::var("CUBBS_BBS_VERIFY_PREPARED_B_CACHE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

// Once a Verify batch satisfies the structural eligibility checks in
// `verify_homogeneous_pre_b_aggregate`, the pre-B path is profitable even for
// a singleton on the CUDA backend. Keep it enabled by default for every
// non-empty eligible batch; callers can still raise the threshold with
// CUBBS_BBS_VERIFY_AGGREGATE_MIN_ITEMS or disable it with
// CUBBS_BBS_PAIRING_AGGREGATE=0.
pub(super) const DEFAULT_VERIFY_PRE_B_AGGREGATE_MIN_ITEMS: usize = 1;

pub(super) fn verify_aggregate_min_items_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_VERIFY_PRE_B_AGGREGATE_MIN_ITEMS)
}

pub(super) fn verify_aggregate_min_items() -> usize {
    let configured = std::env::var("CUBBS_BBS_VERIFY_AGGREGATE_MIN_ITEMS").ok();
    verify_aggregate_min_items_from_env(configured.as_deref())
}

pub(super) fn verify_pre_b_aggregate_policy(
    item_count: usize,
    min_items: usize,
    pairing_aggregate_override: Option<&str>,
) -> bool {
    let aggregate_enabled = pairing_aggregate_override
        .map(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true);
    aggregate_enabled && item_count >= min_items
}

pub(super) fn verify_pre_b_aggregate_enabled(item_count: usize) -> bool {
    let pairing_override = std::env::var("CUBBS_BBS_PAIRING_AGGREGATE").ok();
    // Pre-B aggregation removes the per-item B MSMs before the final pairing
    // check. On the CUDA path it remains profitable for every non-empty
    // structurally eligible batch, independently of the generic pairing-only
    // aggregation threshold (128 items by default).
    verify_pre_b_aggregate_policy(
        item_count,
        verify_aggregate_min_items(),
        pairing_override.as_deref(),
    )
}

pub(super) fn verify_aggregate_lane_count(item_count: usize) -> usize {
    let configured = std::env::var("CUBBS_BBS_GPU_VERIFY_AGGREGATE_LANES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    configured
        .min((item_count / 128).max(1))
        .min(item_count.max(1))
}

pub(super) fn message_scalar_cache_enabled() -> bool {
    std::env::var("CUBBS_BBS_MESSAGE_SCALAR_CACHE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn sign_prepared_cache_enabled() -> bool {
    std::env::var("CUBBS_BBS_SIGN_PREPARED_CACHE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | ""))
        .unwrap_or(true)
}

pub(super) fn sign_prepared_cache_max_batch() -> usize {
    std::env::var("CUBBS_BBS_SIGN_PREPARED_CACHE_MAX_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(24)
}

pub(super) fn proof_bbs_mixed_shape_min_batch() -> usize {
    std::env::var("CUBBS_BBS_PROOF_MIXED_SHAPE_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
}

pub(super) fn proof_bbs_mixed_shape_min_groups() -> usize {
    std::env::var("CUBBS_BBS_PROOF_MIXED_SHAPE_MIN_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
}

pub(super) fn gpu_point_group_parallel_min_groups() -> usize {
    std::env::var("CUBBS_BBS_GPU_POINT_GROUP_PARALLEL_MIN_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

pub(super) fn gpu_native_mixed_msm_grouping_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_NATIVE_MIXED_MSM_GROUPING")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_common_points_msm_abi_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

/// Enable the GPU proof-gen t2Prepare (6 fixed G1 scalar multiplications on
/// the device). Shape and capability checks still gate the specialized path;
/// the GPU path falls back to CPU preparation if disabled or failing.
pub(super) fn gpu_proof_gen_t2_prepare_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_PROOF_GEN_T2_PREPARE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_proof_gen_expanded_t2_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_PROOF_GEN_EXPANDED_T2")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn gpu_proof_gen_t2_prepare_for_shape(item_count: usize, message_count: usize) -> bool {
    if force_gpu_execution() {
        return true;
    }
    if let Ok(value) = std::env::var("CUBBS_BBS_GPU_PROOF_GEN_T2_PREPARE_MIN_ITEMS") {
        return value
            .parse::<usize>()
            .map(|minimum| item_count >= minimum)
            .unwrap_or(false);
    }
    item_count >= 2 || message_count >= 32
}

pub(super) fn gpu_proof_gen_b_common_points_msm_abi_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_PROOF_GEN_B_COMMON_POINTS_MSM_ABI")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

pub(super) fn gpu_native_g1_add_enabled() -> bool {
    std::env::var("CUBBS_BBS_GPU_NATIVE_G1_ADD")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn proof_verify_shared_domain_enabled() -> bool {
    std::env::var("CUBBS_BBS_PROOF_VERIFY_SHARED_DOMAIN")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

pub(super) fn sign_finalize_parallel_min_batch() -> usize {
    std::env::var("CUBBS_BBS_SIGN_FINALIZE_PARALLEL_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn proof_gen_finalize_parallel_min_batch() -> usize {
    std::env::var("CUBBS_BBS_PROOF_GEN_FINALIZE_PARALLEL_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn proof_gpu_prepare_parallel_min_batch() -> usize {
    std::env::var("CUBBS_BBS_PROOF_GPU_PREPARE_PARALLEL_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn proof_verify_parallel_pairing_min_batch() -> usize {
    std::env::var("CUBBS_BBS_PROOF_VERIFY_PARALLEL_PAIRING_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn proof_verify_pair_msm_parallel_min_batch() -> usize {
    std::env::var("CUBBS_BBS_PROOF_VERIFY_PAIR_MSM_PARALLEL_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn verify_parallel_pairing_min_batch() -> usize {
    std::env::var("CUBBS_BBS_VERIFY_PARALLEL_PAIRING_MIN_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn proof_verify_gpu_msm_enabled(req: &ProofVerifyRequest) -> bool {
    if force_gpu_execution() {
        return true;
    }
    let min_messages = std::env::var("CUBBS_BBS_GPU_PROOF_VERIFY_MSM_MIN_MESSAGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    // AUTO retains the measured two-hidden-message crossover. FORCE_GPU bypasses
    // this policy gate; CFRG valid and negative vectors cover zero/one-hidden
    // T2 shapes through the universal GPU MSM path.
    let hidden_messages = req
        .total_message_count
        .saturating_sub(req.disclosed_messages.len());
    hidden_messages >= 2 && req.total_message_count >= min_messages
}

pub(super) fn proof_gen_gpu_t2_msm_enabled(message_count: usize) -> bool {
    if force_gpu_execution() {
        return true;
    }
    let min_messages = std::env::var("CUBBS_BBS_GPU_PROOF_GEN_T2_MSM_MIN_MESSAGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    message_count >= min_messages
}

pub(super) fn proof_gen_overlap_domain_enabled() -> bool {
    std::env::var("CUBBS_BBS_PROOF_GEN_OVERLAP_DOMAIN")
        .map(|value| value != "0" && value != "false" && value != "FALSE")
        .unwrap_or(false)
}

pub(super) fn cpu_message_scalars(messages: &[Vec<u8>]) -> Result<Vec<BBSplusMessage>> {
    BBSplusMessage::messages_to_scalar::<Bls12381Sha256>(messages, Bls12381Sha256::API_ID)
        .map_err(|e| Error::Zkryptium(e.to_string()))
}

pub(super) fn cpu_signature_e_scalar(e_input: &[u8]) -> Result<BBSplusMessage> {
    hash_to_scalar::<Bls12381Sha256>(e_input, &bbs_signature_e_dst())
        .map(BBSplusMessage::new)
        .map_err(|e| Error::Zkryptium(e.to_string()))
}

pub(super) fn cpu_signature_e_scalars_batch(e_inputs: &[Vec<u8>]) -> Result<Vec<BBSplusMessage>> {
    if e_inputs.len() < 4 {
        return e_inputs
            .iter()
            .map(|input| cpu_signature_e_scalar(input))
            .collect();
    }
    let hardware = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8);
    let input_bytes = e_inputs.first().map(Vec::len).unwrap_or(0);
    let tuned_default = if e_inputs.len() < 256 {
        4
    } else if e_inputs.len() < 1024 && input_bytes <= 576 {
        4
    } else {
        4
    };
    let worker_count = std::env::var("CUBBS_BBS_SIGN_E_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(hardware.min(tuned_default))
        .max(1)
        .min(e_inputs.len());
    let chunks = thread::scope(|scope| {
        worker_ranges_with_count(e_inputs.len(), worker_count)
            .into_iter()
            .map(|range| {
                scope.spawn(move || {
                    e_inputs[range]
                        .iter()
                        .map(|input| cpu_signature_e_scalar(input))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("signature e worker panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    Ok(chunks.into_iter().flatten().collect())
}

pub(super) fn bbs_msm_terms_from_template(
    template: &BbsMsmTerms,
    message_scalars: &[BBSplusMessage],
) -> Result<BbsMsmTerms> {
    let scalar_be = bbs_msm_scalar_batch_from_template(template, message_scalars)?;
    Ok(BbsMsmTerms {
        compressed_points: template.compressed_points.clone(),
        scalar_be,
    })
}

pub(super) fn b_msm_domain_scalar(terms: &BbsMsmTerms) -> Result<Scalar> {
    let domain = terms
        .scalar_be
        .get(1)
        .ok_or_else(|| Error::Gpu("BBS MSM terms missing precomputed domain scalar".to_owned()))?;
    Ok(parse_message_scalar_bytes(domain)?.value)
}

pub(super) fn precompute_b_msm_domains(terms: Vec<BbsMsmTerms>) -> Result<Vec<Scalar>> {
    if terms.len() >= proof_gpu_prepare_parallel_min_batch() {
        let chunks = into_worker_chunks(terms);
        return thread::scope(|scope| {
            chunks
                .into_iter()
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(b_msm_domain_scalar)
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("B MSM domain worker panicked"))
                .collect::<Result<Vec<_>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        });
    }

    terms.iter().map(b_msm_domain_scalar).collect()
}

pub(super) fn proof_verify_terms_from_zkryptium(
    terms: BBSplusProofVerifyMsmTerms,
) -> (BbsMsmTerms, BbsMsmTerms, Scalar) {
    let domain = terms.domain;
    (
        BbsMsmTerms {
            compressed_points: terms.t1_compressed_points.into(),
            scalar_be: terms.t1_scalar_be,
        },
        BbsMsmTerms {
            compressed_points: terms.t2_compressed_points.into(),
            scalar_be: terms.t2_scalar_be,
        },
        domain,
    )
}

pub(super) fn g2_generator_compressed() -> Vec<u8> {
    static COMPRESSED: OnceLock<Vec<u8>> = OnceLock::new();
    COMPRESSED
        .get_or_init(|| G2Projective::GENERATOR.to_affine().to_compressed().to_vec())
        .clone()
}

pub(super) fn pairing_aggregate_scalars(count: usize) -> Vec<Scalar> {
    // A non-zero 128-bit coefficient gives the aggregate check a 2^-128
    // soundness error, matching the security level of BLS12-381 while
    // avoiding a 512-bit field reduction for every signature.
    let mut random_bytes = vec![0u8; count * 16];
    rand::thread_rng().fill_bytes(&mut random_bytes);
    random_bytes
        .chunks_exact(16)
        .map(|bytes| {
            let mut low_bytes = [0u8; 8];
            let mut high_bytes = [0u8; 8];
            low_bytes.copy_from_slice(&bytes[..8]);
            high_bytes.copy_from_slice(&bytes[8..]);
            let mut low = u64::from_le_bytes(low_bytes);
            let high = u64::from_le_bytes(high_bytes);
            if low == 0 && high == 0 {
                low = 1;
            }
            Scalar::from_raw_unchecked([low, high, 0, 0])
        })
        .collect()
}

pub(super) fn pairing_aggregate_coefficients_for_items(
    items: &[BbsPairingEqInput],
) -> Result<Vec<Vec<u8>>> {
    let mut transcript = Sha256::new();
    // Frozen domain-separation bytes are not product branding.
    transcript.update(b"OMNIID_BBS_PAIRING_AGGREGATE_TRANSCRIPT_V1");
    transcript.update((items.len() as u64).to_be_bytes());
    for item in items {
        for value in [&item.lhs_g1, &item.lhs_g2, &item.rhs_g1, &item.rhs_g2] {
            transcript.update((value.len() as u64).to_be_bytes());
            transcript.update(value);
        }
    }
    let seed = transcript.finalize();
    Ok((0..items.len())
        .map(|index| {
            let mut wide = [0u8; 64];
            for half in 0..2u8 {
                let mut hasher = Sha256::new();
                hasher.update(b"CUBBS_BBS_PAIRING_AGGREGATE_COEFF_V1");
                hasher.update(seed);
                hasher.update((index as u64).to_be_bytes());
                hasher.update([half]);
                wide[(half as usize) * 32..(half as usize + 1) * 32]
                    .copy_from_slice(&hasher.finalize());
            }
            let scalar = Scalar::from_bytes_wide(&wide);
            let scalar = if scalar == Scalar::from(0u64) {
                Scalar::from(1u64)
            } else {
                scalar
            };
            scalar.to_bytes_be().to_vec()
        })
        .collect())
}

pub(super) fn proof_gen_t2_terms_from_zkryptium(terms: BBSplusProofGenT2MsmTerms) -> BbsMsmTerms {
    BbsMsmTerms {
        compressed_points: terms.t2_compressed_points.into(),
        scalar_be: terms.t2_scalar_be,
    }
}

pub(super) fn proof_verify_challenge_and_pairing_eq_input_from_prepared(
    req: &ProofVerifyRequest,
    disclosed_message_scalars: &[BBSplusMessage],
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    domain: Scalar,
    meta: &ProofVerifyPreparedMeta,
) -> Result<BbsPairingEqInput> {
    let challenge_input = proof_verify_challenge_input_from_prepared(
        req,
        disclosed_message_scalars,
        t1_compressed,
        t2_compressed,
        domain,
        meta,
    )?;
    let challenge = hash_to_scalar::<Bls12381Sha256>(&challenge_input, &bbs_signature_e_dst())
        .map_err(|e| Error::Zkryptium(e.to_string()))?;
    let expected = parse_message_scalar_bytes(&meta.challenge_be)?.value;
    if challenge != expected {
        return Err(Error::InvalidProof);
    }
    Ok(meta.pairing_input.clone())
}

pub(super) fn proof_verify_challenge_input_from_prepared(
    req: &ProofVerifyRequest,
    disclosed_message_scalars: &[BBSplusMessage],
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    domain: Scalar,
    meta: &ProofVerifyPreparedMeta,
) -> Result<Vec<u8>> {
    if disclosed_message_scalars.len() != req.disclosed_indexes.len() {
        return Err(Error::InvalidDisclosedIndexes);
    }
    if t1_compressed.len() != 48 || t2_compressed.len() != 48 {
        return Err(Error::InvalidProof);
    }
    if meta.pairing_input.lhs_g1.len() != 48
        || meta.pairing_input.rhs_g1.len() != 48
        || meta.challenge_be.len() != 32
    {
        return Err(Error::InvalidProof);
    }

    let ph = req.presentation_header.as_slice();
    let r = req.disclosed_indexes.len();
    let mut input = Vec::with_capacity(8 + (r * 40) + (5 * 48) + 32 + 8 + ph.len());
    input.extend_from_slice(&(r as u64).to_be_bytes());
    for (index, scalar) in req
        .disclosed_indexes
        .iter()
        .zip(disclosed_message_scalars.iter())
    {
        input.extend_from_slice(&(*index as u64).to_be_bytes());
        input.extend_from_slice(&scalar.to_bytes_be());
    }
    input.extend_from_slice(&meta.pairing_input.lhs_g1);
    input.extend_from_slice(&meta.pairing_input.rhs_g1);
    input.extend_from_slice(&req.proof[96..144]);
    input.extend_from_slice(t1_compressed);
    input.extend_from_slice(t2_compressed);
    input.extend_from_slice(&domain.to_bytes_be());
    input.extend_from_slice(&(ph.len() as u64).to_be_bytes());
    input.extend_from_slice(ph);
    Ok(input)
}

pub(super) fn bbs_msm_scalar_batch_from_template(
    template: &BbsMsmTerms,
    message_scalars: &[BBSplusMessage],
) -> Result<Vec<Vec<u8>>> {
    let expected_len = message_scalars.len() + 2;
    if template.compressed_points.len() != expected_len || template.scalar_be.len() != expected_len
    {
        return Err(Error::Gpu(format!(
            "BBS MSM template shape mismatch: expected {expected_len} terms, got {}/{}",
            template.compressed_points.len(),
            template.scalar_be.len()
        )));
    }

    let mut scalar_be = Vec::with_capacity(expected_len);
    scalar_be.push(template.scalar_be[0].clone());
    scalar_be.push(template.scalar_be[1].clone());
    scalar_be.extend(
        message_scalars
            .iter()
            .map(|scalar| scalar.to_bytes_be().to_vec()),
    );

    Ok(scalar_be)
}

pub(super) fn signature_e_input_from_msm_terms(
    req: &SignRequest,
    message_scalars: &[BBSplusMessage],
    terms: &BbsMsmTerms,
) -> Result<Vec<u8>> {
    ensure_header(&req.header)?;
    if req.secret_key.len() != SECRET_KEY_LEN {
        return Err(Error::InvalidSecretKey);
    }
    let expected_len = message_scalars.len() + 2;
    if terms.scalar_be.len() != expected_len {
        return Err(Error::Gpu(format!(
            "BBS MSM scalar template shape mismatch: expected {expected_len} terms, got {}",
            terms.scalar_be.len()
        )));
    }

    signature_e_input_from_domain(req, message_scalars, &terms.scalar_be[1])
}

pub(super) fn signature_e_input_from_domain(
    req: &SignRequest,
    message_scalars: &[BBSplusMessage],
    domain_be: &[u8],
) -> Result<Vec<u8>> {
    ensure_header(&req.header)?;
    if req.secret_key.len() != SECRET_KEY_LEN {
        return Err(Error::InvalidSecretKey);
    }
    if domain_be.len() != SECRET_KEY_LEN {
        return Err(Error::Gpu(format!(
            "BBS MSM domain scalar has invalid length: {}",
            domain_be.len()
        )));
    }

    let mut input = Vec::with_capacity((message_scalars.len() + 2) * SECRET_KEY_LEN);
    input.extend_from_slice(&req.secret_key);
    for scalar in message_scalars {
        input.extend_from_slice(&scalar.to_bytes_be());
    }
    input.extend_from_slice(domain_be);
    Ok(input)
}

pub(super) fn gpu_precompute_message_scalars(
    kernel: &GpuKernelLibrary,
    messages: &[Vec<u8>],
) -> Result<Vec<BBSplusMessage>> {
    gpu_precompute_flat_message_scalars(kernel, messages.to_vec(), true)
}

pub(super) fn gpu_precompute_flat_message_scalars(
    kernel: &GpuKernelLibrary,
    messages: Vec<Vec<u8>>,
    fail_fast: bool,
) -> Result<Vec<BBSplusMessage>> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    let scalar_result =
        kernel.messages_to_scalar_batch(messages, bbs_message_scalar_dst(), fail_fast);
    if scalar_result.failed > 0 {
        return Err(Error::Gpu(
            "GPU message scalar precompute failed".to_owned(),
        ));
    }

    scalar_result
        .results
        .into_iter()
        .map(|item| item.result.and_then(parse_gpu_message_scalar))
        .collect()
}

pub(super) fn gpu_precompute_flat_message_scalars_borrowed(
    kernel: &GpuKernelLibrary,
    messages: &[&[u8]],
) -> Result<Arc<[BBSplusMessage]>> {
    if messages.is_empty() {
        return Ok(Arc::from(Vec::<BBSplusMessage>::new()));
    }
    let cache_key = message_scalar_cache_enabled().then(|| {
        let mut key = Sha256::new();
        key.update(b"CUBBS_BBS_MESSAGE_SCALAR_BATCH_CACHE_V1");
        key.update((messages.len() as u64).to_be_bytes());
        for message in messages {
            key.update((message.len() as u64).to_be_bytes());
            key.update(message);
        }
        <[u8; 32]>::from(key.finalize())
    });
    if let Some(key) = cache_key.as_ref() {
        if let Some(cached) = message_scalar_batch_cache()
            .read()
            .unwrap()
            .get(key)
            .cloned()
        {
            return Ok(cached);
        }
    }
    let packed = kernel.messages_to_scalar_packed_borrowed(messages, &bbs_message_scalar_dst())?;
    let scalars = packed
        .chunks_exact(32)
        .map(parse_message_scalar_bytes)
        .collect::<Result<Vec<_>>>()?;
    let scalars = Arc::<[BBSplusMessage]>::from(scalars);
    if let Some(key) = cache_key {
        let mut cache = message_scalar_batch_cache().write().unwrap();
        const MAX_CACHED_SCALAR_BATCHES: usize = 16;
        if cache.len() >= MAX_CACHED_SCALAR_BATCHES {
            cache.clear();
        }
        cache.insert(key, Arc::clone(&scalars));
    }
    Ok(scalars)
}

pub(super) fn message_scalar_batch_cache(
) -> &'static RwLock<HashMap<[u8; 32], Arc<[BBSplusMessage]>>> {
    static CACHE: OnceLock<RwLock<HashMap<[u8; 32], Arc<[BBSplusMessage]>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(super) fn gpu_precompute_signature_e_scalars(
    kernel: &GpuKernelLibrary,
    e_inputs: Vec<Vec<u8>>,
    fail_fast: bool,
) -> Result<Vec<BBSplusMessage>> {
    if e_inputs.is_empty() {
        return Ok(Vec::new());
    }

    let scalar_result = kernel.messages_to_scalar_batch(e_inputs, bbs_signature_e_dst(), fail_fast);
    if scalar_result.failed > 0 {
        return Err(Error::Gpu(
            "GPU signature e scalar precompute failed".to_owned(),
        ));
    }

    scalar_result
        .results
        .into_iter()
        .map(|item| item.result.and_then(parse_gpu_message_scalar))
        .collect()
}

pub(super) fn gpu_precompute_signature_e_scalars_repeated(
    kernel: &GpuKernelLibrary,
    e_input: &[u8],
    count: usize,
) -> Result<Vec<BBSplusMessage>> {
    let inputs = vec![e_input; count];
    kernel
        .messages_to_scalar_packed_borrowed(&inputs, &bbs_signature_e_dst())?
        .chunks_exact(32)
        .map(parse_message_scalar_bytes)
        .collect()
}

pub(super) fn gpu_precompute_challenge_scalars(
    kernel: &GpuKernelLibrary,
    challenge_inputs: Vec<Vec<u8>>,
    fail_fast: bool,
) -> Result<Vec<BBSplusMessage>> {
    if challenge_inputs.is_empty() {
        return Ok(Vec::new());
    }

    let scalar_result =
        kernel.messages_to_scalar_batch(challenge_inputs, bbs_message_scalar_dst(), fail_fast);
    if scalar_result.failed > 0 {
        return Err(Error::Gpu(
            "GPU proof challenge scalar precompute failed".to_owned(),
        ));
    }

    scalar_result
        .results
        .into_iter()
        .map(|item| item.result.and_then(parse_gpu_message_scalar))
        .collect()
}

pub(super) fn parse_gpu_message_scalar(bytes: Vec<u8>) -> Result<BBSplusMessage> {
    parse_message_scalar_bytes(&bytes)
}

pub(super) fn parse_message_scalar_bytes(bytes: &[u8]) -> Result<BBSplusMessage> {
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| Error::Gpu("message scalar has invalid length".to_owned()))?;
    BBSplusMessage::from_bytes_be(&bytes)
        .map_err(|_| Error::Gpu("message scalar is not canonical".to_owned()))
}

pub(super) fn gpu_status_message(status: i32) -> String {
    match status {
        gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT => "gpu invalid input".to_owned(),
        gpu_abi::CUBBS_BBS_GPU_ERR_INSUFFICIENT_OUTPUT => {
            "gpu output buffer is too small".to_owned()
        }
        gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL => "gpu internal error".to_owned(),
        other => format!("gpu call failed with status {other}"),
    }
}

pub(super) fn gpu_item_error(result: gpu_abi::CuBbsBbsGpuResult) -> Error {
    match result.code {
        gpu_abi::CUBBS_BBS_GPU_ERR_INVALID_INPUT => Error::Gpu("gpu item invalid input".to_owned()),
        gpu_abi::CUBBS_BBS_GPU_ERR_INSUFFICIENT_OUTPUT => {
            Error::Gpu("gpu item output buffer is too small".to_owned())
        }
        gpu_abi::CUBBS_BBS_GPU_ERR_INTERNAL => Error::Gpu("gpu item internal error".to_owned()),
        other => Error::Gpu(format!("gpu item failed with status {other}")),
    }
}

pub(super) fn batch_call_error<T>(
    attempted: usize,
    error: Error,
    fail_fast: bool,
) -> BbsBatchRunResult<T> {
    let result_count = if fail_fast {
        attempted.min(1)
    } else {
        attempted
    };
    let results = (0..result_count)
        .map(|index| BbsBatchItemResult {
            index,
            result: Err(error.clone()),
        })
        .collect::<Vec<_>>();

    BbsBatchRunResult {
        attempted: results.len(),
        succeeded: 0,
        failed: results.len(),
        results,
    }
}

pub(super) fn collect_generated_batch<F>(
    attempted: usize,
    fail_fast: bool,
    gpu_results: Vec<gpu_abi::CuBbsBbsGpuResult>,
    mut outputs: Vec<Vec<u8>>,
    item_len: F,
) -> BbsBatchRunResult<Vec<u8>>
where
    F: Fn(usize) -> usize,
{
    let mut results = Vec::with_capacity(attempted);
    let mut succeeded = 0;
    let mut failed = 0;

    for (index, gpu_result) in gpu_results.into_iter().enumerate() {
        if gpu_result.code == gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            let written = gpu_result
                .bytes_written
                .max(item_len(index))
                .min(outputs[index].len());
            outputs[index].truncate(written);
            succeeded += 1;
            results.push(BbsBatchItemResult {
                index,
                result: Ok(std::mem::take(&mut outputs[index])),
            });
        } else {
            failed += 1;
            results.push(BbsBatchItemResult {
                index,
                result: Err(gpu_item_error(gpu_result)),
            });
            if fail_fast {
                break;
            }
        }
    }

    BbsBatchRunResult {
        attempted: results.len(),
        succeeded,
        failed,
        results,
    }
}

pub(super) fn collect_boolean_batch(
    attempted: usize,
    fail_fast: bool,
    gpu_results: Vec<gpu_abi::CuBbsBbsGpuResult>,
) -> BbsBatchRunResult<bool> {
    let mut results = Vec::with_capacity(attempted);
    let mut succeeded = 0;
    let mut failed = 0;

    for (index, gpu_result) in gpu_results.into_iter().enumerate() {
        if gpu_result.code == gpu_abi::CUBBS_BBS_GPU_SUCCESS {
            succeeded += 1;
            results.push(BbsBatchItemResult {
                index,
                result: Ok(gpu_result.value != 0),
            });
        } else {
            failed += 1;
            results.push(BbsBatchItemResult {
                index,
                result: Err(gpu_item_error(gpu_result)),
            });
            if fail_fast {
                break;
            }
        }
    }

    BbsBatchRunResult {
        attempted: results.len(),
        succeeded,
        failed,
        results,
    }
}

pub(super) fn parse_secret_key(bytes: &[u8]) -> Result<BBSplusSecretKey> {
    if bytes.len() != SECRET_KEY_LEN {
        return Err(Error::InvalidSecretKey);
    }

    BBSplusSecretKey::from_bytes(bytes).map_err(|_| Error::InvalidSecretKey)
}

pub(super) fn parse_public_key(bytes: &[u8]) -> Result<BBSplusPublicKey> {
    if bytes.len() != PUBLIC_KEY_LEN {
        return Err(Error::InvalidPublicKey);
    }

    static CACHE: OnceLock<RwLock<HashMap<Vec<u8>, BBSplusPublicKey>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let guard = cache.read().expect("BBS public key parse cache poisoned");
        if let Some(public_key) = guard.get(bytes) {
            return Ok(public_key.clone());
        }
    }

    let public_key = BBSplusPublicKey::from_bytes(bytes).map_err(|_| Error::InvalidPublicKey)?;
    let mut guard = cache.write().expect("BBS public key parse cache poisoned");
    Ok(guard
        .entry(bytes.to_vec())
        .or_insert_with(|| public_key.clone())
        .clone())
}

pub(super) fn parse_signature(bytes: &[u8]) -> Result<ZkSignature<BBSplus<Bls12381Sha256>>> {
    let bytes: &[u8; SIGNATURE_LEN] = bytes.try_into().map_err(|_| Error::InvalidSignature)?;
    ZkSignature::<BBSplus<Bls12381Sha256>>::from_bytes(bytes).map_err(|_| Error::InvalidSignature)
}

// CFRG BBS accepts an arbitrary octet-string header (including empty). The
// 64-byte BBS-2023 header shape belongs to the W3C integration layer, whose
// hashing pipeline constructs exactly that size before entering this facade.
pub(super) fn ensure_header(_header: &[u8]) -> Result<()> {
    Ok(())
}

pub(super) fn ensure_disclosed_indexes(indexes: &[usize], message_count: usize) -> Result<()> {
    let mut previous = None;
    for &index in indexes {
        if index >= message_count || previous.is_some_and(|p| index <= p) {
            return Err(Error::InvalidDisclosedIndexes);
        }
        previous = Some(index);
    }
    Ok(())
}
