use super::*;

fn header() -> Vec<u8> {
    [[0x11; 32], [0x22; 32]].concat()
}

fn messages() -> Vec<Vec<u8>> {
    vec![
        b"issuer=did:key:zExample".to_vec(),
        b"credentialSubject.id=did:example:alice".to_vec(),
        b"credentialSubject.name=Alice".to_vec(),
        b"credentialSubject.age=42".to_vec(),
        b"credentialSubject.license=ABC-123".to_vec(),
    ]
}

#[test]
fn direct_zkryptium_sign_verify_proof_round_trip() {
    let engine = CuBbsProtocolEngine::new();
    let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
    let header = header();
    let messages = messages();

    let signature = engine
        .sign(SignRequest {
            secret_key: keys.secret_key.clone(),
            public_key: keys.public_key.clone(),
            header: header.clone(),
            messages: messages.clone(),
        })
        .unwrap();
    assert_eq!(signature.len(), SIGNATURE_LEN);

    assert!(engine
        .verify(VerifyRequest {
            public_key: keys.public_key.clone(),
            signature: signature.clone(),
            header: header.clone(),
            messages: messages.clone(),
        })
        .unwrap());

    let disclosed_indexes = vec![0, 1, 4];
    let disclosed_messages = disclosed_indexes
        .iter()
        .map(|&i| messages[i].clone())
        .collect::<Vec<_>>();
    let presentation_header = b"presentation-header".to_vec();

    let proof = engine
        .proof_gen(ProofGenRequest {
            public_key: keys.public_key.clone(),
            signature,
            header: header.clone(),
            presentation_header: presentation_header.clone(),
            messages: messages.clone(),
            disclosed_indexes: disclosed_indexes.clone(),
        })
        .unwrap();

    assert!(engine
        .proof_verify(ProofVerifyRequest {
            public_key: keys.public_key.clone(),
            proof: proof.clone(),
            header,
            presentation_header,
            disclosed_messages,
            disclosed_indexes,
            total_message_count: messages.len(),
        })
        .unwrap());
}

#[test]
fn prepared_b_signing_matches_standard_signature() {
    let engine = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
    let req = SignRequest {
        secret_key: keys.secret_key.clone(),
        public_key: keys.public_key.clone(),
        header: header(),
        messages: messages(),
    };

    let standard = CuBbsProtocolEngine::new().sign(req.clone()).unwrap();
    let prepared_b = engine.prepare_b_for_sign(&req).unwrap();
    assert_eq!(prepared_b.compressed_b.len(), 48);

    let prepared = engine
        .sign_with_prepared_b(req.clone(), prepared_b)
        .unwrap();
    assert_eq!(prepared, standard);

    assert!(engine
        .verify(VerifyRequest {
            public_key: keys.public_key,
            signature: prepared,
            header: req.header,
            messages: req.messages,
        })
        .unwrap());
}

#[test]
fn prepare_b_msm_terms_match_bbs_shape() {
    let engine = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
    let req = SignRequest {
        secret_key: keys.secret_key,
        public_key: keys.public_key,
        header: header(),
        messages: messages(),
    };

    let terms = engine.prepare_b_msm_terms_for_sign(&req).unwrap();
    assert_eq!(terms.compressed_points.len(), req.messages.len() + 2);
    assert_eq!(terms.scalar_be.len(), req.messages.len() + 2);
    assert!(terms
        .compressed_points
        .iter()
        .all(|point| point.len() == 48));
    assert!(terms.scalar_be.iter().all(|scalar| scalar.len() == 32));
    assert_eq!(terms.scalar_be[0][..31], [0u8; 31]);
    assert_eq!(terms.scalar_be[0][31], 1);
}

#[test]
fn parses_backend_aliases() {
    assert_eq!(BbsBackend::parse("cpu").unwrap(), BbsBackend::ZkryptiumCpu);
    assert_eq!(
        BbsBackend::parse("zkryptium-cpu").unwrap(),
        BbsBackend::ZkryptiumCpu
    );
    assert_eq!(
        BbsBackend::parse("gpu").unwrap(),
        BbsBackend::GpuRequestedCpuFallback
    );
    assert_eq!(
        BbsBackend::parse("gpu-requested-cpu-fallback").unwrap(),
        BbsBackend::GpuRequestedCpuFallback
    );
}

#[test]
fn proof_verify_gpu_msm_policy_respects_force_mode() {
    let request = |disclosed_count| ProofVerifyRequest {
        public_key: Vec::new(),
        proof: Vec::new(),
        header: Vec::new(),
        presentation_header: Vec::new(),
        disclosed_messages: vec![Vec::new(); disclosed_count],
        disclosed_indexes: (0..disclosed_count).collect(),
        total_message_count: 6,
    };
    if force_gpu_execution() {
        assert!(proof_verify_gpu_msm_enabled(&request(5)));
        assert!(proof_verify_gpu_msm_enabled(&request(4)));
    } else {
        assert!(!proof_verify_gpu_msm_enabled(&request(5)));
        assert!(proof_verify_gpu_msm_enabled(&request(4)));
    }
}

#[test]
fn verify_pre_b_aggregate_policy_is_default_on_for_nonempty_batches() {
    assert_eq!(
        verify_aggregate_min_items_from_env(None),
        DEFAULT_VERIFY_PRE_B_AGGREGATE_MIN_ITEMS
    );
    assert!(!verify_pre_b_aggregate_policy(0, 1, None));
    assert!(verify_pre_b_aggregate_policy(1, 1, None));
    assert!(verify_pre_b_aggregate_policy(64, 1, None));
    assert!(!verify_pre_b_aggregate_policy(1, 1, Some("0")));
    assert!(verify_pre_b_aggregate_policy(1, 1, Some("1")));
    assert_eq!(verify_aggregate_min_items_from_env(Some("128")), 128);
    assert_eq!(
        verify_aggregate_min_items_from_env(Some("invalid")),
        DEFAULT_VERIFY_PRE_B_AGGREGATE_MIN_ITEMS
    );
}

#[test]
fn repeated_verify_scalars_are_aggregated_once_per_lane() {
    let request = VerifyRequest {
        public_key: Vec::new(),
        signature: Vec::new(),
        header: Vec::new(),
        messages: vec![Vec::new(); 2],
    };
    let messages = [
        parse_message_scalar_bytes(&Scalar::from(3u64).to_bytes_be()).unwrap(),
        parse_message_scalar_bytes(&Scalar::from(5u64).to_bytes_be()).unwrap(),
    ];
    let prepared = (0..5)
        .map(|index| (index, &request, messages.as_slice()))
        .collect::<Vec<_>>();
    let coefficients = (1u64..=5).map(Scalar::from).collect::<Vec<_>>();
    let lanes = verify_aggregate_shape_scalar_lanes(
        &prepared,
        &[0, 1, 2, 3, 4],
        &coefficients,
        Scalar::from(7u64),
        4,
        2,
    );

    for (lane, coefficient_sum) in lanes.iter().zip([9u64, 6]) {
        let coefficient_sum = Scalar::from(coefficient_sum);
        assert_eq!(lane[0], coefficient_sum);
        assert_eq!(lane[1], Scalar::from(7u64) * coefficient_sum);
        assert_eq!(lane[2], Scalar::from(3u64) * coefficient_sum);
        assert_eq!(lane[3], Scalar::from(5u64) * coefficient_sum);
    }
}

#[test]
fn repeated_verify_a_terms_are_aggregated_once_per_lane() {
    let mut signature = vec![0u8; SIGNATURE_LEN];
    signature[48..].copy_from_slice(&Scalar::from(7u64).to_bytes_be());
    let request = VerifyRequest {
        public_key: Vec::new(),
        signature,
        header: Vec::new(),
        messages: vec![Vec::new(); 2],
    };
    let messages = [
        parse_message_scalar_bytes(&Scalar::from(3u64).to_bytes_be()).unwrap(),
        parse_message_scalar_bytes(&Scalar::from(5u64).to_bytes_be()).unwrap(),
    ];
    let prepared = (0..5)
        .map(|index| (index, &request, messages.as_slice()))
        .collect::<Vec<_>>();
    let coefficients = (1u64..=5).map(Scalar::from).collect::<Vec<_>>();

    let lanes = repeated_verify_a_scalar_lanes(&prepared, &coefficients, 2).unwrap();
    assert_eq!(lanes[0], (Scalar::from(9u64), -Scalar::from(63u64)));
    assert_eq!(lanes[1], (Scalar::from(6u64), -Scalar::from(42u64)));

    let distinct_request = request.clone();
    let distinct = [
        (0, &request, messages.as_slice()),
        (1, &distinct_request, messages.as_slice()),
    ];
    assert!(repeated_verify_a_scalar_lanes(&distinct, &coefficients[..2], 2).is_none());
}

#[test]
fn verify_dual_same_points_defaults_on_for_non_singleton_aggregates() {
    assert!(!gpu_verify_dual_same_points_enabled_from_env(None, 0));
    assert!(!gpu_verify_dual_same_points_enabled_from_env(None, 1));
    assert!(gpu_verify_dual_same_points_enabled_from_env(None, 2));
    assert!(gpu_verify_dual_same_points_enabled_from_env(None, 512));
    assert!(gpu_verify_dual_same_points_enabled_from_env(Some("1"), 1));
    assert!(!gpu_verify_dual_same_points_enabled_from_env(
        Some("0"),
        512
    ));
}

#[test]
fn gpu_abi_fixed_lengths_match_bbs_constants() {
    assert_eq!(
        gpu_abi::fixed_lengths(),
        (
            SECRET_KEY_LEN,
            PUBLIC_KEY_LEN,
            SIGNATURE_LEN,
            BBS2023_HEADER_LEN
        )
    );
    assert!(gpu_abi::CUBBS_BBS_GPU_SIGN_BATCH_SYMBOL.ends_with(&[0]));
    assert!(gpu_abi::CUBBS_BBS_GPU_PROOF_VERIFY_BATCH_SYMBOL.ends_with(&[0]));
    assert!(gpu_abi::CUBBS_BBS_GPU_PREPARE_B_BATCH_SYMBOL.ends_with(&[0]));
    assert!(gpu_abi::CUBBS_BBS_GPU_G1_MSM_BATCH_SYMBOL.ends_with(&[0]));
    assert_eq!(
        BbsBackend::GpuRequestedCpuFallback.info().gpu_abi_version,
        gpu_abi::CUBBS_BBS_GPU_ABI_VERSION
    );
}

#[test]
fn gpu_requested_backend_preserves_bbs_round_trip() {
    let engine = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
    let header = header();
    let messages = messages();
    let signature = engine
        .sign(SignRequest {
            secret_key: keys.secret_key.clone(),
            public_key: keys.public_key.clone(),
            header: header.clone(),
            messages: messages.clone(),
        })
        .unwrap();

    assert!(engine
        .verify(VerifyRequest {
            public_key: keys.public_key,
            signature,
            header,
            messages,
        })
        .unwrap());
    assert!(engine.backend().is_gpu_requested());
}

#[test]
fn batch_sign_and_verify_preserve_order() {
    let engine = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
    let header = header();
    let batch = (0..4)
        .map(|i| {
            let mut messages = messages();
            messages.push(format!("batch-index={i}").into_bytes());
            SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages,
            }
        })
        .collect::<Vec<_>>();
    let verify_messages = batch
        .iter()
        .map(|req| req.messages.clone())
        .collect::<Vec<_>>();

    let signed = engine.sign_batch(batch, false);
    assert_eq!(signed.attempted, 4);
    assert_eq!(signed.succeeded, 4);
    assert_eq!(signed.failed, 0);
    assert_eq!(
        signed
            .results
            .iter()
            .map(|item| item.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );

    let verify_reqs = signed
        .results
        .into_iter()
        .map(|item| VerifyRequest {
            public_key: keys.public_key.clone(),
            signature: item.result.unwrap(),
            header: header.clone(),
            messages: verify_messages[item.index].clone(),
        })
        .collect::<Vec<_>>();
    let mut tampered_signature = verify_reqs[0].clone();
    tampered_signature.signature[0] ^= 0x01;
    let mut tampered_message = verify_reqs[1].clone();
    tampered_message.messages[0].push(0xff);

    let verified = engine.verify_batch(verify_reqs.clone(), false);
    assert_eq!(verified.attempted, 4);
    assert_eq!(verified.succeeded, 4);
    assert!(verified
        .results
        .into_iter()
        .all(|item| item.result.unwrap()));

    let mut tampered_e = verify_reqs[2].clone();
    tampered_e.signature[SIGNATURE_LEN - 1] ^= 0x01;
    let mixed = engine.verify_batch(
        vec![verify_reqs[3].clone(), tampered_e, tampered_message.clone()],
        false,
    );
    assert_eq!(mixed.attempted, 3);
    assert_eq!(mixed.succeeded, 3);
    assert_eq!(
        mixed
            .results
            .into_iter()
            .map(|item| item.result.unwrap())
            .collect::<Vec<_>>(),
        vec![true, false, false]
    );

    assert!(!engine.verify(tampered_signature).unwrap_or(false));
    assert!(!engine.verify(tampered_message).unwrap());
}

#[test]
fn batch_proof_gen_and_verify_preserve_order() {
    let engine = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
    let header = header();
    let presentation_header = b"batch-proof-presentation".to_vec();
    let disclosed_indexes = vec![0, 2, 4];
    let batch = (0..4)
        .map(|i| {
            let mut messages = messages();
            messages.push(format!("proof-batch-index={i}").into_bytes());
            messages
        })
        .collect::<Vec<_>>();
    let signatures = engine.sign_batch(
        batch
            .iter()
            .map(|messages| SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages: messages.clone(),
            })
            .collect(),
        false,
    );
    assert_eq!(signatures.succeeded, 4);

    let proof_reqs = signatures
        .results
        .iter()
        .map(|item| ProofGenRequest {
            public_key: keys.public_key.clone(),
            signature: item.result.as_ref().unwrap().clone(),
            header: header.clone(),
            presentation_header: presentation_header.clone(),
            messages: batch[item.index].clone(),
            disclosed_indexes: disclosed_indexes.clone(),
        })
        .collect::<Vec<_>>();
    let proofs = engine.proof_gen_batch(proof_reqs, false);
    assert_eq!(proofs.attempted, 4);
    assert_eq!(proofs.succeeded, 4);
    assert_eq!(
        proofs
            .results
            .iter()
            .map(|item| item.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );

    let verify_reqs = proofs
        .results
        .into_iter()
        .map(|item| ProofVerifyRequest {
            public_key: keys.public_key.clone(),
            proof: item.result.unwrap(),
            header: header.clone(),
            presentation_header: presentation_header.clone(),
            disclosed_messages: disclosed_indexes
                .iter()
                .map(|&index| batch[item.index][index].clone())
                .collect(),
            disclosed_indexes: disclosed_indexes.clone(),
            total_message_count: batch[item.index].len(),
        })
        .collect::<Vec<_>>();
    let mut tampered_req = verify_reqs[0].clone();
    tampered_req.presentation_header.push(0xff);
    let verified = engine.proof_verify_batch(verify_reqs, false);
    assert_eq!(verified.attempted, 4);
    assert_eq!(verified.succeeded, 4);
    assert!(verified
        .results
        .into_iter()
        .all(|item| item.result.unwrap()));
    assert!(!engine.proof_verify(tampered_req).unwrap());
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_verify_pre_b_aggregate_at_default_threshold -- --ignored"]
fn loaded_gpu_verify_pre_b_aggregate_at_default_threshold() {
    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );
    assert!(
        force_gpu_execution(),
        "pre-B aggregate correctness must run with CUBBS_BBS_FORCE_GPU=1 so no operation can silently use the CPU fallback"
    );
    assert_eq!(
        verify_aggregate_min_items(),
        DEFAULT_VERIFY_PRE_B_AGGREGATE_MIN_ITEMS,
        "run this boundary test without overriding CUBBS_BBS_VERIFY_AGGREGATE_MIN_ITEMS"
    );
    assert!(verify_pre_b_aggregate_enabled(1));
    assert!(verify_pre_b_aggregate_enabled(64));

    let cpu = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let gpu = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let info = gpu.backend_info();
    assert!(
        info.hardware_acceleration && !info.fallback,
        "pre-B aggregate test requires a hardware GPU backend: {info:?}"
    );

    let keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
    let message_counts = [16usize, 64, 128];
    let messages = (0..64)
        .map(|item_index| {
            let message_count = message_counts[item_index % message_counts.len()];
            (0..message_count)
                .map(|message_index| {
                    format!("pre-b-aggregate:item={item_index}:message={message_index}")
                        .into_bytes()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let headers = messages
        .iter()
        .map(|messages| {
            let mut header = vec![0u8; BBS2023_HEADER_LEN];
            header[..8].copy_from_slice(&(messages.len() as u64).to_be_bytes());
            header
        })
        .collect::<Vec<_>>();
    let signatures = cpu.sign_batch(
        messages
            .iter()
            .zip(headers.iter())
            .map(|(messages, header)| SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages: messages.clone(),
            })
            .collect(),
        false,
    );
    assert_eq!(signatures.succeeded, 64);
    assert_eq!(signatures.failed, 0);

    let verify_requests = signatures
        .results
        .iter()
        .map(|item| VerifyRequest {
            public_key: keys.public_key.clone(),
            signature: item.result.as_ref().unwrap().clone(),
            header: headers[item.index].clone(),
            messages: messages[item.index].clone(),
        })
        .collect::<Vec<_>>();

    let singleton = gpu.verify_batch_borrowed(&verify_requests[..1], false);
    assert_eq!(singleton.attempted, 1);
    assert_eq!(singleton.failed, 0);
    assert!(singleton.results[0].result.as_ref().unwrap());

    let mut invalid_singleton = vec![verify_requests[0].clone()];
    invalid_singleton[0].messages[0].push(0xff);
    let singleton_checked = gpu.verify_batch_borrowed(&invalid_singleton, false);
    assert_eq!(singleton_checked.attempted, 1);
    assert_eq!(singleton_checked.failed, 0);
    assert!(!singleton_checked.results[0].result.as_ref().unwrap());

    let valid = gpu.verify_batch_borrowed(&verify_requests, false);
    assert_eq!(valid.attempted, 64);
    assert_eq!(valid.failed, 0);
    assert!(valid
        .results
        .iter()
        .all(|item| item.result.as_ref().is_ok_and(|value| *value)));

    let invalid_index = 31usize;
    let mut one_invalid = verify_requests;
    one_invalid[invalid_index].messages[0].push(0xff);
    let checked = gpu.verify_batch_borrowed(&one_invalid, false);
    assert_eq!(checked.attempted, 64);
    assert_eq!(checked.failed, 0);
    assert_eq!(checked.results.len(), 64);
    for item in checked.results {
        assert_eq!(
            item.result.unwrap(),
            item.index != invalid_index,
            "aggregate failure must fall back and isolate the invalid request"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_repeated_shared_sign_verify_round_trip -- --ignored --nocapture"]
fn loaded_gpu_repeated_shared_sign_verify_round_trip() {
    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );
    assert!(
        force_gpu_execution(),
        "repeated-path correctness must run with CUBBS_BBS_FORCE_GPU=1"
    );

    let cpu = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let gpu = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let info = gpu.backend_info();
    assert!(
        info.hardware_acceleration && !info.fallback,
        "repeated-path test requires a hardware GPU backend: {info:?}"
    );

    let keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
    let sign_request = Arc::new(SignRequest {
        secret_key: keys.secret_key.clone(),
        public_key: keys.public_key.clone(),
        header: header(),
        messages: (0..64)
            .map(|index| format!("repeated-message-{index}").into_bytes())
            .collect(),
    });
    let shared_sign_requests = vec![Arc::clone(&sign_request); 257];
    let signed = gpu.sign_batch_shared(&shared_sign_requests, false);
    assert_eq!(signed.attempted, shared_sign_requests.len());
    assert_eq!(signed.succeeded, shared_sign_requests.len());
    assert_eq!(signed.failed, 0);
    assert_eq!(signed.results.len(), shared_sign_requests.len());
    for item in &signed.results {
        assert!(cpu
            .verify(VerifyRequest {
                public_key: keys.public_key.clone(),
                signature: item.result.as_ref().unwrap().clone(),
                header: sign_request.header.clone(),
                messages: sign_request.messages.clone(),
            })
            .unwrap());
    }

    let verify_request = Arc::new(VerifyRequest {
        public_key: keys.public_key,
        signature: signed.results[0].result.as_ref().unwrap().clone(),
        header: sign_request.header.clone(),
        messages: sign_request.messages.clone(),
    });
    let shared_verify_requests = vec![Arc::clone(&verify_request); 257];
    let verified = gpu.verify_batch_shared(&shared_verify_requests, false);
    assert_eq!(verified.attempted, shared_verify_requests.len());
    assert_eq!(verified.succeeded, shared_verify_requests.len());
    assert_eq!(verified.failed, 0);
    assert!(verified
        .results
        .iter()
        .all(|item| item.result.as_ref().is_ok_and(|value| *value)));

    let mut tampered = (*verify_request).clone();
    tampered.messages[0].push(0xff);
    let tampered = Arc::new(tampered);
    let checked = gpu.verify_batch_shared(&vec![tampered; 257], false);
    assert_eq!(checked.attempted, 257);
    assert_eq!(checked.failed, 0);
    assert!(checked
        .results
        .iter()
        .all(|item| item.result.as_ref().is_ok_and(|value| !*value)));
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_ -- --ignored"]
fn loaded_gpu_four_operation_matrix_matches_cpu() {
    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );

    let cpu = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let gpu = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let info = gpu.backend_info();
    assert!(
        info.hardware_acceleration && !info.fallback,
        "correctness matrix requires a hardware GPU backend: {info:?}"
    );

    fn run_case(
        label: &str,
        cpu: &BbsEngine,
        gpu: &BbsEngine,
        message_counts: &[usize],
        unique_issuers: bool,
    ) {
        let shared_keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
        let keys = (0..message_counts.len())
            .map(|_| {
                if unique_issuers {
                    cpu.generate_keypair(&mut rand::thread_rng()).unwrap()
                } else {
                    shared_keys.clone()
                }
            })
            .collect::<Vec<_>>();
        let headers = message_counts
            .iter()
            .enumerate()
            .map(|(item_index, &message_count)| {
                let mut value = vec![0u8; BBS2023_HEADER_LEN];
                value[..8].copy_from_slice(&(item_index as u64).to_be_bytes());
                value[8..16].copy_from_slice(&(message_count as u64).to_be_bytes());
                value[16..16 + label.len()].copy_from_slice(label.as_bytes());
                value
            })
            .collect::<Vec<_>>();
        let messages = message_counts
            .iter()
            .enumerate()
            .map(|(item_index, &message_count)| {
                (0..message_count)
                    .map(|message_index| {
                        format!("{label}:item={item_index}:message={message_index}").into_bytes()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let sign_requests = messages
            .iter()
            .zip(headers.iter())
            .zip(keys.iter())
            .map(|((messages, header), keys)| SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages: messages.clone(),
            })
            .collect::<Vec<_>>();

        let cpu_signed = cpu.sign_batch(sign_requests.clone(), false);
        let gpu_signed = gpu.sign_batch(sign_requests, false);
        for signed in [&cpu_signed, &gpu_signed] {
            assert_eq!(signed.attempted, message_counts.len(), "{label}");
            assert_eq!(signed.succeeded, message_counts.len(), "{label}");
            assert_eq!(signed.failed, 0, "{label}");
            assert_eq!(
                signed
                    .results
                    .iter()
                    .map(|item| item.index)
                    .collect::<Vec<_>>(),
                (0..message_counts.len()).collect::<Vec<_>>(),
                "{label}"
            );
        }
        let cpu_signatures = cpu_signed
            .results
            .iter()
            .map(|item| item.result.as_ref().unwrap().clone())
            .collect::<Vec<_>>();
        let gpu_signatures = gpu_signed
            .results
            .iter()
            .map(|item| item.result.as_ref().unwrap().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            gpu_signatures, cpu_signatures,
            "{label}: GPU signatures must match the CPU reference"
        );

        let verify_requests = cpu_signatures
            .iter()
            .zip(messages.iter())
            .zip(headers.iter())
            .zip(keys.iter())
            .map(|(((signature, messages), header), keys)| VerifyRequest {
                public_key: keys.public_key.clone(),
                signature: signature.clone(),
                header: header.clone(),
                messages: messages.clone(),
            })
            .collect::<Vec<_>>();
        for (backend, verified) in [
            ("cpu", cpu.verify_batch(verify_requests.clone(), false)),
            ("gpu", gpu.verify_batch(verify_requests.clone(), false)),
        ] {
            assert_eq!(verified.failed, 0, "{label}:{backend}");
            assert!(
                verified
                    .results
                    .iter()
                    .all(|item| item.result.as_ref().is_ok_and(|valid| *valid)),
                "{label}:{backend}"
            );
        }

        let disclosed_indexes = message_counts
            .iter()
            .map(|&message_count| (0..message_count).step_by(2).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let presentation_header = format!("cubbs-bbs-correctness:{label}").into_bytes();
        let proof_requests = cpu_signatures
            .iter()
            .zip(messages.iter())
            .zip(headers.iter())
            .zip(disclosed_indexes.iter())
            .zip(keys.iter())
            .map(
                |((((signature, messages), header), disclosed_indexes), keys)| ProofGenRequest {
                    public_key: keys.public_key.clone(),
                    signature: signature.clone(),
                    header: header.clone(),
                    presentation_header: presentation_header.clone(),
                    messages: messages.clone(),
                    disclosed_indexes: disclosed_indexes.clone(),
                },
            )
            .collect::<Vec<_>>();
        let cpu_proofs = cpu.proof_gen_batch(proof_requests.clone(), false);
        let gpu_proofs = gpu.proof_gen_batch(proof_requests, false);
        for (backend, proofs) in [("cpu", &cpu_proofs), ("gpu", &gpu_proofs)] {
            assert_eq!(proofs.attempted, message_counts.len(), "{label}:{backend}");
            assert_eq!(proofs.succeeded, message_counts.len(), "{label}:{backend}");
            assert_eq!(proofs.failed, 0, "{label}:{backend}");
            assert_eq!(
                proofs
                    .results
                    .iter()
                    .map(|item| item.index)
                    .collect::<Vec<_>>(),
                (0..message_counts.len()).collect::<Vec<_>>(),
                "{label}:{backend}"
            );
        }

        for (producer, proofs) in [("cpu", cpu_proofs), ("gpu", gpu_proofs)] {
            let proof_verify_requests = proofs
                .results
                .iter()
                .map(|item| ProofVerifyRequest {
                    public_key: keys[item.index].public_key.clone(),
                    proof: item.result.as_ref().unwrap().clone(),
                    header: headers[item.index].clone(),
                    presentation_header: presentation_header.clone(),
                    disclosed_messages: disclosed_indexes[item.index]
                        .iter()
                        .map(|&index| messages[item.index][index].clone())
                        .collect(),
                    disclosed_indexes: disclosed_indexes[item.index].clone(),
                    total_message_count: message_counts[item.index],
                })
                .collect::<Vec<_>>();
            for (verifier, verified) in [
                (
                    "cpu",
                    cpu.proof_verify_batch(proof_verify_requests.clone(), false),
                ),
                (
                    "gpu",
                    gpu.proof_verify_batch(proof_verify_requests.clone(), false),
                ),
            ] {
                assert_eq!(verified.failed, 0, "{label}:{producer}->{verifier}");
                assert!(
                    verified
                        .results
                        .iter()
                        .all(|item| item.result.as_ref().is_ok_and(|valid| *valid)),
                    "{label}:{producer}->{verifier}"
                );
            }

            let mut tampered = proof_verify_requests[0].clone();
            tampered.presentation_header.push(0xff);
            assert!(
                !cpu.proof_verify(tampered.clone()).unwrap(),
                "{label}:{producer}->cpu tampered proof"
            );
            assert!(
                !gpu.proof_verify(tampered).unwrap(),
                "{label}:{producer}->gpu tampered proof"
            );
        }

        let mut tampered_message = verify_requests[0].clone();
        tampered_message.messages[0].push(0xff);
        assert!(
            !cpu.verify(tampered_message.clone()).unwrap(),
            "{label}:cpu tampered signature input"
        );
        assert!(
            !gpu.verify(tampered_message).unwrap(),
            "{label}:gpu tampered signature input"
        );
    }

    run_case(
        "tier-h-mixed",
        &cpu,
        &gpu,
        &[16, 64, 128, 16, 64, 128],
        false,
    );
    run_case(
        "tier-u-mixed",
        &cpu,
        &gpu,
        &[16, 64, 128, 16, 64, 128],
        true,
    );
    run_case("tier-x-16", &cpu, &gpu, &[16, 16, 16], false);
    run_case("tier-x-64", &cpu, &gpu, &[64, 64, 64], false);
    run_case("tier-x-128", &cpu, &gpu, &[128, 128, 128], false);
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_65536_four_operations_match_cpu_and_reject_tampering -- --ignored --nocapture"]
fn loaded_gpu_65536_four_operations_match_cpu_and_reject_tampering() {
    const MESSAGE_COUNT: usize = 65_536;

    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );
    assert!(
        force_gpu_execution(),
        "65,536-message correctness must run with CUBBS_BBS_FORCE_GPU=1 so no operation can silently use the CPU fallback"
    );

    let cpu = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let gpu = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let info = gpu.backend_info();
    assert!(
        info.hardware_acceleration && !info.fallback,
        "65,536-message correctness test requires a hardware GPU backend: {info:?}"
    );

    let keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
    let mut header = vec![0u8; BBS2023_HEADER_LEN];
    header[..8].copy_from_slice(&(MESSAGE_COUNT as u64).to_be_bytes());
    header[8..33].copy_from_slice(b"omniid-route4-correctness");
    let messages = (0..MESSAGE_COUNT)
        .map(|index| {
            let mut message = vec![0u8; 32];
            message[..8].copy_from_slice(&(index as u64).to_be_bytes());
            message[8..].copy_from_slice(b"route4-message-fixture!!");
            message
        })
        .collect::<Vec<_>>();
    let sign_request = SignRequest {
        secret_key: keys.secret_key.clone(),
        public_key: keys.public_key.clone(),
        header: header.clone(),
        messages: messages.clone(),
    };

    let route_terms = cpu.prepare_b_msm_terms_for_sign(&sign_request).unwrap();
    let route = gpu
        .g1_msm_route_summary_for_bench(&[route_terms], false, true)
        .unwrap();
    assert_eq!(
        route.cubbs_generic, 1,
        "65,536 messages must select Route 4"
    );
    assert_eq!(route.unknown, 0);
    assert_eq!(route.blst_item + route.blst_common_points, 0);
    assert_eq!(route.sppark_item + route.sppark_common_points, 0);

    let cpu_signature = cpu.sign(sign_request.clone()).unwrap();
    let gpu_signature = gpu.sign(sign_request).unwrap();
    assert_eq!(
        gpu_signature, cpu_signature,
        "GPU Sign must match the deterministic CPU reference at 65,536 messages"
    );

    let verify_request = |signature: Vec<u8>| VerifyRequest {
        public_key: keys.public_key.clone(),
        signature,
        header: header.clone(),
        messages: messages.clone(),
    };
    for (producer, signature) in [
        ("cpu", cpu_signature.clone()),
        ("gpu", gpu_signature.clone()),
    ] {
        let request = verify_request(signature);
        assert!(
            cpu.verify(request.clone()).unwrap(),
            "{producer} signature must verify on CPU"
        );
        assert!(
            gpu.verify(request).unwrap(),
            "{producer} signature must verify on GPU"
        );
    }

    let mut tampered_signature_input = verify_request(cpu_signature.clone());
    tampered_signature_input.messages[MESSAGE_COUNT / 2].push(0xff);
    assert!(!cpu.verify(tampered_signature_input.clone()).unwrap());
    assert!(!gpu.verify(tampered_signature_input).unwrap());

    let disclosed_indexes = (0..MESSAGE_COUNT).step_by(2).collect::<Vec<_>>();
    assert_eq!(disclosed_indexes.len(), MESSAGE_COUNT / 2);
    let disclosed_messages = disclosed_indexes
        .iter()
        .map(|&index| messages[index].clone())
        .collect::<Vec<_>>();
    let presentation_header = b"omniid-route4-65536-presentation".to_vec();
    let proof_gen_request = ProofGenRequest {
        public_key: keys.public_key.clone(),
        signature: cpu_signature,
        header: header.clone(),
        presentation_header: presentation_header.clone(),
        messages,
        disclosed_indexes: disclosed_indexes.clone(),
    };

    let cpu_proof = cpu.proof_gen(proof_gen_request.clone()).unwrap();
    let gpu_proof = gpu.proof_gen(proof_gen_request).unwrap();
    for (producer, proof) in [("cpu", cpu_proof), ("gpu", gpu_proof)] {
        // ProofGen is randomized, so correctness is established by
        // independent verification rather than byte-for-byte equality.
        let request = ProofVerifyRequest {
            public_key: keys.public_key.clone(),
            proof,
            header: header.clone(),
            presentation_header: presentation_header.clone(),
            disclosed_messages: disclosed_messages.clone(),
            disclosed_indexes: disclosed_indexes.clone(),
            total_message_count: MESSAGE_COUNT,
        };
        assert!(
            cpu.proof_verify(request.clone()).unwrap(),
            "{producer} proof must verify on CPU"
        );
        assert!(
            gpu.proof_verify(request.clone()).unwrap(),
            "{producer} proof must verify on GPU"
        );

        let mut tampered = request;
        tampered.disclosed_messages[MESSAGE_COUNT / 4].push(0xff);
        assert!(
            !cpu.proof_verify(tampered.clone()).unwrap(),
            "{producer} proof with a tampered disclosed message must fail on CPU"
        );
        assert!(
            !gpu.proof_verify(tampered).unwrap(),
            "{producer} proof with a tampered disclosed message must fail on GPU"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_256_disclosure_extremes_match_cpu_and_reject_tampering -- --ignored --nocapture"]
fn loaded_gpu_256_disclosure_extremes_match_cpu_and_reject_tampering() {
    const MESSAGE_COUNT: usize = 256;

    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );
    assert!(
        force_gpu_execution(),
        "this test requires CUBBS_BBS_FORCE_GPU=1 so no cryptographic fallback is allowed"
    );

    let cpu = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let gpu = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let info = gpu.backend_info();
    assert!(
        info.hardware_acceleration && !info.fallback,
        "256-message disclosure-extremes test requires a hardware GPU backend without fallback: {info:?}"
    );

    // A single key pair deliberately makes both disclosure cases same-issuer.
    let keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
    let mut header = vec![0u8; BBS2023_HEADER_LEN];
    header[..8].copy_from_slice(&(MESSAGE_COUNT as u64).to_be_bytes());
    header[8..38].copy_from_slice(b"omniid-256-disclosure-extremes");
    let messages = (0..MESSAGE_COUNT)
        .map(|index| {
            let mut message = vec![0u8; 34];
            message[..8].copy_from_slice(&(index as u64).to_be_bytes());
            message[8..].copy_from_slice(b"disclosure-extreme-fixture");
            message
        })
        .collect::<Vec<_>>();
    let signature = cpu
        .sign(SignRequest {
            secret_key: keys.secret_key.clone(),
            public_key: keys.public_key.clone(),
            header: header.clone(),
            messages: messages.clone(),
        })
        .unwrap();
    let presentation_header = b"omniid-256-disclosure-presentation".to_vec();

    for (label, disclosure_ratio, all_disclosed) in [
        ("disclosure-0pct", 0.0f64, false),
        ("disclosure-100pct", 1.0f64, true),
    ] {
        let disclosed_indexes = if all_disclosed {
            (0..MESSAGE_COUNT).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let disclosed_messages = disclosed_indexes
            .iter()
            .map(|&index| messages[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            disclosed_indexes.len(),
            (MESSAGE_COUNT as f64 * disclosure_ratio) as usize,
            "{label} must use the requested disclosure ratio"
        );

        let proof_gen_request = ProofGenRequest {
            public_key: keys.public_key.clone(),
            signature: signature.clone(),
            header: header.clone(),
            presentation_header: presentation_header.clone(),
            messages: messages.clone(),
            disclosed_indexes: disclosed_indexes.clone(),
        };
        let cpu_proof = cpu.proof_gen(proof_gen_request.clone()).unwrap();
        let gpu_proof = gpu.proof_gen(proof_gen_request).unwrap();

        for (producer, proof) in [("cpu", cpu_proof), ("gpu", gpu_proof)] {
            // ProofGen is randomized.  Cross-verification, rather than
            // byte equality, proves both independently generated proofs.
            let request = ProofVerifyRequest {
                public_key: keys.public_key.clone(),
                proof,
                header: header.clone(),
                presentation_header: presentation_header.clone(),
                disclosed_messages: disclosed_messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
                total_message_count: MESSAGE_COUNT,
            };
            assert!(
                cpu.proof_verify(request.clone()).unwrap(),
                "{label}:{producer} proof must verify on CPU"
            );
            assert!(
                gpu.proof_verify(request.clone()).unwrap(),
                "{label}:{producer} proof must verify on GPU"
            );

            let mut tampered_header = request.clone();
            tampered_header.presentation_header.push(0xff);
            assert!(
                !cpu.proof_verify(tampered_header.clone()).unwrap(),
                "{label}:{producer} proof with a tampered presentation header must fail on CPU"
            );
            assert!(
                !gpu.proof_verify(tampered_header).unwrap(),
                "{label}:{producer} proof with a tampered presentation header must fail on GPU"
            );

            if all_disclosed {
                let mut tampered_disclosed_message = request;
                tampered_disclosed_message.disclosed_messages[MESSAGE_COUNT / 2].push(0xff);
                assert!(
                    !cpu.proof_verify(tampered_disclosed_message.clone())
                        .unwrap(),
                    "{label}:{producer} proof with a tampered disclosed message must fail on CPU"
                );
                assert!(
                    !gpu.proof_verify(tampered_disclosed_message).unwrap(),
                    "{label}:{producer} proof with a tampered disclosed message must fail on GPU"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_ -- --ignored"]
fn loaded_gpu_library_messages_to_scalar_matches_zkryptium_fixture() {
    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );

    let engine = GpuBbsEngine::new();
    let dst = hex_to_bytes(
        "4242535f424c53313233383147315f584d443a5348412d3235365f535357555f524f5f4832475f484d32535f4d41505f4d53475f544f5f5343414c41525f41535f484153485f",
    );
    let messages = vec![
        hex_to_bytes("9872ad089e452c7b6e283dfac2a80d58e8d0ff71cc4d5e310a1debdda4a45f02"),
        hex_to_bytes("c344136d9ab02da4dd5908bbba913ae6f58c2cc844b802a6f811f5fb075f9b80"),
        Vec::new(),
    ];
    let expected = [
        "1cb5bb86114b34dc438a911617655a1db595abafac92f47c5001799cf624b430",
        "154249d503c093ac2df516d4bb88b510d54fd97e8d7121aede420a25d9521952",
        "08e3afeb2b4f2b5f907924ef42856616e6f2d5f1fb373736db1cca32707a7d16",
    ];

    let result = engine.messages_to_scalar_batch_for_test(messages, dst, true);
    assert_eq!(result.succeeded, expected.len());
    for (item, expected_hex) in result.results.iter().zip(expected.iter()) {
        assert_eq!(hex_string(item.result.as_ref().unwrap()), *expected_hex);
    }
}

#[test]
#[ignore = "requires a real CUDA device; run with CUBBS_BBS_GPU_LIBRARY=<absolute .so> and cargo test -p cubbs-bbs --lib loaded_gpu_ -- --ignored"]
fn loaded_gpu_dual_same_points_matches_independent_msms() {
    assert!(
        std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_some(),
        "CUBBS_BBS_GPU_LIBRARY must name the CUDA ABI shared library"
    );

    let kernel = GpuKernelLibrary::load_shared_from_env()
        .unwrap()
        .expect("GPU library must be loaded");
    assert!(kernel.can_g1_dual_msm_same_points());

    let cpu = CuBbsProtocolEngine::new();
    let keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
    let request = SignRequest {
        secret_key: keys.secret_key,
        public_key: keys.public_key,
        header: header(),
        messages: messages(),
    };
    let terms = cpu.prepare_b_msm_terms_for_sign(&request).unwrap();
    let lhs_scalars = terms
        .scalar_be
        .iter()
        .map(|scalar| scalar.as_slice().try_into().unwrap())
        .collect::<Vec<[u8; 32]>>();
    let rhs_scalars = lhs_scalars
        .iter()
        .enumerate()
        .map(|(index, scalar)| {
            (parse_message_scalar_bytes(scalar).unwrap().value * Scalar::from(index as u64 + 2))
                .to_bytes_be()
        })
        .collect::<Vec<_>>();
    let point_refs = terms
        .compressed_points
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();

    let (dual_lhs, dual_rhs) = kernel
        .g1_dual_msm_same_points_borrowed(&point_refs, &lhs_scalars, &rhs_scalars)
        .expect("dual MSM must succeed");
    let expected = kernel.g1_msm_batch(
        &[
            BbsMsmTerms {
                compressed_points: terms.compressed_points.clone(),
                scalar_be: lhs_scalars.iter().map(|scalar| scalar.to_vec()).collect(),
            },
            BbsMsmTerms {
                compressed_points: terms.compressed_points.clone(),
                scalar_be: rhs_scalars.iter().map(|scalar| scalar.to_vec()).collect(),
            },
        ],
        true,
    );
    assert_eq!(expected.failed, 0);
    assert_eq!(expected.results.len(), 2);
    assert_eq!(dual_lhs, *expected.results[0].result.as_ref().unwrap());
    assert_eq!(dual_rhs, *expected.results[1].result.as_ref().unwrap());

    assert!(kernel.can_g1_dual_msm_same_points_lanes());
    let lane_count = 2;
    let (lane_lhs, lane_rhs) = kernel
        .g1_dual_msm_same_points_lanes_borrowed(&point_refs, &lhs_scalars, &rhs_scalars, lane_count)
        .expect("multi-lane dual MSM must succeed");
    for lane in 0..lane_count {
        let lane_terms = |scalars: &[[u8; 32]]| BbsMsmTerms {
            compressed_points: terms.compressed_points.clone(),
            scalar_be: scalars
                .iter()
                .enumerate()
                .map(|(index, scalar)| {
                    if index % lane_count == lane {
                        scalar.to_vec()
                    } else {
                        vec![0u8; 32]
                    }
                })
                .collect(),
        };
        let lane_expected =
            kernel.g1_msm_batch(&[lane_terms(&lhs_scalars), lane_terms(&rhs_scalars)], true);
        assert_eq!(lane_expected.failed, 0);
        assert_eq!(
            lane_lhs[lane],
            *lane_expected.results[0].result.as_ref().unwrap()
        );
        assert_eq!(
            lane_rhs[lane],
            *lane_expected.results[1].result.as_ref().unwrap()
        );
    }
}

#[test]
#[ignore = "requires CUDA; run with CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_GPU_LIBRARY=<absolute .so>"]
fn loaded_gpu_pairing_common_unique_common_preserves_identity() {
    assert!(force_gpu_execution(), "test must forbid CPU fallback");
    let kernel = GpuKernelLibrary::load_shared_from_env().unwrap().unwrap();
    assert!(kernel.uses_hardware_acceleration());
    let equation = |k: u64| {
        let p = G1Projective::GENERATOR;
        let q = G2Projective::GENERATOR;
        BbsPairingEqInput {
            lhs_g1: (p * Scalar::from(k)).to_affine().to_compressed().to_vec(),
            lhs_g2: q.to_affine().to_compressed().to_vec(),
            rhs_g1: p.to_affine().to_compressed().to_vec(),
            rhs_g2: (q * Scalar::from(k)).to_affine().to_compressed().to_vec(),
        }
    };
    let common = vec![equation(2), equation(2)];
    let unique = vec![equation(3), equation(4)];
    // Reserve the full-size TLS buffers first so the test does not rely on
    // reallocation. Common A -> unique B -> common A must upload A again.
    for inputs in [&unique, &common, &unique, &common] {
        let result = kernel.pairing_eq_batch(inputs, false);
        assert_eq!(result.failed, 0, "{result:?}");
        assert_eq!(result.results.len(), 2);
        assert!(result.results.iter().all(|r| r.result == Ok(true)));
    }
    let mut invalid = common.clone();
    invalid[1].rhs_g2 = equation(5).rhs_g2;
    let result = kernel.pairing_eq_batch(&invalid, false);
    assert_eq!(result.results[0].result, Ok(true));
    assert_eq!(result.results[1].result, Ok(false));
}

#[test]
fn cpu_abi_smoke_library_prepare_b_matches_cpu() {
    if std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_none() {
        return;
    }

    let gpu = GpuBbsEngine::new();
    let Some(kernel) = &gpu.kernel else {
        return;
    };
    if kernel.backend_kind != gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CPU_ABI_TEST {
        return;
    }

    let cpu = CuBbsProtocolEngine::new();
    let keys = cpu.generate_keypair(&mut rand::thread_rng()).unwrap();
    let req = SignRequest {
        secret_key: keys.secret_key,
        public_key: keys.public_key,
        header: header(),
        messages: messages(),
    };
    let scalars = cpu_message_scalars(&req.messages).unwrap();
    let gpu_prepared = kernel.prepare_b_batch(std::slice::from_ref(&req), &[scalars], true);
    assert_eq!(gpu_prepared.succeeded, 1);
    assert_eq!(gpu_prepared.failed, 0);

    let cpu_prepared = cpu.prepare_b_for_sign(&req).unwrap();
    assert_eq!(
        gpu_prepared.results[0].result.as_ref().unwrap(),
        &cpu_prepared.compressed_b
    );

    let engine = BbsEngine::GpuRequestedCpuFallback(gpu);
    let signed = engine.sign_batch(vec![req.clone()], true);
    assert_eq!(signed.succeeded, 1);
    assert!(engine
        .verify(VerifyRequest {
            public_key: req.public_key,
            signature: signed.results.into_iter().next().unwrap().result.unwrap(),
            header: req.header,
            messages: req.messages,
        })
        .unwrap());
}

#[test]
fn cuda_abi_does_not_advertise_legacy_prepare_b_stub() {
    if std::env::var_os("CUBBS_BBS_GPU_LIBRARY").is_none() {
        return;
    }

    let gpu = GpuBbsEngine::new();
    let Some(kernel) = &gpu.kernel else {
        return;
    };
    if kernel.backend_kind != gpu_abi::CUBBS_BBS_GPU_BACKEND_KIND_CUDA {
        return;
    }

    assert!(kernel.prepare_b_batch.is_some());
    assert!(!kernel.can_prepare_b());
    assert!(kernel.can_g1_msm());
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0);
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
        .collect()
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
