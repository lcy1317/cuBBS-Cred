use crate::four_config::FourOpConfig;
use cubbs_bbs::{
    reference::{CpuReferenceBackend, ZkryptiumReference},
    BbsEngine, ProofGenRequest, ProofVerifyRequest, SignRequest, VerifyRequest,
};
use rand::thread_rng;

#[derive(Clone)]
pub struct FourOpTemplates {
    pub sign: Option<Vec<SignRequest>>,
    pub verify: Option<Vec<VerifyRequest>>,
    pub proof_gen: Option<Vec<ProofGenRequest>>,
    pub proof_verify: Option<Vec<ProofVerifyRequest>>,
}

pub fn prepare(engine: &BbsEngine, config: &FourOpConfig) -> Result<FourOpTemplates, String> {
    let issuer_count = if config.unique_issuer_per_item {
        config.batch_size
    } else {
        1
    };
    let keypairs = (0..issuer_count)
        .map(|_| {
            engine
                .generate_keypair(&mut thread_rng())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let presentation_header = b"omniid-four-op-preloaded".to_vec();
    let mut sign = Vec::with_capacity(config.batch_size);
    for item in 0..config.batch_size {
        let message_count = config.message_counts[item % config.message_counts.len()];
        let keypair = &keypairs[if config.unique_issuer_per_item {
            item
        } else {
            0
        }];
        sign.push(SignRequest {
            secret_key: keypair.secret_key.clone(),
            public_key: keypair.public_key.clone(),
            header: deterministic_header(message_count),
            messages: deterministic_messages(message_count, item),
        });
    }

    if config.runs("sign")
        && !config.runs("verify-signature")
        && !config.runs("proof-gen")
        && !config.runs("proof-verify")
    {
        return Ok(FourOpTemplates {
            sign: Some(sign),
            verify: None,
            proof_gen: None,
            proof_verify: None,
        });
    }

    let signatures = engine
        .sign_batch(sign.clone(), true)
        .results
        .into_iter()
        .map(|item| item.result.map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if signatures.len() != config.batch_size {
        return Err("signature template batch was incomplete".to_owned());
    }

    let verify = config.runs("verify-signature").then(|| {
        sign.iter()
            .zip(signatures.iter())
            .map(|(request, signature)| VerifyRequest {
                public_key: request.public_key.clone(),
                signature: signature.clone(),
                header: request.header.clone(),
                messages: request.messages.clone(),
            })
            .collect::<Vec<_>>()
    });
    let needs_proof = config.runs("proof-gen") || config.runs("proof-verify");
    let proof_gen = needs_proof.then(|| {
        sign.iter()
            .zip(signatures.iter())
            .map(|(request, signature)| ProofGenRequest {
                public_key: request.public_key.clone(),
                signature: signature.clone(),
                header: request.header.clone(),
                presentation_header: presentation_header.clone(),
                messages: request.messages.clone(),
                disclosed_indexes: disclosed_indexes(
                    request.messages.len(),
                    config.disclosed_ratio,
                ),
            })
            .collect::<Vec<_>>()
    });
    let proof_verify = if config.runs("proof-verify") {
        let proofs = engine
            .proof_gen_batch(
                proof_gen
                    .as_ref()
                    .expect("proof requests were prepared")
                    .clone(),
                true,
            )
            .results
            .into_iter()
            .map(|item| item.result.map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if proofs.len() != config.batch_size {
            return Err("proof template batch was incomplete".to_owned());
        }
        Some(
            sign.iter()
                .zip(proofs)
                .map(|(request, proof)| {
                    let disclosed_indexes =
                        disclosed_indexes(request.messages.len(), config.disclosed_ratio);
                    let disclosed_messages = disclosed_indexes
                        .iter()
                        .map(|&index| request.messages[index].clone())
                        .collect();
                    ProofVerifyRequest {
                        public_key: request.public_key.clone(),
                        proof,
                        header: request.header.clone(),
                        presentation_header: presentation_header.clone(),
                        disclosed_messages,
                        disclosed_indexes,
                        total_message_count: request.messages.len(),
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    Ok(FourOpTemplates {
        sign: config.runs("sign").then_some(sign),
        verify,
        proof_gen: if config.runs("proof-gen") {
            Some(proof_gen.expect("proof requests were prepared for proof generation"))
        } else {
            None
        },
        proof_verify,
    })
}

pub fn prepare_reference(
    reference: &ZkryptiumReference,
    config: &FourOpConfig,
) -> Result<FourOpTemplates, String> {
    let issuer_count = if config.unique_issuer_per_item {
        config.batch_size
    } else {
        1
    };
    let keypairs = (0..issuer_count)
        .map(|_| {
            reference
                .generate_keypair()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let presentation_header = b"omniid-four-op-reference".to_vec();
    let mut sign = Vec::with_capacity(config.batch_size);
    for item in 0..config.batch_size {
        let message_count = config.message_counts[item % config.message_counts.len()];
        let keypair = &keypairs[if config.unique_issuer_per_item {
            item
        } else {
            0
        }];
        sign.push(SignRequest {
            secret_key: keypair.secret_key.clone(),
            public_key: keypair.public_key.clone(),
            header: deterministic_header(message_count),
            messages: deterministic_messages(message_count, item),
        });
    }
    if config.runs("sign")
        && !config.runs("verify-signature")
        && !config.runs("proof-gen")
        && !config.runs("proof-verify")
    {
        return Ok(FourOpTemplates {
            sign: Some(sign),
            verify: None,
            proof_gen: None,
            proof_verify: None,
        });
    }
    let signatures = sign
        .iter()
        .map(|request| reference.sign(request).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let verify = config.runs("verify-signature").then(|| {
        sign.iter()
            .zip(signatures.iter())
            .map(|(request, signature)| VerifyRequest {
                public_key: request.public_key.clone(),
                signature: signature.clone(),
                header: request.header.clone(),
                messages: request.messages.clone(),
            })
            .collect::<Vec<_>>()
    });
    let needs_proof = config.runs("proof-gen") || config.runs("proof-verify");
    let proof_gen = needs_proof.then(|| {
        sign.iter()
            .zip(signatures.iter())
            .map(|(request, signature)| ProofGenRequest {
                public_key: request.public_key.clone(),
                signature: signature.clone(),
                header: request.header.clone(),
                presentation_header: presentation_header.clone(),
                messages: request.messages.clone(),
                disclosed_indexes: disclosed_indexes(
                    request.messages.len(),
                    config.disclosed_ratio,
                ),
            })
            .collect::<Vec<_>>()
    });
    let proof_verify = if config.runs("proof-verify") {
        let proofs = proof_gen
            .as_ref()
            .expect("proof requests were prepared")
            .iter()
            .map(|request| {
                reference
                    .proof_gen(request)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some(
            sign.iter()
                .zip(proofs)
                .map(|(request, proof)| {
                    let disclosed_indexes =
                        disclosed_indexes(request.messages.len(), config.disclosed_ratio);
                    let disclosed_messages = disclosed_indexes
                        .iter()
                        .map(|&index| request.messages[index].clone())
                        .collect();
                    ProofVerifyRequest {
                        public_key: request.public_key.clone(),
                        proof,
                        header: request.header.clone(),
                        presentation_header: presentation_header.clone(),
                        disclosed_messages,
                        disclosed_indexes,
                        total_message_count: request.messages.len(),
                    }
                })
                .collect(),
        )
    } else {
        None
    };
    Ok(FourOpTemplates {
        sign: config.runs("sign").then_some(sign),
        verify,
        proof_gen: if config.runs("proof-gen") {
            Some(proof_gen.expect("proof requests were prepared for proof generation"))
        } else {
            None
        },
        proof_verify,
    })
}

fn deterministic_header(seed: usize) -> Vec<u8> {
    (0..64)
        .map(|index| (seed as u8).wrapping_mul(17).wrapping_add(index as u8))
        .collect()
}

fn deterministic_messages(count: usize, item: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|index| format!("four-op-{count}-{item:04}-{index:04}").into_bytes())
        .collect()
}

fn disclosed_indexes(message_count: usize, disclosed_ratio: f64) -> Vec<usize> {
    let disclosed_count =
        ((message_count as f64 * disclosed_ratio).round() as usize).min(message_count);
    if disclosed_count == 0 {
        return Vec::new();
    }
    (0..disclosed_count)
        .map(|position| position * message_count / disclosed_count)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{disclosed_indexes, prepare_reference};
    use crate::four_config::FourOpConfig;
    use cubbs_bbs::reference::{Zkryptium061Reference, ZkryptiumReference};

    #[test]
    fn disclosure_ratios_are_exact_and_evenly_spaced_for_table_shapes() {
        assert!(disclosed_indexes(256, 0.0).is_empty());
        assert_eq!(
            disclosed_indexes(256, 0.25),
            (0..256).step_by(4).collect::<Vec<_>>()
        );
        assert_eq!(
            disclosed_indexes(256, 0.50),
            (0..256).step_by(2).collect::<Vec<_>>()
        );
        let dense = disclosed_indexes(256, 0.75);
        assert_eq!(dense.len(), 192);
        assert_eq!(dense.first(), Some(&0));
        assert_eq!(dense.last(), Some(&254));
        assert!(dense.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(disclosed_indexes(256, 1.0), (0..256).collect::<Vec<_>>());
    }

    #[test]
    fn reference_templates_honor_issuer_mode() {
        let reference = ZkryptiumReference::V0_6_1(Zkryptium061Reference::new());
        let mut config = FourOpConfig::default();
        config.operation = "sign".to_owned();
        config.message_counts = vec![1];
        config.batch_size = 2;

        let same = prepare_reference(&reference, &config)
            .unwrap()
            .sign
            .unwrap();
        assert_eq!(same[0].public_key, same[1].public_key);

        config.unique_issuer_per_item = true;
        let unique = prepare_reference(&reference, &config)
            .unwrap()
            .sign
            .unwrap();
        assert_ne!(unique[0].public_key, unique[1].public_key);
    }
}
