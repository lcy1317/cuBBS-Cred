//! Protocol lowering through the attributed ZKryptium derivative.

use super::*;

impl CuBbsProtocolEngine {
    pub const fn new() -> Self {
        Self
    }

    pub fn generate_keypair(&self, rng: &mut impl RngCore) -> Result<KeyPair> {
        let mut key_material = [0u8; Bls12381Sha256::IKM_LEN];
        rng.fill_bytes(&mut key_material);
        let pair = ZkKeyPair::<BBSplus<Bls12381Sha256>>::generate(&key_material, None, None)
            .map_err(|e| Error::Zkryptium(e.to_string()))?;
        let (secret_key, public_key) = pair.into_parts();

        Ok(KeyPair {
            secret_key: secret_key.to_bytes().to_vec(),
            public_key: public_key.to_bytes().to_vec(),
        })
    }

    pub fn public_key_from_secret(&self, secret_key: &[u8]) -> Result<Vec<u8>> {
        let secret_key = parse_secret_key(secret_key)?;
        Ok(secret_key.public_key().to_bytes().to_vec())
    }

    pub fn sign(&self, req: SignRequest) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        let secret_key = parse_secret_key(&req.secret_key)?;
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::sign(
            Some(&req.messages),
            &secret_key,
            &public_key,
            Some(&req.header),
        )
        .map(|sig| sig.to_bytes().to_vec())
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub fn profile_sign(&self, req: SignRequest) -> Result<BbsSignProfile> {
        ensure_header(&req.header)?;
        let secret_key = parse_secret_key(&req.secret_key)?;
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::profile_sign(
            Some(&req.messages),
            &secret_key,
            &public_key,
            Some(&req.header),
        )
        .map(|profile| BbsSignProfile {
            message_scalar_ms: profile.message_scalar_ms,
            generator_ms: profile.generator_ms,
            domain_ms: profile.domain_ms,
            e_scalar_ms: profile.e_scalar_ms,
            msm_and_finalize_ms: profile.msm_and_finalize_ms,
            total_ms: profile.total_ms,
            message_count: profile.message_count,
            signature_bytes: profile.signature_bytes,
        })
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn sign_prehashed_message_scalars(
        &self,
        req: SignRequest,
        message_scalars: Vec<BBSplusMessage>,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let secret_key = parse_secret_key(&req.secret_key)?;
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::sign_prehashed_message_scalars(
            &message_scalars,
            &secret_key,
            &public_key,
            Some(&req.header),
        )
        .map(|sig| sig.to_bytes().to_vec())
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn signature_e_input_for_prehashed_message_scalars(
        &self,
        req: &SignRequest,
        message_scalars: &[BBSplusMessage],
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let secret_key = parse_secret_key(&req.secret_key)?;
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::signature_e_input_for_prehashed_message_scalars(
            message_scalars,
            &secret_key,
            &public_key,
            Some(&req.header),
        )
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn sign_prehashed_message_scalars_and_e(
        &self,
        req: SignRequest,
        message_scalars: Vec<BBSplusMessage>,
        e: BBSplusMessage,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let secret_key = parse_secret_key(&req.secret_key)?;
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::sign_prehashed_message_scalars_and_e(
            &message_scalars,
            &e,
            &secret_key,
            &public_key,
            Some(&req.header),
        )
        .map(|sig| sig.to_bytes().to_vec())
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub fn prepare_b_for_sign(&self, req: &SignRequest) -> Result<BbsPreparedB> {
        ensure_header(&req.header)?;
        let message_scalars = cpu_message_scalars(&req.messages)?;
        self.prepare_b_for_prehashed_message_scalars(req, &message_scalars)
    }

    pub fn prepare_b_msm_terms_for_sign(&self, req: &SignRequest) -> Result<BbsMsmTerms> {
        ensure_header(&req.header)?;
        let message_scalars = cpu_message_scalars(&req.messages)?;
        self.prepare_b_msm_terms_for_prehashed_message_scalars(req, &message_scalars)
    }

    pub fn messages_to_scalar_bytes(&self, messages: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        cpu_message_scalars(messages).map(|scalars| {
            scalars
                .iter()
                .map(|scalar| scalar.to_bytes_be().to_vec())
                .collect()
        })
    }

    pub fn prepare_b_for_message_scalars(
        &self,
        public_key: &[u8],
        header: &[u8],
        message_scalars: &[Vec<u8>],
    ) -> Result<BbsPreparedB> {
        ensure_header(header)?;
        let message_scalars = message_scalars
            .iter()
            .map(|scalar| parse_message_scalar_bytes(scalar))
            .collect::<Result<Vec<_>>>()?;
        let req = SignRequest {
            secret_key: Vec::new(),
            public_key: public_key.to_vec(),
            header: header.to_vec(),
            messages: vec![Vec::new(); message_scalars.len()],
        };
        self.prepare_b_for_prehashed_message_scalars(&req, &message_scalars)
    }

    pub fn prepare_b_msm_terms_for_message_scalars(
        &self,
        public_key: &[u8],
        header: &[u8],
        message_scalars: &[Vec<u8>],
    ) -> Result<BbsMsmTerms> {
        ensure_header(header)?;
        let message_scalars = message_scalars
            .iter()
            .map(|scalar| parse_message_scalar_bytes(scalar))
            .collect::<Result<Vec<_>>>()?;
        let req = SignRequest {
            secret_key: Vec::new(),
            public_key: public_key.to_vec(),
            header: header.to_vec(),
            messages: vec![Vec::new(); message_scalars.len()],
        };
        self.prepare_b_msm_terms_for_prehashed_message_scalars(&req, &message_scalars)
    }

    pub(super) fn prepare_b_for_prehashed_message_scalars(
        &self,
        req: &SignRequest,
        message_scalars: &[BBSplusMessage],
    ) -> Result<BbsPreparedB> {
        ensure_header(&req.header)?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::prepare_b_for_prehashed_message_scalars(
            message_scalars,
            &public_key,
            Some(&req.header),
        )
        .map(|prepared| BbsPreparedB {
            compressed_b: prepared.compressed_b,
        })
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn prepare_b_msm_terms_for_prehashed_message_scalars(
        &self,
        req: &SignRequest,
        message_scalars: &[BBSplusMessage],
    ) -> Result<BbsMsmTerms> {
        ensure_header(&req.header)?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let public_key = parse_public_key(&req.public_key)?;

        ZkSignature::<BBSplus<Bls12381Sha256>>::prepare_b_msm_terms_for_prehashed_message_scalars(
            message_scalars,
            &public_key,
            Some(&req.header),
        )
        .map(
            |BBSplusPrepareBMsmTerms {
                 compressed_points,
                 scalar_be,
             }| BbsMsmTerms {
                compressed_points: compressed_points.into(),
                scalar_be,
            },
        )
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub fn sign_with_prepared_b(
        &self,
        req: SignRequest,
        prepared_b: BbsPreparedB,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        let message_scalars = cpu_message_scalars(&req.messages)?;
        let e_input =
            self.signature_e_input_for_prehashed_message_scalars(&req, &message_scalars)?;
        let e = cpu_signature_e_scalar(&e_input)?;
        self.sign_prehashed_message_scalars_e_and_b(req, e, prepared_b)
    }

    pub(super) fn sign_prehashed_message_scalars_e_and_b(
        &self,
        req: SignRequest,
        e: BBSplusMessage,
        prepared_b: BbsPreparedB,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        let secret_key = parse_secret_key(&req.secret_key)?;
        let prepared_b = BBSplusPreparedB {
            compressed_b: prepared_b.compressed_b,
        };

        ZkSignature::<BBSplus<Bls12381Sha256>>::sign_prehashed_message_scalars_e_and_b(
            &e,
            &prepared_b,
            &secret_key,
        )
        .map(|sig| sig.to_bytes().to_vec())
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn sign_prehashed_e_and_b_with_secret(
        &self,
        secret_key: &BBSplusSecretKey,
        e: BBSplusMessage,
        prepared_b: BbsPreparedB,
    ) -> Result<Vec<u8>> {
        let prepared_b = BBSplusPreparedB {
            compressed_b: prepared_b.compressed_b,
        };

        ZkSignature::<BBSplus<Bls12381Sha256>>::sign_prehashed_message_scalars_e_and_b(
            &e,
            &prepared_b,
            secret_key,
        )
        .map(|sig| sig.to_bytes().to_vec())
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub fn verify(&self, req: VerifyRequest) -> Result<bool> {
        ensure_header(&req.header)?;
        let public_key = parse_public_key(&req.public_key)?;
        let signature = parse_signature(&req.signature)?;

        Ok(signature
            .verify(&public_key, Some(&req.messages), Some(&req.header))
            .is_ok())
    }

    pub(super) fn verify_prehashed_message_scalars(
        &self,
        req: &VerifyRequest,
        message_scalars: &[BBSplusMessage],
    ) -> Result<bool> {
        ensure_header(&req.header)?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let public_key = parse_public_key(&req.public_key)?;
        let signature = parse_signature(&req.signature)?;

        Ok(signature
            .verify_prehashed_message_scalars(&public_key, message_scalars, Some(&req.header))
            .is_ok())
    }

    pub(super) fn verify_prehashed_b(
        &self,
        req: VerifyRequest,
        prepared_b: BbsPreparedB,
    ) -> Result<bool> {
        ensure_header(&req.header)?;
        let public_key = parse_public_key(&req.public_key)?;
        let signature = parse_signature(&req.signature)?;
        let prepared_b = BBSplusPreparedB {
            compressed_b: prepared_b.compressed_b,
        };

        Ok(signature
            .verify_prehashed_b(&public_key, &prepared_b)
            .is_ok())
    }

    pub(super) fn verify_prehashed_b_batch_aggregate(
        &self,
        items: &[(VerifyRequest, BbsPreparedB)],
    ) -> Result<bool> {
        let mut signatures = Vec::with_capacity(items.len());
        let mut public_keys = Vec::with_capacity(items.len());
        let mut prepared_b = Vec::with_capacity(items.len());

        for (req, item_b) in items {
            ensure_header(&req.header)?;
            signatures.push(parse_signature(&req.signature)?);
            public_keys.push(parse_public_key(&req.public_key)?);
            prepared_b.push(BBSplusPreparedB {
                compressed_b: item_b.compressed_b.clone(),
            });
        }

        let refs = signatures
            .iter()
            .zip(public_keys.iter())
            .zip(prepared_b.iter())
            .map(|((signature, public_key), item_b)| (signature, public_key, item_b))
            .collect::<Vec<_>>();

        Ok(ZkSignature::<BBSplus<Bls12381Sha256>>::verify_prehashed_b_batch(&refs).is_ok())
    }

    pub(super) fn verify_prehashed_b_pairing_eq_input(
        &self,
        req: &VerifyRequest,
        prepared_b: &BbsPreparedB,
    ) -> Result<BbsPairingEqInput> {
        let (lhs_g1, lhs_g2, rhs) = self.verify_prehashed_b_pairing_projective(req, prepared_b)?;

        Ok(BbsPairingEqInput {
            lhs_g1,
            lhs_g2,
            rhs_g1: rhs.to_affine().to_compressed().to_vec(),
            rhs_g2: g2_generator_compressed(),
        })
    }

    pub(super) fn verify_prehashed_b_pairing_projective(
        &self,
        req: &VerifyRequest,
        prepared_b: &BbsPreparedB,
    ) -> Result<(Vec<u8>, Vec<u8>, G1Projective)> {
        ensure_header(&req.header)?;
        let signature = parse_signature(&req.signature)?;
        let b = parse_compressed_g1(&prepared_b.compressed_b)?;
        Ok((
            req.signature[..48].to_vec(),
            req.public_key.clone(),
            b - signature.a() * signature.e(),
        ))
    }

    pub fn proof_gen(&self, req: ProofGenRequest) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.messages.len())?;
        let public_key = parse_public_key(&req.public_key)?;

        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen(
            &public_key,
            &req.signature,
            Some(&req.header),
            Some(&req.presentation_header),
            Some(&req.messages),
            Some(&req.disclosed_indexes),
        )
        .map(|proof| proof.to_bytes())
        .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_prehashed_message_scalars(
        &self,
        req: ProofGenRequest,
        message_scalars: Vec<BBSplusMessage>,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.messages.len())?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let public_key = parse_public_key(&req.public_key)?;

        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_prehashed_message_scalars(
            &public_key,
            &req.signature,
            Some(&req.header),
            Some(&req.presentation_header),
            &message_scalars,
            Some(&req.disclosed_indexes),
        )
        .map(|proof| proof.to_bytes())
        .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_prehashed_message_scalars_and_b(
        &self,
        req: ProofGenRequest,
        message_scalars: Vec<BBSplusMessage>,
        compressed_b: Vec<u8>,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.messages.len())?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let public_key = parse_public_key(&req.public_key)?;

        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_prehashed_message_scalars_and_b(
            &public_key,
            &req.signature,
            Some(&req.header),
            Some(&req.presentation_header),
            &message_scalars,
            Some(&req.disclosed_indexes),
            &compressed_b,
        )
        .map(|proof| proof.to_bytes())
        .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_t2_msm_terms_for_prehashed_message_scalars_and_b(
        &self,
        req: &ProofGenRequest,
        message_scalars: &[BBSplusMessage],
        compressed_b: &[u8],
        precomputed_domain: Option<Scalar>,
    ) -> Result<(BBSplusProofGenT2State, BbsMsmTerms)> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.messages.len())?;
        if message_scalars.len() != req.messages.len() {
            return Err(Error::Gpu(format!(
                "GPU message scalar count mismatch: expected {}, got {}",
                req.messages.len(),
                message_scalars.len()
            )));
        }
        let public_key = parse_public_key(&req.public_key)?;

        let (state, terms) = if let Some(domain) = precomputed_domain {
            PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_prehashed_message_scalars_and_b_t2_msm_terms_with_domain(
                &public_key,
                &req.signature,
                Some(&req.header),
                message_scalars,
                Some(&req.disclosed_indexes),
                compressed_b,
                domain,
            )
        } else {
            PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_prehashed_message_scalars_and_b_t2_msm_terms(
                &public_key,
                &req.signature,
                Some(&req.header),
                message_scalars,
                Some(&req.disclosed_indexes),
                compressed_b,
            )
        }
        .map_err(|e| Error::Zkryptium(e.to_string()))?;

        Ok((state, proof_gen_t2_terms_from_zkryptium(terms)))
    }

    /// Build T2 state + terms from precomputed points (GPU path). The 6 fixed
    /// G1 muls were done on the device; this only assembles the state and
    /// builds the H_point MSM terms.
    pub(super) fn proof_gen_t2_from_precomputed_points(
        &self,
        req: &ProofGenRequest,
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: &[usize],
        random_scalars: Vec<Scalar>,
        d: bls12_381_plus::G1Projective,
        abar: bls12_381_plus::G1Projective,
        bbar: bls12_381_plus::G1Projective,
        t1: bls12_381_plus::G1Projective,
        domain: Scalar,
    ) -> Result<(BBSplusProofGenT2State, BbsMsmTerms)> {
        ensure_header(&req.header)?;
        let (state, terms) =
            PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_t2_state_from_precomputed_points(
                &req.signature,
                message_scalars,
                Some(disclosed_indexes),
                random_scalars,
                d,
                abar,
                bbar,
                t1,
                domain,
            )
            .map_err(|e| Error::Zkryptium(e.to_string()))?;
        Ok((state, proof_gen_t2_terms_from_zkryptium(terms)))
    }

    pub(super) fn proof_gen_t2_from_precomputed_compressed_points(
        &self,
        req: &ProofGenRequest,
        message_scalars: &[BBSplusMessage],
        random_scalars: Vec<Scalar>,
        d: Vec<u8>,
        abar: Vec<u8>,
        bbar: Vec<u8>,
        t1: Vec<u8>,
        domain: Scalar,
    ) -> Result<(BBSplusProofGenT2CompressedState, BbsMsmTerms)> {
        ensure_header(&req.header)?;
        let (state, terms) = PoKSignature::<BBSplus<Bls12381Sha256>>::
            proof_gen_t2_compressed_state_from_precomputed_points(
                &req.signature,
                message_scalars,
                Some(&req.disclosed_indexes),
                random_scalars,
                d,
                abar,
                bbar,
                t1,
                domain,
            )
            .map_err(|e| Error::Zkryptium(e.to_string()))?;
        Ok((state, proof_gen_t2_terms_from_zkryptium(terms)))
    }

    pub(super) fn proof_gen_state_from_precomputed_compressed_points(
        &self,
        req: &ProofGenRequest,
        message_scalars: &[BBSplusMessage],
        random_scalars: Vec<Scalar>,
        d: Vec<u8>,
        abar: Vec<u8>,
        bbar: Vec<u8>,
        t1: Vec<u8>,
        domain: Scalar,
    ) -> Result<BBSplusProofGenT2CompressedState> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_compressed_state_from_precomputed_points(
            &req.signature,
            message_scalars,
            Some(&req.disclosed_indexes),
            random_scalars,
            d,
            abar,
            bbar,
            t1,
            domain,
        )
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn proof_gen_finalize_compressed_state_and_t2(
        &self,
        req: ProofGenRequest,
        state: BBSplusProofGenT2CompressedState,
        t2_compressed: Vec<u8>,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_finalize_compressed_state_and_t2(
            state,
            Some(&req.presentation_header),
            &t2_compressed,
        )
        .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_finalize_compressed_state_and_t2_with_r3(
        &self,
        req: ProofGenRequest,
        state: BBSplusProofGenT2CompressedState,
        t2_compressed: Vec<u8>,
        r3: Scalar,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_finalize_compressed_state_and_t2_with_r3(
            state,
            Some(&req.presentation_header),
            &t2_compressed,
            r3,
        )
        .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_challenge_input_compressed_state_and_t2(
        &self,
        req: &ProofGenRequest,
        state: &BBSplusProofGenT2CompressedState,
        t2_compressed: &[u8],
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_challenge_input_compressed_state_and_t2(
            state,
            Some(&req.presentation_header),
            t2_compressed,
        )
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn proof_gen_finalize_compressed_state_with_challenge(
        &self,
        req: ProofGenRequest,
        state: BBSplusProofGenT2CompressedState,
        challenge: BBSplusMessage,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_finalize_compressed_state_with_challenge(
            state,
            challenge.value,
        )
        .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_finalize_compressed_state_with_challenge_and_r3(
        &self,
        req: ProofGenRequest,
        state: BBSplusProofGenT2CompressedState,
        challenge: BBSplusMessage,
        r3: Scalar,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::
            proof_gen_finalize_compressed_state_with_challenge_and_r3(
                state,
                challenge.value,
                r3,
            )
            .map_err(|_| Error::InvalidProof)
    }

    pub(super) fn proof_gen_t2_terms_from_compressed_b(
        &self,
        message_count: usize,
        disclosed_indexes: &[usize],
        random_scalars: &[Scalar],
        compressed_b: &[u8],
    ) -> Result<BbsMsmTerms> {
        let terms =
            PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_t2_msm_terms_from_compressed_b(
                message_count,
                Some(disclosed_indexes),
                random_scalars,
                compressed_b,
            )
            .map_err(|e| Error::Zkryptium(e.to_string()))?;
        Ok(proof_gen_t2_terms_from_zkryptium(terms))
    }

    pub(super) fn proof_gen_finalize_prehashed_b_and_t2(
        &self,
        req: ProofGenRequest,
        state: BBSplusProofGenT2State,
        t2_compressed: Vec<u8>,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_finalize_prehashed_b_and_t2(
            state,
            Some(&req.presentation_header),
            &t2_compressed,
        )
        .map_err(|_| Error::InvalidProof)?;
        Ok(proof.to_bytes())
    }

    pub(super) fn proof_gen_challenge_input_prehashed_b_and_t2(
        &self,
        req: &ProofGenRequest,
        state: &BBSplusProofGenT2State,
        t2_compressed: &[u8],
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_challenge_input_prehashed_b_and_t2(
            state,
            Some(&req.presentation_header),
            t2_compressed,
        )
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn proof_gen_finalize_prehashed_b_and_t2_with_challenge(
        &self,
        req: ProofGenRequest,
        state: BBSplusProofGenT2State,
        t2_compressed: Vec<u8>,
        challenge: BBSplusMessage,
    ) -> Result<Vec<u8>> {
        ensure_header(&req.header)?;
        let proof =
            PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen_finalize_prehashed_b_and_t2_with_challenge(
                state,
                &t2_compressed,
                challenge.value,
            )
            .map_err(|_| Error::InvalidProof)?;
        Ok(proof.to_bytes())
    }

    pub fn proof_verify(&self, req: ProofVerifyRequest) -> Result<bool> {
        ensure_header(&req.header)?;
        if ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count).is_err()
            || req.disclosed_messages.len() != req.disclosed_indexes.len()
        {
            return Ok(false);
        }

        let public_key = parse_public_key(&req.public_key)?;
        let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&req.proof)
            .map_err(|_| Error::InvalidProof)?;

        Ok(proof
            .proof_verify(
                &public_key,
                Some(&req.disclosed_messages),
                Some(&req.disclosed_indexes),
                Some(&req.header),
                Some(&req.presentation_header),
            )
            .is_ok())
    }

    pub(super) fn proof_verify_prehashed_disclosed_scalars(
        &self,
        req: ProofVerifyRequest,
        disclosed_message_scalars: Vec<BBSplusMessage>,
    ) -> Result<bool> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count)?;
        if disclosed_message_scalars.len() != req.disclosed_indexes.len() {
            return Err(Error::Gpu(format!(
                "GPU disclosed scalar count mismatch: expected {}, got {}",
                req.disclosed_indexes.len(),
                disclosed_message_scalars.len()
            )));
        }

        let public_key = parse_public_key(&req.public_key)?;
        let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&req.proof)
            .map_err(|_| Error::InvalidProof)?;

        Ok(proof
            .proof_verify_prehashed_disclosed_scalars(
                &public_key,
                &disclosed_message_scalars,
                Some(&req.disclosed_indexes),
                Some(&req.header),
                Some(&req.presentation_header),
            )
            .is_ok())
    }

    pub(super) fn proof_verify_msm_terms_for_prehashed_disclosed_scalars(
        &self,
        req: &ProofVerifyRequest,
        disclosed_message_scalars: &[BBSplusMessage],
    ) -> Result<(BbsMsmTerms, BbsMsmTerms, Scalar)> {
        self.proof_verify_prepared_msm_for_prehashed_disclosed_scalars(
            req,
            disclosed_message_scalars,
        )
        .map(|(t1, t2, domain, _)| (t1, t2, domain))
    }

    pub(super) fn proof_verify_prepared_msm_for_prehashed_disclosed_scalars(
        &self,
        req: &ProofVerifyRequest,
        disclosed_message_scalars: &[BBSplusMessage],
    ) -> Result<(BbsMsmTerms, BbsMsmTerms, Scalar, ProofVerifyPreparedMeta)> {
        self.proof_verify_prepared_msm_for_prehashed_disclosed_scalars_with_domain(
            req,
            disclosed_message_scalars,
            None,
        )
    }

    pub(super) fn proof_verify_prepared_msm_for_prehashed_disclosed_scalars_with_domain(
        &self,
        req: &ProofVerifyRequest,
        disclosed_message_scalars: &[BBSplusMessage],
        precomputed_domain: Option<Scalar>,
    ) -> Result<(BbsMsmTerms, BbsMsmTerms, Scalar, ProofVerifyPreparedMeta)> {
        let public_key = parse_public_key(&req.public_key)?;
        self.proof_verify_prepared_msm_with_public_key_and_domain(
            req,
            disclosed_message_scalars,
            &public_key,
            precomputed_domain,
        )
    }

    pub(super) fn proof_verify_prepared_msm_with_public_key_and_domain(
        &self,
        req: &ProofVerifyRequest,
        disclosed_message_scalars: &[BBSplusMessage],
        public_key: &BBSplusPublicKey,
        precomputed_domain: Option<Scalar>,
    ) -> Result<(BbsMsmTerms, BbsMsmTerms, Scalar, ProofVerifyPreparedMeta)> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count)?;
        if disclosed_message_scalars.len() != req.disclosed_indexes.len() {
            return Err(Error::Gpu(format!(
                "GPU disclosed scalar count mismatch: expected {}, got {}",
                req.disclosed_indexes.len(),
                disclosed_message_scalars.len()
            )));
        }

        let terms = match precomputed_domain {
            Some(domain) => PoKSignature::<BBSplus<Bls12381Sha256>>::
                proof_verify_msm_terms_from_proof_bytes_with_domain(
                &req.proof,
                disclosed_message_scalars,
                Some(&req.disclosed_indexes),
                domain,
            ),
            None => {
                let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&req.proof)
                    .map_err(|_| Error::InvalidProof)?;
                proof.proof_verify_prehashed_disclosed_scalars_msm_terms_with_domain(
                    public_key,
                    disclosed_message_scalars,
                    Some(&req.disclosed_indexes),
                    Some(&req.header),
                    None,
                )
            }
        }
        .map_err(|e| Error::Zkryptium(e.to_string()))?;

        let challenge_be = req
            .proof
            .get(req.proof.len().saturating_sub(32)..)
            .ok_or(Error::InvalidProof)?
            .to_vec();
        let meta = ProofVerifyPreparedMeta {
            pairing_input: BbsPairingEqInput {
                lhs_g1: req.proof[0..48].to_vec(),
                lhs_g2: req.public_key.clone(),
                rhs_g1: req.proof[48..96].to_vec(),
                rhs_g2: g2_generator_compressed(),
            },
            challenge_be,
        };
        let (t1, t2, domain) = proof_verify_terms_from_zkryptium(terms);
        Ok((t1, t2, domain, meta))
    }

    pub(super) fn proof_verify_domain_for_request(
        &self,
        req: &ProofVerifyRequest,
    ) -> Result<Scalar> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count)?;
        let public_key = parse_public_key(&req.public_key)?;
        PoKSignature::<BBSplus<Bls12381Sha256>>::proof_verify_domain_for_total_message_count(
            &public_key,
            req.total_message_count,
            Some(&req.disclosed_indexes),
            Some(&req.header),
        )
        .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn proof_verify_prehashed_disclosed_scalars_and_msm(
        &self,
        req: ProofVerifyRequest,
        disclosed_message_scalars: Vec<BBSplusMessage>,
        t1_compressed: Vec<u8>,
        t2_compressed: Vec<u8>,
    ) -> Result<bool> {
        ensure_header(&req.header)?;
        ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count)?;
        if disclosed_message_scalars.len() != req.disclosed_indexes.len() {
            return Err(Error::Gpu(format!(
                "GPU disclosed scalar count mismatch: expected {}, got {}",
                req.disclosed_indexes.len(),
                disclosed_message_scalars.len()
            )));
        }

        let public_key = parse_public_key(&req.public_key)?;
        let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&req.proof)
            .map_err(|_| Error::InvalidProof)?;

        Ok(proof
            .proof_verify_prehashed_disclosed_scalars_and_msm(
                &public_key,
                &disclosed_message_scalars,
                Some(&req.disclosed_indexes),
                Some(&req.header),
                Some(&req.presentation_header),
                &t1_compressed,
                &t2_compressed,
            )
            .is_ok())
    }

    pub(super) fn proof_verify_prehashed_disclosed_scalars_and_msm_challenge_scalar(
        &self,
        req: &ProofVerifyRequest,
        challenge: BBSplusMessage,
    ) -> Result<()> {
        let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&req.proof)
            .map_err(|_| Error::InvalidProof)?;
        proof
            .proof_verify_prehashed_disclosed_scalars_and_msm_challenge_scalar(challenge.value)
            .map_err(|e| Error::Zkryptium(e.to_string()))
    }

    pub(super) fn proof_verify_prehashed_disclosed_scalars_and_msm_batch_aggregate(
        &self,
        items: &[(
            ProofVerifyRequest,
            Vec<BBSplusMessage>,
            Vec<u8>,
            Vec<u8>,
            Scalar,
        )],
    ) -> Result<bool> {
        let mut public_keys = Vec::with_capacity(items.len());
        let mut proofs = Vec::with_capacity(items.len());

        for (req, scalars, _, _, _) in items {
            ensure_header(&req.header)?;
            ensure_disclosed_indexes(&req.disclosed_indexes, req.total_message_count)?;
            if scalars.len() != req.disclosed_indexes.len() {
                return Err(Error::Gpu(format!(
                    "GPU disclosed scalar count mismatch: expected {}, got {}",
                    req.disclosed_indexes.len(),
                    scalars.len()
                )));
            }
            public_keys.push(parse_public_key(&req.public_key)?);
            proofs.push(
                PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&req.proof)
                    .map_err(|_| Error::InvalidProof)?,
            );
        }

        let batch_items = items
            .iter()
            .zip(proofs.iter())
            .zip(public_keys.iter())
            .map(|(((req, scalars, t1, t2, domain), proof), public_key)| {
                BBSplusProofVerifyPrecomputedItem {
                    proof,
                    pk: public_key,
                    disclosed_message_scalars: scalars,
                    disclosed_indexes: Some(&req.disclosed_indexes),
                    header: Some(&req.header),
                    ph: Some(&req.presentation_header),
                    t1_compressed: t1,
                    t2_compressed: t2,
                    precomputed_domain: Some(*domain),
                }
            })
            .collect::<Vec<_>>();

        Ok(PoKSignature::<BBSplus<Bls12381Sha256>>::proof_verify_prehashed_disclosed_scalars_and_msm_batch(&batch_items).is_ok())
    }

    pub fn sign_batch(
        &self,
        reqs: Vec<SignRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        run_bbs_batch(reqs, fail_fast, |req| self.sign(req))
    }

    pub fn verify_batch(
        &self,
        reqs: Vec<VerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        run_bbs_batch(reqs, fail_fast, |req| self.verify(req))
    }

    pub fn proof_gen_batch(
        &self,
        reqs: Vec<ProofGenRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<Vec<u8>> {
        run_bbs_batch(reqs, fail_fast, |req| self.proof_gen(req))
    }

    pub fn proof_verify_batch(
        &self,
        reqs: Vec<ProofVerifyRequest>,
        fail_fast: bool,
    ) -> BbsBatchRunResult<bool> {
        run_bbs_batch(reqs, fail_fast, |req| self.proof_verify(req))
    }
}
