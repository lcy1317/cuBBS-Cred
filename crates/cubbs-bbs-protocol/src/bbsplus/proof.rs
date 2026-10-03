// Copyright 2023 Fondazione LINKS

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by cuBBS-Cred contributors, 2026.
// See the crate README for upstream provenance and a summary of changes.

use super::{commitment::BlindFactor, keys::BBSplusPublicKey, signature::BBSplusSignature};
use crate::{
    bbsplus::{ciphersuites::BbsCiphersuite, generators::Generators},
    errors::Error,
    schemes::{algorithms::BBSplus, generics::PoKSignature},
    utils::{
        message::bbsplus_message::BBSplusMessage,
        util::{
            bbsplus_utils::{
                calculate_domain, get_disclosed_data, get_messages, hash_to_scalar, i2osp,
                parse_g1_projective, ScalarExt,
            },
            get_remaining_indexes,
        },
    },
};
use bls12_381_plus::{multi_miller_loop, G1Projective, G2Prepared, G2Projective, Scalar};
use elliptic_curve::{group::Curve, hash2curve::ExpandMsg, Group};
use ff::Field;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    thread,
};

#[cfg(not(test))]
use crate::utils::util::bbsplus_utils::calculate_random_scalars;
#[cfg(test)]
use crate::utils::util::bbsplus_utils::seeded_random_scalars;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BBSplusPoKSignature {
    Abar: G1Projective,
    Bbar: G1Projective,
    D: G1Projective,
    e_cap: Scalar,
    r1_cap: Scalar,
    r3_cap: Scalar,
    m_cap: Vec<Scalar>,
    challenge: Scalar,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BBSplusProofVerifyMsmTerms {
    pub t1_compressed_points: Vec<Vec<u8>>,
    pub t1_scalar_be: Vec<Vec<u8>>,
    pub t2_compressed_points: Vec<Vec<u8>>,
    pub t2_scalar_be: Vec<Vec<u8>>,
    pub domain: Scalar,
}

pub struct BBSplusProofVerifyPrecomputedItem<'a, CS: BbsCiphersuite> {
    pub proof: &'a PoKSignature<BBSplus<CS>>,
    pub pk: &'a BBSplusPublicKey,
    pub disclosed_message_scalars: &'a [BBSplusMessage],
    pub disclosed_indexes: Option<&'a [usize]>,
    pub header: Option<&'a [u8]>,
    pub ph: Option<&'a [u8]>,
    pub t1_compressed: &'a [u8],
    pub t2_compressed: &'a [u8],
    pub precomputed_domain: Option<Scalar>,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BBSplusProofGenT2MsmTerms {
    pub t2_compressed_points: Vec<Vec<u8>>,
    pub t2_scalar_be: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BBSplusProofGenT2State {
    pub abar: G1Projective,
    pub bbar: G1Projective,
    pub d: G1Projective,
    pub t1: G1Projective,
    pub domain: Scalar,
    pub e: Scalar,
    pub random_scalars: Vec<Scalar>,
    pub disclosed_indexes: Vec<usize>,
    pub disclosed_messages: Vec<BBSplusMessage>,
    pub undisclosed_messages: Vec<BBSplusMessage>,
}

#[derive(Clone, Debug)]
pub struct BBSplusProofGenT2CompressedState {
    pub abar: Vec<u8>,
    pub bbar: Vec<u8>,
    pub d: Vec<u8>,
    pub t1: Vec<u8>,
    pub domain: Scalar,
    pub e: Scalar,
    pub random_scalars: Vec<Scalar>,
    pub disclosed_indexes: Vec<usize>,
    pub disclosed_messages: Vec<BBSplusMessage>,
    pub undisclosed_messages: Vec<BBSplusMessage>,
}

impl BBSplusProofGenT2State {
    /// Construct a T2 state from externally precomputed points (e.g. computed
    /// on a GPU). Used by the GPU proof-gen path where the 6 fixed G1 scalar
    /// multiplications run on the device.
    pub fn from_precomputed_points(
        abar: G1Projective,
        bbar: G1Projective,
        d: G1Projective,
        t1: G1Projective,
        domain: Scalar,
        e: Scalar,
        random_scalars: Vec<Scalar>,
        disclosed_indexes: Vec<usize>,
        disclosed_messages: Vec<BBSplusMessage>,
        undisclosed_messages: Vec<BBSplusMessage>,
    ) -> Self {
        Self {
            abar,
            bbar,
            d,
            t1,
            domain,
            e,
            random_scalars,
            disclosed_indexes,
            disclosed_messages,
            undisclosed_messages,
        }
    }
}

fn parse_signature_cached(signature: &[u8]) -> Result<BBSplusSignature, Error> {
    let signature_bytes: [u8; BBSplusSignature::BYTES] =
        signature.try_into().map_err(|_| Error::InvalidSignature)?;

    static CACHE: OnceLock<Mutex<HashMap<[u8; BBSplusSignature::BYTES], BBSplusSignature>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(parsed) = cache
        .lock()
        .expect("BBS signature cache poisoned")
        .get(&signature_bytes)
        .cloned()
    {
        return Ok(parsed);
    }

    let parsed = BBSplusSignature::from_bytes(&signature_bytes)?;
    cache
        .lock()
        .expect("BBS signature cache poisoned")
        .insert(signature_bytes, parsed.clone());
    Ok(parsed)
}

impl BBSplusPoKSignature {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();

        bytes.extend_from_slice(&self.Abar.to_affine().to_compressed());
        bytes.extend_from_slice(&self.Bbar.to_affine().to_compressed());
        bytes.extend_from_slice(&self.D.to_affine().to_compressed());
        bytes.extend_from_slice(&self.e_cap.to_bytes_be());
        bytes.extend_from_slice(&self.r1_cap.to_bytes_be());
        bytes.extend_from_slice(&self.r3_cap.to_bytes_be());
        self.m_cap
            .iter()
            .for_each(|v| bytes.extend_from_slice(&v.to_bytes_be()));
        bytes.extend_from_slice(&self.challenge.to_bytes_be());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let Abar = parse_g1_projective(&bytes[0..48])
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
        let Bbar = parse_g1_projective(&bytes[48..96])
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
        let D = parse_g1_projective(&bytes[96..144])
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;

        let e_cap = Scalar::from_bytes_be(&bytes[144..176])
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
        let r1_cap = Scalar::from_bytes_be(&bytes[176..208])
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
        let r3_cap = Scalar::from_bytes_be(&bytes[208..240])
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;

        let mut m_cap: Vec<Scalar> = Vec::new();

        for chunk in bytes[240..].chunks_exact(32) {
            m_cap.push(
                Scalar::from_bytes_be(chunk)
                    .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?,
            );
        }

        let challenge = m_cap.pop().ok_or(Error::InvalidProofOfKnowledgeSignature)?; //at least the challenge should be present (even if all attributes are disclosed)

        Ok(Self {
            Abar,
            Bbar,
            D,
            e_cap,
            r1_cap,
            r3_cap,
            m_cap,
            challenge,
        })
    }

    pub fn pairing_g1_compressed(&self) -> ([u8; 48], [u8; 48]) {
        (
            self.Abar.to_affine().to_compressed(),
            self.Bbar.to_affine().to_compressed(),
        )
    }
}

impl<CS: BbsCiphersuite> PoKSignature<BBSplus<CS>> {
    /// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-proof-generation-proofgen
    ///
    /// # Description
    /// This operation creates BBS proof, which is a zero-knowledge, proof-of-knowledge of a BBS signature, while optionally disclosing any subset of the signed messages.
    ///
    /// # Inputs:
    /// * `pk` (REQUIRED), the Signer public key.
    /// * `signature` (REQUIRED), an octet string.
    /// * `header` (OPTIONAL), an octet string containing context and application.
    /// * `ph` (OPTIONAL), an octet string containing the presentation header.
    /// * `messages` (OPTIONAL), a vector of octet strings representing the signed messages.
    /// * `disclosed_indexes` (OPTIONAL), vector of unsigned integers in ascending order. Indexes of disclosed messages.
    ///
    /// # Output:
    /// a PoK of a Signature [`PoKSignature::BBSplus`] or [`Error`].
    ///
    pub fn proof_gen(
        pk: &BBSplusPublicKey,
        signature: &[u8],
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        messages: Option<&[Vec<u8>]>,
        disclosed_indexes: Option<&[usize]>,
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = BBSplusSignature::from_bytes(
            signature.try_into().map_err(|_| Error::InvalidSignature)?,
        )?;
        let messages = messages.unwrap_or(&[]);
        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);

        let message_scalars = BBSplusMessage::messages_to_scalar::<CS>(messages, CS::API_ID)?;
        let generators = Generators::create::<CS>(messages.len() + 1, Some(CS::API_ID));

        let proof = core_proof_gen::<CS>(
            pk,
            &signature,
            &generators,
            &message_scalars,
            disclosed_indexes,
            header,
            ph,
            Some(CS::API_ID),
            &CS::SEED_MOCKED_SCALAR,
            &CS::MOCKED_SCALAR_DST,
        )?;

        Ok(Self::BBSplus(proof))
    }

    /// Create a BBS disclosure proof from message scalars that were already
    /// mapped with `messages_to_scalar` for the same ciphersuite and API ID.
    pub fn proof_gen_prehashed_message_scalars(
        pk: &BBSplusPublicKey,
        signature: &[u8],
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = parse_signature_cached(signature)?;
        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));

        let proof = core_proof_gen::<CS>(
            pk,
            &signature,
            &generators,
            message_scalars,
            disclosed_indexes,
            header,
            ph,
            Some(CS::API_ID),
            &CS::SEED_MOCKED_SCALAR,
            &CS::MOCKED_SCALAR_DST,
        )?;

        Ok(Self::BBSplus(proof))
    }

    /// Create a BBS disclosure proof from message scalars and a precomputed
    /// B point (`P1 + Q1 * domain + H_i * message_i`).
    pub fn proof_gen_prehashed_message_scalars_and_b(
        pk: &BBSplusPublicKey,
        signature: &[u8],
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        compressed_b: &[u8],
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = parse_signature_cached(signature)?;
        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let b = parse_g1_projective(compressed_b)
            .map_err(|_| Error::ProofGenError("Invalid precomputed B".to_owned()))?;

        let proof = core_proof_gen_with_b::<CS>(
            pk,
            &signature,
            &generators,
            message_scalars,
            disclosed_indexes,
            header,
            ph,
            Some(CS::API_ID),
            &CS::SEED_MOCKED_SCALAR,
            &CS::MOCKED_SCALAR_DST,
            b,
        )?;

        Ok(Self::BBSplus(proof))
    }

    /// Prepare proof generation after B has already been computed externally.
    /// The returned T2 MSM terms can be sent to an accelerator and finalized
    /// with `proof_gen_finalize_prehashed_b_and_t2`.
    pub fn proof_gen_prehashed_message_scalars_and_b_t2_msm_terms(
        pk: &BBSplusPublicKey,
        signature: &[u8],
        header: Option<&[u8]>,
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        compressed_b: &[u8],
    ) -> Result<(BBSplusProofGenT2State, BBSplusProofGenT2MsmTerms), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = parse_signature_cached(signature)?;
        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let b = parse_g1_projective(compressed_b)
            .map_err(|_| Error::ProofGenError("Invalid precomputed B".to_owned()))?;

        proof_gen_t2_msm_terms_with_b::<CS>(
            pk,
            &signature,
            &generators,
            message_scalars,
            disclosed_indexes,
            header,
            Some(CS::API_ID),
            &CS::SEED_MOCKED_SCALAR,
            &CS::MOCKED_SCALAR_DST,
            b,
            None,
        )
    }

    pub fn proof_gen_prehashed_message_scalars_and_b_t2_msm_terms_with_domain(
        _pk: &BBSplusPublicKey,
        signature: &[u8],
        _header: Option<&[u8]>,
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        compressed_b: &[u8],
        domain: Scalar,
    ) -> Result<(BBSplusProofGenT2State, BBSplusProofGenT2MsmTerms), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = parse_signature_cached(signature)?;
        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);
        let compressed_generators =
            proof_verify_compressed_generators_for_count::<CS>(message_scalars.len() + 1);
        let b = parse_g1_projective(compressed_b)
            .map_err(|_| Error::ProofGenError("Invalid precomputed B".to_owned()))?;

        proof_gen_t2_msm_terms_with_compressed_generators::<CS>(
            &signature,
            &compressed_generators,
            message_scalars,
            disclosed_indexes,
            &CS::SEED_MOCKED_SCALAR,
            &CS::MOCKED_SCALAR_DST,
            b,
            domain,
        )
    }

    /// Build the T2 state + MSM terms from externally precomputed G1 points
    /// (d, abar, bbar, t1) — e.g. computed on a GPU. Skips the 6 fixed G1
    /// scalar multiplications. The random scalars and points are provided by
    /// the caller.
    pub fn proof_gen_t2_state_from_precomputed_points(
        signature: &[u8],
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        random_scalars: Vec<Scalar>,
        d: G1Projective,
        abar: G1Projective,
        bbar: G1Projective,
        t1: G1Projective,
        domain: Scalar,
    ) -> Result<(BBSplusProofGenT2State, BBSplusProofGenT2MsmTerms), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = parse_signature_cached(signature)?;
        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);
        let compressed_generators =
            proof_verify_compressed_generators_for_count::<CS>(message_scalars.len() + 1);
        proof_gen_t2_state_from_precomputed_points::<CS>(
            &compressed_generators,
            message_scalars,
            disclosed_indexes,
            random_scalars,
            d,
            abar,
            bbar,
            t1,
            domain,
            signature.e,
        )
    }

    /// Build GPU proof state without decoding points that are already in the
    /// canonical compressed representation used by the challenge and proof.
    pub fn proof_gen_t2_compressed_state_from_precomputed_points(
        signature: &[u8],
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        random_scalars: Vec<Scalar>,
        d: Vec<u8>,
        abar: Vec<u8>,
        bbar: Vec<u8>,
        t1: Vec<u8>,
        domain: Scalar,
    ) -> Result<(BBSplusProofGenT2CompressedState, BBSplusProofGenT2MsmTerms), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        if [&d, &abar, &bbar, &t1]
            .iter()
            .any(|point| point.len() != G1Projective::COMPRESSED_BYTES)
        {
            return Err(Error::ProofGenError(
                "Invalid compressed GPU point".to_owned(),
            ));
        }
        let signature = parse_signature_cached(signature)?;
        let l = message_scalars.len();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();
        if disclosed_indexes.iter().any(|&index| index >= l) {
            return Err(Error::ProofGenError("Invalid disclosed index".to_owned()));
        }
        let undisclosed_indexes = get_remaining_indexes(l, &disclosed_indexes);
        let disclosed_messages = get_messages(message_scalars, &disclosed_indexes);
        let undisclosed_messages = get_messages(message_scalars, &undisclosed_indexes);
        let u = undisclosed_indexes.len();
        if random_scalars.len() != 5 + u {
            return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
        }
        let compressed_generators = proof_verify_compressed_generators_for_count::<CS>(l + 1);
        let h_points = &compressed_generators[2..];
        let mut t2_points = Vec::with_capacity(1 + u);
        let mut t2_scalars = Vec::with_capacity(1 + u);
        push_msm_term_compressed(&mut t2_points, &mut t2_scalars, &d, random_scalars[4]);
        for (offset, &message_index) in undisclosed_indexes.iter().enumerate() {
            push_msm_term_compressed(
                &mut t2_points,
                &mut t2_scalars,
                &h_points[message_index],
                random_scalars[5 + offset],
            );
        }
        Ok((
            BBSplusProofGenT2CompressedState {
                abar,
                bbar,
                d,
                t1,
                domain,
                e: signature.e,
                random_scalars,
                disclosed_indexes,
                disclosed_messages,
                undisclosed_messages,
            },
            BBSplusProofGenT2MsmTerms {
                t2_compressed_points: t2_points,
                t2_scalar_be: t2_scalars,
            },
        ))
    }

    pub fn proof_gen_compressed_state_from_precomputed_points(
        signature: &[u8],
        message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        random_scalars: Vec<Scalar>,
        d: Vec<u8>,
        abar: Vec<u8>,
        bbar: Vec<u8>,
        t1: Vec<u8>,
        domain: Scalar,
    ) -> Result<BBSplusProofGenT2CompressedState, Error> {
        if [&d, &abar, &bbar, &t1]
            .iter()
            .any(|point| point.len() != G1Projective::COMPRESSED_BYTES)
        {
            return Err(Error::ProofGenError(
                "Invalid compressed GPU point".to_owned(),
            ));
        }
        let signature = parse_signature_cached(signature)?;
        let l = message_scalars.len();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();
        if disclosed_indexes.iter().any(|&index| index >= l) {
            return Err(Error::ProofGenError("Invalid disclosed index".to_owned()));
        }
        let undisclosed_indexes = get_remaining_indexes(l, &disclosed_indexes);
        if random_scalars.len() != 5 + undisclosed_indexes.len() {
            return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
        }
        Ok(BBSplusProofGenT2CompressedState {
            abar,
            bbar,
            d,
            t1,
            domain,
            e: signature.e,
            random_scalars,
            disclosed_messages: get_messages(message_scalars, &disclosed_indexes),
            undisclosed_messages: get_messages(message_scalars, &undisclosed_indexes),
            disclosed_indexes,
        })
    }

    pub fn proof_gen_t2_msm_terms_from_compressed_b(
        message_count: usize,
        disclosed_indexes: Option<&[usize]>,
        random_scalars: &[Scalar],
        compressed_b: &[u8],
    ) -> Result<BBSplusProofGenT2MsmTerms, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        if compressed_b.len() != G1Projective::COMPRESSED_BYTES {
            return Err(Error::ProofGenError("Invalid compressed B".to_owned()));
        }
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();
        if disclosed_indexes
            .iter()
            .any(|&index| index >= message_count)
        {
            return Err(Error::ProofGenError("Invalid disclosed index".to_owned()));
        }
        let undisclosed_indexes = get_remaining_indexes(message_count, &disclosed_indexes);
        if random_scalars.len() != 5 + undisclosed_indexes.len() {
            return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
        }
        let compressed_generators =
            proof_verify_compressed_generators_for_count::<CS>(message_count + 1);
        let h_points = &compressed_generators[2..];
        let mut points = Vec::with_capacity(1 + undisclosed_indexes.len());
        let mut scalars = Vec::with_capacity(1 + undisclosed_indexes.len());
        push_msm_term_compressed(
            &mut points,
            &mut scalars,
            compressed_b,
            random_scalars[1] * random_scalars[4],
        );
        for (offset, &message_index) in undisclosed_indexes.iter().enumerate() {
            push_msm_term_compressed(
                &mut points,
                &mut scalars,
                &h_points[message_index],
                random_scalars[5 + offset],
            );
        }
        Ok(BBSplusProofGenT2MsmTerms {
            t2_compressed_points: points,
            t2_scalar_be: scalars,
        })
    }

    pub fn proof_gen_finalize_compressed_state_and_t2(
        state: BBSplusProofGenT2CompressedState,
        ph: Option<&[u8]>,
        t2_compressed: &[u8],
    ) -> Result<Vec<u8>, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let challenge_input = proof_challenge_input_from_all_compressed::<CS>(
            &state.abar,
            &state.bbar,
            &state.d,
            &state.t1,
            t2_compressed,
            state.domain,
            &state.disclosed_indexes,
            &state.disclosed_messages,
            ph,
        )?;
        let challenge_dst = [CS::API_ID, CS::H2S].concat();
        let challenge = hash_to_scalar::<CS>(&challenge_input, &challenge_dst)?;
        Self::proof_gen_finalize_compressed_state_with_challenge(state, challenge)
    }

    pub fn proof_gen_finalize_compressed_state_and_t2_with_r3(
        state: BBSplusProofGenT2CompressedState,
        ph: Option<&[u8]>,
        t2_compressed: &[u8],
        r3: Scalar,
    ) -> Result<Vec<u8>, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let challenge_input = proof_challenge_input_from_all_compressed::<CS>(
            &state.abar,
            &state.bbar,
            &state.d,
            &state.t1,
            t2_compressed,
            state.domain,
            &state.disclosed_indexes,
            &state.disclosed_messages,
            ph,
        )?;
        let challenge_dst = [CS::API_ID, CS::H2S].concat();
        let challenge = hash_to_scalar::<CS>(&challenge_input, &challenge_dst)?;
        Self::proof_gen_finalize_compressed_state_with_challenge_and_r3(state, challenge, r3)
    }

    pub fn proof_gen_challenge_input_compressed_state_and_t2(
        state: &BBSplusProofGenT2CompressedState,
        ph: Option<&[u8]>,
        t2_compressed: &[u8],
    ) -> Result<Vec<u8>, Error>
    where
        CS: BbsCiphersuite,
    {
        proof_challenge_input_from_all_compressed::<CS>(
            &state.abar,
            &state.bbar,
            &state.d,
            &state.t1,
            t2_compressed,
            state.domain,
            &state.disclosed_indexes,
            &state.disclosed_messages,
            ph,
        )
    }

    pub fn proof_gen_finalize_compressed_state_with_challenge(
        state: BBSplusProofGenT2CompressedState,
        challenge: Scalar,
    ) -> Result<Vec<u8>, Error> {
        if state.random_scalars.len() < 2 {
            return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
        }
        let r3 = Option::<Scalar>::from(state.random_scalars[1].invert())
            .ok_or(Error::ProofGenError("Invert scalar failed".to_owned()))?;
        Self::proof_gen_finalize_compressed_state_with_challenge_and_r3(state, challenge, r3)
    }

    pub fn proof_gen_finalize_compressed_state_with_challenge_and_r3(
        state: BBSplusProofGenT2CompressedState,
        challenge: Scalar,
        r3: Scalar,
    ) -> Result<Vec<u8>, Error> {
        let u = state.undisclosed_messages.len();
        if state.random_scalars.len() != 5 + u {
            return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
        }
        let e_cap = state.random_scalars[2] + state.e * challenge;
        let r1_cap = state.random_scalars[3] - state.random_scalars[0] * challenge;
        let r3_cap = state.random_scalars[4] - r3 * challenge;
        let mut proof = Vec::with_capacity(240 + (u * 32) + 32);
        proof.extend_from_slice(&state.abar);
        proof.extend_from_slice(&state.bbar);
        proof.extend_from_slice(&state.d);
        proof.extend_from_slice(&e_cap.to_bytes_be());
        proof.extend_from_slice(&r1_cap.to_bytes_be());
        proof.extend_from_slice(&r3_cap.to_bytes_be());
        for (m_tilde, message) in state.random_scalars[5..]
            .iter()
            .zip(state.undisclosed_messages.iter())
        {
            proof.extend_from_slice(&(*m_tilde + message.value * challenge).to_bytes_be());
        }
        proof.extend_from_slice(&challenge.to_bytes_be());
        Ok(proof)
    }

    /// Finalize proof generation after T2 has been computed externally.
    pub fn proof_gen_finalize_prehashed_b_and_t2(
        state: BBSplusProofGenT2State,
        ph: Option<&[u8]>,
        t2_compressed: &[u8],
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let t2 = parse_g1_projective(t2_compressed)
            .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
        let init_res = ProofInitResult {
            Abar: state.abar,
            Bbar: state.bbar,
            D: state.d,
            T1: state.t1,
            T2: t2,
            domain: state.domain,
        };
        let challenge = proof_challenge_calculate::<CS>(
            &init_res,
            &state.disclosed_indexes,
            &state.disclosed_messages,
            ph,
            Some(CS::API_ID),
        )?;
        let proof = proof_finalize(
            &init_res,
            challenge,
            state.e,
            &state.random_scalars,
            &state.undisclosed_messages,
        )?;

        Ok(Self::BBSplus(proof))
    }

    pub fn proof_gen_challenge_input_prehashed_b_and_t2(
        state: &BBSplusProofGenT2State,
        ph: Option<&[u8]>,
        t2_compressed: &[u8],
    ) -> Result<Vec<u8>, Error>
    where
        CS: BbsCiphersuite,
    {
        proof_challenge_input_from_compressed::<CS>(
            state.abar,
            state.bbar,
            state.d,
            &state.t1.to_affine().to_compressed(),
            t2_compressed,
            state.domain,
            &state.disclosed_indexes,
            &state.disclosed_messages,
            ph,
            Some(CS::API_ID),
        )
    }

    pub fn proof_gen_finalize_prehashed_b_and_t2_with_challenge(
        state: BBSplusProofGenT2State,
        t2_compressed: &[u8],
        challenge: Scalar,
    ) -> Result<Self, Error>
    where
        CS: BbsCiphersuite,
    {
        if t2_compressed.len() != G1Projective::COMPRESSED_BYTES {
            return Err(Error::InvalidProofOfKnowledgeSignature);
        }
        let proof = proof_finalize_parts(
            state.abar,
            state.bbar,
            state.d,
            challenge,
            state.e,
            &state.random_scalars,
            &state.undisclosed_messages,
        )?;

        Ok(Self::BBSplus(proof))
    }

    /// https://datatracker.ietf.org/doc/html/draft-kalos-bbs-blind-signatures-00#name-proof-generation
    ///
    /// # Description
    /// This operation creates a BBS proof, which is a zero-knowledge, proof-of-knowledge, of a BBS signature, while optionally disclosing any subset of the signed messages. Note that in contrast to the [`Self::proof_gen`] operation, this operation accepts 2 different lists of messages and disclosed indexes, one for the messages known to the Signer (messages) and the corresponding disclosed indexes (disclosed_indexes) and one for the messages committed by the Prover (committed_messages) and the corresponding disclosed indexes (disclosed_commitment_indexes).
    /// To Verify a proof however, the Verifier expects only one list of messages and one list of disclosed indexes. This is done to avoid revealing which of the disclosed messages where committed by the Prover and which are known to the Verifier.
    ///
    /// # Inputs:
    /// * `pk` (REQUIRED), the Signer public key.
    /// * `signature` (REQUIRED), an octet string.
    /// * `header` (OPTIONAL), an octet string containing context and application.
    /// * `ph` (OPTIONAL), an octet string containing the presentation header.
    /// * `messages` (OPTIONAL), a vector of octet strings messages supplied by the Signer.  If not supplied, it defaults to the empty array.
    /// * `committed_messages` (OPTIONAL), a vector of octet strings messages committed by the Prover.
    /// * `disclosed_indexes` (OPTIONAL), vector of unsigned integers in ascending order. Indexes of disclosed messages.
    /// * `disclosed_commitment_indexes` (OPTIONAL), vector of unsigned integers in ascending order. Indexes of disclosed committed messages.
    /// * `secret_prover_blind` (OPTIONAL), a scalar value ([`BlindFactor`]).
    /// * `signer_blind` (OPTIONAL), a scalar value ([`BlindFactor`]).
    ///
    /// # Output:
    /// ([`PoKSignature::BBSplus`], [`Vec<Vec<u8>>`], [`Vec<usize>`]) or [`Error`]: a PoK of a Signature, a vector of octet strings representing all the disclosed messages and their indexes.
    ///
    pub fn blind_proof_gen(
        pk: &BBSplusPublicKey,
        signature: &[u8],
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        messages: Option<&[Vec<u8>]>,
        committed_messages: Option<&[Vec<u8>]>,
        disclosed_indexes: Option<&[usize]>,
        disclosed_commitment_indexes: Option<&[usize]>,
        secret_prover_blind: Option<&BlindFactor>,
        signer_blind: Option<&BlindFactor>,
    ) -> Result<(Self, Vec<Vec<u8>>, Vec<usize>), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let signature = parse_signature_cached(signature)?;
        let messages = messages.unwrap_or(&[]);
        let committed_messages = committed_messages.unwrap_or(&[]);
        let L = messages.len();
        let M = committed_messages.len();

        let disclosed_indexes = disclosed_indexes.unwrap_or(&[]);
        let disclosed_commitment_indexes = disclosed_commitment_indexes.unwrap_or(&[]);

        if disclosed_indexes.len() > L {
            return Err(Error::BlindProofGenError(
                "number of disclosed indexes is grater than the number of messages".to_owned(),
            ));
        }

        if disclosed_indexes.iter().any(|&i| i >= L) {
            return Err(Error::BlindProofGenError(
                "disclosed index out of range".to_owned(),
            ));
        }

        if disclosed_commitment_indexes.len() > M {
            return Err(Error::BlindProofGenError("number of commitment disclosed indexes is grater than the number of committed messages".to_owned()));
        }

        if disclosed_commitment_indexes.iter().any(|&i| i >= M) {
            return Err(Error::BlindProofGenError(
                "commitment disclosed index out of range".to_owned(),
            ));
        }

        let mut message_scalars = Vec::new();

        let secret_prover_blind = secret_prover_blind.unwrap_or(&BlindFactor(Scalar::ZERO));

        if secret_prover_blind.0 != Scalar::ZERO {
            let signer_blind = signer_blind.unwrap_or(&BlindFactor(Scalar::ZERO));
            let message = BBSplusMessage::new(secret_prover_blind.0 + signer_blind.0);
            message_scalars.push(message);
        }

        let api_id = CS::API_ID_BLIND;
        message_scalars.extend(BBSplusMessage::messages_to_scalar::<CS>(
            committed_messages,
            api_id,
        )?);
        message_scalars.extend(BBSplusMessage::messages_to_scalar::<CS>(messages, api_id)?);

        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(api_id));

        let (disclosed_messages, disclosed_indexes) = get_disclosed_data(
            messages,
            committed_messages,
            disclosed_indexes,
            disclosed_commitment_indexes,
            secret_prover_blind,
        );

        let proof = core_proof_gen::<CS>(
            pk,
            &signature,
            &generators,
            &message_scalars,
            &disclosed_indexes,
            header,
            ph,
            Some(api_id),
            CS::SEED_MOCKED_SCALAR,
            &CS::BLIND_PROOF_DST,
        )?;

        Ok((Self::BBSplus(proof), disclosed_messages, disclosed_indexes))
    }

    /// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-proof-verification-proofver
    ///
    /// # Description
    /// The ProofVerify operation validates a BBS proof, given the Signer's public key (PK), a header and presentation header values, the disclosed messages and the indexes those messages had in the original vector of signed messages.
    /// Inside is using the [`BbsCiphersuite::API_ID`] api_id.
    ///
    /// # Inputs:
    /// * `self`, a proof.
    /// * `pk` (REQUIRED), the Signer public key.
    /// * `disclosed_messages` (OPTIONAL), a vector of octet string representing the messages disclosed to the Verifier.
    /// * `disclosed_indexes` (OPTIONAL), vector of usize in ascending order. Indexes of disclosed messages.
    /// * `header` (OPTIONAL), an octet string containing context and application.
    /// * `ph` (OPTIONAL), an octet string containing the presentation header.
    ///
    /// # Output:
    /// a result: [`Ok`] or [`Error`].
    ///
    pub fn proof_verify(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_messages: Option<&[Vec<u8>]>,
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let disclosed_messages = disclosed_messages.unwrap_or(&[]);
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let U = proof.m_cap.len();
        let R = disclosed_indexes.len();

        let disclosed_message_scalars =
            BBSplusMessage::messages_to_scalar::<CS>(disclosed_messages, CS::API_ID)?;

        let generators = Generators::create::<CS>(U + R + 1, Some(CS::API_ID));

        let result = core_proof_verify::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            &disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
        );

        result
    }

    /// Verify a BBS disclosure proof from disclosed message scalars that were
    /// already mapped with `messages_to_scalar` for the same ciphersuite and
    /// API ID.
    pub fn proof_verify_prehashed_disclosed_scalars(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let U = proof.m_cap.len();
        let R = disclosed_indexes.len();

        let generators = Generators::create::<CS>(U + R + 1, Some(CS::API_ID));

        core_proof_verify::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
        )
    }

    /// Return the two G1 MSM inputs needed during proof verification. The
    /// caller may compute these MSMs externally, then pass the resulting T1/T2
    /// points to `proof_verify_prehashed_disclosed_scalars_and_msm`.
    pub fn proof_verify_prehashed_disclosed_scalars_msm_terms(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
    ) -> Result<BBSplusProofVerifyMsmTerms, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        self.proof_verify_prehashed_disclosed_scalars_msm_terms_with_domain(
            pk,
            disclosed_message_scalars,
            disclosed_indexes,
            header,
            None,
        )
    }

    pub fn proof_verify_prehashed_disclosed_scalars_msm_terms_with_domain(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        precomputed_domain: Option<Scalar>,
    ) -> Result<BBSplusProofVerifyMsmTerms, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let u = proof.m_cap.len();
        let r = disclosed_indexes.len();
        if let Some(domain) = precomputed_domain {
            let compressed_generators =
                proof_verify_compressed_generators_for_count::<CS>(u + r + 1);
            return proof_verify_msm_terms_with_compressed_generators::<CS>(
                proof,
                &compressed_generators,
                disclosed_message_scalars,
                &disclosed_indexes,
                domain,
                None,
            );
        }

        let generators = Generators::create::<CS>(u + r + 1, Some(CS::API_ID));

        proof_verify_msm_terms::<CS>(
            pk,
            proof,
            &generators,
            header,
            disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
            precomputed_domain,
        )
    }

    /// Build proof-verification MSM terms while reusing the canonical G1
    /// encodings from the serialized proof. `proof_bytes` must be the byte
    /// representation from which this proof was parsed.
    pub fn proof_verify_prehashed_disclosed_scalars_msm_terms_with_domain_and_proof_bytes(
        &self,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        precomputed_domain: Scalar,
        proof_bytes: &[u8],
    ) -> Result<BBSplusProofVerifyMsmTerms, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        if proof_bytes.len() < 144 {
            return Err(Error::InvalidProofOfKnowledgeSignature);
        }
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();
        let compressed_generators = proof_verify_compressed_generators_for_count::<CS>(
            proof.m_cap.len() + disclosed_indexes.len() + 1,
        );
        proof_verify_msm_terms_with_compressed_generators::<CS>(
            proof,
            &compressed_generators,
            disclosed_message_scalars,
            &disclosed_indexes,
            precomputed_domain,
            Some((
                &proof_bytes[0..48],
                &proof_bytes[48..96],
                &proof_bytes[96..144],
            )),
        )
    }

    /// Build verification MSM terms directly from a canonical serialized
    /// proof. Scalar encodings are checked here; compressed G1 encodings are
    /// checked by the MSM/pairing consumer before use.
    pub fn proof_verify_msm_terms_from_proof_bytes_with_domain(
        proof_bytes: &[u8],
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        domain: Scalar,
    ) -> Result<BBSplusProofVerifyMsmTerms, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        if proof_bytes.len() < 272 || (proof_bytes.len() - 240) % 32 != 0 {
            return Err(Error::InvalidProofOfKnowledgeSignature);
        }
        let scalar = |range: std::ops::Range<usize>| {
            Scalar::from_bytes_be(&proof_bytes[range])
                .map_err(|_| Error::InvalidProofOfKnowledgeSignature)
        };
        let e_cap = scalar(144..176)?;
        let r1_cap = scalar(176..208)?;
        let r3_cap = scalar(208..240)?;
        let challenge = scalar(proof_bytes.len() - 32..proof_bytes.len())?;
        let mut m_cap = Vec::with_capacity((proof_bytes.len() - 272) / 32);
        for chunk in proof_bytes[240..proof_bytes.len() - 32].chunks_exact(32) {
            m_cap.push(
                Scalar::from_bytes_be(chunk)
                    .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?,
            );
        }

        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();
        if disclosed_message_scalars.len() != disclosed_indexes.len() {
            return Err(Error::PoKSVerificationError(
                "Invalid disclosed message count".to_owned(),
            ));
        }
        let u = m_cap.len();
        let r = disclosed_indexes.len();
        let l = u + r;
        if disclosed_indexes.iter().any(|&index| index >= l) {
            return Err(Error::PoKSVerificationError(
                "Invalid disclosed index".to_owned(),
            ));
        }
        let compressed_generators = proof_verify_compressed_generators_for_count::<CS>(l + 1);
        let g1_base = &compressed_generators[0];
        let q1 = &compressed_generators[1];
        let h_points = &compressed_generators[2..];
        let undisclosed_indexes = get_remaining_indexes(l, &disclosed_indexes);

        let mut t1_points = Vec::with_capacity(3);
        let mut t1_scalars = Vec::with_capacity(3);
        push_msm_term_compressed(
            &mut t1_points,
            &mut t1_scalars,
            &proof_bytes[48..96],
            challenge,
        );
        push_msm_term_compressed(&mut t1_points, &mut t1_scalars, &proof_bytes[0..48], e_cap);
        push_msm_term_compressed(
            &mut t1_points,
            &mut t1_scalars,
            &proof_bytes[96..144],
            r1_cap,
        );

        let mut t2_points = Vec::with_capacity(4 + r + u);
        let mut t2_scalars = Vec::with_capacity(4 + r + u);
        push_msm_term_compressed(&mut t2_points, &mut t2_scalars, g1_base, challenge);
        push_msm_term_compressed(&mut t2_points, &mut t2_scalars, q1, domain * challenge);
        for i in 0..r {
            push_msm_term_compressed(
                &mut t2_points,
                &mut t2_scalars,
                &h_points[disclosed_indexes[i]],
                disclosed_message_scalars[i].value * challenge,
            );
        }
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &proof_bytes[96..144],
            r3_cap,
        );
        for (scalar, &index) in m_cap.iter().zip(undisclosed_indexes.iter()) {
            push_msm_term_compressed(&mut t2_points, &mut t2_scalars, &h_points[index], *scalar);
        }
        Ok(BBSplusProofVerifyMsmTerms {
            t1_compressed_points: t1_points,
            t1_scalar_be: t1_scalars,
            t2_compressed_points: t2_points,
            t2_scalar_be: t2_scalars,
            domain,
        })
    }

    pub fn proof_verify_domain_for_total_message_count(
        pk: &BBSplusPublicKey,
        total_message_count: usize,
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
    ) -> Result<Scalar, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let domain_input = Self::proof_verify_domain_input_for_total_message_count(
            pk,
            total_message_count,
            disclosed_indexes,
            header,
        )?;
        let domain_dst = [CS::API_ID, CS::H2S].concat();
        hash_to_scalar::<CS>(&domain_input, &domain_dst)
    }

    pub fn proof_verify_domain_input_for_total_message_count(
        pk: &BBSplusPublicKey,
        total_message_count: usize,
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
    ) -> Result<Vec<u8>, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        for &i in &disclosed_indexes {
            if i >= total_message_count {
                return Err(Error::PoKSVerificationError(
                    "Invalid disclosed indexes".to_owned(),
                ));
            }
        }

        // Domain calculation serializes Q1 and every H point. Reusing the
        // compressed generator cache avoids regenerating and recompressing the
        // same shape for every issuer in a multi-tenant verification batch.
        let compressed =
            proof_verify_compressed_generators_for_count::<CS>(total_message_count + 1);
        let q1_and_h = &compressed[1..];
        let header = header.unwrap_or(b"");
        let mut domain_input = Vec::with_capacity(
            pk.to_bytes().len() + 8 + q1_and_h.len() * 48 + CS::API_ID.len() + 8 + header.len(),
        );
        domain_input.extend_from_slice(&pk.to_bytes());
        domain_input.extend_from_slice(&i2osp(total_message_count, 8));
        for point in q1_and_h {
            domain_input.extend_from_slice(point);
        }
        domain_input.extend_from_slice(CS::API_ID);
        domain_input.extend_from_slice(&i2osp(header.len(), 8));
        domain_input.extend_from_slice(header);
        Ok(domain_input)
    }

    /// Verify a BBS disclosure proof when T1/T2 have already been computed,
    /// for example by a GPU MSM backend.
    pub fn proof_verify_prehashed_disclosed_scalars_and_msm(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        t1_compressed: &[u8],
        t2_compressed: &[u8],
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let u = proof.m_cap.len();
        let r = disclosed_indexes.len();
        let generators = Generators::create::<CS>(u + r + 1, Some(CS::API_ID));

        core_proof_verify_with_precomputed_msm::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
            t1_compressed,
            t2_compressed,
        )
    }

    /// Verify only the Fiat-Shamir challenge after externally computing T1/T2.
    /// Pairing verification can then be delegated to an accelerator.
    pub fn proof_verify_prehashed_disclosed_scalars_and_msm_challenge(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        t1_compressed: &[u8],
        t2_compressed: &[u8],
        precomputed_domain: Option<Scalar>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let u = proof.m_cap.len();
        let r = disclosed_indexes.len();
        let generators = Generators::create::<CS>(u + r + 1, Some(CS::API_ID));

        core_proof_verify_precomputed_msm_challenge::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
            t1_compressed,
            t2_compressed,
            precomputed_domain,
        )
    }

    pub fn proof_verify_prehashed_disclosed_scalars_and_msm_challenge_input(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        t1_compressed: &[u8],
        t2_compressed: &[u8],
        precomputed_domain: Option<Scalar>,
    ) -> Result<Vec<u8>, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let u = proof.m_cap.len();
        let r = disclosed_indexes.len();
        let generators = Generators::create::<CS>(u + r + 1, Some(CS::API_ID));

        proof_verify_precomputed_msm_challenge_input::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
            t1_compressed,
            t2_compressed,
            precomputed_domain,
        )
    }

    pub fn proof_verify_prehashed_disclosed_scalars_and_msm_challenge_fast(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_message_scalars: &[BBSplusMessage],
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
        t1_compressed: &[u8],
        t2_compressed: &[u8],
        precomputed_domain: Option<Scalar>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let u = proof.m_cap.len();
        let r = disclosed_indexes.len();
        let generators = Generators::create::<CS>(u + r + 1, Some(CS::API_ID));

        let challenge_input = proof_verify_precomputed_msm_challenge_input_fast::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID),
            t1_compressed,
            t2_compressed,
            precomputed_domain,
        )?;
        let challenge = hash_to_scalar::<CS>(&challenge_input, &[CS::API_ID, CS::H2S].concat())?;

        if proof.challenge == challenge {
            Ok(())
        } else {
            Err(Error::PoKSVerificationError("invalid challenge".to_owned()))
        }
    }

    pub fn proof_verify_prehashed_disclosed_scalars_and_msm_challenge_scalar(
        &self,
        challenge: Scalar,
    ) -> Result<(), Error> {
        if self.to_bbsplus_proof().challenge == challenge {
            Ok(())
        } else {
            Err(Error::PoKSVerificationError("invalid challenge".to_owned()))
        }
    }

    pub fn proof_verify_prehashed_disclosed_scalars_and_msm_batch(
        items: &[BBSplusProofVerifyPrecomputedItem<'_, CS>],
    ) -> Result<(), Error>
    where
        CS: Sync,
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        if items.is_empty() {
            return Ok(());
        }

        let proofs = if items.len() >= 4 {
            thread::scope(|scope| {
                items
                    .iter()
                    .map(|item| scope.spawn(move || proof_verify_precomputed_item::<CS>(item)))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().expect("proof verify worker panicked"))
                    .collect::<Result<Vec<_>, Error>>()
            })?
        } else {
            items
                .iter()
                .map(proof_verify_precomputed_item::<CS>)
                .collect::<Result<Vec<_>, Error>>()?
        };

        core_proof_pairing_verify_batch(&proofs)
    }

    /// https://datatracker.ietf.org/doc/html/draft-kalos-bbs-blind-signatures-00#name-proof-verification
    ///
    /// # Description
    /// It is the same as [`Self::proof_verify`] but inside i using a different api_id parameter ([`BbsCiphersuite::API_ID_BLIND`])
    ///
    pub fn blind_proof_verify(
        &self,
        pk: &BBSplusPublicKey,
        disclosed_messages: Option<&[Vec<u8>]>,
        disclosed_indexes: Option<&[usize]>,
        header: Option<&[u8]>,
        ph: Option<&[u8]>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let proof = self.to_bbsplus_proof();
        let disclosed_messages = disclosed_messages.unwrap_or(&[]);
        let mut disclosed_indexes = disclosed_indexes.unwrap_or(&[]).to_vec();
        disclosed_indexes.sort();
        disclosed_indexes.dedup();

        let U = proof.m_cap.len();
        let R = disclosed_indexes.len();

        let disclosed_message_scalars =
            BBSplusMessage::messages_to_scalar::<CS>(disclosed_messages, CS::API_ID_BLIND)?;

        let generators = Generators::create::<CS>(U + R + 1, Some(CS::API_ID_BLIND));

        let result = core_proof_verify::<CS>(
            pk,
            proof,
            &generators,
            header,
            ph,
            &disclosed_message_scalars,
            &disclosed_indexes,
            Some(CS::API_ID_BLIND),
        );

        result
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bbsplus_proof().to_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::BBSplus(BBSplusPoKSignature::from_bytes(bytes)?))
    }

    pub fn to_bbsplus_proof(&self) -> &BBSplusPoKSignature {
        match self {
            Self::BBSplus(inner) => inner,
            _ => panic!("Cannot happen!"),
        }
    }
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-coreproofgen
///
/// # Description
/// This operation computes a zero-knowledge proof-of-knowledge of a signature, while optionally selectively disclosing from the original set of signed messages. The Prover may also supply a presentation header (ph).
///
/// # Inputs:
/// * `pk` (REQUIRED), the Signer public key.
/// * `signature` (REQUIRED), a [`BBSplusSignature`].
/// * `generators` (REQUIRED), vector of pseudo-random points in G1.
/// * `header` (OPTIONAL), an octet string containing context and application.
/// * `ph` (OPTIONAL), an octet string containing the presentation header.
/// * `messages` (OPTIONAL), a vector of scalars ([`BBSplusMessage`]) representing the signed messages.
/// * `disclosed_indexes` (OPTIONAL), vector of usize in ascending order. Indexes of disclosed messages.
/// * `api_id` (OPTIONAL), an octet string.
///
/// # Output:
/// a PoK of a Signature [`BBSplusPoKSignature`] or [`Error`].
///
fn core_proof_gen<CS>(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    generators: &Generators,
    messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    api_id: Option<&[u8]>,
    _seed: &[u8],
    _dst: &[u8],
) -> Result<BBSplusPoKSignature, Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let L = messages.len();
    if L > generators.values.len() - 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let mut disclosed_indexes = disclosed_indexes.to_vec();
    disclosed_indexes.sort();
    disclosed_indexes.dedup();

    let R = disclosed_indexes.len();

    let U = L
        .checked_sub(R)
        .ok_or(Error::ProofGenError("R > L".to_owned()))?;

    if let Some(invalid_index) = disclosed_indexes.iter().find(|&&i| i > L - 1) {
        return Err(Error::ProofGenError(format!(
            "Invalid disclosed index: {}",
            invalid_index
        )));
    }

    let undisclosed_indexes: Vec<usize> = get_remaining_indexes(L, &disclosed_indexes);

    let disclosed_messages = get_messages(messages, &disclosed_indexes);
    let undisclosed_messages = get_messages(messages, &undisclosed_indexes);

    #[cfg(not(test))]
    let random_scalars = calculate_random_scalars(5 + U);

    #[cfg(test)]
    let random_scalars = seeded_random_scalars::<CS>(5 + U, _seed, _dst);

    let init_res = proof_init::<CS>(
        pk,
        signature,
        generators,
        &random_scalars,
        header,
        messages,
        &undisclosed_indexes,
        api_id,
    )?;

    let challenge = proof_challenge_calculate::<CS>(
        &init_res,
        &disclosed_indexes,
        &disclosed_messages,
        ph,
        api_id,
    )?;

    let proof = proof_finalize(
        &init_res,
        challenge,
        signature.e,
        &random_scalars,
        &undisclosed_messages,
    )?;

    Ok(proof)
}

fn core_proof_gen_with_b<CS>(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    generators: &Generators,
    messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    api_id: Option<&[u8]>,
    _seed: &[u8],
    _dst: &[u8],
    b: G1Projective,
) -> Result<BBSplusPoKSignature, Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let l = messages.len();
    if l > generators.values.len() - 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let mut disclosed_indexes = disclosed_indexes.to_vec();
    disclosed_indexes.sort();
    disclosed_indexes.dedup();

    let r = disclosed_indexes.len();
    let u = l
        .checked_sub(r)
        .ok_or(Error::ProofGenError("R > L".to_owned()))?;

    if let Some(invalid_index) = disclosed_indexes.iter().find(|&&i| i > l - 1) {
        return Err(Error::ProofGenError(format!(
            "Invalid disclosed index: {}",
            invalid_index
        )));
    }

    let undisclosed_indexes: Vec<usize> = get_remaining_indexes(l, &disclosed_indexes);
    let disclosed_messages = get_messages(messages, &disclosed_indexes);
    let undisclosed_messages = get_messages(messages, &undisclosed_indexes);

    #[cfg(not(test))]
    let random_scalars = calculate_random_scalars(5 + u);

    #[cfg(test)]
    let random_scalars = seeded_random_scalars::<CS>(5 + u, _seed, _dst);

    let init_res = proof_init_with_b::<CS>(
        pk,
        signature,
        generators,
        &random_scalars,
        header,
        &undisclosed_indexes,
        api_id,
        b,
    )?;

    let challenge = proof_challenge_calculate::<CS>(
        &init_res,
        &disclosed_indexes,
        &disclosed_messages,
        ph,
        api_id,
    )?;

    let proof = proof_finalize(
        &init_res,
        challenge,
        signature.e,
        &random_scalars,
        &undisclosed_messages,
    )?;

    Ok(proof)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProofInitResult {
    Abar: G1Projective,
    Bbar: G1Projective,
    D: G1Projective,
    T1: G1Projective,
    T2: G1Projective,
    domain: Scalar,
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-proof-initialization
///
/// # Description
/// This operation initializes the proof and returns one of the inputs passed to the challenge calculation operation ([`proof_challenge_calculate`]), during the [`core_proof_gen`] operation.
/// The inputted messages MUST be supplied to this operation in the same order they had when inputted to the `core_sign` operation
///
/// # Inputs:
/// * `pk` (REQUIRED), the Signer public key.
/// * `signature` (REQUIRED), a [`BBSplusSignature`].
/// * `generators` (REQUIRED), vector of pseudo-random points in G1.
/// * `random_scalars` (REQUIRED), vector of scalar values.
/// * `header` (OPTIONAL), an octet string containing context and application.
/// * `messages` (OPTIONAL), a vector of scalars ([`BBSplusMessage`]).
/// * `undisclosed_indexes` (OPTIONAL), vector of usize in ascending order. Indexes of undisclosed messages.
/// * `api_id` (OPTIONAL), an octet string.
///
/// # Output:
/// a [`ProofInitResult`] or [`Error`].
///
fn proof_init<CS>(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    generators: &Generators,
    random_scalars: &[Scalar],
    header: Option<&[u8]>,
    messages: &[BBSplusMessage],
    undisclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
) -> Result<ProofInitResult, Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let L = messages.len();
    let U = undisclosed_indexes.len();

    if random_scalars.len() != 5 + U {
        return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
    }

    if generators.values.len() != L + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let Q1 = generators.values[0];
    let H_points = &generators.values[1..];

    let domain = calculate_domain::<CS>(pk, Q1, H_points, header, api_id)?;

    let mut B = generators.g1_base_point + Q1 * domain;
    for i in 0..L {
        B = B + H_points[i] * messages[i].value;
    }

    let r1 = random_scalars[0];
    let r2 = random_scalars[1];
    let e_tilde = random_scalars[2];
    let r1_tilde = random_scalars[3];
    let r3_tilde = random_scalars[4];
    let m_tilde = &random_scalars[5..(5 + U)];

    let D = B * r2;
    let Abar = signature.A * (r1 * r2);
    let Bbar = D * r1 - Abar * signature.e;

    let T1 = Abar * e_tilde + D * r1_tilde;
    let mut T2 = D * r3_tilde;

    for idx in 0..U {
        T2 = T2 + H_points[undisclosed_indexes[idx]] * m_tilde[idx];
    }

    Ok(ProofInitResult {
        Abar,
        Bbar,
        D,
        T1,
        T2,
        domain,
    })
}

fn proof_init_with_b<CS>(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    generators: &Generators,
    random_scalars: &[Scalar],
    header: Option<&[u8]>,
    undisclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
    b: G1Projective,
) -> Result<ProofInitResult, Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let u = undisclosed_indexes.len();

    if random_scalars.len() != 5 + u {
        return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
    }

    let q1 = generators.values[0];
    let h_points = &generators.values[1..];
    let domain = calculate_domain::<CS>(pk, q1, h_points, header, api_id)?;

    let r1 = random_scalars[0];
    let r2 = random_scalars[1];
    let e_tilde = random_scalars[2];
    let r1_tilde = random_scalars[3];
    let r3_tilde = random_scalars[4];
    let m_tilde = &random_scalars[5..(5 + u)];

    let d = b * r2;
    let abar = signature.A * (r1 * r2);
    let bbar = d * r1 - abar * signature.e;

    let t1 = abar * e_tilde + d * r1_tilde;
    let mut t2 = d * r3_tilde;

    for idx in 0..u {
        t2 = t2 + h_points[undisclosed_indexes[idx]] * m_tilde[idx];
    }

    Ok(ProofInitResult {
        Abar: abar,
        Bbar: bbar,
        D: d,
        T1: t1,
        T2: t2,
        domain,
    })
}

fn proof_gen_t2_msm_terms_with_b<CS>(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    generators: &Generators,
    messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    header: Option<&[u8]>,
    api_id: Option<&[u8]>,
    _seed: &[u8],
    _dst: &[u8],
    b: G1Projective,
    precomputed_domain: Option<Scalar>,
) -> Result<(BBSplusProofGenT2State, BBSplusProofGenT2MsmTerms), Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let l = messages.len();
    if l > generators.values.len() - 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let mut disclosed_indexes = disclosed_indexes.to_vec();
    disclosed_indexes.sort();
    disclosed_indexes.dedup();

    let r = disclosed_indexes.len();
    let u = l
        .checked_sub(r)
        .ok_or(Error::ProofGenError("R > L".to_owned()))?;

    if let Some(invalid_index) = disclosed_indexes.iter().find(|&&i| i > l - 1) {
        return Err(Error::ProofGenError(format!(
            "Invalid disclosed index: {}",
            invalid_index
        )));
    }

    let undisclosed_indexes: Vec<usize> = get_remaining_indexes(l, &disclosed_indexes);
    let disclosed_messages = get_messages(messages, &disclosed_indexes);
    let undisclosed_messages = get_messages(messages, &undisclosed_indexes);

    #[cfg(not(test))]
    let random_scalars = calculate_random_scalars(5 + u);

    #[cfg(test)]
    let random_scalars = seeded_random_scalars::<CS>(5 + u, _seed, _dst);

    if random_scalars.len() != 5 + u {
        return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
    }

    let q1 = generators.values[0];
    let h_points = &generators.values[1..];
    let compressed_generators =
        proof_verify_compressed_generators_for_count::<CS>(generators.values.len());
    let h_points_compressed = &compressed_generators[2..];
    let domain = match precomputed_domain {
        Some(domain) => domain,
        None => calculate_domain::<CS>(pk, q1, h_points, header, api_id)?,
    };

    let r1 = random_scalars[0];
    let r2 = random_scalars[1];
    let e_tilde = random_scalars[2];
    let r1_tilde = random_scalars[3];
    let r3_tilde = random_scalars[4];
    let m_tilde = &random_scalars[5..(5 + u)];

    let d = b * r2;
    let abar = signature.A * (r1 * r2);
    let bbar = d * r1 - abar * signature.e;
    let t1 = abar * e_tilde + d * r1_tilde;

    let mut t2_points = Vec::with_capacity(1 + u);
    let mut t2_scalars = Vec::with_capacity(1 + u);
    push_msm_term(&mut t2_points, &mut t2_scalars, d, r3_tilde);
    for idx in 0..u {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[undisclosed_indexes[idx]],
            m_tilde[idx],
        );
    }

    Ok((
        BBSplusProofGenT2State {
            abar,
            bbar,
            d,
            t1,
            domain,
            e: signature.e,
            random_scalars,
            disclosed_indexes,
            disclosed_messages,
            undisclosed_messages,
        },
        BBSplusProofGenT2MsmTerms {
            t2_compressed_points: t2_points,
            t2_scalar_be: t2_scalars,
        },
    ))
}

fn proof_gen_t2_msm_terms_with_compressed_generators<CS>(
    signature: &BBSplusSignature,
    compressed_generators: &[Vec<u8>],
    messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    _seed: &[u8],
    _dst: &[u8],
    b: G1Projective,
    domain: Scalar,
) -> Result<(BBSplusProofGenT2State, BBSplusProofGenT2MsmTerms), Error>
where
    CS: BbsCiphersuite,
{
    let l = messages.len();
    if compressed_generators.len() < l + 2 {
        return Err(Error::NotEnoughGenerators);
    }

    let mut disclosed_indexes = disclosed_indexes.to_vec();
    disclosed_indexes.sort();
    disclosed_indexes.dedup();

    let r = disclosed_indexes.len();
    let u = l
        .checked_sub(r)
        .ok_or(Error::ProofGenError("R > L".to_owned()))?;

    if let Some(invalid_index) = disclosed_indexes.iter().find(|&&i| i > l - 1) {
        return Err(Error::ProofGenError(format!(
            "Invalid disclosed index: {}",
            invalid_index
        )));
    }

    let undisclosed_indexes: Vec<usize> = get_remaining_indexes(l, &disclosed_indexes);
    let disclosed_messages = get_messages(messages, &disclosed_indexes);
    let undisclosed_messages = get_messages(messages, &undisclosed_indexes);
    let h_points_compressed = &compressed_generators[2..];

    #[cfg(not(test))]
    let random_scalars = calculate_random_scalars(5 + u);

    #[cfg(test)]
    let random_scalars = seeded_random_scalars::<CS>(5 + u, _seed, _dst);

    if random_scalars.len() != 5 + u {
        return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
    }

    let r1 = random_scalars[0];
    let r2 = random_scalars[1];
    let e_tilde = random_scalars[2];
    let r1_tilde = random_scalars[3];
    let r3_tilde = random_scalars[4];
    let m_tilde = &random_scalars[5..(5 + u)];

    let d = b * r2;
    let abar = signature.A * (r1 * r2);
    let bbar = d * r1 - abar * signature.e;
    let t1 = abar * e_tilde + d * r1_tilde;

    let mut t2_points = Vec::with_capacity(1 + u);
    let mut t2_scalars = Vec::with_capacity(1 + u);
    push_msm_term(&mut t2_points, &mut t2_scalars, d, r3_tilde);
    for idx in 0..u {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[undisclosed_indexes[idx]],
            m_tilde[idx],
        );
    }

    Ok((
        BBSplusProofGenT2State {
            abar,
            bbar,
            d,
            t1,
            domain,
            e: signature.e,
            random_scalars,
            disclosed_indexes,
            disclosed_messages,
            undisclosed_messages,
        },
        BBSplusProofGenT2MsmTerms {
            t2_compressed_points: t2_points,
            t2_scalar_be: t2_scalars,
        },
    ))
}

/// Build the proof-gen T2 state + MSM terms from externally precomputed G1
/// points (e.g. computed on a GPU). Same as
/// proof_gen_t2_msm_terms_with_compressed_generators but skips the 6 fixed G1
/// scalar multiplications, accepting d/abar/bbar/t1 precomputed.
pub fn proof_gen_t2_state_from_precomputed_points<CS>(
    compressed_generators: &[Vec<u8>],
    messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    random_scalars: Vec<Scalar>,
    d: G1Projective,
    abar: G1Projective,
    bbar: G1Projective,
    t1: G1Projective,
    domain: Scalar,
    e: Scalar,
) -> Result<(BBSplusProofGenT2State, BBSplusProofGenT2MsmTerms), Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let l = messages.len();
    if compressed_generators.len() < l + 2 {
        return Err(Error::NotEnoughGenerators);
    }
    let mut di = disclosed_indexes.to_vec();
    di.sort();
    di.dedup();
    let r = di.len();
    let u = l
        .checked_sub(r)
        .ok_or(Error::ProofGenError("R > L".to_owned()))?;
    if random_scalars.len() != 5 + u {
        return Err(Error::ProofGenError("Random scalars not valid".to_owned()));
    }
    let undisclosed_indexes: Vec<usize> = get_remaining_indexes(l, &di);
    let disclosed_messages = get_messages(messages, &di);
    let undisclosed_messages = get_messages(messages, &undisclosed_indexes);
    let h_points_compressed = &compressed_generators[2..];
    let r3_tilde = random_scalars[4];
    let m_tilde = &random_scalars[5..(5 + u)];
    let mut t2_points = Vec::with_capacity(1 + u);
    let mut t2_scalars = Vec::with_capacity(1 + u);
    push_msm_term(&mut t2_points, &mut t2_scalars, d, r3_tilde);
    for idx in 0..u {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[undisclosed_indexes[idx]],
            m_tilde[idx],
        );
    }
    Ok((
        BBSplusProofGenT2State {
            abar,
            bbar,
            d,
            t1,
            domain,
            e,
            random_scalars,
            disclosed_indexes: di,
            disclosed_messages,
            undisclosed_messages,
        },
        BBSplusProofGenT2MsmTerms {
            t2_compressed_points: t2_points,
            t2_scalar_be: t2_scalars,
        },
    ))
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-challenge-calculation
///
/// # Description
/// This operation calculates the challenge scalar value, used during the [`core_proof_gen`] and [`core_proof_verify`], as part of the Fiat-Shamir heuristic, for making the proof protocol non-interactive (in a interactive setting, the challenge would be a random value supplied by the Verifier).
///
/// # Inputs:
/// * `init_res` (REQUIRED), [`ProofInitResult`] returned after initializing the proof generation or verification operations, consisting of 5 points of G1 and a scalar value, in that order.
/// * `header` (OPTIONAL), an octet string containing context and application.
/// * `dsclosed_messages` (OPTIONAL), a vector of scalars ([`BBSplusMessage`]) representing the disclosed messages to the Verifier.
/// * `disclosed_indexes` (OPTIONAL), vector of usize in ascending order. Indexes of disclosed messages.
/// * `ph` (OPTIONAL), an octet string containing the presentation header.
/// * `api_id` (OPTIONAL), an octet string.
///
/// # Output:
/// a challenge ([`Scalar`]) or [`Error`].
///
fn proof_challenge_calculate<CS>(
    init_res: &ProofInitResult,
    disclosed_indexes: &[usize],
    disclosed_messages: &[BBSplusMessage],
    ph: Option<&[u8]>,
    api_id: Option<&[u8]>,
) -> Result<Scalar, Error>
where
    CS: BbsCiphersuite,
{
    let api_id = api_id.unwrap_or(b"");
    let challenge_dst = [api_id, CS::H2S].concat();
    let c_arr = proof_challenge_input::<CS>(
        init_res,
        disclosed_indexes,
        disclosed_messages,
        ph,
        Some(api_id),
    )?;
    let challenge = hash_to_scalar::<CS>(&c_arr, &challenge_dst)?;

    Ok(challenge)
}

fn proof_challenge_input<CS>(
    init_res: &ProofInitResult,
    disclosed_indexes: &[usize],
    disclosed_messages: &[BBSplusMessage],
    ph: Option<&[u8]>,
    _api_id: Option<&[u8]>,
) -> Result<Vec<u8>, Error>
where
    CS: BbsCiphersuite,
{
    proof_challenge_input_from_compressed::<CS>(
        init_res.Abar,
        init_res.Bbar,
        init_res.D,
        &init_res.T1.to_affine().to_compressed(),
        &init_res.T2.to_affine().to_compressed(),
        init_res.domain,
        disclosed_indexes,
        disclosed_messages,
        ph,
        _api_id,
    )
}

fn proof_challenge_input_from_compressed<CS>(
    abar: G1Projective,
    bbar: G1Projective,
    d: G1Projective,
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    domain: Scalar,
    disclosed_indexes: &[usize],
    disclosed_messages: &[BBSplusMessage],
    ph: Option<&[u8]>,
    _api_id: Option<&[u8]>,
) -> Result<Vec<u8>, Error>
where
    CS: BbsCiphersuite,
{
    proof_challenge_input_from_all_compressed::<CS>(
        &abar.to_affine().to_compressed(),
        &bbar.to_affine().to_compressed(),
        &d.to_affine().to_compressed(),
        t1_compressed,
        t2_compressed,
        domain,
        disclosed_indexes,
        disclosed_messages,
        ph,
    )
}

fn proof_challenge_input_from_all_compressed<CS>(
    abar_compressed: &[u8],
    bbar_compressed: &[u8],
    d_compressed: &[u8],
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    domain: Scalar,
    disclosed_indexes: &[usize],
    disclosed_messages: &[BBSplusMessage],
    ph: Option<&[u8]>,
) -> Result<Vec<u8>, Error>
where
    CS: BbsCiphersuite,
{
    let r = disclosed_indexes.len();

    if disclosed_messages.len() != r {
        return Err(Error::ProofGenError(
            "Number of disclosed indexes different from number of disclosed messages".to_owned(),
        ));
    }
    if [
        abar_compressed,
        bbar_compressed,
        d_compressed,
        t1_compressed,
        t2_compressed,
    ]
    .iter()
    .any(|point| point.len() != G1Projective::COMPRESSED_BYTES)
    {
        return Err(Error::InvalidProofOfKnowledgeSignature);
    }

    let ph = ph.unwrap_or(b"");
    let mut c_arr =
        Vec::with_capacity(8 + (r * 40) + (5 * G1Projective::COMPRESSED_BYTES) + 32 + 8 + ph.len());
    c_arr.extend_from_slice(&i2osp(r, 8));
    for i in 0..r {
        c_arr.extend_from_slice(&i2osp(disclosed_indexes[i], 8));
        c_arr.extend_from_slice(&disclosed_messages[i].value.to_bytes_be());
    }
    c_arr.extend_from_slice(abar_compressed);
    c_arr.extend_from_slice(bbar_compressed);
    c_arr.extend_from_slice(d_compressed);
    c_arr.extend_from_slice(t1_compressed);
    c_arr.extend_from_slice(t2_compressed);
    c_arr.extend_from_slice(&domain.to_bytes_be());
    c_arr.extend_from_slice(&i2osp(ph.len(), 8));
    c_arr.extend_from_slice(ph);
    Ok(c_arr)
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-proof-finalization
///
/// # Description
/// This operation finalizes the proof calculation during the [`core_proof_gen`] operation and returns the PoK [`BBSplusPoKSignature`].
///
/// # Inputs:
/// * `init_res` (REQUIRED), [`ProofInitResult`] returned after initializing the proof generation or verification operations, consisting of 5 points of G1 and a scalar value, in that order.
/// * `challenge` (REQUIRED), a scalar value.
/// * `e` (REQUIRED), a scalar value.
/// * `random_scalars` (REQUIRED), a vector of random scalars.
/// * `dsclosed_messages` (OPTIONAL), a vector of scalars ([`BBSplusMessage`]) representing the disclosed messages to the Verifier.
///
/// # Output:
/// a proof ([`BBSplusPoKSignature`]) or [`Error`].
///
fn proof_finalize(
    init_res: &ProofInitResult,
    challenge: Scalar,
    e: Scalar,
    random_scalars: &[Scalar],
    undisclosed_messages: &[BBSplusMessage],
) -> Result<BBSplusPoKSignature, Error> {
    proof_finalize_parts(
        init_res.Abar,
        init_res.Bbar,
        init_res.D,
        challenge,
        e,
        random_scalars,
        undisclosed_messages,
    )
}

fn proof_finalize_parts(
    abar: G1Projective,
    bbar: G1Projective,
    d: G1Projective,
    challenge: Scalar,
    e: Scalar,
    random_scalars: &[Scalar],
    undisclosed_messages: &[BBSplusMessage],
) -> Result<BBSplusPoKSignature, Error> {
    let U = undisclosed_messages.len();

    let r1 = random_scalars[0];
    let r2 = random_scalars[1];
    let e_tilde = random_scalars[2];
    let r1_tilde = random_scalars[3];
    let r3_tilde = random_scalars[4];
    let m_tilde = &random_scalars[5..(5 + U)];

    let r3 = Option::<Scalar>::from(r2.invert())
        .ok_or(Error::ProofGenError("Invert scalar failed".to_owned()))?;

    let e_cap = e_tilde + e * challenge;

    let r1_cap = r1_tilde - r1 * challenge;
    let r3_cap = r3_tilde - r3 * challenge;
    let mut m_cap: Vec<Scalar> = Vec::new();

    for j in 0..U {
        let m_cap_j = m_tilde[j] + undisclosed_messages[j].value * challenge;
        m_cap.push(m_cap_j);
    }

    Ok(BBSplusPoKSignature {
        Abar: abar,
        Bbar: bbar,
        D: d,
        e_cap,
        r1_cap,
        r3_cap,
        m_cap,
        challenge,
    })
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-coreproofverify
///
/// # Description
/// This operation checks that a proof is valid for a header, vector of disclosed messages (disclosed_messages) along side their index corresponding to their original position when signed (disclosed_indexes) and presentation header (ph) against a public key (PK).
///
/// # Inputs:
/// * `pk` (REQUIRED), the Signer public key.
/// * `proof` (REQUIRED), a [`BBSplusPoKSignature`].
/// * `generators` (REQUIRED), vector of pseudo-random points in G1.
/// * `header` (OPTIONAL), an octet string containing context and application.
/// * `ph` (OPTIONAL), an octet string containing the presentation header.
/// * `disclosed_messages` (OPTIONAL), a vector of scalars ([`BBSplusMessage`]) representing the messages disclosed to the Verifier.
/// * `disclosed_indexes` (OPTIONAL), vector of unsigned integers in ascending order. Indexes of disclosed messages.
/// * `api_id` (OPTIONAL), an octet string.
///
/// # Output:
/// a result: [`Ok`] or [`Error`].
///
fn core_proof_verify<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
) -> Result<(), Error>
where
    CS: BbsCiphersuite,
{
    let init_res = proof_verify_init::<CS>(
        pk,
        proof,
        generators,
        header,
        disclosed_messages,
        disclosed_indexes,
        api_id,
    )?;

    let challenge = proof_challenge_calculate::<CS>(
        &init_res,
        disclosed_indexes,
        disclosed_messages,
        ph,
        api_id,
    )?;

    if proof.challenge != challenge {
        return Err(Error::PoKSVerificationError("invalid challenge".to_owned()));
    }

    let BP2 = G2Projective::GENERATOR;

    let term1 = (&proof.Abar.to_affine(), &G2Prepared::from(pk.0.to_affine()));
    let term2 = (&proof.Bbar.to_affine(), &G2Prepared::from(-BP2.to_affine()));

    let pairing = multi_miller_loop(&[term1, term2]).final_exponentiation();

    if pairing.is_identity().into() {
        Ok(())
    } else {
        Err(Error::PoKSVerificationError("Invalid Proof".to_owned()))
    }
}

fn core_proof_verify_with_precomputed_msm<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
    t1_compressed: &[u8],
    t2_compressed: &[u8],
) -> Result<(), Error>
where
    CS: BbsCiphersuite,
{
    core_proof_verify_precomputed_msm_challenge::<CS>(
        pk,
        proof,
        generators,
        header,
        ph,
        disclosed_messages,
        disclosed_indexes,
        api_id,
        t1_compressed,
        t2_compressed,
        None,
    )?;

    core_proof_pairing_verify(pk, proof)
}

fn core_proof_verify_precomputed_msm_challenge<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    precomputed_domain: Option<Scalar>,
) -> Result<(), Error>
where
    CS: BbsCiphersuite,
{
    let domain = match precomputed_domain {
        Some(domain) => domain,
        None => proof_verify_domain::<CS>(
            pk,
            proof,
            generators,
            header,
            disclosed_messages,
            disclosed_indexes,
            api_id,
        )?,
    };
    let t1 =
        parse_g1_projective(t1_compressed).map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
    let t2 =
        parse_g1_projective(t2_compressed).map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;

    let init_res = ProofInitResult {
        Abar: proof.Abar,
        Bbar: proof.Bbar,
        D: proof.D,
        T1: t1,
        T2: t2,
        domain,
    };

    let challenge = proof_challenge_calculate::<CS>(
        &init_res,
        disclosed_indexes,
        disclosed_messages,
        ph,
        api_id,
    )?;

    if proof.challenge != challenge {
        return Err(Error::PoKSVerificationError("invalid challenge".to_owned()));
    }

    Ok(())
}

fn proof_verify_precomputed_msm_challenge_input<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    precomputed_domain: Option<Scalar>,
) -> Result<Vec<u8>, Error>
where
    CS: BbsCiphersuite,
{
    let domain = match precomputed_domain {
        Some(domain) => domain,
        None => proof_verify_domain::<CS>(
            pk,
            proof,
            generators,
            header,
            disclosed_messages,
            disclosed_indexes,
            api_id,
        )?,
    };
    let t1 =
        parse_g1_projective(t1_compressed).map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
    let t2 =
        parse_g1_projective(t2_compressed).map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;

    let init_res = ProofInitResult {
        Abar: proof.Abar,
        Bbar: proof.Bbar,
        D: proof.D,
        T1: t1,
        T2: t2,
        domain,
    };

    proof_challenge_input::<CS>(&init_res, disclosed_indexes, disclosed_messages, ph, api_id)
}

fn proof_verify_precomputed_msm_challenge_input_fast<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    ph: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
    t1_compressed: &[u8],
    t2_compressed: &[u8],
    precomputed_domain: Option<Scalar>,
) -> Result<Vec<u8>, Error>
where
    CS: BbsCiphersuite,
{
    let domain = match precomputed_domain {
        Some(domain) => domain,
        None => proof_verify_domain::<CS>(
            pk,
            proof,
            generators,
            header,
            disclosed_messages,
            disclosed_indexes,
            api_id,
        )?,
    };

    proof_challenge_input_from_compressed::<CS>(
        proof.Abar,
        proof.Bbar,
        proof.D,
        t1_compressed,
        t2_compressed,
        domain,
        disclosed_indexes,
        disclosed_messages,
        ph,
        api_id,
    )
}

fn proof_verify_precomputed_item<'a, CS>(
    item: &'a BBSplusProofVerifyPrecomputedItem<'a, CS>,
) -> Result<(&'a BBSplusPoKSignature, &'a BBSplusPublicKey), Error>
where
    CS: BbsCiphersuite + Sync,
    CS::Expander: for<'b> ExpandMsg<'b>,
{
    let proof = item.proof.to_bbsplus_proof();
    let mut disclosed_indexes = item.disclosed_indexes.unwrap_or(&[]).to_vec();
    disclosed_indexes.sort();
    disclosed_indexes.dedup();

    let u = proof.m_cap.len();
    let r = disclosed_indexes.len();
    let generators = Generators::create::<CS>(u + r + 1, Some(CS::API_ID));

    core_proof_verify_precomputed_msm_challenge::<CS>(
        item.pk,
        proof,
        &generators,
        item.header,
        item.ph,
        item.disclosed_message_scalars,
        &disclosed_indexes,
        Some(CS::API_ID),
        item.t1_compressed,
        item.t2_compressed,
        item.precomputed_domain,
    )?;

    Ok((proof, item.pk))
}

fn core_proof_pairing_verify(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
) -> Result<(), Error> {
    let BP2 = G2Projective::GENERATOR;

    let term1 = (&proof.Abar.to_affine(), &G2Prepared::from(pk.0.to_affine()));
    let term2 = (&proof.Bbar.to_affine(), &G2Prepared::from(-BP2.to_affine()));

    let pairing = multi_miller_loop(&[term1, term2]).final_exponentiation();

    if pairing.is_identity().into() {
        Ok(())
    } else {
        Err(Error::PoKSVerificationError("Invalid Proof".to_owned()))
    }
}

fn core_proof_pairing_verify_batch(
    items: &[(&BBSplusPoKSignature, &BBSplusPublicKey)],
) -> Result<(), Error> {
    let BP2 = G2Projective::GENERATOR;
    let neg_bp2_prepared = G2Prepared::from(-BP2.to_affine());
    let prepared = if items.len() >= 4 {
        thread::scope(|scope| {
            items
                .iter()
                .map(|(proof, pk)| {
                    scope.spawn(move || {
                        let coeff = Scalar::random(&mut rand::thread_rng());
                        (
                            proof.Abar * coeff,
                            proof.Bbar * coeff,
                            G2Prepared::from(pk.0.to_affine()),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("proof verify pairing worker panicked"))
                .collect::<Vec<_>>()
        })
    } else {
        items
            .iter()
            .map(|(proof, pk)| {
                let coeff = Scalar::random(&mut rand::thread_rng());
                (
                    proof.Abar * coeff,
                    proof.Bbar * coeff,
                    G2Prepared::from(pk.0.to_affine()),
                )
            })
            .collect::<Vec<_>>()
    };

    let mut terms = Vec::with_capacity(items.len() * 2);
    for (scaled_abar, scaled_bbar, prepared_pk) in &prepared {
        terms.push((scaled_abar.to_affine(), prepared_pk));
        terms.push((scaled_bbar.to_affine(), &neg_bp2_prepared));
    }

    let borrowed_terms = terms.iter().map(|(g1, g2)| (g1, *g2)).collect::<Vec<_>>();
    let pairing = multi_miller_loop(&borrowed_terms).final_exponentiation();

    if pairing.is_identity().into() {
        Ok(())
    } else {
        Err(Error::PoKSVerificationError("Invalid Proof".to_owned()))
    }
}

fn proof_verify_domain<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
) -> Result<Scalar, Error>
where
    CS: BbsCiphersuite,
{
    let u = proof.m_cap.len();
    let r = disclosed_indexes.len();
    let l = u + r;

    for &i in disclosed_indexes {
        if i > l - 1 {
            return Err(Error::PoKSVerificationError(
                "Invalid disclosed indexes".to_owned(),
            ));
        }
    }

    if disclosed_messages.len() != r {
        return Err(Error::PoKSVerificationError(
            "len messages != len indexes".to_owned(),
        ));
    }

    if generators.values.len() != l + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let q1 = generators.values[0];
    let h_points = &generators.values[1..];
    calculate_domain::<CS>(pk, q1, h_points, header, api_id)
}

fn proof_verify_msm_terms<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
    precomputed_domain: Option<Scalar>,
) -> Result<BBSplusProofVerifyMsmTerms, Error>
where
    CS: BbsCiphersuite,
{
    let u = proof.m_cap.len();
    let r = disclosed_indexes.len();
    let l = u + r;
    let domain = match precomputed_domain {
        Some(domain) => domain,
        None => proof_verify_domain::<CS>(
            pk,
            proof,
            generators,
            header,
            disclosed_messages,
            disclosed_indexes,
            api_id,
        )?,
    };

    let compressed_generators = proof_verify_compressed_generators::<CS>(generators);
    let g1_base_compressed = &compressed_generators[0];
    let q1_compressed = &compressed_generators[1];
    let h_points_compressed = &compressed_generators[2..];
    let undisclosed_indexes = get_remaining_indexes(l, disclosed_indexes);

    let mut t1_points = Vec::with_capacity(3);
    let mut t1_scalars = Vec::with_capacity(3);
    push_msm_term(&mut t1_points, &mut t1_scalars, proof.Bbar, proof.challenge);
    push_msm_term(&mut t1_points, &mut t1_scalars, proof.Abar, proof.e_cap);
    push_msm_term(&mut t1_points, &mut t1_scalars, proof.D, proof.r1_cap);

    let mut t2_points = Vec::with_capacity(4 + r + u);
    let mut t2_scalars = Vec::with_capacity(4 + r + u);
    push_msm_term_compressed(
        &mut t2_points,
        &mut t2_scalars,
        g1_base_compressed,
        proof.challenge,
    );
    push_msm_term_compressed(
        &mut t2_points,
        &mut t2_scalars,
        q1_compressed,
        domain * proof.challenge,
    );
    for i in 0..r {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[disclosed_indexes[i]],
            disclosed_messages[i].value * proof.challenge,
        );
    }
    push_msm_term(&mut t2_points, &mut t2_scalars, proof.D, proof.r3_cap);
    for j in 0..u {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[undisclosed_indexes[j]],
            proof.m_cap[j],
        );
    }

    Ok(BBSplusProofVerifyMsmTerms {
        t1_compressed_points: t1_points,
        t1_scalar_be: t1_scalars,
        t2_compressed_points: t2_points,
        t2_scalar_be: t2_scalars,
        domain,
    })
}

fn proof_verify_msm_terms_with_compressed_generators<CS>(
    proof: &BBSplusPoKSignature,
    compressed_generators: &[Vec<u8>],
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    domain: Scalar,
    proof_points: Option<(&[u8], &[u8], &[u8])>,
) -> Result<BBSplusProofVerifyMsmTerms, Error>
where
    CS: BbsCiphersuite,
{
    let u = proof.m_cap.len();
    let r = disclosed_indexes.len();
    let l = u + r;
    if compressed_generators.len() != l + 2 {
        return Err(Error::PoKSVerificationError(
            "Invalid compressed generator count".to_owned(),
        ));
    }
    if disclosed_messages.len() != r {
        return Err(Error::PoKSVerificationError(
            "Invalid disclosed message count".to_owned(),
        ));
    }

    let g1_base_compressed = &compressed_generators[0];
    let q1_compressed = &compressed_generators[1];
    let h_points_compressed = &compressed_generators[2..];
    let undisclosed_indexes = get_remaining_indexes(l, disclosed_indexes);

    let mut t1_points = Vec::with_capacity(3);
    let mut t1_scalars = Vec::with_capacity(3);
    if let Some((abar, bbar, d)) = proof_points {
        push_msm_term_compressed(&mut t1_points, &mut t1_scalars, bbar, proof.challenge);
        push_msm_term_compressed(&mut t1_points, &mut t1_scalars, abar, proof.e_cap);
        push_msm_term_compressed(&mut t1_points, &mut t1_scalars, d, proof.r1_cap);
    } else {
        push_msm_term(&mut t1_points, &mut t1_scalars, proof.Bbar, proof.challenge);
        push_msm_term(&mut t1_points, &mut t1_scalars, proof.Abar, proof.e_cap);
        push_msm_term(&mut t1_points, &mut t1_scalars, proof.D, proof.r1_cap);
    }

    let mut t2_points = Vec::with_capacity(4 + r + u);
    let mut t2_scalars = Vec::with_capacity(4 + r + u);
    push_msm_term_compressed(
        &mut t2_points,
        &mut t2_scalars,
        g1_base_compressed,
        proof.challenge,
    );
    push_msm_term_compressed(
        &mut t2_points,
        &mut t2_scalars,
        q1_compressed,
        domain * proof.challenge,
    );
    for i in 0..r {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[disclosed_indexes[i]],
            disclosed_messages[i].value * proof.challenge,
        );
    }
    if let Some((_, _, d)) = proof_points {
        push_msm_term_compressed(&mut t2_points, &mut t2_scalars, d, proof.r3_cap);
    } else {
        push_msm_term(&mut t2_points, &mut t2_scalars, proof.D, proof.r3_cap);
    }
    for j in 0..u {
        push_msm_term_compressed(
            &mut t2_points,
            &mut t2_scalars,
            &h_points_compressed[undisclosed_indexes[j]],
            proof.m_cap[j],
        );
    }

    Ok(BBSplusProofVerifyMsmTerms {
        t1_compressed_points: t1_points,
        t1_scalar_be: t1_scalars,
        t2_compressed_points: t2_points,
        t2_scalar_be: t2_scalars,
        domain,
    })
}

fn push_msm_term(
    points: &mut Vec<Vec<u8>>,
    scalars: &mut Vec<Vec<u8>>,
    point: G1Projective,
    scalar: Scalar,
) {
    points.push(point.to_affine().to_compressed().to_vec());
    scalars.push(scalar.to_be_bytes().to_vec());
}

fn push_msm_term_compressed(
    points: &mut Vec<Vec<u8>>,
    scalars: &mut Vec<Vec<u8>>,
    compressed_point: &[u8],
    scalar: Scalar,
) {
    points.push(compressed_point.to_vec());
    scalars.push(scalar.to_be_bytes().to_vec());
}

fn proof_verify_compressed_generators<CS>(generators: &Generators) -> Vec<Vec<u8>>
where
    CS: BbsCiphersuite,
{
    type Cache = HashMap<(Vec<u8>, usize), Vec<Vec<u8>>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

    let key = (CS::API_ID.to_vec(), generators.values.len());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().expect("proof verify generator cache poisoned");
        if let Some(cached) = guard.get(&key) {
            return cached.clone();
        }
    }

    let mut compressed = Vec::with_capacity(generators.values.len() + 1);
    compressed.push(
        generators
            .g1_base_point
            .to_affine()
            .to_compressed()
            .to_vec(),
    );
    compressed.extend(
        generators
            .values
            .iter()
            .map(|point| point.to_affine().to_compressed().to_vec()),
    );

    let mut guard = cache.lock().expect("proof verify generator cache poisoned");
    guard.insert(key, compressed.clone());
    compressed
}

fn proof_verify_compressed_generators_for_count<CS>(generator_count: usize) -> Vec<Vec<u8>>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    type Cache = HashMap<(Vec<u8>, usize), Vec<Vec<u8>>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

    let key = (CS::API_ID.to_vec(), generator_count);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().expect("proof verify generator cache poisoned");
        if let Some(cached) = guard.get(&key) {
            return cached.clone();
        }
    }

    let generators = Generators::create::<CS>(generator_count, Some(CS::API_ID));
    let compressed = proof_verify_compressed_generators::<CS>(&generators);
    let mut guard = cache.lock().expect("proof verify generator cache poisoned");
    guard.insert(key, compressed.clone());
    compressed
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-proof-verification-initiali
///
/// # Description
/// This operation initializes the proof verification operation and returns part of the input that will be passed to the challenge calculation operation ([`proof_challenge_calculate`]), during the [`core_proof_verify`] operation.
///
/// # Inputs:
/// * `pk` (REQUIRED), the Signer public key.
/// * `proof` (REQUIRED), a [`BBSplusPoKSignature`].
/// * `generators` (REQUIRED), vector of pseudo-random points in G1.
/// * `header` (OPTIONAL), an octet string containing context and application.
/// * `disclosed_messages` (REQUIRED), a vector of scalars ([`BBSplusMessage`]) representing the messages disclosed to the Verifier. It could be empty.
/// * `disclosed_indexes` (REQUIRED), vector of unsigned integers in ascending order. Indexes of disclosed messages. It could be empty.
/// * `api_id` (OPTIONAL), an octet string.
///
/// # Output:
/// a [`ProofInitResult`] or [`Error`].
///
fn proof_verify_init<CS>(
    pk: &BBSplusPublicKey,
    proof: &BBSplusPoKSignature,
    generators: &Generators,
    header: Option<&[u8]>,
    disclosed_messages: &[BBSplusMessage],
    disclosed_indexes: &[usize],
    api_id: Option<&[u8]>,
) -> Result<ProofInitResult, Error>
where
    CS: BbsCiphersuite,
{
    let U = proof.m_cap.len();
    let R = disclosed_indexes.len();

    let L = U + R;

    for &i in disclosed_indexes {
        if i > L - 1 {
            return Err(Error::PoKSVerificationError(
                "Invalid disclosed indexes".to_owned(),
            ));
        }
    }

    if disclosed_messages.len() != R {
        return Err(Error::PoKSVerificationError(
            "len messages != len indexes".to_owned(),
        ));
    }

    if generators.values.len() != L + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let Q1 = generators.values[0];
    let H_points = &generators.values[1..];

    let undisclosed_indexes = get_remaining_indexes(L, disclosed_indexes);

    let domain = calculate_domain::<CS>(pk, Q1, H_points, header, api_id)?;

    let T1 = proof.Bbar * proof.challenge + proof.Abar * proof.e_cap + proof.D * proof.r1_cap;
    let mut Bv = generators.g1_base_point + Q1 * domain;

    for i in 0..R {
        Bv += H_points[disclosed_indexes[i]] * disclosed_messages[i].value;
    }

    let mut T2 = Bv * proof.challenge + proof.D * proof.r3_cap;

    for j in 0..U {
        T2 += H_points[undisclosed_indexes[j]] * proof.m_cap[j];
    }

    Ok(ProofInitResult {
        Abar: proof.Abar,
        Bbar: proof.Bbar,
        D: proof.D,
        T1,
        T2,
        domain,
    })
}

/// Represents a Commitment proof
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BBSplusZKPoK {
    pub(crate) s_cap: Scalar,
    pub(crate) m_cap: Vec<Scalar>,
    pub(crate) challenge: Scalar,
}

impl BBSplusZKPoK {
    pub fn new(s_cap: Scalar, m_cap: Vec<Scalar>, challenge: Scalar) -> Self {
        Self {
            s_cap,
            m_cap,
            challenge,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();

        bytes.extend_from_slice(&self.s_cap.to_bytes_be());
        self.m_cap
            .iter()
            .for_each(|s| bytes.extend_from_slice(&s.to_bytes_be()));
        bytes.extend_from_slice(&self.challenge.to_bytes_be());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let s_cap = Scalar::from_bytes_be(
            &<[u8; 32]>::try_from(&bytes[0..32])
                .map_err(|_| Error::InvalidProofOfKnowledgeSignature)?,
        )?;

        let mut m_cap: Vec<Scalar> = Vec::new();

        for chunk in bytes[32..].chunks_exact(32) {
            let b =
                <[u8; 32]>::try_from(chunk).map_err(|_| Error::InvalidProofOfKnowledgeSignature)?;
            m_cap.push(Scalar::from_bytes_be(&b)?);
        }

        let challenge = m_cap.pop().ok_or(Error::InvalidProofOfKnowledgeSignature)?; //at least the challenge should be present (even if all attributes are disclosed)

        Ok(Self {
            s_cap,
            m_cap,
            challenge,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        bbsplus::{
            ciphersuites::BbsCiphersuite, commitment::BlindFactor, keys::BBSplusPublicKey,
            proof::seeded_random_scalars, signature::BBSplusSignature,
        },
        schemes::{
            algorithms::{BBSplus, BbsBls12381Sha256, BbsBls12381Shake256, Scheme},
            generics::{PoKSignature, Signature},
        },
        utils::util::bbsplus_utils::{get_messages_vec, ScalarExt},
    };
    use elliptic_curve::hash2curve::ExpandMsg;
    use std::{collections::BTreeMap, fs};

    //mocked_rng - SHA256 - UPDATED
    #[test]
    fn mocked_rng_sha256() {
        mocked_rng::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "mockedRng.json");
    }

    //mocked_rng - SHAKE256 - UPDATED
    #[test]
    fn mocked_rng_shake256() {
        mocked_rng::<BbsBls12381Shake256>("./fixture_data/bls12-381-shake-256/", "mockedRng.json");
    }

    //SIGNATURE POK - SHA256
    #[test]
    fn proof_check_sha256_1() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof001.json")
    }
    #[test]
    fn proof_check_sha256_2() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof002.json")
    }
    #[test]
    fn proof_check_sha256_3() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof003.json")
    }
    #[test]
    fn proof_check_sha256_4() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof004.json")
    }
    #[test]
    fn proof_check_sha256_5() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof005.json")
    }
    #[test]
    fn proof_check_sha256_6() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof006.json")
    }
    #[test]
    fn proof_check_sha256_7() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof007.json")
    }
    #[test]
    fn proof_check_sha256_8() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof008.json")
    }
    #[test]
    fn proof_check_sha256_9() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof009.json")
    }
    #[test]
    fn proof_check_sha256_10() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof010.json")
    }
    #[test]
    fn proof_check_sha256_11() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof011.json")
    }
    #[test]
    fn proof_check_sha256_12() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof012.json")
    }
    #[test]
    fn proof_check_sha256_13() {
        proof_check::<BbsBls12381Sha256>("./fixture_data/bls12-381-sha-256/", "proof/proof013.json")
    }

    //SIGNATURE POK - SHAKE256

    #[test]
    fn proof_check_shake256_1() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof001.json",
        )
    }
    #[test]
    fn proof_check_shake256_2() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof002.json",
        )
    }
    #[test]
    fn proof_check_shake256_3() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof003.json",
        )
    }
    #[test]
    fn proof_check_shake256_4() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof004.json",
        )
    }
    #[test]
    fn proof_check_shake256_5() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof005.json",
        )
    }
    #[test]
    fn proof_check_shake256_6() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof006.json",
        )
    }
    #[test]
    fn proof_check_shake256_7() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof007.json",
        )
    }
    #[test]
    fn proof_check_shake256_8() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof008.json",
        )
    }
    #[test]
    fn proof_check_shake256_9() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof009.json",
        )
    }
    #[test]
    fn proof_check_shake256_10() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof010.json",
        )
    }
    #[test]
    fn proof_check_shake256_11() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof011.json",
        )
    }
    #[test]
    fn proof_check_shake256_12() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof012.json",
        )
    }
    #[test]
    fn proof_check_shake256_13() {
        proof_check::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "proof/proof013.json",
        )
    }

    // BLIND PROOF OF KNOWLEDGE OF A SIGNATURE - SHA256

    #[test]
    fn blind_proof_check_sha256_1() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof001.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_sha256_2() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof002.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_sha256_3() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof003.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_sha256_4() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof004.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_sha256_5() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof005.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_sha256_6() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof006.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_sha256_7() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof007.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    #[ignore]
    fn blind_proof_check_sha256_8() {
        blind_proof_check::<BbsBls12381Sha256>(
            "./fixture_data_blind/bls12-381-sha-256/",
            "proof/proof008.json",
            "./fixture_data_blind/",
        )
    }

    // BLIND PROOF OF KNOWLEDGE OF A SIGNATURE - SHAKE256

    #[test]
    fn blind_proof_check_shake256_1() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof001.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_shake256_2() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof002.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_shake256_3() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof003.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_shake256_4() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof004.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_shake256_5() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof005.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_shake256_6() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof006.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    fn blind_proof_check_shake256_7() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof007.json",
            "./fixture_data_blind/",
        )
    }

    #[test]
    #[ignore]
    fn blind_proof_check_shake256_8() {
        blind_proof_check::<BbsBls12381Shake256>(
            "./fixture_data_blind/bls12-381-shake-256/",
            "proof/proof008.json",
            "./fixture_data_blind/",
        )
    }

    fn mocked_rng<S: Scheme>(pathname: &str, filename: &str)
    where
        S::Ciphersuite: BbsCiphersuite,
        <S::Ciphersuite as BbsCiphersuite>::Expander: for<'a> ExpandMsg<'a>,
    {
        let data = fs::read_to_string([pathname, filename].concat()).expect("Unable to read file");
        let res: serde_json::Value = serde_json::from_str(&data).expect("Unable to parse");
        eprintln!("Mocked Random Scalars");

        let seed_ = hex::decode(res["seed"].as_str().unwrap()).unwrap();
        let dst = hex::decode(res["dst"].as_str().unwrap()).unwrap();
        let count: usize = res["count"].as_u64().unwrap().try_into().unwrap();

        let mocked_scalars_hex: Vec<&str> = res["mockedScalars"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();

        let r = seeded_random_scalars::<S::Ciphersuite>(count, &seed_, &dst);

        let mut results = true;

        for i in 0..count {
            let scalar_hex = hex::encode(r[i].to_bytes_be());

            let scalar_expected = mocked_scalars_hex[i];

            if scalar_hex != scalar_expected {
                if results == true {
                    results = false
                }
                eprintln!(" count: {}", i);
                eprintln!(" Expected scalar: {}", scalar_expected);
                eprintln!(" Computed scalar: {}", scalar_hex);
            }
        }

        assert!(results, "Failed");
    }

    fn proof_check<S: Scheme>(pathname: &str, proof_filename: &str)
    where
        S::Ciphersuite: BbsCiphersuite,
        <S::Ciphersuite as BbsCiphersuite>::Expander: for<'a> ExpandMsg<'a>,
    {
        let data =
            fs::read_to_string([pathname, proof_filename].concat()).expect("Unable to read file");
        let proof_json: serde_json::Value = serde_json::from_str(&data).expect("Unable to parse");

        let signerPK_hex = proof_json["signerPublicKey"].as_str().unwrap();
        let header_hex = proof_json["header"].as_str().unwrap();
        let ph_hex = proof_json["presentationHeader"].as_str().unwrap();
        let input_messages: Vec<String> = proof_json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::from_value(m.clone()).unwrap())
            .collect();

        let proof_expected = proof_json["proof"].as_str().unwrap();
        let result_expected = proof_json["result"]["valid"].as_bool().unwrap();

        let ph = hex::decode(ph_hex).unwrap();

        let revealed_message_indexes: Vec<usize> = proof_json["disclosedIndexes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::from_value(m.clone()).unwrap())
            .collect();

        //Get Message Signature
        let signature_expected = proof_json["signature"].as_str().unwrap();

        let signature = Signature::<BBSplus<S::Ciphersuite>>::from_bytes(
            hex::decode(signature_expected)
                .unwrap()
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();

        let header = hex::decode(header_hex).unwrap();
        let PK = BBSplusPublicKey::from_bytes(&hex::decode(signerPK_hex).unwrap()).unwrap();

        let msgs: Vec<Vec<u8>> = input_messages
            .iter()
            .map(|m| hex::decode(m).unwrap())
            .collect();

        let proof = PoKSignature::<BBSplus<S::Ciphersuite>>::proof_gen(
            &PK,
            &signature.to_bytes(),
            Some(&header),
            Some(&ph),
            Some(&msgs),
            Some(&revealed_message_indexes),
        )
        .unwrap();
        let my_encoded_proof = hex::encode(&proof.to_bytes());
        let result0 = proof_expected == my_encoded_proof;
        let result1 = result0 == result_expected;
        if result1 == false {
            println!("  proofGen: {}", result1);
            println!("  Expected: {}", proof_expected);
            println!("  Computed: {}", my_encoded_proof);
            assert!(result1, "Failed");
        }

        // Verify the Proof
        let disclosed_messages = get_messages_vec(&msgs, &revealed_message_indexes);

        let PROOF = PoKSignature::<BBSplus<S::Ciphersuite>>::from_bytes(
            &hex::decode(proof_expected).unwrap(),
        )
        .unwrap();

        let result2 = PROOF
            .proof_verify(
                &PK,
                Some(&disclosed_messages),
                Some(&revealed_message_indexes),
                Some(&header),
                Some(&ph),
            )
            .is_ok();
        let result3 = result2 == result_expected;
        if !result3 {
            eprintln!("  proofVerify: {}", result3);
            eprintln!("  Expected: {}", result_expected);
            eprintln!("  Computed: {}", result2);
            assert!(result3, "failed");
        } else {
            eprintln!("  Expected: {}", signature_expected);
            eprintln!("  Computed: {}", hex::encode(signature.to_bytes()));

            eprintln!("  proofVerify: {}", result3);
            eprintln!("  Expected: {}", result_expected);
            eprintln!("  Computed: {}", result2);
            if result_expected == false {
                eprintln!(
                    "{} ({})",
                    result3,
                    proof_json["result"]["reason"].as_str().unwrap()
                );
            }
        }
    }

    fn blind_proof_check<S: Scheme>(pathname: &str, proof_filename: &str, messages_path: &str)
    where
        S::Ciphersuite: BbsCiphersuite,
        <S::Ciphersuite as BbsCiphersuite>::Expander: for<'a> ExpandMsg<'a>,
    {
        let data =
            fs::read_to_string([pathname, proof_filename].concat()).expect("Unable to read file");
        let proof_json: serde_json::Value = serde_json::from_str(&data).expect("Unable to parse");

        let messages_data = fs::read_to_string([messages_path, "messages.json"].concat())
            .expect("Unable to read file");
        let messages_json: serde_json::Value =
            serde_json::from_str(&messages_data).expect("Unable to parse");

        println!("{}", proof_json["caseName"]);

        let pk_hex = proof_json["signerPublicKey"].as_str().unwrap();

        let pk = BBSplusPublicKey::from_bytes(&hex::decode(pk_hex).unwrap()).unwrap();

        let committed_messages: Option<Vec<String>> = messages_json["committedMessages"]
            .as_array()
            .and_then(|cm| {
                cm.iter()
                    .map(|m| serde_json::from_value(m.clone()).unwrap())
                    .collect()
            });
        let committed_messages: Option<Vec<Vec<u8>>> = match committed_messages {
            Some(cm) => Some(cm.iter().map(|m| hex::decode(m).unwrap()).collect()),
            None => None,
        };

        let messages: Option<Vec<String>> = messages_json["messages"].as_array().and_then(|cm| {
            cm.iter()
                .map(|m| serde_json::from_value(m.clone()).unwrap())
                .collect()
        });
        let messages: Option<Vec<Vec<u8>>> = match messages {
            Some(m) => Some(m.iter().map(|m| hex::decode(m).unwrap()).collect()),
            None => None,
        };

        let secret_prover_blind = proof_json["proverBlind"].as_str().map(|b| {
            BlindFactor::from_bytes(&hex::decode(b).unwrap().try_into().unwrap()).unwrap()
        });
        let signer_blind = proof_json["signerBlind"].as_str().map(|b| {
            BlindFactor::from_bytes(&hex::decode(b).unwrap().try_into().unwrap()).unwrap()
        });
        let header = hex::decode(proof_json["header"].as_str().unwrap()).unwrap();
        let ph = hex::decode(proof_json["presentationHeader"].as_str().unwrap()).unwrap();
        let signature = BBSplusSignature::from_bytes(
            &hex::decode(proof_json["signature"].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap(),
        )
        .unwrap();

        let disclosed_indexes: Option<Vec<usize>> =
            if let Some(values) = proof_json["revealedMessages"].as_object() {
                Some(values.keys().map(|s| s.parse().unwrap()).collect())
            } else {
                None
            };

        let disclosed_commitment_indexes: Option<Vec<usize>> =
            if let Some(values) = proof_json["revealedCommittedMessages"].as_object() {
                Some(values.keys().map(|s| s.parse().unwrap()).collect())
            } else {
                None
            };

        let mut used_committed_messages: Option<Vec<Vec<u8>>> = None;

        if disclosed_commitment_indexes.is_some() {
            used_committed_messages = committed_messages;
        }

        let (proof, disclosed_msgs, disclosed_idxs) =
            PoKSignature::<BBSplus<S::Ciphersuite>>::blind_proof_gen(
                &pk,
                &signature.to_bytes(),
                Some(&header),
                Some(&ph),
                messages.as_deref(),
                used_committed_messages.as_deref(),
                disclosed_indexes.as_deref(),
                disclosed_commitment_indexes.as_deref(),
                secret_prover_blind.as_ref(),
                signer_blind.as_ref(),
            )
            .unwrap();

        if let Some(values) = proof_json["disclosedData"].as_object() {
            let mut sorted_values = BTreeMap::new();
            for (key, value) in values {
                sorted_values.insert(key.parse::<usize>().unwrap(), value);
            }

            let idxs: Vec<usize> = sorted_values.keys().map(|&s| s).collect();
            assert_eq!(disclosed_idxs, idxs);
            let msgs: Vec<Vec<u8>> = sorted_values
                .values()
                .map(|s| hex::decode(s.as_str().unwrap()).unwrap())
                .collect();
            assert_eq!(disclosed_msgs, msgs);
        }

        // These optional blind-proof fixtures were published for the legacy
        // draft-05 transcript. cuBBS-Cred baseline Sign/Verify/ProofGen/
        // ProofVerify follows draft-10 and is exact-vector gated separately.
        // Keep this extension test as a generated-proof semantic round trip;
        // do not compare its randomized/transcript-version-specific bytes.
        let _legacy_expected_proof = proof_json["proof"].as_str().unwrap();

        let result = proof
            .blind_proof_verify(
                &pk,
                Some(&disclosed_msgs),
                Some(&disclosed_idxs),
                Some(&header),
                Some(&ph),
            )
            .is_ok();

        let expected_result = proof_json["result"]["valid"].as_bool().unwrap();

        assert_eq!(result, expected_result);
    }
}
