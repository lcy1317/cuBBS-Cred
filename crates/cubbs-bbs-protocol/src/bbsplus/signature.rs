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

use super::keys::{BBSplusPublicKey, BBSplusSecretKey};
use crate::{
    bbsplus::{ciphersuites::BbsCiphersuite, generators::Generators},
    errors::Error,
    schemes::{algorithms::BBSplus, generics::Signature},
    utils::{
        message::bbsplus_message::BBSplusMessage,
        util::bbsplus_utils::{
            calculate_domain, hash_to_scalar, parse_g1_projective, serialize, ScalarExt,
        },
    },
};
use bls12_381_plus::{multi_miller_loop, G1Projective, G2Prepared, G2Projective, Gt, Scalar};
use elliptic_curve::{group::Curve, hash2curve::ExpandMsg};
use ff::Field;
use serde::{Deserialize, Serialize};
use std::{thread, time::Instant};

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BBSplusSignature {
    pub A: G1Projective,
    pub e: Scalar,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BBSplusSignProfile {
    pub message_scalar_ms: f64,
    pub generator_ms: f64,
    pub domain_ms: f64,
    pub e_scalar_ms: f64,
    pub msm_and_finalize_ms: f64,
    pub total_ms: f64,
    pub message_count: usize,
    pub signature_bytes: usize,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BBSplusPreparedB {
    pub compressed_b: Vec<u8>,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BBSplusPrepareBMsmTerms {
    pub compressed_points: Vec<Vec<u8>>,
    pub scalar_be: Vec<Vec<u8>>,
}

impl BBSplusSignature {
    pub const BYTES: usize = 80;

    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut bytes = [0u8; Self::BYTES];
        bytes[0..G1Projective::COMPRESSED_BYTES]
            .copy_from_slice(&self.A.to_affine().to_compressed());
        let e = self.e.to_be_bytes();
        bytes[G1Projective::COMPRESSED_BYTES..Self::BYTES].copy_from_slice(&e);
        bytes
    }

    pub fn from_bytes(data: &[u8; Self::BYTES]) -> Result<Self, Error> {
        let A: G1Projective = parse_g1_projective(&data[0..G1Projective::COMPRESSED_BYTES])
            .map_err(|_| Error::InvalidSignature)?;
        let e = Scalar::from_bytes_be(&data[G1Projective::COMPRESSED_BYTES..Self::BYTES])
            .map_err(|_| Error::InvalidSignature)?;

        Ok(Self { A, e })
    }
}

impl<CS: BbsCiphersuite> Signature<BBSplus<CS>> {
    pub fn a(&self) -> G1Projective {
        match self {
            Self::BBSplus(inner) => inner.A,
            _ => panic!("Cannot happen!"),
        }
    }

    pub fn e(&self) -> Scalar {
        match self {
            Self::BBSplus(inner) => inner.e,
            _ => panic!("Cannot happen!"),
        }
    }

    /// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-signature-generation-sign
    /// # Description
    /// The `sign` API returns a BBS signature from a secret key (SK), over a header and a set of messages.
    ///
    /// # Inputs:
    /// * `messages` (OPTIONAL), a vector of octet strings representing the messages, it could be an empty vector.
    /// * `sk` (REQUIRED), a secret key
    /// * `pk` (REQUIRED), a public key
    /// * `header` (OPTIONAL), an octet string containing context and application specific information.
    ///
    /// # Output:
    /// * new [`Signature::BBSplus`] or [`Error`]
    pub fn sign(
        messages: Option<&[Vec<u8>]>,
        sk: &BBSplusSecretKey,
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let messages = messages.unwrap_or(&[]);
        let message_scalars = BBSplusMessage::messages_to_scalar::<CS>(messages, CS::API_ID)?;
        let generators = Generators::create::<CS>(messages.len() + 1, Some(CS::API_ID));
        let signature = core_sign::<CS>(
            sk,
            pk,
            generators,
            header,
            &message_scalars,
            Some(CS::API_ID),
        )?;

        Ok(Self::BBSplus(signature))
    }

    /// Profile the main CPU phases of BBS signature generation while producing
    /// a real signature. Timings are intended for backend development.
    pub fn profile_sign(
        messages: Option<&[Vec<u8>]>,
        sk: &BBSplusSecretKey,
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<BBSplusSignProfile, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let total_started = Instant::now();
        let messages = messages.unwrap_or(&[]);

        let started = Instant::now();
        let message_scalars = BBSplusMessage::messages_to_scalar::<CS>(messages, CS::API_ID)?;
        let message_scalar_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let generators = Generators::create::<CS>(messages.len() + 1, Some(CS::API_ID));
        let generator_ms = started.elapsed().as_secs_f64() * 1000.0;

        let L = message_scalars.len();
        if generators.values.len() != L + 1 {
            return Err(Error::NotEnoughGenerators);
        }
        let Q1 = generators.values[0];
        let H_points = &generators.values[1..];
        let api_id = CS::API_ID;

        let started = Instant::now();
        let domain = calculate_domain::<CS>(pk, Q1, H_points, header, Some(api_id))?;
        let domain_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let signature_dst = [api_id, CS::H2S].concat();
        let mut input: Vec<Scalar> = Vec::new();
        input.push(sk.0);
        message_scalars.iter().for_each(|m| input.push(m.value));
        input.push(domain);
        let e = hash_to_scalar::<CS>(&serialize(&input), &signature_dst)?;
        let e_scalar_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let signature = core_sign_with_domain_and_e(sk, generators, &message_scalars, domain, e)?;
        let msm_and_finalize_ms = started.elapsed().as_secs_f64() * 1000.0;

        Ok(BBSplusSignProfile {
            message_scalar_ms,
            generator_ms,
            domain_ms,
            e_scalar_ms,
            msm_and_finalize_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1000.0,
            message_count: messages.len(),
            signature_bytes: signature.to_bytes().len(),
        })
    }

    /// Create a BBS signature from message scalars that were already mapped
    /// with `messages_to_scalar` for the same ciphersuite and API ID.
    ///
    /// This is useful for callers that offload the message-to-scalar stage while
    /// preserving the standard BBS signature bytes and verification behavior.
    pub fn sign_prehashed_message_scalars(
        message_scalars: &[BBSplusMessage],
        sk: &BBSplusSecretKey,
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let signature = core_sign::<CS>(
            sk,
            pk,
            generators,
            header,
            message_scalars,
            Some(CS::API_ID),
        )?;

        Ok(Self::BBSplus(signature))
    }

    /// Return the byte string hashed to produce the deterministic BBS signature
    /// scalar `e` for a set of already-mapped message scalars.
    pub fn signature_e_input_for_prehashed_message_scalars(
        message_scalars: &[BBSplusMessage],
        sk: &BBSplusSecretKey,
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<Vec<u8>, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let Q1 = generators.values[0];
        let H_points = &generators.values[1..];
        let domain = calculate_domain::<CS>(pk, Q1, H_points, header, Some(CS::API_ID))?;

        let mut input: Vec<Scalar> = Vec::new();
        input.push(sk.0);
        message_scalars.iter().for_each(|m| input.push(m.value));
        input.push(domain);

        Ok(serialize(&input))
    }

    /// Create a BBS signature from already-mapped message scalars and an
    /// already-hashed signature scalar `e`.
    pub fn sign_prehashed_message_scalars_and_e(
        message_scalars: &[BBSplusMessage],
        e: &BBSplusMessage,
        sk: &BBSplusSecretKey,
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<Self, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let signature = core_sign_with_e::<CS>(
            sk,
            pk,
            generators,
            header,
            message_scalars,
            e.value,
            Some(CS::API_ID),
        )?;

        Ok(Self::BBSplus(signature))
    }

    /// Compute the G1 `B` point used by BBS signing:
    /// `B = P1 + Q1 * domain + H_1 * msg_1 + ... + H_L * msg_L`.
    pub fn prepare_b_for_prehashed_message_scalars(
        message_scalars: &[BBSplusMessage],
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<BBSplusPreparedB, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let Q1 = generators.values[0];
        let H_points = &generators.values[1..];
        let domain = calculate_domain::<CS>(pk, Q1, H_points, header, Some(CS::API_ID))?;
        let B = core_compute_b(generators, message_scalars, domain)?;

        Ok(BBSplusPreparedB {
            compressed_b: B.to_affine().to_compressed().to_vec(),
        })
    }

    /// Return the exact G1 MSM terms for the BBS signing `B` point:
    /// `B = P1 + Q1 * domain + H_1 * msg_1 + ... + H_L * msg_L`.
    ///
    /// The caller can hand these compressed G1 points and big-endian scalars to
    /// a generic GPU MSM backend without duplicating BBS generator/domain logic
    /// outside the ciphersuite implementation.
    pub fn prepare_b_msm_terms_for_prehashed_message_scalars(
        message_scalars: &[BBSplusMessage],
        pk: &BBSplusPublicKey,
        header: Option<&[u8]>,
    ) -> Result<BBSplusPrepareBMsmTerms, Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let L = message_scalars.len();
        if generators.values.len() != L + 1 {
            return Err(Error::NotEnoughGenerators);
        }

        let Q1 = generators.values[0];
        let H_points = &generators.values[1..];
        let domain = calculate_domain::<CS>(pk, Q1, H_points, header, Some(CS::API_ID))?;

        let mut compressed_points = Vec::with_capacity(L + 2);
        let mut scalar_be = Vec::with_capacity(L + 2);

        compressed_points.push(
            generators
                .g1_base_point
                .to_affine()
                .to_compressed()
                .to_vec(),
        );
        scalar_be.push(Scalar::ONE.to_be_bytes().to_vec());
        compressed_points.push(Q1.to_affine().to_compressed().to_vec());
        scalar_be.push(domain.to_be_bytes().to_vec());

        for i in 0..L {
            compressed_points.push(H_points[i].to_affine().to_compressed().to_vec());
            scalar_be.push(message_scalars[i].to_bytes_be().to_vec());
        }

        Ok(BBSplusPrepareBMsmTerms {
            compressed_points,
            scalar_be,
        })
    }

    /// Finish BBS signing from a precomputed `B` point and deterministic `e`
    /// scalar. This is the intended boundary for a future GPU MSM backend.
    pub fn sign_prehashed_message_scalars_e_and_b(
        e: &BBSplusMessage,
        prepared_b: &BBSplusPreparedB,
        sk: &BBSplusSecretKey,
    ) -> Result<Self, Error> {
        let B = parse_g1_projective(&prepared_b.compressed_b)
            .map_err(|_| Error::SignatureGenerationError("Invalid precomputed B".to_owned()))?;
        let signature = core_finalize_signature_from_b(sk, B, e.value)?;

        Ok(Self::BBSplus(signature))
    }

    /// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-signature-verification-veri
    /// # Description
    /// The `verify` API validates a BBS signature, given a public key (PK), a header and a set of messages
    /// # Inputs:
    /// * `self`, the signature
    /// * `pk` (REQUIRED), a public key
    /// * `messages` (OPTIONAL), a vector of octet strings representing the messages, it could be an empty vector.
    /// * `header` (OPTIONAL), an octet string containing context and application specific information.
    ///
    /// # Output:
    /// * a result either [`Ok()`] or [`Error`]
    pub fn verify(
        &self,
        pk: &BBSplusPublicKey,
        messages: Option<&[Vec<u8>]>,
        header: Option<&[u8]>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let messages = messages.unwrap_or(&[]);
        let message_scalars = BBSplusMessage::messages_to_scalar::<CS>(messages, CS::API_ID)?;
        let generators = Generators::create::<CS>(messages.len() + 1, Some(CS::API_ID));
        let signature = self.bbsPlusSignature();

        core_verify::<CS>(
            pk,
            signature,
            &message_scalars,
            generators,
            header,
            Some(CS::API_ID),
        )
    }

    /// Verify a BBS signature with already-hashed message scalars. This keeps
    /// the core verification logic unchanged while allowing callers to replace
    /// the message-to-scalar phase with an external accelerator.
    pub fn verify_prehashed_message_scalars(
        &self,
        pk: &BBSplusPublicKey,
        message_scalars: &[BBSplusMessage],
        header: Option<&[u8]>,
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        let generators = Generators::create::<CS>(message_scalars.len() + 1, Some(CS::API_ID));
        let signature = self.bbsPlusSignature();

        core_verify::<CS>(
            pk,
            signature,
            message_scalars,
            generators,
            header,
            Some(CS::API_ID),
        )
    }

    /// Verify a BBS signature from a precomputed `B` point. This is useful when
    /// an external MSM accelerator has already computed
    /// `P1 + Q1 * domain + H_i * message_i`.
    pub fn verify_prehashed_b(
        &self,
        pk: &BBSplusPublicKey,
        prepared_b: &BBSplusPreparedB,
    ) -> Result<(), Error> {
        let B = parse_g1_projective(&prepared_b.compressed_b)
            .map_err(|_| Error::SignatureVerificationError)?;
        core_verify_precomputed_b(pk, self.bbsPlusSignature(), B)
    }

    /// Batch verify BBS signatures from precomputed `B` points.
    ///
    /// This combines all pairing terms into a single Miller-loop/final
    /// exponentiation pass. Each item is multiplied by a deterministic
    /// hash-to-scalar coefficient derived from the ciphersuite API ID and the
    /// item bytes, preventing simple cancellation between invalid items. If the
    /// aggregate check fails, callers that need per-item results should fall
    /// back to `verify_prehashed_b` for each item.
    pub fn verify_prehashed_b_batch(
        items: &[(&Self, &BBSplusPublicKey, &BBSplusPreparedB)],
    ) -> Result<(), Error>
    where
        CS::Expander: for<'a> ExpandMsg<'a>,
    {
        if items.is_empty() {
            return Ok(());
        }

        let mut parsed = Vec::with_capacity(items.len());
        let mut rng = rand::thread_rng();
        for (signature, pk, prepared_b) in items {
            let B = parse_g1_projective(&prepared_b.compressed_b)
                .map_err(|_| Error::SignatureVerificationError)?;
            let signature = signature.bbsPlusSignature();
            let coeff = Scalar::random(&mut rng);
            parsed.push((signature, *pk, B, coeff));
        }

        core_verify_precomputed_b_batch(&parsed)
    }

    pub fn bbsPlusSignature(&self) -> &BBSplusSignature {
        match self {
            Self::BBSplus(inner) => inner,
            _ => panic!("Cannot happen!"),
        }
    }

    pub fn to_bytes(&self) -> [u8; BBSplusSignature::BYTES] {
        self.bbsPlusSignature().to_bytes()
    }

    pub fn from_bytes(data: &[u8; BBSplusSignature::BYTES]) -> Result<Self, Error> {
        Ok(Self::BBSplus(BBSplusSignature::from_bytes(data)?))
    }

    /// # Description
    /// Update signature with a new value of a signed message
    ///
    /// # Inputs:
    /// * `sk` (REQUIRED), Signer private key.
    /// * `old_message` (REQUIRED), message octet string old value.
    /// * `new_message` (REQUIRED), message octet string new value.
    /// * `update_index` (REQUIRED), index of the message to update.
    /// * `n` (REQUIRED), total number of signed messages.
    ///
    /// # Output:
    /// * new [`BBSplusSignature`] or [`Error`]
    pub fn update_signature(
        &self,
        sk: &BBSplusSecretKey,
        old_message: &[u8],
        new_message: &[u8],
        update_index: usize,
        n: usize,
    ) -> Result<Self, Error> {
        let generators = Generators::create::<CS>(n + 1, Some(CS::API_ID));

        if generators.values.len() <= update_index + 1 {
            return Err(Error::UpdateSignatureError(
                "len(generators) <= update_index".to_owned(),
            ));
        }

        let old_message_scalar =
            BBSplusMessage::map_message_to_scalar_as_hash::<CS>(old_message, CS::API_ID)?;
        let new_message_scalar =
            BBSplusMessage::map_message_to_scalar_as_hash::<CS>(new_message, CS::API_ID)?;

        let H_points = &generators.values[1..];
        let H_i = H_points.get(update_index).ok_or(Error::Unspecified)?;
        let sk_e = sk.0 + self.e();
        let mut B = self.a() * sk_e;
        B = B + (-H_i * old_message_scalar.value);
        B = B + (H_i * new_message_scalar.value);

        let sk_e_inv = Option::<Scalar>::from(sk_e.invert()).ok_or(Error::UpdateSignatureError(
            "Invert scalar failed".to_owned(),
        ))?;
        let A = B * sk_e_inv;

        if A == G1Projective::IDENTITY {
            return Err(Error::UpdateSignatureError("A == IDENTITY G1".to_owned()));
        }

        return Ok(Self::BBSplus(BBSplusSignature { A, e: self.e() }));
    }
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-coresign
/// # Description
/// This operation computes a deterministic signature from a secret key (SK), a set of generators (points of G1) and optionally a header and a vector of messages.
///
/// # Inputs:
/// * `sk` (REQUIRED), a secret key
/// * `pk` (REQUIRED), a public key
/// * `generators` (REQUIRED), vector of pseudo-random points in G1.
/// * `header` (OPTIONAL), an octet string containing context and application specific information.
/// * `messages` (REQUIRED), a vector of scalars (`BBSplusMessage`) representing the messages, it could be an empty vector.
/// * `api_id` (OPTIONAL), an octet string. If not supplied it defaults to theempty octet string ("").
///
/// # Output:
/// * new [`BBSplusSignature`] or [`Error`]
fn core_sign<CS>(
    sk: &BBSplusSecretKey,
    pk: &BBSplusPublicKey,
    generators: Generators,
    header: Option<&[u8]>,
    messages: &[BBSplusMessage],
    api_id: Option<&[u8]>,
) -> Result<BBSplusSignature, Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let L = messages.len();

    if generators.values.len() != L + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let Q1 = generators.values[0];
    let H_points = &generators.values[1..];

    let api_id = api_id.unwrap_or(b"");

    let signature_dst = [api_id, CS::H2S].concat();

    let domain = calculate_domain::<CS>(pk, Q1, H_points, header, Some(api_id))?;

    //serialize
    let mut input: Vec<Scalar> = Vec::new();
    input.push(sk.0);
    messages.iter().for_each(|m| input.push(m.value));
    input.push(domain);

    let e = hash_to_scalar::<CS>(&serialize(&input), &signature_dst)?;

    core_sign_with_e::<CS>(sk, pk, generators, header, messages, e, Some(api_id))
}

fn core_sign_with_e<CS>(
    sk: &BBSplusSecretKey,
    pk: &BBSplusPublicKey,
    generators: Generators,
    header: Option<&[u8]>,
    messages: &[BBSplusMessage],
    e: Scalar,
    api_id: Option<&[u8]>,
) -> Result<BBSplusSignature, Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let L = messages.len();

    if generators.values.len() != L + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let Q1 = generators.values[0];
    let H_points = &generators.values[1..];

    let domain = calculate_domain::<CS>(pk, Q1, H_points, header, api_id)?;

    core_sign_with_domain_and_e(sk, generators, messages, domain, e)
}

fn core_sign_with_domain_and_e(
    sk: &BBSplusSecretKey,
    generators: Generators,
    messages: &[BBSplusMessage],
    domain: Scalar,
    e: Scalar,
) -> Result<BBSplusSignature, Error> {
    let B = core_compute_b(generators, messages, domain)?;
    core_finalize_signature_from_b(sk, B, e)
}

fn core_compute_b(
    generators: Generators,
    messages: &[BBSplusMessage],
    domain: Scalar,
) -> Result<G1Projective, Error> {
    let L = messages.len();
    if generators.values.len() != L + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let Q1 = generators.values[0];
    let H_points = &generators.values[1..];

    // B = P1 + Q_1 * domain + H_1 * msg_1 + ... + H_L * msg_L
    let mut B = generators.g1_base_point + Q1 * domain;
    for i in 0..L {
        B = B + H_points[i] * messages[i].value;
    }

    Ok(B)
}

fn core_finalize_signature_from_b(
    sk: &BBSplusSecretKey,
    B: G1Projective,
    e: Scalar,
) -> Result<BBSplusSignature, Error> {
    // A = B * (1 / (SK + e))
    let sk_e_inv = Option::<Scalar>::from((sk.0 + e).invert()).ok_or(
        Error::SignatureGenerationError("Invert scalar failed".to_owned()),
    )?;
    let A = B * sk_e_inv;

    if A == G1Projective::IDENTITY {
        return Err(Error::SignatureGenerationError(
            "A == Identity_G1".to_owned(),
        ));
    }

    Ok(BBSplusSignature { A, e: e })
}

/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-bbs-signatures-05#name-coreverify
/// # Description
/// This operation checks that a signature is valid for a given set of generators, header and vector of messages, against a supplied public key (PK). The set of messages MUST be supplied in this operation in the same order they were supplied to `core_sign` when creating the signature.
///
/// # Inputs:
/// * `pk` (REQUIRED), a public key
/// * `signature` (REQUIRED), a `BBSplusSignature`
/// * `messages` (REQUIRED), a vector of scalars (`BBSplusMessage`) representing the messages, it could be an empty vector.
/// * `generators` (REQUIRED), vector of pseudo-random points in G1.
/// * `header` (OPTIONAL), an octet string containing context and application specific information.
/// * `api_id` (OPTIONAL), an octet string. If not supplied it defaults to theempty octet string ("").
///
/// # Output:
/// * a result either [`Ok()`] or [`Error`]
pub(super) fn core_verify<CS>(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    messages: &[BBSplusMessage],
    generators: Generators,
    header: Option<&[u8]>,
    api_id: Option<&[u8]>,
) -> Result<(), Error>
where
    CS: BbsCiphersuite,
    CS::Expander: for<'a> ExpandMsg<'a>,
{
    let L = messages.len();

    if generators.values.len() != L + 1 {
        return Err(Error::NotEnoughGenerators);
    }

    let Q1 = generators.values[0];
    let H_points: &[G1Projective] = &generators.values[1..];

    let domain = calculate_domain::<CS>(pk, Q1, H_points, header, api_id)?;

    let mut B = generators.g1_base_point + Q1 * domain;

    for i in 0..L {
        B = B + H_points[i] * messages[i].value;
    }

    core_verify_precomputed_b(pk, signature, B)
}

fn core_verify_precomputed_b(
    pk: &BBSplusPublicKey,
    signature: &BBSplusSignature,
    B: G1Projective,
) -> Result<(), Error> {
    let BP2 = G2Projective::GENERATOR;
    let A2 = pk.0 + BP2 * signature.e;

    let identity_GT = Gt::IDENTITY;

    let term1 = (&signature.A.to_affine(), &G2Prepared::from(A2.to_affine()));
    let term2 = (&B.to_affine(), &G2Prepared::from(-BP2.to_affine()));

    let pairing = multi_miller_loop(&[term1, term2]).final_exponentiation();

    if pairing == identity_GT {
        Ok(())
    } else {
        Err(Error::SignatureVerificationError)
    }
}

fn core_verify_precomputed_b_batch(
    items: &[(&BBSplusSignature, &BBSplusPublicKey, G1Projective, Scalar)],
) -> Result<(), Error> {
    let BP2 = G2Projective::GENERATOR;
    let neg_bp2_prepared = G2Prepared::from(-BP2.to_affine());
    let prepared = if items.len() >= 4 {
        thread::scope(|scope| {
            items
                .iter()
                .map(|(signature, pk, B, coeff)| {
                    scope.spawn(move || {
                        let A2 = pk.0 + BP2 * signature.e;
                        (
                            signature.A * *coeff,
                            *B * *coeff,
                            G2Prepared::from(A2.to_affine()),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("signature verify worker panicked"))
                .collect::<Vec<_>>()
        })
    } else {
        items
            .iter()
            .map(|(signature, pk, B, coeff)| {
                let A2 = pk.0 + BP2 * signature.e;
                (
                    signature.A * *coeff,
                    *B * *coeff,
                    G2Prepared::from(A2.to_affine()),
                )
            })
            .collect::<Vec<_>>()
    };

    let mut terms = Vec::with_capacity(items.len() * 2);
    for (scaled_a, scaled_b, prepared_a2) in &prepared {
        terms.push((scaled_a.to_affine(), prepared_a2));
        terms.push((scaled_b.to_affine(), &neg_bp2_prepared));
    }

    let borrowed_terms = terms.iter().map(|(g1, g2)| (g1, *g2)).collect::<Vec<_>>();
    let pairing = multi_miller_loop(&borrowed_terms).final_exponentiation();

    if pairing == Gt::IDENTITY {
        Ok(())
    } else {
        Err(Error::SignatureVerificationError)
    }
}

#[cfg(test)]
mod tests {

    use crate::bbsplus::ciphersuites::BbsCiphersuite;
    use crate::keys::pair::KeyPair;
    use crate::schemes::algorithms::Scheme;
    use crate::schemes::algorithms::{BbsBls12381Sha256, BbsBls12381Shake256};
    use crate::{
        bbsplus::keys::{BBSplusPublicKey, BBSplusSecretKey},
        schemes::{algorithms::BBSplus, generics::Signature},
    };
    use elliptic_curve::hash2curve::ExpandMsg;
    use std::fs;

    //MSG SIGNATURE
    #[test]
    fn msg_signature_sha256_1() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature001.json",
        );
    }
    #[test]
    fn msg_signature_sha256_2() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature002.json",
        );
    }
    #[test]
    fn msg_signature_sha256_3() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature003.json",
        );
    }
    #[test]
    fn msg_signature_sha256_4() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature004.json",
        );
    }
    #[test]
    fn msg_signature_sha256_5() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature005.json",
        );
    }
    #[test]
    fn msg_signature_sha256_6() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature006.json",
        );
    }
    #[test]
    fn msg_signature_sha256_7() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature007.json",
        );
    }
    #[test]
    fn msg_signature_sha256_8() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature008.json",
        );
    }
    #[test]
    fn msg_signature_sha256_9() {
        msg_signature::<BbsBls12381Sha256>(
            "./fixture_data/bls12-381-sha-256/",
            "signature/signature009.json",
        );
    }

    //MSG SIGNATURE - SHAKE256
    #[test]
    fn msg_signature_shake256_1() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature001.json",
        );
    }
    #[test]
    fn msg_signature_shake256_2() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature002.json",
        );
    }
    #[test]
    fn msg_signature_shake256_3() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature003.json",
        );
    }
    #[test]
    fn msg_signature_shake256_4() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature004.json",
        );
    }
    #[test]
    fn msg_signature_shake256_5() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature005.json",
        );
    }
    #[test]
    fn msg_signature_shake256_6() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature006.json",
        );
    }
    #[test]
    fn msg_signature_shake256_7() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature007.json",
        );
    }
    #[test]
    fn msg_signature_shake256_8() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature008.json",
        );
    }
    #[test]
    fn msg_signature_shake256_9() {
        msg_signature::<BbsBls12381Shake256>(
            "./fixture_data/bls12-381-shake-256/",
            "signature/signature009.json",
        );
    }

    //Update Signature - SHA256
    #[test]
    fn update_signature_sha256() {
        update_signature::<BbsBls12381Sha256>();
    }

    //Update Blinded Signature - SHAKE256
    #[test]
    fn update_signature_shake256() {
        update_signature::<BbsBls12381Shake256>();
    }

    fn msg_signature<S: Scheme>(pathname: &str, filename: &str)
    where
        S::Ciphersuite: BbsCiphersuite,
        <S::Ciphersuite as BbsCiphersuite>::Expander: for<'a> ExpandMsg<'a>,
    {
        let data = fs::read_to_string([pathname, filename].concat()).expect("Unable to read file");
        let res: serde_json::Value = serde_json::from_str(&data).expect("Unable to parse");
        eprintln!("{}", res["caseName"]);

        let header_hex = res["header"].as_str().unwrap();
        let msgs_hex: Vec<String> = res["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::from_value(m.clone()).unwrap())
            .collect();
        let SK_hex = res["signerKeyPair"]["secretKey"].as_str().unwrap();
        let PK_hex = res["signerKeyPair"]["publicKey"].as_str().unwrap();
        let SIGNATURE_expected = res["signature"].as_str().unwrap();
        let RESULT_expected = res["result"]["valid"].as_bool().unwrap();

        let header = hex::decode(header_hex).unwrap();
        let SK = BBSplusSecretKey::from_bytes(&hex::decode(SK_hex).unwrap()).unwrap();
        let PK = BBSplusPublicKey::from_bytes(&hex::decode(PK_hex).unwrap()).unwrap();

        let messages: Vec<Vec<u8>> = msgs_hex.iter().map(|m| hex::decode(m).unwrap()).collect();

        let signature =
            Signature::<BBSplus<S::Ciphersuite>>::sign(Some(&messages), &SK, &PK, Some(&header))
                .unwrap();

        let result0 = hex::encode(signature.to_bytes()) == SIGNATURE_expected;

        let result1 = result0 == RESULT_expected;
        if !result1 {
            eprintln!("  SIGN: {}", result1);
            eprintln!("  Expected: {}", SIGNATURE_expected);
            eprintln!("  Computed: {}", hex::encode(signature.to_bytes()));
            assert!(result1, "failed")
        }

        //Verify the signature

        let signature_expected = Signature::<BBSplus<S::Ciphersuite>>::from_bytes(
            &hex::decode(SIGNATURE_expected).unwrap().try_into().unwrap(),
        )
        .unwrap();
        let result2 = signature_expected
            .verify(&PK, Some(&messages), Some(&header))
            .is_ok();
        let result3 = result2 == RESULT_expected;

        if !result3 {
            eprintln!("  VERIFY: {}", result3);
            eprintln!("  Expected: {}", RESULT_expected);
            eprintln!("  Computed: {}", result2);
            assert!(result3, "failed");
        } else {
            eprintln!("  SIGN: {}", result1);
            eprintln!("  Expected: {}", SIGNATURE_expected);
            eprintln!("  Computed: {}", hex::encode(signature.to_bytes()));

            eprintln!("  VERIFY: {}", result3);
            eprintln!("  Expected: {}", RESULT_expected);
            eprintln!("  Computed: {}", result2);
            if RESULT_expected == false {
                eprintln!(
                    "{} ({})",
                    result3,
                    res["result"]["reason"].as_str().unwrap()
                );
            }
        }
    }

    fn update_signature<S: Scheme>()
    where
        S::Ciphersuite: BbsCiphersuite,
        <S::Ciphersuite as BbsCiphersuite>::Expander: for<'a> ExpandMsg<'a>,
    {
        const IKM: &str = "746869732d49532d6a7573742d616e2d546573742d494b4d2d746f2d67656e65726174652d246528724074232d6b6579";
        const KEY_INFO: &str = "746869732d49532d736f6d652d6b65792d6d657461646174612d746f2d62652d757365642d696e2d746573742d6b65792d67656e";
        const msgs: [&str; 3] = [
            "9872ad089e452c7b6e283dfac2a80d58e8d0ff71cc4d5e310a1debdda4a45f02",
            "87a8bd656d49ee07b8110e1d8fd4f1dcef6fb9bc368c492d9bc8c4f98a739ac6",
            "96012096adda3f13dd4adbe4eea481a4c4b5717932b73b00e31807d3c5894b90",
        ];
        const header_hex: &str = "11223344556677889900aabbccddeeff";
        let header = hex::decode(header_hex).unwrap();

        let keypair = KeyPair::<BBSplus<S::Ciphersuite>>::generate(
            &hex::decode(&IKM).unwrap(),
            Some(&hex::decode(&KEY_INFO).unwrap()),
            None,
        )
        .unwrap();

        let sk = keypair.private_key();
        let pk = keypair.public_key();

        let messages: Vec<Vec<u8>> = msgs.iter().map(|m| hex::decode(m).unwrap()).collect();

        let signature =
            Signature::<BBSplus<S::Ciphersuite>>::sign(Some(&messages), sk, pk, Some(&header))
                .unwrap();
        let verify = signature.verify(pk, Some(&messages), Some(&header)).is_ok();

        assert!(verify, "Signature NOT VALID!");

        const new_message: &str =
            "8872ad089e452c7b6e283dfac2a80d58e8d0ff71cc4d5e310a1debdda4a45f02";
        const update_index: usize = 0usize;

        let new_message_bytes = hex::decode(new_message).unwrap();
        let old_message_bytes = messages.get(update_index).unwrap();

        let updated_signature = signature
            .update_signature(
                sk,
                &old_message_bytes,
                &new_message_bytes,
                update_index,
                messages.len(),
            )
            .unwrap();

        let mut new_messages = messages.clone();
        new_messages[update_index] = new_message_bytes;

        let verify = updated_signature
            .verify(pk, Some(&new_messages), Some(&header))
            .is_ok();

        assert!(verify, "Signature NOT VALID!");

        const new_message_wrong: &str =
            "9872ad089e452c7b6e283dfac2a80d58e8d0ff71cc4d5e310a1debdda4a45f01";
        let new_message_bytes_wrong = hex::decode(new_message_wrong).unwrap();
        let mut new_messages_wrong = messages.clone();
        new_messages_wrong[update_index] = new_message_bytes_wrong;

        let verify = updated_signature
            .verify(pk, Some(&new_messages_wrong), Some(&header))
            .is_ok();

        assert!(!verify, "Signature MUST BE NOT VALID!");
    }
}
