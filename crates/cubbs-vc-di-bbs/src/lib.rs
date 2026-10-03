//! cuBBS-Cred's own W3C `bbs-2023` Data Integrity core.
//!
//! This crate owns the W3C-facing BBS-2023 algorithm glue. It intentionally
//! does not depend on SpruceID's W3C layer or on `ssi-bbs`; the only BBS math
//! backend currently used is the cuBBS-Cred protocol facade in `cubbs-bbs`.
//!
//! The public functions in this first layer operate on already-canonicalized
//! N-Quads and grouped indexes. A JSON-LD/RDFC adapter can feed those prepared
//! inputs without changing the BBS primitive or future GPU/batch backends.

mod batch;
mod canonical;
mod hashing;
mod hmac_label;
mod jsonld;
mod key;
mod prepared;
mod proof_config;
mod select;
mod suite;

pub use batch::{
    acquire_w3c_verify_crypto_admission_for_backend, assemble_jsonld_disclosure_proof,
    derive_jsonld_disclosure_proof_batch, issue_jsonld_base_proof_batch, jsonld_batch_workers,
    prepare_jsonld_base_issue, prepare_jsonld_base_verification, prepare_jsonld_disclosure_proof,
    prepare_jsonld_disclosure_verification, prepared_cache_snapshot,
    verify_jsonld_base_proof_batch, verify_jsonld_disclosure_proof_batch,
    w3c_batch_timing_snapshot, w3c_verify_crypto_admission_snapshot,
    w3c_verify_crypto_admission_snapshot_for_backend, BatchItemResult, BatchRunResult,
    DeriveJsonLdDisclosureProofBatchRequest, IssueJsonLdBaseProofBatchRequest,
    PreparedCacheSnapshot, PreparedJsonLdBaseIssue, PreparedJsonLdDisclosureProof,
    VerifyJsonLdBaseProofBatchRequest, VerifyJsonLdDisclosureProofBatchRequest,
    W3cBatchTimingSnapshot, W3cVerifyCryptoAdmissionSnapshot, W3cVerifyCryptoPermit,
};
pub use canonical::{
    canonical_label_map_from_input_map, canonicalize_nquads, relabel_and_sort_canonical_nquads,
    replace_blank_node_labels, split_canonical_nquads, CanonicalizationError,
    CanonicalizationResult, CanonicalizedNQuads,
};
pub use hashing::{
    bbs_header, hash_canonical_proof, hash_joined_strings, hash_mandatory, Sha256Digest,
    BBS2023_HEADER_LEN,
};
pub use hmac_label::{hmac_sha256, shuffled_bnode_label_map, HMAC_KEY_LEN};
pub use jsonld::{
    canonicalization_cache_snapshot, canonicalize_and_group_jsonld, canonicalize_jsonld,
    json_content_digest, jsonld_to_nquads, jsonld_to_nquads_using,
    label_replacement_canonicalize_jsonld, shared_jsonld_loader_enabled,
    CanonicalizationCacheSnapshot, CanonicalizedJsonLdGroup, CanonicalizedJsonLdGroups,
    JsonLdCanonicalizationError, LabelReplacementCanonicalizedJsonLd,
};
pub use key::{
    did_key_from_public_key, public_key_from_did_key, public_key_from_multibase,
    public_key_from_verification_method, public_key_to_multibase, verification_method_document,
    verification_method_from_public_key, KeyEncodingError, MULTIBASE_BASE58BTC_PREFIX,
    MULTICODEC_G2_PUBLIC_KEY_HEADER,
};
pub use prepared::{
    complete_disclosure_proof, derive_disclosure_proof, derive_disclosure_proof_batch,
    group_canonical_nquads, issue_base_proof, issue_base_proof_batch, lower_base_verification,
    lower_disclosure_proof, lower_disclosure_verification, mandatory_indexes_in_revealed_document,
    selective_indexes_in_bbs_messages, verify_base_proof, verify_base_proof_batch,
    verify_disclosure_proof, verify_disclosure_proof_batch, Bbs2023Core, DerivedProofMaterial,
    DisclosureGroups, Error, GroupedNQuads, IssueBaseProofBatchRequest, IssueBaseProofRequest,
    LoweredDisclosureProof, PreparedBaseVerification, PreparedBaseVerificationBatchRequest,
    PreparedBatchItemResult, PreparedBatchRunResult, PreparedDisclosureBatchRequest,
    PreparedDisclosureRequest, PreparedDisclosureVerification,
    PreparedDisclosureVerificationBatchRequest, Result,
};
pub use proof_config::{
    build_proof_configuration, default_bbs_2023_proof_options, normalize_bbs_2023_proof_options,
    ProofConfigError, BBS_2023_CRYPTOSUITE, DATA_INTEGRITY_PROOF_TYPE,
};
pub use select::{parse_json_pointer, select_json, JsonPointerToken, SelectionError};
pub use suite::{
    derive_jsonld_disclosure_proof, issue_jsonld_base_proof, verify_jsonld_base_proof,
    verify_jsonld_disclosure_proof, DeriveJsonLdDisclosureProofRequest,
    IssueJsonLdBaseProofRequest, SuiteJsonLdError,
};
