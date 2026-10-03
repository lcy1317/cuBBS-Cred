use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{SystemTime, UNIX_EPOCH},
};

use bls12_381_plus::{group::Curve, Scalar};
use cubbs_bbs::{
    BbsBackend, BbsEngine, Error as BackendError, ProofGenRequest, ProofVerifyRequest, SignRequest,
    VerifyRequest,
};
use cubbs_bbs_protocol::{
    bbsplus::{
        ciphersuites::{BbsCiphersuite as _, Bls12381Sha256},
        generators::Generators,
        keys::{BBSplusPublicKey, BBSplusSecretKey},
    },
    keys::pair::KeyPair as ZkKeyPair,
    schemes::{
        algorithms::BBSplus,
        generics::{PoKSignature, Signature as ZkSignature},
    },
    utils::{
        message::bbsplus_message::BBSplusMessage,
        util::bbsplus_utils::{hash_to_scalar, ScalarExt},
    },
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const UPSTREAM_REPOSITORY: &str = "https://github.com/decentralized-identity/bbs-signature";
const UPSTREAM_COMMIT: &str = "a6a445dd3c2a382fb8affa08a4bc077bd27bf753";
const CIPHERSUITE: &str = "BBS_BLS12381G1_XMD:SHA-256_SSWU_RO_";

type AnyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Passed,
    Failed,
    Blocked,
}

#[derive(Debug, Serialize)]
struct Check {
    category: &'static str,
    name: String,
    status: CheckStatus,
    expected: String,
    actual: String,
    detail: String,
}

#[derive(Debug, Serialize)]
struct Suite {
    name: String,
    backend_info: Option<BackendInfo>,
    passed: usize,
    failed: usize,
    blocked: usize,
    checks: Vec<Check>,
}

#[derive(Debug, Serialize)]
struct BackendInfo {
    requested: &'static str,
    active: &'static str,
    hardware_acceleration: bool,
    fallback: bool,
    detail: &'static str,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    generated_unix_seconds: u64,
    upstream_repository: &'static str,
    upstream_commit: &'static str,
    ciphersuite: &'static str,
    vector_root: String,
    provider: Suite,
    backends: Vec<Suite>,
    notes: Vec<&'static str>,
}

#[derive(Debug)]
struct Options {
    vector_root: PathBuf,
    output: PathBuf,
    backend: String,
    strict: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(strict_failure) => {
            if strict_failure {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("cfrg-ietf-conformance: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> AnyResult<bool> {
    let options = parse_options()?;
    let sha_root = options.vector_root.join("bls12-381-sha-256");
    require_file(&sha_root.join("keypair.json"))?;
    require_file(&sha_root.join("signature/signature001.json"))?;
    require_file(&sha_root.join("proof/proof001.json"))?;

    let provider_checks = run_provider_checks(&sha_root)?;
    let provider = suite("cubbs-bbs-protocol-direct", None, provider_checks);

    let mut backends = Vec::new();
    if options.backend == "cpu" || options.backend == "both" {
        backends.push(run_backend_checks(
            "cubbs-bbs-cpu",
            BbsBackend::ZkryptiumCpu,
            &sha_root,
        )?);
    }
    if options.backend == "gpu" || options.backend == "both" {
        backends.push(run_backend_checks(
            "cubbs-bbs-gpu-requested",
            BbsBackend::GpuRequestedCpuFallback,
            &sha_root,
        )?);
    }

    let report = Report {
        schema_version: 1,
        generated_unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        upstream_repository: UPSTREAM_REPOSITORY,
        upstream_commit: UPSTREAM_COMMIT,
        ciphersuite: CIPHERSUITE,
        vector_root: options.vector_root.display().to_string(),
        provider,
        backends,
        notes: vec![
            "The direct provider suite exercises the cuBBS-Cred protocol crate independently of the public engine facade; both paths accept the arbitrary octet-string header required by the CFRG BBS protocol.",
            "The direct provider suite bypasses the cuBBS-Cred 64-byte bbs-2023 header guard to separate cryptographic-version drift from facade restrictions.",
            "BBS Sign output is randomized; generated-signature checks use Sign -> Verify, while fixed CFRG signature bytes are verified independently.",
            "ProofGen output is randomized in production; its provider check is a generate-then-verify conformance check, not an exact-byte comparison.",
            "A blocked facade check is not counted as a cryptographic pass, including negative vectors rejected before cryptographic processing.",
            "SHAKE-256 fixtures are downloaded but not executed because the cuBBS-Cred backend currently exposes only BLS12-381-SHA-256.",
        ],
    };

    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.output, serde_json::to_vec_pretty(&report)?)?;

    print_suite(&report.provider);
    for backend in &report.backends {
        print_suite(backend);
    }
    println!("report={}", options.output.display());

    let has_failure = report.provider.failed > 0
        || report.backends.iter().any(|suite| suite.failed > 0)
        || report.backends.iter().any(|suite| suite.blocked > 0);
    Ok(options.strict && has_failure)
}

fn parse_options() -> AnyResult<Options> {
    let mut vector_root = PathBuf::from("CFRG-IETF-BBS/fixture_data");
    let mut output = PathBuf::from("CFRG-IETF-BBS/results/latest.json");
    let mut backend = String::from("both");
    let mut strict = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--vectors" => {
                vector_root = PathBuf::from(args.next().ok_or("--vectors requires a path")?)
            }
            "--output" => output = PathBuf::from(args.next().ok_or("--output requires a path")?),
            "--backend" => backend = args.next().ok_or("--backend requires cpu, gpu, or both")?,
            "--strict" => strict = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p cubbs-bench --bin cfrg-ietf-conformance -- \
                     [--vectors PATH] [--output PATH] [--backend cpu|gpu|both] [--strict]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if !matches!(backend.as_str(), "cpu" | "gpu" | "both") {
        return Err("--backend must be cpu, gpu, or both".into());
    }
    Ok(Options {
        vector_root,
        output,
        backend,
        strict,
    })
}

fn run_provider_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let mut checks = Vec::new();
    checks.push(provider_keygen_check(root)?);
    checks.push(provider_hash_to_scalar_check(root)?);
    checks.extend(provider_map_message_checks(root)?);
    checks.extend(provider_generator_checks(root)?);
    checks.extend(provider_mocked_rng_checks(root)?);
    checks.extend(provider_signature_checks(root)?);
    checks.extend(provider_proof_checks(root)?);
    checks.extend(provider_proof_gen_checks(root)?);
    Ok(checks)
}

fn provider_keygen_check(root: &Path) -> AnyResult<Check> {
    let vector = read_json(&root.join("keypair.json"))?;
    let key_material = hex_field(&vector, "keyMaterial")?;
    let key_info = hex_field(&vector, "keyInfo")?;
    let key_dst = hex_field(&vector, "keyDst")?;
    let pair = ZkKeyPair::<BBSplus<Bls12381Sha256>>::generate(
        &key_material,
        Some(&key_info),
        Some(&key_dst),
    )?;
    let (secret_key, public_key) = pair.into_parts();
    let expected_sk = string_field(&vector["keyPair"], "secretKey")?;
    let expected_pk = string_field(&vector["keyPair"], "publicKey")?;
    let actual_sk = hex_encode(&secret_key.to_bytes());
    let actual_pk = hex_encode(&public_key.to_bytes());
    Ok(pass_fail(
        "keygen",
        "deterministic key pair",
        actual_sk == expected_sk && actual_pk == expected_pk,
        format!("sk={expected_sk};pk={expected_pk}"),
        format!("sk={actual_sk};pk={actual_pk}"),
        "CFRG deterministic KeyGen",
    ))
}

fn provider_hash_to_scalar_check(root: &Path) -> AnyResult<Check> {
    let vector = read_json(&root.join("h2s.json"))?;
    let message = hex_field(&vector, "message")?;
    let dst = hex_field(&vector, "dst")?;
    let expected = string_field(&vector, "scalar")?;
    let actual = hex_encode(&hash_to_scalar::<Bls12381Sha256>(&message, &dst)?.to_bytes_be());
    Ok(pass_fail(
        "hash_to_scalar",
        "h2s.json",
        actual == expected,
        expected,
        actual,
        "CFRG hash-to-scalar primitive",
    ))
}

fn provider_map_message_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let vector = read_json(&root.join("MapMessageToScalarAsHash.json"))?;
    let cases = array_field(&vector, "cases")?;
    let messages = cases
        .iter()
        .map(|case| hex_field(case, "message"))
        .collect::<AnyResult<Vec<_>>>()?;
    let scalars =
        BBSplusMessage::messages_to_scalar::<Bls12381Sha256>(&messages, Bls12381Sha256::API_ID)?;
    cases
        .iter()
        .zip(scalars.iter())
        .enumerate()
        .map(|(index, (case, scalar))| {
            let expected = string_field(case, "scalar")?;
            let actual = hex_encode(&scalar.to_bytes_be());
            Ok(pass_fail(
                "map_message_to_scalar",
                format!("message[{index}]"),
                actual == expected,
                expected,
                actual,
                "CFRG message-to-scalar mapping",
            ))
        })
        .collect()
}

fn provider_generator_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let vector = read_json(&root.join("generators.json"))?;
    let expected_messages = array_field(&vector, "MsgGenerators")?
        .iter()
        .map(|value| value.as_str().ok_or("MsgGenerators must contain strings"))
        .collect::<Result<Vec<_>, _>>()?;
    let generators = Generators::create::<Bls12381Sha256>(
        expected_messages.len() + 1,
        Some(Bls12381Sha256::API_ID),
    );
    let mut checks = Vec::new();
    let expected_p1 = string_field(&vector, "P1")?;
    let actual_p1 = hex_encode(&generators.g1_base_point.to_affine().to_compressed());
    checks.push(pass_fail(
        "generators",
        "P1",
        actual_p1 == expected_p1,
        expected_p1,
        actual_p1,
        "fixed base point",
    ));
    let expected_q1 = string_field(&vector, "Q1")?;
    let actual_q1 = hex_encode(&generators.values[0].to_affine().to_compressed());
    checks.push(pass_fail(
        "generators",
        "Q1",
        actual_q1 == expected_q1,
        expected_q1,
        actual_q1,
        "domain generator",
    ));
    for (index, expected) in expected_messages.iter().enumerate() {
        let actual = hex_encode(&generators.values[index + 1].to_affine().to_compressed());
        checks.push(pass_fail(
            "generators",
            format!("H[{index}]"),
            actual == *expected,
            (*expected).to_owned(),
            actual,
            "message generator",
        ));
    }
    Ok(checks)
}

fn provider_mocked_rng_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let vector = read_json(&root.join("mockedRng.json"))?;
    let seed = hex_field(&vector, "seed")?;
    let dst = hex_field(&vector, "dst")?;
    let count = vector["count"].as_u64().ok_or("count must be an integer")? as usize;
    let expected = array_field(&vector, "mockedScalars")?
        .iter()
        .map(|value| value.as_str().ok_or("mockedScalars must contain strings"))
        .collect::<Result<Vec<_>, _>>()?;
    let actual = mocked_scalars_sha256(count, &seed, &dst)?;
    expected
        .iter()
        .zip(actual.iter())
        .enumerate()
        .map(|(index, (expected, actual))| {
            let actual = hex_encode(&actual.to_bytes_be());
            Ok(pass_fail(
                "mocked_rng",
                format!("scalar[{index}]"),
                actual == *expected,
                (*expected).to_owned(),
                actual,
                "CFRG deterministic proof-test RNG",
            ))
        })
        .collect()
}

fn provider_signature_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let mut checks = Vec::new();
    for path in json_files(&root.join("signature"))? {
        let vector = read_json(&path)?;
        let name = string_field(&vector, "caseName")?;
        let expected_valid = bool_field(&vector["result"], "valid")?;
        let messages = message_field(&vector)?;
        let header = hex_field(&vector, "header")?;
        let signature_bytes = hex_field(&vector, "signature")?;
        let public_key =
            BBSplusPublicKey::from_bytes(&hex_field(&vector["signerKeyPair"], "publicKey")?)?;
        let actual_valid = parse_signature(&signature_bytes)
            .map(|signature| {
                signature
                    .verify(&public_key, Some(&messages), Some(&header))
                    .is_ok()
            })
            .unwrap_or(false);
        checks.push(pass_fail(
            "signature_verify",
            name.clone(),
            actual_valid == expected_valid,
            expected_valid.to_string(),
            actual_valid.to_string(),
            path.display().to_string(),
        ));

        if expected_valid {
            let secret_key =
                BBSplusSecretKey::from_bytes(&hex_field(&vector["signerKeyPair"], "secretKey")?)?;
            let generated = ZkSignature::<BBSplus<Bls12381Sha256>>::sign(
                Some(&messages),
                &secret_key,
                &public_key,
                Some(&header),
            )?;
            let actual_valid = generated
                .verify(&public_key, Some(&messages), Some(&header))
                .is_ok();
            checks.push(pass_fail(
                "signature_generate_roundtrip",
                name,
                actual_valid,
                "true",
                actual_valid.to_string(),
                "BBS signatures are randomized; fixed CFRG bytes are checked separately by signature_verify",
            ));
        }
    }
    Ok(checks)
}

fn provider_proof_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let mut checks = Vec::new();
    for path in json_files(&root.join("proof"))? {
        let vector = read_json(&path)?;
        let name = string_field(&vector, "caseName")?;
        let expected_valid = bool_field(&vector["result"], "valid")?;
        let messages = message_field(&vector)?;
        let indexes = index_field(&vector)?;
        let disclosed = disclosed_messages(&messages, &indexes);
        let header = hex_field(&vector, "header")?;
        let presentation_header = hex_field(&vector, "presentationHeader")?;
        let public_key = BBSplusPublicKey::from_bytes(&hex_field(&vector, "signerPublicKey")?)?;
        let proof_bytes = hex_field(&vector, "proof")?;
        let actual_valid = PoKSignature::<BBSplus<Bls12381Sha256>>::from_bytes(&proof_bytes)
            .ok()
            .and_then(|proof| {
                disclosed.as_ref().ok().map(|disclosed| {
                    proof
                        .proof_verify(
                            &public_key,
                            Some(disclosed),
                            Some(&indexes),
                            Some(&header),
                            Some(&presentation_header),
                        )
                        .is_ok()
                })
            })
            .unwrap_or(false);
        checks.push(pass_fail(
            "proof_verify",
            name,
            actual_valid == expected_valid,
            expected_valid.to_string(),
            actual_valid.to_string(),
            path.display().to_string(),
        ));
    }
    Ok(checks)
}

fn provider_proof_gen_checks(root: &Path) -> AnyResult<Vec<Check>> {
    let key_vector = read_json(&root.join("keypair.json"))?;
    let secret_key =
        BBSplusSecretKey::from_bytes(&hex_field(&key_vector["keyPair"], "secretKey")?)?;
    let public_key =
        BBSplusPublicKey::from_bytes(&hex_field(&key_vector["keyPair"], "publicKey")?)?;
    let mut checks = Vec::new();
    for path in json_files(&root.join("proof"))? {
        let vector = read_json(&path)?;
        if !bool_field(&vector["result"], "valid")? {
            continue;
        }
        let name = string_field(&vector, "caseName")?;
        let messages = message_field(&vector)?;
        let indexes = index_field(&vector)?;
        let disclosed = disclosed_messages(&messages, &indexes)?;
        let header = hex_field(&vector, "header")?;
        let presentation_header = hex_field(&vector, "presentationHeader")?;
        let signature = ZkSignature::<BBSplus<Bls12381Sha256>>::sign(
            Some(&messages),
            &secret_key,
            &public_key,
            Some(&header),
        )?;
        let proof = PoKSignature::<BBSplus<Bls12381Sha256>>::proof_gen(
            &public_key,
            &signature.to_bytes(),
            Some(&header),
            Some(&presentation_header),
            Some(&messages),
            Some(&indexes),
        )?;
        let valid = proof
            .proof_verify(
                &public_key,
                Some(&disclosed),
                Some(&indexes),
                Some(&header),
                Some(&presentation_header),
            )
            .is_ok();
        checks.push(pass_fail(
            "proof_generate_roundtrip",
            name,
            valid,
            "true",
            valid.to_string(),
            "production randomized ProofGen followed by ProofVerify",
        ));
    }
    Ok(checks)
}

fn run_backend_checks(name: &str, backend: BbsBackend, root: &Path) -> AnyResult<Suite> {
    let engine = BbsEngine::new(backend);
    let info = engine.backend_info();
    let backend_info = BackendInfo {
        requested: info.requested,
        active: info.active,
        hardware_acceleration: info.hardware_acceleration,
        fallback: info.fallback,
        detail: info.detail,
    };
    let mut checks = Vec::new();

    let key_vector = read_json(&root.join("keypair.json"))?;
    let secret_key = hex_field(&key_vector["keyPair"], "secretKey")?;
    let expected_public = string_field(&key_vector["keyPair"], "publicKey")?;
    let actual_public = engine.public_key_from_secret(&secret_key)?;
    checks.push(pass_fail(
        "backend_public_key",
        "public key from CFRG secret key",
        hex_encode(&actual_public) == expected_public,
        expected_public,
        hex_encode(&actual_public),
        "cuBBS-Cred public facade",
    ));

    let map_vector = read_json(&root.join("MapMessageToScalarAsHash.json"))?;
    let map_cases = array_field(&map_vector, "cases")?;
    let messages = map_cases
        .iter()
        .map(|case| hex_field(case, "message"))
        .collect::<AnyResult<Vec<_>>>()?;
    let actual_scalars = engine.messages_to_scalar_bytes(&messages)?;
    for (index, (case, actual)) in map_cases.iter().zip(actual_scalars.iter()).enumerate() {
        let expected = string_field(case, "scalar")?;
        let actual = hex_encode(actual);
        checks.push(pass_fail(
            "backend_map_message_to_scalar",
            format!("message[{index}]"),
            actual == expected,
            expected,
            actual,
            "cuBBS-Cred public facade",
        ));
    }

    for path in json_files(&root.join("signature"))? {
        let vector = read_json(&path)?;
        let case_name = string_field(&vector, "caseName")?;
        let expected_valid = bool_field(&vector["result"], "valid")?;
        let request = VerifyRequest {
            public_key: hex_field(&vector["signerKeyPair"], "publicKey")?,
            signature: hex_field(&vector, "signature")?,
            header: hex_field(&vector, "header")?,
            messages: message_field(&vector)?,
        };
        checks.push(backend_boolean_check(
            "backend_signature_verify",
            case_name.clone(),
            expected_valid,
            engine.verify(request),
            path.display().to_string(),
        ));

        if expected_valid {
            let public_key = hex_field(&vector["signerKeyPair"], "publicKey")?;
            let header = hex_field(&vector, "header")?;
            let messages = message_field(&vector)?;
            let request = SignRequest {
                secret_key: hex_field(&vector["signerKeyPair"], "secretKey")?,
                public_key: public_key.clone(),
                header: header.clone(),
                messages: messages.clone(),
            };
            let result = engine.sign(request).and_then(|signature| {
                engine.verify(VerifyRequest {
                    public_key,
                    signature,
                    header,
                    messages,
                })
            });
            checks.push(backend_boolean_check(
                "backend_signature_generate_roundtrip",
                case_name,
                true,
                result,
                "cuBBS-Cred randomized Sign -> Verify; fixed CFRG bytes are checked separately by backend_signature_verify",
            ));
        }
    }

    for path in json_files(&root.join("proof"))? {
        let vector = read_json(&path)?;
        let case_name = string_field(&vector, "caseName")?;
        let expected_valid = bool_field(&vector["result"], "valid")?;
        let messages = message_field(&vector)?;
        let indexes = index_field(&vector)?;
        let disclosed = disclosed_messages(&messages, &indexes).unwrap_or_default();
        let request = ProofVerifyRequest {
            public_key: hex_field(&vector, "signerPublicKey")?,
            proof: hex_field(&vector, "proof")?,
            header: hex_field(&vector, "header")?,
            presentation_header: hex_field(&vector, "presentationHeader")?,
            disclosed_messages: disclosed,
            disclosed_indexes: indexes,
            total_message_count: messages.len(),
        };
        checks.push(backend_boolean_check(
            "backend_proof_verify",
            case_name,
            expected_valid,
            engine.proof_verify(request),
            path.display().to_string(),
        ));
    }

    for path in json_files(&root.join("proof"))? {
        let vector = read_json(&path)?;
        if !bool_field(&vector["result"], "valid")? {
            continue;
        }
        let case_name = string_field(&vector, "caseName")?;
        let messages = message_field(&vector)?;
        let indexes = index_field(&vector)?;
        let header = hex_field(&vector, "header")?;
        let presentation_header = hex_field(&vector, "presentationHeader")?;
        let public_key = hex_field(&key_vector["keyPair"], "publicKey")?;
        let sign_request = SignRequest {
            secret_key: secret_key.clone(),
            public_key: public_key.clone(),
            header: header.clone(),
            messages: messages.clone(),
        };
        let generated = engine.sign(sign_request).and_then(|signature| {
            engine.proof_gen(ProofGenRequest {
                public_key: public_key.clone(),
                signature,
                header: header.clone(),
                presentation_header: presentation_header.clone(),
                messages: messages.clone(),
                disclosed_indexes: indexes.clone(),
            })
        });
        match generated {
            Ok(proof) => {
                let verified = engine.proof_verify(ProofVerifyRequest {
                    public_key,
                    proof,
                    header,
                    presentation_header,
                    disclosed_messages: disclosed_messages(&messages, &indexes)?,
                    disclosed_indexes: indexes,
                    total_message_count: messages.len(),
                });
                checks.push(backend_boolean_check(
                    "backend_proof_generate_roundtrip",
                    case_name,
                    true,
                    verified,
                    "cuBBS-Cred Sign -> ProofGen -> ProofVerify",
                ));
            }
            Err(error) => checks.push(blocked_backend_error(
                "backend_proof_generate_roundtrip",
                case_name,
                "true",
                error,
                "cuBBS-Cred Sign -> ProofGen",
            )),
        }
    }

    Ok(suite(name, Some(backend_info), checks))
}

fn backend_boolean_check(
    category: &'static str,
    name: String,
    expected: bool,
    result: cubbs_bbs::Result<bool>,
    detail: impl Into<String>,
) -> Check {
    match result {
        Ok(actual) => pass_fail(
            category,
            name,
            actual == expected,
            expected.to_string(),
            actual.to_string(),
            detail,
        ),
        Err(error) => blocked_backend_error(category, name, expected.to_string(), error, detail),
    }
}

fn blocked_backend_error(
    category: &'static str,
    name: String,
    expected: impl Into<String>,
    error: BackendError,
    detail: impl Into<String>,
) -> Check {
    Check {
        category,
        name,
        status: CheckStatus::Blocked,
        expected: expected.into(),
        actual: format!("error:{error}"),
        detail: detail.into(),
    }
}

fn pass_fail(
    category: &'static str,
    name: impl Into<String>,
    passed: bool,
    expected: impl Into<String>,
    actual: impl Into<String>,
    detail: impl Into<String>,
) -> Check {
    Check {
        category,
        name: name.into(),
        status: if passed {
            CheckStatus::Passed
        } else {
            CheckStatus::Failed
        },
        expected: expected.into(),
        actual: actual.into(),
        detail: detail.into(),
    }
}

fn suite(name: impl Into<String>, backend_info: Option<BackendInfo>, checks: Vec<Check>) -> Suite {
    let passed = checks
        .iter()
        .filter(|check| matches!(check.status, CheckStatus::Passed))
        .count();
    let failed = checks
        .iter()
        .filter(|check| matches!(check.status, CheckStatus::Failed))
        .count();
    let blocked = checks
        .iter()
        .filter(|check| matches!(check.status, CheckStatus::Blocked))
        .count();
    Suite {
        name: name.into(),
        backend_info,
        passed,
        failed,
        blocked,
        checks,
    }
}

fn print_suite(suite: &Suite) {
    println!(
        "{}: passed={} failed={} blocked={}",
        suite.name, suite.passed, suite.failed, suite.blocked
    );
    if let Some(info) = &suite.backend_info {
        println!(
            "  active={} hardware={} fallback={} detail={}",
            info.active, info.hardware_acceleration, info.fallback, info.detail
        );
    }
    for check in suite
        .checks
        .iter()
        .filter(|check| !matches!(check.status, CheckStatus::Passed))
    {
        println!(
            "  {:?} {} :: {} (expected {}, actual {})",
            check.status, check.category, check.name, check.expected, check.actual
        );
    }
}

fn read_json(path: &Path) -> AnyResult<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn require_file(path: &Path) -> AnyResult<()> {
    if !path.is_file() {
        return Err(format!("missing fixture: {}", path.display()).into());
    }
    Ok(())
}

fn json_files(path: &Path) -> AnyResult<Vec<PathBuf>> {
    let mut files = fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == "json"));
    files.sort();
    Ok(files)
}

fn string_field(value: &Value, field: &str) -> AnyResult<String> {
    value[field]
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{field} must be a string").into())
}

fn bool_field(value: &Value, field: &str) -> AnyResult<bool> {
    value[field]
        .as_bool()
        .ok_or_else(|| format!("{field} must be a boolean").into())
}

fn array_field<'a>(value: &'a Value, field: &str) -> AnyResult<&'a Vec<Value>> {
    value[field]
        .as_array()
        .ok_or_else(|| format!("{field} must be an array").into())
}

fn hex_field(value: &Value, field: &str) -> AnyResult<Vec<u8>> {
    hex_decode(&string_field(value, field)?)
}

fn message_field(value: &Value) -> AnyResult<Vec<Vec<u8>>> {
    array_field(value, "messages")?
        .iter()
        .map(|message| {
            message
                .as_str()
                .ok_or_else(|| "messages must contain strings".into())
                .and_then(hex_decode)
        })
        .collect()
}

fn index_field(value: &Value) -> AnyResult<Vec<usize>> {
    array_field(value, "disclosedIndexes")?
        .iter()
        .map(|index| {
            index
                .as_u64()
                .map(|index| index as usize)
                .ok_or_else(|| "disclosedIndexes must contain integers".into())
        })
        .collect()
}

fn disclosed_messages(messages: &[Vec<u8>], indexes: &[usize]) -> AnyResult<Vec<Vec<u8>>> {
    indexes
        .iter()
        .map(|&index| {
            messages
                .get(index)
                .cloned()
                .ok_or_else(|| format!("disclosed index {index} is out of range").into())
        })
        .collect()
}

fn parse_signature(bytes: &[u8]) -> AnyResult<ZkSignature<BBSplus<Bls12381Sha256>>> {
    let bytes: &[u8; 80] = bytes
        .try_into()
        .map_err(|_| "signature must contain exactly 80 bytes")?;
    Ok(ZkSignature::<BBSplus<Bls12381Sha256>>::from_bytes(bytes)?)
}

fn hex_decode(value: &str) -> AnyResult<Vec<u8>> {
    if value.len() % 2 != 0 {
        return Err("hex string must have even length".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| Ok(u8::from_str_radix(&value[index..index + 2], 16)?))
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn mocked_scalars_sha256(count: usize, seed: &[u8], dst: &[u8]) -> AnyResult<Vec<Scalar>> {
    const EXPAND_LEN: usize = 48;
    let uniform = expand_message_xmd_sha256(seed, dst, EXPAND_LEN * count)?;
    uniform
        .chunks_exact(EXPAND_LEN)
        .map(|chunk| {
            let okm: &[u8; EXPAND_LEN] = chunk
                .try_into()
                .map_err(|_| "invalid mocked scalar chunk length")?;
            Ok(Scalar::from_okm(okm))
        })
        .collect()
}

fn expand_message_xmd_sha256(message: &[u8], dst: &[u8], output_len: usize) -> AnyResult<Vec<u8>> {
    const HASH_LEN: usize = 32;
    const BLOCK_LEN: usize = 64;
    if dst.len() > 255 || output_len > u16::MAX as usize {
        return Err("expand_message_xmd input is too large".into());
    }
    let ell = output_len.div_ceil(HASH_LEN);
    if ell == 0 || ell > 255 {
        return Err("expand_message_xmd ell is out of range".into());
    }

    let mut dst_prime = dst.to_vec();
    dst_prime.push(dst.len() as u8);
    let mut hasher = Sha256::new();
    hasher.update([0u8; BLOCK_LEN]);
    hasher.update(message);
    hasher.update((output_len as u16).to_be_bytes());
    hasher.update([0u8]);
    hasher.update(&dst_prime);
    let b0 = hasher.finalize();

    let mut hasher = Sha256::new();
    hasher.update(b0);
    hasher.update([1u8]);
    hasher.update(&dst_prime);
    let mut previous = hasher.finalize();
    let mut uniform = Vec::with_capacity(ell * HASH_LEN);
    uniform.extend_from_slice(&previous);

    for index in 2..=ell {
        let mut xor = [0u8; HASH_LEN];
        for byte in 0..HASH_LEN {
            xor[byte] = b0[byte] ^ previous[byte];
        }
        let mut hasher = Sha256::new();
        hasher.update(xor);
        hasher.update([index as u8]);
        hasher.update(&dst_prime);
        previous = hasher.finalize();
        uniform.extend_from_slice(&previous);
    }
    uniform.truncate(output_len);
    Ok(uniform)
}
