use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use hyper::{
    body::to_bytes,
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, StatusCode,
};
use serde_json::{json, Value as JsonValue};
use ssi::{
    claims::{
        data_integrity::{
            AnyDataIntegrity, AnySelectionOptions, AnySignatureOptions, AnySuite,
            CryptographicSuite, ProofOptions,
        },
        vc::v2::JsonCredential,
        SignatureEnvironment, VerificationParameters,
    },
    dids::{AnyDidMethod, DIDKey, VerificationMethodDIDResolver},
    json_ld::syntax,
    verification_methods::{AnyMethod, SingleSecretSigner},
    JWK,
};

struct State {
    jwk: JWK,
    operations: AtomicU64,
    failures: AtomicU64,
}

fn string_pointers(body: &JsonValue, field: &str) -> Result<Vec<ssi::JsonPointerBuf>, String> {
    body.get("options")
        .and_then(|options| options.get(field))
        .and_then(JsonValue::as_array)
        .into_iter()
        .flatten()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{field} entry is not a string"))?
                .parse()
                .map_err(|error| format!("invalid {field} entry: {error}"))
        })
        .collect()
}

async fn issue_one(state: &State, body: &JsonValue) -> Result<JsonValue, String> {
    let did_url = DIDKey::generate_url(&state.jwk).map_err(|error| error.to_string())?;
    let resolver = VerificationMethodDIDResolver::<_, AnyMethod>::new(AnyDidMethod::default());
    let credential: JsonCredential = serde_json::from_value(
        body.get("credential")
            .cloned()
            .ok_or_else(|| "missing credential".to_owned())?,
    )
    .map_err(|error| error.to_string())?;
    let mut signature_options = AnySignatureOptions::default();
    signature_options.mandatory_pointers = string_pointers(body, "mandatoryPointers")?;
    let signed = AnySuite::Bbs2023
        .sign_with(
            SignatureEnvironment::default(),
            credential,
            &resolver,
            SingleSecretSigner::new(state.jwk.clone()).into_local(),
            ProofOptions::from_method(did_url.into_iri().into()),
            signature_options,
        )
        .await
        .map_err(|error| error.to_string())?;
    serde_json::to_value(signed).map_err(|error| error.to_string())
}

async fn derive_one(body: &JsonValue) -> Result<JsonValue, String> {
    let resolver = VerificationMethodDIDResolver::<_, AnyMethod>::new(AnyDidMethod::default());
    let base: AnyDataIntegrity<JsonCredential> = serde_json::from_value(
        body.get("verifiableCredential")
            .cloned()
            .ok_or_else(|| "missing verifiableCredential".to_owned())?,
    )
    .map_err(|error| error.to_string())?;
    let params = VerificationParameters::from_resolver(&resolver);
    let mut selection = AnySelectionOptions::default();
    selection.selective_pointers = string_pointers(body, "selectivePointers")?;
    selection.presentation_header = body
        .get("options")
        .and_then(|options| options.get("presentationHeader"))
        .and_then(JsonValue::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| "invalid presentationHeader byte".to_owned())
                })
                .collect()
        })
        .transpose()?;
    let derived = base
        .select(&params, selection)
        .await
        .map_err(|error| error.to_string())?
        .map(|object| syntax::from_value::<JsonCredential>(syntax::Value::Object(object)).unwrap());
    serde_json::to_value(derived).map_err(|error| error.to_string())
}

async fn verify_derived_one(body: &JsonValue) -> Result<JsonValue, String> {
    let resolver = VerificationMethodDIDResolver::<_, AnyMethod>::new(AnyDidMethod::default());
    let document: AnyDataIntegrity<JsonCredential> = serde_json::from_value(
        body.get("verifiableCredential")
            .cloned()
            .ok_or_else(|| "missing verifiableCredential".to_owned())?,
    )
    .map_err(|error| error.to_string())?;
    document
        .verify(VerificationParameters::from_resolver(&resolver))
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())?;
    Ok(json!({"verified": true}))
}

fn json_response(status: StatusCode, body: JsonValue) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn handle(request: Request<Body>, state: Arc<State>) -> Result<Response<Body>, Infallible> {
    if request.method() == Method::GET && request.uri().path() == "/healthz" {
        return Ok(json_response(
            StatusCode::OK,
            json!({
                "status": "ok",
                "implementation": "spruceid-ssi-0.16.0",
                "operations": ["issue", "derive", "verifyDerived"],
                "verifyBase": false,
                "batching": false,
                "oneHttpRequestPerOperation": true,
                "issuer": DIDKey::generate_url(&state.jwk)
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                "operationsCompleted": state.operations.load(Ordering::Relaxed),
                "failures": state.failures.load(Ordering::Relaxed)
            }),
        ));
    }
    if request.method() != Method::POST {
        return Ok(json_response(
            StatusCode::METHOD_NOT_ALLOWED,
            json!({"error": "use POST"}),
        ));
    }
    let path = request.uri().path().to_owned();
    let body = match to_bytes(request.into_body()).await {
        Ok(body) => body,
        Err(error) => {
            return Ok(json_response(
                StatusCode::BAD_REQUEST,
                json!({"error": error.to_string()}),
            ))
        }
    };
    let body = match serde_json::from_slice::<JsonValue>(&body) {
        Ok(body) => body,
        Err(error) => {
            return Ok(json_response(
                StatusCode::BAD_REQUEST,
                json!({"error": error.to_string()}),
            ))
        }
    };
    let result = match path.as_str() {
        "/credentials/issue" => issue_one(&state, &body).await,
        "/credentials/derive" => derive_one(&body).await,
        "/credentials/verify-derived" => verify_derived_one(&body).await,
        _ => {
            return Ok(json_response(
                StatusCode::NOT_FOUND,
                json!({"error": "unsupported operation endpoint"}),
            ))
        }
    };
    match result {
        Ok(value) => {
            state.operations.fetch_add(1, Ordering::Relaxed);
            Ok(json_response(StatusCode::OK, value))
        }
        Err(error) => {
            state.failures.fetch_add(1, Ordering::Relaxed);
            Ok(json_response(
                StatusCode::BAD_REQUEST,
                json!({"verified": false, "error": error}),
            ))
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(40643);
    let state = Arc::new(State {
        jwk: JWK::generate_bls12381g2(),
        operations: AtomicU64::new(0),
        failures: AtomicU64::new(0),
    });
    let make = make_service_fn(move |_| {
        let state = state.clone();
        async move { Ok::<_, Infallible>(service_fn(move |request| handle(request, state.clone()))) }
    });
    let address = ([127, 0, 0, 1], port).into();
    eprintln!("SpruceID W3C adapter listening on http://{address}");
    Server::bind(&address).serve(make).await.unwrap();
}
