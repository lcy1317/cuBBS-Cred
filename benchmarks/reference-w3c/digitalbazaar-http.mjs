#!/usr/bin/env node

import http from "node:http";
import {webcrypto} from "node:crypto";
import * as Bls12381Multikey from
  "../../ref/implementations/bbs-2023-cryptosuite/node_modules/@digitalbazaar/bls12-381-multikey/lib/index.js";
import {
  createDiscloseCryptosuite,
  createSignCryptosuite,
  createVerifyCryptosuite
} from "../../ref/implementations/bbs-2023-cryptosuite/lib/index.js";
import {
  bls12381MultikeyKeyPair
} from "../../ref/implementations/bbs-2023-cryptosuite/test/mock-data.js";
import {DataIntegrityProof} from
  "../../ref/implementations/bbs-2023-cryptosuite/node_modules/@digitalbazaar/data-integrity/lib/index.js";
import jsigs from
  "../../ref/implementations/bbs-2023-cryptosuite/node_modules/jsonld-signatures/lib/jsonld-signatures.js";
import {loader} from
  "../../ref/implementations/bbs-2023-cryptosuite/test/documentLoader.js";

globalThis.crypto ??= webcrypto;

const port = Number(process.env.PORT ?? 40743);
const listenBacklog = Number(process.env.LISTEN_BACKLOG ?? 4096);
if(!Number.isSafeInteger(listenBacklog) || listenBacklog < 1 ||
  listenBacklog > 65535) {
  throw new Error("LISTEN_BACKLOG must be an integer in 1..65535");
}
const keepAliveTimeoutMs = Number(process.env.KEEP_ALIVE_TIMEOUT_MS ?? 120000);
if(!Number.isSafeInteger(keepAliveTimeoutMs) || keepAliveTimeoutMs < 1 ||
  keepAliveTimeoutMs > 600000) {
  throw new Error("KEEP_ALIVE_TIMEOUT_MS must be an integer in 1..600000");
}
const algorithm = Bls12381Multikey.ALGORITHMS.BBS_BLS12381_SHA256;
const keyPair = await Bls12381Multikey.from(
  {...bls12381MultikeyKeyPair}, {algorithm});
const documentLoader = loader.build();
const {purposes: {AssertionProofPurpose}} = jsigs;
const purpose = new AssertionProofPurpose();
const date = "2023-03-01T21:29:24Z";
const operationEndpoints = new Set([
  "/credentials/issue",
  "/credentials/derive",
  "/credentials/verify-derived"
]);
let operations = 0;
let failures = 0;

async function issueOne(body) {
  const signSuite = new DataIntegrityProof({
    signer: keyPair.signer(),
    date,
    cryptosuite: createSignCryptosuite({
      mandatoryPointers: body.options?.mandatoryPointers ?? []
    })
  });
  return jsigs.sign(structuredClone(body.credential), {
    suite: signSuite,
    purpose,
    documentLoader
  });
}

async function deriveOne(body) {
  const discloseSuite = new DataIntegrityProof({
    cryptosuite: createDiscloseCryptosuite({
      selectivePointers: body.options?.selectivePointers ?? [],
      presentationHeader: new Uint8Array(
        body.options?.presentationHeader ?? [])
    })
  });
  return jsigs.derive(structuredClone(body.verifiableCredential), {
    suite: discloseSuite,
    purpose,
    documentLoader
  });
}

async function verifyDerivedOne(body) {
  const verifySuite = new DataIntegrityProof({
    cryptosuite: createVerifyCryptosuite()
  });
  const result = await jsigs.verify(
    structuredClone(body.verifiableCredential), {
    suite: verifySuite,
    purpose,
    documentLoader
  });
  if(!result.verified) {
    throw result.error ?? new Error("derived verification failed");
  }
  return {verified: true};
}

async function route(path, body) {
  if(path === "/credentials/issue") {
    return issueOne(body);
  }
  if(path === "/credentials/derive") {
    return deriveOne(body);
  }
  if(path === "/credentials/verify-derived") {
    return verifyDerivedOne(body);
  }
  throw new Error(`unsupported operation endpoint: ${path}`);
}

const server = http.createServer(async (request, response) => {
  try {
    if(request.method === "GET" && request.url === "/healthz") {
      response.writeHead(200, {"content-type": "application/json"});
      response.end(JSON.stringify({
        status: "ok",
        implementation: "digitalbazaar-bbs-2023-cryptosuite-2.0.2-0",
        operations: ["issue", "derive", "verifyDerived"],
        verifyBase: false,
        batching: false,
        oneHttpRequestPerOperation: true,
        issuer: keyPair.controller,
        operationsCompleted: operations,
        failures
      }));
      return;
    }
    if(request.method !== "POST" || !operationEndpoints.has(request.url)) {
      response.writeHead(404);
      response.end();
      return;
    }
    const chunks = [];
    for await (const chunk of request) {
      chunks.push(chunk);
    }
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    const result = await route(request.url, body);
    operations += 1;
    response.writeHead(200, {"content-type": "application/json"});
    response.end(JSON.stringify(result));
  } catch(error) {
    failures += 1;
    response.writeHead(400, {"content-type": "application/json"});
    response.end(JSON.stringify({
      verified: false,
      error: String(error?.stack ?? error)
    }));
  }
});
// The default Node keep-alive timeout is only 5 s.  At high offered
// concurrency, an otherwise valid pooled request can be queued longer than
// that interval and race a server-side idle close; preserve HTTP capacity
// semantics by keeping the transport alive through the measured 30 s window.
server.keepAliveTimeout = keepAliveTimeoutMs;
server.headersTimeout = keepAliveTimeoutMs + 5000;

server.listen({port, host: "127.0.0.1", backlog: listenBacklog}, () => {
  console.log(`Digital Bazaar W3C adapter listening on 127.0.0.1:${port}`);
});
