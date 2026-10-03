#!/usr/bin/env node

import http from "node:http";
import {readFile} from "node:fs/promises";
import cbor from "../../ref/implementations/grotto-bbs-vc-library/node_modules/cbor/lib/cbor.js";
import {base58btc} from "../../ref/implementations/grotto-bbs-vc-library/node_modules/multiformats/src/bases/base58.js";
import {base64url} from "../../ref/implementations/grotto-bbs-vc-library/node_modules/multiformats/src/bases/base64.js";
import {
  API_ID_BBS_SHA,
  prepareGenerators
} from "../../ref/implementations/grotto-bbs-vc-library/node_modules/@grottonetworking/bbs-signatures/lib/BBS.js";
import {
  concatBytes,
  hexToBytes
} from "../../ref/implementations/grotto-bbs-vc-library/node_modules/@noble/hashes/esm/utils.js";
import {signBase} from "../../ref/implementations/grotto-bbs-vc-library/lib/BBSsignBase.js";
import {verifyBase} from "../../ref/implementations/grotto-bbs-vc-library/lib/BBSverifyBase.js";
import {derive} from "../../ref/implementations/grotto-bbs-vc-library/lib/BBSderive.js";
import {verifyDerived} from "../../ref/implementations/grotto-bbs-vc-library/lib/BBSverifyDerived.js";
import {localLoader} from "../../ref/implementations/grotto-bbs-vc-library/examples/documentLoader.js";

const port = Number(process.env.PORT ?? 40543);
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
const sourceRoot = new URL(
  "../../ref/implementations/grotto-bbs-vc-library/examples/",
  import.meta.url
);
const keyMaterial = JSON.parse(
  await readFile(new URL("input/BBSKeyMaterial.json", sourceRoot))
);
const keyPair = {
  priv: hexToBytes(keyMaterial.privateKeyHex),
  pub: hexToBytes(keyMaterial.publicKeyHex)
};
// The table workload carries 64 credentialSubject fields plus the mandatory
// W3C credential statements.  Prepare enough generators once at service
// startup so the timed path never truncates or regenerates the public basis.
const generators = await prepareGenerators(128, API_ID_BBS_SHA);
const publicKeyMultibase = base58btc.encode(
  concatBytes(new Uint8Array([0xeb, 0x01]), keyPair.pub)
);
const proofConfig = {
  type: "DataIntegrityProof",
  cryptosuite: "bbs-2023",
  created: "2023-08-15T23:36:38Z",
  verificationMethod:
    `did:key:${publicKeyMultibase}#${publicKeyMultibase}`,
  proofPurpose: "assertionMethod"
};
const operationNames = [
  "issue",
  "verifySignature",
  "derive",
  "proofVerify"
];
const operationEndpoints = new Set([
  "/credentials/issue",
  "/credentials/derive",
  "/credentials/verify"
]);
const counters = Object.fromEntries(operationNames.map(name => [
  name,
  {dispatches: 0, dispatchedItems: 0, totalQueueWaitMs: 0,
    maxObservedBatch: 1, lastBatchSize: 1, batchSizeCounts: [0, 0]}
]));

function snapshot(name) {
  const value = counters[name];
  const counts = [...value.batchSizeCounts];
  counts[1] = value.dispatches;
  return {...value, batchSizeCounts: counts};
}

function health() {
  return {
    status: "ok",
    bbs_backend_requested: "reference",
    bbs_backend_active: "grotto-bbs-vc-library-0.1.0",
    bbs_backend_hardware_acceleration: false,
    bbs_backend_fallback: false,
    bbs_backend_gpu_abi_version: 0,
    operations: ["issue", "verifyBase", "derive", "verifyDerived"],
    oneHttpRequestPerOperation: true,
    w3c_runtime: {
      jsonldWorkers: 1,
      sharedJsonldLoader: true,
      canonicalizationCacheCapacity: 0
    },
    batching: {
      enabled: false,
      config: {intervalMs: 0, maxBatchSize: 1, queueCapacity: 1},
      issue: snapshot("issue"),
      verifySignature: snapshot("verifySignature"),
      derive: snapshot("derive"),
      proofVerify: snapshot("proofVerify")
    }
  };
}

function record(name) {
  counters[name].dispatches += 1;
  counters[name].dispatchedItems += 1;
}

function presentationHeader(document) {
  const encoded = document?.proof?.proofValue;
  if(typeof encoded !== "string" || !encoded.startsWith("u")) {
    return new Uint8Array();
  }
  const bytes = base64url.decode(encoded);
  if(bytes[0] !== 0xd9 || bytes[1] !== 0x5d || bytes[2] !== 0x03) {
    return new Uint8Array();
  }
  const decoded = cbor.decode(bytes.slice(3));
  return decoded[4] instanceof Uint8Array ?
    decoded[4] : new Uint8Array(decoded[4] ?? []);
}

async function route(path, body) {
  if(path.endsWith("/credentials/issue")) {
    record("issue");
    return signBase(
      structuredClone(body.credential),
      keyPair,
      body.options?.mandatoryPointers ?? [],
      {documentLoader: localLoader, proofConfig},
      generators
    );
  }
  if(path.endsWith("/credentials/derive")) {
    record("derive");
    const ph = new Uint8Array(body.options?.presentationHeader ?? []);
    return derive(
      body.verifiableCredential,
      body.options?.selectivePointers ?? [],
      {documentLoader: localLoader},
      generators,
      ph
    );
  }
  if(path.endsWith("/credentials/verify")) {
    const document = body.verifiableCredential;
    const derived = document?.proof?.proofValue?.startsWith("u2V0D");
    record(derived ? "proofVerify" : "verifySignature");
    const verified = derived ?
      await verifyDerived(
        document,
        keyPair.pub,
        {documentLoader: localLoader},
        generators,
        presentationHeader(document)
      ) :
      await verifyBase(
        document,
        keyPair.pub,
        {documentLoader: localLoader},
        generators
      );
    return {verified};
  }
  throw new Error(`unsupported path: ${path}`);
}

const server = http.createServer(async (request, response) => {
  try {
    if(request.method === "GET" && request.url.endsWith("/healthz")) {
      response.writeHead(200, {"content-type": "application/json"});
      response.end(JSON.stringify(health()));
      return;
    }
    if(request.method !== "POST" || !operationEndpoints.has(request.url)) {
      response.writeHead(request.method === "POST" ? 404 : 405);
      response.end();
      return;
    }
    const chunks = [];
    for await (const chunk of request) {
      chunks.push(chunk);
    }
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    const result = await route(request.url, body);
    response.writeHead(200, {"content-type": "application/json"});
    response.end(JSON.stringify(result));
  } catch(error) {
    response.writeHead(400, {"content-type": "application/json"});
    response.end(JSON.stringify({error: String(error?.stack ?? error)}));
  }
});
// See the Digital Bazaar adapter: a 5 s default races pooled requests once
// high-concurrency crypto work queues past the idle interval.
server.keepAliveTimeout = keepAliveTimeoutMs;
server.headersTimeout = keepAliveTimeoutMs + 5000;

server.listen({port, host: "127.0.0.1", backlog: listenBacklog}, () => {
  console.log(`Grotto reference W3C adapter listening on 127.0.0.1:${port}`);
});
