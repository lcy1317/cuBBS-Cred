#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ref_root="$repo_root/ref/implementations"

(
  cd "$ref_root/bbs-2023-cryptosuite"
  NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--experimental-global-webcrypto" \
    npm test
)

(
  cd "$ref_root/grotto-bbs-vc-library"
  npm test
)

(
  cd "$ref_root/spruceid-ssi"
  cargo test -p ssi-data-integrity-suites \
    --features 'w3c,bbs' bbs_2023 --lib
)

if [[ ${TRY_PROCIVIS:-0} == 1 ]]; then
  (
    cd "$ref_root/procivis-one-open-core"
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
      cargo test -p one-providers json_ld_bbsplus --lib
  )
else
  jq -e '
    [.w3cImplementations[]
      | select(.id == "procivis-one-open-core")
      | .correctness.status] == ["blocked-dependency"]
  ' "$repo_root/benchmarks/reference-w3c/inventory.json" >/dev/null
  echo "Procivis build skipped: pinned pairing_crypto source is unavailable"
fi

echo "runnable reference W3C implementation verification passed"
