# Third-Party Software and Protocol Sources

cuBBS-Cred is licensed under Apache-2.0. It builds on the open-source software
and public protocol specifications listed below. This file records direct code
reuse and native build inputs; the complete Rust dependency graph and exact
registry checksums remain authoritative in `Cargo.lock`.

## Direct and Modified Sources

| Component | Version/source used | License | Use in cuBBS-Cred |
|---|---|---|---|
| [ZKryptium](https://github.com/Cybersecurity-LINKS/ZKryptium) | derivative/registry 0.2.2: commit `0e21c20f4c84473e7eb69a1aef136159c9d085b8`, crate SHA-256 `c734c171ed591a19dc1127351eb1b4d91864d3e53b2b6e9992bffcb7febf364a`; registry 0.6.1: commit `69e8e273a4ad92a59b52e9b9a2744b2a48b9bbd6`, crate SHA-256 `39159a1cd33b28bad7c3502528f77da679dd6f45a055581f5ac56954c458c6e5` | Apache-2.0 | `crates/cubbs-bbs-protocol` derives from 0.2.2 and implements the production protocol lowering. Registry 0.6.1 is the default independent CPU reference and recoverable GPU fallback; registry 0.2.2 is retained for legacy comparison. |
| [BLST](https://github.com/supranational/blst) | crate 0.3.16; upstream commit `e7f90de551e8df682f3cc99067d204d8b90d27ad`; crate SHA-256 `dcdb4c7013139a150f9fc55d123186dbfaba0d912817466282c73ac49e71fb45` | Apache-2.0 | CPU MSM Routes 5/6, BLS12-381 point validation and serialization, pairing support, and low-level curve/field facilities. Some CUDA code documents compatible constants and GLV decomposition derived from BLST source. `BLST_SRC` may override this audited default. |
| [SPPARK](https://github.com/supranational/sppark) | tag v0.1.15; commit `17278d74295392f9813f009300b257a688422b7a`; Git tree `47f0157dc6285f3a999545788a4a536b82418214` | Apache-2.0 | finite-field and curve templates used by cuBBS-Cred CUDA Routes 1–4; generic Pippenger is linked only when optional reference Routes 7/8 are enabled. `SPPARK_ROOT` may override this audited default. |
| [CFRG BBS conformance snapshot](https://github.com/decentralized-identity/bbs-signature) | pinned commit and archive checksum recorded in [docs/testing.md](docs/testing.md) | Apache-2.0 for the bundled snapshot | protocol conformance vectors and draft-10 semantic checks. |
| [W3C JSON-LD contexts](https://www.w3.org/copyright/software-license/) | canonical HTTP copies verified 2026-09-15; per-file sources and SHA-256 values in `crates/cubbs-vc-di-bbs/src/contexts/NOTICE.md` | W3C Software and Document License; the W3C CCG VDL repository also applies Apache-2.0 to software | four offline contexts embedded in `cubbs-vc-di-bbs` with `include_str!`. |

ZKryptium upstream copyright headers and its license are retained in
`crates/cubbs-bbs-protocol`. Every modified Rust source file and manifest
derived from the 0.2.2 source base carries a cuBBS-Cred modification notice,
while unchanged source files retain only the upstream header. `docs/protocol-provenance.md`
identifies the derivative, distinguishes the 0.6.1 reference/fallback role,
summarizes the changes, and records the replacement CFRG fixture provenance.
The bundled CFRG snapshot retains its own `LICENSE` file.

The four JSON-LD context files under
`crates/cubbs-vc-di-bbs/src/contexts/` are redistribution inputs, not merely
test fixtures: their bytes are compiled into the W3C service. Their directory
NOTICE records the canonical URI, upstream timestamp, local hash, comparison
date, and formatting-only modification statement. The complete 2015 and 2023
W3C license notices are retained under `LICENSES/` and must accompany binary
redistribution.

`native/cuda/Makefile` compiles BLST's `server.c` and platform assembly into the
native shared library. SPPARK is not tracked in this Git/source distribution;
an audited snapshot may exist in the Git-ignored local `ref/` tree. Its headers
are a required build input for the production CUDA field and curve types.
Enabling `CUBBS_BBS_ENABLE_SPPARK_REFERENCE=1` additionally compiles SPPARK's
generic Pippenger implementation for benchmark-only Routes 7/8.

## Code Boundaries

The following are third-party or modified-third-party layers:

- `crates/cubbs-bbs-protocol`: modified ZKryptium-derived protocol layer;
- Cargo packages named `zkryptium`: unmodified registry reference adapters;
- BLST sources compiled by `native/cuda/Makefile` and BLST calls in the native
  bridge;
- SPPARK headers included by `native/cuda/src/prepare_b_msm.cu` and the CUDA
  kernel headers;
- optional SPPARK Pippenger reference object used only by MSM Routes 7/8;
- conformance vectors under `CFRG-IETF-BBS`.

The following are cuBBS-Cred implementation layers built on those foundations:

- the ABI v2 host/runtime interface and MSM Route 1–4 registry;
- operation- and shape-aware dynamic batching and planner IR;
- common-points and variable-points lowering;
- Route 1–4 scheduling, specialized/universal CUDA kernels, caches, telemetry,
  and failure policy;
- ProofGen/ProofVerify split and fused execution paths;
- W3C service integration and end-to-end HTTP microbatch scheduling.

cuBBS-Cred does not claim ownership of the BLS12-381 curve, CFRG BBS equations,
W3C Data Integrity semantics, or general techniques such as GLV, Pippenger,
fixed-base comb, window decomposition, streams, caching, and batching.

## Rust Dependencies

The direct and transitive Rust packages are resolved by `Cargo.lock` and remain
under their respective upstream licenses. The dependency graph is predominantly
licensed under permissive Apache-2.0, MIT, BSD, ISC, Unicode, Zlib, BSL, or
Unlicense terms.

The locked JSON-LD dependency tree also contains the following unmodified
`MPL-2.0+` packages:

| Package | Cargo.lock SHA-256 | Corresponding source |
|---|---|---|
| `bitmaps 2.1.0` | `031043d04099746d8db04daf1fa424b2bc8bd69d92b25962dcde24da39ab64a2` | [crates.io archive](https://crates.io/api/v1/crates/bitmaps/2.1.0/download) |
| `im 15.1.0` | `d0acd33ff0285af998aaf9b57342af478078f53492322fafc47450e09397e0e9` | [crates.io archive](https://crates.io/api/v1/crates/im/15.1.0/download) |
| `sized-chunks 0.6.5` | `16d69225bde7a69b235da73377861095455d298f2b970996eec25ddbb42b3d1e` | [crates.io archive](https://crates.io/api/v1/crates/sized-chunks/0.6.5/download) |

cuBBS-Cred does not vendor or modify their source files. A binary distributor
must provide the applicable MPL notice and make the corresponding source
available as required by that license. Release inventories must be generated
from the exact committed lockfile rather than treating this summary as a
complete replacement for dependency license texts.

`r-efi 6.0.0` offers `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; cuBBS-Cred may
redistribute it under the Apache-2.0 option and does not rely on the LGPL
alternative.

The modified `cubbs-bbs-protocol` manifest no longer declares the unused
`cargo-license` dependency; `UPSTREAM-Cargo.toml` retains that upstream
provenance.
The unmodified registry `zkryptium 0.2.2` package still declares `cargo-license`,
so it remains in `Cargo.lock` while that CPU reference adapter is enabled.

## Benchmark-Only External Implementations

The source trees under the Git-ignored `ref/implementations/` directory are
not part of a cuBBS-Cred commit or binary. Scripts under
`benchmarks/reference-w3c` interoperate with them for correctness and
performance comparison:

| Reference implementation | Upstream license |
|---|---|
| Digital Bazaar `bbs-2023-cryptosuite` | BSD-3-Clause |
| Grotto `bbs-vc-library` | BSD-3-Clause |
| SpruceID `ssi` | Apache-2.0 |
| Procivis `one-open-core` | Apache-2.0 |

These names identify independent upstream projects and do not imply
endorsement. If any reference source or executable is redistributed with a
cuBBS-Cred benchmark bundle, its own license, copyright notice, and any required
attribution must accompany that bundle.

## Reproducible and Binary Releases

**Release gate:** this source tree does not itself constitute a generated
dependency-license bundle for prebuilt binaries. Do not publish a binary
artifact until the exact build's third-party license/notice texts and the MPL
source-access information above are packaged beside it.

`native/cuda/Makefile` prefers the audited BLST 0.3.16 checkout under
`ref/implementations/supranational-blst` and falls back to an already-present
Cargo cache copy. BLST is not in this workspace's Cargo dependency graph, so
`cargo fetch --locked` does not materialize it. The default `SPPARK_ROOT`
points to the audited v0.1.15 checkout under
`ref/implementations/supranational-sppark`. Neither checkout is included in
this Git repository. Overrides remain possible. Therefore a source or binary
release must:

1. verify and record the actual BLST source version and checksum;
2. verify and record the actual SPPARK upstream commit/tree or archive
   checksum;
3. record CUDA/NVCC and target GPU architecture;
4. include all license and notice material required by the selected upstream
   revisions;
5. keep Route 7/8 reference support disabled unless that optional dependency is
   intentionally included.

Run `python3 tools/build_release_license_bundle.py OUTPUT_DIR` from the exact
release revision. The generator refuses non-independent, dirty, or incorrectly
pinned BLST/SPPARK checkouts; rejects missing or unknown Rust license metadata;
copies package license/notice files; supplies the exact upstream MIT text absent
from the `dotenv 0.15.0` crate archive; includes the embedded-context notices;
and verifies and packages the locked source archives for all MPL-2.0+
dependencies. The generated inventory is tied to the SHA-256 of the committed
`Cargo.lock`. This notice supplements, and does not replace, the license terms
of each upstream project.

NVIDIA's CUDA toolkit, driver, and runtime remain under NVIDIA's own terms,
not this repository's Apache-2.0 license. Local compilation and testing do not
require generating a CUDA redistribution bundle. Binary redistribution, if
undertaken separately, requires its own review of the applicable terms.
