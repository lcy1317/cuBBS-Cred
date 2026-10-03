# cuBBS-Cred

cuBBS-Cred accelerates the four complete BBS operations with CUDA and exposes
a W3C `bbs-2023` HTTP service with dynamic batching, prepared-state caching,
and optional managed-credential endpoints.

## Documentation

Start with the [documentation index](docs/README.md).
Build, deployment, HTTP examples, correctness tests and reproducible benchmarks
are maintained there rather than in per-directory READMEs.

## Source layout

- `crates/`: Rust protocol, W3C, service and benchmark implementations.
- `native/cuda/`: CUDA ABI, MSM/pairing kernels and native tests.
- `benchmarks/`: maintained runners, schemas, fixtures and comparison adapters.
- `CFRG-IETF-BBS/`: pinned conformance vectors and their upstream license.
- `docs/`: project documentation and editable documentation figures.
- `ref/` and `paper-figures/`: local dependencies and frozen research artifacts;
  see the [data and archive contract](docs/benchmarks.md).

First-party crates and binaries use `cubbs-*`, environment variables use
`CUBBS_*`, and native entry points use `cubbs_*`. Rebuild the Rust executables
and CUDA library together; old binaries/libraries must not be mixed with this
namespace. Third-party identities, protocol identifiers, pinned input octets
and archived evidence are not renamed.

Existing deployments must migrate their `OMNIID_*` settings to `CUBBS_*` and
point `CUBBS_BBS_GPU_LIBRARY` at `libcubbs_bbs_cuda.so`. Historical benchmark
input bytes are intentionally preserved for reproducible comparisons; they
are not public product names. See the [ownership map](docs/gpu-bbs-backend-design.md)
for the refactored module and cache boundaries.

## Licensing

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and
[THIRD_PARTY.md](THIRD_PARTY.md). Native dependencies retain their own notices.
Binary releases must pass the license-bundle gate documented in the
[build and release guide](docs/build-run-http-benchmark.zh-CN.md).
