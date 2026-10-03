# cuBBS-Cred 编译、HTTP 调用与四操作测试

本文给出从源码构建到实际调用的最短完整路径，覆盖：

- CPU-only 与 CUDA 构建；
- 启动 W3C BBS HTTP 服务；
- Issue → Verify Base → Derive → Verify Derived 调用闭环；
- HTTP 持续负载测试；
- Table 1 所对应的四个完整 BBS 操作测试；
- 二进制发布前的第三方许可证检查。

以下命令均从仓库根目录执行。现有 `cubbs-*` 二进制名和 `CUBBS_*`
环境变量是为兼容旧部署保留的接口，并不表示项目仍名为 cuBBS-Cred。

## 1. 入口与测量边界

| 产物 | 用途 | 测量边界 |
|---|---|---|
| `cubbs-w3c-service` | W3C `bbs-2023` HTTP 服务 | HTTP、JSON-LD、规范化、合批和完整 BBS 操作 |
| `w3c-http-load` | 完整 HTTP workflow 负载 | Issue、Base Verify、Derive、Derived Verify |
| `w3c-http-op-load` | 单操作 HTTP 负载 | 一次 HTTP 请求对应一个或一批 W3C 操作 |
| `cubbs-four-op-preloaded` | cuBBS-Cred CPU/GPU 后端测试 | 输入预生成后的完整 Sign、Signature Verify、ProofGen、ProofVerify API |
| `cubbs-four-op-reference` | 论文 CPU 基线 | 硬固定 ZKryptium 0.6.1 public single-item API |
| `cubbs-gpu-ceiling` | GPU 诊断 | 仅 common-points G1 MSM device ceiling |

`cubbs-gpu-ceiling` 不包含完整 BBS 协议，也不包含 HTTP/W3C 开销，不能用它的
outputs/s 替代 Fig. 1 或 Table 1 的完整操作吞吐率。

## 2. 构建

### 2.1 基础环境

CPU-only 构建需要 Rust/Cargo 和常规 C/C++ 构建工具。GPU 构建还需要：

- 可用的 NVIDIA driver 和 CUDA toolkit；
- `nvcc`、GNU Make、C/C++ compiler；
- 下述固定版本的 BLST 与 SPPARK 源码 checkout。

HTTP 示例另外使用 `curl` 和 `jq`。先确认当前目录和工具链：

```bash
# 在本仓库根目录执行
rustc --version
cargo --version
nvidia-smi
nvcc --version
```

CPU-only 使用者可以跳过 `nvidia-smi`、`nvcc` 和 2.2 节。

### 2.2 准备固定的 CUDA 构建输入

首次构建 GPU 后端时执行：

```bash
git clone --branch v0.1.15 --depth 1 \
  https://github.com/supranational/sppark.git \
  ref/implementations/supranational-sppark
test "$(git -C ref/implementations/supranational-sppark rev-parse HEAD)" = \
  17278d74295392f9813f009300b257a688422b7a

git clone --branch v0.3.16 --depth 1 \
  https://github.com/supranational/blst.git \
  ref/implementations/supranational-blst
test "$(git -C ref/implementations/supranational-blst rev-parse HEAD)" = \
  e7f90de551e8df682f3cc99067d204d8b90d27ad
```

这两个目录已被 Git 忽略。若目录已存在，不要覆盖；先用上述 `rev-parse` 和
`git status --short` 确认它们是对应 commit 的独立、干净 checkout。
`cargo fetch --locked` 不会下载 BLST，因为它不是本 workspace 的 Cargo
dependency。

非 Git 的 SPPARK 文件快照可以满足本地编译，但不能通过正式发布许可证 gate。
若本机默认目录已有这种快照，应保留它，并把同一固定版本另行 clone 到新目录；
构建时可用 `SPPARK_ROOT`、`BLST_SRC` 指向新 checkout。

构建并执行 CUDA smoke tests：

```bash
make -C native/cuda CUDA_ARCH=native -j"$(nproc)"
make -C native/cuda CUDA_ARCH=native smoke
```

默认产物为：

```text
target/native/cuda/libcubbs_bbs_cuda.so
```

`CUDA_ARCH=native` 应在目标 GPU 机器上构建。跨机器构建时应改为目标架构，例如
Ada RTX 4090 使用 `sm_89`；发布记录必须保存实际架构、CUDA 和 driver 版本。

### 2.3 构建 Rust 二进制

```bash
cargo build --release --locked \
  -p cubbs-w3c-service \
  -p cubbs-gpu-ceiling \
  -p cubbs-bench --bins
```

依赖已完整缓存时可以增加 `--offline`。主要产物位于：

```text
target/release/cubbs-w3c-service
target/release/cubbs-four-op-preloaded
target/release/cubbs-four-op-reference
target/release/cubbs-gpu-ceiling
target/release/w3c-http-load
target/release/w3c-http-op-load
```

CPU 服务只需要 Rust 可执行文件及其系统动态库。GPU 服务还必须同时部署匹配目标
GPU 架构的 `libcubbs_bbs_cuda.so`，并在启动时提供其路径。

## 3. 启动 HTTP 服务

服务配置使用环境变量，目前没有 CLI 参数解析器。不要用
`cubbs-w3c-service --help` 探测参数；该二进制会按正常服务入口启动。

### 3.1 CPU quick start

```bash
CUBBS_PORT=40443 \
CUBBS_BBS_BACKEND=cpu \
target/release/cubbs-w3c-service
```

这里的 CPU 路径是 cuBBS-Cred 本地修改的 ZKryptium 0.2.2 派生协议引擎，适合
功能验证，但不是论文中独立的 ZKryptium 0.6.1 CPU comparator。

### 3.2 严格 GPU 启动

```bash
CUDA_VISIBLE_DEVICES=0 \
CUBBS_PORT=40443 \
CUBBS_BBS_BACKEND=gpu \
CUBBS_BBS_GPU_LIBRARY="$PWD/target/native/cuda/libcubbs_bbs_cuda.so" \
CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 \
target/release/cubbs-w3c-service
```

`CUBBS_BBS_REQUIRE_HARDWARE_GPU=1` 是正式 GPU 测试的必要条件：动态库、driver、
架构或设备不可用时启动直接失败，不允许静默记录 CPU fallback 数据。

服务目前固定监听 `127.0.0.1`，默认端口为 `40443`，只有端口可通过
`CUBBS_PORT` 修改。需要外部访问时应在本机反向代理或 SSH tunnel 后使用，不能
把当前二进制描述成直接监听 `0.0.0.0` 的公网服务。

未设置 `CUBBS_SECRET_KEY` 时，每次进程启动都会生成临时 issuer key，重启后 DID
和 verification method 会改变。生产部署必须从 secret manager 注入固定的 32-byte
key；支持 `hex:`、`base64url:` 和 multibase `z...` 编码。

### 3.3 健康检查与 GPU fail-closed 验收

在另一个终端执行：

```bash
BASE_URL=http://127.0.0.1:40443/id
curl -fsS "$BASE_URL/healthz" | jq .
```

正式 GPU 数据至少必须通过：

```bash
curl -fsS "$BASE_URL/healthz" | jq -e '
  .status == "ok" and
  .bbs_backend_requested == "gpu" and
  .bbs_backend_hardware_acceleration == true and
  .bbs_backend_fallback == false and
  .bbs_backend_gpu_abi_version == 2
'
```

若这个检查失败，不应继续记录 GPU 性能数据。

## 4. 完整 HTTP 调用样例

以下四步使用单请求 API。每个非 `/id` 路径都有相同的 `/id` alias；本文统一使用
`/id`。先设置：

```bash
BASE_URL=http://127.0.0.1:40443/id
```

### 4.1 Issue：签发 base credential

```bash
curl -fsS -X POST "$BASE_URL/credentials/issue" \
  -H 'content-type: application/json' \
  --data-binary @- \
  -o /tmp/cubbs-signed-vc.json <<'JSON'
{
  "credential": {
    "@context": [
      "https://www.w3.org/ns/credentials/v2",
      {
        "ExampleCredential": "https://example.com/ExampleCredential",
        "name": "https://schema.org/name",
        "employeeId": "https://schema.org/identifier"
      }
    ],
    "id": "urn:uuid:cubbs-quickstart-1",
    "type": ["VerifiableCredential", "ExampleCredential"],
    "issuer": "did:example:placeholder",
    "credentialSubject": {
      "id": "did:example:alice",
      "name": "Alice",
      "employeeId": "E-1001"
    }
  },
  "options": {
    "mandatoryPointers": ["/issuer"]
  }
}
JSON

jq -e '.proof.proofValue | length > 0' /tmp/cubbs-signed-vc.json
```

### 4.2 Verify Base：验证 base signature

```bash
jq -n --slurpfile vc /tmp/cubbs-signed-vc.json \
  '{verifiableCredential: $vc[0]}' |
curl -fsS -X POST "$BASE_URL/credentials/verify" \
  -H 'content-type: application/json' \
  --data-binary @- |
jq -e '.verified == true and (.checks | index("proof") != null)'
```

### 4.3 Derive：生成选择披露 credential

```bash
jq -n --slurpfile vc /tmp/cubbs-signed-vc.json '
  {
    verifiableCredential: $vc[0],
    options: {
      selectivePointers: ["/credentialSubject/name"],
      presentationHeader: [111, 109, 110, 105, 105, 100]
    }
  }
' |
curl -fsS -X POST "$BASE_URL/credentials/derive" \
  -H 'content-type: application/json' \
  --data-binary @- \
  -o /tmp/cubbs-derived-vc.json

jq -e '
  .credentialSubject.name == "Alice" and
  (.credentialSubject | has("employeeId") | not) and
  (.proof.proofValue | length > 0)
' /tmp/cubbs-derived-vc.json
```

### 4.4 Verify Derived：验证选择披露证明

```bash
jq -n --slurpfile vc /tmp/cubbs-derived-vc.json \
  '{verifiableCredential: $vc[0]}' |
curl -fsS -X POST "$BASE_URL/credentials/verify" \
  -H 'content-type: application/json' \
  --data-binary @- |
jq -e '.verified == true and (.checks | index("proof") != null)'
```

两个 Verify 命令都应以状态码 0 退出并输出包含 `"verified": true` 的结果。

## 5. HTTP 服务负载测试

生产容量使用持续 workflow 模式，并为每次 workflow 提供 unique VC：

```bash
target/release/w3c-http-load \
  --mode sustained \
  --base-url http://127.0.0.1:40443/id \
  --concurrency 24 \
  --duration-secs 10 \
  --warmup-rounds 1 \
  --startup-spread-ms 500 \
  --generate-dataset /tmp/cubbs-w3c-unique.jsonl \
  --dataset-size 100000 \
  --output /tmp/cubbs-w3c-sustained.json
```

数据集必须覆盖 warm-up 与正式阶段发起的全部 workflow；不足时增大
`--dataset-size`。报告同时保存 backend identity、四操作吞吐/延迟以及测试前后
`/healthz` batching delta。正式 GPU 报告仍需确认：

```bash
jq -e '
  .backendInfo.hardwareAcceleration == true and
  .backendInfo.fallback == false
' /tmp/cubbs-w3c-sustained.json
```

Fig. 8 / Table 2 的“一次 HTTP 请求对应一个 W3C 操作”正式测试使用
`w3c-http-op-load` 和 `benchmarks/w3c/full-flow/` 中锁定的 runner，不应拿上述
四阶段 workflow 总吞吐直接替换单操作数据。详细的合批语义见
[`W3C 服务`](w3c-service.md)。

## 6. 四个完整 BBS 操作测试

四操作名称为 `sign`、`verify-signature`、`proof-gen` 和 `proof-verify`。
`cubbs-four-op-preloaded` 在计时前生成输入，但计时区间保留完整协议操作、GPU
提交、结果构造和释放；它不是仅测 CUDA kernel 的 microbenchmark。

### 6.1 GPU 四操作 quick smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
CUBBS_BBS_GPU_LIBRARY="$PWD/target/native/cuda/libcubbs_bbs_cuda.so" \
target/release/cubbs-four-op-preloaded \
  --backend gpu \
  --api batch \
  --operation all \
  --messages 16 \
  --same-issuer \
  --disclosed-ratio 0.5 \
  --batch 8 \
  --sign-batches 1 --sign-streams 1 \
  --verify-batches 1 --verify-streams 1 \
  --proof-gen-batches 1 --proof-gen-streams 1 \
  --proof-verify-batches 1 --proof-verify-streams 1 \
  > /tmp/cubbs-four-op-gpu-smoke.json
```

该入口本身会 fail closed：请求 GPU 时如果未加载真实 GPU backend 或发生整体
fallback，进程返回非零。再检查报告：

```bash
jq -e '
  .tool == "cubbs-four-op-preloaded" and
  .backend.hardwareAcceleration == true and
  .backend.fallback == false and
  .validationPassed == true and
  ([.metrics[].failed] | all(. == 0))
' /tmp/cubbs-four-op-gpu-smoke.json
```

quick smoke 只验证构建、四操作和结果正确性，不是正式吞吐数据。正式测量应将四个
operation 分进程运行，并锁定 GPU、message count、issuer mode、disclosure ratio、
batch、streams、warm-up 和重复次数；当前 publication runner 说明见
[`测试与复现指南`](benchmarks.md)。

首次执行还会初始化 CUDA context 和进程内 prepared state，因此总运行时间可能
明显长于 JSON 中的计时区间；不要把首次启动耗时计入 steady-state 吞吐。

### 6.2 ZKryptium 0.6.1 CPU baseline smoke

```bash
target/release/cubbs-four-op-reference \
  --backend cpu \
  --api single \
  --operation all \
  --messages 16 \
  --same-issuer \
  --disclosed-ratio 0.5 \
  --batch 8 \
  --sign-batches 1 --sign-streams 1 \
  --verify-batches 1 --verify-streams 1 \
  --proof-gen-batches 1 --proof-gen-streams 1 \
  --proof-verify-batches 1 --proof-verify-streams 1 \
  > /tmp/cubbs-four-op-cpu-reference-smoke.json
```

这个专用二进制在代码中硬固定 ZKryptium 0.6.1；无需依赖环境变量选择版本。合格
CPU baseline 报告必须满足：

```bash
jq -e '
  .tool == "cubbs-four-op-reference" and
  .config.apiMode == "single" and
  .backend.requested == "cpu-reference" and
  .backend.active == "zkryptium-0.6.1" and
  .backend.fallback == false and
  .cpuReference == "zkryptium-0.6.1" and
  .validationPassed == true
' /tmp/cubbs-four-op-cpu-reference-smoke.json
```

不要用 `cubbs-four-op-preloaded --backend cpu` 或 HTTP 服务的 CPU 模式替代论文
CPU comparator；它们运行的是本地 0.2.2 派生生产引擎，测量身份不同。

## 7. 源码标注与二进制发布

当前源码树采用三层标注：

1. 修改自 ZKryptium 0.2.2 的 Rust 文件保留 LINKS Foundation 的原始版权和
   Apache-2.0 header，并增加 cuBBS-Cred 修改声明；原始 manifest 保存在
   `crates/cubbs-bbs-protocol/UPSTREAM-Cargo.toml`。
2. BLST 派生的常量、布局声明和 GLV 片段在对应 CUDA 源码旁注明 Supranational
   copyright、Apache-2.0 和来源；SPPARK/BLST checkout 保留其自身许可证。
3. 仓库级 `NOTICE`、`THIRD_PARTY.md`、`LICENSES/` 以及嵌入式 W3C context 的
   `contexts/NOTICE.md` 记录整体归属和再分发要求。

这意味着源码归属标注已完成，但“源码可以编译”不等于“预编译二进制已自动满足
全部再分发义务”。发布每一组二进制前，必须从与该构建一致的、独立且干净的固定
BLST/SPPARK checkout 生成许可证包：

```bash
python3 tools/build_release_license_bundle.py dist/cuBBS-Cred-licenses
```

独立 checkout 不在默认路径时显式指定：

```bash
python3 tools/build_release_license_bundle.py \
  --blst-root /path/to/clean/blst \
  --sppark-root /path/to/clean/sppark \
  dist/cuBBS-Cred-licenses
```

生成器是 fail closed 的：checkout 不是固定 commit、工作树不干净、Cargo license
metadata 未审查、MPL 对应源码缺失或 W3C context hash 不匹配时都会拒绝生成。
应把生成目录与 Rust 二进制、CUDA shared library 一起发布。完整边界和义务以
[`../THIRD_PARTY.md`](../THIRD_PARTY.md) 为准。

本机编译、正确性验证和性能复测不属于二进制发布流程，无需为此生成再分发包。

## 8. 最小验收清单

- `make -C native/cuda ...` 和 CUDA smoke tests 全部通过；
- `cargo build --release --locked ...` 生成预期二进制；
- GPU `/healthz` 显示 hardware acceleration、ABI v2、无 fallback；
- HTTP Issue → Verify Base → Derive → Verify Derived 全链成功；
- 四操作报告 `validationPassed=true` 且所有 `failed=0`；
- 论文 CPU 报告明确为 `cubbs-four-op-reference` / ZKryptium 0.6.1；
- 发布二进制前生成并随包携带 release license bundle。

完整 CPU/GPU、CFRG 和负例验收命令见[正确性验证](testing.md)。

## 9. 平台、ABI 与部署兼容性

当前构建路径面向 64-bit little-endian x86-64 Linux、GNU toolchain 和 NVIDIA
CUDA。没有经本仓库发布的 Windows/macOS、AArch64 或非 NVIDIA GPU 构建流程，
不能把 Rust 的潜在可移植性等同于完整 native 后端已经验证。

Rust 默认 release 命令未强制 `target-cpu=native`；但 ELF 二进制仍依赖构建时的
glibc/libstdc++。部署到更老的系统时，在兼容系统上重建并检查实际动态库依赖，
不应把某次本机产物的 GLIBC 版本写成源码的永久最低要求。

CUDA shared library 默认针对一个架构编译。换 GPU 时使用目标 `CUDA_ARCH`，
并使用独立构建目录避免误复用旧架构 object，例如：

```bash
make -C native/cuda CUDA_ARCH=sm_89 BUILD_DIR=../../target/native/cuda-sm89
```

只有设置 `CUBBS_BBS_GPU_LIBRARY` 指向该目录的 `.so` 才会实际使用它。
新架构要重跑 native smoke、四操作与 GPU 规范向量；batch/streams 的饱和参数
也需独立确认，不从另一张 GPU 直接推定。

ABI v2 是进程内 C ABI，不是网络或文件协议。Rust/native pointer width、
`size_t`、结构体布局和 BLST/SPPARK Montgomery-limb 表示必须匹配。
更改这些约定需新的 ABI 与完整正确性验收；详见[后端设计](gpu-bbs-backend-design.md)。
