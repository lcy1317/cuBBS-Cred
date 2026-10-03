# 测试与复现指南

所有命令从仓库根目录执行。构建与快速调用见
[构建运行指南](build-run-http-benchmark.zh-CN.md)，正确性验收见
[正确性验证](testing.md)。性能冻结不会使复现工具、fixture 或配置失去用途。

## 测量入口

| 用途 | 入口 | 计时边界 |
|---|---|---|
| Fig. 1 / Table 1 GPU | `cubbs-four-op-preloaded` | 输入预生成后的完整四操作；保留协议计算、提交、结果构造与释放 |
| Fig. 1 / Table 1 CPU comparator | `cubbs-four-op-reference` | 硬固定 ZKryptium 0.6.1 public single-item API |
| Fig. 2 | `native/cuda/msm_route_bench` | Route 1–4 MSM，不是完整 BBS 操作 |
| Fig. 3 / Fig. 4 | `cubbs-bench --bbs-throughput-stress` | 不同 message/issuer workload 的完整 BBS 操作 |
| Fig. 8 / Table 2 | `cubbs-w3c-service` + `w3c-http-op-load` | 每次 HTTP 请求完成一个 W3C 操作 |
| HTTP 全流程回归 | `w3c-http-load --mode sustained` | Issue → Base Verify → Derive → Proof Verify |
| GPU 输入不足诊断 | `cubbs-gpu-ceiling` | 仅预加载 common-points MSM device ceiling |
| 规范正确性 | `cfrg-ietf-conformance` | 规范向量，不是性能测试 |

`items/s`、`requests/s`、`workflows/s` 和 MSM `outputs/s` 不能互换。
HTTP 并发数 `c` 也不是 CUDA batch size；必须同时记录实际 batch 分布。
`cubbs-four-op-preloaded --backend cpu` 是本地修改的 0.2.2 派生引擎，
不能代替独立的 0.6.1 comparator。

## 原语与 GPU 诊断

```bash
cargo build --release --locked -p cubbs-gpu-ceiling -p cubbs-bench --bins
target/release/cubbs-four-op-preloaded --help
target/release/cubbs-four-op-reference --help
```

四操作为 `sign`、`verify-signature`、`proof-gen`、`proof-verify`。
正式容量测试应分别运行，锁定 message 数/混合分布、issuer、披露比例、batch、
streams、warm-up、重复次数和时长。输入预生成及每个 stream 的预热在计时前完成，
不能把输入克隆开销的变化误报为密码学优化。

若完整操作吞吐低，才用独立 MSM 工具区分设备上限与 host 供给不足：

```bash
CUBBS_BBS_GPU_LIBRARY="$PWD/target/native/cuda/libcubbs_bbs_cuda.so" \
  target/release/cubbs-gpu-ceiling \
  --messages 16,64,128 --batch 8192 --streams 4 \
  --warmup-secs 1 --duration-secs 10
```

该工具在计时前准备 point/scalar/ABI buffers，在计时前后验证输出。
它不含完整消息映射、domain/challenge、签名/证明收尾，不能用于填充原语表。
诊断源码在 `benchmarks/gpu-ceiling/src/`：`config.rs` 解析配置，`workload.rs`
准备输入，`runner.rs` 执行与校验，`report.rs` 记录硬件、CPU 占用及吞吐。

## CPU 饱和测试

`benchmarks/cpu-baseline/` 的 wrapper 使用 `cubbs-four-op-reference`，
检查 adapter identity、验证结果与零失败，并拒绝复用已有原始 JSON 的输出目录。
测量前确认实际 NUMA/SMT 拓扑；默认 `0-63,128-191`、memory node 0 只适用于
对应拓扑，不能把它当成所有 EPYC 主机的固定编号。

```bash
STREAMS_LIST="32 64 96 128" \
MESSAGES="256" ISSUER_MODE=same DISCLOSED_RATIO=0.5 \
  benchmarks/cpu-baseline/run-host-sweep.sh \
  cpu-9554 /remote/path/cubbs-four-op-reference native \
  benchmarks/results/cpu-baseline/epyc9554/new-sweep
```

可用 `CUBBS_CPUSET`、`CUBBS_MEMORY_NODE` 覆盖绑定。多进程候选：

```bash
benchmarks/cpu-baseline/run-host-multiprocess.sh \
  cpu-9554 /remote/path/cubbs-four-op-reference native proof-gen \
  4 32 256 benchmarks/results/cpu-baseline/epyc9554/new-proof-gen-p4-s32
```

该 wrapper 将一个 NUMA 节点分区，拓扑可由 `CUBBS_PHYSICAL_CPU_BASE`、
`CUBBS_SMT_CPU_BASE`、`CUBBS_CORES_PER_NUMA` 和 `CUBBS_MEMORY_NODE` 覆盖。
最终结果必须保留各进程报告，并确认计时窗口重叠；不能只相加不同时间的峰值。
CPU 满载与重复结果稳定性是两个独立检查，达到线程数不代表已取得饱和容量。
已知服务器的连接/同步仍需遵循其访问和资源使用约束。

## HTTP 全流程回归

`benchmarks/w3c/full-flow/run-baseline.sh` 是持续四操作 workflow 回归入口，
不是 Table 2 的单操作容量入口。wrapper 默认使用 10K unique-VC manifest、
30 秒发起窗口、c24、500 ms startup spread、20 ms coalescing interval、
batch 128、queue 16384 和 16 个 JSON-LD worker。它们是该回归配置，
不是所有设备的推荐饱和参数；数据量不足时应另行准备更大的数据集。

```bash
benchmarks/w3c/full-flow/run-baseline.sh \
  --backend gpu --host-label rtx5060ti --run-id new-gpu-c24-d30
benchmarks/w3c/full-flow/run-baseline.sh \
  --backend cpu --host-label local-cpu --run-id new-cpu-c24-d30

benchmarks/w3c/full-flow/compare-runs.sh \
  benchmarks/w3c/full-flow/results/new-cpu-c24-d30/run.json \
  benchmarks/w3c/full-flow/results/new-gpu-c24-d30/run.json \
  benchmarks/w3c/full-flow/results/new-paired-comparison.json
```

`verify-results.sh` 保留的是固定 10K dataset、既定参数和历史 comparison 文件名
的回归契约，并非任意新 campaign 的通用 validator；使用前检查其目标路径。
吞吐必须包含启动到 drain 完成的总时间。phase-barrier 模式可用于确定性回归，
不能宣称是连续到达的服务容量。不要将旧 README 的历史 workflow 速率写入当前表格。

## 外部 W3C HTTP 对照

| 实现 | 适配器 | 默认端口 |
|---|---|---:|
| Digital Bazaar | `benchmarks/reference-w3c/digitalbazaar-http.mjs` | 40743 |
| Grotto | `benchmarks/reference-w3c/grotto-http.mjs` | 40543 |
| SpruceID | `benchmarks/reference-w3c/spruce-http/` | 40643 |
| Procivis One | 冻结 campaign 内的 formal adapter 与锁定依赖 | 由 campaign 指定 |

依赖按各实现的锁定版本安装，源码位于本地 `ref/implementations/`：

```bash
NODE_OPTIONS=--experimental-global-webcrypto \
  node benchmarks/reference-w3c/digitalbazaar-http.mjs
node benchmarks/reference-w3c/grotto-http.mjs
cargo run --release --manifest-path benchmarks/reference-w3c/spruce-http/Cargo.toml
```

这些是分别启动的长期进程。用同一单操作 HTTP load generator 测量，例如：

```bash
cargo run --release -p cubbs-bench --bin w3c-http-op-load -- \
  --operation derive --url http://127.0.0.1:40643/credentials/derive \
  --payload /path/to/derive-request.json --expect document \
  --concurrency 128 --duration-secs 30
```

适配器不引入跨请求 BBS batch。只比较各实现实际支持的端点，unsupported 记为
`N/S`，不能把接口不支持当成零吞吐。正确性入口为
`benchmarks/reference-w3c/verify-implementations.sh`。

## Dataset 与结果契约

可复现 W3C workload 的 manifest 位于
`benchmarks/w3c/datasets/<dataset-id>/manifest.json`，声明 schema version、
dataset ID、说明、entry count、issuer/shape/feature 分布、确定性 seed、
生成 JSONL 的相对路径与 SHA-256。unique-input workload 每行是完整 Issue
请求，`credential.id` 不重复，预热和正式计时消耗互不重叠的输入区间。
复用工作集的 cache 实验须另行声明规模、冷热状态和命中率。

每次 run 保存：

- fixture/manifest digest、源码 commit、二进制与依赖版本；
- 完整服务/负载命令、硬件、NUMA/affinity、driver/toolchain；
- requested/active backend、hardware acceleration、fallback 与 route telemetry；
- message、issuer、disclosure、batch、streams、并发、缓存与队列配置；
- warm-up、计时时间、drain、成功/失败、吞吐和延迟分位数；
- HTTP `/healthz` 前后快照、实际 batch 分布及 cache/timing delta。

原始每次测量不能被汇总 CSV 取代。保留 batch、streams、并发、重复次数、
统计量和 CV 等信息；冻结口径使用 median 时不能在描述中改称 mean。
同一 GPU 不并发运行相互干扰的性能实验。比较 kernel 优化时固定其余配置，
不能把增加 streams 带来的饱和提升写成单 kernel 加速。

## 冻结数据与本地目录

- `paper-figures/table1/table_bbs_capacity.csv`：原语表。
- `paper-figures/table2/table_w3c_service_reviewed_source_long.csv`：服务表长表。
- `paper-figures/fig8/fig8_source_data.csv`：Fig. 8 数据。
- `ref/results/paper-archive/`：不可变 campaign、原始报告及被替代版本。
- `benchmarks/baselines/`：明确标记为历史来源的锁定基线，不能冒充当前运行。
- `benchmarks/bbs/results/`、`benchmarks/w3c/results/` 等 runner 输出目录：
  本地生成产物，验收后归档到 `ref/results/`。
- `ref/research/`：本地文献；`ref/implementations/`：构建/对照实现依赖。

以上生成数据及论文包不随源码清理而删除。新 run 使用新目录，不覆盖冻结结果。
Git 不跟踪 `ref/` 和 `paper-figures/` 的产物不等于它们可以丢弃。

保留原有来源警告：历史原语 CSV 的 CPU 列并非全部已经由可追溯的 0.6.1
报告覆盖。只有报告同时记录 `cubbs-four-op-reference`、`apiMode=single`、
`active=zkryptium-0.6.1`、`fallback=false`，并通过正确性检查，才能据此标为
该 comparator。本次整理不重测、不替换数值，也不更改这些结果的来源身份。
