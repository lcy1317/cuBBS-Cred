# 正确性验证

性能数据冻结不替代当前代码验证。以下命令从仓库根目录执行；一次旧通过记录
不能自动视为当前版本已通过。构建见[运行指南](build-run-http-benchmark.zh-CN.md)。

## CPU 与服务回归

```bash
cargo test --locked --workspace
cargo check --locked -p cubbs-bench --bins
python3 tools/check_docs.py
```

硬件相关 `loaded_gpu_` 测试被显式忽略；默认测试通过不代表 GPU 向量已执行。
上游保留的 ignored 用例和硬件用例必须在报告中区分。

## Native 与真实 GPU

在具备驱动且获准使用的 GPU 上，加载与本次源码匹配的 shared library：

```bash
make -C native/cuda CUDA_ARCH=native smoke
CUBBS_BBS_GPU_LIBRARY="$PWD/target/native/cuda/libcubbs_bbs_cuda.so" \
CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 \
  cargo test --release --locked -p cubbs-bbs --lib loaded_gpu_ -- --ignored --nocapture --test-threads=1
CUBBS_BBS_GPU_LIBRARY="$PWD/target/native/cuda/libcubbs_bbs_cuda.so" \
CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 \
  cargo test --release --locked -p cubbs-vc-di-bbs --lib loaded_gpu_ -- --ignored --nocapture --test-threads=1
```

这些测试要求真实 CUDA、无静默 CPU fallback。GPU/CPU 比较需包括：

- 确定性 Sign、H2S、MSM 中间值及编码的精确比较；
- 随机 ProofGen 的双向交叉验证，固定测试随机数时的向量比较；
- 篡改消息、签名、证明、header 与披露索引后的拒绝；
- mixed shape/issuer 的输出顺序、逐项错误隔离和 aggregate 失败后定位；
- 256-message 的披露端点，以及 65,536-message 四操作边界；
- 非法 scalar、点/子群、长度/容量与 ABI version 的拒绝。

W3C lowering 回归还覆盖隐藏末尾消息：BBS 总消息数由证明长度中的隐藏消息响应
数量与披露数量相加得到，不能使用最大披露索引加一。单条、组批与 managed
验证共用这一准备路径。

Route 4 的扩展 native smoke：

```bash
CUBBS_BBS_GPU_ULTRA_SMOKE=1 make -C native/cuda smoke
```

其中 65,538-term MSM 应留在生产 CUDA route，而不是被静默替换成 CPU MSM。
Route 选择与预算见 [MSM 路由](msm-route-production-design-v2.zh-CN.md)。

## CFRG / IETF BBS 冻结向量

向量保持在 `CFRG-IETF-BBS/`，来源和身份如下：

- 上游：<https://github.com/decentralized-identity/bbs-signature>
- Commit：`a6a445dd3c2a382fb8affa08a4bc077bd27bf753`
- 下载日期：2026-07-18
- 上游 archive SHA-256：
  `1d6809598cc9174c651513b07f0177b424fdf9b80649dcf6aa17ac75b433c1ea`

保留 SHA-256 和 SHAKE-256 fixtures 及其原始 LICENSE。公共 `cubbs-bbs`
后端对外使用 SHA-256 suite；派生 protocol crate 的底层向量覆盖另见
[协议来源](protocol-provenance.md)。不要把两层覆盖混为一谈。

```bash
cargo run --release --locked -p cubbs-bench --bin cfrg-ietf-conformance -- \
  --backend cpu --strict --output CFRG-IETF-BBS/results/cpu-latest.json

CUBBS_BBS_GPU_LIBRARY="$PWD/target/native/cuda/libcubbs_bbs_cuda.so" \
CUBBS_BBS_FORCE_GPU=1 \
cargo run --release --locked -p cubbs-bench --bin cfrg-ietf-conformance -- \
  --backend gpu --strict --output CFRG-IETF-BBS/results/gpu-latest.json
```

`--strict` 在失败或规范向量被公共 facade 提前阻断时返回非零。报告应区分
passed、failed、blocked 和未执行；不能把 CPU 通过写成 GPU 通过。
生成的 `latest.json` 仅供当前验收；正式冻结报告另存新 campaign 目录。
旧报告在 `ref/results/paper-archive/payload/correctness/cfrg-ietf-bbs/results/`。

## W3C 服务验收

按运行指南完成 Issue → Base Verify → Derive → Proof Verify 闭环，并检查
`/healthz` 的 requested/active backend、ABI、fallback、队列和 cache 配置。
使用外部 `vc-di-bbs-test-suite` 时保存其版本、配置、完整报告和服务 build identity。
HTTP 成功或 CUDA 动态库成功加载，均不能单独证明全部密码学阶段运行在 GPU。

仓库提供 `benchmarks/w3c/conformance/localConfig.cjs`，仅选择本机 cuBBS-Cred
实现并开启互操作测试，不修改上游测试、fixture 或其原有 `localConfig.cjs`。
服务使用 `CUBBS_W3C_VERIFY_CACHE_CAPACITY=0`，避免验证结果缓存代替计算。
在服务已于端口 40444 启动后，可使用上游已安装的 Mocha：

```bash
# CUBBS_SUITE 指向已安装依赖的上游 vc-di-bbs-test-suite checkout（绝对路径）。
CUBBS_CONFIG_ROOT="$PWD/benchmarks/w3c/conformance"
CUBBS_SUITE_RUN=$(mktemp -d /tmp/cubbs-w3c.XXXXXX)
cp -a "$CUBBS_SUITE/tests" "$CUBBS_SUITE/config" "$CUBBS_SUITE/package.json" "$CUBBS_SUITE_RUN/"
ln -s "$CUBBS_SUITE/node_modules" "$CUBBS_SUITE_RUN/node_modules"
mkdir "$CUBBS_SUITE_RUN/reports"
jq '.suites["bbs-2023"].vcHolder.holderName = "cuBBS-Cred"' \
  "$CUBBS_SUITE/config/runner.json" > "$CUBBS_SUITE_RUN/config/runner.json"
cd "$CUBBS_SUITE_RUN"
APP_ROOT_PATH="$CUBBS_CONFIG_ROOT" \
CUBBS_TEST_BASE_URL=http://127.0.0.1:40444/id \
node --experimental-global-webcrypto node_modules/mocha/bin/mocha.js tests/ \
  --reporter json --timeout 30000 > /tmp/cubbs-w3c-conformance.json
```

只在隔离副本中设置被测 holder 名称；上游测试、fixture、原 checkout 的配置不动。
保留该副本及完整报告用于追溯。Node 18 的 WebCrypto 开关只补充运行环境，
不更改测试断言。此配置检查本地实现的规范用例和 Issue/Derive/Verify 互通，
不代表已遍历所有远程第三方实现组合。

冻结性能结果只接纳通过正确性检查、无失败的测量；规范向量、交叉验证和负例
各有用途，不能相互替代。CPU/GPU 数值比较与硬件饱和测试的边界见
[测试与复现指南](benchmarks.md)。
