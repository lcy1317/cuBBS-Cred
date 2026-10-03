# 协议实现来源与修改边界

`crates/cubbs-bbs-protocol` 是 LINKS Foundation ZKryptium 0.2.2 的
Apache-2.0 派生实现，上游 commit 为
`0e21c20f4c84473e7eb69a1aef136159c9d085b8`。
原始 manifest 保存在 `crates/cubbs-bbs-protocol/UPSTREAM-Cargo.toml`；
源码保留原始版权及许可证，并标注 cuBBS-Cred 修改。

上游：<https://github.com/Cybersecurity-LINKS/ZKryptium>。

本地修改包括 draft-10 Sign transcript、prehashed/precomputed lowering hooks、
prepared-state cache 和 GPU planner 使用的 batch 接口。生产 GPU 协议 lowering
依赖此 crate，不是直接运行 registry ZKryptium 0.6.1。
0.6.1 是独立 CPU reference / 可恢复 GPU fallback 的默认 adapter；
registry 0.2.2 adapter 仅作为独立版本化的 legacy comparison 保留。

## 向量与范围

baseline BBS `fixture_data` 与仓库 CFRG draft-10 snapshot 内容对应，
除规范化换行外一致；测试校验 SHA-256 与 SHAKE-256 的精确输出。
该 snapshot 的 commit、archive digest 和验收命令见[正确性验证](testing.md)。

可选 blind-proof fixtures 仍为上游 draft-05 资产，其测试检验生成证明的可验证性，
不将旧 proof bytes 当作 draft-10 oracle。Blind issuance 不属于 cuBBS-Cred
对外四个基础操作。协议 crate 的全部功能也不等于公共 GPU 后端的支持范围。

## 实现归属

cuBBS-Cred 实现 BBS-aware batching/lowering、ABI v2、Route 1–4 调度与
专用 kernel、workspace/cache/telemetry 和 HTTP 组批；不宣称发明 BBS、
BLS12-381、GLV、Pippenger、comb 或一般批处理技术。
BLST 提供 CPU/host 曲线设施，SPPARK 提供 CUDA 域/曲线模板；其 Pippenger
仅用于显式开启的 reference routes。完整版本、版权和发布义务以
[THIRD_PARTY.md](../THIRD_PARTY.md) 及源文件声明为准。

## 仓库历史

cuBBS-Cred 初始主线合并了前身 GPU/W3C 服务历史 `5542f47`、Verify 聚合与
Route 4 历史 `a13732e` 及当时通过验证的工作区修改。原始生成结果和对照源码
保存在本地 `ref/`，不属于源码 Git 分发。
