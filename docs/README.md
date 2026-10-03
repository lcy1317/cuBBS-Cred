# cuBBS-Cred 文档

项目说明统一在此维护；以下命令及源码路径均以仓库根目录为基准。
每个主题只保留一个维护入口，实验数值以冻结结果文件为准，不在多份说明中复制。

| 主题 | 入口 |
|---|---|
| 编译、部署、HTTP 调用与发布 | [构建运行指南](build-run-http-benchmark.zh-CN.md) |
| 后端边界、调度、ABI 与源码导航 | [GPU 后端设计](gpu-bbs-backend-design.md) |
| 四操作公式与安全不变量 | [BBS 四操作](bbs-four-operation-cryptography.zh-CN.md) |
| MSM 数据布局、Route 1–8 与遥测 | [MSM 路由](msm-route-production-design-v2.zh-CN.md) |
| HTTP 端点、组批、缓存与托管模式 | [W3C 服务](w3c-service.md) |
| 原语、HTTP 和 CPU 基线的测量与归档 | [测试与复现指南](benchmarks.md) |
| CPU/GPU 正确性、规范向量与验收 | [正确性验证](testing.md) |
| 协议派生版本及修改边界 | [协议来源](protocol-provenance.md) |
| 文档图源与可编辑绘图工具 | [图源说明](figures/README.md) |

## 文档与产物边界

- 根目录 `README.md` 只提供项目入口。
- `LICENSE`、`NOTICE`、`THIRD_PARTY.md`、`LICENSES/` 和第三方版权声明保留原位，
  不作为重复开发文档清理。
- `paper-figures/` 是本地冻结图表包；`ref/results/paper-archive/` 保存实验原始证据。
  其中的结果说明、配置、版本与校验清单属于实验产物，不迁移或重写。
- `ref/implementations/` 包含构建及外部对比所需的独立源码，不属于弃用目录。
- 已确认的旧调优报告、重复说明和候选图稿已清理，历史内容可从 Git 恢复。
  尚存的未列入索引旧稿不作为当前接口或性能依据。

文档整理后运行 `python3 tools/check_docs.py` 检查本地链接、Cargo readme 与弃用路径引用。
