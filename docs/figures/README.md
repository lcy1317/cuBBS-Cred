# 文档图源

此目录保存说明当前设计所需的图，以及可编辑的绘图工具。
它不是 `paper-figures/` 冻结性能图表包，不更改论文数据或最终论文配图。

| 主题 | 预览 | 可编辑源 |
|---|---|---|
| GPU 后端 | [SVG](gpu-backend.svg) | [Mermaid](gpu-backend.mmd)、`generate_gpu_backend_paper_ppt.py` |
| MSM 路由位置 | [SVG](msm-routes.svg) | [Mermaid](msm-routes.mmd) |
| 四操作 GPU 分工 | [SVG](four-operations.svg) | [Mermaid](four-operations.mmd)、[源 SVG](four-operations.source.svg) |
| 普通及托管 HTTP 流程 | [SVG](w3c-http.svg) | [Mermaid](w3c-http.mmd) |

Mermaid、源 SVG 与 PPT 是编辑入口，不保证不同格式自动往返后逐像素一致。
旧的 overview/balanced/engineering/paper-v2/paper-v3 等候选布局不再作为文档入口。

## 可编辑 PPT 工具

以下工具原位于 `tools/`，现在与图源放在一起；它们不参与生产构建或性能测试。

- `generate_gpu_backend_paper_ppt.py`：论文式后端组件图。
- `generate_route_comb_ppt.py`：comb kernel 图和共用绘图 helper。
- `generate_gpu_backend_pipeline_ppt.py`：详细数据流水线讲解。
- `generate_section2_gpu_architecture_ppt.py`：宏观系统/W3C 视图。

后两项导入 comb 脚本的 helper，必须作为同一组维护。保留这些可编辑来源，
不把“未被构建脚本调用”误判为“不再用于论文编辑”。

在已安装 `python-pptx` 的环境中，从仓库根目录执行，例如：

```bash
python3 docs/figures/generate_gpu_backend_paper_ppt.py
```

输出统一写入本目录的 `generated/`，不进入 Git，也不覆盖冻结论文图。
