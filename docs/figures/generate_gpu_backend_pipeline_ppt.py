#!/usr/bin/env python3
"""Generate an editable, GPU-backend-only architecture deck for cuBBS-Cred."""

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from generate_route_comb_ppt import (
    BG, BLUE, GREEN, MUTED, ORANGE, PANEL, PANEL_2, RED, TEAL, WHITE,
    arrow, card, footer, line, rgb, set_bg, text_box, title,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "cubbs-cred-gpu-backend-pipeline.zh-CN.pptx"


def panel(slide, x, y, w, h, label, color=BLUE, fill="091B2C"):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(1.1)
    text_box(slide, x + 0.14, y + 0.04, w - 0.28, 0.25, label, 9.1, color, True)
    return shape


def pill(slide, x, y, w, text, color=TEAL, fill="11283C", size=8.2):
    return card(slide, x, y, w, 0.34, text, "", fill, color,
                title_size=size, radius=True)


def slide_pipeline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "cuBBS-Cred GPU 后端：请求如何变成 GPU 并行任务",
          "核心不是“把一个 HTTP 请求扔给 GPU”，而是把兼容请求整理成矩阵/批任务，再复用点表并一次产生 N 个结果")

    # Stage 0/1: requests and operation queues
    panel(slide, 0.40, 1.27, 3.02, 2.34, "① REQUEST INTAKE + OPERATION BATCH", BLUE)
    ops = [("Sign", TEAL), ("Verify", BLUE), ("ProofGen", ORANGE), ("ProofVerify", GREEN)]
    for i, (name, color) in enumerate(ops):
        x = 0.64 + (i % 2) * 1.28
        y = 1.70 + (i // 2) * 0.55
        pill(slide, x, y, 1.12, name, color, "12283C", 8.8)
    card(slide, 0.64, 2.90, 2.53, 0.45, "Coalesce", "interval / max batch / capacity；保存 original index", "103441", TEAL,
         title_size=10, body_size=7.8, accent=TEAL)

    arrow(slide, 3.48, 2.23, "right", TEAL, 0.85)

    # Stage 2: exact shape keys
    panel(slide, 3.88, 1.27, 4.20, 2.34, "② COMPATIBILITY BUCKETS · 第一级分组", TEAL)
    card(slide, 4.12, 1.66, 1.74, 0.66, "Sign", "header + message_count L", PANEL, TEAL,
         title_size=10.5, body_size=8.1, accent=TEAL)
    card(slide, 6.02, 1.66, 1.82, 0.66, "ProofGen", "L + disclosed_indexes", PANEL, ORANGE,
         title_size=10.5, body_size=8.1, accent=ORANGE)
    card(slide, 4.12, 2.50, 1.74, 0.66, "ProofVerify", "total L + D + indexes", PANEL, GREEN,
         title_size=10.5, body_size=8.1, accent=GREEN)
    card(slide, 6.02, 2.50, 1.82, 0.66, "Verify", "直接进入 primitive grouping", PANEL, BLUE,
         title_size=10.5, body_size=8.1, accent=BLUE)
    text_box(slide, 4.10, 3.25, 3.72, 0.22,
             "issuer 不是统一 key；最终看实际 point set 是否可复用",
             8.2, MUTED, True, PP_ALIGN.CENTER)

    arrow(slide, 8.14, 2.23, "right", TEAL, 0.85)

    # Stage 3: canonical jobs
    panel(slide, 8.53, 1.27, 4.40, 2.34, "③ JOB BUILDER · 第二级分组", ORANGE)
    card(slide, 8.79, 1.65, 1.83, 0.72, "Common-points", "P[K] shared\nS[N×K] scalar matrix", "103441", TEAL,
         title_size=10.5, body_size=8.5, accent=TEAL)
    card(slide, 10.82, 1.65, 1.83, 0.72, "Variable/fused", "P[N×K]\nper-request points", PANEL_2, BLUE,
         title_size=10.5, body_size=8.5, accent=BLUE)
    card(slide, 8.79, 2.56, 1.83, 0.62, "H2S batch", "flat messages → Fr", "2A2B32", ORANGE,
         title_size=10.3, body_size=8.1, accent=ORANGE)
    card(slide, 10.82, 2.56, 1.83, 0.62, "Pairing batch", "equation inputs", "2D2832", GREEN,
         title_size=10.3, body_size=8.1, accent=GREEN)
    text_box(slide, 8.78, 3.25, 3.88, 0.22,
             "按 exact points / term count / layout 决定复用与 route",
             8.2, MUTED, True, PP_ALIGN.CENTER)

    card(slide, 3.94, 3.83, 5.44, 0.48, "CUDA ABI v2 · 一桶一次提交", "packed descriptors · flat scalars / S[N×K] · compressed points · original indexes", "18314A", TEAL,
         title_size=11.3, body_size=8.1, accent=TEAL)
    line(slide, 10.70, 3.61, 8.30, 3.83, TEAL, 1.5)
    arrow(slide, 6.56, 4.35, "down", TEAL, 0.8)

    # GPU execution plane
    panel(slide, 0.40, 4.70, 12.53, 2.20, "④ GPU EXECUTION · 这里才是真正的加速后端", TEAL, "081D27")
    card(slide, 0.67, 5.11, 2.10, 1.06, "H2S kernels", "一批 message 并行 SHA-256/XMD\n输出连续 Fr scalars", "103441", ORANGE,
         title_size=12.5, body_size=8.9, accent=ORANGE)

    card(slide, 3.02, 5.02, 6.95, 1.25, "", "", "102A43", BLUE,
         title_size=12.4, accent=BLUE)
    routes = [
        ("MSM R1\nvariable/fused", BLUE),
        ("MSM R2\ncommon tuned\ncache point table", TEAL),
        ("MSM R3\nlarge tiled", ORANGE),
        ("MSM R4\nuniversal CUDA", "94A9BF"),
    ]
    for i, (label, color) in enumerate(routes):
        card(slide, 3.23 + i * 1.63, 5.57, 1.47, 0.52, label, "", "11283C", color,
             title_size=7.9, radius=True)

    card(slide, 10.21, 5.11, 2.43, 1.06, "Pairing kernels", "batch / aggregate equation\n与 MSM/host prep 可重叠", "2D2832", GREEN,
         title_size=12.2, body_size=8.9, accent=GREEN)

    card(slide, 0.67, 6.39, 11.97, 0.27,
         "复用层：point-table / prepared-state cache · reusable device buffers · per-point-set nonblocking streams · route telemetry", "",
         "0D2930", "3A8D82", title_size=8.5, accent="3A8D82")
    text_box(slide, 9.74, 4.33, 2.84, 0.20, "compressed G1 / scalar / status → finalize + reorder", 8.0, MUTED, True, PP_ALIGN.RIGHT)
    footer(slide, 1)


def slide_why_faster(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "为什么这种 batch 架构会加速",
          "加速来源不是 BBS 算法改变，而是把 N 次重复控制/传输/预计算变成一次，并把 N 行算术同时铺到 GPU")

    panel(slide, 0.48, 1.38, 5.84, 4.30, "WITHOUT BACKEND BATCHING · N 次独立调用", RED, "241B29")
    y = 1.85
    slow = [
        ("Request 0", "decode points → H2D → precompute → launch → sync"),
        ("Request 1", "decode points → H2D → precompute → launch → sync"),
        ("…", "重复相同 generator/point 工作"),
        ("Request N−1", "decode points → H2D → precompute → launch → sync"),
    ]
    for i, (head, body) in enumerate(slow):
        card(slide, 0.80, y + i * 0.79, 5.18, 0.61, head, body, "1A2636", RED,
             title_size=10.4, body_size=8.6, accent=RED)
    text_box(slide, 0.84, 5.14, 5.05, 0.27,
             "GPU 每次只拿到很小工作量；launch/H2D/sync 占比高",
             10.0, RED, True, PP_ALIGN.CENTER)

    panel(slide, 6.68, 1.38, 6.17, 4.30, "OMNI BATCHED BACKEND · 一桶一次提交", TEAL, "0B2B30")
    card(slide, 7.02, 1.83, 5.49, 0.64, "1. Point reuse", "公共 P 只上传/解码/预计算一次，并由 cache 跨 batch 复用", "102D3A", TEAL,
         title_size=10.7, body_size=8.8, accent=TEAL)
    card(slide, 7.02, 2.66, 5.49, 0.64, "2. Packed transfer", "所有 scalar rows 合并成 S[N×K]；一次 H2D，而不是 N 次小拷贝", "102D3A", BLUE,
         title_size=10.7, body_size=8.8, accent=BLUE)
    card(slide, 7.02, 3.49, 5.49, 0.64, "3. Multi-output kernel", "一个 batched launch 产生 R₀…Rₙ₋₁；row/block 并行填满 SM", "102D3A", ORANGE,
         title_size=10.7, body_size=8.8, accent=ORANGE)
    card(slide, 7.02, 4.32, 5.49, 0.64, "4. Mixed-shape overlap", "不同 point-set bucket 使用独立 stream；GPU 计算时 CPU prepare/finalize 可并行", "102D3A", GREEN,
         title_size=10.7, body_size=8.8, accent=GREEN)
    text_box(slide, 7.04, 5.18, 5.45, 0.26,
             "结果：固定成本被 N 个请求摊薄，GPU 算术占比上升",
             10.2, TEAL, True, PP_ALIGN.CENTER)

    card(slide, 0.48, 5.93, 12.37, 0.73,
         "直观成本模型",
         "独立调用：N·(Tlaunch + Tpoints-H2D + Tprecompute + Trow)    →    batched：Tgroup + Tscalar-H2D + Tlaunch + parallel_rows(N) + amortized(Tcache)",
         "152339", "566F8B", title_size=11.0, body_size=10.0, accent="566F8B")
    card(slide, 0.48, 6.80, 12.37, 0.27,
         "边界：N 太小、point set 全不同、W3C/CPU prepare 占主导时，GPU 仍可能喂不满；所以 benchmark 必须报告 effective batch、point reuse、route 和 queue wait。", "",
         "2B2631", ORANGE, title_size=8.7, accent=ORANGE)
    footer(slide, 2)


def slide_batch_keys(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "Batch 到底按什么区分",
          "先保证协议/shape 兼容，再看底层点布局；这两层不能合并成一个简单的 issuer key")

    headers = ["Operation", "第一级 shape bucket（当前代码）", "第二级 primitive grouping", "GPU 形成的主要工作"]
    widths = [1.55, 3.58, 4.15, 3.08]
    x0, y0 = 0.48, 1.44
    x = x0
    for label, w in zip(headers, widths):
        card(slide, x, y0, w, 0.54, label, "", "17324A", "355D7B", title_size=9.6, radius=False)
        x += w
    rows = [
        ("Sign", "header + message_count L", "exact common generator points；final/variable terms 单独处理", "H2S + B common MSM + sign finalize", TEAL),
        ("Verify", "无独立 mixed-shape key；直接准备 primitive terms", "按 compressed point set / term count 分 common 或 variable", "H2S + B MSM + pairing", BLUE),
        ("ProofGen", "message_count L + disclosed_indexes", "B/T2 common generators；proof-specific D/Abar/Bbar/T1", "H2S + common MSM + fused MSM", ORANGE),
        ("ProofVerify", "total L + disclosed_count D + disclosed_indexes", "T2 hidden generators 与 Abar/Bbar/D variable terms 分解", "H2S + paired MSM + pairing", GREEN),
    ]
    for ri, row in enumerate(rows):
        y = y0 + 0.54 + ri * 0.87
        x = x0
        for ci, (value, w) in enumerate(zip(row[:4], widths)):
            fill = "10273C" if ri % 2 == 0 else "0D2235"
            box = card(slide, x, y, w, 0.87, value, "", fill, "294860", title_size=9.1, radius=False)
            if ci == 0:
                box.text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(row[4])
            x += w

    card(slide, 0.48, 5.68, 4.02, 0.91, "Issuer / public key", "不是统一的一级 bucket key。不同 issuer 仍可能共享标准 generators；但 issuer-specific pairing/variable data 不复用。", PANEL, ORANGE,
         title_size=12, body_size=9.2, accent=ORANGE)
    card(slide, 4.65, 5.68, 4.02, 0.91, "Common-points 判断", "真正决定 Route 2/3 的是 P[K] 是否逐字节相同，以及 K、N 和 cache memory budget。", PANEL, TEAL,
         title_size=12, body_size=9.2, accent=TEAL)
    card(slide, 8.82, 5.68, 4.02, 0.91, "Variable/fused 判断", "每项 points 不同但代数形态和 term count 相同，可作为 item batch 进入 Route 1。", PANEL, BLUE,
         title_size=12, body_size=9.2, accent=BLUE)
    card(slide, 0.48, 6.78, 12.36, 0.28,
         "因此：同一 HTTP batch 可以被拆成多个 GPU bucket；每个 bucket 独立执行，最终按 original index 合并。", "",
         "103441", TEAL, title_size=9.2, accent=TEAL)
    footer(slide, 3)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "cuBBS-Cred GPU Backend Pipeline"
    prs.core_properties.subject = "Request bucketing, GPU execution and acceleration mechanism"
    prs.core_properties.author = "cuBBS-Cred"
    prs.core_properties.keywords = "cuBBS-Cred, BBS, GPU, CUDA, batch, MSM, architecture"
    slide_pipeline(prs)
    slide_why_faster(prs)
    slide_batch_keys(prs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    build()
