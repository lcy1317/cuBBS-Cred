#!/usr/bin/env python3
"""Generate editable Section 2 macro architecture slides for cuBBS-Cred."""

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from generate_route_comb_ppt import (
    BG, BLUE, FONT, GREEN, GRID, MUTED, ORANGE, PANEL, PANEL_2, PANEL_3,
    RED, TEAL, TEAL_DARK, WHITE, arrow, card, footer, line, rgb, set_bg,
    text_box, title,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "cubbs-cred-section2-gpu-architecture.zh-CN.pptx"


def layer_panel(slide, x, y, w, h, label, label_color, fill="091B2C", stroke="29455E"):
    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = rgb(fill)
    panel.line.color.rgb = rgb(stroke)
    panel.line.width = Pt(1.1)
    text_box(slide, x + 0.16, y + 0.05, w - 0.32, 0.28, label, 9.5, label_color, True)
    return panel


def small_pill(slide, x, y, w, text, color=TEAL, fill="10273B", size=8.4):
    return card(slide, x, y, w, 0.34, text, "", fill, color,
                title_size=size, radius=True)


def slide_w3c_end_to_end(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "A · cuBBS-Cred 端到端 GPU 架构（包含 W3C）",
          "论文 Section 2 系统视角：并发 HTTP 请求被聚合为兼容 batch，再下沉为 GPU 密码学原语")

    layer_panel(slide, 0.42, 1.28, 12.49, 2.48, "CPU CONTROL PLANE · W3C / SERVICE / SCHEDULER / PROTOCOL", BLUE)

    card(slide, 0.66, 1.68, 2.17, 1.20, "并发请求", "W3C Data Integrity\nHTTP single / batch API\nissue · verify · derive", PANEL_2, BLUE,
         title_size=14, body_size=10.1, accent=BLUE)
    arrow(slide, 2.88, 2.16, "right", TEAL, 0.8)
    card(slide, 3.20, 1.68, 2.44, 1.20, "W3C 服务层", "JSON-LD / RDFC\nframing / disclosed set\nvalidation / encoding", PANEL, "5E7F9D",
         title_size=14, body_size=10.1, accent="5E7F9D")
    arrow(slide, 5.69, 2.16, "right", TEAL, 0.8)
    card(slide, 6.01, 1.68, 2.73, 1.20, "操作感知微批调度", "4 条 operation queue\ninterval / max batch / capacity\npipeline slots + backpressure", "103441", TEAL,
         title_size=13.2, body_size=9.7, accent=TEAL)
    arrow(slide, 8.80, 2.16, "right", TEAL, 0.8)
    card(slide, 9.12, 1.68, 3.54, 1.20, "Rust BBS protocol runtime", "Sign · Verify · ProofGen · ProofVerify\nshape grouping + prepare / finalize\n恢复原请求顺序并返回", PANEL_2, BLUE,
         title_size=13.2, body_size=9.8, accent=BLUE)

    small_pill(slide, 0.70, 3.10, 2.06, "API compatibility boundary", "5E7F9D")
    small_pill(slide, 3.25, 3.10, 2.34, "W3C 语义不进入 CUDA", "5E7F9D")
    small_pill(slide, 6.07, 3.10, 2.60, "按 operation + compatible shape 分组", TEAL)
    small_pill(slide, 9.18, 3.10, 3.42, "canonical H2S / MSM / pairing batches", BLUE)

    line(slide, 10.88, 2.88, 10.88, 3.89, TEAL, 2.2)
    arrow(slide, 10.79, 3.67, "down", TEAL, 0.8)
    card(slide, 3.58, 3.86, 6.34, 0.49, "CUDA ABI v2", "packed descriptors · scalar matrices · compressed points · per-item status · route telemetry", "18314A", TEAL,
         title_size=12.3, body_size=8.8, accent=TEAL)

    layer_panel(slide, 0.42, 4.49, 12.49, 2.43, "GPU EXECUTION PLANE · BATCHED CRYPTOGRAPHIC PRIMITIVES", TEAL, "081D27", "24615E")
    card(slide, 0.67, 4.91, 2.30, 1.10, "H2S / SHA-256", "message-to-scalar batch\ndomain-separated hashing", "103441", TEAL,
         title_size=14, body_size=9.8, accent=TEAL)

    card(slide, 3.22, 4.81, 6.72, 1.30, "G1 MSM dispatcher", "", "102A43", BLUE,
         title_size=14.5, body_size=9.3, accent=BLUE)
    route_specs = [
        ("R1 variable/fused", BLUE),
        ("R2 tuned common · comb inside", TEAL),
        ("R3 large tiled", ORANGE),
        ("R4 universal", "94A9BF"),
    ]
    for i, (label, color) in enumerate(route_specs):
        small_pill(slide, 3.43 + i * 1.58, 5.58, 1.43, label, color, "12283D", 7.5)

    card(slide, 10.20, 4.91, 2.45, 1.10, "Pairing equation", "signature/proof relation\nbatched Miller/final exponent", "2D2832", ORANGE,
         title_size=13.5, body_size=9.5, accent=ORANGE)

    card(slide, 0.67, 6.26, 11.98, 0.40,
         "持久化 GPU 资源", "generator / point-table / prepared-state cache · reusable scalar/output buffers · per-point-set nonblocking streams",
         "0D2930", "3A8D82", title_size=9.4, body_size=8.3, accent="3A8D82")

    line(slide, 1.82, 4.35, 1.82, 4.91, TEAL, 1.6)
    line(slide, 6.58, 4.35, 6.58, 4.81, BLUE, 1.6)
    line(slide, 11.42, 4.35, 11.42, 4.91, ORANGE, 1.6)
    text_box(slide, 10.02, 3.98, 2.72, 0.22, "GPU results → host finalize", 8.8, MUTED, True, PP_ALIGN.RIGHT)
    footer(slide, 1)


def slide_four_primitives(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "B · cuBBS-Cred 四原语 GPU 核心架构（不包含 W3C）",
          "密码学后端视角：四个公开 API 共享 prepare → primitive batches → CUDA execution → finalize 骨架")

    ops = [
        ("Sign", "signature", TEAL),
        ("Verify", "valid / invalid", BLUE),
        ("ProofGen", "selective-disclosure proof", ORANGE),
        ("ProofVerify", "valid / invalid", GREEN),
    ]
    for i, (name, out, color) in enumerate(ops):
        x = 0.52 + i * 3.18
        card(slide, x, 1.34, 2.80, 0.72, name, "input → " + out, PANEL_2, color,
             title_size=14, body_size=9.3, accent=color)
        line(slide, x + 1.40, 2.06, x + 1.40, 2.31, color, 1.5)

    layer_panel(slide, 0.42, 2.28, 12.49, 1.68, "RUST PROTOCOL RUNTIME · OPERATION-SPECIFIC CONTROL", BLUE)
    card(slide, 0.67, 2.68, 2.43, 0.92, "① Decode / validate", "key · point · scalar · index\nlength / range / transcript inputs", PANEL, "6988A6", title_size=12.5, body_size=8.8, accent="6988A6")
    arrow(slide, 3.14, 3.01, "right", TEAL, 0.75)
    card(slide, 3.46, 2.68, 2.57, 0.92, "② Prepare", "messages → Fr · domain\nproof state · common/variable terms", PANEL, TEAL, title_size=12.5, body_size=8.8, accent=TEAL)
    arrow(slide, 6.07, 3.01, "right", TEAL, 0.75)
    card(slide, 6.39, 2.68, 2.60, 0.92, "③ Group / plan", "operation + compatible shape\ncommon / fused / large / untuned", PANEL, BLUE, title_size=12.5, body_size=8.8, accent=BLUE)
    arrow(slide, 9.03, 3.01, "right", TEAL, 0.75)
    card(slide, 9.35, 2.68, 3.30, 0.92, "④ Finalize", "challenge / response · point encoding\npairing + challenge 双条件判定", PANEL, ORANGE, title_size=12.5, body_size=8.8, accent=ORANGE)

    card(slide, 3.58, 4.12, 6.34, 0.47, "CUDA ABI v2 · canonical primitive batches", "H2S batch · packed G1 MSM jobs · pairing-equation batch · status / telemetry", "18314A", TEAL,
         title_size=11.6, body_size=8.4, accent=TEAL)
    line(slide, 7.68, 3.60, 7.68, 4.12, TEAL, 1.8)

    layer_panel(slide, 0.42, 4.76, 12.49, 2.08, "GPU DATA PLANE · SHARED BY ALL FOUR OPERATIONS", TEAL, "081D27", "24615E")
    card(slide, 0.67, 5.16, 2.33, 1.04, "H2S batch", "SHA-256 / XMD\nmessage and domain scalars", "103441", TEAL,
         title_size=13.5, body_size=9.3, accent=TEAL)
    card(slide, 3.24, 5.06, 6.70, 1.24, "G1 MSM execution", "Rⱼ = Σᵢ sⱼᵢPᵢ · shared cache and multi-output kernels", "102A43", BLUE,
         title_size=14, body_size=9.1, accent=BLUE)
    for i, (label, color) in enumerate([
        ("R1 variable/fused", BLUE),
        ("R2 common tuned", TEAL),
        ("R3 large tiled", ORANGE),
        ("R4 universal", "94A9BF"),
    ]):
        small_pill(slide, 3.45 + i * 1.58, 5.78, 1.43, label, color, "12283D", 7.7)
    card(slide, 10.19, 5.16, 2.46, 1.04, "Pairing batch", "signature / proof equation\nGPU gate when profitable", "2D2832", ORANGE,
         title_size=13.5, body_size=9.1, accent=ORANGE)
    card(slide, 0.67, 6.42, 11.98, 0.25,
         "Cache / workspace / streams: generators + fixed point tables + prepared states + reusable device buffers", "",
         "0D2930", "3A8D82", title_size=8.7, accent="3A8D82")

    text_box(slide, 9.95, 4.24, 2.62, 0.20, "results return to ④ finalize", 8.3, MUTED, True, PP_ALIGN.RIGHT)
    footer(slide, 2)


def slide_relationship(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "两张架构图的边界关系", "Figure A 讨论系统吞吐；Figure B 讨论四原语如何共享 GPU 后端。二者不能用同一个性能数字替代")

    card(slide, 0.62, 1.48, 5.62, 4.95, "", "", "0B2236", BLUE, title_size=17, accent=BLUE)
    card(slide, 0.94, 2.08, 4.98, 0.69, "HTTP + Data Integrity", "JSON-LD/RDFC · framing · serialization", PANEL, "6485A5", title_size=12, body_size=9.3)
    arrow(slide, 3.35, 2.83, "down", TEAL, 0.7)
    card(slide, 0.94, 3.18, 4.98, 0.88, "Micro-batching / scheduling", "4 queues · coalescing · pipeline slots · backpressure", "103441", TEAL, title_size=12.5, body_size=9.5)
    arrow(slide, 3.35, 4.12, "down", TEAL, 0.7)
    card(slide, 0.94, 4.48, 4.98, 0.85, "Figure B · four primitives", "Sign · Verify · ProofGen · ProofVerify", PANEL_2, BLUE, title_size=12.5, body_size=10)
    arrow(slide, 3.35, 5.40, "down", TEAL, 0.7)
    card(slide, 0.94, 5.73, 4.98, 0.45, "GPU primitive layer", "H2S · MSM Route 1–4 · pairing", "103441", TEAL, title_size=10.5, body_size=8.5)

    card(slide, 6.60, 1.48, 6.12, 4.95, "", "", "0D2631", TEAL, title_size=17, accent=TEAL)
    items = [
        ("2.1 System boundary", "W3C/API 是兼容层；GPU 优化不改变 CFRG/W3C 语义。", BLUE),
        ("2.2 Batch scheduler", "把并发单请求聚合成 operation/shape-compatible batches。", TEAL),
        ("2.3 Hybrid runtime", "CPU 负责控制密集型 prepare/finalize，GPU 承担算术热点。", ORANGE),
        ("2.4 GPU execution plane", "H2S、G1 MSM、pairing；Route 1–4 是 MSM 子系统。", GREEN),
        ("2.5 Resource reuse", "点表、prepared state、buffers 和 streams 跨 batch 复用。", "9A8BFF"),
    ]
    for i, (head, body, color) in enumerate(items):
        card(slide, 6.92, 2.08 + i * 0.79, 5.47, 0.62, head, body, PANEL, color,
             title_size=11.5, body_size=8.7, accent=color)

    card(slide, 0.62, 6.68, 12.10, 0.29,
         "计量边界：HTTP req/s ≠ BBS op/s ≠ kernel item/s；实验章节必须分别报告 queue wait、effective batch、route、GPU occupancy 与端到端吞吐。", "",
         "2B2631", ORANGE, title_size=9.3, accent=ORANGE)
    footer(slide, 3)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "cuBBS-Cred Section 2 GPU Architecture"
    prs.core_properties.subject = "W3C end-to-end and four-primitive GPU architecture"
    prs.core_properties.author = "cuBBS-Cred"
    prs.core_properties.keywords = "cuBBS-Cred, W3C, BBS, CUDA, architecture, batch"
    slide_w3c_end_to_end(prs)
    slide_four_primitives(prs)
    slide_relationship(prs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    build()
