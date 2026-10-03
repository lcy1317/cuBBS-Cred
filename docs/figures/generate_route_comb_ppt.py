#!/usr/bin/env python3
"""Generate the editable cuBBS-Cred Route/comb architecture deck."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "cubbs-cred-route-comb-architecture.zh-CN.pptx"

BG = "071525"
PANEL = "0E2236"
PANEL_2 = "12304A"
PANEL_3 = "14375A"
TEAL = "21D4B4"
TEAL_DARK = "0D4B48"
BLUE = "4EA4FF"
ORANGE = "FFB454"
RED = "F9797D"
WHITE = "F7FAFD"
MUTED = "AFC2D5"
GRID = "31506D"
GREEN = "79E2AA"
FONT = "Noto Sans CJK SC"
MONO = "DejaVu Sans Mono"


def rgb(value):
    return RGBColor.from_string(value)


def set_bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(color)


def text_box(slide, x, y, w, h, text, size=15, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, font=FONT,
             margin=0.08):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin / 2)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = rgb(color)
    return box


def card(slide, x, y, w, h, title, body="", fill=PANEL, stroke=GRID,
         title_color=WHITE, body_color=MUTED, title_size=15, body_size=11,
         radius=True, accent=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke)
    shape.line.width = Pt(1.25)
    if accent:
        stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.07), Inches(h))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = rgb(accent)
        stripe.line.fill.background()
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.09)
    tf.margin_bottom = Inches(0.06)
    p = tf.paragraphs[0]
    p.space_after = Pt(3)
    run = p.add_run()
    run.text = title
    run.font.name = FONT
    run.font.size = Pt(title_size)
    run.font.bold = True
    run.font.color.rgb = rgb(title_color)
    if body:
        p2 = tf.add_paragraph()
        p2.space_before = Pt(1)
        p2.line_spacing = 1.0
        run2 = p2.add_run()
        run2.text = body
        run2.font.name = FONT
        run2.font.size = Pt(body_size)
        run2.font.color.rgb = rgb(body_color)
    return shape


def line(slide, x1, y1, x2, y2, color=GRID, width=1.8):
    connector = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(width)
    return connector


def arrow(slide, x, y, direction="down", color=TEAL, scale=1.0):
    if direction == "right":
        shp, w, h = MSO_SHAPE.RIGHT_ARROW, 0.34 * scale, 0.18 * scale
    else:
        shp, w, h = MSO_SHAPE.DOWN_ARROW, 0.18 * scale, 0.34 * scale
    a = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(color)
    a.line.fill.background()
    return a


def title(slide, heading, subtitle):
    text_box(slide, 0.5, 0.24, 12.3, 0.5, heading, 25, WHITE, True)
    text_box(slide, 0.52, 0.76, 12.1, 0.34, subtitle, 11.5, MUTED)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.10), Inches(1.25), Inches(0.045))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(TEAL)
    bar.line.fill.background()


def footer(slide, page):
    text_box(slide, 0.52, 7.18, 9.0, 0.18, "cuBBS-Cred · Route ABI v2 · editable architecture deck", 8.5, "7890A8")
    text_box(slide, 12.25, 7.16, 0.5, 0.2, str(page), 9, "7890A8", align=PP_ALIGN.RIGHT)


def slide_route_map(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "comb 在 cuBBS-Cred 架构中的位置", "comb8 / comb9 / comb10 不是 Route 编号；它们是 Route 2 内部的固定基 signed-digit MSM 变体")

    ops = [
        ("Sign", "签名中的公共生成元 MSM"),
        ("Verify", "签名关系的 G1 MSM"),
        ("ProofGen", "证明承诺与响应 MSM"),
        ("ProofVerify", "公开/隐藏项重构 MSM"),
    ]
    for i, (name, desc) in enumerate(ops):
        x = 0.52 + i * 3.18
        card(slide, x, 1.36, 2.81, 0.73, name, desc, PANEL_2, "2A6C75", title_size=14, body_size=9.5, accent=TEAL)
        line(slide, x + 1.40, 2.09, 6.66, 2.42, "2D7991", 1.2)

    card(slide, 3.82, 2.31, 5.68, 0.74, "统一成同一种 GPU 工作", "S ∈ Frᴺˣᴷ，P ∈ G1ᴷ；逐行计算 Rⱼ = Σᵢ SⱼᵢPᵢ", "0B3543", TEAL, title_size=14, body_size=11)
    arrow(slide, 6.58, 3.08, "down", TEAL, 0.9)
    card(slide, 4.25, 3.43, 4.82, 0.66, "MSM dispatcher", "依据：点是否复用、K(shape)、N(batch)、显存预算", "162A40", BLUE, title_size=14, body_size=10.5)

    routes = [
        ("Route 1", "Variable / fused\nGLV + signed-radix\n点随任务变化", PANEL, BLUE),
        ("Route 2", "Common points / tuned\ncomb10 · comb9 · comb8\n复用预计算点表", TEAL_DARK, TEAL),
        ("Route 3", "Large common points\ntiled GLV / window-4\n大 K 避免点表爆炸", PANEL, ORANGE),
        ("Route 4", "Universal fallback\nbounded generic GLV\n未调优 shape", PANEL, "92A7BE"),
    ]
    for i, (name, body, fill, stroke) in enumerate(routes):
        x = 0.52 + i * 3.18
        line(slide, 6.66, 4.09, x + 1.40, 4.47, stroke, 1.3)
        card(slide, x, 4.45, 2.81, 1.20, name, body, fill, stroke, title_size=15, body_size=10.5, accent=stroke)

    card(slide, 3.60, 5.93, 6.13, 0.55, "Route 2 放大", "同一组 P 被大量标量行复用 → 把昂贵的点倍增/组合搬到一次性缓存，在线阶段只做查表、符号处理与归约", "123A47", TEAL, title_size=12.5, body_size=9.5)
    line(slide, 5.10, 5.65, 5.10, 5.93, TEAL, 2.0)
    arrow(slide, 5.01, 5.75, "down", TEAL, 0.8)
    card(slide, 0.52, 6.68, 12.20, 0.36, "不属于 comb：hash-to-scalar · transcript/challenge · pairing · 编解码 · 最终协议判定", "", "111F31", "3A5168", title_size=10.5, body_size=9)
    footer(slide, 1)


def slide_comb_internals(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "Route 2 内部：signed-comb8 / 9 / 10", "固定公共点 P 预计算一次；每个请求只提交标量矩阵 S，并让每个输出行独立占用一个 CUDA block")

    card(slide, 0.48, 1.35, 2.30, 1.38, "输入矩阵", "P = (P₀…Pₖ₋₁) ∈ G1ᴷ\nS = [sⱼᵢ] ∈ Frᴺˣᴷ\nK = 消息数 + 2", PANEL_2, BLUE, title_size=15, body_size=11.5, accent=BLUE)
    arrow(slide, 2.88, 1.91, "right", TEAL, 0.9)
    card(slide, 3.31, 1.35, 2.05, 1.38, "shape selector", "18 ≤ K ≤ 66 → 10\n67 ≤ K ≤ 160 → 9\n其余 K ≤ 258 → 8", "122D40", TEAL, title_size=15, body_size=11.3, accent=TEAL)

    combs = [
        ("comb10", "26 windows × 513 magnitudes / point", "18–66 terms", TEAL),
        ("comb9", "29 windows × 257 magnitudes / point", "67–160 terms", BLUE),
        ("comb8", "32 windows × 129 magnitudes / point", "other ≤258", ORANGE),
    ]
    for i, (name, detail, scope, stroke) in enumerate(combs):
        y = 1.21 + i * 1.06
        line(slide, 5.36, 2.04, 5.70, y + 0.43, stroke, 1.3)
        card(slide, 5.72, y, 2.55, 0.88, name, detail + "\n" + scope, PANEL_3, stroke, title_size=14, body_size=9.3, accent=stroke)

    card(slide, 8.63, 1.27, 4.15, 2.18, "一次性：point-table cache", "T[Pᵢ,t,|d|] = |d| · 2^(wt) · Pᵢ\ncache key: points / K / comb bits / affine mode\n默认预算 256 MiB；命中后跨 batch / request 复用", "103441", TEAL, title_size=15, body_size=10.5, accent=TEAL)
    for i, stroke in enumerate([TEAL, BLUE, ORANGE]):
        line(slide, 8.27, 1.64 + i * 1.06, 8.63, 2.28, stroke, 1.2)

    arrow(slide, 10.60, 3.52, "down", TEAL, 0.9)
    card(slide, 8.63, 3.93, 4.15, 1.85, "在线：scalar rows", "1 block / output row Rⱼ\nscalar → signed digits dⱼᵢₜ → table lookup\n负 digit 条件取负；线程部分和 → block reduction", PANEL_2, BLUE, title_size=15, body_size=10.5, accent=BLUE)
    arrow(slide, 10.60, 5.84, "down", GREEN, 0.8)
    card(slide, 8.63, 6.18, 4.15, 0.61, "输出", "R₀ … Rₙ₋₁ ∈ G1", "123A36", GREEN, title_size=13, body_size=10.5, accent=GREEN)

    card(slide, 0.48, 4.62, 7.80, 1.24, "核心等式", "s = Σₜ dₜ · 2^(wt)    ⇒    sP = Σₜ sign(dₜ) · T[P,t,|dₜ|]\n更大的 w：窗口更少、在线加法更少；但预计算表按 2^(w−1)+1 指数增长。", "121F33", "4A6E91", title_size=14, body_size=13, accent="4A6E91")
    card(slide, 0.48, 6.15, 7.80, 0.64, "关键边界", "comb 优化的是 Route 2 的 G1 MSM 热点，不替代 pairing，也不改变 BBS / W3C 的协议语义。", "2B2631", ORANGE, title_size=12, body_size=10, accent=ORANGE)
    footer(slide, 2)


def slide_tradeoffs(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    title(slide, "为什么不是永远用 comb10", "调度器优化的是总成本：预计算/显存成本 + N 个在线标量行成本，而不是单独追求最大窗口")

    headers = ["variant", "windows", "magnitudes", "entries / point", "默认 K 区间", "设计意图"]
    widths = [1.35, 1.20, 1.46, 1.55, 1.64, 4.02]
    x0, y0 = 0.55, 1.42
    x = x0
    for label, w in zip(headers, widths):
        card(slide, x, y0, w, 0.52, label, "", "17324A", "355D7B", title_size=10.2, radius=False)
        x += w
    rows = [
        ("comb10", "26", "513", "13,338", "18–66", "短 K：用更大点表换最少窗口/在线工作", TEAL),
        ("comb9", "29", "257", "7,453", "67–160", "中等 K：平衡查表量、显存和在线加法", BLUE),
        ("comb8", "32", "129", "4,128", "other ≤258", "较长 K：控制点表体积与缓存压力", ORANGE),
    ]
    for ri, row in enumerate(rows):
        y = y0 + 0.52 + ri * 0.62
        x = x0
        for ci, (value, w) in enumerate(zip(row[:6], widths)):
            fill = "10273C" if ri % 2 == 0 else "0D2235"
            box = card(slide, x, y, w, 0.62, value, "", fill, "294860", title_size=10.2, radius=False)
            if ci == 0:
                box.text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(row[6])
            x += w

    card(slide, 0.55, 4.08, 3.78, 1.10, "常用 message shape", "L=16/32/64 → K=18/34/66 → comb10", "103441", TEAL, title_size=13, body_size=11.5, accent=TEAL)
    card(slide, 4.51, 4.08, 3.78, 1.10, "中等 message shape", "L=128 → K=130 → comb9", "122D40", BLUE, title_size=13, body_size=11.5, accent=BLUE)
    card(slide, 8.47, 4.08, 3.78, 1.10, "长 message shape", "L=256 → K=258 → comb8（预算允许）", "352C22", ORANGE, title_size=13, body_size=11.5, accent=ORANGE)

    card(slide, 0.55, 5.48, 3.78, 1.07, "X：同构大批", "同一 P、同一 K；最大化 cache 命中和 N 行并行，Route 2 最理想。", PANEL, TEAL, title_size=13, body_size=10.3, accent=TEAL)
    card(slide, 4.51, 5.48, 3.78, 1.07, "H：混合 shape", "按 K 分桶；每个 shape 各用 comb10/9/8，并复用各自点表。", PANEL, BLUE, title_size=13, body_size=10.3, accent=BLUE)
    card(slide, 8.47, 5.48, 3.78, 1.07, "U：不同 issuer", "生成元可能公共，但 issuer 相关准备会拆组；复用程度低于 X/H。", PANEL, ORANGE, title_size=13, body_size=10.3, accent=ORANGE)

    card(slide, 0.55, 6.78, 11.70, 0.34, "K 超过专用边界或固定点表不再划算：Route 3（large common points）；未命中调优条件：Route 4。", "", "111F31", "3A5168", title_size=10.5)
    footer(slide, 3)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "cuBBS-Cred Route 1–4 与 signed-comb 架构"
    prs.core_properties.subject = "Editable architecture deck for Route ABI v2"
    prs.core_properties.author = "cuBBS-Cred"
    prs.core_properties.keywords = "cuBBS-Cred, BBS, CUDA, MSM, signed-comb, route"
    slide_route_map(prs)
    slide_comb_internals(prs)
    slide_tradeoffs(prs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    build()
