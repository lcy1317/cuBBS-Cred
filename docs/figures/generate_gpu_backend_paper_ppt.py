#!/usr/bin/env python3
"""Generate an editable, paper-style PowerPoint diagram for the cuBBS-Cred GPU backend."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "cubbs-cred-gpu-backend-paper-editable.pptx"

FONT = "Times New Roman"
TEXT = "17212B"
LINE = "4B5563"
NAVY = "1F4E79"
HOST_FILL = "EAF0F6"
HOST_STROKE = "456A8A"
GPU_FILL = "FBFCFD"
GPU_STROKE = "52677A"
RUNTIME_FILL = "EEF2F7"
RUNTIME_STROKE = "68788A"
H2S_FILL = "FFF3E0"
H2S_STROKE = "A65E00"
PAIR_FILL = "F2EAF4"
PAIR_STROKE = "77507D"
MSM_FILL = "E7F2ED"
MSM_STROKE = "1F6F54"
ROUTE_FILL = "F5FAF7"
RESULT_FILL = "F3F4F6"
RESULT_STROKE = "5B6573"
EDGE_LABEL = "9A287D"
WHITE = "FFFFFF"


def rgb(value):
    return RGBColor.from_string(value)


def add_text(slide, x, y, w, h, text, size=12, color=TEXT, bold=False,
             align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin=0.03):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = p.space_after = Pt(0)
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = rgb(color)
    return box


def add_box(slide, x, y, w, h, text, fill, stroke, bold=False,
            radius=True, size=12, color=TEXT, line_width=1.2):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke)
    shape.line.width = Pt(line_width)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.07)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.space_before = p.space_after = Pt(0)
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = rgb(color)
    return shape


def add_boundary(slide, x, y, w, h, title, fill, stroke, title_color=TEXT):
    boundary = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    boundary.fill.solid()
    boundary.fill.fore_color.rgb = rgb(fill)
    boundary.fill.transparency = 18
    boundary.line.color.rgb = rgb(stroke)
    boundary.line.width = Pt(1.5)
    add_text(slide, x + 0.12, y + 0.05, w - 0.24, 0.30, title, 12,
             title_color, True, PP_ALIGN.CENTER)
    return boundary


def add_line(slide, x1, y1, x2, y2, color=LINE, width=1.1, dash=False):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    return c


def add_down_arrow(slide, x, y, color=LINE, scale=1.0):
    a = slide.shapes.add_shape(
        MSO_SHAPE.DOWN_ARROW, Inches(x), Inches(y), Inches(0.15 * scale), Inches(0.23 * scale)
    )
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(color)
    a.line.fill.background()
    return a


def add_right_arrow(slide, x, y, color=LINE, scale=1.0):
    a = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(0.25 * scale), Inches(0.14 * scale)
    )
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(color)
    a.line.fill.background()
    return a


def build_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)

    # CPU Host is shown only as the system boundary; the figure focuses on GPU internals.
    add_box(slide, 4.95, 0.12, 3.43, 0.54,
            "CPU Host · dynamic batching and job lowering",
            HOST_FILL, HOST_STROKE, True, size=12)
    add_text(slide, 5.08, 0.69, 3.18, 0.23, "packed primitive batches", 12,
             EDGE_LABEL, False)
    add_line(slide, 6.665, 0.66, 6.665, 0.91, HOST_STROKE, 1.2)
    add_down_arrow(slide, 6.59, 0.80, HOST_STROKE, 0.95)

    add_boundary(slide, 0.28, 0.92, 12.77, 5.72,
                 "cuBBS-Cred GPU Backend", GPU_FILL, GPU_STROKE, NAVY)

    dispatcher = add_box(slide, 0.65, 1.43, 2.45, 0.64,
                         "Job dispatcher\nshape-aware scheduling",
                         NAVY, NAVY, True, size=12, color=WHITE)
    runtime = add_box(slide, 3.77, 1.43, 5.82, 0.64,
                      "Shared runtime resources · CUDA streams · prepared states · telemetry",
                      RUNTIME_FILL, RUNTIME_STROKE, True, size=12)

    # Runtime/dispatch fan-out. Lines remain separately editable.
    add_line(slide, 1.87, 2.07, 1.87, 2.76, NAVY, 1.1)
    add_line(slide, 1.87, 2.76, 10.06, 2.76, NAVY, 1.1)
    add_line(slide, 4.44, 2.76, 4.44, 3.02, NAVY, 1.1)
    add_line(slide, 10.06, 2.76, 10.06, 3.02, NAVY, 1.1)
    add_down_arrow(slide, 1.795, 2.78, NAVY, 0.9)
    add_down_arrow(slide, 4.365, 2.78, NAVY, 0.9)
    add_down_arrow(slide, 9.985, 2.78, NAVY, 0.9)

    add_line(slide, 6.68, 2.07, 6.68, 2.42, RUNTIME_STROKE, 1.0, True)
    add_line(slide, 1.88, 2.42, 10.10, 2.42, RUNTIME_STROKE, 1.0, True)
    add_line(slide, 1.88, 2.42, 1.88, 3.02, RUNTIME_STROKE, 1.0, True)
    add_line(slide, 4.45, 2.42, 4.45, 3.02, RUNTIME_STROKE, 1.0, True)
    add_line(slide, 10.10, 2.42, 10.10, 3.02, RUNTIME_STROKE, 1.0, True)

    # Engines. H2S and Pairing are direct engines; only MSM has Route 1-4.
    add_box(slide, 0.65, 3.04, 2.45, 0.70,
            "Hash-to-Scalar Engine\nhash expansion + mod r",
            H2S_FILL, H2S_STROKE, True, size=12)
    add_box(slide, 3.18, 3.04, 2.58, 0.70,
            "Pairing Engine\nMiller loop + final exponentiation",
            PAIR_FILL, PAIR_STROKE, True, size=12)

    add_boundary(slide, 6.01, 2.91, 6.61, 2.45,
                 "MSM Engine over 𝔾₁", MSM_FILL, MSM_STROKE, MSM_STROKE)
    add_box(slide, 7.56, 3.37, 3.53, 0.52,
            "MSM classifier · point reuse + K + N",
            MSM_STROKE, MSM_STROKE, True, size=12, color=WHITE)

    route_x = [6.31, 7.86, 9.41, 10.96]
    route_labels = [
        "Route 1\nvariable points",
        "Route 2\ncommon points",
        "Route 3\nlarge common points",
        "Route 4\nuniversal fallback",
    ]
    add_line(slide, 9.325, 3.89, 9.325, 4.07, MSM_STROKE, 1.05)
    add_line(slide, 6.96, 4.07, 11.61, 4.07, MSM_STROKE, 1.05)
    for x, label in zip(route_x, route_labels):
        add_line(slide, x + 0.65, 4.07, x + 0.65, 4.20, MSM_STROKE, 1.05)
        add_down_arrow(slide, x + 0.575, 4.11, MSM_STROKE, 0.75)
        add_box(slide, x, 4.25, 1.30, 0.58, label,
                ROUTE_FILL, MSM_STROKE, True, size=12)

    add_line(slide, 6.96, 4.83, 6.96, 5.00, MSM_STROKE, 1.0)
    add_line(slide, 8.51, 4.83, 8.51, 5.00, MSM_STROKE, 1.0)
    add_line(slide, 9.96, 4.83, 9.96, 5.00, MSM_STROKE, 1.0)
    add_line(slide, 11.61, 4.83, 11.61, 5.00, MSM_STROKE, 1.0)
    add_line(slide, 6.96, 5.00, 11.61, 5.00, MSM_STROKE, 1.0)
    add_box(slide, 8.10, 4.91, 2.43, 0.36, "shared reduction and output packing",
            MSM_FILL, MSM_STROKE, False, size=12)

    # Engine outputs and protocol result boundary.
    add_line(slide, 1.88, 3.74, 1.88, 4.47, H2S_STROKE, 1.15)
    add_down_arrow(slide, 1.805, 4.28, H2S_STROKE, 0.9)
    add_box(slide, 1.23, 4.51, 1.30, 0.42, "𝔽ᵣ scalars",
            H2S_FILL, H2S_STROKE, True, size=12)

    add_line(slide, 4.47, 3.74, 4.47, 4.47, PAIR_STROKE, 1.15)
    add_down_arrow(slide, 4.395, 4.28, PAIR_STROKE, 0.9)
    add_box(slide, 3.54, 4.51, 1.86, 0.42, "e: 𝔾₁ × 𝔾₂ → 𝔾ₜ",
            PAIR_FILL, PAIR_STROKE, True, size=12)

    add_box(slide, 4.59, 5.74, 4.15, 0.56,
            "GPU result interface · scalars · 𝔾₁ outputs · pairing checks",
            RESULT_FILL, RESULT_STROKE, True, size=12)
    add_line(slide, 1.88, 4.93, 1.88, 5.52, H2S_STROKE, 1.0)
    add_line(slide, 1.88, 5.52, 5.23, 5.52, H2S_STROKE, 1.0)
    add_line(slide, 5.23, 5.52, 5.23, 5.74, H2S_STROKE, 1.0)
    add_line(slide, 4.47, 4.93, 4.47, 5.39, PAIR_STROKE, 1.0)
    add_line(slide, 4.47, 5.39, 6.66, 5.39, PAIR_STROKE, 1.0)
    add_line(slide, 6.66, 5.39, 6.66, 5.74, PAIR_STROKE, 1.0)
    add_line(slide, 9.32, 5.27, 9.32, 5.52, MSM_STROKE, 1.0)
    add_line(slide, 9.32, 5.52, 8.11, 5.52, MSM_STROKE, 1.0)
    add_line(slide, 8.11, 5.52, 8.11, 5.74, MSM_STROKE, 1.0)
    add_down_arrow(slide, 5.155, 5.56, H2S_STROKE, 0.75)
    add_down_arrow(slide, 6.585, 5.56, PAIR_STROKE, 0.75)
    add_down_arrow(slide, 8.035, 5.56, MSM_STROKE, 0.75)

    add_text(slide, 4.90, 6.32, 3.53, 0.21, "typed device results", 12,
             EDGE_LABEL, False)
    add_line(slide, 6.665, 6.30, 6.665, 6.68, HOST_STROKE, 1.2)
    add_down_arrow(slide, 6.59, 6.51, HOST_STROKE, 0.9)
    add_box(slide, 4.95, 6.70, 3.43, 0.54,
            "CPU Host · protocol assembly and final decision",
            HOST_FILL, HOST_STROKE, True, size=12)

    # Small editable legend clarifies the architectural invariant.
    add_text(slide, 0.56, 6.70, 3.95, 0.52,
             "Only MSM jobs enter Route 1–4; H2S and pairing use dedicated engines.",
             12, TEXT, False, PP_ALIGN.LEFT)
    return slide


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    build_slide(prs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
