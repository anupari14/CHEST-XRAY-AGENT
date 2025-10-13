# utils/pdf_export.py
from typing import Optional, List
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm


def export_pdf_with_images(
    report_json: dict,
    patient: dict,
    input_img: str,
    output_img: Optional[str],
    out_path: str,
    overlay_imgs: Optional[List[str]] = None,
) -> str:
    """
    Nicely formatted A4 report:
      • Header banner
      • Patient card (3 columns, aligned)
      • Image section: input (large) + overlay cards (2 per row, up to 3)
      • Report text with consistent spacing and section dividers
    """
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(p_out), pagesize=A4)
    W, H = A4
    M = 1.6 * cm                    # margin
    y = H - M

    # ---------- helpers ----------
    def banner(title: str, subtitle: str = ""):
        nonlocal y
        bar_h = 0.9 * cm
        c.setFillColor(colors.HexColor("#0F172A"))  # slate-900
        c.rect(M, y - bar_h, W - 2 * M, bar_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(M + 0.5 * cm, y - 0.65 * cm, title)
        if subtitle:
            c.setFont("Helvetica", 9)
            c.drawRightString(W - M - 0.5 * cm, y - 0.62 * cm, subtitle)
        c.setFillColor(colors.black)
        y -= (bar_h + 0.6 * cm)

    def section(title_txt: str):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(M, y, title_txt)
        y -= 0.35 * cm
        c.setStrokeColor(colors.HexColor("#E5E7EB"))  # gray-200
        c.setLineWidth(0.8)
        c.line(M, y, W - M, y)
        c.setStrokeColor(colors.black)
        y -= 0.5 * cm

    def kv(x: float, y0: float, label: str, value: str) -> float:
        """Draw key/value vertically with consistent spacing; return new y."""
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#6B7280"))  # gray-500
        c.drawString(x, y0, label.upper())
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y0 - 0.4 * cm, str(value or "—"))
        return y0 - 1.05 * cm

    def caption(x: float, y0: float, text: str):
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#6B7280"))
        c.drawString(x, y0, text)
        c.setFillColor(colors.black)

    def wrap(text: str, max_chars=96) -> List[str]:
        if not text:
            return []
        out, line = [], ""
        for w in text.split():
            if len(line) + len(w) + 1 <= max_chars:
                line = (line + " " + w).strip()
            else:
                out.append(line); line = w
        if line:
            out.append(line)
        return out

    def card(x: float, y_top: float, w: float, h: float):
        """Light card container."""
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#E5E7EB"))
        c.setLineWidth(1.0)
        c.roundRect(x, y_top - h, w, h, 6, fill=1, stroke=1)
        c.setStrokeColor(colors.black)

    # ---------- header ----------
    banner(
        title="Chest X-ray Report",
        subtitle=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ---------- patient section ----------
    section("Patient")

    full_name = f"{patient.get('first', '')} {patient.get('last', '')}".strip() or "—"
    col_gap = 0.9 * cm
    col_w = (W - 2 * M - 2 * col_gap) / 3.0
    x1, x2, x3 = M, M + col_w + col_gap, M + 2 * (col_w + col_gap)
    y1 = y2 = y3 = y

    y1 = kv(x1, y1, "MRN", patient.get("mrn"))
    y1 = kv(x1, y1, "SEX", patient.get("sex"))
    y1 = kv(x1, y1, "DOB", patient.get("dob"))

    y2 = kv(x2, y2, "PATIENT ID", patient.get("patient_id"))
    y2 = kv(x2, y2, "NAME", full_name)

    y3 = kv(x3, y3, "PHONE", patient.get("phone"))
    y3 = kv(x3, y3, "EMAIL", patient.get("email"))

    y = min(y1, y2, y3) - 0.4 * cm

    # ---------- images section ----------
    section("Images")

    # Left: input card (larger)
    left_w, left_h = 8.6 * cm, 10.2 * cm
    right_x0 = M + left_w + 1.2 * cm
    right_w = W - right_x0 - M

    # Input card
    card(M, y, left_w, left_h)
    try:
        c.drawImage(
            ImageReader(str(input_img)),
            M + 0.25 * cm, y - left_h + 0.25 * cm,
            width=left_w - 0.5 * cm, height=left_h - 0.95 * cm,
            preserveAspectRatio=True, anchor="s",
        )
    except Exception:
        caption(M + 0.35 * cm, y - left_h + 0.5 * cm, "[Input image unavailable]")
    caption(M + 0.35 * cm, y - left_h - 0.35 * cm, "Input X-ray")

    # Right: overlay cards (2-per-row)
    imgs: List[str] = list(overlay_imgs or [])
    if not imgs and output_img:
        imgs = [output_img]
    imgs = imgs[:3]

    card_gap = 0.8 * cm
    card_w = (right_w - card_gap) / 2.0
    card_h = (left_h - card_gap) / 2.0  # stack up to 2 rows

    def draw_overlay(ix: int, px: float, py: float, path: str):
        card(px, py, card_w, card_h)
        try:
            c.drawImage(
                ImageReader(str(path)),
                px + 0.25 * cm, py - card_h + 0.25 * cm,
                width=card_w - 0.5 * cm, height=card_h - 0.95 * cm,
                preserveAspectRatio=True, anchor="s",
            )
        except Exception:
            caption(px + 0.35 * cm, py - card_h + 0.5 * cm, f"[Overlay {ix+1} unavailable]")
        caption(px + 0.35 * cm, py - card_h - 0.35 * cm, f"Overlay {ix+1}")

    # Row 1
    if len(imgs) >= 1:
        draw_overlay(0, right_x0, y, imgs[0])
    if len(imgs) >= 2:
        draw_overlay(1, right_x0 + card_w + card_gap, y, imgs[1])
    # Row 2 (only if 3rd present)
    next_row_top = y - (card_h + card_gap)
    if len(imgs) >= 3:
        draw_overlay(2, right_x0, next_row_top, imgs[2])

    # move y below image block
    used_h = max(left_h, card_h + (card_h + card_gap if len(imgs) > 2 else 0))
    y = y - used_h - 1.0 * cm

    # ---------- report section ----------
    section("Report")

    rep = report_json.get("report", {}) if "report" in report_json else report_json

    # metadata table-like alignment
    meta_x_lab = M
    meta_x_val = M + 3.0 * cm
    line_h = 0.5 * cm

    def meta_row(label: str, value: str):
        nonlocal y
        c.setFont("Helvetica-Bold", 10)
        c.drawString(meta_x_lab, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(meta_x_val, y, str(value or "—"))
        y -= line_h

    meta_row("Indication", rep.get("indication"))
    meta_row("Technique", rep.get("technique"))
    meta_row("Comparison", rep.get("comparison"))

    # Findings
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M, y, "Findings:")
    y -= 0.45 * cm
    c.setFont("Helvetica", 10)
    for ln in wrap(rep.get("findings") or "", max_chars=104):
        c.drawString(M, y, ln)
        y -= 0.38 * cm
        if y < M + 4 * cm:
            c.showPage(); y = H - M

    # Impression
    y -= 0.25 * cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M, y, "Impression:")
    y -= 0.45 * cm
    c.setFont("Helvetica", 10)
    for it in rep.get("impression", []) or []:
        for ln in wrap(f"• {it}", max_chars=100):
            c.drawString(M, y, ln)
            y -= 0.38 * cm
            if y < M + 2 * cm:
                c.showPage(); y = H - M

    # footer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawRightString(W - M, M * 0.7, "AI-generated draft; requires radiologist review. Not for diagnostic use.")
    c.setFillColor(colors.black)

    c.showPage()
    c.save()
    return str(p_out)
