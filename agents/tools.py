# agents/tools.py
import re, json
from typing import Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ICD10_MAP = {
    "Cardiomegaly": "I51.7",
    "Atelectasis": "J98.11",
    "Consolidation": "J18.9",
    "Pleural Effusion": "J90",
}

def simple_qc(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    # naive rule: if no label >= 0.15, mark low-quality/uncertain
    high = any(f["prob"] >= 0.15 for f in findings)
    return {"low_signal": not high}

def map_codes(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    icd = []
    for f in findings:
        code = ICD10_MAP.get(f["label"])
        if code and f["present"]:
            icd.append({"code": code, "desc": f["label"]})
    return {"icd10": icd, "radlex": []}

def phi_scrub(text: str) -> str:
    text = re.sub(r"\b(\d{2}/\d{2}/\d{4}|\d{1,2}-\d{1,2}-\d{4})\b", "[DATE]", text)
    text = re.sub(r"\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b", "[NAME]", text)
    text = re.sub(r"\bMRN[:\s]*\d+\b", "MRN:[REDACTED]", text)
    return text

def export_pdf(report_json: Dict[str, Any], out_path: str) -> str:
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 50
    def line(txt):
        nonlocal y
        c.drawString(50, y, txt[:1000]); y -= 18
    line("Chest X-ray Report (AI Draft)")
    line("------------------------------------------------")
    r = report_json.get("report", {})
    for k in ["indication","technique","comparison","findings"]:
        v = r.get(k, "")
        for ln in (v if isinstance(v, list) else [v]):
            line(f"{k.title()}: {ln}")
    line("Impression:")
    for i in r.get("impression", []):
        line(f" - {i}")
    c.showPage(); c.save()
    return out_path
