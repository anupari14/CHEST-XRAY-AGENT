# app.py (snippet)
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json
from typing import Dict, Any
from openai import OpenAI
from db.vectors import ingest_pdf_report,vector_search
from fastapi import Query


from starlette.staticfiles import StaticFiles
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)



class ReportBody(BaseModel):
    indication: str
    technique: str
    comparison: str
    findings: str
    impression: List[str]

class ReportPackage(BaseModel):
    report: ReportBody
    codes: Dict[str, Any] = Field(default_factory=dict)
    flags: Dict[str, Any] = Field(default_factory=dict)
    disclaimers: List[str] = Field(default_factory=list)

# --- NORMALIZERS (add near your other helpers) ---

def coerce_report(report):
    """Case-insensitive normalization for report keys + types."""
    if not isinstance(report, dict):
        report = {}

    # make a lower -> value map (so "Indication"/"indication" both work)
    lower_map = { (k or "").strip().lower(): v for k, v in report.items() }

    out = {
        "indication":   str(lower_map.get("indication", "") or ""),
        "technique":    str(lower_map.get("technique", "") or ""),
        "comparison":   str(lower_map.get("comparison", "") or ""),
        "findings":     str(lower_map.get("findings", "") or ""),
        "impression":   []
    }

    imp = lower_map.get("impression", [])
    if isinstance(imp, list):
        out["impression"] = [str(x) for x in imp]
    elif isinstance(imp, str) and imp.strip():
        out["impression"] = [imp.strip()]

    return out


def coerce_codes(codes):
    """Normalize 'codes' to dict: {icd10:[{code,desc}], radlex:[str]}."""
    out = {"icd10": [], "radlex": []}
    if not codes:
        return out

    # If model sent a list (e.g., []), just return empty structured object
    if isinstance(codes, list):
        # try to recognize list of dicts as icd10 entries
        icd = []
        for x in codes:
            if isinstance(x, dict):
                code = x.get("code") or x.get("icd10") or x.get("id")
                desc = x.get("desc") or x.get("description") or ""
                if code:
                    icd.append({"code": str(code), "desc": str(desc)})
        if icd:
            out["icd10"] = icd
        return out

    # dict path
    if isinstance(codes, dict):
        icd10 = codes.get("icd10", [])
        radlex = codes.get("radlex", [])
        if isinstance(icd10, list):
            out["icd10"] = [
                {"code": str(i.get("code") or i.get("icd10") or i), "desc": str(i.get("desc") or i.get("description") or "")}
                for i in icd10 if (isinstance(i, dict) and (i.get("code") or i.get("icd10")))
                or isinstance(i, str)
            ]
        if isinstance(radlex, list):
            out["radlex"] = [str(r) for r in radlex if r]
    return out


def merge_codes(a, b):
    """Merge two codes dicts with de-dupe."""
    out = {"icd10": [], "radlex": []}
    seen_icd, seen_rad = set(), set()
    for item in (a.get("icd10", []) + b.get("icd10", [])):
        code = item.get("code")
        if code and code not in seen_icd:
            seen_icd.add(code)
            out["icd10"].append({"code": code, "desc": item.get("desc", "")})
    for rid in (a.get("radlex", []) + b.get("radlex", [])):
        if rid and rid not in seen_rad:
            seen_rad.add(rid)
            out["radlex"].append(rid)
    return out


def coerce_flags(flags):
    """Normalize flags into a dict with expected keys."""
    base = {"critical": False, "needs_overread": True}
    if isinstance(flags, dict):
        out = dict(base)
        out["critical"] = bool(flags.get("critical", base["critical"]))
        out["needs_overread"] = bool(flags.get("needs_overread", base["needs_overread"]))
        if isinstance(flags.get("follow_up"), str):
            out["follow_up"] = flags["follow_up"]
        return out

    # if list of strings like ["critical"] etc., map heuristically
    if isinstance(flags, list):
        s = {str(x).lower() for x in flags}
        return {
            "critical": "critical" in s or "urgent" in s,
            "needs_overread": True,
            "follow_up": None
        }
    return base


def coerce_disclaimers(disclaimers):
    """Ensure disclaimers is a list of whole strings, not split chars."""
    MUST = "AI-generated draft; requires radiologist review."
    if disclaimers is None:
        return [MUST]

    # If it's a single string, wrap it
    if isinstance(disclaimers, str):
        items = [disclaimers]
    elif isinstance(disclaimers, list):
        # If it looks like a list of single chars, join them
        if disclaimers and all(isinstance(x, str) and len(x) == 1 for x in disclaimers):
            items = ["".join(disclaimers)]
        else:
            items = [str(x) for x in disclaimers]
    else:
        items = []

    # Ensure MUST disclaimer present and dedupe
    s = {i.strip() for i in items if i and i.strip()}
    s.add(MUST)
    return list(s)




def llm_report(findings_payload: dict, prompt_path="prompts/report_prompt.md") -> dict:
    system = open(prompt_path, "r", encoding="utf-8").read()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("OPENAI_REPORT_MODEL", "gpt-4o-mini")

    # NOTE: The word "JSON" must appear in messages when using json_object
    system_with_json = (
        system
        + "\n\nReturn your answer as a single **JSON** object only. "
          "Do not include prose outside of JSON."
    )

    user_with_json = {
        "role": "user",
        "content": (
            "Here is the findings payload. "
            "Generate the report as **JSON** with fields: report, codes, flags, disclaimers.\n\n"
            + json.dumps(findings_payload)
        )
    }

    resp = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_with_json},
            user_with_json
        ],
        temperature=0.2,
        max_tokens=500,
    )

    data = json.loads(resp.choices[0].message.content)

    # ensure disclaimer
    disc = set(data.get("disclaimers", []))
    disc.add("AI-generated draft; requires radiologist review.")
    data["disclaimers"] = list(disc)
    return data

def _build_where(
    critical_only: Optional[bool] = None,
    sex: Optional[str] = None ) -> Optional[Dict[str, Any]]:
    conds: List[Dict[str, Any]] = []
    if critical_only:
        conds.append({"critical": {"$eq": True}})
    if sex:
        conds.append({"patient_sex": {"$eq": sex}})
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


# app.py
import os, json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from models.cxr import CXRInferencer
from agents.tools import simple_qc, map_codes, phi_scrub, export_pdf
from pydantic import ValidationError
from typing import Optional
from pathlib import Path


from app import ReportPackage  # if split; or paste classes here directly

app = FastAPI(title="Medical Agent: CXR → Report")
app.mount("/static", StaticFiles(directory=str(ARTIFACTS_DIR)), name="static")

# ---- YOLO singleton (loads once) ----
# If you trained in a specific path, set env: export YOLO_MODEL_PATH="/path/to/best.pt"
# or pass a string to LocalYOLO(...)

inferencer = CXRInferencer(weights="densenet121-res224-chex")
def _to_url(path: Path) -> str:
    """Convert a local artifacts path to a /static URL."""
    rel = path.relative_to(ARTIFACTS_DIR)
    return f"/static/{rel.as_posix()}"

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    threshold: float = Form(0.5),
    topk: int = Form(3)  # allow UI to control number of CAMs
):
    # ---- Create per-encounter folder under artifacts/ ----
    encounter_id = f"enc_{int(time.time())}"
    enc_dir = ARTIFACTS_DIR / encounter_id
    cam_dir = enc_dir / "cams"
    enc_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save upload inside encounter ----
    upload_path = enc_dir / f"upload_{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # ---- Predict (torchxrayvision CheXpert) ----
    cxr = inferencer.predict(str(upload_path), threshold=threshold)
    qc  = simple_qc(cxr["findings"])

    payload = {
        "patient_context": {"age": patient_age, "sex": patient_sex},
        "technical": {"qc": qc},
        "findings": cxr["findings"],
        "model": cxr["model"],
        "uncertainty": "Model-only; needs human review."
    }

    # ---- LLM Draft (JSON) ----
    draft = llm_report(payload)
    # Normalize + merge codes/flags
    draft["report"] = coerce_report(draft.get("report"))
    draft["codes"]  = coerce_codes(draft.get("codes"))
    mapped_codes    = coerce_codes(map_codes(cxr["findings"]))
    draft["codes"]  = merge_codes(draft["codes"], mapped_codes)
    draft["flags"]  = coerce_flags(draft.get("flags"))
    draft["disclaimers"] = coerce_disclaimers(draft.get("disclaimers"))

    # PHI scrub (now safe)
    for k in ["indication", "technique", "comparison", "findings"]:
        draft["report"][k] = phi_scrub(draft["report"][k])

    # Ensure disclaimer
    disc = set(draft.get("disclaimers", []))
    disc.add("AI-generated draft; requires radiologist review.")
    draft["disclaimers"] = list(disc)

    # ---- Validate JSON shape ----
    try:
        pkg = ReportPackage(**draft)
    except ValidationError as e:
        return {"error": "Schema validation failed", "details": json.loads(e.json())}

    # ---- Export PDF into encounter folder ----
    pdf_path = enc_dir / "report.pdf"
    out_pdf = export_pdf(pkg.model_dump(), out_path=str(pdf_path))

    # ---- Generate Grad-CAM overlays into encounter/cams ----
    cam_out = inferencer.topk_cam(str(upload_path), topk=int(topk), save_dir=str(cam_dir))
    cam_assets: List[Dict[str, Any]] = []
    for r in cam_out["results"]:
        p = Path(r["cam_png"])
        cam_assets.append({
            "rank": r["rank"],
            "label": r["label"],
            "score": r["score"],
            "url": _to_url(p),      # public URL for UI
            "path": str(p)          # local path (optional)
        })

    # ---- Ingest PDF into vector DB (use local path, not URL) ----
    patient_meta = {"age": patient_age, "sex": patient_sex}
    ing = ingest_pdf_report(
        pdf_path=str(pdf_path),
        report_json=pkg.model_dump(),
        patient_id=str(patient_meta.get("mrn", "anon")),  # replace with true MRN/ID if available (ensure compliance)
        encounter_id=encounter_id,
        patient_meta=patient_meta,
    )

    # ---- Build response ----
    return {
        "findings": cxr["findings"],
        "qc": qc,
        "report": pkg.model_dump(),
        "artifacts": {
            "pdf": _to_url(pdf_path),
            "cams": cam_assets,
            "encounter_id": encounter_id
        },
        "ingestion": ing
    }

# run: uvicorn app:app --reload

@app.get("/search")
async def search_reports(
    q: str,
    k: int = 5,
    critical_only: bool = False,
    sex: Optional[str] = None,   # <-- instead of: str | None
    ):
    where = _build_where(critical_only=critical_only, sex=sex)
    hits = vector_search(q, k=k, where=where)
    return {"query": q, "k": k, "where": where, "results": hits}
