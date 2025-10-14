# app.py (snippet)
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json , uuid, datetime
from typing import Dict, Any, Optional
from openai import OpenAI
from db.vectors import ingest_pdf_report,vector_search
from utils.pdf_export import export_pdf_with_images
from fastapi import Query, Depends, HTTPException, Header, status, Form
from fastapi.responses import JSONResponse
from models.yolo import YOLOInferencer


from starlette.staticfiles import StaticFiles
from pathlib import Path

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts")).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Patients: models & storage (JSON-backed) ----
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
import uuid, json

DB_DIR = ARTIFACTS_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
PAT_DB_PATH = DB_DIR / "patients.json"
yolo_inferencer = YOLOInferencer(weights=Path("models/best.pt"))
model_name = "Ultralytics YOLOv8 (class-wise max confidence)"

class PatientBase(BaseModel):
    mrn: Optional[str] = Field(None, description="Medical Record Number / external ID")
    first: str
    last: str
    sex: Optional[str] = Field(None, pattern="^(M|F|Other|)$")
    dob: Optional[str] = None                 # ISO date string (YYYY-MM-DD) for simplicity
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    notes: Optional[str] = None

class PatientCreate(PatientBase):
    first: str
    last: str

class PatientUpdate(BaseModel):
    mrn: Optional[str] = None
    first: Optional[str] = None
    last: Optional[str] = None
    sex: Optional[str] = Field(None, pattern="^(M|F|Other|)$")
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    notes: Optional[str] = None

class Patient(PatientBase):
    patient_id: str
    created_at: float
    updated_at: float


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

def _load_patients() -> Dict[str, Dict[str, Any]]:
    if PAT_DB_PATH.exists():
        try:
            return json.loads(PAT_DB_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_patients(db: Dict[str, Dict[str, Any]]) -> None:
    PAT_DB_PATH.write_text(json.dumps(db, indent=2))

def _search_index(rec: Dict[str, Any]) -> str:
    """Simple concat index for naive search."""
    parts = [
        rec.get("patient_id",""), rec.get("mrn",""), rec.get("first",""), rec.get("last",""),
        rec.get("sex",""), rec.get("dob",""), rec.get("phone",""), rec.get("email",""),
        rec.get("address",""), rec.get("notes",""),
    ]
    return " ".join([str(p or "") for p in parts]).lower()


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

# ---- Simple Users & Tokens (demo) ----
# Comma-separated "user:password" pairs (DEMO ONLY; use a real DB in prod)
RAW_USERS = os.getenv("APP_USERS", "demo:demo").split(",")
USERS: Dict[str, str] = {}
for pair in RAW_USERS:
    pair = pair.strip()
    if not pair or ":" not in pair:
        continue
    u, p = pair.split(":", 1)
    USERS[u.strip()] = p.strip()

# token store: token -> {"user": str, "exp": unix_epoch}
TOKENS: Dict[str, Dict[str, float]] = {}

TOKEN_TTL_SECONDS = int(os.getenv("APP_TOKEN_TTL", "7200"))  # 2 hours


def _now() -> float:
    return time.time()

def _new_token(user: str) -> str:
    t = uuid.uuid4().hex
    TOKENS[t] = {"user": user, "exp": _now() + TOKEN_TTL_SECONDS}
    return t

def _check_token(token: str) -> str:
    rec = TOKENS.get(token)
    if not rec:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if rec["exp"] < _now():
        TOKENS.pop(token, None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return rec["user"]

def auth_required(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.split(None, 1)[1].strip()
    return _check_token(token)




# ---- YOLO singleton (loads once) ----
# If you trained in a specific path, set env: export YOLO_MODEL_PATH="/path/to/best.pt"
# or pass a string to LocalYOLO(...)

# inferencer = CXRInferencer(weights="densenet121-res224-chex")
def _to_url(path: Path) -> str:
    """Convert a local artifacts path to a /static URL."""
    rel = path.relative_to(ARTIFACTS_DIR)
    return f"/static/{rel.as_posix()}"

@app.post("/auth/login")
async def auth_login(username: str = Form(...), password: str = Form(...)):
    real = USERS.get(username)
    if not real or real != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")
    tok = _new_token(username)
    return {
        "token": tok,
        "user": username,
        "expires_in": TOKEN_TTL_SECONDS,
        "expires_at": int(_now() + TOKEN_TTL_SECONDS),
    }

@app.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.lower().startswith("bearer "):
        tok = authorization.split(None, 1)[1].strip()
        TOKENS.pop(tok, None)
    return {"ok": True}


from fastapi import Depends, HTTPException, Header, status, Form, UploadFile, File

# ...

from pathlib import Path
from fastapi import UploadFile, File, Form, Depends, HTTPException
from pydantic import ValidationError

@app.post("/analyze", dependencies=[Depends(auth_required)])
async def analyze(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    topk: int = Form(3),              # how many CAMs to render
    patient_id: str = Form(...),
):
    # ---- validate patient ----
    db = _load_patients()
    patient = db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # ---- encounter dirs ----
    encounter_id = f"enc_{int(time.time())}"
    enc_dir  = ARTIFACTS_DIR / encounter_id
    cams_dir = enc_dir / "cams"
    enc_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    # ---- save upload ----
    upload_path = enc_dir / f"upload_{file.filename}"
    raw = await file.read()
    with open(upload_path, "wb") as f:
        f.write(raw)

    # ---- model prediction (no CAMs here) ----
    # try:
    #     cxr = inferencer.predict(str(upload_path), threshold=threshold)
    # except TypeError:
    #     cxr = inferencer.predict(str(upload_path))
    # findings = cxr.get("findings", [])
    # print(findings)
    # qc = simple_qc(findings)

    findings, cams_raw = yolo_inferencer.infer_probs_and_overlays(
        image_path=upload_path,
        out_dir=cams_dir,
        conf=float(threshold),
        iou=0.45,
        topk=int(topk),
    )
    qc = simple_qc(findings)

    # ---- Grad-CAMs (THIS uses your topk_cam) ----
    # try:
    #     cam_pack = inferencer.topk_cam(str(upload_path), topk=int(topk), save_dir=str(cams_dir))
    #     cams_raw = cam_pack.get("results", [])  # [{rank,label,score,cam_png}]
    #     print(cams_raw)
    # except Exception:
    #     cams_raw = []

    # normalize for response + choose overlay for PDF
    def _static_url(p: Path) -> str:
        """Return /static/<relative> for files under ARTIFACTS_DIR.
        If not under ARTIFACTS_DIR, try to make it so (copy) or fall back to file name.
        """
        p = Path(p).resolve()
        root = ARTIFACTS_DIR  # already resolved above
        try:
            rel = p.relative_to(root)
            return f"/static/{rel.as_posix()}"
        except Exception:
            # If file accidentally outside, try to mirror it into artifacts so it can be served
            # (shouldn't happen if you save CAMs into enc_dir/cams)
            # As a last resort, just return a name under current encounter won't be reliable to serve.
            # Prefer copying into encounter dir before calling this helper.
            return f"/static/{p.name}"

    # overlay_path: Path = upload_path  # default fallback
    overlay_paths = []
    cams_for_api = []
    for c in cams_raw:
        pp = Path(c.get("cam_png", "")).resolve()
        if pp.exists():
            overlay_paths.append(str(pp))
            cams_for_api.append({
                "rank": c["rank"],
                "label": c["label"],
                "score": float(c["score"]),
                "path": str(pp),
                "url": _static_url(pp),   # assumes _static_url uses ARTIFACTS_DIR.resolve()
            })

    # choose first overlay for hero image (fallback to input if none)
    overlay_path = Path(overlay_paths[0]) if overlay_paths else upload_path

    # ---- LLM report ----
    payload = {
        "patient_context": {"age": patient.get("dob"), "sex": patient.get("sex")},
        "technical": {"qc": qc},
        "findings": findings,
        "model": model_name,
        "uncertainty": "Model-only; needs human review.",
    }
    draft = llm_report(payload)
    draft["report"] = coerce_report(draft.get("report"))
    draft["codes"]  = merge_codes(coerce_codes(draft.get("codes")), coerce_codes(map_codes(findings)))
    draft["flags"]  = coerce_flags(draft.get("flags"))
    draft["disclaimers"] = coerce_disclaimers(draft.get("disclaimers"))
    for k in ["indication", "technique", "comparison", "findings"]:
        draft["report"][k] = phi_scrub(draft["report"][k])
    disc = set(draft.get("disclaimers", [])); disc.add("AI-generated draft; requires radiologist review.")
    draft["disclaimers"] = list(disc)

    try:
        pkg = ReportPackage(**draft)
    except ValidationError as e:
        return {"error": "Schema validation failed", "details": json.loads(e.json())}
    
    overlay_paths = []
    for c in cams_for_api:
        p = c.get("path")
        if p and Path(p).is_file():
            overlay_paths.append(p)
    # ensure first overlay used for hero image too
    overlay_path = Path(overlay_paths[0]) if overlay_paths else upload_path


    # ---- PDF (patient + input + overlay + report) ----
    pdf_path = enc_dir / "report.pdf"
    export_pdf_with_images(
        report_json=pkg.model_dump(),
        patient=patient,
        input_img=str(upload_path),
        output_img=str(overlay_path),         # first overlay for hero/back-compat
        out_path=str(pdf_path),
        overlay_imgs=overlay_paths[:3],       # <<< show top-3 overlays
    )

    # ---- (optional) vector ingest ----
    ing = ingest_pdf_report(
        pdf_path=str(pdf_path),
        report_json=pkg.model_dump(),
        patient_id=patient_id,
        encounter_id=encounter_id,
        patient_meta={
            "mrn": patient.get("mrn"),
            "first": patient.get("first"),
            "last": patient.get("last"),
            "sex": patient.get("sex"),
            "dob": patient.get("dob"),
        },
    )

    # ---- artifacts ----
    artifacts = {
        "pdf": _static_url(pdf_path),
        "input_image": _static_url(upload_path),
        "output_image": _static_url(overlay_path),
        "cams": cams_for_api,          # <-- all three here with .url
    }

    return {
        "patient": patient,
        "encounter_id": encounter_id,
        "findings": findings,
        "qc": qc,
        "report": pkg.model_dump(),
        "artifacts": artifacts,
        "ingestion": ing,
    }



# run: uvicorn app:app --reload

@app.get("/search", dependencies=[Depends(auth_required)])
async def search_reports(
    q: str,
    k: int = 5,
    critical_only: bool = False,
    sex: Optional[str] = None,   # <-- instead of: str | None
    ):
    where = _build_where(critical_only=critical_only, sex=sex)
    hits = vector_search(q, k=k, where=where)
    return {"query": q, "k": k, "where": where, "results": hits}

# ---------------- Patients API ----------------

@app.post("/patients", response_model=Patient, dependencies=[Depends(auth_required)])
async def create_patient(payload: PatientCreate):
    db = _load_patients()

    # Prevent duplicate MRN if provided
    if payload.mrn:
        for rec in db.values():
            if (rec.get("mrn") or "").strip() and rec.get("mrn") == payload.mrn:
                raise HTTPException(status_code=409, detail="MRN already exists")

    now = time.time()
    pid = uuid.uuid4().hex
    rec = {
        "patient_id": pid,
        "created_at": now,
        "updated_at": now,
        **payload.model_dump(),
    }
    db[pid] = rec
    _save_patients(db)
    return rec


@app.get("/patients", dependencies=[Depends(auth_required)])
async def list_patients(q: Optional[str] = None, limit: int = 50, offset: int = 0):
    """
    List patients with a simple 'q' search across fields. Returns {total, items}.
    """
    db = _load_patients()
    items = list(db.values())
    if q:
        qq = q.lower().strip()
        items = [r for r in items if qq in _search_index(r)]
    total = len(items)
    items = items[offset : offset + max(0, min(limit, 500))]
    return {"total": total, "items": items}


@app.get("/patients/{patient_id}", response_model=Patient, dependencies=[Depends(auth_required)])
async def get_patient(patient_id: str):
    db = _load_patients()
    rec = db.get(patient_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Patient not found")
    return rec


@app.patch("/patients/{patient_id}", response_model=Patient, dependencies=[Depends(auth_required)])
async def update_patient(patient_id: str, patch: PatientUpdate):
    db = _load_patients()
    rec = db.get(patient_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Patient not found")

    data = patch.model_dump(exclude_unset=True)
    # MRN uniqueness if being changed
    if "mrn" in data and data["mrn"]:
        for pid, other in db.items():
            if pid != patient_id and (other.get("mrn") or "") == data["mrn"]:
                raise HTTPException(status_code=409, detail="MRN already exists")

    rec.update(data)
    rec["updated_at"] = time.time()
    db[patient_id] = rec
    _save_patients(db)
    return rec


@app.delete("/patients/{patient_id}", dependencies=[Depends(auth_required)])
async def delete_patient(patient_id: str):
    db = _load_patients()
    if patient_id not in db:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.pop(patient_id)
    _save_patients(db)
    return {"ok": True}

