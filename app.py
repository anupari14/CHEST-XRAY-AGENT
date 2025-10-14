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
import hashlib


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

class ChatRequest(BaseModel):
    query: str
    k: int = 4
    filters: Optional[dict] = None
    session_id: Optional[str] = None

class ChatSource(BaseModel):
    rank: int
    score: float
    encounter_id: Optional[str] = None
    mrn: Optional[str] = None
    patient_name: Optional[str] = None
    pdf: Optional[str] = None     # /static/... (UI will prefix with API_URL)
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
    usage: Dict[str, Optional[int]] = {}

def _usage_to_dict(u: Any) -> Dict[str, Optional[int]]:
    """Normalize OpenAI usage to a plain dict (object- or dict-safe)."""
    if u is None:
        return {}
    try:
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    except Exception:
        return {
            "prompt_tokens": (u or {}).get("prompt_tokens"),
            "completion_tokens": (u or {}).get("completion_tokens"),
            "total_tokens": (u or {}).get("total_tokens"),
        }

def _build_context(hits: List[dict]) -> str:
    """Builds the text context given retrieval hits (each hit has .text and .meta)."""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.get("meta", {}) or {}
        # robust patient extraction
        patient = m.get("patient") or m.get("patient_meta") or {}
        first = patient.get("first") or m.get("first") or ""
        last  = patient.get("last")  or m.get("last")  or ""
        name  = m.get("patient_name") or m.get("name") or f"{first} {last}".strip() or "—"
        mrn   = patient.get("mrn") or m.get("mrn") or m.get("patient_mrn") or m.get("patient_id") or "—"
        score = float(h.get("score", 0.0))
        body  = (h.get("text") or "")[:2500]  # keep context bounded
        blocks.append(f"[{i}] Patient: {name} | MRN: {mrn} | Score: {score:.4f}\n{body}")
    return "\n\n---\n".join(blocks)

def _report_key(meta: Dict[str, Any]) -> str:
    """
    Build a stable key for a report from metadata with multiple fallbacks.
    This prevents duplicate rows when multiple chunks match the same report.
    Priority:
      1) doc_id / encounter_id / report_id
      2) PDF path (prefer enc_ folder name if present)
      3) local path (path/file/report_path)
      4) precomputed hash / text_id
      5) SHA1 of text (first 1000 chars)
    """
    m = meta or {}

    # 1) explicit ids if you have them
    for k in ("doc_id", "encounter_id", "report_id"):
        if m.get(k):
            return f"{k}:{m[k]}"

    # 2) PDF path (often /static/enc_xxx/report.pdf)
    pdf = m.get("pdf") or (m.get("artifacts") or {}).get("pdf")
    if pdf:
        try:
            p = Path(pdf)
            # try to use the encounter folder if available
            for part in reversed(p.parts):
                if part.startswith("enc_"):
                    return f"enc:{part}"
            return f"pdf:{p.stem}"
        except Exception:
            return f"pdf:{pdf}"

    # 3) local path fields you might have saved
    for k in ("path", "file", "report_path"):
        if m.get(k):
            try:
                return f"path:{Path(m[k]).stem}"
            except Exception:
                return f"path:{m[k]}"

    # 4) precomputed hash or text id
    for k in ("hash", "text_id"):
        if m.get(k):
            return f"{k}:{m[k]}"

    # 5) last resort: hash text
    txt = (m.get("full_text") or m.get("text") or "")
    if txt:
        h = hashlib.sha1(txt[:1000].encode("utf-8", "ignore")).hexdigest()
        return f"sha1:{h}"

    # absolute fallback (should rarely be hit)
    return f"row:{id(m)}"


def _dedupe_hits_by_report(hits: List[Dict[str, Any]], limit: Optional[int] = None) -> List[dict]:
    """
    Collapse multiple chunks from the same report into one row using _report_key.
    Keep the highest-scoring hit per report; return sorted DESC by score.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for h in hits or []:
        key = _report_key((h.get("meta") or {}))
        sc = float(h.get("score", 0.0))
        if key not in buckets or sc > buckets[key]["score"]:
            buckets[key] = {"hit": h, "score": sc}

    uniq = sorted(buckets.values(), key=lambda x: x["score"], reverse=True)
    out = [g["hit"] for g in uniq]
    if limit is not None:
        out = out[: int(limit)]
    return out


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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # 1) Retrieve similar chunks
    try:
        raw = vector_search(req.query, k=int(req.k), where=req.filters or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    raw_hits = raw.get("results", []) if isinstance(raw, dict) else raw or []

    # 2) Dedupe by report (so UI shows 1 row per report)
    uniq_hits = _dedupe_hits_by_report(raw_hits, limit=req.k)


    # 3) Build model context (you can choose uniq_hits or raw_hits for richer context)
    context = _build_context(uniq_hits)

    system = (
        "You are a careful radiology assistant. Use ONLY the provided CONTEXT. "
        "Each context block starts with a header like: "
        "[n] Patient: <NAME> | MRN: <MRN> | Score: <SCORE>. "
        "When referring to a patient, ALWAYS use the name and MRN from that header "
        "instead of 'context [n]'. "
        "Give a concise bulleted list (e.g., '• NAME (MRN: …): <brief evidence>') "
        "followed by a short summary. Do NOT invent details."
    )
    user = f"User question: {req.query}\n\nCONTEXT:\n{context}"

    # 4) LLM call
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        answer = resp.choices[0].message.content
        usage_dict = _usage_to_dict(getattr(resp, "usage", None))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # 5) Build sources from UNIQUE hits
    def _src_row(i: int, h: dict) -> Dict[str, Any]:
        m = h.get("meta", {}) or {}
        patient = m.get("patient") or m.get("patient_meta") or {}
        first = patient.get("first") or m.get("first") or ""
        last  = patient.get("last")  or m.get("last")  or ""
        name  = m.get("patient_name") or m.get("name") or f"{first} {last}".strip() or "—"
        mrn   = patient.get("mrn") or m.get("mrn") or m.get("patient_mrn") or m.get("patient_id") or "—"
        pdf_rel = (
            m.get("pdf")
            or (m.get("artifacts") or {}).get("pdf")
            or (f"/static/{m['encounter_id']}/report.pdf" if m.get("encounter_id") else None)
        )
        return {
            "rank": i + 1,
            "score": float(h.get("score", 0.0)),
            "encounter_id": m.get("encounter_id") or m.get("encounter") or "",
            "mrn": mrn,
            "patient_name": name,
            "pdf": pdf_rel,   # UI will prefix with MEDAGENT_API_URL
            "text": (h.get("text") or "")[:1000],
        }

    sources = [_src_row(i, h) for i, h in enumerate(uniq_hits)]

    return ChatResponse(answer=answer, sources=sources, usage=usage_dict)