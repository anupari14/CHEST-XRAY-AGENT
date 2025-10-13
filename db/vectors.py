# db/vectors.py
import os, time, hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# Embeddings (OpenAI with graceful fallback)
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if USE_OPENAI:
    from openai import OpenAI
    _oclient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _EMB_MODEL = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small")
else:
    # tiny local fallback (no network) to keep the pipeline working
    # pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer(os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2"))

def _sanitize_meta(value):
    # Chroma metadata must be str | int | float | bool | None
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    # collapse lists/dicts to compact strings (or drop if empty)
    if isinstance(value, (list, tuple)):
        flat = [str(v) for v in value if v not in (None, "")]
        return ", ".join(flat) if flat else None
    if isinstance(value, dict):
        # compress small dicts as key=value;key2=value2
        parts = [f"{k}={v}" for k, v in value.items() if v not in (None, "")]
        return "; ".join(parts) if parts else None
    return str(value)

def _embed(texts: List[str]) -> List[List[float]]:
    if USE_OPENAI:
        # batch to be kind to the API
        embs = []
        B = int(os.getenv("EMB_BATCH", "64"))
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            resp = _oclient.embeddings.create(model=_EMB_MODEL, input=batch)
            embs += [d.embedding for d in resp.data]
        return embs
    else:
        return _sbert.encode(texts, normalize_embeddings=True).tolist()

# Simple, dependable splitter (no hard dependency on tokenizers)
def split_text(s: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    s = " ".join(s.split())  # collapse whitespace
    chunks, i = [], 0
    while i < len(s):
        j = min(len(s), i + chunk_size)
        # soft break at sentence boundary if possible
        cut = s.rfind(". ", i, j)
        if cut == -1 or cut <= i + 200:
            cut = j
        chunks.append(s[i:cut].strip())
        i = max(cut - overlap, i + 1)
    return [c for c in chunks if c]

def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()

def get_collection(name: str = "cxr_reports"):
    client = chromadb.PersistentClient(
        path=os.getenv("CHROMA_DIR", "./vectordb"),
        settings=Settings(anonymized_telemetry=False)
    )
    # using cosine by default; set metadata indexing as needed
    col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return col

def _stable_id(patient_id: str, encounter_id: str, chunk_ix: int) -> str:
    raw = f"{patient_id}:{encounter_id}:{chunk_ix}"
    return hashlib.sha1(raw.encode()).hexdigest()

def ingest_pdf_report(
    pdf_path: str,
    report_json: Dict[str, Any],
    patient_id: str,
    encounter_id: str,
    patient_meta: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> Dict[str, Any]:
    """
    1) Extract text from PDF, 2) chunk, 3) embed, 4) upsert to Chroma.
    """
    col = get_collection()
    text = read_pdf_text(pdf_path)

    # Optional: prepend structured JSON for richer search
    structured = (
        f"Indication: {report_json['report'].get('indication','')}\n"
        f"Technique: {report_json['report'].get('technique','')}\n"
        f"Comparison: {report_json['report'].get('comparison','')}\n"
        f"Findings: {report_json['report'].get('findings','')}\n"
        f"Impression: {'; '.join(report_json['report'].get('impression', []))}\n"
    )
    fulltext = (structured + "\n" + text).strip()

    chunks = split_text(fulltext, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        chunks = [fulltext]

    embeddings = _embed(chunks)

    ids = [_stable_id(patient_id, encounter_id, i) for i in range(len(chunks))]

    icd_codes = None
    codes = report_json.get("codes")
    if isinstance(codes, dict):
        icd_list = []
        for item in codes.get("icd10", []) or []:
            if isinstance(item, dict) and item.get("code"):
                icd_list.append(str(item["code"]))
            elif isinstance(item, str):
                icd_list.append(item)
        icd_codes = ", ".join(icd_list) if icd_list else None

    metadatas = []
    for i, chunk in enumerate(chunks):
        md = {
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "page_chunk": i,
            "source": "pdf",
            "pdf_path": pdf_path,
            # use CSV string instead of list
            "icd10_codes": icd_codes,
            "critical": bool(report_json.get("flags", {}).get("critical")) if isinstance(report_json.get("flags"), dict) else False,
        }
        if patient_meta:
            md.update({f"patient_{k}": v for k, v in patient_meta.items()})

        # sanitize every value to Chroma-allowed primitives
        md = {k: _sanitize_meta(v) for k, v in md.items()}
        metadatas.append(md)

    col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)
    return {"chunks": len(chunks), "collection": col.name, "ids": ids}

def vector_search(query: str, k: int = 5, where: Optional[Dict[str, Any]] = None):
    col = get_collection()
    q_emb = _embed([query])[0]

    kwargs: Dict[str, Any] = {"query_embeddings": [q_emb], "n_results": k}
    if where:  # only include when not None/empty
        kwargs["where"] = where

    res = col.query(**kwargs)
    hits: List[Dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "score": float(res.get("distances", [[None]])[0][i]) if "distances" in res else None,
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
        })
    return hits
