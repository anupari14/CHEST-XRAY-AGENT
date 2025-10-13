# ui.py
import os
import io
import json
from typing import Optional

import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = os.getenv("MEDAGENT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Medical Agent • CXR", layout="wide")
st.title("Medical Agent — Chest X-ray (Demo)")
st.caption("AI-generated draft; requires radiologist review. Not for diagnostic use.")

# ---------------- State ----------------
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None
if "last_uploaded_preview" not in st.session_state:
    st.session_state["last_uploaded_preview"] = None

# ---------------- Controls (FIRST SECTION) ----------------
st.subheader("Controls")
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    uploaded = st.file_uploader(
        "Upload Chest X-ray (.png/.jpg/.dcm)", type=["png", "jpg", "jpeg", "dcm"]
    )
with c2:
    thr = st.slider("Prediction threshold", 0.0, 1.0, 0.60, 0.01)
with c3:
    topk = st.slider("Top-K CAMs", 1, 6, 3)
with c4:
    run = st.button("Analyze", use_container_width=True)

# Update preview from the current upload (no rerun)
if uploaded and (uploaded.type or "").startswith("image/"):
    try:
        img = Image.open(io.BytesIO(uploaded.getvalue())).convert("L")
        st.session_state["last_uploaded_preview"] = img
    except Exception:
        pass

# Analyze call
if run and uploaded:
    with st.spinner("Running analysis…"):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
        data = {"threshold": str(thr), "topk": str(topk)}
        try:
            r = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=180)
            r.raise_for_status()
            st.session_state["last_response"] = r.json()
            st.success("Analysis complete.")
        except Exception as e:
            st.error(f"Request failed: {e}")

# Cache current response for all sections below
resp = st.session_state.get("last_response") or {}

# ---------------- Helper: make CAM URL from url/path ----------------
def _to_static_url(cam_item: dict) -> Optional[str]:
    """
    Accepts either:
      - cam_item['url'] = '/static/enc_xxx/cams/cam_1_*.png'
      - cam_item['path'] = 'artifacts/enc_xxx/cams/cam_1_*.png' (local path)
    Returns absolute URL: f'{API_URL}/static/enc_xxx/cams/…'
    """
    u = cam_item.get("url")
    if isinstance(u, str) and u.startswith("/"):
        return f"{API_URL.rstrip('/')}{u}"

    p = cam_item.get("path")
    if isinstance(p, str):
        norm = p.replace("\\", "/")
        key = "artifacts/"
        idx = norm.find(key)
        if idx != -1:
            rel = norm[idx + len(key):]
            return f"{API_URL.rstrip('/')}/static/{rel}"
    return None

# ---------------- Row: Input image | Grad-CAM overlays ----------------
st.divider()
col_img, col_cam = st.columns([1, 1], vertical_alignment="top")

with col_img:
    st.subheader("Input Image")
    if st.session_state["last_uploaded_preview"] is not None:
        st.image(st.session_state["last_uploaded_preview"], use_container_width=True)
    else:
        st.info("Upload a chest X-ray above to preview.")

with col_cam:
    st.subheader("Grad-CAM Overlays")
    arts = resp.get("artifacts", {}) or {}
    cam_list = arts.get("cams") or []
    if cam_list:
        urls = []
        for c in cam_list:
            url = _to_static_url(c)
            if url:
                urls.append((url, c.get("label"), c.get("score"), c.get("rank")))
        if urls:
            grid = st.columns(3)
            for i, (abs_url, label, score, rank) in enumerate(urls):
                cap = f"{rank}. {label} (score={float(score):.3f})" if isinstance(score, (int, float, float)) else f"{rank}. {label}"
                with grid[i % 3]:
                    st.image(abs_url, caption=cap, use_container_width=True)
        else:
            st.warning("CAM payload present but no valid URLs/paths could be resolved.")
            with st.expander("Debug CAM payload"):
                st.json(cam_list)
    else:
        st.info("Run analysis to view Grad-CAM overlays.")

# ---------------- Report + PDF ----------------
st.divider()
st.header("Report Analysis")
report = resp.get("report", {}).get("report", {}) or {}
if report:
    st.markdown(f"**Indication**: {report.get('indication','')}")
    st.markdown(f"**Technique**: {report.get('technique','')}")
    st.markdown(f"**Comparison**: {report.get('comparison','')}")
    st.markdown("**Findings**:")
    st.write(report.get("findings",""))
    st.markdown("**Impression**:")
    for item in report.get("impression", []):
        st.write(f"- {item}")
else:
    st.info("No report yet. Upload an image and click **Analyze**.")

pdf_val = arts.get("pdf")
if pdf_val:
    try:
        if isinstance(pdf_val, str) and pdf_val.startswith("/"):
            abs_pdf_url = f"{API_URL.rstrip('/')}{pdf_val}"
            rr = requests.get(abs_pdf_url, timeout=60)
            rr.raise_for_status()
            st.download_button(
                "⬇️ Download PDF",
                data=rr.content,
                file_name="report.pdf",
                mime="application/pdf",
            )
            st.caption(f"PDF: {abs_pdf_url}")
        else:
            with open(pdf_val, "rb") as fh:
                st.download_button(
                    "⬇️ Download PDF",
                    data=fh.read(),
                    file_name="report.pdf",
                    mime="application/pdf",
                )
    except Exception as e:
        st.error(f"Could not load PDF: {e}")

# ---------------- Top-3 findings ----------------
st.divider()
st.header("Top-3 Findings")
findings = resp.get("findings", []) or []
if findings:
    try:
        top3 = sorted(findings, key=lambda z: float(z.get("prob", 0.0)), reverse=True)[:3]
    except Exception:
        top3 = findings[:3]
    if top3:
        df3 = pd.DataFrame(top3)[["label", "prob"]]
        df3["prob"] = df3["prob"].map(lambda x: round(float(x), 3))
        st.dataframe(df3, use_container_width=True, hide_index=True)
    else:
        st.info("No findings available.")
else:
    st.info("No findings available. Run analysis first.")

st.caption("© Demo UI — do not use for clinical decision-making.")

# ---------------- Semantic search (last) ----------------
st.divider()
st.header("Search")
q = st.text_input("Query across reports", placeholder="e.g., large right pleural effusion with cardiomegaly")
kk = st.slider("Top K", 1, 20, 5)
if st.button("Search") and q.strip():
    try:
        r = requests.get(f"{API_URL}/search", params={"q": q, "k": kk}, timeout=60)
        r.raise_for_status()
        res = r.json()
        for i, hit in enumerate(res.get("results", []), 1):
            score = hit.get("score")
            st.markdown(f"**{i}. score:** {score:.4f}" if isinstance(score, (int, float)) else f"**{i}.**")
            st.write((hit.get("text","") or "")[:800] + ("…" if hit.get("text") else ""))
            st.json(hit.get("meta", {}) or {})
            st.divider()
    except Exception as e:
        st.error(f"Search failed: {e}")
