# ui.py
import os
import io
import json
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = os.getenv("MEDAGENT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Medical Agent • Imaging Suite", layout="wide")

# ---------------- Session State ----------------
defaults = {
    "token": None,
    "user": None,
    "page": "cxr",                # default page after login
    "last_response": None,        # CXR results
    "last_uploaded_preview": None # CXR preview
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if st.session_state["token"]:
        h["Authorization"] = f"Bearer {st.session_state['token']}"
    return h

# ---------------- Styles ----------------
st.markdown("""
<style>
.app-topbar {display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin-bottom:4px;}
.app-title  {font-size: 1.5rem; font-weight: 700; margin: 0;}
.app-subtle {margin-top:-6px; color:#666; font-size:0.90rem;}
.navbar {display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;}
.navbtn {padding:.35rem .75rem; border-radius:10px; border:1px solid rgba(0,0,0,.08); cursor:pointer; font-weight:600; background:#f5f6f8;}
.navbtn.active {background:white; border-color:rgba(0,0,0,.15); box-shadow:0 1px 3px rgba(0,0,0,.06);}
.userbox {font-size:.95rem;}
.userbox .user {font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ---------------- Auth helpers ----------------
def do_login(username: str, password: str) -> bool:
    try:
        r = requests.post(f"{API_URL}/auth/login", data={"username": username, "password": password}, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.session_state["token"] = data.get("token")
        st.session_state["user"] = data.get("user")
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

def do_logout():
    try:
        if st.session_state.get("token"):
            requests.post(f"{API_URL}/auth/logout", headers=_headers(), timeout=10)
    except Exception:
        pass
    st.session_state.update({"token": None, "user": None, "page": "cxr",
                             "last_response": None, "last_uploaded_preview": None})

# ---------------- Login gate ----------------
if not st.session_state.get("token"):
    st.markdown("""
    <div class="app-topbar">
      <div>
        <div class="app-title">Medical Agent — Imaging Suite</div>
        <div class="app-subtle">AI-generated draft; requires radiologist review. Not for diagnostic use.</div>
      </div>
      <div></div>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted and do_login(u, p):
        st.rerun()
    st.stop()

# ---------------- Header + Navbar (shown only after login) ----------------
left, mid, right = st.columns([1.4, 2.2, 1])
with left:
    st.markdown("""
    <div class="app-topbar">
      <div>
        <div class="app-title">🩻 Medical Agent — Imaging Suite</div>
        <div class="app-subtle">AI-generated draft; requires radiologist review. Not for diagnostic use.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with mid:
    # Single-row top nav (no duplicates)
    labels = ["Patient Registration", "Chest X-Ray Analysis", "Brain Tumour Detection"]
    key_by_label = {
        "Patient Registration": "patient",
        "Chest X-Ray Analysis": "cxr",
        "Brain Tumour Detection": "brain",
    }
    label_by_key = {v: k for k, v in key_by_label.items()}
    current_idx = labels.index(label_by_key.get(st.session_state.get("page", "cxr"), "Chest X-Ray Analysis"))

    choice = st.radio(
        "Navigation",
        labels,
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["page"] = key_by_label[choice]


with right:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""<div class="userbox">👤 <span class="user">{st.session_state['user']}</span></div>""",
                    unsafe_allow_html=True)
    with c2:
        if st.button("Logout", key="logout_btn", type="secondary", use_container_width=True):
            do_logout()
            st.rerun()

# ---------------- Common helpers ----------------
def load_preview_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

# ======================================================================
# PAGE 1: PATIENT REGISTRATION
# ======================================================================
if st.session_state["page"] == "patient":
    st.divider()
    st.header("🧾 Patient Registration")

    with st.form("patient_form"):
        colA, colB = st.columns(2)
        with colA:
            mrn = st.text_input("MRN / Patient ID *")
            first = st.text_input("First Name *")
            last = st.text_input("Last Name *")
            sex = st.selectbox("Sex", ["", "M", "F", "Other"])
        with colB:
            dob = st.date_input("Date of Birth")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
        addr = st.text_area("Address")
        notes = st.text_area("Clinical Notes / Indication")
        submitted = st.form_submit_button("Register Patient")

    if submitted:
        payload = {
            "mrn": mrn, "first": first, "last": last,
            "sex": sex, "dob": str(dob),
            "phone": phone, "email": email,
            "address": addr, "notes": notes,
        }
        # If you add a backend route (e.g., POST /patients), this will work; otherwise shows a tip.
        try:
            r = requests.post(f"{API_URL}/patients", json=payload, headers=_headers(), timeout=30)
            if r.status_code >= 400:
                st.warning("Backend patient API not available; saved locally for now.")
                raise RuntimeError("No /patients endpoint")
            r.raise_for_status()
            st.success("Patient registered.")
        except Exception:
            # local fallback: keep one record in session
            st.session_state.setdefault("patients_local", []).append(payload)
            st.success("Patient stored locally (demo mode).")

    # show local list (demo)
    if st.session_state.get("patients_local"):
        st.subheader("Recent Patients (local)")
        dfp = pd.DataFrame(st.session_state["patients_local"])
        st.dataframe(dfp, use_container_width=True)

# ======================================================================
# PAGE 2: CHEST X-RAY ANALYSIS (existing flow)
# ======================================================================
elif st.session_state["page"] == "cxr":
    st.divider()
    st.header("Chest X-Ray Analysis")

    # Controls
    st.subheader("Controls")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        uploaded = st.file_uploader("Upload Chest X-ray (.png/.jpg/.dcm)", type=["png", "jpg", "jpeg", "dcm"])
    with c2:
        thr = st.slider("Prediction threshold", 0.0, 1.0, 0.60, 0.01)
    with c3:
        topk = st.slider("Top-K CAMs", 1, 6, 3)
    with c4:
        run_cxr = st.button("Analyze", use_container_width=True)

    if uploaded and (uploaded.type or "").startswith("image/"):
        img = load_preview_from_bytes(uploaded.getvalue())
        if img is not None:
            st.session_state["last_uploaded_preview"] = img

    if run_cxr and uploaded:
        with st.spinner("Analyzing…"):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
            data = {"threshold": str(thr), "topk": str(topk)}
            try:
                r = requests.post(f"{API_URL}/analyze", files=files, data=data, headers=_headers(), timeout=180)
                r.raise_for_status()
                st.session_state["last_response"] = r.json()
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Request failed: {e}")

    resp = st.session_state.get("last_response") or {}

    # Row: input image | overlays
    col_img, col_ov = st.columns([1, 1], vertical_alignment="top")
    with col_img:
        st.subheader("Input Image")
        if st.session_state["last_uploaded_preview"] is not None:
            st.image(st.session_state["last_uploaded_preview"], use_container_width=True)
        else:
            st.info("Upload a chest X-ray above to preview.")
    with col_ov:
        st.subheader("Overlays")
        arts = resp.get("artifacts", {}) or {}
        cam_list = arts.get("cams") or []
        if cam_list:
            cols = st.columns(3)
            for i, cam in enumerate(cam_list):
                url = cam.get("url") or cam.get("path")
                if isinstance(url, str):
                    if url.startswith("/"):
                        abs_url = f"{API_URL.rstrip('/')}{url}"
                    elif "artifacts/" in url:
                        rel = url.replace("\\", "/").split("artifacts/", 1)[1]
                        abs_url = f"{API_URL.rstrip('/')}/static/{rel}"
                    else:
                        abs_url = url
                    cap = f"{cam.get('rank')}. {cam.get('label')} (score={cam.get('score'):.3f})" if cam.get("score") is not None else f"{cam.get('rank')}. {cam.get('label')}"
                    with cols[i % 3]:
                        st.image(abs_url, caption=cap, use_container_width=True)
        else:
            st.info("No overlays for this analysis.")

    # Report + PDF
    st.divider()
    st.subheader("📝 Draft Report")
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

    pdf_val = (resp.get("artifacts", {}) or {}).get("pdf")
    if pdf_val:
        try:
            if isinstance(pdf_val, str) and pdf_val.startswith("/"):
                abs_pdf_url = f"{API_URL.rstrip('/')}{pdf_val}"
                rr = requests.get(abs_pdf_url, headers=_headers(), timeout=60)
                rr.raise_for_status()
                st.download_button("⬇️ Download PDF", data=rr.content, file_name="report.pdf", mime="application/pdf")
                st.caption(f"PDF: {abs_pdf_url}")
            else:
                with open(pdf_val, "rb") as fh:
                    st.download_button("⬇️ Download PDF", data=fh.read(), file_name="report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Could not load PDF: {e}")

    # Top-3 findings
    st.divider()
    st.subheader("🏷️ Top-3 Findings")
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

    # Search (last)
    st.divider()
    st.subheader("🔎 Semantic Search")
    q = st.text_input("Query across reports", placeholder="e.g., large right pleural effusion with cardiomegaly")
    kk = st.slider("Top K", 1, 20, 5)
    if st.button("Search") and q.strip():
        try:
            r = requests.get(f"{API_URL}/search", params={"q": q, "k": kk}, headers=_headers(), timeout=60)
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

# ======================================================================
# PAGE 3: BRAIN TUMOUR DETECTION (placeholder; optional /brain/infer)
# ======================================================================
elif st.session_state["page"] == "brain":
    st.divider()
    st.header("🧠 Brain Tumour Detection")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        brain_file = st.file_uploader("Upload Brain MRI (.png/.jpg/.nii/.nii.gz)",
                                      type=["png", "jpg", "jpeg"], key="brain_upl")  # extend types when backend ready
    with c2:
        b_conf = st.slider("Detection threshold", 0.0, 1.0, 0.35, 0.01)
    with c3:
        run_brain = st.button("Run Detection", use_container_width=True)

    if run_brain and brain_file:
        with st.spinner("Running brain tumour detection…"):
            try:
                # If you expose a backend route, e.g., POST /brain/infer
                files = {"file": (brain_file.name, brain_file.getvalue(), brain_file.type or "application/octet-stream")}
                data = {"threshold": str(b_conf)}
                r = requests.post(f"{API_URL}/brain/infer", files=files, data=data, headers=_headers(), timeout=180)
                r.raise_for_status()
                out = r.json()
                st.success("Inference complete.")
                overlay = (out.get("artifact") or {}).get("overlay")
                if overlay:
                    if overlay.startswith("/"):
                        abs_url = f"{API_URL.rstrip('/')}{overlay}"
                    else:
                        abs_url = overlay
                    st.image(abs_url, caption="Detections", use_container_width=True)
                st.subheader("Detections (raw)")
                st.json(out.get("detections") or out)
            except Exception:
                st.info("Brain inference backend not configured yet. This is a placeholder UI.")

    st.caption("Note: wire this page to your `/brain/infer` endpoint when available.")
