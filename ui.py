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

def _auth_hdrs():
    return st.session_state.get("auth_headers") or {}

def _auth_cookies():
    return st.session_state.get("auth_cookies") or {}

def do_nav_search():
    q = (st.session_state.get("nav_search_q") or "").strip()
    if not q:
        st.session_state["nav_search_results"] = []
        return
    try:
        r = requests.get(
            f"{API_URL}/search",
            params={"q": q, "k": 3},  # Top K hardcoded
            headers=_headers(),
            timeout=30,
        )
        r.raise_for_status()
        hits = r.json().get("results", []) or []
        # sort DESC by score & cap to 3
        hits = sorted(hits, key=lambda h: float(h.get("score", 0.0)), reverse=True)[:3]
        st.session_state["nav_search_results"] = hits
    except Exception as e:
        st.session_state["nav_search_results"] = []
        st.session_state["nav_search_error"] = str(e)

def clear_nav_search():
    st.session_state["nav_search_q"] = ""
    st.session_state["nav_search_results"] = []
    st.session_state["nav_search_error"] = ""

def _extract_patient(meta: dict):
    """Return (name, mrn) from a variety of possible metadata shapes."""
    meta = meta or {}
    p = meta.get("patient") or meta.get("patient_meta") or {}

    # names
    first = p.get("first") or meta.get("first") or meta.get("patient_first") or meta.get("fname") or ""
    last  = p.get("last")  or meta.get("last")  or meta.get("patient_last")  or meta.get("lname")  or ""
    name  = meta.get("name") or p.get("name") or f"{first} {last}".strip()
    name  = name if (name and name.strip()) else "—"

    # mrn/id
    mrn = (
        p.get("mrn")
        or meta.get("mrn")
        or meta.get("patient_mrn")
        or meta.get("patient_id")   # sometimes you stored MRN here
        or meta.get("id")
        or "—"
    )
    return name, mrn



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
    
# Make /static/... absolute for the browser; pass through http(s) and local paths
def _abs_or_static(p) -> Optional[str]:
    if not p:
        return None
    s = str(p)
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if s.startswith("/static/"):
        return f"{API_URL.rstrip('/')}{s}"
    if s.startswith("static/"):
        return f"{API_URL.rstrip('/')}/{s}"
    return s


# ---------------- Login gate ----------------
if not st.session_state.get("token"):
    st.markdown("""
    <div class="app-topbar">
      <div>
        <div class="app-title">Medical Agent</div>
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

# ---------------- Header (with navbar + user/Logout) ----------------
left, mid, right = st.columns([0.5, 2.2, 1])
with left:
    # Single top nav
    labels = ["Patient Registration", "Chest X-Ray Analysis", "Brain Tumour Detection"]
    key_by_label = {
        "Patient Registration": "patient",
        "Chest X-Ray Analysis": "cxr",
        "Brain Tumour Detection": "brain",
    }
    label_by_key = {v: k for k, v in key_by_label.items()}
    current_idx = labels.index(label_by_key.get(st.session_state.get("page", "cxr"), "Chest X-Ray Analysis"))
    choice = st.radio("Navigation", labels, index=current_idx, horizontal=True, label_visibility="collapsed")
    st.session_state["page"] = key_by_label[choice]

with mid:
    with st.container(border=True):
        st.text_input(
            "Search reports",
            key="nav_search_q",
            placeholder="Search reports…",
            label_visibility="collapsed",
            on_change=do_nav_search,   # <-- ENTER triggers search
        )


with right:
    if st.button(f"Logout ( {st.session_state.get('user','')} )", key="logout_btn", type="secondary", use_container_width=True):
        do_logout()
        st.rerun()

# --- Search Results (from top navbar search) ---
# --- Search Results (under navbar) ---
if "nav_search_error" in st.session_state and st.session_state.get("nav_search_error"):
    st.error(f"Search failed: {st.session_state['nav_search_error']}")

results = st.session_state.get("nav_search_results") or []
if results:
    st.subheader("Search results")
    for i, hit in enumerate(results, 1):
        meta = hit.get("meta", {}) or {}
        patient = meta.get("patient") or meta.get("patient_meta") or {}
        name, mrn = _extract_patient(meta)
        score = float(hit.get("score", 0.0))

        # guess a report link
        pdf_rel = meta.get("pdf") \
                  or (meta.get("artifacts") or {}).get("pdf") \
                  or (f"/static/{meta['encounter_id']}/report.pdf" if meta.get("encounter_id") else None)
        pdf_url = _abs_or_static(pdf_rel) if pdf_rel else None

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
            c1.markdown(f"**{i}. {name}**")
            c2.markdown(f"**MRN:** {mrn}")
            c3.markdown(f"**Score:** {score:.4f}")
            if pdf_url:
                c4.markdown(f"[Open report]({pdf_url})")

    st.button("Clear results", on_click=clear_nav_search, type="secondary")

        

# ---------------- Common helpers ----------------
def load_preview_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def _abs_or_static(url_or_path: Optional[str]) -> Optional[str]:
    if not isinstance(url_or_path, str) or not url_or_path:
        return None
    if url_or_path.startswith("/"):
        return f"{API_URL.rstrip('/')}{url_or_path}"
    if "artifacts/" in url_or_path:
        rel = url_or_path.replace("\\", "/").split("artifacts/", 1)[1]
        return f"{API_URL.rstrip('/')}/static/{rel}"
    return url_or_path

# ======================================================================
# PAGE 1: PATIENT REGISTRATION  (integrated with backend)
# ======================================================================
if st.session_state["page"] == "patient":
    st.divider()
    st.header("🧾 Patient Registration")

    # ---------- Create patient ----------
    with st.form("patient_form"):
        colA, colB = st.columns(2)
        with colA:
            mrn = st.text_input("MRN / Patient ID")
            first = st.text_input("First Name *")
            last = st.text_input("Last Name *")
            sex = st.selectbox("Sex", ["", "M", "F", "Other"])
        with colB:
            dob = st.date_input("Date of Birth", format="YYYY-MM-DD")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
        address = st.text_area("Address")
        notes = st.text_area("Clinical Notes / Indication")
        submitted = st.form_submit_button("Register Patient")

    if submitted:
        if not first.strip() or not last.strip():
            st.error("First and Last name are required.")
        else:
            payload = {
                "mrn": mrn.strip() or None,
                "first": first.strip(),
                "last": last.strip(),
                "sex": sex if sex else None,
                "dob": str(dob) if dob else None,
                "phone": phone.strip() or None,
                "email": email.strip() or None,
                "address": address.strip() or None,
                "notes": notes.strip() or None,
            }
            try:
                r = requests.post(f"{API_URL}/patients", json=payload, headers=_headers(), timeout=30)
                if r.status_code == 409:
                    st.warning("A patient with this MRN already exists.")
                r.raise_for_status()
                st.success("Patient registered.")
                st.session_state["__patients_refresh__"] = st.session_state.get("__patients_refresh__", 0) + 1
            except requests.HTTPError as e:
                try:
                    detail = r.json().get("detail")
                except Exception:
                    detail = str(e)
                st.error(f"Create failed: {detail}")
            except Exception as e:
                st.error(f"Create failed: {e}")

    # ---------- Search & list ----------
    st.divider()
    st.subheader("Patients")

    ls1, ls2, ls3 = st.columns([2, 1, 1])
    with ls1:
        q = st.text_input("Search", value=st.session_state.get("__patients_q__", ""), placeholder="name / MRN / phone / notes")
    with ls2:
        limit = st.number_input("Page size", min_value=5, max_value=100, value=20, step=5)
    with ls3:
        if st.button("Refresh"):
            st.session_state["__patients_refresh__"] = st.session_state.get("__patients_refresh__", 0) + 1
    st.session_state["__patients_q__"] = q

    # fetch list
    try:
        resp_list = requests.get(
            f"{API_URL}/patients",
            params={"q": q, "limit": int(limit), "offset": 0},
            headers=_headers(),
            timeout=30,
        )
        resp_list.raise_for_status()
        data = resp_list.json()
        items = data.get("items", [])
        total = data.get("total", len(items))
    except Exception as e:
        items, total = [], 0
        st.error(f"List failed: {e}")

    # render table with delete actions
    if items:
        st.markdown("**Results:** " + str(total))
        for i, p in enumerate(items, 1):
            with st.container(border=True):
                row1 = st.columns([2, 2, 1.2, 1.2, 1.2, 0.8])
                with row1[0]:
                    st.markdown(f"**{p.get('first','')} {p.get('last','')}**")
                    st.caption(f"MRN: {p.get('mrn') or '—'}\nID: {p.get('patient_id') or '—'}")
                with row1[1]:
                    st.write(p.get("notes") or "")
                with row1[2]:
                    st.caption("Sex / DOB")
                    st.write(f"{p.get('sex') or '—'} / {p.get('dob') or '—'}")
                with row1[3]:
                    st.caption("Phone")
                    st.write(p.get("phone") or "—")
                with row1[4]:
                    st.caption("Email")
                    st.write(p.get("email") or "—")
                with row1[5]:
                    if st.button("🗑️", key=f"del_{p['patient_id']}", help="Delete"):
                        try:
                            dr = requests.delete(
                                f"{API_URL}/patients/{p['patient_id']}",
                                headers=_headers(),
                                timeout=20,
                            )
                            dr.raise_for_status()
                            st.success("Deleted.")
                            st.session_state["__patients_refresh__"] = st.session_state.get("__patients_refresh__", 0) + 1
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
    else:
        st.info("No patients found.")

# ======================================================================
# PAGE 2: CHEST X-RAY ANALYSIS (MRN required + validated; PDF includes images)
# ======================================================================
elif st.session_state["page"] == "cxr":
    st.divider()
    st.header("Chest X-Ray Analysis")

    # Controls
    st.subheader("Controls")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        uploaded = st.file_uploader("Upload Chest X-ray (.png/.jpg/.dcm)", type=["png", "jpg", "jpeg", "dcm"])
        patient_id = st.text_input("Patient ID (MRN) *", help="Enter a valid patient MRN/ID already registered.")
        run_cxr = st.button("Analyze", use_container_width=True)
    with c2:
        thr = 0.1
    with c3:
        topk = 20
    
        

    if uploaded and (uploaded.type or "").startswith("image/"):
        img = load_preview_from_bytes(uploaded.getvalue())
        if img is not None:
            st.session_state["last_uploaded_preview"] = img

    # Validate & Analyze
    if run_cxr:
        if not patient_id.strip():
            st.error("Patient ID (MRN) is required.")
        elif not uploaded:
            st.error("Please upload a chest X-ray image.")
        else:
            # Validate patient exists
            try:
                rcheck = requests.get(f"{API_URL}/patients/{patient_id.strip()}",
                                      headers=_headers(), timeout=20)
                rcheck.raise_for_status()
            except Exception:
                st.error("Invalid Patient ID. Please register the patient or check the MRN.")
            else:
                with st.spinner("Analyzing…"):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
                    data = {
                        "threshold": str(thr),
                        "topk": str(topk),
                        "patient_id": patient_id.strip()
                    }
                    try:
                        r = requests.post(f"{API_URL}/analyze", files=files, data=data,
                                          headers=_headers(), timeout=180)
                        r.raise_for_status()
                        st.session_state["last_response"] = r.json()
                        st.success("Analysis complete.")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    # Normalize response dict safely
    resp = st.session_state.get("last_response")
    if not isinstance(resp, dict):
        resp = {}

    # Patient summary + images row
    # ---- Patient summary & images row (replace this whole section) ----
    st.divider()
    pinfo = resp.get("patient") or {}
    # NEW
    if pinfo:
        st.subheader("Patient")
        full_name = f"{pinfo.get('first','')} {pinfo.get('last','')}".strip() or "—"
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**MRN**<br/>{pinfo.get('mrn','—')}", unsafe_allow_html=True)
                st.markdown(f"**Sex**<br/>{pinfo.get('sex','—')}", unsafe_allow_html=True)
                st.markdown(f"**DOB**<br/>{pinfo.get('dob','—')}", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Patient ID**<br/>{pinfo.get('patient_id','—')}", unsafe_allow_html=True)
                st.markdown(f"**Name**<br/>{full_name}", unsafe_allow_html=True)
            with col3:
                st.markdown(f"**Phone**<br/>{pinfo.get('phone','—')}", unsafe_allow_html=True)
                st.markdown(f"**Email**<br/>{pinfo.get('email','—')}", unsafe_allow_html=True)


    arts = resp.get("artifacts", {}) or {}
    in_url  = _abs_or_static(arts.get("input_image"))
    out_url = _abs_or_static(arts.get("output_image"))

    # If backend didn’t set output_image, try first CAM
    if not out_url:
        cams = arts.get("cams") or []
        if cams:
            out_url = _abs_or_static(cams[0].get("url") or cams[0].get("path"))

    # If overlay accidentally equals input, try to find a different cam
    if out_url and in_url and out_url == in_url:
        for cam in (arts.get("cams") or []):
            cand = _abs_or_static(cam.get("url") or cam.get("path"))
            if cand and cand != in_url:
                out_url = cand
                break

    ci, co = st.columns(2)
    with ci:
        st.subheader("Input X-ray")
        if in_url:
            st.image(in_url, use_container_width=True)
        elif st.session_state["last_uploaded_preview"] is not None:
            st.image(st.session_state["last_uploaded_preview"], use_container_width=True)
        else:
            st.info("Upload a chest X-ray above to preview.")

    with co:
        st.subheader("AI Findings")
        cam_list = arts.get("cams") or []
        if cam_list:
            cols = st.columns(3)
            for i, cam in enumerate(cam_list[:3]):  # show top-3
                # Convert backend "/static/..." or relative paths to absolute URLs
                abs_url = _abs_or_static(cam.get("url") or cam.get("path"))
                if not abs_url:
                    continue
                cap = (
                    f"{cam.get('rank')}. {cam.get('label')} (score={cam.get('score'):.3f})"
                    if isinstance(cam.get("score"), (int, float))
                    else f"{cam.get('rank')}. {cam.get('label')}"
                )
                with cols[i % 3]:
                    st.image(abs_url, caption=cap, use_container_width=True)
        else:
            st.info("No overlays for this analysis.")



    # Report + PDF
    st.divider()
    st.subheader("Report")
    report_block = resp.get("report", {}) or {}
    report = report_block.get("report", {}) or {}
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
            abs_pdf_url = _abs_or_static(pdf_val)
            if abs_pdf_url and abs_pdf_url.startswith("http"):
                rr = requests.get(abs_pdf_url, headers=_headers(), timeout=60)
                rr.raise_for_status()
                st.download_button("⬇️ Download PDF", data=rr.content, file_name="report.pdf", mime="application/pdf")
                st.caption(f"PDF: {abs_pdf_url}")
            else:
                with open(pdf_val, "rb") as fh:
                    st.download_button("⬇️ Download PDF", data=fh.read(), file_name="report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Could not load PDF: {e}")
    
    # ---- Chatbot (RAG) ----
    st.divider()
    st.header("💬 Report Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # render prior messages
    for role, content in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("Ask about findings, impressions, trends…")
    if prompt:
        # show user message
        st.session_state["chat_history"].append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # call backend
        try:
            r = requests.post(
                f"{API_URL}/chat",
                json={"query": prompt, "k": 4},
                headers=_auth_hdrs(),
                cookies=_auth_cookies(),
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", []) or []

            # assistant msg + citations
            with st.chat_message("assistant"):
                st.markdown(answer)

                sources = data.get("sources", []) or []
                if sources:
                    st.markdown("**Matches**")
                    for s in sources:
                        name = s.get("patient_name") or "—"
                        mrn  = s.get("mrn") or "—"
                        score = float(s.get("score", 0.0))
                        pdf = _abs_or_static(s.get("pdf")) if s.get("pdf") else None

                        cols = st.columns([5, 2, 2, 3])
                        cols[0].markdown(f"**{name}**")
                        cols[1].markdown(f"**MRN:** {mrn}")
                        cols[2].markdown(f"**Score:** {score:.4f}")
                        if pdf:
                            cols[3].markdown(f"[Open report]({pdf})")
            st.session_state["chat_history"].append(("assistant", answer))
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Chat failed: {e}")

    # clear thread
    if st.button("Clear chat", type="secondary"):
        st.session_state["chat_history"] = []



# ======================================================================
# PAGE 3: BRAIN TUMOUR DETECTION (placeholder; optional /brain/infer)
# ======================================================================
elif st.session_state["page"] == "brain":
    st.divider()
    st.header("Brain Tumour Detection")

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
                files = {"file": (brain_file.name, brain_file.getvalue(), brain_file.type or "application/octet-stream")}
                data = {"threshold": str(b_conf)}
                r = requests.post(f"{API_URL}/brain/infer", files=files, data=data, headers=_headers(), timeout=180)
                r.raise_for_status()
                out = r.json()
                st.success("Inference complete.")
                overlay = (out.get("artifact") or {}).get("overlay")
                if overlay:
                    abs_url = _abs_or_static(overlay)
                    if abs_url:
                        st.image(abs_url, caption="Detections", use_container_width=True)
                st.subheader("Detections (raw)")
                st.json(out.get("detections") or out)
            except Exception:
                st.info("Brain inference backend not configured yet. This is a placeholder UI.")

    st.caption("Note: wire this page to your `/brain/infer` endpoint when available.")
