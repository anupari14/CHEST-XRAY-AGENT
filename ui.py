# ui.py
import os
import io
import json
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

ASSETS = Path("assets")
LOGO = ASSETS / "AIDx-logo.png"   # update filename if different

st.set_page_config(
    page_title="Medical Agent",
    page_icon=str(LOGO),  # favicon
    layout="wide"
)
st.markdown("""
<style>
:root{
  --brand-primary:#435766;   /* Primary Blue-Grey */
  --brand-accent:#5E7382;    /* Accent Teal-Grey  */
  --brand-soft:#F5F7F8;      /* Soft White        */
  --brand-shadow:#2D3B44;    /* Shadow Grey       */
  --brand-highlight:#6E8797; /* Highlight Blue    */
}

/* Header bar + subtle divider */
[data-testid="stHeader"]{
  background: transparent;
  border-bottom: none;
}

/* Headings and section titles */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3{
  color: var(--brand-primary) !important;
}

/* Inputs: border tint */
.stTextInput input, .stPassword input, .stSelectbox div[data-baseweb="select"] > div {
  border: 1px solid var(--brand-accent) !important;
  box-shadow: none !important;
}

/* Primary button */
.stButton > button {
  background: var(--brand-primary) !important;
  color: #fff !important;
  border: 1px solid var(--brand-shadow) !important;
}
.stButton > button:hover {
  background: var(--brand-highlight) !important;
  border-color: var(--brand-shadow) !important;
}

/* Accent details: help text, small labels */
small, .stCaption, .st-emotion-cache-16idsys p {
  color: var(--brand-accent) !important;
}

/* Optional: “card” feel for your Login block */
.section-card {
  background: #FFFFFF;
  border: 1px solid var(--brand-accent);
  border-radius: 10px;
  padding: 1.2rem 1.2rem 1.0rem;
  box-shadow: 0 2px 10px rgba(45,59,68,0.08);
}
/* HIDE Streamlit's default header + toolbar/menu */
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }  /* removes Deploy & kebab menu */
#MainMenu { visibility: hidden; }                         /* legacy menu */

/* Pull content up now that the default header is gone */
.block-container { padding-top: 0.75rem; }
            
/* === Chatbot panel === */
.chat-panel{
  background:#ffffff;
  border:1px solid var(--brand-accent);
  border-radius:16px;
  box-shadow:0 6px 20px rgba(0,0,0,.08);
  max-width: 980px;
}
.chat-header{
  display:flex; align-items:center; gap:.6rem;
  padding:12px 16px; border-bottom:1px solid #e7eef4;
}
.chat-scroll{
  max-height: 520px;               /* adjust height */
  overflow-y: auto;
  padding: 14px 16px 4px 16px;
  background:#fbfdff;
}

/* message bubbles */
.msg{ display:flex; gap:.6rem; margin:8px 0; }
.msg .avatar{
  width:28px; height:28px; border-radius:50%;
  background: var(--brand-primary); color:#fff;
  display:flex; align-items:center; justify-content:center;
  font-size:.8rem; flex-shrink:0;
}
.msg .bubble{
  max-width:72%;
  padding:10px 14px; border-radius:16px;
  box-shadow:0 2px 8px rgba(67,87,102,.12);
  line-height:1.45;
}
.msg.assistant .bubble{
  background:#F5F7F8; color:#2D3B44;
  border-top-left-radius:6px;
}
.msg.user{ justify-content:flex-end; }
.msg.user .bubble{
  background:#E8EEF3; color:#0F172A;
  border-top-right-radius:6px;
}

/* input bar inside the card */
.chat-inputbar{
  display:flex; gap:.6rem; padding:10px; border-top:1px solid #e7eef4; background:#fff;
}
.chat-inputbar .stTextInput>div>div>input{
  background:#F5F7F8 !important;
  border:1px solid var(--brand-accent) !important;
}
.chat-inputbar .stButton>button{
  background:var(--brand-primary) !important; color:#fff !important;
  border:1px solid var(--brand-primary) !important;
}
/* Header layout fixes */
.topbar { display: block; }  /* ensure normal flow */

/* Keep the tabs inside their column and allow horizontal scroll if tight */
.topbar .stTabs { margin-bottom: 0; border-bottom: none; }
.topbar .stTabs [role="tablist"]{
  overflow-x: auto;           /* prevents pushing into the logout column */
  white-space: nowrap;
  gap: 22px;
  scrollbar-width: none;
}
.topbar .stTabs [role="tablist"]::-webkit-scrollbar{ display: none; }

/* Optional: reduce tab underline thickness */
.topbar .stTabs [data-baseweb="tab-highlight"]{ height: 2px; }

/* Make sure the logout column content stays right aligned and on top */
.topbar .logout-col { display:flex; justify-content:flex-end; align-items:center; }
            

/* Footer chat dock */
.chat-footer { height: 260px; }
.chat-footer.collapsed{ height: 56px; } 

/* Footer header bar */
.chat-footer-head {
  display:flex; align-items:center; justify-content:space-between;
  gap: .75rem; padding: 8px 14px; background: #f7fafc;
  border-bottom: 1px solid #e7eef4;
}
.chat-footer-head h4 { margin:0; color: var(--brand-primary); }

/* Scrollable chat area */
.chat-footer-scroll {
  flex: 1 1 auto; overflow-y: auto; padding: 12px 16px; background:#fbfdff;
}

/* Input bar */
.chat-footer-input {
  flex: 0 0 auto; display:flex; gap:.5rem; padding: 10px 12px;
  border-top: 1px solid #e7eef4; background:#fff;
}
.chat-footer-input .stTextInput>div>div>input{
  background:#F5F7F8 !important;
  border:1px solid var(--brand-accent) !important;
}
.chat-footer-input .stButton>button{
  background:var(--brand-primary) !important; color:#fff !important;
  border:1px solid var(--brand-primary) !important;
}

/* Bubbles (simple) */
.msg{ display:flex; gap:.6rem; margin:8px 0; }
.msg .avatar{
  width:26px; height:26px; border-radius:50%; background:var(--brand-primary);
  color:#fff; display:flex; align-items:center; justify-content:center; font-size:.75rem;
}
.msg .bubble{
  max-width:76%; padding:10px 14px; border-radius:16px; box-shadow:0 2px 8px rgba(67,87,102,.12);
  line-height:1.45;
}
.msg.assistant .bubble{ background:#F5F7F8; color:#2D3B44; border-top-left-radius:6px; }
.msg.user{ justify-content:flex-end; }
.msg.user .bubble{ background:#E8EEF3; color:#0F172A; border-top-right-radius:6px; }

/* Collapsed state */
.chat-footer.collapsed{ height: 56px; }
.chat-footer.collapsed .chat-footer-scroll,
.chat-footer.collapsed .chat-footer-input{ display:none; }
            
/* Hide Streamlit’s default footer/menu to avoid extra bars */
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }


</style>
""", unsafe_allow_html=True)


# Streamlit ≥ 1.31 shows a proper header logo (top-left)
try:
    st.logo(str(LOGO))
except Exception:
    # Fallback for older versions: simple header row
    c1, c2 = st.columns([1, 8])
    with c1:
        st.image(str(LOGO), use_container_width=True)
    with c2:
        st.markdown("### Medical Agent")
        st.caption("AI-generated draft; requires radiologist review. Not for diagnostic use.")


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

def _append_chat(role: str, content: str):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append((role, content))


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
/* Top app header bar (our custom one) */
/* Sticky top app header */
/* Transparent, borderless header (no horizontal bar) */
.topbar {
  position: sticky; top: 0; z-index: 999;
  background: transparent;     /* was var(--brand-shadow) */
  border-bottom: none;         /* was 1px solid var(--brand-accent) */
  padding: 4px 0;              /* reduce height */
  margin: 0 0 8px 0;           /* remove negative margins that created a stripe */
}


/* Keep header widgets readable on dark */
.topbar label, .topbar p, .topbar span, .topbar div { color: var(--brand-soft) !important; }


/* Logout button on header */
.topbar .stButton > button {
  background: var(--brand-primary) !important;
  color: #fff !important;
  border: 1px solid var(--brand-primary) !important;
}
.topbar .stButton > button:hover { background: var(--brand-highlight) !important; }

/* Logo sizing/alignment */
.topbar .logo-box { display:flex; align-items:center; gap:.5rem; }
.topbar .logo-box img { width: 120px; height:auto; } /* tweak 100–160 as you like */
            
/* Make the chat popover wider and nicer */
[data-testid="stPopover"]{
  min-width: 520px;            /* popover width */
  border: 1px solid var(--brand-accent);
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0,0,0,.18);
}
.header-chat-scroll{
  max-height: 380px;           /* scrollable history height */
  overflow-y: auto;
  padding: 8px 2px 0 2px;
}
.header-chat-inputbar .stTextInput>div>div>input{
  background:#F5F7F8 !important;
  border:1px solid var(--brand-accent) !important;
}
.header-chat-inputbar .stButton>button{
  background:var(--brand-primary) !important;
  color:#fff !important;
  border:1px solid var(--brand-primary) !important;
}


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
    st.markdown("### Login")

    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted and do_login(u, p):
        st.rerun()
    st.stop()

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
def render_patient_page():
    st.divider()
    st.header("New Patient Registration")

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
def render_cxr_page():
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
    




# ======================================================================
# PAGE 3: BRAIN TUMOUR DETECTION (placeholder; optional /brain/infer)
# ======================================================================
def render_brain_page():
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

# ---- put these helpers near render_footer_chat() ----
# ----- helpers used by the footer callbacks -----
def _footer_send():
    msg = (st.session_state.get("footer_chat_text") or "").strip()
    if not msg:
        return
    st.session_state.setdefault("chat_history", []).append(("user", msg))
    # call backend
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"query": msg, "k": 4},
            headers=_auth_hdrs(),
            cookies=_auth_cookies(),
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "")
        st.session_state["chat_history"].append(("assistant", answer))

        sources = data.get("sources", []) or []
        if sources:
            lines = []
            for s in sources[:3]:
                name = s.get("patient_name") or "—"
                mrn  = s.get("mrn") or "—"
                score = float(s.get("score", 0.0))
                lines.append(f"• {name} (MRN {mrn}) — {score:.3f}")
            st.session_state["chat_history"].append(("assistant", "**Matches**  \n" + "\n".join(lines)))
    except Exception as e:
        st.session_state["chat_history"].append(("assistant", f"⚠️ Chat failed: {e}"))

    # safe to clear inside callback
    st.session_state.footer_chat_text = ""

def _footer_clear():
    st.session_state["chat_history"] = []
    st.session_state["footer_chat_text"] = ""


def render_footer_chat():
    # init state
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("chat_open", True)
    st.session_state.setdefault("footer_chat_text", "")

    # dynamic bottom padding so content isn't hidden under fixed footer
    pad_px = 260 if st.session_state["chat_open"] else 56
    st.markdown(f"<style>.page-bottom-pad{{padding-bottom:{pad_px}px}}</style>", unsafe_allow_html=True)
    st.markdown('<div class="page-bottom-pad"></div>', unsafe_allow_html=True)

    # footer outer
    collapsed_cls = " collapsed" if not st.session_state["chat_open"] else ""
    st.markdown(f'<div class="chat-footer{collapsed_cls}">', unsafe_allow_html=True)

    # header row with toggle
    head_l, head_r = st.columns([6, 2])
    with head_l:
        st.markdown('<div class="chat-footer-head"><h4>💬 Report Chatbot</h4></div>', unsafe_allow_html=True)
    with head_r:
        if st.button("Hide" if st.session_state["chat_open"] else "Show",
                     key="chat_toggle", use_container_width=True):
            st.session_state["chat_open"] = not st.session_state["chat_open"]
            st.rerun()

    if st.session_state["chat_open"]:
        # history
        st.markdown('<div class="chat-footer-scroll">', unsafe_allow_html=True)
        if st.session_state["chat_history"]:
            for role, content in st.session_state["chat_history"]:
                role = role if role in ("user", "assistant") else "assistant"
                av = "U" if role == "user" else "AI"
                html = f'''
                <div class="msg {'user' if role=='user' else 'assistant'}">
                  <div class="avatar">{av}</div>
                  <div class="bubble">{content}</div>
                </div>'''
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="msg assistant"><div class="avatar">AI</div>'
                '<div class="bubble">Hi! Ask about findings, impressions, or trends.</div></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # input bar (callbacks handle actions)
        st.markdown('<div class="chat-footer-input">', unsafe_allow_html=True)
        col_in, col_send, col_clear = st.columns([8, 1.2, 1.2])
        with col_in:
            st.text_input(
                "Type a message…",
                key="footer_chat_text",
                label_visibility="collapsed",
                placeholder="Type a message…",
            )
        with col_send:
            st.button("Send", use_container_width=True, key="footer_chat_send", on_click=_footer_send)
        with col_clear:
            st.button("Clear", type="secondary", use_container_width=True, key="footer_chat_clear", on_click=_footer_clear)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end .chat-footer




# ---------------- Top Header Bar (sticky with logo + TABS + logout) ----------------
# ---------------- Top Header Bar (logo + tabs + CHAT + logout) ----------------
st.markdown('<div class="topbar">', unsafe_allow_html=True)

c_logo, c_tabs, c_logout = st.columns([1.1, 8.2, 1.5])

with c_logo:
    st.image(str(LOGO), use_container_width=False, width=120)

with c_tabs:
    tab_patient, tab_cxr, tab_brain = st.tabs(
        ["Patient Registry", "Chest X-Ray Analysis", "Brain Tumour Detection"]
    )

with c_logout:
    # ensure right alignment via CSS class
    st.markdown('<div class="logout-col">', unsafe_allow_html=True)
    user_lbl = st.session_state.get("user", "")
    if st.button(f"Logout ( {user_lbl} )", type="primary", use_container_width=True, key="logout_btn"):
        do_logout(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---- Render tab contents below ----
with tab_patient: render_patient_page()
with tab_cxr:     render_cxr_page()
with tab_brain:   render_brain_page()
render_footer_chat()





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

        




