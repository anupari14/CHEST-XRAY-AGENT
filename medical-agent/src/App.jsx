// src/App.jsx
import { useMemo, useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './styles/theme.css';
import LoginPage from './components/LoginPage.jsx';
import Logo from './components/Logo.jsx';
import FloatingChatbot from './components/FloatingChatbot.jsx';

const API_URL =
  import.meta?.env?.VITE_MEDAGENT_API_URL ||
  'http://127.0.0.1:8000';

export default function App() {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('cxr'); // 'patient' | 'cxr' | 'brain'

  // Navbar search (optional wiring)
  const [navQ, setNavQ] = useState('');
  const [navResults, setNavResults] = useState([]);
  const [navErr, setNavErr] = useState('');

  const headers = useMemo(() => {
    const h = { Accept: 'application/json' };
    if (token) h['Authorization'] = `Bearer ${token}`;
    return h;
  }, [token]);

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    setActiveTab('cxr');
  };

  if (!token) {
    return <LoginPage onSuccess={(t, u) => { setToken(t); setUser(u); }} />;
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--brand-soft)' }}>
      {/* Topbar */}
      <header style={{ position: 'sticky', top: 0, zIndex: 10, background: 'transparent' }}>
        <div style={{ width: '100%', padding: '10px var(--page-gutter)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="logo-lockup">
            <img src="/assets/AIDx-logo.png" alt="AIDx" style={{ height: 28, width: 'auto' }} />
            <h1>Medical Agent</h1>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="subtle" style={{ fontSize: '.9rem' }}>{user ? String(user) : ''}</span>
            <button className="btn" onClick={handleLogout}>Logout</button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ width: '100%', padding: '0 24px 8px', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <TabButton label="Patient Registry" active={activeTab === 'patient'} onClick={() => setActiveTab('patient')} />
          <TabButton label="Chest X-Ray Analysis" active={activeTab === 'cxr'} onClick={() => setActiveTab('cxr')} />
          <TabButton label="Brain Tumour Detection" active={activeTab === 'brain'} onClick={() => setActiveTab('brain')} />
        </div>
      </header>

      {/* Main content */}
      <main style={{ width: '100%', padding: '12px var(--page-gutter) 96px' }}>
        {activeTab === 'patient' && <PatientRegistry headers={headers} />}
        {activeTab === 'cxr' && <CXRPage headers={headers} />}
        {activeTab === 'brain' && <BrainPage headers={headers} />}

        {!!navErr && <div className="subtle" style={{ color: '#b42318', marginTop: 8 }}>Search failed: {navErr}</div>}
        {!!navResults?.length && (
          <div className="card" style={{ marginTop: 12, padding: 12, width: '100%', maxWidth: 'none', minWidth: 0 }}>
            <h3 style={{ margin: 0, fontSize: '1rem' }}>Search results</h3>
            <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
              {navResults.map((hit, i) => <SearchCard key={i} hit={hit} index={i} />)}
            </div>
            <button
              className="btn"
              style={{ marginTop: 10, background: 'white', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }}
              onClick={() => { setNavResults([]); setNavQ(''); setNavErr(''); }}
            >
              Clear results
            </button>
          </div>
        )}
      </main>

      <FloatingChatbot
        apiUrl={API_URL}
        headers={headers}
        assistantName="Dr. Aster"
        subtitle="Clinical AI Assistant"
      />
    </div>
  );
}

/* -------------------- */
/* Tabs / Small pieces  */
/* -------------------- */
function TabButton({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className="btn"
      style={{
        background: active ? 'var(--brand-primary)' : '#fff',
        color: active ? '#fff' : 'var(--brand-ink)',
        borderColor: active ? 'var(--brand-primary)' : 'var(--card-border)'
      }}
    >
      {label}
    </button>
  );
}

/* -------------------- */
/* Patient Registry     */
/* -------------------- */
function PatientRegistry({ headers }) {
  // list state
  const [q, setQ] = useState('');
  const [limit, setLimit] = useState(20);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [message, setMessage] = useState('');

  // modal state
  const [showForm, setShowForm] = useState(false);

  const fetchList = async () => {
    try {
      const r = await axios.get(`${API_URL}/patients`, {
        params: { q, limit, offset: 0 },
        headers, timeout: 30000
      });
      const data = r.data || {};
      setItems(data.items || []);
      setTotal(data.total ?? (data.items || []).length);
    } catch (ex) {
      setMessage(`List failed: ${ex.message}`);
    }
  };

  useEffect(() => { fetchList(); /* initial */ }, []);
  const refreshAndToast = async () => { await fetchList(); setMessage('Patient list updated.'); };

  const onDelete = async (pid) => {
    try {
      await axios.delete(`${API_URL}/patients/${pid}`, { headers, timeout: 20000 });
      await refreshAndToast();
    } catch (ex) {
      setMessage(`Delete failed: ${ex.message}`);
    }
  };

  return (
    <section className="space-y-6" style={{ position: 'relative' }}>
      {/* Search + List */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1 }}>
            <TextField label="Search" placeholder="name / MRN / phone / notes" value={q} onChange={setQ} />
          </div>
          <div>
            <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>Page size</label>
            <input
              type="number" min={5} max={100} step={5} value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value || 20))}
              className="input" style={{ width: 120 }}
            />
          </div>
          <button
            className="btn"
            style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }}
            onClick={fetchList}
          >
            Refresh
          </button>
        </div>

        <div style={{ marginTop: 12 }}>
          {items?.length ? (
            <>
              <div className="subtle" style={{ fontWeight: 600 }}>Results: {total}</div>
              <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
                {items.map((p, i) => (
                  <PatientRow
                    key={p.patient_id || i}
                    p={p}
                    headers={headers}
                    onDelete={() => onDelete(p.patient_id)}
                  />
                ))}
              </div>
            </>
          ) : (
            <div className="subtle">No patients found.</div>
          )}
        </div>

        {!!message && <div className="subtle" style={{ marginTop: 8 }}>{message}</div>}
      </div>

      {/* Floating "Register Patient" button */}
      {!showForm && (
        <button
          className="btn"
          onClick={() => setShowForm(true)}
          title="Register Patient"
          style={{
            position: 'fixed',
            right: 'var(--page-gutter)',
            bottom: 24 + 56, // keep above chatbot
            zIndex: 35,
            padding: '12px 16px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8
          }}
        >
          ➕ Register Patient
        </button>
      )}

      {/* Modal: Patient Registration */}
      {showForm && (
        <PatientFormModal
          headers={headers}
          onClose={() => setShowForm(false)}
          onCreated={async () => {
            await refreshAndToast();
            setShowForm(false); // hide and reveal the button again
          }}
        />
      )}
    </section>
  );
}

function PatientRow({ p, headers, onDelete }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="card" style={{ padding: 12 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 2fr 1fr 1fr 1fr auto', gap: 12, alignItems: 'start' }}>
        <div>
          <div style={{ fontWeight: 600 }}>{`${p.first || ''} ${p.last || ''}`}</div>
          <div className="subtle" style={{ fontSize: '.85rem' }}>MRN: {p.mrn || '—'}</div>
          <div className="subtle" style={{ fontSize: '.85rem' }}>ID: {p.patient_id || '—'}</div>
        </div>
        <div style={{ whiteSpace: 'pre-line' }}>{p.notes || ''}</div>
        <div>
          <div className="subtle" style={{ fontSize: '.85rem' }}>Sex / DOB</div>
          <div>{(p.sex || '—') + ' / ' + (p.dob || '—')}</div>
        </div>
        <div>
          <div className="subtle" style={{ fontSize: '.85rem' }}>Phone</div>
          <div>{p.phone || '—'}</div>
        </div>
        <div>
          <div className="subtle" style={{ fontSize: '.85rem' }}>Email</div>
          <div>{p.email || '—'}</div>
        </div>
        <div style={{ textAlign: 'right', display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            className="btn"
            style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }}
            onClick={() => setOpen(o => !o)}
            title={open ? 'Hide records & reports' : 'Show records & reports'}
          >
            {open ? 'Hide' : 'View'}
          </button>
          <button
            onClick={onDelete}
            className="btn"
            style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }}
            title="Delete"
          >
            🗑️
          </button>
        </div>
      </div>

      {open && (
        <PatientDetails patientId={p.patient_id} headers={headers} />
      )}
    </div>
  );
}

function PatientDetails({ patientId, headers }) {
  const [records, setRecords] = useState({ items: [] });
  const [reports, setReports] = useState({ items: [] });
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setErr('');
      try {
        const [r1, r2] = await Promise.all([
          axios.get(`${API_URL}/patients/${patientId}/records`, { headers, timeout: 30000 }),
          axios.get(`${API_URL}/patients/${patientId}/reports`, { headers, timeout: 30000 })
        ]);
        if (!alive) return;
        setRecords(r1.data || { items: [] });
        setReports(r2.data || { items: [] });
      } catch (ex) {
        if (alive) setErr(ex.message || 'Fetch failed');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [patientId, headers]);

  if (loading) {
    return <div className="subtle" style={{ marginTop: 10 }}>Loading records…</div>;
  }
  if (err) {
    return <div className="subtle" style={{ marginTop: 10, color: '#b42318' }}>Failed to load: {err}</div>;
  }

  return (
    <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>
      {/* Records grid */}
      <div className="card" style={{ padding: 12 }}>
        <h4 style={{ margin: 0 }}>Patient Records</h4>
        {records.items?.length ? (
          <div
            style={{
              marginTop: 8,
              display: 'grid',
              gap: 10,
              gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))'
            }}
          >
            {records.items.map((rec, i) => (
              <RecordCard key={rec.study_id || i} rec={rec} />
            ))}
          </div>
        ) : (
          <div className="subtle" style={{ marginTop: 6 }}>No records yet.</div>
        )}
      </div>

      {/* Reports list */}
      <div className="card" style={{ padding: 12 }}>
        <h4 style={{ margin: 0 }}>Reports</h4>
        {reports.items?.length ? (
          <ul style={{ margin: '8px 0 0 0', padding: 0, listStyle: 'none', display: 'grid', gap: 8 }}>
            {reports.items.map((r, i) => (
              <ReportRow key={r.report_id || i} row={r} />
            ))}
          </ul>
        ) : (
          <div className="subtle" style={{ marginTop: 6 }}>No reports yet.</div>
        )}
      </div>
    </div>
  );
}

function RecordCard({ rec }) {
  const img = absOrStatic(rec.image_url);
  const overlay = absOrStatic(rec.overlay_url);
  const when = rec.created_at ? new Date(rec.created_at * 1000).toLocaleString() : '—';
  return (
    <div className="card" style={{ padding: 10 }}>
      <div className="subtle" style={{ fontSize: '.85rem' }}>{rec.modality || '—'} • {when}</div>
      <div style={{ marginTop: 8, display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
        <figure className="card" style={{ padding: 6 }}>
          {img ? <img src={img} style={{ width: '100%', borderRadius: 8 }} /> : <div className="subtle">No input</div>}
          <figcaption className="subtle" style={{ fontSize: '.75rem', marginTop: 4 }}>Input</figcaption>
        </figure>
        <figure className="card" style={{ padding: 6 }}>
          {overlay ? <img src={overlay} style={{ width: '100%', borderRadius: 8 }} /> : <div className="subtle">No overlay</div>}
          <figcaption className="subtle" style={{ fontSize: '.75rem', marginTop: 4 }}>Overlay</figcaption>
        </figure>
      </div>
    </div>
  );
}

function ReportRow({ row }) {
  const when = row.created_at ? new Date(row.created_at * 1000).toLocaleString() : '—';
  const pdf = absOrStatic(row.pdf_url);
  return (
    <li className="card" style={{ padding: 10, display: 'grid', gridTemplateColumns: '1fr auto', alignItems: 'center', gap: 8 }}>
      <div>
        <div className="subtle" style={{ fontSize: '.85rem' }}>{when}</div>
        {Array.isArray(row.impression) && row.impression.length > 0 ? (
          <ul style={{ margin: '6px 0 0 18px' }}>
            {row.impression.slice(0, 3).map((it, idx) => <li key={idx}>{it}</li>)}
          </ul>
        ) : (
          <div className="subtle" style={{ marginTop: 4 }}>No impression summary.</div>
        )}
      </div>
      {pdf ? (
        <a
          className="btn"
          href={pdf}
          target="_blank"
          rel="noreferrer"
          style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }}
        >
          Open PDF
        </a>
      ) : (
        <span className="subtle">No PDF</span>
      )}
    </li>
  );
}

function PatientFormModal({ headers, onClose, onCreated }) {
  const [form, setForm] = useState({
    mrn: '', first: '', last: '', sex: '', dob: '', phone: '', email: '', address: '', notes: ''
  });
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState('');

  // ESC to close
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const onSubmit = async (e) => {
    e.preventDefault();
    setErr('');
    if (!form.first.trim() || !form.last.trim()) {
      setErr('First and Last name are required.');
      return;
    }
    try {
      setSaving(true);
      await axios.post(`${API_URL}/patients`, { ...form }, { headers, timeout: 30000 });
      await onCreated(); // refresh list + close
    } catch (ex) {
      setErr(ex?.response?.data?.detail || ex.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        background: 'rgba(15, 23, 42, .35)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16
      }}
      onClick={onClose}
    >
      <div
        className="card"
        style={{ width: 'min(920px, 96vw)', padding: 16, maxHeight: '90vh', overflow: 'auto' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
          <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Register Patient</h2>
          <button className="btn" style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }} onClick={onClose}>Close</button>
        </div>

        <form onSubmit={onSubmit} style={{ marginTop: 12, display: 'grid', gap: 12, gridTemplateColumns: '1fr 1fr' }}>
          <TextField label="MRN / Patient ID" value={form.mrn} onChange={(v) => setForm(s => ({ ...s, mrn: v }))} />
          <TextField label="First Name *" value={form.first} onChange={(v) => setForm(s => ({ ...s, first: v }))} />
          <TextField label="Last Name *" value={form.last} onChange={(v) => setForm(s => ({ ...s, last: v }))} />
          <SelectField label="Sex" options={['', 'M', 'F', 'Other']} value={form.sex} onChange={(v) => setForm(s => ({ ...s, sex: v }))} />
          <TextField label="Date of Birth (YYYY-MM-DD)" value={form.dob} onChange={(v) => setForm(s => ({ ...s, dob: v }))} placeholder="YYYY-MM-DD" />
          <TextField label="Phone" value={form.phone} onChange={(v) => setForm(s => ({ ...s, phone: v }))} />
          <TextField label="Email" value={form.email} onChange={(v) => setForm(s => ({ ...s, email: v }))} />
          <TextArea label="Address" className="md:col-span-2" value={form.address} onChange={(v) => setForm(s => ({ ...s, address: v }))} />
          <TextArea label="Clinical Notes / Indication" className="md:col-span-2" value={form.notes} onChange={(v) => setForm(s => ({ ...s, notes: v }))} />
          <div style={{ gridColumn: '1 / -1', display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button type="button" className="btn" style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }} onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn" disabled={saving}>
              {saving ? 'Saving…' : 'Create Patient'}
            </button>
          </div>
        </form>

        {!!err && <div className="subtle" style={{ marginTop: 8, color: '#b42318' }}>{err}</div>}
      </div>
    </div>
  );
}

/* -------------------- */
/* CXR Page             */
/* -------------------- */
function CXRPage({ headers }) {
  const [file, setFile] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState(null);
  const [preview, setPreview] = useState(null);
  const inputImgBoxRef = useRef(null);  

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(f);
  };

  const analyze = async () => {
    if (!patientId.trim()) return alert('Patient ID (MRN) is required.');
    if (!file) return alert('Please upload a chest X-ray image.');
    try {
      await axios.get(`${API_URL}/patients/${patientId.trim()}`, { headers, timeout: 20000 });
    } catch {
      return alert('Invalid Patient ID. Please register the patient or check the MRN.');
    }

    const fd = new FormData();
    fd.append('file', file, file.name);
    fd.append('threshold', String(0.1));
    fd.append('topk', String(20));
    fd.append('patient_id', patientId.trim());

    setBusy(true);
    try {
      const r = await axios.post(`${API_URL}/analyze`, fd, { headers, timeout: 180000 });
      setResp(r.data || {});
    } catch (ex) {
      alert(`Request failed: ${ex.message}`);
    } finally {
      setBusy(false);
    }
  };

  const arts = resp?.artifacts || {};
  const inUrl = absOrStatic(arts?.input_image);
  let outUrl = absOrStatic(arts?.output_image);
  const cams = Array.isArray(arts?.cams) ? arts.cams : [];
  if (!outUrl && cams?.length) outUrl = absOrStatic(cams[0]?.url || cams[0]?.path);
  if (outUrl && inUrl && outUrl === inUrl) {
    for (const cam of cams) {
      const cand = absOrStatic(cam?.url || cam?.path);
      if (cand && cand !== inUrl) { outUrl = cand; break; }
    }
  }

  const pinfo = resp?.patient || {};
  const reportBlock = resp?.report || {};
  const report = reportBlock?.report || {};
  const pdfVal = arts?.pdf;

  return (
    <section className="space-y-4" style={{ width: '100%' }}>
      {/* Top card */}
      <div className="card" style={{ padding: 16, width: '100%', maxWidth: '100%' }}>
        <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Chest X-Ray Analysis</h2>
        <div
          style={{
            marginTop: 12,
            display: 'grid',
            gap: 12,
            gridTemplateColumns: 'minmax(0, 4fr)',
            alignItems: 'stretch'
          }}
        >
          <div className="card" style={{ padding: 12, width: '100%', maxWidth: '100%', height: '100%' }}>
            <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>
              Upload Chest X-ray (.png/.jpg/.dcm)
            </label>
            <input type="file" accept=".png,.jpg,.jpeg,.dcm" onChange={onFile} />
            <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>
              <PatientPicker
                headers={headers}
                onSelect={(p) => {
                  // store the true patient_id used by the backend
                  setPatientId(p.patient_id);
                }}
              />
              {/* Show what will be sent to backend for clarity */}
              <div className="subtle" style={{ fontSize: '.85rem' }}>
                Selected Patient ID to use: <b>{patientId || '—'}</b>
              </div>
              <button onClick={analyze} disabled={busy} className="btn" style={{ width: '100%', marginTop: 4 }}>
                {busy ? 'Analyzing…' : 'Analyze'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Patient summary */}
      {(pinfo && (Object.keys(pinfo).length > 0)) && (
        <div className="card" style={{ padding: 16, width: '100%', maxWidth: '100%' }}>
          <h3 style={{ margin: 0, fontSize: '1.05rem' }}>Patient</h3>
          <div style={{ marginTop: 8, display: 'grid', gap: 12, gridTemplateColumns: 'repeat(3, 1fr)' }}>
            <InfoKV k="MRN" v={pinfo.mrn || '—'} />
            <InfoKV k="Sex" v={pinfo.sex || '—'} />
            <InfoKV k="DOB" v={pinfo.dob || '—'} />
            <InfoKV k="Patient ID" v={pinfo.patient_id || '—'} />
            <InfoKV k="Name" v={`${pinfo.first || ''} ${pinfo.last || ''}`.trim() || '—'} />
            <InfoKV k="Phone" v={pinfo.phone || '—'} />
            <InfoKV k="Email" v={pinfo.email || '—'} />
          </div>
        </div>
      )}

      {/* Images */}
      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: '1fr 1fr', width: '100%' }}>
        <div ref={inputImgBoxRef} className="card" style={{ padding: 16, width: '100%', maxWidth: '100%' }}>
          <h3 style={{ margin: 0, fontSize: '1.05rem' }}>Input X-ray</h3>
          {inUrl ? (
            <img src={inUrl} style={{ marginTop: 8, borderRadius: 12, border: '1px solid var(--card-border)', width: '100%' }} />
          ) : preview ? (
            <img src={preview} style={{ marginTop: 8, borderRadius: 12, border: '1px solid var(--card-border)', width: '100%' }} />
          ) : (
            <div className="subtle" style={{ marginTop: 8, minHeight: 320, display: 'flex', alignItems: 'center' }}>
              Upload a chest X-ray above to preview.
            </div>
          )}
        </div>

        <ImageCarousel items={cams} matchRef={inputImgBoxRef} title="AI Findings" />
          {/* <h3 style={{ margin: 0, fontSize: '1.05rem' }}>AI Findings</h3>
          {cams?.length ? (
            <div
              style={{
                marginTop: 8,
                display: 'grid',
                gap: 8,
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))'
              }}
            >
              {cams.slice(0, 6).map((cam, i) => {
                const url = absOrStatic(cam?.url || cam?.path);
                const score = typeof cam?.score === 'number' ? cam.score.toFixed(3) : null;
                const cap = `${cam?.rank}. ${cam?.label}${score ? ` (score=${score})` : ''}`;
                return (
                  <figure key={i} className="card" style={{ padding: 8 }}>
                    {url && <img src={url} style={{ borderRadius: 8, width: '100%' }} />}
                    <figcaption className="subtle" style={{ fontSize: '.8rem', marginTop: 4 }}>{cap}</figcaption>
                  </figure>
                );
              })} 
            </div>
          ) : (
            <div className="subtle" style={{ marginTop: 8, minHeight: 320, display: 'flex', alignItems: 'center' }}>
              No overlays for this analysis.
            </div>
          )} */}
        </div>

      {/* Report */}
      <div className="card" style={{ padding: 16, width: '100%', maxWidth: '100%' }}>
        <h3 style={{ margin: 0, fontSize: '1.05rem' }}>Report</h3>
        {Object.keys(report || {}).length ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
            {report.indication && <div><b>Indication:</b> {report.indication}</div>}
            {report.technique && <div><b>Technique:</b> {report.technique}</div>}
            {report.comparison && <div><b>Comparison:</b> {report.comparison}</div>}
            {report.findings && (
              <div>
                <b>Findings:</b>
                <div style={{ whiteSpace: 'pre-wrap' }}>{report.findings}</div>
              </div>
            )}
            {Array.isArray(report.impression) && report.impression.length > 0 && (
              <div>
                <b>Impression:</b>
                <ul style={{ margin: '6px 0 0 18px' }}>
                  {report.impression.map((it, idx) => <li key={idx}>{it}</li>)}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="subtle" style={{ marginTop: 8, minHeight: 160, display: 'flex', alignItems: 'center' }}>
            No report yet. Upload an image and click <b>Analyze</b>.
          </div>
        )}

        {pdfVal && <PDFDownloader src={absOrStatic(pdfVal)} headers={headers} />}
      </div>
    </section>
  );
}


/* -------------------- */
/* Brain Page           */
/* -------------------- */
function BrainPage({ headers }) {
  const [file, setFile] = useState(null);
  const [thr, setThr] = useState(0.35);
  const [img, setImg] = useState(null);
  const [raw, setRaw] = useState(null);
  const [msg, setMsg] = useState('');

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
  };

  const run = async () => {
    setMsg('');
    if (!file) { setMsg('Please select a file'); return; }
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      fd.append('threshold', String(thr));
      const r = await axios.post(`${API_URL}/brain/infer`, fd, { headers, timeout: 180000 });
      const out = r.data || {};
      setRaw(out?.detections || out);
      const overlay = out?.artifact?.overlay;
      if (overlay) setImg(absOrStatic(overlay));
    } catch (ex) {
      setMsg('Brain inference backend not configured yet. This is a placeholder UI.');
    }
  };

  return (
    // ↓ full-width section (no theme max-width)
    <section style={{ width: '100%', maxWidth: '100%' }}>
      <div className="card" style={{ padding: 16, width: '100%', maxWidth: '100%' }}>
        <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Brain Tumour Detection</h2>
        <div
          style={{
            marginTop: 12,
            display: 'grid',
            gap: 12,
            gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)',
            alignItems: 'stretch'
          }}
        >
          <div className="card" style={{ padding: 12, width: '100%', maxWidth: '100%' }}>
            <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>
              Upload Brain MRI (.png/.jpg/.nii/.nii.gz)
            </label>
            <input type="file" accept=".png,.jpg,.jpeg" onChange={onFile} />
            <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <label className="subtle" style={{ fontSize: '.9rem' }}>Detection threshold</label>
              <input type="number" step="0.01" min={0} max={1} value={thr}
                onChange={(e) => setThr(parseFloat(e.target.value || 0))}
                className="input" style={{ width: 100 }} />
              <button onClick={run} className="btn" style={{ marginLeft: 'auto' }}>Run Detection</button>
            </div>
            {!!msg && <div className="subtle" style={{ marginTop: 8 }}>{msg}</div>}
          </div>
          <div className="card" style={{ padding: 12, width: '100%', maxWidth: '100%' }}>
            <div className="subtle" style={{ fontSize: '.9rem' }}>Detections</div>
            {img ? (
              <img src={img} style={{ marginTop: 8, borderRadius: 8, border: '1px solid var(--card-border)', width: '100%' }} />
            ) : (
              <div className="subtle" style={{ marginTop: 8, minHeight: 280, display: 'flex', alignItems: 'center' }}>
                No overlay yet.
              </div>
            )}
          </div>
        </div>
        <div style={{ marginTop: 12 }}>
          <div className="subtle" style={{ fontSize: '.9rem' }}>Raw output</div>
          <pre className="card" style={{ padding: 12, fontSize: '.8rem', overflow: 'auto', maxHeight: 320, width: '100%', maxWidth: '100%', minHeight: 160 }}>
            {JSON.stringify(raw, null, 2)}
          </pre>
        </div>
      </div>
      <p className="subtle" style={{ fontSize: '.85rem' }}>
        Note: wire this page to your <code>/brain/infer</code> endpoint when available.
      </p>
    </section>
  );
}


/* -------------------- */
/* Footer Chat (unused inline version kept for reference) */
/* -------------------- */
function FooterChat() { return null; }

function Msg({ role, text }) {
  const isUser = role === 'user';
  return (
    <div style={{ display: 'flex', gap: 8, margin: '6px 0', justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      {!isUser && <div style={avatarStyle}>AI</div>}
      <div style={{
        maxWidth: '72%', padding: '8px 12px', borderRadius: 16, boxShadow: '0 2px 8px rgba(67,87,102,.12)',
        background: isUser ? '#E8EEF3' : '#F5F7F8', color: isUser ? '#0F172A' : '#2D3B44',
        borderTopRightRadius: isUser ? 6 : 16, borderTopLeftRadius: isUser ? 16 : 6
      }}>
        <RichText text={text} />
      </div>
      {isUser && <div style={avatarStyle}>U</div>}
    </div>
  );
}

const avatarStyle = {
  height: 26, width: 26, borderRadius: '50%', background: 'var(--brand-primary)',
  color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12
};

function RichText({ text }) {
  const html = String(text || '')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\n/g, '<br/>');
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

/* -------------------- */
/* Small Inputs         */
/* -------------------- */
function TextField({ label, value, onChange, placeholder, help, className, style }) {
  return (
    <div className={className} style={style}>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>{label}</label>
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="input" />
      {help && <div className="subtle" style={{ fontSize: '.8rem', marginTop: 4 }}>{help}</div>}
    </div>
  );
}

function TextArea({ label, value, onChange, className, style }) {
  return (
    <div className={className} style={style}>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>{label}</label>
      <textarea value={value} onChange={(e) => onChange(e.target.value)} className="textarea" style={{ height: 96 }} />
    </div>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <div>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="select">
        {options.map((o, i) => <option key={i} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function InfoKV({ k, v }) {
  return (
    <div>
      <div className="subtle" style={{ fontSize: '.85rem' }}>{k}</div>
      <div style={{ fontSize: '.95rem' }}>{String(v)}</div>
    </div>
  );
}

function PDFDownloader({ src, headers }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [err, setErr] = useState('');
  const mounted = useRef(true);

  useEffect(() => () => { mounted.current = false; }, []);
  useEffect(() => {
    if (!src) return;
    if (!String(src).startsWith('http')) { setBlobUrl(src); return; }
    (async () => {
      try {
        const r = await axios.get(src, { responseType: 'blob', headers, timeout: 60000 });
        const url = URL.createObjectURL(r.data);
        if (mounted.current) setBlobUrl(url);
      } catch (ex) {
        setErr(`Could not load PDF: ${ex.message}`);
      }
    })();
  }, [src]);

  if (err) return <div style={{ color: '#b42318', fontSize: '.9rem' }}>{err}</div>;
  if (!blobUrl) return null;
  return (
    <a href={blobUrl} download="report.pdf" className="btn" style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)', marginTop: 8 }}>
      ⬇️ Download PDF
    </a>
  );
}

function PatientPicker({ headers, onSelect }) {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [options, setOptions] = useState([]);
  const [error, setError] = useState('');

  // tiny debounce
  useEffect(() => {
    const t = setTimeout(async () => {
      if (!q.trim()) { setOptions([]); setOpen(false); return; }
      try {
        setLoading(true);
        setError('');
        const r = await axios.get(`${API_URL}/patients`, {
          params: { q, limit: 10, offset: 0 },
          headers, timeout: 20000
        });
        const items = r.data?.items || [];
        setOptions(items);
        setOpen(items.length > 0);
      } catch (e) {
        setError('Search failed');
        setOptions([]);
        setOpen(false);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [q]);

  const pick = (p) => {
    setQ(`${p.first || ''} ${p.last || ''}`.trim() || (p.mrn || p.patient_id));
    setOpen(false);
    onSelect?.(p); // hand back the whole patient (contains patient_id & mrn)
  };

  // also allow direct paste of MRN or patient_id:
  const onBlurTryResolve = async () => {
    if (!q.trim() || open) return;
    try {
      // try treat as patient_id
      const r = await axios.get(`${API_URL}/patients/${q.trim()}`, { headers, timeout: 10000 });
      if (r.data?.patient_id) return pick(r.data);
    } catch(_) { /* ignore */ }

    try {
      const s = await axios.get(`${API_URL}/patients`, {
        params: { q, limit: 10, offset: 0 }, headers, timeout: 10000
      });
      const items = s.data?.items || [];
      // prefer exact MRN match if present
      const exactMRN = items.find(p => (p.mrn || '').toLowerCase() === q.trim().toLowerCase());
      if (exactMRN) return pick(exactMRN);
      if (items.length === 1) return pick(items[0]);
    } catch(_) { /* ignore */ }
  };

  return (
    <div style={{ position: 'relative' }}>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>
        Patient (search by Name or MRN)
      </label>
      <input
        className="input"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onFocus={() => setOpen(options.length > 0)}
        onBlur={onBlurTryResolve}
        placeholder="Type name or MRN…"
        autoComplete="off"
      />
      {loading && <div className="subtle" style={{ marginTop: 4, fontSize: '.85rem' }}>Searching…</div>}
      {error && <div className="subtle" style={{ marginTop: 4, color: '#b42318' }}>{error}</div>}
      {open && (
        <div
          className="card"
          style={{
            position: 'absolute', left: 0, right: 0, top: 'calc(100% + 6px)', zIndex: 40,
            padding: 6, maxHeight: 260, overflowY: 'auto'
          }}
          onMouseDown={(e) => e.preventDefault()} /* keep focus */
        >
          {options.map((p) => (
            <button
              key={p.patient_id}
              className="btn"
              onClick={() => pick(p)}
              style={{
                width: '100%', justifyContent: 'flex-start', gap: 10,
                background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)',
                margin: '4px 0'
              }}
            >
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600 }}>
                  {(p.first || '') + ' ' + (p.last || '')}
                </div>
                <div className="subtle" style={{ fontSize: '.85rem' }}>
                  MRN: {p.mrn || '—'} &nbsp;•&nbsp; Sex/DOB: {(p.sex || '—') + ' / ' + (p.dob || '—')}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


/* -------------------- */
/* Utils                */
/* -------------------- */
function absOrStatic(path) {
  if (!path || typeof path !== 'string') return null;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return `${API_URL.replace(/\/$/, '')}${path}`;
  if (path.includes('artifacts/')) {
    const rel = path.replace(/\\/g, '/').split('artifacts/', 2)[1];
    return `${API_URL.replace(/\/$/, '')}/static/${rel}`;
  }
  return path;
}


/* Simple Image Carousel */
/* -------------------- */
import { useLayoutEffect } from 'react';

function ImageCarousel({ items = [], matchRef, title = 'AI Findings' }) {
  const [idx, setIdx] = useState(0);
  const [w, setW] = useState(640); // default; will sync to Input X-ray width
  const wrapRef = useRef(null);

  // Keep the carousel width synced to the left image container
  useLayoutEffect(() => {
    if (!matchRef?.current) return;
    const node = matchRef.current;
    const apply = () => setW(Math.max(320, Math.floor(node.clientWidth)));
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(node);
    return () => ro.disconnect();
  }, [matchRef]);

  const has = Array.isArray(items) ? items.length : 0;
  const cur = has ? items[Math.max(0, Math.min(idx, has - 1))] : null;

  const go = (d) => setIdx((p) => {
    const n = (p + d + has) % has;
    return n;
  });

  const imgUrl = cur ? (cur.url || cur.path) : null;
  const caption = cur
    ? `${cur.rank}. ${cur.label}${typeof cur.score === 'number' ? ` (score=${cur.score.toFixed(3)})` : ''}`
    : '';

  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ padding: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0, fontSize: '1.05rem' }}>{title}</h3>
        <span className="subtle" style={{ fontWeight: 600 }}>{has ? `${idx + 1}/${has}` : '0/0'}</span>
      </div>

      {/* Image stage */}
      <div
        ref={wrapRef}
        style={{
          position: 'relative',
          width: '100%',
          overflow: 'hidden',
          display: 'flex',
          justifyContent: 'center',
          padding: 12
        }}
      >
        <div
          style={{
            width: w,           // match input image width
            maxWidth: '100%',
            borderRadius: 12,
            border: '1px solid var(--card-border)',
            overflow: 'hidden',
            position: 'relative',
            background: '#fff'
          }}
        >
          {imgUrl ? (
            <img
              src={absOrStatic(imgUrl)}
              alt={caption}
              style={{ display: 'block', width: '100%', height: 'auto' }}
            />
          ) : (
            <div className="subtle" style={{ padding: 24, textAlign: 'center' }}>No overlays for this analysis.</div>
          )}

          {/* Click zones */}
          {has > 1 && (
            <>
              <button
                className="carousel-zone"
                aria-label="Previous"
                onClick={() => go(-1)}
                style={{
                  position: 'absolute', inset: '0 auto 0 0', width: '24%',
                  background: 'transparent', border: 'none', cursor: 'pointer'
                }}
              />
              <button
                className="carousel-zone"
                aria-label="Next"
                onClick={() => go(+1)}
                style={{
                  position: 'absolute', inset: '0 0 0 auto', width: '24%',
                  background: 'transparent', border: 'none', cursor: 'pointer'
                }}
              />
              {/* optional chevrons */}
              <div style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}>◀︎</div>
              <div style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}>▶︎</div>
            </>
          )}
        </div>
      </div>

      {/* Caption */}
      {caption && (
        <div className="subtle" style={{ padding: '0 16px 16px' }}>
          {caption}
        </div>
      )}
    </div>
  );
}