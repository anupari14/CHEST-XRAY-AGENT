// src/components/PatientRegistry.jsx
import { useEffect, useState } from 'react';
import axios from 'axios';
import { API_URL } from '../lib/client.js'
import { TextField, TextArea, SelectField } from './common/ui.jsx';
import { absOrStatic } from '../lib/paths.js';

export default function PatientRegistry({ headers }) {
  const [q, setQ] = useState('');
  const [limit, setLimit] = useState(20);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [message, setMessage] = useState('');
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

  useEffect(() => { fetchList(); }, []); // initial
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

      {!showForm && (
        <button
          className="btn"
          onClick={() => setShowForm(true)}
          title="Register Patient"
          style={{
            position: 'fixed',
            right: 'var(--page-gutter)',
            bottom: 24 + 56,
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

      {showForm && (
        <PatientFormModal
          headers={headers}
          onClose={() => setShowForm(false)}
          onCreated={async () => {
            await refreshAndToast();
            setShowForm(false);
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

      {open && <PatientDetails patientId={p.patient_id} headers={headers} />}
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

  if (loading) return <div className="subtle" style={{ marginTop: 10 }}>Loading records…</div>;
  if (err) return <div className="subtle" style={{ marginTop: 10, color: '#b42318' }}>Failed to load: {err}</div>;
  // ---- Pair each record with its report
  const repByEnc = new Map(
    (reports.items || []).map(r => [r.encounter_id || r.study_id || r.record_id || '', r])
    );
  const byTime = [...(reports.items || [])].sort((a,b)=>(a.created_at||0)-(b.created_at||0));
  const findReportFor = (rec) => {
  const key = rec.encounter_id || rec.study_id || rec.record_id || '';
  if (key && repByEnc.has(key)) return repByEnc.get(key);
    // fallback: nearest by time (±30 min window), else undefined
  const t = rec.created_at || 0;
  let best = undefined, bestDiff = Infinity;
  for (const r of byTime) {
    const d = Math.abs((r.created_at||0) - t);
    if (d < bestDiff) { best = r; bestDiff = d; }
  }
  return bestDiff <= 1800 ? best : undefined; // 1800s = 30 min
  };
  const merged = (records.items || []).map(rec => ({ rec, rep: findReportFor(rec) }));

  return (
    <div className="card" style={{ padding: 12 }}>
       <h4 style={{ margin: 0 }}>Patient Records</h4>
          {merged.length ? (
           <div
            style={{
            marginTop: 8,
            display: 'grid',
            gap: 12,
            gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
            alignItems: 'start'
          }}
        >
          {merged.map(({ rec, rep }, i) => (
            <RecordWithReportCard key={rec.study_id || rec.encounter_id || i} rec={rec} rep={rep} />
           ))}
        </div>
      ) : (
        <div className="subtle" style={{ marginTop: 6 }}>No records yet.</div>
      )}
    </div>
  );
}

function RecordCard({ rec }) {
  const img = absOrStatic(rec?.image_url);
  const when = rec.created_at ? new Date(rec.created_at * 1000).toLocaleString() : '—';
  return (
    <div className="card" style={{ padding: 12 }}>
      <div className="subtle" style={{ fontSize: '.85rem' }}>{rec.modality || 'CXR'} • {when}</div>
      <div style={{ marginTop: 8, border: '1px solid var(--card-border)',borderRadius: 12,overflow: 'hidden',width: '100%',aspectRatio: '1 / 1',background: '#fff' }}>
        {img ? ( <img src={img} alt="Input X-Ray" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} /> ) : <div className="subtle">No input</div>}        
      </div>
    </div>
  );
}

function ReportRow({ row }) {
  const when = row.created_at ? new Date(row.created_at * 1000).toLocaleString() : '—';
  const pdf = row.pdf_url;
  return (
    <li className="card" style={{ padding: 10, display: 'grid', gridTemplateColumns: '1fr auto', alignItems: 'center', gap: 8 }}>
      <div>
        <div className="subtle" style={{ fontSize: '.85rem' }}>{when}</div>
        {Array.isArray(row.impression) && row.impression.length > 0 ? (
          <ul style={{ margin: '6px 0 0 18px' }}>
            {row.impression.slice(0, 3).map((it, idx) => <li key={idx}>{it}</li>)}
          </ul>
        ) : <div className="subtle" style={{ marginTop: 4 }}>No impression summary.</div>}
      </div>
      {pdf ? (
        <a className="btn" href={pdf} target="_blank" rel="noreferrer"
           style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }}>
          Open PDF
        </a>
      ) : <span className="subtle">No PDF</span>}
    </li>
  );
}

function PatientFormModal({ headers, onClose, onCreated }) {
  const [form, setForm] = useState({
    mrn: '', first: '', last: '', sex: '', dob: '', phone: '', email: '', address: '', notes: ''
  });
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState('');

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
      await onCreated();
    } catch (ex) {
      setErr(ex?.response?.data?.detail || ex.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div role="dialog" aria-modal="true"
      style={{ position: 'fixed', inset: 0, zIndex: 50, background: 'rgba(15, 23, 42, .35)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}
      onClick={onClose}
    >
      <div className="card" style={{ width: 'min(920px, 96vw)', padding: 16, maxHeight: '90vh', overflow: 'auto' }}
           onClick={(e) => e.stopPropagation()}>
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
            <button type="button" className="btn" style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)' }} onClick={onClose}>Cancel</button>
            <button type="submit" className="btn" disabled={saving}>{saving ? 'Saving…' : 'Create Patient'}</button>
          </div>
        </form>

        {!!err && <div className="subtle" style={{ marginTop: 8, color: '#b42318' }}>{err}</div>}
      </div>
    </div>
  );
}

function RecordWithReportCard({ rec, rep }) {
  const when = rec.created_at ? new Date(rec.created_at * 1000).toLocaleString() : '—';
  const img = absOrStatic(rec?.image_url);          // already absolute from API now
  const pdf = rep?.pdf_url;
  const impressions = Array.isArray(rep?.impression) ? rep.impression : [];
  const summary = impressions.slice(0, 3);

  return (
    <div className="card" style={{ padding: 12 }}>
      {/* Header */}
      <div className="subtle" style={{ fontSize: '.85rem', marginBottom: 8 }}>
        {rec.modality || 'CXR'} • {when}
      </div>

      {/* Image */}
      <div
        style={{
          border: '1px solid var(--card-border)',
          borderRadius: 12,
          overflow: 'hidden',
          width: '100%',
          aspectRatio: '1/1',
          background: '#fff'
        }}
      >
        {img ? (
          <img
            src={img}
            alt="Input X-ray"
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          />
        ) : (
          <div className="subtle" style={{ padding: 16, textAlign: 'center' }}>No input</div>
        )}
      </div>

      {/* Report snippet */}
      <div style={{ marginTop: 10 }}>
        <div style={{ fontWeight: 600, fontSize: '.95rem' }}>Report</div>
        {summary.length ? (
          <ul style={{ margin: '6px 0 0 18px' }}>
            {summary.map((it, idx) => <li key={idx}>{it}</li>)}
          </ul>
        ) : rep?.findings ? (
          <div className="subtle" style={{ marginTop: 4 }}>
            {String(rep.findings).slice(0, 180)}{String(rep.findings).length > 180 ? '…' : ''}
          </div>
        ) : (
          <div className="subtle" style={{ marginTop: 4 }}>No impression summary.</div>
        )}
        {pdf ? (
          <a
            className="btn"
            href={pdf}
            target="_blank"
            rel="noreferrer"
            style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)', marginTop: 8 }}
          >
            Open PDF
          </a>
        ) : null}
      </div>
    </div>
  );
}

