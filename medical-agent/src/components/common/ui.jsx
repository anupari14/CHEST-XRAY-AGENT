// src/components/common/ui.jsx
import { useEffect, useRef, useState, useLayoutEffect } from 'react';
import axios from 'axios';
import { API_URL, absOrStatic } from '../../lib/client.js';

/* ---- Small inputs ---- */
export function TextField({ label, value, onChange, placeholder, help, className, style }) {
  return (
    <div className={className} style={style}>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>{label}</label>
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="input" />
      {help && <div className="subtle" style={{ fontSize: '.8rem', marginTop: 4 }}>{help}</div>}
    </div>
  );
}
export function TextArea({ label, value, onChange, className, style }) {
  return (
    <div className={className} style={style}>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>{label}</label>
      <textarea value={value} onChange={(e) => onChange(e.target.value)} className="textarea" style={{ height: 96 }} />
    </div>
  );
}
export function SelectField({ label, value, onChange, options }) {
  return (
    <div>
      <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="select">
        {options.map((o, i) => <option key={i} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

export function InfoKV({ k, v }) {
  return (
    <div>
      <div className="subtle" style={{ fontSize: '.85rem' }}>{k}</div>
      <div style={{ fontSize: '.95rem' }}>{String(v)}</div>
    </div>
  );
}

export function PDFDownloader({ src, headers }) {
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
  }, [src, headers]);

  if (err) return <div style={{ color: '#b42318', fontSize: '.9rem' }}>{err}</div>;
  if (!blobUrl) return null;
  return (
    <a href={blobUrl} download="report.pdf" className="btn" style={{ background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)', marginTop: 8 }}>
      ⬇️ Download PDF
    </a>
  );
}

/* ---- Patient picker (Name or MRN → emits whole patient object) ---- */
export function PatientPicker({ headers, onSelect }) {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [options, setOptions] = useState([]);
  const [error, setError] = useState('');

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
      } catch {
        setError('Search failed');
        setOptions([]);
        setOpen(false);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [q, headers]);

  const pick = (p) => {
    setQ(`${p.first || ''} ${p.last || ''}`.trim() || (p.mrn || p.patient_id));
    setOpen(false);
    onSelect?.(p);
  };

  const onBlurTryResolve = async () => {
    if (!q.trim() || open) return;
    try {
      const r = await axios.get(`${API_URL}/patients/${q.trim()}`, { headers, timeout: 10000 });
      if (r.data?.patient_id) return pick(r.data);
    } catch {}
    try {
      const s = await axios.get(`${API_URL}/patients`, {
        params: { q, limit: 10, offset: 0 }, headers, timeout: 10000
      });
      const items = s.data?.items || [];
      const exactMRN = items.find(p => (p.mrn || '').toLowerCase() === q.trim().toLowerCase());
      if (exactMRN) return pick(exactMRN);
      if (items.length === 1) return pick(items[0]);
    } catch {}
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
          style={{ position: 'absolute', left: 0, right: 0, top: 'calc(100% + 6px)', zIndex: 40, padding: 6, maxHeight: 260, overflowY: 'auto' }}
          onMouseDown={(e) => e.preventDefault()}
        >
          {options.map((p) => (
            <button
              key={p.patient_id}
              className="btn"
              onClick={() => pick(p)}
              style={{ width: '100%', justifyContent: 'flex-start', gap: 10, background: '#fff', color: 'var(--brand-ink)', borderColor: 'var(--card-border)', margin: '4px 0' }}
            >
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600 }}>{(p.first || '') + ' ' + (p.last || '')}</div>
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

/* ---- Image Carousel (kept as-is) ---- */
export function ImageCarousel({ items = [], matchRef, title = 'AI Findings' }) {
  const [idx, setIdx] = useState(0);
  const [w, setW] = useState(640);
  const has = Array.isArray(items) ? items.length : 0;
  const cur = has ? items[(idx + has) % has] : null;

  useLayoutEffect(() => {
    if (!matchRef?.current) return;
    const node = matchRef.current;
    const apply = () => setW(Math.max(320, Math.floor(node.clientWidth)));
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(node);
    return () => ro.disconnect();
  }, [matchRef]);

  const go = (d) => setIdx(p => (p + d + has) % has);

  const imgUrl = cur ? (cur.url || cur.path) : null;
  const caption = cur
    ? `${cur.rank}. ${cur.label}${typeof cur.score === 'number' ? ` (score=${cur.score.toFixed(3)})` : ''}`
    : '';

  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ padding: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0, fontSize: '1.05rem' }}>{title}</h3>
        <span className="subtle" style={{ fontWeight: 600 }}>{has ? `${(idx % has) + 1}/${has}` : '0/0'}</span>
      </div>
      <div style={{ position: 'relative', width: '100%', overflow: 'hidden', display: 'flex', justifyContent: 'center', padding: 12 }}>
        <div style={{ width: w, maxWidth: '100%', borderRadius: 12, border: '1px solid var(--card-border)', overflow: 'hidden', position: 'relative', background: '#fff' }}>
          {imgUrl ? (
            <img src={absOrStatic(imgUrl)} alt={caption} style={{ display: 'block', width: '100%', height: 'auto' }} />
          ) : (
            <div className="subtle" style={{ padding: 24, textAlign: 'center' }}>No overlays for this analysis.</div>
          )}

          {has > 1 && (
            <>
              <button
                aria-label="Previous"
                onClick={() => go(-1)}
                style={{ position: 'absolute', inset: '0 auto 0 0', width: '24%', background: 'transparent', border: 'none', cursor: 'pointer' }}
              />
              <button
                aria-label="Next"
                onClick={() => go(+1)}
                style={{ position: 'absolute', inset: '0 0 0 auto', width: '24%', background: 'transparent', border: 'none', cursor: 'pointer' }}
              />
              {/* transparent arrow hints */}
              <div style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', opacity: .5 }}>◀︎</div>
              <div style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', opacity: .5 }}>▶︎</div>
            </>
          )}
        </div>
      </div>
      {caption && <div className="subtle" style={{ padding: '0 16px 16px' }}>{caption}</div>}
    </div>
  );
}

export { absOrStatic, API_URL };
