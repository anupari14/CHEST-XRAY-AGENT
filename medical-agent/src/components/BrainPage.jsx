// src/components/BrainPage.jsx
import { useState } from 'react';
import axios from 'axios';
import { API_URL, absOrStatic } from '../lib/client.js';

export default function BrainPage({ headers }) {
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
    } catch {
      setMsg('Brain inference backend not configured yet. This is a placeholder UI.');
    }
  };

  return (
    <section style={{ width: '100%', maxWidth: '100%' }}>
      <div className="card" style={{ padding: 16 }}>
        <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Brain Tumour Detection</h2>
        <div style={{ marginTop: 12, display: 'grid', gap: 12, gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)' }}>
          <div className="card" style={{ padding: 12 }}>
            <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>
              Upload Brain MRI (.png/.jpg/.nii/.nii.gz)
            </label>
            <input type="file" accept=".png,.jpg,.jpeg" onChange={onFile} />
            <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <label className="subtle" style={{ fontSize: '.9rem' }}>Detection threshold</label>
              <input
                type="number" step="0.01" min={0} max={1} value={thr}
                onChange={(e) => setThr(parseFloat(e.target.value || 0))}
                className="input" style={{ width: 100 }}
              />
              <button onClick={run} className="btn" style={{ marginLeft: 'auto' }}>Run Detection</button>
            </div>
            {!!msg && <div className="subtle" style={{ marginTop: 8 }}>{msg}</div>}
          </div>
          <div className="card" style={{ padding: 12 }}>
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
          <pre className="card" style={{ padding: 12, fontSize: '.8rem', overflow: 'auto', maxHeight: 320, minHeight: 160 }}>
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
