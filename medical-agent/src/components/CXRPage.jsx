// src/components/CXRPage.jsx
import { useRef, useState } from 'react';
import axios from 'axios';
import { API_URL, absOrStatic } from '../lib/client.js';
import { InfoKV, PDFDownloader, PatientPicker, ImageCarousel } from './common/ui.jsx';

export default function CXRPage({ headers }) {
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
      <div className="card" style={{ padding: 16, width: '100%', maxWidth: '100%' }}>
        <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Chest X-Ray Analysis</h2>
        <div style={{ marginTop: 12, display: 'grid', gap: 12, gridTemplateColumns: 'minmax(0, 4fr)' }}>
          <div className="card" style={{ padding: 12 }}>
            <label className="subtle" style={{ fontSize: '.9rem', display: 'block', marginBottom: 6 }}>
              Upload Chest X-ray (.png/.jpg/.dcm)
            </label>
            <input type="file" accept=".png,.jpg,.jpeg,.dcm" onChange={onFile} />
            <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>
              <PatientPicker headers={headers} onSelect={(p) => setPatientId(p.patient_id)} />
              <div className="subtle" style={{ fontSize: '.85rem' }}>
                Selected Patient ID to use: <b>{patientId || '—'}</b>
              </div>
              <button onClick={analyze} disabled={busy} className="btn" style={{ width: '100%', marginTop: 4, position: 'relative' }}>
                {busy ? 'Analyzing…' : 'Analyze'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {(pinfo && (Object.keys(pinfo).length > 0)) && (
        <div className="card" style={{ padding: 16 }}>
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

      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: '1fr 1fr' }}>
        <div ref={inputImgBoxRef} className="card" style={{ padding: 16 }}>
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
      </div>

      <div className="card" style={{ padding: 16 }}>
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
