import { useState } from 'react';
import axios from 'axios';
import Logo from './Logo.jsx';

const API_URL =
  import.meta?.env?.VITE_MEDAGENT_API_URL ||
  'http://127.0.0.1:8000';

export default function LoginPage({ onSuccess }) {
  const [u, setU] = useState('');
  const [p, setP] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setErr('');
    setBusy(true);
    try {
      const res = await axios.post(
        `${API_URL}/auth/login`,
        new URLSearchParams({ username: u, password: p }),
        { timeout: 30000 }
      );
      const { token, user } = res.data || {};
      if (!token) throw new Error('No token returned');
      onSuccess(token, user ?? u);
    } catch (ex) {
      setErr(String(ex?.response?.data?.detail || ex.message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{minHeight:'100vh', display:'grid', gridTemplateColumns:'1.15fr 1fr'}}>
      {/* LEFT: brand + background illustration */}
      <div style={{position:'relative', overflow:'hidden', background:'linear-gradient(135deg, #F0F6FA 0%, #E8F1F6 100%)'}}>
        {/* subtle pattern */}
        <svg aria-hidden="true" viewBox="0 0 600 600" style={{position:'absolute', inset:0, opacity:.22}}>
          <defs>
            <radialGradient id="bg" cx="50%" cy="40%" r="75%">
              <stop offset="0%" stopColor="#9BC0D3"/>
              <stop offset="100%" stopColor="#E8F1F6"/>
            </radialGradient>
          </defs>
          <rect width="600" height="600" fill="url(#bg)"/>
          <g stroke="#bcd1dd" strokeWidth="0.6" fill="none">
            {Array.from({length:16}).map((_,i)=>(
              <circle key={i} cx="320" cy="260" r={30 + i*18}/>
            ))}
          </g>
        </svg>

        {/* hero copy */}
        <div style={{position:'relative', padding:'48px 56px', height:'100%', display:'flex', flexDirection:'column', justifyContent:'space-between'}}>
          <div className="logo-lockup">
            <img src="/assets/AIDx-logo.png" alt="AIDx" style={{height:36, width:'auto'}}/>
            <h1>Medical Agent</h1>
          </div>

          <div style={{maxWidth: 560}}>
            <h2 style={{fontSize:'2rem', margin:'0 0 .5rem 0', color:'#2F4858'}}>Clinical Imaging Suite</h2>
            <p className="subtle" style={{fontSize:'1rem', lineHeight:1.6}}>
              Secure, AI-assisted workflows for radiology teams. Generate draft reports, visualize heatmaps,
              and keep patient data private with least-privilege access controls.
            </p>
          </div>

          <div className="subtle" style={{fontSize:'.85rem'}}>
            © {new Date().getFullYear()} AIDx • For research use only. Not for diagnostic purposes.
          </div>
        </div>
      </div>

      {/* RIGHT: login card */}
      <div style={{display:'grid', placeItems:'center', padding:'48px 24px'}}>
        <div className="card" style={{width:'min(520px, 92vw)', padding:'28px 28px 24px'}}>
          <div style={{display:'flex', alignItems:'center', gap:12, marginBottom:12}}>
            <Logo size={28}/>
            <div>
              <div style={{fontWeight:700}}>Welcome back</div>
              <div className="subtle" style={{fontSize:'.9rem'}}>Sign in to continue</div>
            </div>
          </div>

          <form onSubmit={submit} style={{marginTop:12, display:'grid', gap:12}}>
            <div>
              <label className="subtle" style={{fontSize:'.9rem', display:'block', marginBottom:6}}>Username</label>
              <input className="input" value={u} onChange={e=>setU(e.target.value)} autoComplete="username" />
            </div>
            <div>
              <div style={{display:'flex', justifyContent:'space-between', alignItems:'baseline'}}>
                <label className="subtle" style={{fontSize:'.9rem'}}>Password</label>
                {/* Optional forgot link */}
                {/* <a className="link" href="#">Forgot password?</a> */}
              </div>
              <input className="input" type="password" value={p} onChange={e=>setP(e.target.value)} autoComplete="current-password" />
            </div>

            {err && <div style={{color:'#b42318', background:'#fee4e2', border:'1px solid #fecdca', padding:'8px 10px', borderRadius:10, fontSize:'.9rem'}}>{err}</div>}

            <button type="submit" className="btn" disabled={busy}>
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <div className="subtle" style={{marginTop:14, fontSize:'.85rem'}}>
            By continuing, you agree to our <a className="link" href="#">Terms</a> and <a className="link" href="#">Privacy Policy</a>.
          </div>
        </div>
      </div>
    </div>
  );
}
