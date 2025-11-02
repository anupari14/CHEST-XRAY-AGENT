// src/App.jsx
import { useMemo, useState } from 'react';
import './styles/theme.css';
import { API_URL } from './lib/client.js';
import LoginPage from './components/LoginPage.jsx';
import FloatingChatbot from './components/FloatingChatbot.jsx';
import PatientRegistry from './components/PatientRegistry.jsx';
import CXRPage from './components/CXRPage.jsx';
import BrainPage from './components/BrainPage.jsx';


export default function App() {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('cxr');

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

      {/* Main */}
      <main style={{ width: '100%', padding: '12px var(--page-gutter) 96px' }}>
        {activeTab === 'patient' && <PatientRegistry headers={headers} />}
        {activeTab === 'cxr' && <CXRPage headers={headers} />}
        {activeTab === 'brain' && <BrainPage headers={headers} />}
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
