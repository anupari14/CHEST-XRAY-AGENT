import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import ClinicalIcon from "./ClinicalIcon.jsx";

/**
 * Floating website-style chatbot
 * - Bottom-right launcher button
 * - Expands to a panel (mobile = full-screen)
 * - Enter to send, Shift+Enter for newline
 */
export default function FloatingChatbot({
  apiUrl,
  headers = {},
  assistantName = "Dr. Aster",
  subtitle = "Clinical AI Assistant",
}) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [unread, setUnread] = useState(0);
  const [hist, setHist] = useState([
    ["assistant", `Hello, I’m ${assistantName}. Ask me about findings, impressions, or trends.`],
  ]);

  const listRef = useRef(null);

  const api = useMemo(() => axios.create({ baseURL: apiUrl, timeout: 60000 }), [apiUrl]);

  // Auto-scroll
  useEffect(() => {
    if (!open) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [hist, open]);

  // Close with ESC
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const quickPrompts = [
    "Summarise key findings",
    "Explain the impression in plain English",
    "List differential diagnoses",
    "Compare with the previous study",
  ];

  const sendPrompt = (p) => {
    setText(p);
    setTimeout(() => send(), 0);
  };

  const send = async () => {
    const msg = (text || "").trim();
    if (!msg) return;
    setErr("");
    setHist((h) => [...h, ["user", msg]]);
    setText("");
    setBusy(true);
    try {
      const r = await api.post("/chat", { query: msg, k: 4 }, { headers });
      const data = r.data || {};
      const answer = data.answer || "";
      const sources = Array.isArray(data.sources) ? data.sources : [];
      setHist((h) => [...h, ["assistant", answer]]);
      if (sources.length) {
        const lines = sources.slice(0, 3).map((s) => {
          const name = s.patient_name || "—";
          const mrn = s.mrn || "—";
          const score = Number.parseFloat(s.score || 0).toFixed(3);
          return `• ${name} (MRN ${mrn}) — ${score}`;
        });
        setHist((h) => [...h, ["assistant", `**Matches**\n${lines.join("\n")}`]]);
      }
    } catch (ex) {
      const m = ex?.response?.data?.detail || ex.message;
      setErr(m);
      setHist((h) => [...h, ["assistant", `⚠️ Chat failed: ${m}`]]);
    } finally {
      setBusy(false);
    }
  };

  // Count unread if panel is closed
  useEffect(() => {
    if (!open) {
      const last = hist[hist.length - 1];
      if (last && last[0] === "assistant") setUnread((n) => Math.min(n + 1, 9));
    } else {
      setUnread(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hist.length, open]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <>
      {/* Launcher button */}
      {!open && (
        <button
          aria-label="Open clinical assistant"
          className="chat-launcher"
          onClick={() => setOpen(true)}
          title={`${assistantName}`}
        >
          <ClinicalIcon size={22} />
          <span className="chat-launcher-label">Ask {assistantName}</span>
          {unread > 0 && <span className="chat-badge">{unread}</span>}
        </button>
      )}

      {/* Panel */}
      {open && (
        <div className="chat-panel-wrap" role="dialog" aria-label="Clinical assistant">
          <div className="chat-panel card">
            {/* Header */}
            <div className="chat-head">
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="chat-avatar">
                  <ClinicalIcon size={16} />
                </div>
                <div>
                  <div style={{ fontWeight: 700 }}>{assistantName}</div>
                  <div className="subtle" style={{ fontSize: ".8rem", marginTop: -2 }}>{subtitle}</div>
                </div>
              </div>
              <button
                className="btn"
                style={{ background: "#fff", color: "var(--brand-ink)", borderColor: "var(--card-border)" }}
                onClick={() => setOpen(false)}
              >
                Close
              </button>
            </div>

            {/* Quick prompts */}
            {/* <div className="chat-prompts">
              {quickPrompts.map((p, i) => (
                <button key={i} onClick={() => sendPrompt(p)} className="btn chat-chip">
                  {p}
                </button>
              ))}
            </div> */}

            {/* Messages */}
            <div ref={listRef} className="chat-body">
              {hist.map((m, i) => <Msg key={i} role={m[0]} text={m[1]} />)}
              {busy && <TypingDots />}
            </div>

            {/* Error */}
            {!!err && <div className="chat-error">{String(err)}</div>}

            {/* Input */}
            <div className="chat-input">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Type a message… (Shift+Enter for new line)"
                className="textarea"
                style={{ height: 44, resize: "none", flex: 1 }}
              />
              <button className="btn" onClick={send} disabled={busy}>Send</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* --------- pieces ---------- */

function Msg({ role, text }) {
  const isUser = role === "user";
  return (
    <div style={{ display: "flex", gap: 8, margin: "6px 0", justifyContent: isUser ? "flex-end" : "flex-start" }}>
      {!isUser && <div className="chat-bubble-avatar"><ClinicalIcon size={12} /></div>}
      <div
        className="chat-bubble"
        style={{
          background: isUser ? "#E8EEF3" : "#F5F7F8",
          color: isUser ? "#0F172A" : "#2D3B44",
          borderTopRightRadius: isUser ? 6 : 16,
          borderTopLeftRadius: isUser ? 16 : 6,
        }}
      >
        <RichText text={text} />
      </div>
      {isUser && <div className="chat-bubble-avatar user">U</div>}
    </div>
  );
}

function RichText({ text }) {
  const html = String(text || "")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\*(.+?)\*/g, "<i>$1</i>")
    .replace(/\n/g, "<br/>");
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

function TypingDots() {
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", margin: "6px 0" }}>
      <div className="chat-bubble-avatar"><ClinicalIcon size={12} /></div>
      <div className="chat-typing"><Dot /><Dot delay="0.12s" /><Dot delay="0.24s" /></div>
    </div>
  );
}

function Dot({ delay = "0s" }) {
  return <span className="chat-dot" style={{ animationDelay: delay }} />;
}
