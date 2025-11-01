// A compact stethoscope + cross mark, looks clinical at small sizes
export default function ClinicalIcon({ size = 18, className = "", title = "Clinical Assistant" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label={title} className={className}>
      <defs>
        <linearGradient id="clin-ink" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#3F7187"/>
          <stop offset="100%" stopColor="#2F4858"/>
        </linearGradient>
      </defs>
      {/* cross */}
      <rect x="20.5" y="6" width="3" height="8" rx="1" fill="url(#clin-ink)"/>
      <rect x="18" y="8.5" width="8" height="3" rx="1" fill="url(#clin-ink)"/>
      {/* stethoscope */}
      <g fill="none" stroke="url(#clin-ink)" strokeWidth="1.8" strokeLinecap="round">
        <path d="M7 6 v6 a5 5 0 0 0 10 0 V6" />
        <path d="M12 17 v3 a5 5 0 0 0 10 0 v-3" />
        <circle cx="24" cy="20" r="3" fill="#fff" />
      </g>
    </svg>
  );
}
