export default function Logo({ size = 28 }) {
  // Inline medical cross+shield icon, tweak color to your brand
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#3F7187"/>
          <stop offset="100%" stopColor="#2F4858"/>
        </linearGradient>
      </defs>
      <path fill="url(#g)" d="M12 2l7 3v6c0 5-3.7 9.4-7 11-3.3-1.6-7-6-7-11V5l7-3z"/>
      <rect x="8.25" y="9.25" width="7.5" height="1.5" rx=".75" fill="#fff"/>
      <rect x="11.25" y="6.75" width="1.5" height="7.5" rx=".75" fill="#fff"/>
    </svg>
  );
}
