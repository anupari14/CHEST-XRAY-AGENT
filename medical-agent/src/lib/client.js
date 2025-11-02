// src/lib/client.js
export const API_URL =
  import.meta?.env?.VITE_MEDAGENT_API_URL || 'http://127.0.0.1:8000';

export function absOrStatic(path) {
  if (!path || typeof path !== 'string') return null;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return `${API_URL.replace(/\/$/, '')}${path}`;
  if (path.includes('artifacts/')) {
    const rel = path.replace(/\\/g, '/').split('artifacts/', 2)[1];
    return `${API_URL.replace(/\/$/, '')}/static/${rel}`;
  }
  return path;
}
