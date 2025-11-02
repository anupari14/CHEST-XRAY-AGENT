// src/lib/paths.js
import { API_URL } from './client';

export function absOrStatic(path) {
  if (!path || typeof path !== 'string') return null;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return `${API_URL.replace(/\/$/, '')}${path}`;
  if (path.includes('artifacts/')) {
    // convert repo/artifacts style to /static/<rel>
    const rel = path.replace(/\\/g, '/').split('artifacts/', 2)[1];
    return `${API_URL.replace(/\/$/, '')}/static/${rel}`;
  }
  return `${API_URL.replace(/\/$/, '')}/${path.replace(/^\//,'')}`;
}
