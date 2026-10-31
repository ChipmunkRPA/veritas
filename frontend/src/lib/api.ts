import { sha256Blob } from './hash'
import { authHeaders } from './api/client'
import { refusedSession } from './api/sessionEnd'
import { readEngineState, type EngineState } from '../features/engine/engineState'

export type Kind = 'image' | 'video' | 'audio' | 'document'

export interface Signal {
  id: string; name: string; category: string; observed: string; value: number | null; reference: string
  ai_probability: number | null; weight: number; direction: string; strength: string; reasoning: string
  artifacts: string[]; details: Record<string, unknown>; decisive: boolean
}
export interface Verdict {
  probability_ai: number | null; band: string; confidence: string; label: string; summary: string
  reasoning: string[]; caveats: string[]; signals_for_ai: number; signals_for_authentic: number; signals_total: number
}
export interface SentenceOut {
  idx: number; start: number; end: number; text: string; paragraph: number; p_ai: number | null; band: string
  reasons: string[]; scores: Record<string, number>; flags?: string[]
}
export interface Analysis {
  id: string; kind: Kind; filename: string; content_type: string; size_bytes: number; sha256: string; status: string
  created_at: string | null; started_at: string | null; finished_at: string | null; error: string | null
  probability_ai: number | null; band: string | null; has_report: boolean
  result?: {
    verdict: Verdict; signals: Signal[]; artifacts: string[]; timings: Record<string, number>
    faces?: { face: { x: number; y: number; w: number; h: number; score: number }; scores: Record<string, number>; p_ai?: number }[]
    image?: { width: number; height: number; format: string }
    video?: Record<string, unknown> & { duration?: number; width?: number; height?: number; fps?: number }
    /** Audio analyses (A-1): ffprobe facts, content shares and assessability (`engine/stages/audio.py::fuse`). */
    audio?: Record<string, unknown>

    frames?: { index: number; time: number; p_frame: number; p_face: number | null; p_general: number; faces: number; scores: Record<string, number> }[]
    text?: string; sentences?: SentenceOut[]; paragraphs?: [number, number][]
    stats?: { words: number; sentences: number; scored_sentences?: number; ai_sentences: number; mixed_sentences: number; human_sentences: number; ai_share: number; mixed_share: number; weighted_ai_share: number; binoculars_document: number | null }
    metadata?: Record<string, unknown>
  } | null
}

const KEY = localStorage.getItem('veritas_api_key') || ''
export const hasApiKey = KEY.length > 0  // media URLs need a signed token when a key is configured
/** The stored key, and on writes the session's CSRF token (L-6). */
const headers = (method = 'GET'): Record<string, string> => authHeaders(method)

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail ?? msg } catch { /* ignore */ }
    const text = typeof msg === 'string' ? msg : JSON.stringify(msg)
    throw new Error(r.status === 401 ? refusedSession(text) : text)  // N-19a: a session that ended while the page is open
  }
  return r.status === 204 ? (undefined as T) : r.json()
}

/** Guided-demo sample as served by `/api/demo` (no id) and `/api/demo/start` (with id + status). */
export interface DemoItem { kind: Kind; title: string; blurb: string; id?: string; status?: string; decision_code?: string | null }

export const api = {
  demo: (): Promise<{ items: DemoItem[] }> => fetch('/api/demo', { headers: headers() }).then((r) => handle<{ items: DemoItem[] }>(r)),
  /** 202: analyses for the five samples in walkthrough order; completed ones are reused so repeat demos are instant. */
  demoStart: (): Promise<{ items: DemoItem[] }> => fetch('/api/demo/start', { method: 'POST', headers: headers('POST') }).then((r) => handle<{ items: DemoItem[] }>(r)),
  list: (): Promise<Analysis[]> => fetch('/api/analyses?limit=200', { headers: headers() }).then((r) => handle<Analysis[]>(r)),
  /** L-2: the published decision code (v2 detail), so the demo tour describes what the screen shows. */
  decisionCode: (id: string): Promise<string | null> => fetch(`/api/v2/analyses/${id}`, { headers: headers() })
    .then((r) => handle<{ decision?: { code?: string } | null }>(r)).then((a) => a.decision?.code ?? null),
  get: (id: string): Promise<Analysis> => fetch(`/api/analyses/${id}`, { headers: headers() }).then((r) => handle<Analysis>(r)),
  remove: (id: string): Promise<void> => fetch(`/api/analyses/${id}`, { method: 'DELETE', headers: headers('DELETE') }).then((r) => handle<void>(r)),
  /** The file is hashed locally first (`client_sha256`, UX §5.1); the server refuses the upload when its own hash differs. */
  upload: async (kind: Kind, file: File): Promise<Analysis> => {
    const fd = new FormData(); fd.append('client_sha256', await sha256Blob(file)); fd.append('file', file)
    return fetch(`/api/analyze/${kind}`, { method: 'POST', body: fd, headers: headers('POST') }).then((r) => handle<Analysis>(r))
  },
  /** Same as `upload` but reports determinate progress (0..1) via XMLHttpRequest. Key travels in the header only. Without a
   *  `profile` the instance default applies. */
  uploadWithProgress: async (kind: Kind, file: File, onProgress: (frac: number) => void, profile?: 'quick' | 'full' | 'deep'): Promise<Analysis> => {
    const sha256 = await sha256Blob(file)
    return new Promise<Analysis>((resolve, reject) => {
      const fd = new FormData(); fd.append('client_sha256', sha256); if (profile) fd.append('profile', profile); fd.append('file', file)
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `/api/analyze/${kind}`)
      for (const [k, v] of Object.entries(headers('POST'))) xhr.setRequestHeader(k, v)
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total) }
      xhr.onerror = () => reject(new Error('Network error during upload'))
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText) as Analysis) } catch { reject(new Error('Malformed response')) }
        } else {
          let msg = xhr.statusText || `HTTP ${xhr.status}`
          try { const d = JSON.parse(xhr.responseText).detail; if (d) msg = typeof d === 'string' ? d : JSON.stringify(d) } catch { /* ignore */ }
          reject(new Error(xhr.status === 401 ? refusedSession(msg) : msg))
        }
      }
      xhr.send(fd)
    })
  },
  text: (text: string, title = 'pasted-text.txt', profile?: 'quick' | 'full' | 'deep'): Promise<Analysis> =>
    fetch('/api/analyze-text', { method: 'POST', body: JSON.stringify({ text, title, ...(profile ? { profile } : {}) }), headers: { 'Content-Type': 'application/json', ...headers('POST') } }).then((r) => handle<Analysis>(r)),
  /** The engine as `/api/ready` tells it to this caller (the stored key or the session travels with it). A 503 that is a
   *  readiness answer resolves; anything else rejects, which the shell shows as "offline". */
  health: (): Promise<EngineState> => fetch('/api/ready', { headers: headers() }).then(async (r) => {
    const state = await readEngineState(r)
    if (state) return state
    if (r.ok) throw new Error('Malformed response')
    return handle<EngineState>(r)
  }),
  // Browser-opened URLs (images, PDF, JSON) never carry the API key; they use a short-lived signed token
  // fetched per analysis with the key in the header. Without an API key configured, plain URLs work.
  downloadToken: async (id: string): Promise<string> => {
    if (!KEY) return ''
    const cached = tokenCache.get(id)
    if (cached && cached.exp > Date.now() + 30_000) return cached.token
    const r = await fetch(`/api/analyses/${id}/download-token`, { headers: headers() }).then((x) => handle<{ token: string; ttl_seconds: number }>(x))
    tokenCache.set(id, { token: r.token, exp: Date.now() + r.ttl_seconds * 1000 })
    return r.token
  },
  artifactUrl: (id: string, name: string, token = '') => `/api/analyses/${id}/artifacts/${name}${token ? `?token=${token}` : ''}`,
  sourceUrl: (id: string, token = '') => `/api/analyses/${id}/source${token ? `?token=${token}` : ''}`,
  reportUrl: (id: string, token = '') => `/api/analyses/${id}/report.pdf${token ? `?token=${token}` : ''}`,
  exportUrl: (id: string, token = '') => `/api/analyses/${id}/export.json${token ? `?token=${token}` : ''}`,
}

const tokenCache = new Map<string, { token: string; exp: number }>()

/** Band labels of the v1 view. D-66: a label never claims authenticity; the clean bands read as the v2 headlines do. */
export const BAND_LABEL: Record<string, string> = {
  very_likely_authentic: 'No indications of AI generation or manipulation found',
  likely_authentic: 'No significant indications of AI generation or manipulation', uncertain: 'Inconclusive',
  likely_ai: 'Likely AI / manipulated', very_likely_ai: 'Very likely AI / manipulated',
}
export const BAND_COLOR: Record<string, string> = {
  very_likely_authentic: '#2e7d32', likely_authentic: '#4f9d69', uncertain: '#e3a21a', likely_ai: '#d9662a', very_likely_ai: '#c0392b',
}
export const pct = (x: number | null | undefined, d = 0) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)

/** Position of each band on the 5-step verdict scale; components resolve `var(--color-verdict-n)`. */
export const BAND_STEP: Record<string, 1 | 2 | 3 | 4 | 5> = {
  very_likely_authentic: 1, likely_authentic: 2, uncertain: 3, likely_ai: 4, very_likely_ai: 5,
}
/** Short band labels for dense contexts (list rows). */
export const BAND_SHORT: Record<string, string> = {
  very_likely_authentic: 'No indications', likely_authentic: 'No significant indications', uncertain: 'Inconclusive',
  likely_ai: 'Likely AI', very_likely_ai: 'AI / manipulated',
}
export const bandStep = (band: string | null | undefined): 1 | 2 | 3 | 4 | 5 => (band && BAND_STEP[band]) || 3
/** Sentence-level band (ai / mixed / human) to verdict step. */
export const sentenceStep = (band: string): 1 | 3 | 5 => (band === 'ai' ? 5 : band === 'mixed' ? 3 : 1)
/** Frame-level probability to verdict step (direction A §4.7 band strip thresholds). */
export const frameStep = (p: number): 1 | 2 | 3 | 4 | 5 => (p >= 0.8 ? 5 : p >= 0.65 ? 4 : p <= 0.2 ? 1 : p <= 0.35 ? 2 : 3)
