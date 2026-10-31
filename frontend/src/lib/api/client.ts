/** Typed client for the backend API, generated from the FastAPI OpenAPI schema (`make api-types`, ARCHITECTURE §5.5).
 *  v2 reads go through `queries.ts`; the hand-written `lib/api.ts` keeps serving the v1 routes of the legacy views. */
import createClient, { type Middleware } from 'openapi-fetch'
import type { paths } from './schema'
import { refusedSession } from './sessionEnd'

const KEY_STORAGE = 'veritas_api_key'
const CSRF_COOKIE = 'veritas_csrf'
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

/** Session mode (L-6): the CSRF token the sign-in set in the readable `veritas_csrf` cookie, or null without a session. */
export function csrfToken(cookies: string = typeof document === 'undefined' ? '' : document.cookie): string | null {
  for (const part of cookies.split(';')) {
    const [name, ...value] = part.trim().split('=')
    if (name === CSRF_COOKIE && value.length > 0) return decodeURIComponent(value.join('='))
  }
  return null
}

/** Key mode sends `X-API-Key` when a key is stored. Session mode (the `veritas_session` cookie travels by itself) sends
 *  `X-CSRF-Token` on every write, which the server requires (double submit, C-1). */
export function apiKeyMiddleware(getKey: () => string | null, getCsrf: () => string | null = csrfToken): Middleware {
  return {
    onRequest({ request }) {
      for (const [k, v] of Object.entries(credentialHeaders(request.method, getKey(), getCsrf()))) request.headers.set(k, v)
      return request
    },
  }
}

function credentialHeaders(method: string, key: string | null, csrf: string | null): Record<string, string> {
  const headers: Record<string, string> = {}
  if (key) headers['X-API-Key'] = key
  if (csrf && !SAFE_METHODS.has(method.toUpperCase())) headers['X-CSRF-Token'] = csrf
  return headers
}

const storedKey = (): string | null => (typeof localStorage === 'undefined' ? null : localStorage.getItem(KEY_STORAGE))

/** The credential headers for requests that bypass openapi-fetch (XHR uploads with progress, blob downloads, the v1 client):
 *  pass the method so a write carries the session's CSRF token. */
export const authHeaders = (method = 'GET'): Record<string, string> => credentialHeaders(method, storedKey(), csrfToken())

export function createApiClient(baseUrl = '', getKey: () => string | null = storedKey, getCsrf: () => string | null = csrfToken) {
  const client = createClient<paths>({ baseUrl })
  client.use(apiKeyMiddleware(getKey, getCsrf))
  return client
}

export const apiClient = createApiClient()

/** A failed API call. `message` is the problem+json `detail` (already sanitised server-side), else the HTTP status text. */
export class ApiError extends Error {
  constructor(readonly status: number, readonly title: string, detail: string) {
    super(detail)
    this.name = 'ApiError'
  }
}

/** The body of a successful openapi-fetch call; raises `ApiError` otherwise. `signIn`: the call is the sign-in itself, whose 401
 *  is a refused password and not an ended session. */
export async function unwrap<T>(call: Promise<{ data?: T; error?: unknown; response: Response }>, { signIn = false }: { signIn?: boolean } = {}): Promise<T> {
  const { data, error, response } = await call
  if (!response.ok || error !== undefined) {
    const body = (error && typeof error === 'object' ? error : {}) as { title?: unknown; detail?: unknown }
    const fallback = response.statusText || `HTTP ${response.status}`
    const detail = typeof body.detail === 'string' ? body.detail : fallback
    // N-19a (browser QA P1-11): a 401 while the application is open is a session that ended, which the gate answers with the
    // sign-in form in place. A refused password is the answer of the sign-in itself and keeps its sentence.
    const ended = response.status === 401 && !signIn
    throw new ApiError(response.status, typeof body.title === 'string' ? body.title : fallback, ended ? refusedSession(detail) : detail)
  }
  return data as T
}
