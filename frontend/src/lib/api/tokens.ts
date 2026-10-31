/** Resource-scoped tokens (ARCHITECTURE §5.3, §7.3) for GET URLs that cannot carry a header: media elements and
 *  EventSource. Grants are cached until 30 s before expiry, which generalises the v1 download-token cache in `lib/api.ts`,
 *  and media renders only once its token is ready (the transient-401 fix of production-readiness §E2E). */
import { useQuery } from '@tanstack/react-query'
import { apiClient, unwrap } from './client'
import { qk } from './queries'

export const TOKEN_TTL_S = 600
export const REFRESH_MARGIN_MS = 30_000

export interface Grant {
  token: string
  /** Epoch milliseconds. */
  expiresAt: number
}

export type GrantFetcher = (resource: string) => Promise<Grant>

export class TokenCache {
  private readonly grants = new Map<string, Grant>()
  private readonly inflight = new Map<string, Promise<Grant>>()

  constructor(private readonly fetcher: GrantFetcher, private readonly now: () => number = Date.now) {}

  /** A grant that stays valid for more than the refresh margin. Concurrent callers share one request; failures are not cached. */
  get(resource: string): Promise<Grant> {
    const cached = this.grants.get(resource)
    if (cached && cached.expiresAt - this.now() > REFRESH_MARGIN_MS) return Promise.resolve(cached)
    let pending = this.inflight.get(resource)
    if (!pending) {
      pending = this.fetcher(resource)
        .then((grant) => { this.grants.set(resource, grant); return grant })
        .finally(() => this.inflight.delete(resource))
      this.inflight.set(resource, pending)
    }
    return pending
  }

  /** Milliseconds until the grant is due for renewal (0 when it is due now). */
  refreshIn(grant: Grant): number {
    return Math.max(0, grant.expiresAt - REFRESH_MARGIN_MS - this.now())
  }

  clear(): void {
    this.grants.clear()
  }
}

export const fetchGrant: GrantFetcher = async (resource) => {
  const body = await unwrap(apiClient.POST('/api/v2/tokens', { body: { resource, ttl_s: TOKEN_TTL_S } }))
  return { token: body.token, expiresAt: Date.parse(body.expires_at) }
}

export const tokenCache = new TokenCache(fetchGrant)

/** The token for `resource` (`events:<id>`, `exhibit:<id>`, `media:<id>`, `analysis:<id>`), renewed before it expires.
 *  `null` disables the query. */
export function useResourceToken(resource: string | null) {
  const q = useQuery({
    queryKey: qk.token(resource ?? ''),
    queryFn: () => tokenCache.get(resource as string),
    enabled: resource !== null,
    staleTime: (query) => (query.state.data ? tokenCache.refreshIn(query.state.data) : 0),
    refetchInterval: (query) => (query.state.data ? Math.max(1000, tokenCache.refreshIn(query.state.data)) : false),
    gcTime: TOKEN_TTL_S * 1000,
    retry: 1,
  })
  return { token: q.data?.token ?? null, ready: q.data !== undefined, error: q.error }
}
