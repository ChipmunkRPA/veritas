import { MutationCache, QueryClient } from '@tanstack/react-query'
import { ApiError } from './client'

/** The detail of one analysis (`qk.analysis(id)`), not the queries under it. */
const isAnalysisDetail = (key: readonly unknown[]): boolean => key[0] === 'analysis' && key.length === 2

/** The application's query client.
 *
 *  A write that the server refuses with 409 was sent from a page whose picture of the analysis is out of date: a re-run of a
 *  revision that another tab has superseded, a sign-off of a conclusion that has changed. The refusal says so, and the page
 *  then reads the analysis again, so the state line and the buttons are those of the server (N-10b; code re-verification
 *  R2-08: the state line of a page left open did not change after the refusal). */
export function makeQueryClient(): QueryClient {
  const client: QueryClient = new QueryClient({
    defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
    mutationCache: new MutationCache({
      onError: (error) => {
        if (error instanceof ApiError && error.status === 409) {
          void client.invalidateQueries({ predicate: (q) => isAnalysisDetail(q.queryKey) })
        }
      },
    }),
  })
  return client
}
