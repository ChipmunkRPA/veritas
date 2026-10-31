/** Cooperative cancellation retains the source and completed evidence. The server owns the terminal state. */
import { apiClient, unwrap } from './client'

/** `note` (N-19a): why the run is cancelled; recorded with the request and shown on the analysis. */
export function requestCancellation(aid: string, note?: string) {
  const text = note?.trim()
  return unwrap(apiClient.POST('/api/v2/analyses/{aid}/cancel', { params: { path: { aid } }, body: text ? { note: text } : {} }))
}
