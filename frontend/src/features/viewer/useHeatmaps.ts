import { useEffect, useRef, useState } from 'react'
import { HeatmapClient, type LoadedMap } from './heatmapClient'
import type { HeatmapPayload } from './lenses'
import { exhibitUrl, fetchBlob } from './useBase'

export interface HeatLayers {
  decoded: Readonly<Record<string, LoadedMap>>
  errors: Readonly<Record<string, string>>
  /** The rendered layer of a map at the current threshold (the previous threshold's layer while a new one renders). */
  layer: (id: string | null | undefined) => ImageBitmap | null
}

/** Decodes every map lens of the analysis once in the worker, and renders each at the `Show ≥` threshold. Decoding all maps up front
 *  keeps a lens toggle a redraw (ARCHITECTURE §8.2: layer toggle ≤ 100 ms). */
export function useHeatmaps(maps: readonly HeatmapPayload[], token: string, enabled: boolean, show: number): HeatLayers {
  const [client, setClient] = useState<HeatmapClient | null>(null)
  const [decoded, setDecoded] = useState<Record<string, LoadedMap>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [rendered, setRendered] = useState<{ show: number; layers: Record<string, ImageBitmap> }>({ show, layers: {} })
  const tokenRef = useRef(token)
  tokenRef.current = token
  const ids = maps.map((m) => `${m.id}${m.reliability_mask_artefact_id ? '+m' : ''}`).join(',')

  useEffect(() => {
    const c = new HeatmapClient()
    setClient(c)
    return () => c.dispose()
  }, [])

  useEffect(() => {
    if (!client || !enabled) return
    let live = true
    setDecoded({})
    setErrors({})
    for (const m of maps) {
      const hasMask = m.reliability_mask_artefact_id != null
      Promise.all([fetchBlob(exhibitUrl(m.id, 'payload', tokenRef.current)), hasMask ? fetchBlob(exhibitUrl(m.id, 'mask', tokenRef.current)) : null])
        .then(([payload, mask]) => client.decode(m.id, payload, mask))
        .then((loaded) => { if (live) setDecoded((d) => ({ ...d, [m.id]: loaded })) })
        .catch((e: unknown) => { if (live) setErrors((x) => ({ ...x, [m.id]: e instanceof Error ? e.message : String(e) })) })
    }
    return () => { live = false }
    // maps is represented by ids
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, enabled, ids])

  useEffect(() => {
    if (!client) return
    const todo = Object.keys(decoded).filter((id) => !(rendered.show === show && rendered.layers[id]))
    if (!todo.length) return
    let live = true
    Promise.all(todo.map(async (id) => [id, await client.render(id, show)] as const))
      .then((pairs) => {
        if (!live) {
          pairs.forEach(([, b]) => b.close())
          return
        }
        setRendered((prev) => {
          if (prev.show === show) return { show, layers: { ...prev.layers, ...Object.fromEntries(pairs) } }
          return { show, layers: Object.fromEntries(pairs) }
        })
      })
      .catch((e: unknown) => { if (live) setErrors((x) => ({ ...x, render: e instanceof Error ? e.message : String(e) })) })
    return () => { live = false }
    // rendered is read to skip layers that exist; re-running on its change would loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, decoded, show])

  // Release layers of a previous threshold once they are replaced.
  const previous = useRef(rendered)
  useEffect(() => {
    const old = previous.current
    if (old !== rendered && old.show !== rendered.show) {
      for (const [id, b] of Object.entries(old.layers)) if (rendered.layers[id] !== b) b.close()
    }
    previous.current = rendered
  }, [rendered])

  return { decoded, errors, layer: (id) => (id ? rendered.layers[id] ?? null : null) }
}
