import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/api'
import type { Size } from '../../lib/coords'
import type { Base } from './scene'
import { TileSource, type TileSetInfo } from './TileSource'

export const tokenQuery = (token: string): string => (token ? `?token=${encodeURIComponent(token)}` : '')
export const exhibitUrl = (exhibitId: string, part: 'payload' | 'mask' | 'preview', token: string): string =>
  `/api/v2/exhibits/${exhibitId}/${part}${tokenQuery(token)}`

export async function fetchBlob(url: string): Promise<Blob> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url.split('?')[0]}`)
  return r.blob()
}

export interface BaseState { base: Base | null; source: Size | null; error: string | null }

/** The source image (ARCHITECTURE §7.4): the tile pyramid when the heatmaps carry one (above 4096 px), else one decoded bitmap of
 *  the original. Requests start once `enabled` (the media token is ready); a renewed token is used for later tiles without
 *  reloading. `onUpdate` fires whenever a tile arrives. */
export function useBase({ enabled, analysisId, tiles, token, onUpdate }: {
  enabled: boolean
  analysisId: string
  tiles: { exhibitId: string; info: TileSetInfo } | null
  token: string
  onUpdate: () => void
}): BaseState {
  const [state, setState] = useState<BaseState>({ base: null, source: null, error: null })
  const update = useRef(onUpdate)
  update.current = onUpdate
  const tokenRef = useRef(token)
  tokenRef.current = token
  const tilesKey = tiles ? `${tiles.exhibitId}:${tiles.info.width}x${tiles.info.height}:${tiles.info.levels}:${tiles.info.tile_px}` : ''

  useEffect(() => {
    if (!enabled) return
    let live = true
    if (tiles) {
      const { exhibitId, info } = tiles
      const source = new TileSource(info, (z, x, y) => `/api/v2/exhibits/${exhibitId}/tiles/${z}/${x}_${y}.png${tokenQuery(tokenRef.current)}`,
        () => update.current(), (message) => { if (live) setState((s) => ({ ...s, error: message })) })
      setState({ base: { kind: 'tiles', tiles: source }, source: { width: info.width, height: info.height }, error: null })
      return () => {
        live = false
        source.dispose()
      }
    }
    let bitmap: ImageBitmap | null = null
    setState({ base: null, source: null, error: null })
    fetchBlob(api.sourceUrl(analysisId, tokenRef.current))
      .then((blob) => createImageBitmap(blob))
      .then((b) => {
        if (!live) {
          b.close()
          return
        }
        bitmap = b
        setState({ base: { kind: 'bitmap', bitmap: b }, source: { width: b.width, height: b.height }, error: null })
      })
      .catch((e: unknown) => {
        if (live) setState({ base: null, source: null, error: e instanceof Error ? e.message : String(e) })
      })
    return () => {
      live = false
      const closing = bitmap
      if (!closing) return
      // Clear the state before closing, so no later frame draws a detached bitmap.
      setState((s) => (s.base?.kind === 'bitmap' && s.base.bitmap === closing ? { base: null, source: null, error: null } : s))
      closing.close()
    }
    // tiles is represented by tilesKey
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, analysisId, tilesKey])
  return state
}
