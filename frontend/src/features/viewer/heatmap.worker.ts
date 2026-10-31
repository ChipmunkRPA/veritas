/** Heat layer worker (ARCHITECTURE §7.4): decodes map payloads and renders colormapped layers with `OffscreenCanvas`, so the
 *  main thread is not blocked. The protocol is in `heatmapClient.ts`. */
import { HeatEngine } from './heatmap'
import type { HeatRequest, HeatResponse } from './heatmapClient'

const engine = new HeatEngine()
const scope = self as unknown as { onmessage: ((e: MessageEvent<HeatRequest>) => void) | null; postMessage: (m: HeatResponse, t: Transferable[]) => void }

async function handle(req: HeatRequest): Promise<void> {
  try {
    if (req.type === 'decode') {
      const { map, unreliable } = await engine.decode(req.key, req.payload, req.mask)
      // The worker keeps its bytes for rendering; the main thread gets copies for the pixel probe.
      const values = map.values.slice()
      const mask = map.mask ? map.mask.slice() : null
      const transfer: Transferable[] = [values.buffer, ...(mask ? [mask.buffer] : []), ...(unreliable ? [unreliable] : [])]
      scope.postMessage({ id: req.id, type: 'decoded', width: map.width, height: map.height, values, mask, unreliable }, transfer)
    } else if (req.type === 'render') {
      const bitmap = engine.render(req.key, req.show)
      scope.postMessage({ id: req.id, type: 'rendered', bitmap }, [bitmap])
    } else {
      engine.release(req.key)
      scope.postMessage({ id: req.id, type: 'released' }, [])
    }
  } catch (e) {
    scope.postMessage({ id: req.id, type: 'error', message: e instanceof Error ? e.message : String(e) }, [])
  }
}

scope.onmessage = (e) => { void handle(e.data) }
