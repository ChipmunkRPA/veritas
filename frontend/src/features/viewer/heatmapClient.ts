/** Main-thread side of the heat layer worker. Requests carry an id and resolve with the matching response. Where module workers
 *  are unavailable (or the worker fails to start) the same `HeatEngine` runs in-thread. */
import { HeatEngine, type DecodedMap } from './heatmap'

export type HeatRequest =
  | { id: number; type: 'decode'; key: string; payload: Blob; mask: Blob | null }
  | { id: number; type: 'render'; key: string; show: number }
  | { id: number; type: 'release'; key: string }

export type HeatResponse =
  | { id: number; type: 'decoded'; width: number; height: number; values: Uint8Array; mask: Uint8Array | null; unreliable: ImageBitmap | null }
  | { id: number; type: 'rendered'; bitmap: ImageBitmap }
  | { id: number; type: 'released' }
  | { id: number; type: 'error'; message: string }

export interface LoadedMap extends DecodedMap {
  /** Unreliable cells as an alpha stencil at map resolution (null without a mask or unreliable cells). */
  unreliable: ImageBitmap | null
}

type Pending = { resolve: (r: HeatResponse) => void; reject: (e: Error) => void }

export class HeatmapClient {
  private worker: Worker | null = null
  private engine: HeatEngine | null = null
  private nextId = 1
  private readonly pending = new Map<number, Pending>()

  constructor(useWorker = typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined') {
    if (useWorker) {
      try {
        this.worker = new Worker(new URL('./heatmap.worker.ts', import.meta.url), { type: 'module' })
        this.worker.onmessage = (e: MessageEvent<HeatResponse>) => {
          const p = this.pending.get(e.data.id)
          if (!p) return
          this.pending.delete(e.data.id)
          if (e.data.type === 'error') p.reject(new Error(e.data.message))
          else p.resolve(e.data)
        }
        this.worker.onerror = (e) => {
          e.preventDefault()
          this.fail(new Error(`The heat layer worker stopped: ${e.message || 'script error'}.`))
        }
      } catch {
        this.worker = null
      }
    }
    if (!this.worker) this.engine = new HeatEngine()
  }

  private fail(error: Error): void {
    this.worker?.terminate()
    this.worker = null
    this.engine = new HeatEngine()
    for (const p of this.pending.values()) p.reject(error)
    this.pending.clear()
  }

  private send(req: HeatRequest): Promise<HeatResponse> {
    const worker = this.worker
    if (!worker) return Promise.reject(new Error('no worker'))
    return new Promise((resolve, reject) => {
      this.pending.set(req.id, { resolve, reject })
      worker.postMessage(req)
    })
  }

  async decode(key: string, payload: Blob, mask: Blob | null): Promise<LoadedMap> {
    if (this.engine) {
      const { map, unreliable } = await this.engine.decode(key, payload, mask)
      return { ...map, unreliable }
    }
    const r = await this.send({ id: this.nextId++, type: 'decode', key, payload, mask })
    if (r.type !== 'decoded') throw new Error(`unexpected ${r.type}`)
    return { width: r.width, height: r.height, values: r.values, mask: r.mask, unreliable: r.unreliable }
  }

  async render(key: string, show: number): Promise<ImageBitmap> {
    if (this.engine) return this.engine.render(key, show)
    const r = await this.send({ id: this.nextId++, type: 'render', key, show })
    if (r.type !== 'rendered') throw new Error(`unexpected ${r.type}`)
    return r.bitmap
  }

  dispose(): void {
    this.worker?.terminate()
    this.worker = null
    for (const p of this.pending.values()) p.reject(new Error('disposed'))
    this.pending.clear()
  }
}
