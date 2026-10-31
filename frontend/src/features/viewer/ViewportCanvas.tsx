import { useEffect, useRef } from 'react'
import type { Size, View } from '../../lib/coords'
import { cx } from '../../lib/cx'
import { drawScene, type Scene } from './scene'

/** One raster surface of the viewer (ARCHITECTURE §7.4): a canvas of `viewport × devicePixelRatio` device px that redraws the scene
 *  after every render, coalesced into one animation frame. `onDrawn` reports whether the source was complete in that frame. */
export default function ViewportCanvas({ scene, view, viewport, lens, lensFromX = null, grid = true, onDrawn, className, testId }: {
  scene: Scene
  view: View | null
  viewport: Size
  lens: boolean
  lensFromX?: number | null
  grid?: boolean
  onDrawn?: (r: { baseComplete: boolean }) => void
  className?: string
  testId?: string
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const frame = useRef(0)
  const props = useRef({ scene, view, viewport, lens, lensFromX, grid, onDrawn })
  props.current = { scene, view, viewport, lens, lensFromX, grid, onDrawn }

  useEffect(() => {
    if (frame.current) return
    frame.current = requestAnimationFrame(() => {
      frame.current = 0
      const c = ref.current
      const p = props.current
      if (!c || !p.view || p.viewport.width <= 0 || p.viewport.height <= 0) return
      const dpr = window.devicePixelRatio || 1
      const w = Math.round(p.viewport.width * dpr)
      const h = Math.round(p.viewport.height * dpr)
      if (c.width !== w) c.width = w
      if (c.height !== h) c.height = h
      const ctx = c.getContext('2d')
      if (!ctx) return
      try {
        const r = drawScene(ctx, p.scene, p.view, p.viewport, { dpr, lens: p.lens, lensFromX: p.lensFromX, grid: p.grid })
        c.dataset.frames = String(Number(c.dataset.frames ?? 0) + 1)
        c.dataset.base = `${p.scene.base?.kind ?? 'none'}:${r.baseComplete ? 'complete' : 'loading'}`
        p.onDrawn?.(r)
      } catch (e) {
        console.error('evidence viewer: drawing the scene failed', e)
      }
    })
  })

  useEffect(() => () => {
    if (frame.current) cancelAnimationFrame(frame.current)
    frame.current = 0
  }, [])

  return <canvas ref={ref} data-testid={testId} aria-hidden className={cx('absolute left-0 top-0 block', className)} style={{ width: viewport.width, height: viewport.height }} />
}
