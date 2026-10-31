import { toOriginal, type Point, type View } from '../../lib/coords'
import type { Scene } from './scene'
import ViewportCanvas from './ViewportCanvas'

export const LOUPE_PX = 200
export const LOUPE_FACTORS = [4, 8, 16] as const

/** Loupe (UX §5.9, L): a 200 px circle at 4× the current zoom (8×, 16× with `[` `]`), centred on the pointer or the keyboard
 *  crosshair, with a hairline ring and halo. The same scene draws inside, so map lenses show at the same opacity. */
export default function Loupe({ scene, view, at, factor, lens }: { scene: Scene; view: View; at: Point; factor: number; lens: boolean }) {
  const o = toOriginal(view, at)
  const scale = view.scale * factor
  const half = LOUPE_PX / 2
  const loupeView: View = { scale, tx: half - o.x * scale, ty: half - o.y * scale }
  return (
    <div data-testid="loupe" aria-hidden className="absolute pointer-events-none rounded-full overflow-hidden bg-stage"
      style={{ left: at.x - half, top: at.y - half, width: LOUPE_PX, height: LOUPE_PX, boxShadow: '0 0 0 1px var(--color-lens-ink), 0 0 0 3px var(--color-lens-halo)' }}>
      <ViewportCanvas scene={scene} view={loupeView} viewport={{ width: LOUPE_PX, height: LOUPE_PX }} lens={lens} grid={false} />
      <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 block w-3 h-px bg-lens-ink" />
      <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 block w-px h-3 bg-lens-ink" />
      <span className="absolute bottom-3 left-1/2 -translate-x-1/2 px-1 rounded-xs bg-stage-material font-mono text-micro text-stage-text num">{factor}×</span>
    </div>
  )
}
