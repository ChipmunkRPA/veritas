import { useRef, type KeyboardEvent, type PointerEvent } from 'react'

/** Swipe divider (UX §5.9 "Compare modes"): source left of the line, lenses right. Draggable, and a focusable slider moved with
 *  ←/→ at 1 % (Shift 10 %), Home and End (WCAG 2.5.7). */
export default function CompareSlider({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const dragging = useRef(false)
  const clamp = (v: number) => Math.min(100, Math.max(0, Math.round(v)))
  const fromPointer = (e: PointerEvent<HTMLDivElement>) => {
    const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect()
    onChange(clamp(((e.clientX - rect.left) / rect.width) * 100))
  }
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 10 : 1
    const next = e.key === 'ArrowLeft' ? value - step : e.key === 'ArrowRight' ? value + step : e.key === 'Home' ? 0 : e.key === 'End' ? 100 : null
    if (next === null) return
    e.preventDefault()
    e.stopPropagation()
    onChange(clamp(next))
  }
  return (
    <div data-testid="swipe-divider" role="slider" tabIndex={0} aria-label="Swipe divider" aria-orientation="horizontal"
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={value} aria-valuetext={`${value} % source, ${100 - value} % lens`}
      onKeyDown={onKey}
      onPointerDown={(e) => { e.stopPropagation(); dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId) }}
      onPointerMove={(e) => { if (dragging.current) { e.stopPropagation(); fromPointer(e) } }}
      onPointerUp={(e) => { dragging.current = false; e.currentTarget.releasePointerCapture(e.pointerId) }}
      className="absolute top-0 bottom-0 w-6 -ml-3 cursor-ew-resize flex justify-center outline-none group"
      style={{ left: `${value}%` }}>
      <span aria-hidden className="block w-px h-full bg-stage-text" style={{ boxShadow: '0 0 0 1px var(--color-lens-halo)' }} />
      <span aria-hidden className="absolute top-1/2 -translate-y-1/2 h-8 w-3 rounded-sm border border-stage-text bg-stage-material group-focus-visible:border-2" />
    </div>
  )
}
