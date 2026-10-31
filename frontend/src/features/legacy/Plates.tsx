import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'
import { cx } from '../../lib/cx'
import { Button, Eyebrow } from '../../components/ui'

/** Descriptive captions per artifact file name (second caption line). */
const CAPTIONS: Record<string, string> = {
  'faces.jpg': 'Detected faces; swap detectors ran on each box',
  'spectrum.png': 'Frequency spectrum and radial profile; periodic peaks or abnormal tails indicate synthesis',
  'ela.jpg': 'Error Level Analysis; regions compressed differently from their surroundings',
  'noise_map.jpg': 'Local noise-residual map; sensor noise should be spatially consistent',
  'timeline.png': 'Per-frame detector scores over time',
  'key_frames.jpg': 'Highest-scoring frames',
  'sentence_scores.png': 'Per-sentence AI likelihood',
}
const TITLES: Record<string, string> = {
  'faces.jpg': 'Detected faces', 'spectrum.png': 'Frequency spectrum', 'ela.jpg': 'Error Level Analysis', 'noise_map.jpg': 'Noise residual',
  'timeline.png': 'Frame timeline', 'key_frames.jpg': 'Key frames', 'sentence_scores.png': 'Sentence scores',
}

const modelOf = (name: string) => name.replace(/^gradcam_/, '').replace(/\.(jpe?g|png|webp)$/i, '')
export const isGradcam = (name: string) => name.startsWith('gradcam')
/** Per-frame key-frame crops emitted by the video pipeline (`keyframe_3.jpg`); `null` when the name is not one. */
export const keyFrameIndex = (name: string): number | null => { const m = name.match(/^keyframe_(\d+)\.(jpe?g|png|webp)$/i); return m ? Number(m[1]) : null }
export const isPoster = (name: string) => /^poster\.(jpe?g|png|webp)$/i.test(name)
/**
 * Whether an artifact belongs in the FIGURES grid. The exhibit already renders the poster and per-frame key frames, and
 * on the document page the manuscript supersedes the sentence-score bar chart (the PDF report keeps it).
 */
export const isFigure = (name: string, kind?: string | null) =>
  !isPoster(name) && keyFrameIndex(name) == null && !(kind === 'document' && name === 'sentence_scores.png')
export const captionFor = (name: string) => {
  const kf = keyFrameIndex(name)
  if (kf != null) return `Key frame ${kf}`
  if (isPoster(name)) return 'First frame of the source video'
  return CAPTIONS[name] ?? (isGradcam(name) ? `Grad-CAM: regions driving the ${modelOf(name)} decision` : name)
}
export const titleFor = (name: string) => {
  const kf = keyFrameIndex(name)
  if (kf != null) return `Key frame ${kf}`
  if (isPoster(name)) return 'Poster'
  return TITLES[name] ?? (isGradcam(name) ? `Grad-CAM (${modelOf(name)})` : name)
}

export interface PlateItem { name: string; src: string; title: string; caption: string; n?: number }

export const figLabel = (n?: number) => (n ? `Fig. ${n}` : 'Fig.')

/**
 * A numbered figure: hairline image, two-line caption. `onOpen` makes the image a button (lightbox).
 * Without `height` the image sits in a fixed 4:3 plate (`object-fit: contain`) so grids stay rectangular whatever the
 * artifact's aspect; with `height` (inline plates in the ledger) the image keeps its own aspect at that height.
 */
export function Plate({ item, onOpen, height, className, imgClassName }: { item: PlateItem; onOpen?: () => void; height?: number; className?: string; imgClassName?: string }) {
  const img = height ? (
    <img src={item.src} alt={`${item.title}. ${item.caption}`} loading="lazy"
      className={cx('rounded-md border border-border bg-surface-2 transition-[filter] duration-fast w-auto', onOpen && 'hover:brightness-[.97]', imgClassName)}
      style={{ height, maxWidth: '100%', objectFit: 'contain' }} />
  ) : (
    <span className={cx('block aspect-[4/3] w-full overflow-hidden rounded-md border border-border bg-surface-2 transition-[filter] duration-fast', onOpen && 'hover:brightness-[.97]')}>
      <img src={item.src} alt={`${item.title}. ${item.caption}`} loading="lazy" className={cx('w-full h-full object-contain', imgClassName)} />
    </span>
  )
  return (
    <figure className={cx('m-0 min-w-0', className)}>
      {onOpen ? <button type="button" onClick={onOpen} aria-label={`Open ${figLabel(item.n)} ${item.title} in viewer`} className="block w-full rounded-md text-left">{img}</button> : img}
      <figcaption className="mt-2">
        <span className="block text-small font-medium text-text"><span className="num">{figLabel(item.n)}</span> — {item.title}</span>
        <span className="block text-caption text-text-muted">{item.caption}</span>
      </figcaption>
    </figure>
  )
}

/** Full-viewport viewer: surface at 96%, image centred, caption bottom-left, Esc / click to close, arrows step. */
export function Lightbox({ items, index, onClose, onIndex }: { items: PlateItem[]; index: number; onClose: () => void; onIndex: (i: number) => void }) {
  const it = items[index]
  const closeRef = useRef<HTMLButtonElement>(null)
  const prev = useCallback(() => onIndex((index - 1 + items.length) % items.length), [index, items.length, onIndex])
  const next = useCallback(() => onIndex((index + 1) % items.length), [index, items.length, onIndex])
  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft') prev()
      else if (e.key === 'ArrowRight') next()
    }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prevOverflow }
  }, [onClose, prev, next])
  // Rudimentary focus trap: keep Tab within the dialog.
  const onTrap = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Tab') return
    const f = Array.from(e.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled])'))
    if (!f.length) return
    const first = f[0], last = f[f.length - 1]
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
  }
  if (!it) return null
  return (
    <div role="dialog" aria-modal="true" aria-label={`${figLabel(it.n)} ${it.title}`} onKeyDown={onTrap} onClick={onClose}
      className="fixed inset-0 z-50 bg-surface/96 animate-fade-in flex flex-col">
      <div className="flex items-center justify-end gap-1 p-4" onClick={(e) => e.stopPropagation()}>
        {items.length > 1 && (
          <>
            <Button variant="icon" aria-label="Previous figure" onClick={prev}><ChevronLeft size={16} strokeWidth={1.5} aria-hidden /></Button>
            <span className="font-mono text-mono text-text-muted num px-1">{index + 1} / {items.length}</span>
            <Button variant="icon" aria-label="Next figure" onClick={next}><ChevronRight size={16} strokeWidth={1.5} aria-hidden /></Button>
          </>
        )}
        <Button ref={closeRef} variant="icon" aria-label="Close viewer" onClick={onClose}><X size={16} strokeWidth={1.5} aria-hidden /></Button>
      </div>
      <div className="flex-1 min-h-0 flex items-center justify-center px-8">
        <img src={it.src} alt={`${it.title}. ${it.caption}`} className="max-h-full max-w-full rounded-md border border-border object-contain" onClick={(e) => e.stopPropagation()} />
      </div>
      <figcaption className="p-6 max-w-[68ch]" onClick={(e) => e.stopPropagation()}>
        <span className="block text-small font-medium text-text"><span className="num">{figLabel(it.n)}</span> — {it.title}</span>
        <span className="block text-caption text-text-muted">{it.caption}</span>
      </figcaption>
    </div>
  )
}

/** Hook: manage lightbox state for a list of items. */
export function useLightbox(items: PlateItem[]) {
  const [index, setIndex] = useState<number | null>(null)
  const open = useCallback((i: number) => setIndex(i), [])
  const close = useCallback(() => setIndex(null), [])
  const node = index != null ? <Lightbox items={items} index={index} onClose={close} onIndex={setIndex} /> : null
  return { open, close, node }
}

/** FIGURES section grid: auto-fill minmax(320px, 1fr), gap 32 x 40; no cards. */
export function PlateGrid({ items, eyebrow = 'Figures' }: { items: PlateItem[]; eyebrow?: string }) {
  const lb = useLightbox(items)
  if (!items.length) return null
  return (
    <section aria-label={eyebrow} data-tour="figures">
      <Eyebrow as="h2">{eyebrow}</Eyebrow>
      <div className="mt-4 grid gap-x-8 gap-y-10" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' }}>
        {items.map((it, i) => <Plate key={it.name} item={it} onOpen={() => lb.open(i)} />)}
      </div>
      {lb.node}
    </section>
  )
}
