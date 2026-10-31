import { useMemo, useState, type KeyboardEvent } from 'react'
import { api, type Analysis } from '../../lib/api'
import { c2paWord, detectorName, fmtBytes, fmtExifDate, prob } from '../../lib/format'
import { cx } from '../../lib/cx'
import { Dl, Eyebrow } from '../../components/ui'
import { isGradcam } from './Plates'

type Overlay = 'none' | 'heatmap' | 'ela' | 'noise'

/** Stage height cap (§3 image slot: "source image, max-h 560"). Inner media = cap minus stage padding (24 + 40). */
const STAGE_MAX = 560
const MEDIA_MAX = STAGE_MAX - 64

/** Overlay toggles live in a 32 px control row above the stage, so they use the page tokens, not the stage tokens. */
const toggleCls = (on: boolean) => cx(
  'h-8 px-2.5 text-small font-medium rounded-sm transition-colors duration-fast',
  on ? 'bg-text text-surface' : 'text-text-muted hover:text-text hover:bg-surface-3',
)

function provenanceRows(meta: Record<string, unknown> | undefined) {
  if (!meta) return []
  const rows: { key: string; value: string }[] = []
  const get = (p: string) => p.split('.').reduce<unknown>((o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), meta)
  const pick = (label: string, fmt: (v: unknown) => string, ...paths: string[]) => {
    for (const p of paths) {
      const v = get(p)
      if (v !== undefined && v !== null && v !== '') { rows.push({ key: label, value: fmt(v) }); return }
    }
  }
  const str = (v: unknown) => (typeof v === 'object' ? JSON.stringify(v) : String(v))
  // C2PA: the engine emits `{present: bool, validation_state, claim_generator, …}`; never print the raw boolean.
  pick('C2PA', c2paWord, 'c2pa')
  if (!rows.length) rows.push({ key: 'C2PA', value: 'absent' })
  pick('EXIF Make', str, 'exif.Make', 'exif.make', 'Make')
  pick('EXIF Model', str, 'exif.Model', 'exif.model', 'Model')
  pick('Software', str, 'exif.Software', 'exif.software', 'Software')
  pick('Captured', fmtExifDate, 'exif.DateTimeOriginal', 'exif.DateTime', 'DateTimeOriginal')
  pick('Format', str, 'format')
  return rows
}

export default function ImageExhibit({ a, token = '' }: { a: Analysis; token?: string }) {
  const r = a.result!
  const img = r.image
  const faces = r.faces ?? []
  const arts = r.artifacts ?? []
  const gradcams = useMemo(() => arts.filter(isGradcam), [arts])
  const hasEla = arts.includes('ela.jpg'), hasNoise = arts.includes('noise_map.jpg')
  const [showFaces, setShowFaces] = useState(faces.length > 0)
  const [overlay, setOverlay] = useState<Overlay>('none')
  const [gradcam, setGradcam] = useState(gradcams[0] ?? '')
  const [opacity, setOpacity] = useState(70)
  const [hover, setHover] = useState<number | null>(null)
  const [loaded, setLoaded] = useState(false)

  const overlayName = overlay === 'heatmap' ? gradcam : overlay === 'ela' ? 'ela.jpg' : overlay === 'noise' ? 'noise_map.jpg' : ''
  const overlaySrc = overlayName ? api.artifactUrl(a.id, overlayName, token) : ''
  const setO = (o: Overlay) => setOverlay((cur) => (cur === o ? 'none' : o))
  const hasControls = faces.length > 0 || gradcams.length > 0 || hasEla || hasNoise

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement
    if (t.tagName === 'INPUT' || t.tagName === 'SELECT') return
    const k = e.key.toLowerCase()
    if (k === 'f' && faces.length) setShowFaces((s) => !s)
    else if (k === 'h' && gradcams.length) setO('heatmap')
    else if (k === 'e' && hasEla) { e.stopPropagation(); setO('ela') }
    else if (k === 'n' && hasNoise) setO('noise')
    else if (/^[0-9]$/.test(k) && overlay !== 'none') setOpacity(k === '0' ? 100 : Number(k) * 10)
    else return
    e.preventDefault()
  }

  const provenance = provenanceRows(r.metadata)
  const overlayWord = overlay === 'heatmap' ? 'Grad-CAM heatmap' : overlay === 'ela' ? 'ELA' : overlay === 'noise' ? 'noise residual' : null
  const caption = `Exhibit — Source image${faces.length ? ' with detected faces' : ''}${overlayWord ? `; ${overlayWord} overlay at ${opacity}%` : ''}.`

  return (
    <section aria-label="Exhibit" data-tour="exhibit" onKeyDown={onKey}>
      <div className="flex items-end justify-between gap-6 flex-wrap">
        <Eyebrow as="h2">Exhibit</Eyebrow>
        {hasControls && (
          <div className="flex items-center gap-1 h-8" role="group" aria-label="Overlays">
            {faces.length > 0 && <button type="button" aria-pressed={showFaces} className={toggleCls(showFaces)} onClick={() => setShowFaces((s) => !s)} title="Show detected face boxes (F)">Faces</button>}
            {gradcams.length > 0 && <button type="button" aria-pressed={overlay === 'heatmap'} className={toggleCls(overlay === 'heatmap')} onClick={() => setO('heatmap')} title="Stack the Grad-CAM heatmap over the source (H)">Heatmap</button>}
            {hasEla && <button type="button" aria-pressed={overlay === 'ela'} className={toggleCls(overlay === 'ela')} onClick={() => setO('ela')} title="Stack the Error Level Analysis map over the source (E)">ELA</button>}
            {hasNoise && <button type="button" aria-pressed={overlay === 'noise'} className={toggleCls(overlay === 'noise')} onClick={() => setO('noise')} title="Stack the noise-residual map over the source (N)">Noise</button>}
            {overlay === 'heatmap' && gradcams.length > 1 && (
              <select value={gradcam} onChange={(e) => setGradcam(e.target.value)} aria-label="Heatmap model" title="Heatmap model" className="h-8 px-2 ml-1 rounded-sm bg-surface text-text font-mono text-micro border border-graphic">
                {gradcams.map((g) => <option key={g} value={g}>{detectorName(g.replace(/^gradcam_/, '').replace(/\.\w+$/, ''))}</option>)}
              </select>
            )}
            {overlay !== 'none' && (
              <label className="flex items-center gap-2 pl-3 pr-1 text-micro font-mono text-text-muted" title="Overlay opacity (keys 1–9, 0 = 100%)">
                <span className="sr-only">Overlay opacity</span>
                <input type="range" min={0} max={100} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Overlay opacity" className="w-24 accent-text" />
                <span className="num w-8 text-right">{opacity}%</span>
              </label>
            )}
          </div>
        )}
      </div>

      <div className="stage mt-3 relative rounded-md bg-stage text-stage-text border border-stage-border flex items-center justify-center p-6 pb-10 outline-none"
        style={{ maxHeight: STAGE_MAX }} tabIndex={0} aria-label="Media stage. Keys: F faces, H heatmap, E ELA, N noise, 1 to 9 overlay opacity.">
        <div className="relative inline-block max-w-full">
          <img src={api.sourceUrl(a.id, token)} alt={a.filename} onLoad={() => setLoaded(true)}
            className="block max-w-full w-auto h-auto rounded-[2px]" style={{ maxHeight: MEDIA_MAX }} />
          {overlaySrc && (
            <img src={overlaySrc} alt="" aria-hidden className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-base rounded-[2px]"
              style={{ opacity: opacity / 100, objectFit: 'fill' }} />
          )}
          {img && showFaces && loaded && faces.map((f, i) => {
            const pAi = f.p_ai ?? null
            return (
              <button key={i} type="button" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}
                aria-label={`Face ${i + 1}, ${f.face.w} by ${f.face.h} pixels${pAi != null ? `, probability ${prob(pAi)}` : ''}`}
                className={cx('absolute border border-stage-text/85 transition-[border-color,box-shadow] duration-fast', hover === i && 'border-stage-text shadow-[0_0_0_1px_var(--color-stage-text)]')}
                style={{ left: `${(f.face.x / img.width) * 100}%`, top: `${(f.face.y / img.height) * 100}%`, width: `${(f.face.w / img.width) * 100}%`, height: `${(f.face.h / img.height) * 100}%`, boxShadow: hover === i ? undefined : '0 0 0 2px var(--color-stage-shadow)' }}>
                <span aria-hidden className="absolute top-0.5 left-0.5 rounded-sm bg-stage-material px-1.5 py-0.5 font-mono text-micro text-stage-text whitespace-nowrap num">face {i + 1}{pAi != null ? ` · ${prob(pAi)}` : ''}</span>
              </button>
            )
          })}
        </div>

        <p className="absolute bottom-3 right-4 font-mono text-caption text-stage-muted num">
          {img ? `${img.width} × ${img.height} · ${(img.format ?? a.content_type.split('/')[1] ?? '').toUpperCase()} · ` : ''}{fmtBytes(a.size_bytes)}
        </p>
      </div>
      <p className="mt-2 text-caption text-text-muted">{caption}</p>

      {(faces.length > 0 || provenance.length > 0) && (
        <div className="mt-8 grid gap-10 min-[960px]:grid-cols-2">
          {faces.length > 0 && (
            <div>
              <Eyebrow as="h3">Faces</Eyebrow>
              <ul className="mt-3 m-0 p-0 list-none space-y-4">
                {faces.map((f, i) => (
                  <li key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} className={cx('transition-colors duration-fast rounded-sm -mx-2 px-2 py-1', hover === i && 'bg-surface-3')}>
                    <div className="flex items-baseline justify-between gap-4 text-small">
                      <span className="text-text font-medium">Face {i + 1} <span className="font-mono text-mono text-text-muted num">{f.face.w}×{f.face.h} px</span></span>
                      {f.p_ai != null && <span className="text-text-muted">p(AI) <span className="font-mono text-mono text-text num">{prob(f.p_ai)}</span></span>}
                    </div>
                    <dl className="mt-1.5 m-0 grid grid-cols-[200px_1fr_48px] gap-x-3 gap-y-1 items-center">
                      {Object.entries(f.scores).map(([k, v]) => (
                        <div key={k} className="contents">
                          <dt className="text-small text-text-muted truncate" title={k}>{detectorName(k)}</dt>
                          <dd className="m-0 h-1.5 bg-border" style={{ borderRadius: 1 }} aria-hidden><span className="block h-full bg-text" style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%`, borderRadius: 1 }} /></dd>
                          <dd className="m-0 text-right font-mono text-mono text-text num">{prob(v)}</dd>
                        </div>
                      ))}
                    </dl>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {provenance.length > 0 && (
            <div>
              <Eyebrow as="h3">Provenance</Eyebrow>
              <Dl className="mt-3" keyWidth={112} rows={provenance.map((p) => ({ key: p.key, value: <span className={cx('break-words', /^(absent|present)/.test(p.value) ? 'text-small' : 'font-mono text-mono')}>{p.value}</span> }))} />
            </div>
          )}
        </div>
      )}
    </section>
  )
}
