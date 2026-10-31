import { useEffect, useRef, useState } from 'react'
import type { Kind } from '../lib/api'
import { MediaIcon } from './MediaTypes'

/** A local preview of the selected source; it never presents analysis or generated evidence. */
export default function SourcePreview({ file, kind }: { file: File; kind: Kind }) {
  const [url, setUrl] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  // React clears element refs before passive cleanups run, so keep the last media element here.
  const media = useRef<HTMLMediaElement | null>(null)
  const keep = (el: HTMLMediaElement | null) => { if (el) media.current = el }
  useEffect(() => {
    setFailed(false)
    if (kind === 'document') { setUrl(null); return }
    const next = URL.createObjectURL(file)
    setUrl(next)
    return () => {
      // A detached video/audio element keeps range-fetching its blob; revoking first logs ERR_FILE_NOT_FOUND.
      const el = media.current
      if (el && el.getAttribute('src') === next) {
        el.pause()
        el.removeAttribute('src')
        el.load()
      }
      URL.revokeObjectURL(next)
    }
  }, [file, kind])

  if (kind === 'document') return null
  return (
    <figure className="m-0 mt-3 overflow-hidden rounded-md border border-border bg-surface-2" data-testid="source-preview">
      <div className="flex items-center justify-center min-h-24 p-4">
        {failed ? <p className="m-0 text-small text-text-muted flex items-center gap-2"><MediaIcon kind={kind} />This browser cannot preview this format. You can still submit the file.</p>
          : !url ? <span className="text-small text-text-muted">Preparing preview…</span>
            : kind === 'image' ? <img src={url} alt={`Selected source: ${file.name}`} className="max-h-64 max-w-full object-contain" onError={() => setFailed(true)} />
              : kind === 'video' ? <video ref={keep} src={url} aria-label={`Selected source: ${file.name}`} controls playsInline preload="metadata" className="max-h-64 max-w-full" onError={() => setFailed(true)} />
                : <audio ref={keep} src={url} aria-label={`Selected source: ${file.name}`} controls preload="metadata" className="w-full" onError={() => setFailed(true)} />}
      </div>
      <figcaption className="border-t border-border px-4 py-2 text-caption text-text-muted">Source preview · no analysis has run</figcaption>
    </figure>
  )
}
