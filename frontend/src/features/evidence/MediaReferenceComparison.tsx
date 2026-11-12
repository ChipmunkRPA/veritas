import { useQuery } from '@tanstack/react-query'
import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { Button } from '../../components/ui'
import { authHeaders, unwrap } from '../../lib/api/client'
import { saveBlob } from '../../lib/api/reports'

type Kind = 'audio' | 'video'
type Side = 'earlier' | 'submitted'
const sides = ['earlier', 'submitted'] as const
const names = { earlier: 'Earlier clip', submitted: 'Analyzed clip' }
type Asset = { file: string; media_type: 'image/png' | 'audio/wav'; sha256: string }
type Extent = { id: string; start_s: number; end_s: number | null; last_s?: number }
type PictureRange = Extent & {
  strongest_frame: { changed_pixels: number; difference_bounds: { x: number; y: number; w: number; h: number } }
  previews: Record<string, { time_s: number; assets: Record<Side, Asset> }>
}
type Waveform = { low: number[]; high: number[] }
type PictureDimensions = { width: number; height: number }
type AudioRange = Extent & {
  changed_samples: number
  start_sample?: number
  end_sample?: number
  difference?: { peak: number; rms: number }
  preview?: { end_s: number; waveforms: Record<Side, Waveform>; assets: Record<Side, Asset>; difference_waveform?: Waveform }
}
type Comparison = {
  status: string
  interpretation: string
  audio?: { status: string; changed_samples?: number; submitted?: { samples: number }; ranges?: AudioRange[] }
} & ({
  schema: 'chipmunk.video-reference-comparison/1'
  submitted: PictureDimensions
  picture?: { matching_frames: number; changed_frames: number; ranges: PictureRange[] }
} | {
  schema: 'chipmunk.audio-reference-comparison/1'
  picture?: never
})
type Result = { comparison: Comparison; assets: Record<string, string>; pdf?: string | null }

const unassessed: Record<string, string> = {
  comparison_limit: 'At least one clip exceeds the picture comparison limits.',
  unverified_picture_clock: 'The comparison could not verify every picture timestamp.',
  picture_geometry_differs: 'The clips have different display dimensions.',
  picture_timelines_differ: 'The clips have different relative picture timestamps.',
  no_stream: 'Neither clip has a soundtrack.',
  audio_unassessed: 'Both soundtracks need a decodable continuous sample clock.',
  audio_timelines_differ: 'The soundtracks have different start times or sample counts.',
}

const audioUnassessed: Record<string, string> = {
  comparison_limit: 'At least one recording exceeds 30 seconds of decoded audio.',
  no_stream: 'The files have no readable audio track.',
  audio_unassessed: 'Both recordings need a readable, continuous audio sample clock.',
  audio_timelines_differ: 'The recordings have different start times or decoded sample counts.',
}

function route(analysisId: string, kind: Kind) {
  return '/api/v2/analyses/' + encodeURIComponent(analysisId) + '/' + kind + '-reference-comparison'
}

async function request<T>(url: string, init: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  const data: unknown = await response.json().catch(() => null)
  return unwrap(Promise.resolve({ response, data: response.ok ? data as T : undefined, error: response.ok ? undefined : data }))
}

function assetBody(assets: Result['assets'], asset: Asset, type: Asset['media_type']) {
  if (asset.media_type !== type || !Object.prototype.hasOwnProperty.call(assets, asset.file)) return undefined
  const data = assets[asset.file]
  const prefix = type === 'image/png' ? 'iVBORw0KGgo' : 'UklGR'
  return data.startsWith(prefix) && /^[A-Za-z0-9+/]+={0,2}$/.test(data) ? data : undefined
}

function interval(span: Extent) {
  return span.end_s === null ? span.start_s.toFixed(3) + ' s to final picture at ' + span.last_s?.toFixed(3) + ' s'
    : span.start_s.toFixed(3) + '–' + span.end_s.toFixed(3) + ' s'
}

function ClipWaveform({ wave, maximum, name }: { wave: Waveform; maximum: number; name: string }) {
  const lines = wave.low.map((value, i) => {
    const x = (i + .5) / wave.low.length * 300
    return 'M' + x + ',' + (30 - wave.high[i] / maximum * 27) + 'V' + (30 - value / maximum * 27)
  }).join('')
  return <svg viewBox="0 0 300 60" className="w-full max-w-80 rounded border border-border bg-surface-2" role="img"
    aria-label={name + ' waveform'}>
    <path d="M0,30H300" className="stroke-border" /><path d={lines} className="fill-none stroke-accent" />
  </svg>
}

function Playback({ body, name }: { body?: string; name: string }) {
  const [url, setUrl] = useState<string>()
  useEffect(() => {
    if (!body) { setUrl(undefined); return }
    const next = URL.createObjectURL(new Blob([Uint8Array.from(atob(body), c => c.charCodeAt(0))], { type: 'audio/wav' }))
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [body])
  return url ? <audio className="mt-1 block w-full max-w-80" controls preload="none" src={url} aria-label={name} /> : null
}

function PictureEvidence({ span, assets, dimensions }: { span: PictureRange; assets: Result['assets']; dimensions: PictureDimensions }) {
  const frame = span.strongest_frame
  const box = frame.difference_bounds
  return <details open={Object.keys(span.previews).length > 0} className="border-t border-border pt-3">
    <summary className="cursor-pointer font-medium">{span.id} · {interval(span)}</summary>
    <p className="mt-2 mb-0 text-small text-text-muted">{frame.changed_pixels.toLocaleString()} differing pixels at the strongest change, within x {box.x}, y {box.y}, {box.w} × {box.h} px.</p>
    {['before', 'during', 'after'].filter(position => span.previews[position]).map(position => {
      const preview = span.previews[position]
      return <div key={position} className="mt-3">
        <p className="mt-0 mb-2 text-small font-medium capitalize">{position} · {preview.time_s.toFixed(3)} s</p>
        <div className="grid grid-cols-2 gap-3">
          {sides.map(side => {
            const body = assetBody(assets, preview.assets[side], 'image/png')
            return <figure key={side} className="m-0">
              {body && <div className="relative inline-block max-w-full">
                <img className="block max-h-56 max-w-full rounded" loading="lazy"
                  src={'data:image/png;base64,' + body} alt={names[side] + ' at ' + preview.time_s.toFixed(3) + ' seconds'} />
                {position === 'during' && <span className="pointer-events-none absolute border-2 border-amber-400"
                  aria-hidden style={{ left: 100 * box.x / dimensions.width + '%', top: 100 * box.y / dimensions.height + '%',
                    width: 100 * box.w / dimensions.width + '%', height: 100 * box.h / dimensions.height + '%' }} />}
              </div>}
              <figcaption className="mt-1 text-caption text-text-muted">{names[side]}</figcaption>
            </figure>
          })}
        </div>
      </div>
    })}
  </details>
}

function SoundEvidence({ span, assets }: { span: AudioRange; assets: Result['assets'] }) {
  const preview = span.preview
  const residual = preview?.difference_waveform
  const waves = preview ? Object.values(preview.waveforms) : []
  const peak = Math.max(.01, ...waves.flatMap(wave => [...wave.low, ...wave.high].map(Math.abs)))
  const residualScale = residual ? Math.max(1 / 32768, ...[...residual.low, ...residual.high].map(Math.abs)) : 1
  return <details open={!!preview} className="border-t border-border pt-3">
    <summary className="cursor-pointer font-medium">{span.id} · {interval(span)}</summary>
    <p className="mt-2 mb-0 text-small text-text-muted">{span.changed_samples.toLocaleString()} differing samples in this range.</p>
    {span.difference && <p className="mt-2 mb-0 text-caption text-text-muted">Full range difference: peak {span.difference.peak.toFixed(6)}, RMS {span.difference.rms.toFixed(6)} of full scale.</p>}
    {preview && <>
      <p className="mt-2 mb-0 text-caption text-text-muted">Waveforms and playback cover {span.start_s.toFixed(3)}–{preview.end_s.toFixed(3)} s.</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {sides.map(side => <div key={side} className="min-w-0">
          <p className="mt-0 mb-1 text-small font-medium">{names[side]}</p>
          <ClipWaveform wave={preview.waveforms[side]} maximum={peak} name={names[side]} />
          <Playback body={assetBody(assets, preview.assets[side], 'audio/wav')} name={'Play ' + names[side].toLowerCase()} />
        </div>)}
      </div>
      {residual && <div className="mt-3">
        <p className="mt-0 mb-1 text-small font-medium">Difference waveform</p>
        <ClipWaveform wave={residual} maximum={residualScale} name="Difference" />
        <p className="mt-1 mb-0 text-caption text-text-muted">Analyzed minus earlier samples. This plot uses a separate scale of ±{residualScale.toFixed(6)}.</p>
      </div>}
    </>}
  </details>
}

function ResultView({ result }: { result: Result }) {
  const { comparison, assets, pdf } = result
  const { picture, audio } = comparison
  const isAudio = comparison.schema === 'chipmunk.audio-reference-comparison/1'
  const downloadName = 'chipmunk-verify-' + (isAudio ? 'audio' : 'video') + '-comparison'
  const notes = isAudio ? audioUnassessed : unassessed
  const firstAudioRange = audio?.ranges?.[0]
  const wholeRecording = audio?.ranges?.length === 1 && firstAudioRange?.start_sample === 0
    && firstAudioRange.end_sample === audio.submitted?.samples
  return <div className="mt-5 space-y-4 border-t border-border pt-4">
    <div className="flex flex-wrap gap-2">
      <Button onClick={() => saveBlob(new Blob([JSON.stringify({ comparison, assets }, null, 2) + '\n'],
        { type: 'application/json' }), downloadName + '.json')}>Download JSON evidence</Button>
      {pdf && <Button onClick={() => saveBlob(new Blob([Uint8Array.from(atob(pdf), c => c.charCodeAt(0))],
        { type: 'application/pdf' }), downloadName + '.pdf')}>Download PDF</Button>}
    </div>
    <p className="m-0 text-small text-text-muted">{comparison.interpretation}</p>
    {comparison.status !== 'measured' && <p className="m-0 text-small">{notes[comparison.status] ?? 'The files could not be compared.'}</p>}
    {!isAudio && picture && <>
      <h3 className="m-0 text-small font-medium">Pictures</h3>
      <p className="m-0 text-small">{picture.matching_frames} exact matches and {picture.changed_frames} changed decoded pictures.</p>
      {!picture.matching_frames && <p className="m-0 text-small text-text-muted">Every picture differs. Re-encoding can produce this pattern.</p>}
      {picture.ranges.map(span => <PictureEvidence key={span.id} span={span} assets={assets} dimensions={comparison.submitted} />)}
    </>}
    {audio && (!isAudio || comparison.status === 'measured') && <>
      <h3 className="m-0 text-small font-medium">{isAudio ? 'Decoded audio' : 'Soundtrack'}</h3>
      <p className="m-0 text-small">{audio.status === 'measured'
        ? (audio.changed_samples ?? 0).toLocaleString() + ' decoded mono samples differ.'
        : notes[audio.status] ?? 'Soundtrack comparison unavailable.'}</p>
      {isAudio && wholeRecording && <p className="m-0 text-caption text-text-muted">Differences span the whole recording. Encoding or gain changes can cause this.</p>}
      {audio.ranges?.map(span => <SoundEvidence key={span.id} span={span} assets={assets} />)}
    </>}
    {comparison.status === 'measured' && <p className="m-0 text-caption text-text-muted">{isAudio
      ? 'Times start at the first decoded sample. Ranges group adjacent 20 ms bins containing differing samples.'
      : 'Times are relative to the first picture. Soundtrack ranges group 20 ms bins containing different samples.'}</p>}

  </div>
}

function MediaReferenceForm({ analysisId, kind }: { analysisId: string; kind: Kind }) {
  const heading = useId()
  const input = useId()
  const [file, setFile] = useState<File | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<Result | null>(null)
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!file || active.current) return
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    setResult(null)
    setError('')
    const body = new FormData()
    body.append('earlier', file)
    body.append('include_pdf', 'true')
    try {
      const response = await request<Result>(route(analysisId, kind), { method: 'POST', headers: authHeaders('POST'), body, signal: controller.signal })
      if (response?.comparison?.schema !== 'chipmunk.' + kind + '-reference-comparison/1') throw new Error('The comparison response is incomplete.')
      if (!controller.signal.aborted) setResult(response)
    } catch (error) {
      if (!controller.signal.aborted) setError(error instanceof Error ? error.message : 'Could not compare the files.')
    } finally {
      if (!controller.signal.aborted && active.current === controller) {
        active.current = null
        setPending(false)
      }
    }
  }

  return <section aria-labelledby={heading} className="rounded-md border border-border p-4">
    <h2 id={heading} className="m-0 text-body font-medium">Compare with an earlier {kind === 'audio' ? 'audio recording' : 'video'}</h2>
    <p className="mt-2 mb-0 text-small text-text-muted">{kind === 'audio'
      ? 'Choose an earlier recording with the same start time and sample count. Each file can have up to 30 seconds of decoded audio and 48 MiB.'
      : 'Choose an earlier copy with matching picture dimensions and timing. Each clip can have up to 12 seconds of picture timestamps, 360 frames, and 48 MiB.'}</p>
    <form onSubmit={event => void submit(event)} className="mt-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0">
        <label htmlFor={input} className="mb-1 block text-small font-medium">Earlier {kind}</label>
        <input id={input} type="file" accept={kind === 'audio' ? 'audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus' : 'video/*,.mov,.mkv,.mp4,.webm'} className="block max-w-full text-small" disabled={pending}
          onChange={event => { setFile(event.currentTarget.files?.[0] ?? null); setResult(null); setError('') }} />
      </div>
      <Button type="submit" variant="primary" disabled={!file || pending}>{pending ? 'Comparing…' : kind === 'audio' ? 'Compare recordings' : 'Compare videos'}</Button>
    </form>
    {pending && <p role="status" className="mt-3 mb-0 text-small text-text-muted">{kind === 'audio' ? 'Comparing decoded audio…' : 'Comparing pictures and soundtracks…'}</p>}
    {error && <p role="alert" className="mt-3 mb-0 text-small text-danger">{error}</p>}
    {result && <ResultView result={result} />}
  </section>
}

export default function MediaReferenceComparison({ analysisId, kind }: { analysisId: string; kind: Kind }) {
  const availability = useQuery({ queryKey: ['analysis', analysisId, kind + '-reference-comparison'],
    queryFn: ({ signal }) => request<{ available: boolean }>(route(analysisId, kind), { headers: authHeaders(), signal }),
    refetchOnMount: 'always', refetchOnWindowFocus: false, retry: false })
  return availability.isFetchedAfterMount && availability.data?.available && !availability.isError
    ? <MediaReferenceForm key={kind + analysisId} analysisId={analysisId} kind={kind} /> : null
}
