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
type Tile = { x: number; y: number; w: number; h: number }
type PicturePreviews = Record<string, { time_s: number; assets: Record<Side, Asset>; crop_assets?: Record<Side, Asset> }>
type PictureRange = Extent & {
  strongest_frame: { changed_pixels: number; difference_bounds: Tile }
  previews: PicturePreviews
}
type LocalPictureRange = Extent & { tile: Tile; previews: PicturePreviews }
type LocalComparison = { scope: string; summary: string; range?: LocalPictureRange }
type Waveform = { low: number[]; high: number[] }
type PictureDimensions = { width: number; height: number }
type TimingSide = {
  frame_index: number
  picture_time_s: number
  pcm_start_s: number
  pcm_end_s: number
  crop: Asset
  speech: Asset
  waveform: Waveform
}
type TimingComparison = {
  status: 'timing_difference' | 'no_large_timing_difference' | 'unassessed'
  summary: string
  scope: string
  offset_ms?: number
  event_range_s?: number[]
} & Partial<Record<Side, TimingSide>>
type SoundtrackSide = { start_s: number; end_s: number; audio: Asset; waveform: Waveform }
type SoundtrackTiming = {
  summary: string
  scope: string
  range?: Record<Side, SoundtrackSide> & { picture: { frame_index: number; time_s: number; asset: Asset } }
}
type AdditionalSoundtrack = {
  scope: string
  status: string
  stream_indices: Record<Side, number | null>
  inventory: Record<Side, { unassessed_stream_indices: number[] }>
  changed_samples?: number
  ranges?: AudioRange[]
}
type SourceRange = { start_pts: number; end_exclusive_pts: number }
type FileSampleRange = SourceRange & { start_sample: number; end_sample: number }
type FileSampleRanges = Record<Side, FileSampleRange>
type AudioTrimAlignment = {
  earlier_minus_submitted_samples: number
  compared_samples: number
  overlap: FileSampleRanges
}
type AudioRange = Extent & {
  changed_samples: number
  start_sample?: number
  end_sample?: number
  file_ranges?: FileSampleRanges
  source_ranges?: Record<Side, SourceRange>
  difference?: { peak: number; rms: number }
  preview?: { end_s: number; end_sample?: number; waveforms: Record<Side, Waveform>; assets: Record<Side, Asset>; difference_waveform?: Waveform }
}
type LevelAdjustedAudio = {
  gain_db: number
  changed_samples: number
  interpretation: string
  ranges: AudioRange[]
}
type Comparison = {
  status: string
  interpretation: string
  audio?: { status: string; changed_samples?: number; submitted?: { samples: number }; ranges?: AudioRange[]; trim_alignment?: AudioTrimAlignment }
} & ({
  schema: 'chipmunk.video-reference-comparison/1' | 'chipmunk.video-reference-comparison/2' | 'chipmunk.video-reference-comparison/3'
    | 'chipmunk.video-reference-comparison/4' | 'chipmunk.video-reference-comparison/5' | 'chipmunk.video-reference-comparison/6'
  submitted: PictureDimensions
  picture?: { matching_frames: number; changed_frames: number; ranges: PictureRange[]; local_comparison?: LocalComparison }
  timing?: TimingComparison
  soundtrack_alignment?: SoundtrackTiming
  additional_soundtrack?: AdditionalSoundtrack
  level_adjusted?: never
} | {
  schema: 'chipmunk.audio-reference-comparison/1' | 'chipmunk.audio-reference-comparison/2' | 'chipmunk.audio-reference-comparison/3'
    | 'chipmunk.audio-reference-comparison/4'
  picture?: never
  timing?: never
  soundtrack_alignment?: never
  additional_soundtrack?: never
  level_adjusted?: LevelAdjustedAudio
})
type Result = { comparison: Comparison; assets: Record<string, string>; pdf?: string | null }
type Availability = { available: boolean; timing_available?: boolean }
const sourceSeconds = new Intl.NumberFormat('en-US', { minimumFractionDigits: 3, maximumFractionDigits: 7, useGrouping: false })
const trimSampleRate = 16000

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

function fileTime(start: number, end: number) {
  return `${sourceSeconds.format(start / trimSampleRate)}–${sourceSeconds.format(end / trimSampleRate)} s`
}

function FileTimelines({ ranges }: { ranges: FileSampleRanges }) {
  return <dl className="mt-2 mb-0 grid gap-2 text-small sm:grid-cols-2">
    {sides.map(side => {
      const range = ranges[side]
      return <div key={side}>
        <dt className="font-medium">{names[side]}</dt>
        <dd className="m-0">{fileTime(range.start_sample, range.end_sample)}</dd>
        <dd className="m-0 text-caption text-text-muted">Samples [{range.start_sample}, {range.end_sample}) · source ticks [{range.start_pts}, {range.end_exclusive_pts})</dd>
      </div>
    })}
  </dl>
}

function ClipWaveform({ wave, maximum, name }: { wave: Waveform; maximum: number; name: string }) {
  const step = 300 / wave.low.length
  const lines = wave.low.map((value, i) => {
    const y = 30 - value / maximum * 27
    if (value === wave.high[i]) return `M${i * step},${y}h${step}`
    const x = (i + .5) * step
    return `M${x},${30 - wave.high[i] / maximum * 27}V${y}`
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

function PictureEvidence({ span, assets, dimensions }: { span: PictureRange | LocalPictureRange; assets: Result['assets']; dimensions: PictureDimensions }) {
  const box = 'tile' in span ? span.tile : span.strongest_frame.difference_bounds
  return <details open={Object.keys(span.previews).length > 0} className="border-t border-border pt-3">
    <summary className="cursor-pointer font-medium">{span.id} · {interval(span)}</summary>
    <p className="mt-2 mb-0 text-small text-text-muted">{'tile' in span ? 'Measured comparison area at' :
      `${span.strongest_frame.changed_pixels.toLocaleString()} differing pixels at the strongest change, within`} x {box.x}, y {box.y}, {box.w} × {box.h} px.</p>
    {['before', 'during', 'after'].filter(position => span.previews[position]).map(position => {
      const preview = span.previews[position]
      return <div key={position} className="mt-3">
        <p className="mt-0 mb-2 text-small font-medium capitalize">{position} · {preview.time_s.toFixed(3)} s</p>
        <div className="grid grid-cols-2 gap-3">
          {sides.map(side => {
            const body = assetBody(assets, preview.assets[side], 'image/png')
            const crop = preview.crop_assets && assetBody(assets, preview.crop_assets[side], 'image/png')
            return <figure key={side} className="m-0">
              {body && <div className="relative inline-block max-w-full">
                <img className="block max-h-56 max-w-full rounded" loading="lazy"
                  src={'data:image/png;base64,' + body} alt={names[side] + ' at ' + preview.time_s.toFixed(3) + ' seconds'} />
                {(position === 'during' || 'tile' in span) && <span className="pointer-events-none absolute border-2 border-amber-400"
                  aria-hidden style={{ left: 100 * box.x / dimensions.width + '%', top: 100 * box.y / dimensions.height + '%',
                    width: 100 * box.w / dimensions.width + '%', height: 100 * box.h / dimensions.height + '%' }} />}
              </div>}
              {crop && <img className="mt-2 block h-auto w-24 max-w-full rounded border border-border" loading="lazy"
                src={'data:image/png;base64,' + crop} alt={names[side] + ' comparison crop at ' + preview.time_s.toFixed(3) + ' seconds'} />}
              <figcaption className="mt-1 text-caption text-text-muted">{names[side]}</figcaption>
            </figure>
          })}
        </div>
      </div>
    })}
  </details>
}

function SoundEvidence({ span, assets, levelAdjusted = false, showSourceTicks = false }: {
  span: AudioRange; assets: Result['assets']; levelAdjusted?: boolean; showSourceTicks?: boolean
}) {
  const preview = span.preview
  const residual = preview?.difference_waveform
  const waves = preview ? Object.values(preview.waveforms) : []
  const peak = Math.max(.01, ...waves.flatMap(wave => [...wave.low, ...wave.high].map(Math.abs)))
  const residualScale = residual ? Math.max(1 / 32768, ...[...residual.low, ...residual.high].map(Math.abs)) : 1
  const fileRanges = span.file_ranges
  const sourceRanges = showSourceTicks ? span.source_ranges : undefined
  const previewSamples = preview?.end_sample != null && span.start_sample != null ? preview.end_sample - span.start_sample : null
  return <details open={!!preview} className="border-t border-border pt-3">
    <summary className="cursor-pointer font-medium">{span.id} · {fileRanges
      ? `${names.submitted} ${fileTime(fileRanges.submitted.start_sample, fileRanges.submitted.end_sample)}` : interval(span)}</summary>
    <p className="mt-2 mb-0 text-small text-text-muted">{span.changed_samples.toLocaleString()} {levelAdjusted
      ? 'samples in this range cannot be explained by the measured volume adjustment.' : 'differing samples in this range.'}</p>
    {fileRanges && <FileTimelines ranges={fileRanges} />}
    {sourceRanges && <p className="mt-2 mb-0 text-caption text-text-muted">
      Source ticks {sides.map(side => `${names[side]} [${sourceRanges[side].start_pts}, ${sourceRanges[side].end_exclusive_pts})`).join(' · ')}
    </p>}
    {span.difference && <p className="mt-2 mb-0 text-caption text-text-muted">{levelAdjusted ? 'Adjusted difference' : 'Full range difference'}: peak {span.difference.peak.toFixed(6)}, RMS {span.difference.rms.toFixed(6)} of full scale.</p>}
    {preview && <>
      {(!fileRanges || previewSamples == null) && <p className="mt-2 mb-0 text-caption text-text-muted">Waveforms and playback cover {span.start_s.toFixed(3)}–{preview.end_s.toFixed(3)} s.</p>}
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {sides.map(side => <div key={side} className="min-w-0">
          <p className="mt-0 mb-1 text-small font-medium">{names[side]}</p>
          {fileRanges && previewSamples != null && <p className="mt-0 mb-1 text-caption text-text-muted">Playback and waveform {fileTime(fileRanges[side].start_sample, fileRanges[side].start_sample + previewSamples)}</p>}
          <ClipWaveform wave={preview.waveforms[side]} maximum={peak} name={names[side] + (levelAdjusted ? ' at adjusted difference' : '')} />
          <Playback body={assetBody(assets, preview.assets[side], 'audio/wav')}
            name={'Play ' + names[side].toLowerCase() + (levelAdjusted ? ' at adjusted difference' : '')} />
        </div>)}
      </div>
      {residual && <div className="mt-3">
        <p className="mt-0 mb-1 text-small font-medium">{levelAdjusted ? 'Volume-adjusted difference waveform' : 'Difference waveform'}</p>
        <ClipWaveform wave={residual} maximum={residualScale} name={levelAdjusted ? 'Volume-adjusted difference' : 'Difference'} />
        <p className="mt-1 mb-0 text-caption text-text-muted">{levelAdjusted
          ? 'Analyzed minus volume-adjusted earlier samples.' : 'Analyzed minus earlier samples.'} This plot uses a separate scale of ±{residualScale.toFixed(6)}.</p>
      </div>}
    </>}
  </details>
}

function LevelAdjustedEvidence({ result, assets, aligned }: { result: LevelAdjustedAudio; assets: Result['assets']; aligned: boolean }) {
  const magnitude = Math.abs(result.gain_db)
  const gain = magnitude.toFixed(magnitude < .0005 ? 6 : 3)
  return <section className="space-y-2" aria-label="Differences after a volume adjustment">
    <h3 className="m-0 text-small font-medium">Differences after a volume adjustment</h3>
    <p className="m-0 text-small">The analyzed recording is {gain} dB {result.gain_db < 0 ? 'quieter' : 'louder'}
      {' '}{aligned ? 'in two supporting sections of the shared span.' : 'in the matching sections at both ends.'}</p>
    <p className="m-0 text-small">{result.changed_samples === 0 ? `The same volume adjustment explains ${aligned ? 'the shared span' : 'the entire recording'}.`
      : `${result.changed_samples.toLocaleString()} samples remain different after accounting for the volume adjustment.`}</p>
    {!aligned && <p className="m-0 text-caption text-text-muted">{result.interpretation}</p>}
    {result.ranges.map(span => <SoundEvidence key={span.id} span={span} assets={assets} levelAdjusted />)}
  </section>
}

function TimingEvidence({ timing, assets }: { timing: TimingComparison; assets: Result['assets'] }) {
  const matched = sides.flatMap(side => {
    const evidence = timing[side]
    return evidence ? [{ side, evidence }] : []
  })
  const peak = Math.max(.01, ...matched.flatMap(({ evidence }) => [...evidence.waveform.low, ...evidence.waveform.high].map(Math.abs)))
  const range = timing.event_range_s
  return <section className="space-y-2" aria-label="Mouth and speech timing">
    <h3 className="m-0 text-small font-medium">Mouth and speech timing</h3>
    <p className="m-0 text-small">{timing.summary}</p>
    {timing.status === 'timing_difference' && range?.length === 2 && <p className="m-0 text-small">
      Compared event times {sourceSeconds.format(range[0])}–{sourceSeconds.format(range[1])} s.
    </p>}
    {matched.length > 0 && <div className="grid gap-3 sm:grid-cols-2">
      {matched.map(({ side, evidence }) => {
        const crop = assetBody(assets, evidence.crop, 'image/png')
        const time = sourceSeconds.format(evidence.picture_time_s)
        return <div key={side} className="min-w-0 rounded-md border border-border p-3">
          <h4 className="m-0 text-small font-medium">{names[side]}</h4>
          {crop && <img className="mt-2 block h-auto max-h-40 max-w-full rounded border border-border" loading="lazy"
            src={'data:image/png;base64,' + crop} alt={`${names[side]} mouth crop at ${time} seconds`} />}
          <p className="mt-1 mb-2 text-caption text-text-muted">Frame {evidence.frame_index} at {time} s</p>
          <p className="mt-0 mb-1 text-caption text-text-muted">
            Speech {sourceSeconds.format(evidence.pcm_start_s)}–{sourceSeconds.format(evidence.pcm_end_s)} s
          </p>
          <ClipWaveform wave={evidence.waveform} maximum={peak} name={`${names[side]} timing speech`} />
          <Playback body={assetBody(assets, evidence.speech, 'audio/wav')} name={`Play ${names[side].toLowerCase()} timing speech`} />
        </div>
      })}
    </div>}
    <p className="m-0 text-caption text-text-muted">{timing.scope}</p>
  </section>
}

function SoundtrackTimingEvidence({ timing, assets }: { timing: SoundtrackTiming; assets: Result['assets'] }) {
  const span = timing.range
  const image = span && assetBody(assets, span.picture.asset, 'image/png')
  const maximum = span ? Math.max(.01, ...sides.flatMap(side => [...span[side].waveform.low, ...span[side].waveform.high].map(Math.abs))) : 1
  return <section className="space-y-2" aria-label="Local soundtrack timing">
    <h3 className="m-0 text-small font-medium">Local soundtrack timing</h3>
    <p className="m-0 text-small">{timing.summary}</p>
    {span && <>
      {image && <figure className="m-0 max-w-80 space-y-1">
        <img src={'data:image/png;base64,' + image} className="h-auto w-full rounded border border-border"
          alt={`Analyzed clip picture ${span.picture.frame_index} at ${sourceSeconds.format(span.picture.time_s)} seconds`} />
        <figcaption className="text-caption text-text-muted">Picture at {sourceSeconds.format(span.picture.time_s)} s in the analyzed clip.</figcaption>
      </figure>}
      <div className="grid gap-4 sm:grid-cols-2">{sides.map(side => {
        const row = span[side]
        return <div className="min-w-0 space-y-2" key={side}>
          <p className="m-0 text-small font-medium">{names[side]} · {sourceSeconds.format(row.start_s)}–{sourceSeconds.format(row.end_s)} s</p>
          <ClipWaveform wave={row.waveform} maximum={maximum} name={`${names[side]} matched soundtrack`} />
          <Playback body={assetBody(assets, row.audio, 'audio/wav')} name={`Play ${names[side].toLowerCase()} matched soundtrack`} />
        </div>
      })}</div>
    </>}
    <p className="m-0 text-caption text-text-muted">{timing.scope}</p>
  </section>
}

function ResultView({ result }: { result: Result }) {
  const { comparison, assets, pdf } = result
  const { picture, audio } = comparison
  const hasTrim = comparison.schema === 'chipmunk.audio-reference-comparison/3' || comparison.schema === 'chipmunk.audio-reference-comparison/4'
  const isAudio = hasTrim || comparison.schema === 'chipmunk.audio-reference-comparison/1' || comparison.schema === 'chipmunk.audio-reference-comparison/2'
  const alignment = hasTrim ? audio?.trim_alignment : undefined
  const downloadName = 'chipmunk-verify-' + (isAudio ? 'audio' : 'video') + '-comparison'
  const notes = isAudio ? audioUnassessed : unassessed
  const firstAudioRange = audio?.ranges?.[0]
  const wholeRecording = audio?.ranges?.length === 1 && firstAudioRange?.start_sample === 0
    && firstAudioRange.end_sample === (alignment?.compared_samples ?? audio.submitted?.samples)
  return <div className="mt-5 space-y-4 border-t border-border pt-4">
    <div className="flex flex-wrap gap-2">
      <Button onClick={() => saveBlob(new Blob([JSON.stringify({ comparison, assets }, null, 2) + '\n'],
        { type: 'application/json' }), downloadName + '.json')}>Download JSON evidence</Button>
      {pdf && <Button onClick={() => saveBlob(new Blob([Uint8Array.from(atob(pdf), c => c.charCodeAt(0))],
        { type: 'application/pdf' }), downloadName + '.pdf')}>Download PDF</Button>}
    </div>
    <p className="m-0 text-small text-text-muted">{comparison.interpretation}</p>
    {comparison.status !== 'measured' && <p className="m-0 text-small">{notes[comparison.status] ?? 'The files could not be compared.'}</p>}
    {comparison.timing && <TimingEvidence timing={comparison.timing} assets={assets} />}
    {comparison.soundtrack_alignment && <SoundtrackTimingEvidence timing={comparison.soundtrack_alignment} assets={assets} />}
    {picture && !comparison.soundtrack_alignment && (picture.changed_frames > 0 || audio?.status !== 'measured') &&
      <p className="m-0 text-caption text-text-muted">{picture.changed_frames > 0
        ? 'Local soundtrack timing was skipped because the decoded pictures differ.'
        : 'Local soundtrack timing requires two continuous soundtracks with matching start times and sample counts.'}</p>}
    {comparison.level_adjusted && !hasTrim && <LevelAdjustedEvidence result={comparison.level_adjusted} assets={assets} aligned={false} />}
    {!isAudio && picture && <>
      <h3 className="m-0 text-small font-medium">Pictures</h3>
      <p className="m-0 text-small">{picture.matching_frames} exact matches and {picture.changed_frames} changed decoded pictures.</p>
      {!picture.matching_frames && <p className="m-0 text-small text-text-muted">Every picture differs. Re-encoding can produce this pattern.</p>}
      {picture.local_comparison && <section className="space-y-2" aria-label="Local picture differences">
        <h4 className="m-0 text-small font-medium">Local picture differences</h4>
        <p className="m-0 text-small">{picture.local_comparison.summary}</p>
        <p className="m-0 text-caption text-text-muted">{picture.local_comparison.scope}</p>
        {picture.local_comparison.range && <PictureEvidence span={picture.local_comparison.range} assets={assets} dimensions={comparison.submitted} />}
      </section>}
      {picture.ranges.map(span => <PictureEvidence key={span.id} span={span} assets={assets} dimensions={comparison.submitted} />)}
    </>}
    {audio && (!isAudio || comparison.status === 'measured') && <>
      <h3 className="m-0 text-small font-medium">{isAudio ? 'Decoded audio' : comparison.additional_soundtrack ? 'First soundtrack' : 'Soundtrack'}</h3>
      {alignment && <section aria-label="Shared audio after alignment" className="rounded-md border border-border p-3">
        <h4 className="m-0 text-small font-medium">Shared audio after alignment</h4>
        <p className="mt-2 mb-0 text-small">Compared {alignment.compared_samples.toLocaleString()} samples across the two recordings.</p>
        <FileTimelines ranges={alignment.overlap} />
      </section>}
      {comparison.level_adjusted && hasTrim && <LevelAdjustedEvidence result={comparison.level_adjusted} assets={assets} aligned />}
      <p className="m-0 text-small">{alignment ? audio.changed_samples
        ? `${audio.changed_samples.toLocaleString()} decoded samples differ in the shared span.`
        : 'No decoded samples differ in the shared span.'
        : audio.status === 'measured'
        ? (audio.changed_samples ?? 0).toLocaleString() + ' decoded mono samples differ.'
        : notes[audio.status] ?? 'Soundtrack comparison unavailable.'}</p>
      {isAudio && wholeRecording && <p className="m-0 text-caption text-text-muted">Differences span {alignment ? 'the shared span' : 'the whole recording'}. Encoding or gain changes can cause this.</p>}
      {audio.ranges?.map(span => <SoundEvidence key={span.id} span={span} assets={assets} />)}
    </>}
    {comparison.additional_soundtrack && <SecondSoundtrack result={comparison.additional_soundtrack} assets={assets} />}
    {comparison.status === 'measured' && <p className="m-0 text-caption text-text-muted">{isAudio
      ? `${alignment ? 'Each timeline starts at its file’s first decoded sample. Ends are exclusive. ' : 'Times start at the first decoded sample. '}Ranges group adjacent 20 ms bins containing differing samples.`
      : 'Times are relative to the first picture. Soundtrack ranges group 20 ms bins containing different samples.'}</p>}

  </div>
}

function SecondSoundtrack({ result, assets }: { result: AdditionalSoundtrack; assets: Result['assets'] }) {
  const unavailable: Record<string, string> = { ...unassessed, missing_counterpart: 'The second soundtrack has no counterpart.',
    comparison_limit: 'At least one second soundtrack exceeds 12 seconds.' }
  const summary = result.status === 'measured' ? `${(result.changed_samples ?? 0).toLocaleString()} decoded mono samples differ.`
    : unavailable[result.status] ?? 'The second soundtracks could not be compared.'
  return <section aria-label="Second soundtrack comparison" className="space-y-2">
    <h3 className="m-0 text-small font-medium">Second soundtrack comparison</h3>
    <p className="m-0 text-small">{summary}</p>
    {sides.map(side => {
      const index = result.stream_indices[side]
      const count = result.inventory[side].unassessed_stream_indices.length
      return <p key={side} className="m-0 text-caption text-text-muted">{names[side]} {index === null ? 'has no second soundtrack.' : `uses source stream ${index}.`}
        {count > 0 && ` ${count} ${count === 1 ? 'track' : 'tracks'} beyond the second remained unassessed.`}</p>
    })}
    <p className="m-0 text-caption text-text-muted">{result.scope}</p>
    {result.ranges?.map(span => <SoundEvidence key={span.id} span={span} assets={assets} showSourceTicks />)}
  </section>
}

function MediaReferenceForm({ analysisId, kind, timingAvailable }: { analysisId: string; kind: Kind; timingAvailable: boolean }) {
  const heading = useId()
  const input = useId()
  const timingInput = useId()
  const [file, setFile] = useState<File | null>(null)
  const [timingEvent, setTimingEvent] = useState('')
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
    if (timingAvailable && timingEvent) body.append('timing_event_s', timingEvent)
    try {
      const response = await request<Result>(route(analysisId, kind), { method: 'POST', headers: authHeaders('POST'), body, signal: controller.signal })
      const schema = response?.comparison?.schema
      const accepted = kind === 'audio' ? [1, 2, 3, 4] : [1, 2, 3, 4, 5, 6]
      if (!accepted.some(version => schema === `chipmunk.${kind}-reference-comparison/${version}`)) {
        throw new Error('The comparison response is incomplete.')
      }
      if (schema === 'chipmunk.audio-reference-comparison/3' && !response.comparison.audio?.trim_alignment) {
        throw new Error('The comparison response is missing its shared audio range.')
      }
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
      ? 'Choose an earlier copy of the recording. Each file can have up to 30 seconds of decoded audio and 48 MiB.'
      : 'Choose an earlier copy with matching picture dimensions and timing. Each clip can have up to 12 seconds of picture timestamps, 360 frames, and 48 MiB.'}</p>
    {kind === 'audio' && <p className="mt-2 mb-0 text-caption text-text-muted">
      Matching audio timelines can be compared directly. For lossless 16 kHz mono recordings, the comparison can also account for a constant volume change and up to 250 ms of trim per file when the samples support both.
    </p>}
    {kind === 'video' && <p className="mt-2 mb-0 text-caption text-text-muted">
      When present in both clips, a second soundtrack is compared separately. Local timing on the first soundtrack requires identical decoded pictures and matching soundtrack clocks.
    </p>}
    <form onSubmit={event => void submit(event)} className="mt-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0">
        <label htmlFor={input} className="mb-1 block text-small font-medium">Earlier {kind}</label>
        <input id={input} type="file" accept={kind === 'audio' ? 'audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus' : 'video/*,.mov,.mkv,.mp4,.webm' + (timingAvailable ? ',.mpg,.mpeg' : '')} className="block max-w-full text-small" disabled={pending}
          onChange={event => { setFile(event.currentTarget.files?.[0] ?? null); setTimingEvent(''); setResult(null); setError('') }} />
      </div>
      {timingAvailable && <div className="min-w-0 basis-full">
        <label htmlFor={timingInput} className="mb-1 block text-small font-medium">Earlier mouth closure (seconds)</label>
        <input id={timingInput} type="number" min="0" step="0.001" aria-describedby={timingInput + '-hint'}
          value={timingEvent} disabled={pending || !file} className="w-32 rounded border border-border bg-surface px-2 py-1 text-small"
          onChange={event => { setTimingEvent(event.currentTarget.value); setResult(null); setError('') }} />
        <p id={timingInput + '-hint'} className="mt-1 mb-0 text-caption text-text-muted">
          Optional. Mark a moment in the earlier clip when the lips close just before reopening. This check supports matching 360 × 288 clips with 75 pictures at 25 fps and continuous audio.
        </p>
      </div>}
      <Button type="submit" variant="primary" disabled={!file || pending}>{pending ? 'Comparing…' : kind === 'audio' ? 'Compare recordings' : 'Compare videos'}</Button>
    </form>
    {pending && <p role="status" className="mt-3 mb-0 text-small text-text-muted">{kind === 'audio' ? 'Comparing decoded audio…' : 'Comparing pictures and soundtracks…'}</p>}
    {error && <p role="alert" className="mt-3 mb-0 text-small text-danger">{error}</p>}
    {result && <ResultView result={result} />}
  </section>
}

export default function MediaReferenceComparison({ analysisId, kind }: { analysisId: string; kind: Kind }) {
  const availability = useQuery({ queryKey: ['analysis', analysisId, kind + '-reference-comparison'],
    queryFn: ({ signal }) => request<Availability>(route(analysisId, kind), { headers: authHeaders(), signal }),
    refetchOnMount: 'always', refetchOnWindowFocus: false, retry: false })
  return availability.isFetchedAfterMount && availability.data?.available && !availability.isError
    ? <MediaReferenceForm key={kind + analysisId} analysisId={analysisId} kind={kind}
      timingAvailable={kind === 'video' && availability.data.timing_available === true} /> : null
}
