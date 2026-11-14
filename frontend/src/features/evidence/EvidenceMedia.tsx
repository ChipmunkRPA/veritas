import { useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../../components/ui'
import { measurementValue } from './imagePackets'
import { evidenceAssetUrl, type EvidenceAsset, type MultimediaEvidenceSet, type PacketBase, type EvidenceMeasure } from './multimediaPackets'

export interface MediaProps { set: MultimediaEvidenceSet; artifact: EvidenceAsset; token: string; ready: boolean; alt: string }
type PictureProps = MediaProps & { chart?: boolean; overlay?: ReactNode }
function Picture({ set, artifact, token, ready, alt, chart = false, overlay }: PictureProps) {
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const url = ready && artifact.media_type === 'image/png' ? evidenceAssetUrl(set, artifact, token) : null
  const image = url && <img key={attempt} src={url} alt={alt} width={artifact.width} height={artifact.height} loading="lazy"
    className={overlay ? 'block h-auto w-full' : 'block w-full h-full object-contain'} onError={() => setFailed(true)} />
  return <div className={`rounded-md border border-border bg-surface-2 overflow-hidden flex items-center justify-center ${chart ? 'min-h-24' : 'min-h-40 aspect-[4/3]'}`}>
    {!ready ? <p className="p-4 text-small text-text-muted">Preparing secure media links…</p>
      : !url || failed ? <div role="alert" className="p-4 text-small text-text-muted">This evidence image could not be loaded. Its observations remain available.
        {url && <Button variant="text" onClick={() => { setFailed(false); setAttempt((v) => v + 1) }}>Retry image</Button>}</div>
        : overlay ? <div className="relative w-full">{image}{overlay}</div> : image}
  </div>
}
export function EvidencePicture(props: PictureProps) {
  return <Picture key={`${props.artifact.sha256}:${props.token}:${props.ready}`} {...props} />
}
function Clip({ set, artifact, token, ready, alt }: MediaProps) {
  const [failed, setFailed] = useState(false)
  const url = ready && artifact.media_type === 'audio/wav' ? evidenceAssetUrl(set, artifact, token) : null
  return <div className="p-3 rounded-md border border-border bg-surface-2">
    <p className="m-0 mb-2 text-small font-medium">{alt}</p>
    {!ready ? <p className="m-0 text-small text-text-muted">Preparing secure media links…</p>
      : !url || failed ? <div role="alert" className="text-small text-text-muted">This clip could not be loaded.
        {url && <Button variant="text" onClick={() => setFailed(false)}>Retry clip</Button>}</div>
        : <audio controls preload="none" aria-label={alt} src={url} className="w-full max-w-full" onError={() => setFailed(true)} />}
  </div>
}
export function EvidenceAudio(props: MediaProps) { return <Clip key={`${props.artifact.sha256}:${props.token}:${props.ready}`} {...props} /> }
export function EvidenceLimits({ lines }: { lines: string[] }) {
  return lines.length ? <ul className="mt-3 mb-0 pl-4 space-y-1 list-disc text-small text-text-muted">{lines.map((line, i) => <li key={i}>{line}</li>)}</ul> : null
}
export function EvidenceMeasurements({ values }: { values: EvidenceMeasure[] }) {
  return <dl className="mt-3 mb-0 space-y-1.5">{values.map((value, i) => <div key={i} className="flex flex-wrap justify-between gap-x-4 text-small">
    <dt className="text-text-muted">{value.label}</dt><dd className="m-0 font-mono num">
      {typeof value.value === 'number' ? measurementValue(value.value) : value.value}{value.unit && ` ${value.unit}`}
    </dd>
  </div>)}</dl>
}
export function EvidenceReferences({ packet }: { packet: PacketBase }) {
  return <p className="mt-4 mb-0 text-caption text-text-muted">
    {packet.finding_ids.length > 0 && <>Related findings: {packet.finding_ids.map((id, i) => <span key={id}>
      {i > 0 && ', '}<Link to={`#finding-${encodeURIComponent(id)}`} className="underline underline-offset-2">{id}</Link></span>)}. </>}
    {packet.signal_refs.length > 0 && <>Recorded checks: {[...new Set(packet.signal_refs.map((ref) => ref.display_id ?? 'Check'))].join(', ')}.</>}
  </p>
}
export const evidenceGrid = 'grid grid-cols-2 gap-4 max-[639px]:grid-cols-1'
export const evidenceCard = 'border-t border-border pt-6 scroll-mt-20'
