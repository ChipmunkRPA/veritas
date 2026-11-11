import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import VideoEvidenceContent from './VideoEvidenceContent'
import VideoNestedEvidenceContent from './VideoNestedEvidenceContent'
import type { VideoEvidenceSet } from './multimediaPackets'

type Lane = 'visual' | 'voice' | 'timing' | 'all'
const lanes: { id: Lane; title: string; description: string }[] = [
  { id: 'visual', title: 'Video frames', description: 'Inspect selected moments and original frame crops.' },
  { id: 'voice', title: 'Voice intervals', description: 'Listen to voice intervals and inspect background and soundtrack measurements.' },
  { id: 'timing', title: 'Audio/picture timing', description: 'Compare recorded mouth inputs with the sound track.' },
  { id: 'all', title: 'All evidence', description: 'Read the complete evidence record in sequence.' },
]
export function videoLaneForHash(hash: string): Lane | null {
  if (/^#media-evidence-(?:timing|AV\d+)/.test(hash)) return 'timing'
  if (/^#media-evidence-(?:voice|background|sounds|stereo|repetition|similar-audio|[ABC]\d+|RS?\d+)/.test(hash)) return 'voice'
  if (/^#media-evidence-(?:visual|V(?:PH)?\d+)/.test(hash)) return 'visual'
  if (hash === '#media-evidence-all') return 'all'
  return null
}
export default function VideoEvidenceWorkspace({ set, token, mediaReady }: {
  set: VideoEvidenceSet; token: string; mediaReady: boolean
}) {
  const location = useLocation(), navigate = useNavigate()
  const [active, setActive] = useState<Lane>(() => videoLaneForHash(location.hash) ?? 'visual')
  useEffect(() => { const lane = videoLaneForHash(location.hash); if (lane) setActive(lane) }, [location.hash])
  const select = (lane: Lane) => {
    setActive(lane)
    navigate({ pathname: location.pathname, search: location.search, hash: `#media-evidence-${lane}` }, { replace: true, preventScrollReset: true })
  }
  const counts = { visual: set.packets.length, voice: set.audio?.packets.length ?? 0, timing: set.av?.packets.length ?? 0, all: 0 }
  const status = (lane: Lane) => {
    if (lane === 'all') return 'Complete record'
    if (lane === 'voice') {
      const acoustic = set.audio?.acoustic_context
      const drops = set.audio?.stereo_context?.windows.filter((window) => window.downmix?.strong_attenuation).length ?? 0
      const repeats = (set.audio?.repetition_context?.pairs.length ?? 0) + (set.audio?.similarity_context?.pairs.length ?? 0)
      if (acoustic || drops || repeats) {
        const summary = [`${counts.voice} voice intervals`]
        if (acoustic) summary.push(`${acoustic.events.length} acoustic observations`)
        if (drops) summary.push(`${drops} mono level ${drops === 1 ? 'drop' : 'drops'}`)
        if (repeats) summary.push(`${repeats} repeated audio ${repeats === 1 ? 'pair' : 'pairs'}`)
        return summary.join(' · ')
      }
    }
    if (counts[lane]) return `${counts[lane]} selected ${lane === 'voice' ? 'intervals' : lane === 'visual' ? 'frame groups' : 'windows'}`
    const channel = lane === 'voice' ? set.audio : lane === 'timing' ? set.av : null
    return channel?.status === 'absent' ? 'No audio stream' : channel?.status === 'unavailable' ? 'Unavailable' : 'No selected evidence'
  }
  return <div className="mt-5">
    <h3 className="m-0 text-body font-medium">Choose what to inspect</h3>
    <div role="tablist" aria-label="Video evidence channels" className="mt-3 grid grid-cols-2 gap-2">
      {lanes.map((lane, index) => <button key={lane.id} type="button" role="tab" id={`evidence-tab-${lane.id}`}
        aria-controls="video-evidence-panel" aria-selected={active === lane.id} tabIndex={active === lane.id ? 0 : -1}
        onClick={() => select(lane.id)} onKeyDown={(event) => {
          const next = event.key === 'ArrowRight' ? (index + 1) % lanes.length : event.key === 'ArrowLeft' ? (index + lanes.length - 1) % lanes.length
            : event.key === 'Home' ? 0 : event.key === 'End' ? lanes.length - 1 : null
          if (next !== null) { event.preventDefault(); select(lanes[next].id); document.getElementById(`evidence-tab-${lanes[next].id}`)?.focus() }
        }} className={`cursor-pointer rounded-md border p-3 text-left transition-colors ${active === lane.id ? 'border-accent bg-accent/10' : 'border-border bg-surface hover:bg-surface-2'}`}>
        <span className="block text-small font-medium">{lane.title}</span>
        <span className="block mt-1 text-caption text-text-muted">{status(lane.id)}</span>
      </button>)}
    </div>
    <p className="mt-3 mb-0 text-small text-text-muted">{lanes.find((lane) => lane.id === active)?.description}</p>
    <div role="tabpanel" id="video-evidence-panel" aria-labelledby={`evidence-tab-${active}`} className="mt-4">
      {(active === 'visual' || active === 'all') && <VideoEvidenceContent set={{ ...set, coverage: set.coverage ? { ...set.coverage, channels: undefined } : set.coverage }} token={token} mediaReady={mediaReady} />}
      {active !== 'visual' && <VideoNestedEvidenceContent set={set} token={token} mediaReady={mediaReady}
        channel={active === 'voice' ? 'audio' : active === 'timing' ? 'av' : 'all'} />}
    </div>
  </div>
}
