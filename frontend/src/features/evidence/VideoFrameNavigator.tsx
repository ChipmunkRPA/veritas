import { EvidencePicture, evidenceGrid } from './EvidenceMedia'
import { seconds, type VideoEvidenceFrame, type VideoEvidenceSet } from './multimediaPackets'

export default function VideoFrameNavigator({ set, token, mediaReady, selected, onSelect }: {
  set: VideoEvidenceSet; token: string; mediaReady: boolean; selected: VideoEvidenceFrame | null
  onSelect: (frame: VideoEvidenceFrame) => void
}) {
  const entries = set.packets.flatMap((packet) => [
    ...packet.frames.map((frame) => ({ packet, frame, pairedBoundary: false })),
    ...(packet.face_boundary?.method === 'face-appearance-paired-edge/1' ? packet.face_boundary_frames ?? [] : [])
      .map((frame) => ({ packet, frame, pairedBoundary: true })),
  ])
  if (!entries.length) return null
  const current = entries.find(({ frame }) => frame.id === selected?.id)
    ?? entries.find(({ frame }) => frame.is_peak && frame.region_meaning === 'detected_face_context')
    ?? entries.find(({ frame }) => frame.is_peak) ?? entries[0]
  return <section aria-label="Selected video moments" className="mt-5 rounded-md border border-border p-4">
    <h3 className="m-0 text-body font-medium">Moments to inspect</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">Choose a recorded sample to jump to its time in the video and compare the area checked. The spaces between samples were not continuously assessed.</p>
    <div className="mt-3 flex flex-wrap gap-2">{entries.map(({ frame, packet, pairedBoundary }) => {
      const known = frame.anchor.playback_time_s != null && Number.isFinite(frame.anchor.playback_time_s) && frame.anchor.playback_time_s >= 0
      return <button key={frame.id} type="button" aria-label={`Inspect ${frame.id} at source ${seconds(frame.anchor.stream_time_s)}`}
        aria-pressed={current.frame.id === frame.id} disabled={!known || !mediaReady} onClick={() => onSelect(frame)}
        className={`rounded-md border px-3 py-2 text-left cursor-pointer disabled:opacity-50 ${current.frame.id === frame.id ? 'border-accent bg-accent/10' : 'border-border bg-surface'}`}>
        <span className="block font-mono text-small">{seconds(frame.anchor.stream_time_s)}</span>
        <span className="block text-caption text-text-muted">{frame.id}{frame.is_peak ? ' · peak sample' : ''}
          {frame.region_meaning === 'detected_face_context' ? ' · face review' : ''}</span>
        <span className="block text-caption">{frame.interval_role ? `${frame.interval_role[0].toUpperCase()}${frame.interval_role.slice(1)} ${pairedBoundary ? 'refined' : 'flagged'} range`
          : packet.claim_scope === 'sampled_window' || frame.score == null ? 'Context for a scored window' : `Response ${frame.score.toFixed(3)}`}</span>
      </button>
    })}</div>
    <div className="mt-4 border-t border-border pt-4">
      <p className="m-0 text-small font-medium">{current.frame.id} · source {seconds(current.frame.anchor.stream_time_s)} · frame {current.frame.anchor.frame_index}</p>
      <p className="mt-2 mb-3 text-small">{current.packet.observation}</p>
      <div className={evidenceGrid}>
        <figure className="m-0"><EvidencePicture set={set} artifact={current.packet.artifacts[current.frame.artifact_refs.context]} token={token} ready={mediaReady}
          alt={`${current.frame.id}: selected frame with the inspection area marked`} />
          <figcaption className="mt-2 text-caption text-text-muted">The outline locates the area inspected. It is not a pixel-level edit mask.</figcaption></figure>
        {current.frame.artifact_refs.crop && <figure className="m-0"><EvidencePicture set={set} artifact={current.packet.artifacts[current.frame.artifact_refs.crop]} token={token} ready={mediaReady}
          alt={`${current.frame.id}: selected original face crop`} />
          <figcaption className="mt-2 text-caption text-text-muted">Original source crop for checking the suspicious sample against its surroundings.</figcaption></figure>}
      </div>
      <p className="mt-3 mb-0 text-caption text-text-muted">{current.pairedBoundary
        ? 'Compare the original face before, during, and after the two measured transitions. The interval ends at the first frame after the measured change.'
        : current.packet.face_return
        ? 'Compare the original face before, during, and after the measured change.'
        : current.packet.face_interval || current.packet.face_sequence
        ? 'The range reflects measured changes on one track. Review the surrounding face samples for context.'
        : current.packet.claim_scope === 'sampled_window'
        ? 'The response belongs to the recorded window; this representative frame has no separate AI score.'
        : 'A high sampled-frame response is an inspection clue, not proof of a deepfake or an edited pixel boundary.'}</p>
    </div>
  </section>
}
