import { useEffect, useRef, useState } from 'react'
import { Button } from '../../components/ui'
import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import { evidenceAssetUrl, evidenceSourceMediaUrl, hasNestedVideoEvidence, seconds, type VideoEvidenceFrame, type VideoEvidencePacket, type VideoEvidenceSet } from './multimediaPackets'
import VideoFrameNavigator from './VideoFrameNavigator'

function PictureChangeSupport({ tiles }: { tiles: number[] }) {
  return <svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"
    className="pointer-events-none absolute inset-0 h-full w-full">
    {tiles.map(tile => {
      const x = (tile % 10) * 10
      const y = Math.floor(tile / 10) * 10
      return <g key={tile}>
        <rect x={x} y={y} width="10" height="10" fill="#f6ca62" fillOpacity="0.15"
          stroke="#172923" strokeWidth="0.65" />
        <rect x={x} y={y} width="10" height="10" fill="none" stroke="#f6ca62" strokeWidth="0.28" />
      </g>
    })}
  </svg>
}

export function ContextClip({ packet, set, token, ready }: {
  packet: Pick<VideoEvidencePacket, 'id' | 'artifacts' | 'frames' | 'clip'> & { claim_scope: string }
  set: VideoEvidenceSet; token: string; ready: boolean
}) {
  const player = useRef<HTMLVideoElement>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => { setFailed(false) }, [token])
  useEffect(() => { const media = player.current; return () => { if (media && !media.paused) media.pause() } }, [ready, token, attempt, failed])
  const clip = packet.clip
  if (!clip) return null // Historical evidence keeps its original source player.
  if (clip.status !== 'ready') return <p className="mt-4 text-small text-text-muted">Context clip unavailable: {clip.reason} Exact recorded frame images remain below.</p>
  const asset = packet.artifacts[clip.artifact_ref]
  const url = asset && evidenceAssetUrl(set, asset, token)
  const [numerator, denominator] = clip.clip_time_base.split('/').map(Number)
  const timeFor = (index: number) => clip.clip_pts[index - clip.start_frame_index] * numerator / denominator
  const seek = (index = selected) => {
    if (index != null && player.current && Number.isFinite(timeFor(index))) player.current.currentTime = timeFor(index)
  }
  return <div className="mt-5 rounded-md border border-border p-3">
    <h4 className="m-0 mb-2 text-small font-medium">Playable visual context · no audio</h4>
    {!ready ? <p className="text-small text-text-muted">Preparing secure media links…</p>
      : failed || !url ? <div role="alert" className="text-small text-text-muted">The context clip could not be loaded. Exact recorded frame images remain below.
        <Button variant="text" onClick={() => { setFailed(false); setAttempt((value) => value + 1) }}>Retry context clip</Button></div>
        : <video key={`${token}:${attempt}`} ref={player} controls preload="metadata" playsInline muted
          aria-label={`${packet.id}: silent source-derived context clip`} src={url} className="w-full max-h-80 rounded-md bg-surface-2"
          onLoadedMetadata={() => seek()} onError={() => setFailed(true)} />}
    <p className="mt-2 mb-0 text-caption text-text-muted">Source {seconds(clip.start_stream_s)}–{seconds(clip.last_stream_s)} · frames {clip.start_frame_index}–{clip.end_frame_index_exclusive - 1}.
      {' '}This is a re-encoded, silent visual copy. It is not the original compressed bytes or exact analyzed pixels.</p>
    <p className="mt-2 mb-0 text-caption text-text-muted">{clip.sampled_frame_indices.length} {packet.claim_scope === 'sampled_frames' ? 'displayed evidence frame(s)' : 'recorded window input frame(s)'} occur in this clip. Continuous playback does not establish continuous analysis coverage.</p>
    {clip.last_frame_duration.basis.startsWith('inferred_') && <p className="mt-2 mb-0 text-caption text-text-muted">The final frame’s display duration is inferred because a following source timestamp is unavailable.</p>}
    <div className="mt-2 flex flex-wrap gap-2">{packet.frames.filter((frame) => clip.sampled_frame_indices.includes(frame.anchor.frame_index)).map((frame) =>
      <Button key={frame.id} variant="text" disabled={!ready || failed || !url} onClick={() => { setSelected(frame.anchor.frame_index); seek(frame.anchor.frame_index) }}>
        Inspect frame {frame.anchor.frame_index} in clip</Button>)}</div>
    {selected != null && <p role="status" className="mt-2 mb-0 text-caption font-mono">Frame {selected} · clip {seconds(timeFor(selected))}. Browser seeking can be approximate; use the exact frame image below.</p>}
  </div>
}

export default function VideoEvidenceContent({ set, token, mediaReady }: { set: VideoEvidenceSet; token: string; mediaReady: boolean }) {
  const player = useRef<HTMLVideoElement>(null)
  const [selected, setSelected] = useState<VideoEvidenceFrame | null>(null)
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => { const media = player.current; return () => { if (media && !media.paused) media.pause() } }, [mediaReady, token, attempt, failed])
  const target = selected?.anchor.playback_time_s
  const seek = () => {
    if (player.current && target != null && Number.isFinite(target) && target >= 0) player.current.currentTime = target
  }
  useEffect(() => { if (player.current?.readyState) seek() }, [target, selected]) // a click seeks; never starts autoplay
  useEffect(() => { setFailed(false) }, [token])
  const chooseFrame = (frame: VideoEvidenceFrame) => {
    setSelected(frame)
    player.current?.scrollIntoView?.({ block: 'center', behavior: 'auto' })
  }
  const hold = set.picture_hold
  const sceneReview = set.coverage?.scene_return_review
  const mouthHold = hold?.mouth_local_hold
  const motion = hold?.motion_sequence_review
  const motionRate = motion?.parameters.frame_rate ?? 20
  const motionLimit = motion?.frames_scanned || motionRate * 9
  const laterFrames = hold && set.coverage ? set.coverage.frame_count - hold.frames_scanned : 0
  const renderFrames = (packet: VideoEvidencePacket, frames: VideoEvidenceFrame[], refined: boolean) =>
    <div className="mt-5 space-y-5">{frames.map((frame) => {
      const cut = packet.face_spatial_cut_frames?.some(item => item.id === frame.id)
      const crop = frame.artifact_refs.crop && packet.artifacts[frame.artifact_refs.crop]
      const mouthCrop = packet.mouth_local_hold && crop
      const sceneSupport = frame.interval_role === 'during' && packet.scene_return
        ? packet.scene_return.tiles : null
      const navigationKnown = frame.anchor.playback_time_s != null && Number.isFinite(frame.anchor.playback_time_s) && frame.anchor.playback_time_s >= 0
      const occurrence = packet.picture_sequence && (frame.anchor.frame_index <= packet.picture_sequence.first.end_frame_index
        ? 'First occurrence' : 'Second occurrence')
      return <div key={frame.id} className="rounded-md border border-border p-3">
        <p className="m-0 mb-3 text-small font-medium">{occurrence && `${occurrence} · `}{frame.interval_role && <span className="capitalize">{frame.interval_role} {packet.scene_return ? 'picture change' : `${cut ? 'picture-cut' : refined ? 'refined' : packet.mouth_local_hold ? 'hold' : packet.face_spatial_cut ? 'review' : 'flagged'} range`} · </span>}Frame {frame.anchor.frame_index}{frame.is_peak ? ' · highest response in this sample sequence' : ''}
          {frame.score !== null && <span className="font-mono"> · response {frame.score.toFixed(3)}</span>}</p>
        <div className={evidenceGrid}>
          <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={packet.artifacts[frame.artifact_refs.original]} token={token} ready={mediaReady} alt={`${frame.id}: exact original video frame`} />
            <figcaption className="mt-2 text-caption text-text-muted">Exact source frame pixels, without diagnostic overlays.</figcaption></figure>
          {sceneSupport && <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={packet.artifacts[frame.artifact_refs.original]}
            token={token} ready={mediaReady} alt={`${frame.id}: source frame with picture-change support tiles`}
            overlay={<PictureChangeSupport tiles={sceneSupport} />} />
            <figcaption className="mt-2 text-caption text-text-muted">The {sceneSupport.length} outlined tiles show texture measurements that support the start and end times. Ordinary cuts can produce this pattern.</figcaption>
          </figure>}
          {!packet.picture_hold && !packet.picture_sequence && !packet.scene_return && <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={crop || packet.artifacts[frame.artifact_refs.context]} token={token} ready={mediaReady}
            alt={`${frame.id}: ${mouthCrop ? 'exact measured mouth region' : crop ? 'source face crop for context' : 'frame context with its recorded label'}`} />
            <figcaption className="mt-2 text-caption text-text-muted">{mouthCrop ? 'Exact pixels of the measured mouth region.'
              : crop ? 'Detected-face crop for inspection; not an edit boundary or the exact aligned classifier input.'
              : 'Source context with a navigation label; no local edit mask is available.'}</figcaption></figure>}
        </div>
        <p className="mt-3 mb-0 text-caption font-mono text-text-muted">Source {seconds(frame.anchor.stream_time_s)} · PTS {frame.anchor.pts} at {frame.anchor.time_base} · player {navigationKnown ? seconds(frame.anchor.playback_time_s!) : 'mapping unavailable'}</p>
        <Button variant="text" className="mt-2" disabled={!navigationKnown || !mediaReady} onClick={() => chooseFrame(frame)}>Go to frame {frame.anchor.frame_index} in playback</Button>
        {!navigationKnown && <p className="mt-1 mb-0 text-caption text-text-muted">Playback timing is not verified for this frame. Use the exact source image above.</p>}
      </div>
    })}</div>

  return <div id="media-evidence-visual" data-testid="video-evidence-content" className="scroll-mt-20">
    {hold && <section aria-labelledby="video-picture-hold-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-picture-hold-heading" className="m-0 text-body font-medium">Identical consecutive frames</h3>
      {hold.status === 'unavailable' ? <p className="mt-2 mb-0 text-small text-text-muted">{hold.reason ?? 'This check could not verify a sequence of frames.'}</p>
        : <p className="mt-2 mb-0 text-small">Checked {hold.frames_scanned} frames from the start{hold.last_stream_time_s != null
          && ` through ${seconds(hold.last_stream_time_s)} on the source clock`}. {hold.status === 'no_match'
          ? 'No qualifying hold was found in that range.' : <a className="underline" href={`#media-evidence-${hold.packet_id}`}>Inspect the recorded hold</a>}</p>}
      {!!hold.continuity_breaks && <p className="mt-2 mb-0 text-caption text-text-muted">The scan found {hold.continuity_breaks} gaps or unverifiable transitions in the frame sequence.</p>}
      {hold.status !== 'unavailable' && laterFrames > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">{laterFrames} later {laterFrames === 1 ? 'frame was' : 'frames were'} outside this check.</p>}
      <p className="mt-2 mb-0 text-caption text-text-muted">Still cards, capture interruptions, and ordinary editing can repeat a frame. This check carries no AI confidence weight.</p>
    </section>}
    {mouthHold && <section aria-labelledby="video-mouth-local-hold-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-mouth-local-hold-heading" className="m-0 text-body font-medium">Mouth-region picture hold</h3>
      <p className="mt-2 mb-0 text-small">{mouthHold.status === 'matched' && mouthHold.packet_id
        ? <a className="underline" href={`#media-evidence-${mouthHold.packet_id}`}>Inspect the measured mouth-region hold</a>
        : mouthHold.status === 'no_match' ? 'No qualifying mouth-region hold was found in the checked frames.'
        : mouthHold.status === 'no_stable_face' ? 'A stable mouth region could not be tracked in the checked frames.'
        : 'A mouth-region hold could not be assessed in this video.'}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">
        {mouthHold.frames_scanned > 0 && (mouthHold.status === 'unavailable'
          ? <>The first {mouthHold.frames_scanned} source frames were verified. </>
          : <>Coverage includes only the first {mouthHold.frames_scanned} source frames. </>)}
        This observation carries no weight in overall AI confidence.
      </p>
      {!!mouthHold.assessed_window_starts?.length && <p className="mt-2 mb-0 text-caption text-text-muted">
        Mouth geometry was measured in {mouthHold.assessed_window_starts.length} of 3 overlapping windows.
        {' '}Source frames {mouthHold.assessed_window_starts.map(start => `${start}–${start + 59}`).join(', ')}.
      </p>}
    </section>}
    {hold?.sequence_review && <section aria-labelledby="video-picture-sequence-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-picture-sequence-heading" className="m-0 text-body font-medium">Repeated moving sequences</h3>
      <p className="mt-2 mb-0 text-small">{hold.sequence_review.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR01">Inspect both matching source ranges</a>
        : hold.sequence_review.status === 'no_match' ? 'No exact moving sequence repeated within the checked frames.'
        : 'The checked footage did not contain two usable ranges of at least half a second.'}</p>
    </section>}
    {motion && <section aria-labelledby="video-motion-sequence-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-motion-sequence-heading" className="m-0 text-body font-medium">Similar repeated moving sequences</h3>
      <p className="mt-2 mb-0 text-small">{motion.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR02">Inspect both ranges with similar pictures and motion</a>
        : motion.status === 'no_match' ? `No qualifying repeat was found in the first ${motionLimit} frames.`
        : motion.reason === 'exact_sequence' ? 'The exact check already found a moving sequence.'
        : `This check needs verified ${1000 / motionRate} ms spacing throughout the first ${motionLimit} frames.`}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">Coverage is limited to the first {motionLimit} native {set.source.width} × {set.source.height} frames in supported {motionRate} fps video. A repeat may come from ordinary editing and carries no AI confidence weight.</p>
    </section>}
    {!!set.coverage?.brief_face_review?.observations.length && <section aria-labelledby="video-brief-face-coverage-heading"
      className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-brief-face-coverage-heading" className="m-0 text-body font-medium">Brief face-change check</h3>
      {set.coverage.brief_face_review.observations.map((observation) => <p key={observation}
        className="mt-2 mb-0 text-small">{observation}</p>)}
      <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero AI confidence weight.</p>
    </section>}
    {!!set.coverage?.brief_face_windows?.observations.length && <section aria-labelledby="video-brief-face-windows-heading"
      className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-brief-face-windows-heading" className="m-0 text-body font-medium">Overlapping face-change windows</h3>
      {set.coverage.brief_face_windows.observations.map((observation) => <p key={observation}
        className="mt-2 mb-0 text-small">{observation}</p>)}
      <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero AI confidence weight.</p>
    </section>}
    {sceneReview && <section aria-labelledby="video-scene-return-coverage-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-scene-return-coverage-heading" className="m-0 text-body font-medium">Picture-change coverage</h3>
      <p className="mt-2 mb-0 text-small">{sceneReview.observation}{sceneReview.packet_id &&
        <> <a className="underline" href={`#media-evidence-${sceneReview.packet_id}`}>Inspect the source frames and measured tiles</a></>}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">{sceneReview.limitation}</p>
    </section>}
    {set.coverage ? <div className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">{set.coverage.scored_frame_count} frame(s) were inputs to recorded {hasNestedVideoEvidence(set.version) ? 'visual frame or window' : 'frame or window'} checks in a table of {set.coverage.frame_count} frames.
        {' '}{set.coverage.window_check_count} window check(s) cover {set.coverage.window_frame_indices.length} distinct sampled frames.</p>
      {hasNestedVideoEvidence(set.version) && <p className="mt-2 mb-0">These counts cover visual checks only. Voice and audio/picture timing coverage appear in their separate sections.</p>}
      {set.coverage.face_detail_repeat && <p className="mt-2 mb-0">{set.coverage.face_detail_repeat.observation}</p>}
      <p className="mt-2 mb-0">These counts may overlap. Decoded frames are not all scored; unscored frames and the gaps between samples have not been cleared.</p>
      {set.coverage.table_truncated && <p className="mt-2 mb-0">The frame table is truncated; this does not represent the entire video.</p>}
      {set.coverage.unmapped_observations > 0 && <p className="mt-2 mb-0">{set.coverage.unmapped_observations} observation(s) could not be mapped reliably and are omitted.</p>}
      {set.coverage.channels && <div className="mt-3 border-t border-border pt-3">{(['audio', 'av_sync'] as const).map((key) => {
        const channel = set.coverage!.channels![key]
        return channel && <p key={key} className="mt-2 mb-0"><strong>{key === 'audio' ? 'Voice evidence' : 'Audio/video synchronization evidence'}:</strong> {channel.reason}</p>
      })}</div>}
    </div> : <p className="mt-3 text-small text-text-muted">Frame coverage is unavailable. Missing evidence does not establish authenticity.</p>}
    {set.packets.length > 0 && <div className="mt-5">
      <h3 className="m-0 mb-2 text-small font-medium">Original video for context</h3>
      {!mediaReady ? <p className="text-small text-text-muted">Preparing secure media links…</p>
        : failed ? <div role="alert" className="text-small text-text-muted">The original video could not be played. Exact recorded frame images remain below.
          <Button variant="text" onClick={() => { setFailed(false); setAttempt((v) => v + 1) }}>Retry video</Button></div>
          : <video key={`${token}:${attempt}`} ref={player} controls preload="metadata" playsInline aria-label="Original video evidence playback"
            src={evidenceSourceMediaUrl(set.analysis_id, token)} className="w-full max-h-80 rounded-md bg-surface-2"
            onLoadedMetadata={seek} onError={() => setFailed(true)} />}
      <p className="mt-2 mb-0 text-caption text-text-muted">Browser playback can land near a frame. The exact analyzed frame images below are the evidence; playback between sampled frames is context.</p>
      {selected && <p role="status" className="mt-2 mb-0 text-caption font-mono">Selected frame {selected.anchor.frame_index} · source {seconds(selected.anchor.stream_time_s)} · player {target != null ? seconds(target) : 'time unavailable'}</p>}
    </div>}
    <VideoFrameNavigator set={set} token={token} mediaReady={mediaReady} selected={selected} onSelect={chooseFrame} />
    <div className="mt-6 space-y-7">{set.packets.map((packet) => <article key={packet.id} id={`media-evidence-${packet.id}`} className={evidenceCard}>
      <h3 className="m-0 text-body font-medium"><span className="font-mono text-accent mr-3">{packet.id}</span>{packet.title}</h3>
      {!packet.picture_hold && <p className="mt-2 mb-0 text-body">{packet.observation}</p>}
      {!packet.scene_return && <p className="mt-2 mb-0 text-small text-text-muted">{packet.picture_hold
        ? <>{packet.picture_hold.frame_count} consecutive decoded frames have identical pixels over <strong>{seconds(packet.picture_hold.span_s)}</strong> between the first and last source timestamps.</>
        : packet.picture_sequence
        ? <>First occurrence <strong>{seconds(packet.picture_sequence.first.start_stream_s)}–{seconds(packet.picture_sequence.first.end_stream_s)}</strong> · Second occurrence <strong>{seconds(packet.picture_sequence.second.start_stream_s)}–{seconds(packet.picture_sequence.second.end_stream_s)}</strong> on the source clock.</>
        : packet.mouth_local_hold && packet.window
        ? <>Measured mouth-region hold <strong>{seconds(packet.window.start_stream_s)}–{seconds(packet.window.end_stream_s)}</strong> on the source clock. The end is exclusive. Source frames {packet.mouth_local_hold.frames[0]}–{packet.mouth_local_hold.frames[1] - 1}.</>
        : packet.whole_face_review
        ? 'Two checks compared faces at six sampled positions across the clip. The three frames shown below are examples of those inputs.'
        : packet.face_sequence
        ? <>Measured face-change range <strong>{seconds(packet.face_sequence.start_stream_s)}–{seconds(packet.face_sequence.end_stream_s)}</strong>. Three consecutive crops support each of {packet.face_sequence.centers.during.length} centers in this range. The response belongs to the sequence.</>
        : packet.face_return
        ? <>Brief face-change range <strong>{seconds(packet.face_return.start_stream_s)}–{seconds(packet.face_return.end_stream_s)}</strong> on the source clock, with an exclusive end. {packet.face_return.frame_inputs.length} sampled face views support this observation. It carries no weight in overall AI confidence.</>
        : packet.face_interval
        ? <>Flagged source range <strong>{seconds(packet.face_interval.start_stream_s)}–{seconds(packet.face_interval.end_stream_s)}</strong> across {packet.face_interval.samples.during.length} scored samples.</>
        : packet.claim_scope === 'sampled_window'
        ? 'One response for a sampled window. Representative frames do not have individual scores.'
        : 'Individual sampled-frame responses. A face crop is context, not a mask of edited pixels.'}</p>}
      {packet.window && <details className="mt-3 text-small text-text-muted"><summary className="cursor-pointer">Exact window coverage</summary>
        <p className="mt-2 mb-0">Source timestamps {seconds(packet.window.start_stream_s)}–{seconds(packet.window.end_stream_s)}. {packet.whole_face_review ? 'Planned frame indices' : 'Input frame indices'}: {packet.window.frame_indices.join(', ')}.</p>
        <p className="mt-2 mb-0">{packet.face_return && ['face-short-return/9', 'face-short-return/10', 'face-short-return/11', 'face-short-return/12'].includes(packet.face_return.method ?? '')
          ? `Nearby motion was reviewed in ${packet.face_return.scene_frame_inputs?.length ?? 0} native pictures. Face appearances were measured in ${packet.face_return.frame_inputs.length} sampled pictures.`
          : packet.mouth_local_hold ? 'The recorded window lists the source frames used for the mouth-region measurement.'
          : packet.whole_face_review ? 'These positions show sample coverage. Some frames may lack a usable face, and intervals between samples were unassessed.'
          : 'Only these frames were input to this window. This does not establish continuous coverage between them.'}</p>
      </details>}
      {!packet.picture_hold && !packet.picture_sequence && !packet.whole_face_review && !packet.scene_return && <ContextClip packet={packet} set={set} token={token} ready={mediaReady} />}
      {packet.face_spatial_cut && packet.face_spatial_cut_frames && <section className="mt-5 rounded-md border border-border p-3">
        <h4 className="m-0 text-small font-medium">Picture-cut context</h4>
        <p className="mt-2 mb-0 text-small text-text-muted">Picture detail outside the face changed at both ends of <strong>{seconds(packet.face_spatial_cut.start_stream_s)}–{seconds(packet.face_spatial_cut.end_stream_s)}</strong> on the source clock, with an exclusive end. A cut between shots is one plausible explanation for this face change.</p>
        {renderFrames(packet, packet.face_spatial_cut_frames, false)}
      </section>}
      {packet.face_boundary && packet.face_boundary_frames && <section className="mt-5 rounded-md border border-border p-3">
        <h4 className="m-0 text-small font-medium">Refined boundary estimate</h4>
        <p className="mt-2 mb-0 text-small text-text-muted"><strong>{seconds(packet.face_boundary.start_stream_s)}–{seconds(packet.face_boundary.end_stream_s)}</strong> on the source clock.
          {packet.face_boundary.method === 'face-appearance-paired-edge/1'
            ? ' The end is exclusive. Two appearance transitions on the same tracked face bracket the interval, with limited motion measured in nearby pixels.'
            : ` ${packet.face_boundary.frame_inputs.length} denser samples on the same tracked face support this estimate.`}
          {' '}It carries no weight in overall AI confidence.</p>
        {renderFrames(packet, packet.face_boundary_frames, true)}
      </section>}
      {renderFrames(packet, packet.frames, false)}
      {packet.mouth_local_hold && packet.artifacts.mouth_hold_waveform && packet.artifacts.mouth_hold_audio &&
        <div className="mt-4 space-y-3">
          <EvidencePicture set={set} artifact={packet.artifacts.mouth_hold_waveform} token={token} ready={mediaReady}
            alt="Waveform during the measured mouth-region hold" chart />
          <EvidenceAudio set={set} artifact={packet.artifacts.mouth_hold_audio} token={token} ready={mediaReady}
            alt="Listen to the measured mouth-region hold" />
        </div>}
      <p className="mt-4 mb-0 text-small">{packet.interpretation}</p><EvidenceMeasurements values={packet.measurements} />
      <EvidenceLimits lines={packet.limitations} /><EvidenceReferences packet={packet} />
    </article>)}</div>
  </div>
}
