import { useEffect, useRef, useState } from 'react'
import { Button } from '../../components/ui'
import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import { evidenceAssetUrl, evidenceSourceMediaUrl, hasNestedVideoEvidence, seconds, type VideoEvidenceFrame, type VideoEvidencePacket, type VideoEvidenceSet } from './multimediaPackets'
import VideoFrameNavigator from './VideoFrameNavigator'

function occurrenceLabel(sequence: NonNullable<VideoEvidencePacket['picture_sequence']>, side: 'first' | 'second') {
  if (sequence.frame_steps) {
    const other = side === 'first' ? 'second' : 'first'
    return sequence.frame_steps[side] < sequence.frame_steps[other] ? 'Faster occurrence' : 'Slower occurrence'
  }
  if (sequence.framing && sequence.order === 'reverse') {
    const position = side === 'first' ? 'First range' : 'Second range'
    const framing = sequence.framing.wider_range === side ? 'wider framing' : 'tighter framing'
    return `${position} · ${framing}`
  }
  if (side === 'first') return 'First occurrence'
  return sequence.order === 'reverse' ? 'Reversed occurrence' : 'Second occurrence'
}

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

export default function VideoEvidenceContent({ set, token, mediaReady, scope = 'all' }: {
  set: VideoEvidenceSet; token: string; mediaReady: boolean; scope?: 'visual' | 'timing' | 'all'
}) {
  const player = useRef<HTMLVideoElement>(null)
  const [selected, setSelected] = useState<VideoEvidenceFrame | null>(null)
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => { const media = player.current; return () => { if (media && !media.paused) media.pause() } }, [mediaReady, token, attempt, failed])
  const target = selected?.anchor.playback_time_s
  const showVisual = scope !== 'timing'
  const packets = scope === 'all' ? set.packets : set.packets.filter(packet => showVisual ? !packet.lip_timing_review : packet.lip_timing_review)
  const playableContext = packets.find(packet => packet.clip?.status === 'ready')
  const seek = () => {
    if (player.current && target != null && Number.isFinite(target) && target >= 0) player.current.currentTime = target
  }
  useEffect(() => { if (player.current?.readyState) seek() }, [target, selected]) // a click seeks; never starts autoplay
  useEffect(() => { setFailed(false) }, [token])
  const chooseFrame = (frame: VideoEvidenceFrame) => {
    setSelected(frame)
    player.current?.scrollIntoView?.({ block: 'center', behavior: 'auto' })
  }
  const coverage = showVisual ? set.coverage : undefined
  const hold = showVisual ? set.picture_hold : undefined
  const sceneReview = coverage?.scene_return_review
  const transitionReview = coverage?.face_single_boundary_coverage
  const transition = packets.find(packet => packet.face_single_boundary)?.face_single_boundary
  const mouthHold = hold?.mouth_local_hold
  const eyeHold = hold?.eye_local_hold
  const lipTiming = scope !== 'visual' ? set.lip_timing_review : undefined
  const motion = hold?.motion_sequence_review
  const reverseMotion = hold?.reverse_motion_sequence_review
  const retimedMotion = hold?.retimed_motion_sequence_review
  const reflectedMotion = hold?.reflected_motion_sequence_review
  const croppedMotion = hold?.cropped_motion_sequence_review
  const motionRate = motion?.parameters.frame_rate ?? 20
  const motionLimit = motion?.frames_scanned || motionRate * 9
  const laterFrames = hold && coverage ? coverage.frame_count - hold.frames_scanned : 0
  const renderFrames = (packet: VideoEvidencePacket, frames: VideoEvidenceFrame[], refined: boolean) =>
    <div className="mt-5 space-y-5">{frames.map((frame) => {
      const cut = packet.face_spatial_cut_frames?.some(item => item.id === frame.id)
      const crop = frame.artifact_refs.crop && packet.artifacts[frame.artifact_refs.crop]
      const measuredRegion = crop && (packet.eye_local_hold ? 'eye' : (packet.mouth_local_hold || packet.lip_timing_review) ? 'mouth' : null)
      const mismatch = packet.lip_timing_review?.articulation || packet.lip_timing_review?.motion_sound_agreement
      const timingLabel = packet.lip_timing_review?.speech_activity_gap ? 'reduced speech activity'
        : packet.lip_timing_review?.silent_mouth ? 'near-silent' : mismatch ? 'mismatch' : 'timing'
      const rangeLabel = packet.scene_return ? 'picture change'
        : `${cut ? 'picture-cut' : refined ? 'refined' : packet.mouth_local_hold || packet.eye_local_hold ? 'hold' : packet.lip_timing_review ? timingLabel : packet.face_spatial_cut ? 'review' : 'flagged'} range`
      const sceneSupport = frame.interval_role === 'during' && packet.scene_return
        ? packet.scene_return.tiles : null
      const navigationKnown = frame.anchor.playback_time_s != null && Number.isFinite(frame.anchor.playback_time_s) && frame.anchor.playback_time_s >= 0
      const occurrence = packet.picture_sequence && occurrenceLabel(packet.picture_sequence,
        frame.anchor.frame_index <= packet.picture_sequence.first.end_frame_index ? 'first' : 'second')
      const transitionLabel = packet.face_single_boundary
        ? frame.interval_role === 'before' ? 'Last frame before transition' : frame.interval_role === 'transition'
          ? 'First frame at transition' : 'Later supporting frame' : null
      return <div key={frame.id} className="rounded-md border border-border p-3">
        <p className="m-0 mb-3 text-small font-medium">{occurrence && `${occurrence} · `}{transitionLabel
          ? `${transitionLabel} · ` : frame.interval_role && <span className="capitalize">{frame.interval_role} {rangeLabel} · </span>}Frame {frame.anchor.frame_index}{frame.is_peak ? ' · highest response in this sample sequence' : ''}
          {frame.score !== null && <span className="font-mono"> · response {frame.score.toFixed(3)}</span>}</p>
        <div className={evidenceGrid}>
          <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={packet.artifacts[frame.artifact_refs.original]} token={token} ready={mediaReady} alt={`${frame.id}: exact original video frame`} />
            <figcaption className="mt-2 text-caption text-text-muted">Exact source frame pixels, without diagnostic overlays.</figcaption></figure>
          {sceneSupport && <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={packet.artifacts[frame.artifact_refs.original]}
            token={token} ready={mediaReady} alt={`${frame.id}: source frame with picture-change support tiles`}
            overlay={<PictureChangeSupport tiles={sceneSupport} />} />
            <figcaption className="mt-2 text-caption text-text-muted">The {sceneSupport.length} outlined tiles show texture measurements that support the start and end times. Ordinary cuts can produce this pattern.</figcaption>
          </figure>}
          {packet.picture_sequence?.framing && crop && <figure className="m-0 min-w-0">
            <EvidencePicture set={set} artifact={crop} token={token} ready={mediaReady}
              alt={`Frame ${frame.anchor.frame_index}: wider source cropped to the matching area`} />
            <figcaption className="mt-2 text-caption text-text-muted">Exact crop from the wider source frame.</figcaption>
          </figure>}
          {!packet.picture_hold && !packet.picture_sequence && !packet.scene_return && <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={crop || packet.artifacts[frame.artifact_refs.context]} token={token} ready={mediaReady}
            alt={`${frame.id}: ${measuredRegion ? `exact measured ${measuredRegion} region` : crop ? 'source face crop for context' : 'frame context with its recorded label'}`} />
            <figcaption className="mt-2 text-caption text-text-muted">{measuredRegion ? `Exact pixels of the measured ${measuredRegion} region.`
              : packet.face_single_boundary ? 'Exact crop of the face from this source frame.'
              : crop ? 'Detected-face crop for inspection; not an edit boundary or the exact aligned classifier input.'
              : 'Source context with a navigation label; no local edit mask is available.'}</figcaption></figure>}
        </div>
        <p className="mt-3 mb-0 text-caption font-mono text-text-muted">Source {seconds(frame.anchor.stream_time_s)} · PTS {frame.anchor.pts} at {frame.anchor.time_base} · player {navigationKnown ? seconds(frame.anchor.playback_time_s!) : 'mapping unavailable'}</p>
        <Button variant="text" className="mt-2" disabled={!navigationKnown || !mediaReady || failed} onClick={() => chooseFrame(frame)}>Go to frame {frame.anchor.frame_index} in playback</Button>
        {!navigationKnown && <p className="mt-1 mb-0 text-caption text-text-muted">Playback timing is not verified for this frame. Use the exact source image above.</p>}
      </div>
    })}</div>

  return <div id={showVisual ? 'media-evidence-visual' : undefined} data-testid="video-evidence-content" className="scroll-mt-20">
    {transitionReview && <section aria-labelledby="video-face-transition-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-face-transition-heading" className="m-0 text-body font-medium">Face transition check</h3>
      <p className="mt-2 mb-0 text-small">{transition
        ? <><a className="underline" href="#media-evidence-VFB01">Inspect the transition</a> at <strong>{seconds(transition.time_s)}</strong> on the source clock, frame {transition.boundary}.</>
        : transitionReview.status === 'no_match' ? 'No qualifying face appearance transition was found in the portion checked.'
        : transitionReview.reason === 'shot_change' ? 'This check requires one continuous shot. A shot change was detected.'
        : transitionReview.reason === 'profile' ? transitionReview.method === 'single-face-boundary/3'
          ? 'This check covers four-second, 640 × 360 H.264 MP4 with a verified 25 fps or qualified 30 fps clock.'
          : 'This check covers four-second, 640 × 360 H.264 MP4 with a verified 25 fps clock.'
        : transitionReview.reason === 'provenance' ? 'Verified source credentials determined the analysis path.'
        : transitionReview.reason === 'insufficient_face_context' ? 'The face views and surrounding picture did not support a stable comparison.'
        : transitionReview.reason === 'insufficient_native_boundary' ? 'The nearby source frames did not establish a precise transition.'
        : transitionReview.reason === 'conflicting_boundary_context' ? 'The background checks pointed to different source frames, so this check did not display a transition.'
        : 'The face transition check was unavailable for this upload.'}</p>
      {transitionReview.frames_scanned > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">
        The check read {transitionReview.frames_scanned} native pictures for shot changes and measured {transitionReview.measured_face_frames} usable face views.
        {transitionReview.method === 'single-face-boundary/3' && ' Source timing was verified at 30 frames per second.'}
      </p>}
      {transition?.method === 'single-face-boundary/3' && <p className="mt-2 mb-0 text-caption text-text-muted">A focus change can produce a similar pattern and can also occur alongside a face swap.</p>}
      <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero AI confidence weight.</p>
    </section>}
    {lipTiming && <section aria-labelledby="video-lip-timing-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-lip-timing-heading" className="m-0 text-body font-medium">{['chipmunk.video-lip-timing/8', 'chipmunk.video-lip-timing/9'].includes(lipTiming.schema ?? '')
        ? 'Speech activity and visible mouth movement'
        : ['chipmunk.video-lip-timing/3', 'chipmunk.video-lip-timing/4', 'chipmunk.video-lip-timing/5', 'chipmunk.video-lip-timing/6', 'chipmunk.video-lip-timing/7', 'chipmunk.video-lip-timing/10', 'chipmunk.video-lip-timing/11'].includes(lipTiming.schema ?? '')
          ? 'Speech and mouth comparison' : 'Speech and mouth timing trial'}</h3>
      <p className="mt-2 mb-0 text-small text-text-muted">{lipTiming.observation}</p>
      {lipTiming.selected_audio && <p className="mt-2 mb-0 text-caption text-text-muted">
        Compared the picture with default audio track {lipTiming.selected_audio.source_stream_position + 1}.{' '}
        Dubbing, commentary, or a different voice can affect this comparison.
      </p>}
      {lipTiming.shot_coverage ? <ul className="mt-2 mb-0 space-y-1 pl-5 text-small">
        {lipTiming.shot_coverage.map(shot => <li key={shot.frames[0]}>
          {seconds(shot.audio_samples[0] / 16000)}–{seconds(shot.audio_samples[1] / 16000)} · {shot.observation}.
          {shot.exhibit && <> <a className="underline" href={`#media-evidence-${shot.exhibit}`}>Inspect source frames and waveform</a></>}
        </li>)}
      </ul> : lipTiming.status === 'matched' && <a className="mt-2 inline-block text-small underline" href="#media-evidence-VLT01">Inspect source frames and waveform</a>}
    </section>}
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
    {eyeHold && <section aria-labelledby="video-eye-local-hold-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-eye-local-hold-heading" className="m-0 text-body font-medium">Eye-region picture hold</h3>
      <p className="mt-2 mb-0 text-small">{eyeHold.status === 'matched' && eyeHold.packet_id
        ? <a className="underline" href={`#media-evidence-${eyeHold.packet_id}`}>Inspect the measured eye-region hold</a>
        : eyeHold.status === 'no_match' ? 'No qualifying eye-region hold was found in the checked frames.'
        : eyeHold.status === 'no_stable_face' ? 'The check could not track a stable eye region in the checked frames.'
        : eyeHold.status === 'outside_scope' ? 'This upload falls outside the supported video profile for this check.'
        : 'The eye-region check was unavailable for this analysis.'}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">
        {eyeHold.frames_scanned > 0 && <>Checked {eyeHold.frames_scanned} verified source frames. </>}
        This check covers two-second, upright 480 × 360 H.264 MP4 at 30 fps with 4:2:0 chroma and a verified frame clock.
        {' '}Ordinary blur or re-encoding can produce a similar pattern. This check carries zero AI confidence weight.
      </p>
    </section>}
    {hold?.sequence_review && <section aria-labelledby="video-picture-sequence-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-picture-sequence-heading" className="m-0 text-body font-medium">Repeated moving sequences</h3>
      <p className="mt-2 mb-0 text-small">{hold.sequence_review.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR01">Inspect both matching source ranges</a>
        : hold.sequence_review.status === 'no_match' ? 'No exact repeat in the same frame order was found in the checked frames.'
        : 'The checked footage did not contain two usable ranges of at least half a second.'}</p>
    </section>}
    {hold?.reverse_sequence_review && <section aria-labelledby="video-reverse-sequence-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-reverse-sequence-heading" className="m-0 text-body font-medium">Reversed moving sequences</h3>
      <p className="mt-2 mb-0 text-small">{hold.reverse_sequence_review.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR03">Inspect both source ranges and their endpoint frames</a>
        : hold.reverse_sequence_review.status === 'no_match' ? 'No exact reversed sequence was found in the checked frames.'
        : 'The checked footage did not contain two usable ranges of at least half a second.'}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">Checked the first {hold.frames_scanned} source frames for exact decoded matches. Reverse playback can be part of ordinary editing.</p>
    </section>}
    {reverseMotion && reverseMotion.reason !== 'exact_sequence' && <section aria-labelledby="video-reverse-motion-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-reverse-motion-heading" className="m-0 text-body font-medium">Similar moving pictures in reverse order</h3>
      <p className="mt-2 mb-0 text-small">{reverseMotion.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR04">Inspect both source ranges and their endpoint frames</a>
        : reverseMotion.status === 'no_match' ? `No qualifying reverse match was found in the first ${reverseMotion.frames_scanned} frames.`
        : 'This check needs regular, verified source timestamps throughout the first nine seconds.'}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">The check compares whole-picture appearance and reversed motion. Reverse playback can be ordinary editing and carries no AI confidence weight.</p>
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
    {retimedMotion && <section aria-labelledby="video-retimed-motion-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-retimed-motion-heading" className="m-0 text-body font-medium">Footage repeated at two playback speeds</h3>
      <p className="mt-2 mb-0 text-small">{retimedMotion.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR05">Inspect the faster and slower source ranges</a>
        : retimedMotion.status === 'no_match' ? `No match at ${retimedMotion.parameters?.playback_step_pairs ? 'a 2:1, 3:2, or 4:3' : 'a two-to-one'} speed ratio was found in the first ${retimedMotion.frames_scanned} frames.`
        : 'This check needs regular, verified source timestamps throughout the first nine seconds.'}</p>
    </section>}
    {reflectedMotion && <section aria-labelledby="video-reflected-motion-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-reflected-motion-heading" className="m-0 text-body font-medium">Mirrored footage check</h3>
      <p className="mt-2 mb-0 text-small">{reflectedMotion.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR06">Inspect both source ranges and frames</a>
        : reflectedMotion.status === 'no_match' ? `No mirrored repeat was found in the first ${reflectedMotion.frames_scanned} frames.`
        : 'This check needs regular, verified source timestamps throughout the checked opening frames.'}</p>
    </section>}
    {croppedMotion && <section aria-labelledby="video-cropped-motion-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-cropped-motion-heading" className="m-0 text-body font-medium">Cropped footage check</h3>
      <p className="mt-2 mb-0 text-small">{croppedMotion.status === 'matched'
        ? <a className="underline" href="#media-evidence-VPR07">Inspect both time ranges and matching crops</a>
        : croppedMotion.status === 'no_match' ? `No qualifying cropped repeat was found in the first ${croppedMotion.frames_scanned} frames.`
        : 'This check needs regular, verified source timestamps throughout the checked opening frames.'}</p>
    </section>}
    {!!coverage?.brief_face_review?.observations.length && <section aria-labelledby="video-brief-face-coverage-heading"
      className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-brief-face-coverage-heading" className="m-0 text-body font-medium">Brief face-change check</h3>
      {coverage.brief_face_review.observations.map((observation) => <p key={observation}
        className="mt-2 mb-0 text-small">{observation}</p>)}
      <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero AI confidence weight.</p>
    </section>}
    {!!coverage?.brief_face_windows?.observations.length && <section aria-labelledby="video-brief-face-windows-heading"
      className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-brief-face-windows-heading" className="m-0 text-body font-medium">Overlapping face-change windows</h3>
      {coverage.brief_face_windows.observations.map((observation) => <p key={observation}
        className="mt-2 mb-0 text-small">{observation}</p>)}
      <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero AI confidence weight.</p>
    </section>}
    {sceneReview && <section aria-labelledby="video-scene-return-coverage-heading" className="mt-4 rounded-md border border-border p-4">
      <h3 id="video-scene-return-coverage-heading" className="m-0 text-body font-medium">Picture-change coverage</h3>
      <p className="mt-2 mb-0 text-small">{sceneReview.observation}{sceneReview.packet_id &&
        <> <a className="underline" href={`#media-evidence-${sceneReview.packet_id}`}>Inspect the source frames and measured tiles</a></>}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted">{sceneReview.limitation}</p>
    </section>}
    {coverage ? <div className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">{coverage.scored_frame_count} frame(s) were inputs to recorded {hasNestedVideoEvidence(set.version) ? 'visual frame or window' : 'frame or window'} checks in a table of {coverage.frame_count} frames.
        {' '}{coverage.window_check_count} window check(s) cover {coverage.window_frame_indices.length} distinct sampled frames.</p>
      {hasNestedVideoEvidence(set.version) && <p className="mt-2 mb-0">These counts cover visual checks only. Voice and audio/picture timing coverage appear in their separate sections.</p>}
      {coverage.face_detail_repeat && <p className="mt-2 mb-0">{coverage.face_detail_repeat.observation}</p>}
      <p className="mt-2 mb-0">These counts may overlap. Decoded frames are not all scored; unscored frames and the gaps between samples have not been cleared.</p>
      {coverage.table_truncated && <p className="mt-2 mb-0">The frame table is truncated; this does not represent the entire video.</p>}
      {coverage.unmapped_observations > 0 && <p className="mt-2 mb-0">{coverage.unmapped_observations} observation(s) could not be mapped reliably and are omitted.</p>}
      {coverage.channels && <div className="mt-3 border-t border-border pt-3">{(['audio', 'av_sync'] as const).map((key) => {
        const channel = coverage.channels?.[key]
        return channel && <p key={key} className="mt-2 mb-0"><strong>{key === 'audio' ? 'Voice evidence' : 'Audio/video synchronization evidence'}:</strong> {channel.reason}</p>
      })}</div>}
    </div> : showVisual && <p className="mt-3 text-small text-text-muted">Frame coverage is unavailable. Missing evidence does not establish authenticity.</p>}
    {packets.length > 0 && <div className="mt-5">
      <h3 className="m-0 mb-2 text-small font-medium">Original video for context</h3>
      {!mediaReady ? <p className="text-small text-text-muted">Preparing secure media links…</p>
        : failed ? <div role="alert" className="text-small text-text-muted">Original video playback is unavailable. Exact recorded frame images remain below.
          {playableContext && <> <a className="underline" href={`#media-evidence-${playableContext.id}`}>Open the playable context clip</a>.</>}
          <Button variant="text" onClick={() => { setFailed(false); setAttempt((v) => v + 1) }}>Retry video</Button></div>
          : <video key={`${token}:${attempt}`} ref={player} controls preload="metadata" playsInline aria-label="Original video evidence playback"
            src={evidenceSourceMediaUrl(set.analysis_id, token)} className="w-full max-h-80 rounded-md bg-surface-2"
            onLoadedMetadata={seek} onError={() => setFailed(true)} />}
      <p className="mt-2 mb-0 text-caption text-text-muted">Browser playback can land near a frame. The exact analyzed frame images below are the evidence; playback between sampled frames is context.</p>
      {selected && <p role="status" className="mt-2 mb-0 text-caption font-mono">Selected frame {selected.anchor.frame_index} · source {seconds(selected.anchor.stream_time_s)} · player {target != null ? seconds(target) : 'time unavailable'}</p>}
    </div>}
    <VideoFrameNavigator set={{ ...set, packets }} token={token} mediaReady={mediaReady} selected={selected} onSelect={chooseFrame} />
    <div className="mt-6 space-y-7">{packets.map((packet) => <article key={packet.id} id={`media-evidence-${packet.id}`} className={evidenceCard}>
      <h3 className="m-0 text-body font-medium"><span className="font-mono text-accent mr-3">{packet.id}</span>{packet.title}</h3>
      {!packet.picture_hold && <p className="mt-2 mb-0 text-body">{packet.observation}</p>}
      {packet.face_single_boundary && <p className="mt-2 mb-0 text-small text-text-muted">
        Transition at <strong>{seconds(packet.face_single_boundary.time_s)}</strong> on the source clock · frame {packet.face_single_boundary.boundary}.
      </p>}
      {!packet.scene_return && !packet.lip_timing_review && !packet.face_single_boundary && <p className="mt-2 mb-0 text-small text-text-muted">{packet.picture_hold
        ? <>{packet.picture_hold.frame_count} consecutive decoded frames have identical pixels over <strong>{seconds(packet.picture_hold.span_s)}</strong> between the first and last source timestamps.</>
        : packet.picture_sequence
        ? <>{occurrenceLabel(packet.picture_sequence, 'first')} <strong>{seconds(packet.picture_sequence.first.start_stream_s)}–{seconds(packet.picture_sequence.first.end_stream_s)}</strong> · {occurrenceLabel(packet.picture_sequence, 'second')} <strong>{seconds(packet.picture_sequence.second.start_stream_s)}–{seconds(packet.picture_sequence.second.end_stream_s)}</strong> on the source clock.</>
        : packet.mouth_local_hold && packet.window
        ? <>Measured mouth-region hold <strong>{seconds(packet.window.start_stream_s)}–{seconds(packet.window.end_stream_s)}</strong> on the source clock. The end is exclusive. Source frames {packet.mouth_local_hold.frames[0]}–{packet.mouth_local_hold.frames[1] - 1}.</>
        : packet.eye_local_hold && packet.window
        ? <>Measured eye-region hold <strong>{seconds(packet.window.start_stream_s)}–{seconds(packet.window.end_stream_s)}</strong> on the source clock. The end is exclusive. Source frames {packet.eye_local_hold.frames[0]}–{packet.eye_local_hold.frames[1] - 1}.</>
        : packet.whole_face_review
        ? packet.whole_face_review.schema === 'same-frame-face-samples/1'
          ? 'Both readings were combined at each jointly measured face. The three frames below show representative inputs.'
          : packet.whole_face_review.schema === 'relative-face-detail-samples/1'
          ? 'The check compares detail and visual appearance inside the face with the surrounding picture. The three frames below show representative inputs.'
          : 'Two checks compared faces at six sampled positions across the clip. The three frames shown below are examples of those inputs.'
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
        <p className="mt-2 mb-0">Source timestamps {seconds(packet.window.start_stream_s)}–{seconds(packet.window.end_stream_s)}. {packet.whole_face_review && packet.whole_face_review.schema !== 'relative-face-detail-samples/1' ? 'Planned frame indices' : 'Input frame indices'}: {packet.window.frame_indices.join(', ')}.</p>
        {packet.eye_local_hold && <p className="mt-2 mb-0">Source PTS [{packet.window.start_pts}, {packet.window.end_pts}) at {packet.window.time_base}.
          {' '}Measured crop: x {packet.eye_local_hold.box[0]}, y {packet.eye_local_hold.box[1]}, {packet.eye_local_hold.box[2]} × {packet.eye_local_hold.box[3]} pixels.</p>}
        <p className="mt-2 mb-0">{packet.face_return?.edges || (packet.face_return && ['face-short-return/9', 'face-short-return/10', 'face-short-return/11', 'face-short-return/12', 'face-short-return/13'].includes(packet.face_return.method ?? ''))
          ? `Nearby motion was reviewed in ${packet.face_return.scene_frame_inputs?.length ?? 0} native pictures. Face appearances were measured in ${packet.face_return.frame_inputs.length} sampled pictures.`
          : packet.mouth_local_hold ? 'The recorded window lists the source frames used for the mouth-region measurement.'
          : packet.eye_local_hold ? 'The recorded window lists the source frames used for the eye-region measurement.'
          : packet.whole_face_review?.schema === 'relative-face-detail-samples/1' ? 'These sampled faces define the measured track. The timing of any edit remains undetermined.'
          : packet.whole_face_review ? 'These positions show sample coverage. Some frames may lack a usable face, and intervals between samples were unassessed.'
          : 'Only these frames were input to this window. This does not establish continuous coverage between them.'}</p>
      </details>}
      {packet.face_return?.edges && <div className="mt-4 rounded-md border border-border p-3 text-small">
        <h4 className="m-0 font-medium">Source frames compared at each boundary</h4>
        <p className="mt-2 mb-0">Start of change · frames {packet.face_return.edges.entry.frame_indices.join(', ')}</p>
        <p className="mt-1 mb-0">End of change · frames {packet.face_return.edges.exit.frame_indices.join(', ')}</p>
        <p className="mt-2 mb-0 text-caption text-text-muted">At each boundary the check compared the first three frames with the last three.</p>
      </div>}
      {!packet.picture_hold && !packet.picture_sequence && !packet.whole_face_review && !packet.scene_return && !packet.lip_timing_review && !packet.face_single_boundary && !packet.eye_local_hold && <ContextClip packet={packet} set={set} token={token} ready={mediaReady} />}
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
      {packet.lip_timing_review && <div className="mt-4 space-y-3">
        <EvidencePicture set={set} artifact={packet.artifacts.lip_timing_waveform} token={token} ready={mediaReady}
          alt={packet.lip_timing_review.speech_activity_gap ? 'Waveform around the reduced speech activity interval'
            : packet.lip_timing_review.silent_mouth ? 'Waveform around the near-silent soundtrack interval' :
            (packet.lip_timing_review.articulation || packet.lip_timing_review.motion_sound_agreement)
              ? 'Waveform during the speech and mouth mismatch' :
            'Waveform on the audio source clock during the timing change'} chart />
        <p className="m-0 text-caption text-text-muted">{packet.lip_timing_review.speech_activity_gap
          ? <>The waveform and playback use a mix of the original channels. The highlighted interval spans samples [{packet.lip_timing_review.audio_source_pts[0]}, {packet.lip_timing_review.audio_source_pts[1]}) at 16 kHz.</>
          : 'The shaded interval matches the playback below. The waveform is scaled to make quiet speech visible.'}</p>
        <EvidenceAudio set={set} artifact={packet.artifacts.lip_timing_audio} token={token} ready={mediaReady}
          alt={packet.lip_timing_review.speech_activity_gap ? 'Listen to the original channels mixed during reduced speech activity'
            : packet.lip_timing_review.silent_mouth ? 'Listen to the near-silent soundtrack interval' :
            (packet.lip_timing_review.articulation || packet.lip_timing_review.motion_sound_agreement)
              ? 'Listen to the soundtrack during the speech and mouth mismatch' :
            'Listen to the soundtrack during the measured timing change'} />
      </div>}
      {packet.mouth_local_hold && packet.artifacts.mouth_hold_waveform && packet.artifacts.mouth_hold_audio &&
        <div className="mt-4 space-y-3">
          <EvidencePicture set={set} artifact={packet.artifacts.mouth_hold_waveform} token={token} ready={mediaReady}
            alt="Waveform during the measured mouth-region hold" chart />
          <p className="m-0 text-caption text-text-muted">{packet.artifacts.mouth_hold_waveform.time_axis === 'source_pts'
            ? 'The waveform uses the same source clock as the pictures above.'
            : 'The waveform axis starts from the decoded audio origin. The picture range above uses the source clock.'}</p>
          <EvidenceAudio set={set} artifact={packet.artifacts.mouth_hold_audio} token={token} ready={mediaReady}
            alt="Listen to the measured mouth-region hold" />
        </div>}
      <p className="mt-4 mb-0 text-small">{packet.interpretation}</p><EvidenceMeasurements values={packet.measurements} />
      <EvidenceLimits lines={packet.limitations} /><EvidenceReferences packet={packet} />
    </article>)}</div>
  </div>
}
