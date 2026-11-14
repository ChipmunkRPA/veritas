import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import { ContextClip } from './VideoEvidenceContent'
import { rationalSeconds, seconds, type VideoAVChannel, type VideoAVPacket, type VideoEvidenceSet, type VideoTimingStatus } from './multimediaPackets'

const signed = (value: number) => `${value > 0 ? '+' : ''}${value}`
const hasTimingChange = (packet: VideoAVPacket) => packet.measurement.overlaps_offset_step || packet.measurement.overlaps_correlation_collapse
const unassessedTiming: Partial<Record<VideoTimingStatus, string>> = {
  insufficient_speech: 'Too little speech was detected for this window.',
  missing_picture_timestamps: 'No source picture was available close enough to every audio tick.',
  insufficient_audio_context: 'The full timing search would extend beyond the audio.',
  no_face_track: 'No tracked face was available.',
  missing_landmarks: 'No single face track had usable mouth landmarks across the entire window.',
  track_limit: 'The window was outside the selected face tracks.',
  missing_mouth_input: 'Some source mouth crops could not be verified.',
  unchanged_mouth_pixels: 'The mouth crops were identical across the window.',
  flat_mouth_proxy: 'The mouth-opening measurement stayed flat.',
  speaker_unresolved: 'Speech could not be linked to a visible mouth for this window.',
}

function intervalLabel(range: VideoAVPacket['anchor']['source_timestamp_ranges'][number]) {
  const start = rationalSeconds(range.source_start_pts, range.time_base)
  const end = rationalSeconds(range.source_end_pts, range.time_base)
  return start !== null && end !== null ? `${seconds(start)}–${seconds(end)}` : 'time conversion unavailable'
}

function SourceInterval({ packet }: { packet: VideoAVPacket }) {
  return <div className="mt-3 text-caption text-text-muted">
    {packet.anchor.source_timestamp_ranges.map((range, index) => <p key={index} className="mt-1 mb-0 font-mono">Selected source interval: {intervalLabel(range)}
      {' '}· PTS [{range.source_start_pts}, {range.source_end_pts}) at {range.time_base}</p>)}
    <p className="mt-1 mb-0">Analyzed audio samples [{packet.interval.start_sample}, {packet.interval.end_sample});
      {' '}search context [{packet.context_interval.start_sample}, {packet.context_interval.end_sample}). WAV controls begin at zero inside each clip.</p>
  </div>
}

export default function VideoAVEvidenceContent({ set, channel, token, mediaReady }: {
  set: VideoEvidenceSet; channel: VideoAVChannel; token: string; mediaReady: boolean
}) {
  const coverage = channel.coverage
  const timingWindows = coverage?.timing_windows?.windows
  const unmeasured = timingWindows?.filter((row) => row.status !== 'measured')
  const flaggedWindows = channel.packets.filter(hasTimingChange)
  return <div data-testid="video-av-evidence">
    <p className="mt-2 mb-0 text-small text-text-muted">Compare measured mouth appearance and audio energy on their original clocks. Correlation and timing differences do not prove AI generation, editing, or who is speaking.</p>
    {flaggedWindows.length > 0 ? <nav aria-label="Flagged timing intervals" className="mt-4 rounded-md border border-border p-4">
      <h4 className="m-0 text-small font-medium">Timing intervals to review</h4>
      <p className="mt-2 mb-0 text-small text-text-muted">These measured windows overlap a flagged change. They can extend beyond the change itself.</p>
      <ul className="mt-2 mb-0 space-y-1 text-small">{flaggedWindows.map((packet) => <li key={packet.id}>
        <a className="text-accent underline" href={`#media-evidence-${packet.id}`}>{packet.id} · {packet.anchor.source_timestamp_ranges.map(intervalLabel).join(' · ')}</a>
        {' '}· {packet.measurement.overlaps_offset_step ? 'Offset change' : 'Drop in mouth and audio agreement'}
      </li>)}</ul>
    </nav> : channel.packets.length > 0 && <p className="mt-3 mb-0 text-small text-text-muted">The displayed windows contain no flagged timing change. A consistent delay can still be present, and a redrawn mouth can match its audio.</p>}
    {coverage ? <div className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">{coverage.selected_windows} of {coverage.recorded_windows} recorded alignment windows are shown; {coverage.omitted_windows} are omitted.
        {' '}Flagged windows are shown first, followed by measured examples in time order. Overlapping windows are not independent confirmations.</p>
      <p className="mt-2 mb-0">{coverage.recorded_windows > 0 ? <>The nearest source frame is measured at each {channel.source.tick_rate ?? 25} Hz tick.</>
        : 'No measured alignment window is available.'} Repeated frames and gaps do not establish continuous analysis coverage.</p>
      {coverage.unsupported_checks.map((check, index) => <p key={index} className="mt-2 mb-0">Unmapped check: {check.reason} It does not establish synchronization.</p>)}
    </div> : <p className="mt-3 text-small text-text-muted">Alignment coverage is unavailable. Unmeasured intervals have not been cleared.</p>}
    {timingWindows && unmeasured && <section aria-labelledby="speech-mouth-timing-coverage" className="mt-4 rounded-md border border-border p-4 text-small">
      <h4 id="speech-mouth-timing-coverage" className="m-0 font-medium">Speech and mouth timing coverage</h4>
      <p className="mt-2 mb-0 text-text-muted">{timingWindows.length - unmeasured.length} of {timingWindows.length} scheduled one-second windows reached a timing measurement. These overlapping windows are scheduled within the decoded soundtrack. A measurement can still yield an inconclusive timing result.</p>
      {unmeasured.length > 0 && <details open={unmeasured.length <= 12} className="mt-3">
        <summary className="cursor-pointer">Unassessed windows on the source clock ({unmeasured.length})</summary>
        <ul className="mt-2 mb-0 space-y-2 text-text-muted">{unmeasured.map((row) => <li key={row.interval.start_sample}>
          <span className="font-mono">{intervalLabel(row.interval)}</span> · {unassessedTiming[row.status] ?? 'Timing was unassessed.'}
        </li>)}</ul>
      </details>}
    </section>}
    <div className="mt-6 space-y-7">{channel.packets.map((packet) => {
      const measurement = packet.measurement
      const flagged = hasTimingChange(packet)
      const sampleRate = packet.anchor.sample_rate
      return <article key={packet.id} id={`media-evidence-${packet.id}`} className={evidenceCard}>
        <h4 className="m-0 text-body font-medium"><span className="font-mono text-accent mr-3">{packet.id}</span>{packet.title}</h4>
        <p className="mt-2 mb-0 text-body">{packet.observation}</p>
        <p className="mt-2 mb-0 text-small text-text-muted">{flagged ? 'This window overlaps a recorded timing or correlation change.' : 'Informational measured example; no recorded timing-change or correlation-collapse segment overlaps this window.'}</p>
        <SourceInterval packet={packet} />
        <EvidenceMeasurements values={packet.measurements} />
        <p className="mt-2 mb-0 text-small text-text-muted">The selected lag is the best match among {measurement.lag_profile.length} tested shifts. Positive means the audio is later than the picture. The correlation does not estimate the probability of manipulation.</p>
        {measurement.peak_correlation < measurement.minimum_trusted_peak && <p className="mt-2 mb-0 text-small text-text-muted">The strongest match is below the recorded threshold of {measurement.minimum_trusted_peak}. This window cannot support a reliable timing estimate.</p>}
        <div className={`${evidenceGrid} mt-4`}>
          <EvidenceAudio set={set} artifact={packet.artifacts.selected_clip} token={token} ready={mediaReady} alt={`${packet.id}: exact unshifted source audio window`} />
          <EvidenceAudio set={set} artifact={packet.artifacts.context_clip} token={token} ready={mediaReady} alt={`${packet.id}: exact audio search context`} />
        </div>
        <p className="mt-2 mb-0 text-caption text-text-muted">Selected audio lasts {seconds((packet.interval.end_sample - packet.interval.start_sample) / sampleRate)}. The context includes the additional audio actually searched around this interval.
          {' '}Audio is not shifted to the selected lag. These controls and the silent visual clip play independently; browser playback is not a synchronization measurement.</p>
        <ContextClip packet={packet} set={set} token={token} ready={mediaReady} />
        <div className={`${evidenceGrid} mt-4`}>
          {(['lag_profile', 'alignment', 'waveform'] as const).map((role) => packet.artifacts[role] && <figure key={role} className="m-0 min-w-0">
            <EvidencePicture set={set} artifact={packet.artifacts[role]} token={token} ready={mediaReady} chart
              alt={`${packet.id}: ${role === 'lag_profile' ? 'complete measured lag search' : role === 'alignment' ? 'mouth and audio comparison curves' : 'audio waveform and selected interval'}`} />
            <figcaption className="mt-2 text-caption text-text-muted">{role === 'lag_profile'
              ? 'Every measured lag candidate is retained, including alternatives to the selected peak.'
              : role === 'alignment' ? 'Curves are standardized separately for comparison; raw measurements remain in the record.'
                : 'Waveform provides listening context and is not independent evidence of synthesis.'}</figcaption>
          </figure>)}
        </div>
        <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">All {measurement.lag_profile.length} measured lag candidates</summary>
          <div className="overflow-x-auto"><table className="mt-3 w-full text-left text-caption">
            <thead><tr><th className="pr-4">Audio lag</th><th className="pr-4">Correlation</th><th>Selection</th></tr></thead>
            <tbody>{measurement.lag_profile.map((point) => <tr key={point.offset_ticks}>
              <td className="pr-4 font-mono">{signed(point.offset_ticks * 1000 / (channel.source.tick_rate ?? 25))} ms</td>
              <td className="pr-4 font-mono">{point.correlation.toFixed(6)}</td>
              <td>{point.offset_ticks === measurement.offset_ticks ? 'Selected maximum' : 'Compared alternative'}</td>
            </tr>)}</tbody>
          </table></div>
        </details>
        <div className="mt-5 space-y-5">{packet.frames.map((frame) => <div key={frame.id} className="rounded-md border border-border p-3">
          <p className="m-0 mb-3 text-small font-medium">Frame {frame.anchor.frame_index} · measured tick {frame.av_tick}</p>
          <div className="grid grid-cols-3 gap-4 max-[799px]:grid-cols-1">
            {(['original', 'crop', 'measured_patch'] as const).map((role) => {
              const reference = frame.artifact_refs[role], artifact = reference && packet.artifacts[reference]
              return artifact && <figure key={role} className="m-0 min-w-0">
                <EvidencePicture set={set} artifact={artifact} token={token} ready={mediaReady}
                  alt={`${frame.id}: ${role === 'original' ? 'exact original video frame' : role === 'crop' ? 'exact source mouth crop' : 'exact measured grayscale mouth patch'}`} />
                <figcaption className="mt-2 text-caption text-text-muted">{role === 'original' ? 'Recorded native frame pixels.'
                  : role === 'crop' ? 'Original mouth-region crop used to derive the measurement; not an edit boundary.'
                    : 'Downsampled grayscale input actually measured. Appearance is a proxy for mouth activity, not a verified speaker identity.'}</figcaption>
              </figure>
            })}
          </div>
          <p className="mt-3 mb-0 font-mono text-caption text-text-muted">Source {seconds(frame.anchor.stream_time_s)} · PTS {frame.anchor.pts} at {frame.anchor.time_base}.
            {' '}Representative frames have no individual AI score.</p>
        </div>)}</div>
        <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">All {packet.window.frame_inputs.length} measured mouth inputs and exact clocks</summary>
          <p className="mt-2 mb-0">Ticks [{packet.window.start_tick}, {packet.window.end_tick}) preserve the actual frame selections, including any repeated source frame.</p>
          <div className="mt-3 grid grid-cols-3 gap-3 max-[799px]:grid-cols-2 max-[479px]:grid-cols-1">{packet.window.frame_inputs.map((input) => <figure key={input.tick} className="m-0 min-w-0">
            <EvidencePicture set={set} artifact={packet.artifacts[input.artifact_ref]} token={token} ready={mediaReady}
              alt={`${packet.id}: measured mouth input at tick ${input.tick}`} />
            <figcaption className="mt-2 text-caption text-text-muted">Tick {input.tick} · frame {input.frame_index}<br />
              Video PTS {input.frame_pts} at {channel.source.video_time_base ?? 'unrecorded time base'}<br />
              Audio PTS {input.source_audio_pts} at {input.source_audio_time_base}<br />Mouth measurement {input.mouth_value.toFixed(6)}</figcaption>
          </figure>)}</div>
        </details>
        <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">All {measurement.audio_curve.length} measured audio-energy blocks</summary>
          <p className="mt-2 mb-0">RMS is signal energy, not an AI or speaker-identity score. The extended blocks belong to the actual lag search context.</p>
          <div className="mt-3 max-h-64 overflow-auto"><table className="w-full text-left text-caption">
            <thead><tr><th>Tick</th><th>Analyzed sample interval</th><th>RMS energy</th></tr></thead>
            <tbody>{measurement.audio_curve.map((point) => <tr key={point.tick}>
              <td>{point.tick}</td><td className="font-mono">[{point.sample_start}, {point.sample_end})</td><td className="font-mono">{point.rms.toFixed(6)}</td>
            </tr>)}</tbody>
          </table></div>
        </details>
        <p className="mt-4 mb-0 text-small">{packet.interpretation}</p>
        <EvidenceLimits lines={packet.limitations} /><EvidenceReferences packet={packet} />
      </article>
    })}</div>
  </div>
}
