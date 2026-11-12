import { EvidenceAudio, EvidencePicture } from './EvidenceMedia'
import { rationalSeconds, seconds, type AudioEvidenceSet, type StereoChannelContext } from './multimediaPackets'

const unassessed = {
  mono: 'The first audio stream is mono. Channel comparison does not apply.',
  unsupported: 'The first audio stream does not have a supported two-channel layout.',
  too_short: 'The stereo input has less than one complete second of decoded audio.',
  unavailable: 'The stereo measurements could not be aligned to the recorded source clock.',
}

const channels = { left: 'Decoded left channel', right: 'Decoded right channel', mono: 'Exact analyzed mono' } as const

export default function AudioStereoContext({ set, token, mediaReady, videoOriginS, pauseAudio }: {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; videoOriginS?: number | null; pauseAudio: (playing?: HTMLAudioElement) => void
}) {
  const context = set.stereo_context
  if (!context) return null
  const hasDownmix = context.schema === 'chipmunk.stereo-channel-context/2'
  const bound = (context.schema === 'chipmunk.stereo-channel-context/1'
    || (hasDownmix && Boolean(context.mono_pcm_sha256) && context.mono_pcm_sha256 === set.source.pcm_sha256))
    && context.evidence_type === 'context' && context.affects_overall_confidence === false
    && context.source_sha256 === set.source.sha256
    && context.source_stream_index === set.source.source_stream_index
    && context.source_channel_count === set.source.source_channels
    && context.sample_rate === set.source.sample_rate && context.sample_rate > 0
  const range = (window: StereoChannelContext['windows'][number]) =>
    seconds(window.start_sample / context.sample_rate) + '–' + seconds(window.end_sample / context.sample_rate)
  const strong = context.windows.filter((window) => window.status === 'strong_imbalance')
  const attenuated = hasDownmix ? context.windows.filter((window) => window.downmix?.strong_attenuation) : []
  const listening = set.stereo_listening
  const comparisons = hasDownmix && listening?.schema === 'chipmunk.stereo-listening/1'
    && listening.affects_overall_confidence === false ? listening.windows.flatMap(({ stereo_window_index, artifacts }) => {
      const window = context.windows[stereo_window_index]
      return window ? [{ window, artifacts }] : []
    }) : []
  return <section id="media-evidence-stereo" aria-labelledby="media-evidence-stereo-heading" className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="media-evidence-stereo-heading" className="m-0 text-body font-medium">Stereo channel comparison</h3>
    {!bound ? <p role="alert" className="mt-2 mb-0 text-small text-text-muted">The channel measurements do not match this analyzed audio input.</p>
      : context.status === 'measured' || context.status === 'quiet' ? <>
        <p className="mt-2 mb-0 text-small text-text-muted">Compared {context.windows.length} complete one-second {context.windows.length === 1 ? 'window' : 'windows'} in the first {context.max_seconds} seconds {hasDownmix ? 'using the decoded stereo channels and the analyzed mono mix' : 'before mono mixing'}. Stereo placement, mixing and microphone position can all affect these readings. They do not contribute to the AI assessment.</p>
        {context.status === 'quiet'
          ? <p className="mt-2 mb-0 text-small">All measured windows fell below the {context.quiet_floor_dbfs} dBFS review floor.</p>
          : <p className="mt-2 mb-0 text-small">{strong.length} {strong.length === 1 ? 'window had' : 'windows had'} a strong channel-level difference{strong.length === 0 ? '.' : <>. {strong.slice(0, 3).map((window) => range(window) + ' (' + window.louder_channel + ' louder)').join(', ')}{strong.length > 3 ? ', with more in the table.' : '.'}</>}</p>}
        {attenuated.length > 0 && <p className="mt-2 mb-0 text-small">
          <strong>Strong level drop in the analyzed mono mix.</strong>{' '}
          Decoded audio {attenuated.slice(0, 3).map(range).join(', ')}{attenuated.length > 3 ? ', with more in the table.' : '.'}{' '}
          Sounds present in stereo may be harder to hear in the analyzed copy.
        </p>}
        {comparisons.length > 0 && <details className="mt-3 text-small"
          onPlayCapture={(event) => {
            if (event.target instanceof HTMLAudioElement) pauseAudio(event.target)
          }}
          onToggle={(event) => {
            if (!event.currentTarget.open) event.currentTarget.querySelectorAll('audio').forEach((audio) => {
              if (!audio.paused) audio.pause()
            })
          }}>
          <summary className="cursor-pointer font-medium">Compare stereo and analyzed mono</summary>
          <p className="mt-2 mb-0 text-text-muted">Playback covers only the {comparisons.length === 1 ? 'selected interval' : `${comparisons.length} selected intervals`} below. Each clip starts at zero within its interval. These comparisons carry no weight in the AI assessment.</p>
          {comparisons.map(({ window, artifacts }) => <article key={window.start_sample} className="mt-4 border-t border-border pt-3">
            <h4 className="m-0 font-medium">Decoded audio {range(window)}</h4>
            <p className="mt-1 mb-0 font-mono text-caption text-text-muted">Samples [{window.start_sample}, {window.end_sample}) at {context.sample_rate} Hz.</p>
            {window.source_timestamp_ranges.map((item) => {
              const start = rationalSeconds(item.source_start_pts, item.time_base)
              const end = rationalSeconds(item.source_end_pts, item.time_base)
              return <p key={item.start_sample} className="m-0 text-caption text-text-muted">
                Source PTS [{item.source_start_pts}, {item.source_end_pts}) at {item.time_base}.
                {typeof videoOriginS === 'number' && Number.isFinite(videoOriginS) && start !== null && end !== null
                  && start >= videoOriginS && ` Video playback ${seconds(start - videoOriginS)}–${seconds(end - videoOriginS)}.`}
              </p>
            })}
            <figure className="m-0 mt-3">
              <EvidencePicture set={set} artifact={artifacts.waveform} token={token} ready={mediaReady} chart
                alt={`Decoded left and right channels and exact analyzed mono waveforms, ${range(window)}`} />
            </figure>
            <div className="mt-3 grid gap-3 min-[900px]:grid-cols-3">
              {(['left', 'right', 'mono'] as const).map((channel) => <EvidenceAudio key={channel} set={set}
                artifact={artifacts[channel]} token={token} ready={mediaReady} alt={channels[channel]} />)}
            </div>
          </article>)}
        </details>}
        <details className="mt-3 text-small text-text-muted">
          <summary className="cursor-pointer">Channel levels and source timestamps</summary>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-caption">
              <thead><tr><th scope="col">Decoded range</th><th scope="col">Source timestamps (PTS)</th><th scope="col">L / R{hasDownmix ? ' / mono' : ''} (dBFS)</th><th scope="col">Gap (dB)</th><th scope="col">Correlation</th><th scope="col">Observation</th></tr></thead>
              <tbody>{context.windows.map((window) => <tr className="border-t border-border" key={window.start_sample}>
                <th scope="row" className="py-2 font-normal whitespace-nowrap">{range(window)}</th>
                <td>{window.source_timestamp_ranges.map((item) => {
                  const start = rationalSeconds(item.source_start_pts, item.time_base)
                  const end = rationalSeconds(item.source_end_pts, item.time_base)
                  return <div className="whitespace-nowrap" key={item.start_sample}>
                    {start !== null && end !== null && <div>{seconds(start)}–{seconds(end)}</div>}
                    <div>[{item.source_start_pts}, {item.source_end_pts}) at {item.time_base}</div>
                    {typeof videoOriginS === 'number' && Number.isFinite(videoOriginS) && start !== null && end !== null
                      && start >= videoOriginS && <div>Video playback {seconds(start - videoOriginS)}–{seconds(end - videoOriginS)}</div>}
                  </div>
                })}</td>
                <td className="whitespace-nowrap">{window.left_rms_dbfs} / {window.right_rms_dbfs}{hasDownmix && window.downmix && ' / ' + window.downmix.mono_rms_dbfs}</td>
                <td>{window.level_gap_db ?? '—'}</td>
                <td>{window.correlation ?? '—'}</td>
                <td>{window.status === 'strong_imbalance' ? window.louder_channel + ' louder' : window.status === 'quiet' ? 'Below review floor' : 'Measured'}
                  {hasDownmix && window.downmix?.strong_attenuation && <div>{window.downmix.mono_rms_dbfs === context.silence_floor_dbfs
                    ? 'Mono is digitally silent' : window.downmix.attenuation_db + ' dB quieter in mono'}</div>}
                </td>
              </tr>)}</tbody>
            </table>
          </div>
          <p className="mt-2 mb-0 text-caption">A gap of {context.strong_level_gap_db} dB or more marks a strong difference for review. {context.silence_floor_dbfs} dBFS represents digital silence. Correlation measures signal similarity on a scale of −1 to +1.{hasDownmix && <> A strong mono level drop means the analyzed mono input is at least {context.strong_downmix_attenuation_db} dB quieter than the louder stereo channel.</>}</p>
        </details>
      </> : <p className="mt-2 mb-0 text-small text-text-muted">{unassessed[context.status]}</p>}
  </section>
}
