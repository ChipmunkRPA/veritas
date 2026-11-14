import type { Finding, SignalV2 } from '../../lib/api/queries'
import type { ImageEvidencePacket, ImageEvidenceSet } from '../evidence/imagePackets'
import {
  hasNestedVideoEvidence, type AudioEvidencePacket, type DocumentEvidencePacket, type MultimediaEvidenceSet,
  type PacketBase, type SampleInterval, type VideoAVPacket, type VideoEvidenceFrame, type VideoEvidencePacket, type VideoEvidenceSet,
} from '../evidence/multimediaPackets'
import { anchorOf } from '../signals/ledger'

export type FindingSupportEvidence = ImageEvidenceSet | MultimediaEvidenceSet
export interface FindingSupportLink {
  id: string
  label: string
  href: string
  /** Recorded wording, not a new interpretation. Anonymize before displaying it. */
  summary: string
  association: 'finding' | 'shared_check' | 'check'
  scope: 'image_region' | 'video_frames' | 'audio_interval' | 'audio_coverage' | 'av_interval' | 'text_passage' | 'check'
  scopeNote: string
}
export interface FindingSupportResult { packets: FindingSupportLink[]; checks: FindingSupportLink[]; voiceCoverage?: FindingSupportLink }

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const count = (value: unknown): value is number => finite(value) && value >= 0
const index = (value: unknown): value is number => count(value) && Number.isSafeInteger(value)
const seconds = (value: number) => `${value.toFixed(3)} s`
const interval = (value: SampleInterval | undefined) => value && index(value.start_sample) && index(value.end_sample)
  && value.end_sample > value.start_sample
const references = (finding: Finding) => new Set([
  ...(finding.evidence_ids ?? []), ...(finding.sentences ?? []).flatMap((sentence) => sentence.evidence_ids ?? []),
])

/** Resolve only records that exist. Display labels and partial identifiers are not identity keys. */
function linkedChecks(finding: Finding, signals: readonly SignalV2[]) {
  const refs = references(finding)
  return signals.filter((signal, position) => signals.findIndex((other) => other.id === signal.id) === position
    && (refs.has(signal.id) || (!!signal.legacy?.v1_id && refs.has(signal.legacy.v1_id))
      || refs.has(`detector:${signal.detector_id}`)))
}

type LocatedPacket = { packet: PacketBase | ImageEvidencePacket; label: string; scope: FindingSupportLink['scope']; note: string }

function imageLocation(packet: ImageEvidencePacket, set: ImageEvidenceSet): LocatedPacket | null {
  const box = packet.region
  if (!/^E\d+$/.test(packet.id) || !box || ![box.x, box.y, box.w, box.h].every(finite)
    || box.x < 0 || box.y < 0 || box.w <= 0 || box.h <= 0
    || (finite(set.source.width) && box.x + box.w > set.source.width)
    || (finite(set.source.height) && box.y + box.h > set.source.height)) return null
  return { packet, label: `${packet.id} · Region x ${box.x}, y ${box.y} · ${box.w} × ${box.h} px`, scope: 'image_region',
    note: 'Original crop and inspection overlay. This region is an inspection location, not proof that every highlighted pixel was edited.' }
}

function validFrame(frame: VideoEvidenceFrame) {
  return frame?.anchor && index(frame.anchor.frame_index) && finite(frame.anchor.stream_time_s)
}

function videoLocation(packet: VideoEvidencePacket): LocatedPacket | null {
  if (!/^V\d+$/.test(packet.id)) return null
  const frames = (packet.frames ?? []).filter(validFrame)
  const frame = frames.find((item) => item.is_peak) ?? frames[0]
  if (!frame) return null
  const cut = packet.face_spatial_cut
  if (cut && finite(cut.start_stream_s) && finite(cut.end_stream_s) && cut.end_stream_s > cut.start_stream_s
    && packet.face_spatial_cut_frames?.some(validFrame)) {
    return { packet, label: `${packet.id} · Picture-cut context ${seconds(cut.start_stream_s)}–${seconds(cut.end_stream_s)} (source clock)`,
      scope: 'video_frames', note: 'Picture detail outside the face changed at both ends. The three original full frames show the surrounding picture.' }
  }
  const boundary = packet.face_boundary
  if (boundary?.method === 'face-appearance-paired-edge/1'
    && finite(boundary.start_stream_s) && finite(boundary.end_stream_s) && boundary.end_stream_s > boundary.start_stream_s
    && packet.face_boundary_frames?.some(validFrame)) {
    return { packet, label: `${packet.id} · Refined face-change range ${seconds(boundary.start_stream_s)}–${seconds(boundary.end_stream_s)} (source clock)`,
      scope: 'video_frames', note: 'Two face transitions with limited nearby pixel motion bracket the range. The end is exclusive.' }
  }
  const short = packet.face_return
  if (short && finite(short.start_stream_s) && finite(short.end_stream_s) && short.end_stream_s > short.start_stream_s) {
    return { packet, label: `${packet.id} · Brief face-change range ${seconds(short.start_stream_s)}–${seconds(short.end_stream_s)} (source clock)`,
      scope: 'video_frames', note: 'Three source frames show the face before, during and after the measured change. The end is exclusive.' }
  }
  const face = packet.face_sequence ?? packet.face_interval
  if (face && finite(face.start_stream_s) && finite(face.end_stream_s) && face.end_stream_s > face.start_stream_s) {
    return { packet, label: `${packet.id} · Face review ${seconds(face.start_stream_s)}–${seconds(face.end_stream_s)} (source clock)`,
      scope: 'video_frames', note: 'Before, during and after frames from one tracked face. The range marks sampled responses.' }
  }
  let location = `Frame ${frame.anchor.frame_index} · ${seconds(frame.anchor.stream_time_s)} (source clock)`
  const window = packet.window
  if (packet.claim_scope === 'sampled_window' && window && finite(window.start_stream_s) && finite(window.end_stream_s)
    && window.end_stream_s >= window.start_stream_s) {
    location = `Sampled window ${seconds(window.start_stream_s)}–${seconds(window.end_stream_s)} (source clock)`
  }
  return { packet, label: `${packet.id} · ${location}`, scope: 'video_frames',
    note: `${frames.length} recorded frame${frames.length === 1 ? '' : 's'} with source timestamps. A sampled-window response does not identify which individual frames were manipulated.` }
}

function audioLocation(packet: AudioEvidencePacket | VideoAVPacket, av = false): LocatedPacket | null {
  const rate = packet.anchor?.sample_rate
  if (!(av ? /^AV\d+$/ : /^A\d+$/).test(packet.id) || !interval(packet.interval) || !finite(rate) || rate <= 0) return null
  const start = packet.interval.start_sample / rate, end = packet.interval.end_sample / rate
  // The rendered audio cards use these frozen seconds. Never link a contradictory clock label.
  if ('start_s' in packet && (!finite(packet.start_s) || !finite(packet.end_s)
    || Math.abs(packet.start_s - start) > 1 / rate || Math.abs(packet.end_s - end) > 1 / rate)) return null
  return { packet, label: `${packet.id} · ${av ? 'Audio/picture timing' : 'Audio'} ${seconds(start)}–${seconds(end)} (decoded audio)`,
    scope: av ? 'av_interval' : 'audio_interval',
    note: `Samples [${packet.interval.start_sample}, ${packet.interval.end_sample}) at ${rate} Hz. ${av
      ? 'Timing measurements and sampled frames describe alignment; they do not by themselves establish a deepfake.'
      : 'Inspect the waveform and listen to the selected interval and surrounding context. The window response is an inspection score, not the probability that this interval was altered.'}` }
}

function documentLocation(packet: DocumentEvidencePacket, chars?: number): LocatedPacket | null {
  const focus = packet.focus
  if (!/^T\d+$/.test(packet.id) || !focus || !index(focus.start) || !index(focus.end) || focus.end <= focus.start
    || (index(chars) && focus.end > chars)) return null
  const locations = new Set<string>()
  for (const anchor of packet.anchors ?? []) {
    if (index(anchor.page) && anchor.page > 0) locations.add(`Page ${anchor.page}`)
    else if (index(anchor.paragraph_index)) locations.add(`Paragraph ${anchor.paragraph_index + 1}`)
    else if (index(anchor.paragraph) && anchor.paragraph > 0) locations.add(`Paragraph ${anchor.paragraph}`)
  }
  const position = [...locations].join(', ')
  const measured = packet.inspection_scope
  if (measured && (!measured.spans.length || measured.spans.some(({ start, end }) =>
    !index(start) || !index(end) || end <= start || (index(chars) && end > chars)))) return null
  const matched = !!packet.matched_context_review
  const ranges = (matched ? [focus] : measured?.spans ?? [focus]).map(({ start, end }) => `[${start}, ${end})`).join(', ')
  const label = matched ? 'Selected sentence' : measured?.basis === 'scored_context' ? 'Scored context' : measured?.basis === 'ratio_reduction' ? 'Measured ratio range' : 'Characters'
  return { packet, label: `${packet.id} · ${position ? `${position} · ` : ''}${label} ${ranges}`, scope: 'text_passage',
    note: matched ? 'Positions use Unicode code points. Open the evidence to see the selected sentence and the three sentences read together.' : measured
      ? 'Positions use Unicode code points in the extracted text. The marked range records what a check measured. It does not establish authorship.'
      : 'Positions use Unicode code points in the extracted text. The passage and recorded responses provide context. A response is not proof of authorship.' }
}

/** Keep nested links aligned with the channels the evidence workspace can actually render. */
function videoLocations(set: VideoEvidenceSet): (LocatedPacket | null)[] {
  const result = set.packets.map(videoLocation)
  if (!hasNestedVideoEvidence(set.version)) return result
  const audio = set.audio, av = set.av
  const audioCoverage = audio?.coverage, avCoverage = av?.coverage
  if (audio && audio.source.sha256 === set.source.sha256 && ['ready', 'not_assessed'].includes(audio.status)
    && audioCoverage && [audioCoverage.decoded_samples, audioCoverage.scored_samples, audioCoverage.unscored_samples].every(count)
    && Array.isArray(audioCoverage.windows) && finite(audio.source.sample_rate) && audio.source.sample_rate > 0) {
    result.push(...audio.packets.map((packet) => audioLocation(packet)))
  }
  if (av && av.source.sha256 === set.source.sha256 && ['ready', 'not_assessed'].includes(av.status)
    && avCoverage && [avCoverage.recorded_windows, avCoverage.selected_windows, avCoverage.omitted_windows].every(count)
    && Array.isArray(avCoverage.unsupported_checks)) {
    result.push(...av.packets.map((packet) => audioLocation(packet, true)))
  }
  // Background events only record temporal overlap with voice packets, not finding/check attribution.
  return result
}

export function findingSupport({ analysisId, finding, signals, evidence }: {
  analysisId: string; finding: Finding; signals: readonly SignalV2[]; evidence?: FindingSupportEvidence | null
}): FindingSupportResult {
  const base = `/analyses/${encodeURIComponent(analysisId)}`
  const linked = linkedChecks(finding, signals)
  const checks: FindingSupportLink[] = linked.map((signal) => ({ id: `check:${signal.id}`,
    label: `${signal.display_id && /^S\d+$/.test(signal.display_id) ? `${signal.display_id} · ` : ''}Recorded check`,
    // The ledger's not-run rows use the v2 id; successful rows retain their v1 anchor when available.
    href: `${base}/ledger?view=expert#${encodeURIComponent(signal.status === 'ok' ? anchorOf(signal) : `signal-${signal.id}`)}`,
    summary: signal.observed, association: 'check', scope: 'check',
    scopeNote: 'Recorded result cited by this finding.' }))
  const result: FindingSupportResult = { packets: [], checks }
  // Older audio/document builders attached coverage findings to packets. Coverage is not a local detection claim.
  if (['coverage', 'F9_distribution_conditions'].includes(finding.family) || !evidence || evidence.status !== 'ready'
    || !evidence.id || evidence.analysis_id !== analysisId || !Array.isArray(evidence.packets)) return result

  const signalIds = new Set(linked.flatMap((signal) => [signal.id, ...(signal.legacy?.v1_id ? [signal.legacy.v1_id] : [])]))
  const findingRefs = references(finding)
  let locations: (LocatedPacket | null)[]
  switch (evidence.schema) {
    case 'chipmunk.image-evidence/1': locations = evidence.packets.map((packet) => imageLocation(packet, evidence)); break
    case 'chipmunk.video-evidence/1': locations = videoLocations(evidence); break
    case 'chipmunk.audio-evidence/1': locations = evidence.packets.map((packet) => audioLocation(packet)); break
    case 'chipmunk.document-evidence/1': locations = evidence.packets.map((packet) => documentLocation(packet, evidence.source.chars)); break
    default: return result
  }
  const image = evidence.schema === 'chipmunk.image-evidence/1'
  const seen = new Set<string>()
  for (const location of locations) {
    if (!location || seen.has(location.packet.id)) continue
    const { packet, label, scope, note } = location
    const direct = (packet.finding_ids ?? []).includes(finding.id)
      || ('exhibit_refs' in packet && (packet.exhibit_refs ?? []).some((ref) => findingRefs.has(ref)))
    const shared = (packet.signal_refs ?? []).some((ref) => signalIds.has(ref.id))
    if (!direct && !shared) continue
    seen.add(packet.id)
    result.packets.push({ id: packet.id, label, href: `${base}/${image ? 'image' : 'media'}-evidence?view=plain#${image ? 'image' : 'media'}-evidence-${encodeURIComponent(packet.id)}`,
      summary: packet.observation, association: direct ? 'finding' : 'shared_check', scope, scopeNote: note })
  }
  result.packets.sort((a, b) => Number(a.association === 'shared_check') - Number(b.association === 'shared_check'))
  if (finding.family === 'A1_voice' && evidence.schema === 'chipmunk.audio-evidence/1' && !evidence.packets.length) {
    const { coverage, source } = evidence
    const checkedIds = new Set(linked.filter((signal) => signal.status === 'ok').map((signal) => signal.id))
    if (coverage && finite(source.sample_rate) && source.sample_rate > 0
      && coverage.windows?.some((window) => interval(window) && window.end_sample <= coverage.decoded_samples
        && window.signal_refs?.some((ref) => checkedIds.has(ref.id)))) {
      result.voiceCoverage = { id: 'voice-coverage', label: 'Recorded voice window coverage',
        href: `${base}/media-evidence?view=plain#media-evidence-voice-coverage`, summary: '',
        association: 'finding', scope: 'audio_coverage', scopeNote: 'Review the exact intervals that received a voice response.' }
    }
  }
  return result
}
