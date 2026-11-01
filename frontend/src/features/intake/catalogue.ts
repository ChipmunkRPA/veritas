import type { Kind } from '../../lib/api'
import type { AnalysisProfile } from '../../lib/api/intake'
import type { ModelCard } from '../../lib/api/queries'
import { availableChecks } from '../models/availability'

export const PROFILE_NAMES: Record<AnalysisProfile, string> = { quick: 'Quick', full: 'Full', deep: 'Deep' }
const PROFILES = Object.keys(PROFILE_NAMES) as AnalysisProfile[]
const VIDEO_AUDIO_EXCLUDED = new Set(['provenance', 'face-detection', 'face-tracking'])

/** The video pipeline also runs audio checks, excluding its separate provenance/face stages.
 * Document is the intake name for the catalogue's text modality. */
export function intakeChecks(cards: readonly ModelCard[], kind: Kind, profile: AnalysisProfile | ''): ModelCard[] {
  return availableChecks(cards).filter((card) => {
    const sourceMatches = kind === 'document' ? card.modality === 'text'
      : kind === 'video' ? card.modality === 'video' || (card.modality === 'audio' && !VIDEO_AUDIO_EXCLUDED.has(card.family))
        : card.modality === kind
    const profileMatches = profile ? card.profiles?.includes(profile) : PROFILES.some((p) => card.profiles?.includes(p))
    return sourceMatches && profileMatches
  })
}

const CATEGORIES = {
  source: { title: 'Source history', detail: 'Inspect recorded source information and signs of editing.' },
  imagery: { title: 'Generated imagery', detail: 'Look for patterns associated with generated images or sampled video frames and clips.' },
  faces: { title: 'Face locations', detail: 'Locate visible faces for any available face-specific checks.' },
  manipulation: { title: 'Face manipulation', detail: 'Examine usable faces for signs of replacement or alteration.' },
  edits: { title: 'Image edits', detail: 'Inspect image structure for traces that may warrant closer review.' },
  voice: { title: 'Synthetic speech', detail: 'Examine selected speech windows for patterns associated with generated voices.' },
  sync: { title: 'Speech and lip movement', detail: 'Compare speech timing with the movement of a visible speaking face.' },
  recording: { title: 'Recording structure', detail: 'Inspect frame timing and how the recording was encoded.' },
  writing: { title: 'Writing patterns', detail: 'Examine language patterns that may help assess how the text was produced.' },
  other: { title: 'Additional checks', detail: 'Other checks may also be available for this source.' },
} as const
type Category = keyof typeof CATEGORIES

function category(card: ModelCard): Category {
  if (card.family === 'provenance') return 'source'
  if (card.family === 'face-detection' || card.family === 'face-tracking') return 'faces'
  if (card.lane === 'face_swap' || card.family === 'learned-face' || card.family === 'temporal-face') return 'manipulation'
  if (card.family === 'learned-whole-image' || card.family === 'synthetic-video') return 'imagery'
  if (card.family === 'forensic' || card.family === 'localiser') return 'edits'
  if (card.family === 'voice' || card.lane === 'voice') return 'voice'
  if (card.family === 'av-sync' || card.lane === 'lip_sync') return 'sync'
  if (card.family === 'codec') return 'recording'
  if (card.modality === 'text') return 'writing'
  return 'other'
}

export function intakeGroups(checks: readonly ModelCard[]) {
  return (Object.keys(CATEGORIES) as Category[]).flatMap((key) => {
    const members = checks.filter((card) => category(card) === key)
    if (!members.length) return []
    return [{ key, ...CATEGORIES[key], count: members.length,
      profiles: PROFILES.filter((p) => members.some((card) => card.profiles?.includes(p))) }]
  })
}
