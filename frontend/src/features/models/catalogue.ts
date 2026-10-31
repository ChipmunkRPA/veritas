/** Model catalogue rules (ARCHITECTURE §7.2 `/models`, VG-01; BUILD_SPEC E-7): state words, licence classes, pins, measured rows
 *  and the catalogue filters. Every word is derived from the card the API returns; nothing is filled in here. */
import type { ModelCard } from '../../lib/api/queries'

export type ModelEntryState = NonNullable<ModelCard['model_entries']>[number]
export type EvalRow = NonNullable<ModelCard['eval']>[number]
export type StateFilter = 'all' | 'enabled' | 'disabled' | 'installed' | 'not_installed' | 'loaded'

export type CardCopy = NonNullable<ModelCard['client']>

export const NOT_ESTABLISHED = 'not established'
export const LICENCE_CLASS_WORD: Record<string, string> = { commercial: 'commercial', non_commercial: 'non-commercial', unknown: 'not established' }
export const DEVICE_STATE_WORD: Record<string, string> = { ok: 'supported', fallback_cpu: 'runs on the CPU instead', unsupported: 'not supported' }
const SLICE_MODALITY: Record<string, string> = { image: 'images', video: 'video', audio: 'audio', text: 'documents', document: 'documents' }
const SLICE_SPLIT: Record<string, string> = { test: 'test split', cal: 'calibration split', fit: 'fitting split' }

const spaced = (value: string) => value.replace(/[_-]+/g, ' ').trim()

/** The card in client words. The API derives it (no ticket ids, repository paths or raw enums); a card served without it prints
 *  its enums as words and none of the recorded prose, so nothing unvetted reaches the page. */
export function clientOf(card: ModelCard): CardCopy {
  return card.client ?? {
    family: spaced(card.family), hypotheses: [], threats: [], lane: card.lane ? spaced(card.lane) : null, citation: null, training_data: null,
    coverage_date: null, input_constraints: [], contamination_sets: [], failure_modes: [], enablement_reasons: [],
    licence_policy: card.licence_policy.replace(/\+/g, ' + '), disclosures: [], licence_notices: {}, record_titles: {},
    truncated: [], engineering: [],
  }
}

/** N-19d: a check without a calibration of its own. The ledger of a result under a fusion model says "calibrated by the fusion
 *  model" for the same check, so the card says where that calibration comes from. */
export const CALIBRATION_NONE = 'none of its own (a raw score); a fusion model that uses this check calibrates the score'

/** `image/test` → "images · test split"; an unknown part keeps its words. */
export function sliceWord(slice: string): string {
  return slice.split('/').map((part, i) => (i === 0 ? SLICE_MODALITY[part] : SLICE_SPLIT[part]) ?? spaced(part)).join(' · ')
}

/** N-19d: the address of a card's source when its client wording is an address alone; null for a check that is part of Veritas
 *  (its record names files of the repository, which is no address: the sheet's "Source" link led nowhere). */
export function sourceLink(client: CardCopy): string | null {
  return client.source && /^https?:\/\/\S+$/.test(client.source) ? client.source : null
}

/** Recorded notes on failure modes that the client wording leaves out (they are printed in the engineering record). */
export function withheldFailureNotes(client: CardCopy): number {
  return (client.engineering ?? []).filter((n) => n.withheld !== false && (n.field === 'False-positive mode' || n.field === 'Known limitation')).length
}

export const MODALITY_WORD: Record<string, string> = { image: 'Image', video: 'Video', audio: 'Audio', text: 'Document', document: 'Document' }
export const ADMISSION_WORD: Record<string, string> = { pending: 'pending', admitted: 'admitted', rejected: 'rejected', retired: 'retired' }

/** Licence classes of the models the detector needs, in first-seen order; a model-free detector has none. */
export function licenceClasses(card: ModelCard): string[] {
  return [...new Set((card.model_entries ?? []).map((m) => m.licence.class))]
}

/** `installed` / `not installed` / `model-free`, and `loaded on mps` / `not loaded` / `model-free`. */
export function installedWord(card: ModelCard): string {
  if ((card.model_entries ?? []).length === 0) return 'model-free'
  return card.installed ? 'installed' : 'not installed'
}

/** Immediate parents retain the registry's stable model order. */
export function modelParents(card: ModelCard, modelId: string): string[] {
  return [...new Set((card.model_entries ?? []).filter((m) => (m.depends_on ?? []).includes(modelId)).map((m) => m.id))]
}

/** Dependencies can be constructed inside a runtime bundle without their own manager entry. */
export function modelMemoryWord(card: ModelCard, model: ModelEntryState): string {
  if (model.loaded) {
    const devices = model.loaded_devices ?? []
    return devices.length ? `loaded on ${devices.join(', ')}` : 'loaded'
  }
  return !(card.models ?? []).includes(model.id) && modelParents(card, model.id).length > 0
    ? 'not tracked separately' : 'not loaded'
}

export function loadedWord(card: ModelCard): string {
  const allEntries = card.model_entries ?? []
  const entries = (card.models ?? []).length > 0
    ? allEntries.filter((m) => card.models!.includes(m.id)) : allEntries
  if (card.loaded == null || entries.length === 0) return 'model-free'
  if (!card.loaded) return entries.some((m) => m.loaded) ? 'partly loaded' : 'not loaded'
  const devices = [...new Set(entries.flatMap((m) => m.loaded_devices ?? []))]
  return devices.length ? `loaded on ${devices.join(', ')}` : 'loaded'
}

export function enabledWord(card: ModelCard): string {
  return card.enabled ? 'enabled' : 'disabled'
}

export function admissionWord(card: ModelCard): string {
  const a = card.admission
  if (!a) return 'not recorded'
  const word = ADMISSION_WORD[a.decision] ?? a.decision
  // a rejected / retired detector the served fusion model still holds keeps running until that model is refitted
  const served = a.served_until && (a.decision === 'rejected' || a.decision === 'retired')
    ? `${word}; still part of the served fusion model ${a.served_until} until its refit` : word
  return [served, a.date, a.reviewed_by].filter(Boolean).join(' · ')
}

export interface Pin { label: string; value: string }

/** What pins a model: the HF or torch.hub commit, then every file with its SHA-256. */
export function pins(m: ModelEntryState): Pin[] {
  const s = m.source
  const out: Pin[] = []
  if (s.revision) out.push({ label: 'revision', value: s.revision })
  if (s.ref) out.push({ label: 'commit', value: s.ref })
  for (const f of s.files ?? []) if (f.sha256) out.push({ label: f.path, value: f.sha256 })
  return out
}

/** Where the weights come from, in one line. */
export function sourceLine(m: ModelEntryState): string {
  const s = m.source
  if (s.type === 'hf') return `Hugging Face ${s.repo ?? ''}`.trim()
  if (s.type === 'torch_hub') return `torch.hub ${s.repo ?? ''}${s.entrypoint ? ` · ${s.entrypoint}` : ''}`.trim()
  if (s.type === 'url') return s.url ?? 'url'
  if (s.type === 'gdrive') return `Google Drive ${s.file_id ?? ''}`.trim()
  if (s.type === 'local') return `produced locally${s.produced_by ? ` by ${s.produced_by}` : ''}`
  return s.type
}

/** Benchmark rows that carry at least one measured number; rows written with every metric null are placeholders, not results. */
export function measuredRows(card: ModelCard): EvalRow[] {
  return (card.eval ?? []).filter((r) => r.auc != null || r.tpr_at_fpr01 != null || r.ece != null)
}

export const fmtMetric = (v: number | null | undefined, d = 3): string => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d))

/** `0.889 [0.862, 0.913]`; the interval only when both bounds are recorded. */
export function withCi(v: number | null | undefined, ci: readonly (number | null | undefined)[] | null | undefined, d = 3): string {
  if (v == null) return NOT_ESTABLISHED
  return ci && ci.length === 2 && ci[0] != null && ci[1] != null ? `${fmtMetric(v, d)} [${fmtMetric(ci[0], d)}, ${fmtMetric(ci[1], d)}]` : fmtMetric(v, d)
}

/** Search over id, name and family; modality and state menus. */
export function filterCards(cards: readonly ModelCard[], search: string, modality: string, state: StateFilter): ModelCard[] {
  const needle = search.trim().toLowerCase()
  return cards.filter((c) => {
    if (modality !== 'all' && c.modality !== modality) return false
    if (state === 'enabled' && !c.enabled) return false
    if (state === 'disabled' && c.enabled) return false
    if (state === 'installed' && !(c.installed && (c.model_entries ?? []).length > 0)) return false
    if (state === 'not_installed' && c.installed) return false
    if (state === 'loaded' && c.loaded !== true) return false
    return needle === '' || [c.id, c.name, c.family].some((v) => v.toLowerCase().includes(needle))
  })
}

/** The facts line under the title. */
export function catalogueFacts(cards: readonly ModelCard[]): string {
  const withModels = cards.filter((c) => (c.model_entries ?? []).length > 0)
  const policy = cards[0] ? clientOf(cards[0]).licence_policy : undefined
  return [
    `${cards.length} ${cards.length === 1 ? 'detector' : 'detectors'}`,
    `${cards.filter((c) => c.enabled).length} enabled${policy ? ` under the licence policy in force (${policy})` : ''}`,
    // N-19k (design re-review D2-07): the second number is the detectors that need weights, not the enabled ones before it
    `${withModels.filter((c) => c.installed).length} of the ${withModels.length} that need weights have them installed`,
    `${cards.filter((c) => c.loaded === true).length} loaded`,
  ].join(' · ')
}
