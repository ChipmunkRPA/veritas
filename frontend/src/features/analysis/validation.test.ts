import { describe, expect, it } from 'vitest'
import type { AnalysisDetail, ModelCard } from '../../lib/api/queries'
import { detectorValidation, recordedAdmission } from './methods'
import { admissionRecord, admissionWord, measuredRows, withCi } from './validation'

const card = (extra: Partial<ModelCard> = {}): ModelCard => ({ id: 'check.opaque', name: 'Analysis check', family: 'localiser', version: '1', modality: 'image',
  enabled: true, installed: true, loaded: false, licence_policy: 'commercial', ...extra })

describe('public validation records', () => {
  it('keeps finite measured zeroes but never treats malformed metric values as results', () => {
    const rows = measuredRows(card({ eval: [
      { slice: 'image/test', auc: 0, tpr_at_fpr01: 0.7, ece: null, n: 200, date: '2026-10-31', bench: 'private detector benchmark', source: 'private model source' },
      { slice: 'image/test', auc: '0.9', ece: Infinity, tpr_at_fpr01: 1.1 },
      { slice: 'image/test', auc: null, ece: null, tpr_at_fpr01: null },
    ] }))
    expect(rows).toHaveLength(1)
    expect(rows[0]).toEqual({ slice: 'image/test', auc: 0, tpr_at_fpr01: 0.7, ece: null, n: 200, date: '2026-10-31', benchmark_recorded: true })
    expect(JSON.stringify(rows)).not.toContain('private')
  })
  it('uses only the public admission decision and date, retaining historical decisions separately', () => {
    const current = card({ admission: { decision: 'retired', date: '2026-10-31', served_until: 'private-model', reviewed_by: 'private-reviewer' } })
    expect(admissionWord(current)).toBe('retired · 2026-10-31')
    expect(recordedAdmission({ id: current.id, version: '1', admission: { decision: 'admitted', date: '2025-01-01' } }, current))
      .toEqual({ text: 'admitted (card now: retired, 2026-10-31)', decision: 'admitted' })
    expect(admissionRecord(card({ admission: { decision: { name: 'private' }, date: 42 } })))
      .toEqual({ decision: null, date: null })
    expect(admissionWord(card({ admission: { decision: 'constructor', date: 'private-data' } }))).toBe('not recorded')
  })
  it('shows measured data without private benchmark labels and never borrows a different slice', () => {
    const current = card({ eval: [{ slice: 'image/test', auc: 0.8, n: 100, bench: 'private-detector-eval', date: '2026-10-31' }] })
    const analysis = { manifest: { detectors: [{ id: current.id, version: '1' }] }, condition_profile: { slice_id: 'image/test' } } as AnalysisDetail
    const matched = detectorValidation(analysis, [current])[0]
    expect(matched.measured).toContain('AUC 0.800')
    expect(matched.measured).toContain('recorded benchmark, 2026-10-31')
    expect(matched.measured).not.toContain('private')
    const other = detectorValidation({ ...analysis, condition_profile: { ...analysis.condition_profile!, slice_id: 'image/other' } }, [current])[0]
    expect(other.established).toBe(false)
    expect(other.measured).toContain('not established for this slice')
  })
  it('does not print non-finite measurements or intervals as established facts', () => {
    expect(withCi(NaN, [0.1, 0.2])).toBe('not established')
    expect(withCi(0.2, [0.1, Infinity])).toBe('0.200')
    expect(withCi(0.2, [0.1, 0.3])).toBe('0.200 [0.100, 0.300]')
  })
})
