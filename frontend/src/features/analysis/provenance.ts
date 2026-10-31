/** Provenance & metadata (UX §5.8, §7.6). Pure readers of the table exhibits the provenance and codec detectors publish:
 *  `img.meta.consistency` "Metadata tree" (group, tag, value, severity, rule, note), `img.prov.c2pa` "C2PA Content Credentials"
 *  (field, value) and "C2PA ingredient chain", `img.meta.exif` "EXIF tags" / "JPEG structure", `vid.codec.ffprobe` "Codec facts". */
import type { AnalysisDetail, ExhibitMeta, TableExhibit } from '../../lib/api/queries'
import type { Severity } from '../../components/ui'

export type Cell = string | number | null

export const TITLES = {
  tree: 'Metadata tree',
  c2pa: 'C2PA Content Credentials',
  chain: 'C2PA ingredient chain',
  codes: 'C2PA validation codes',
  exif: 'EXIF tags',
  jpeg: 'JPEG structure',
  codec: 'Codec facts',
} as const

export const findTable = (tables: readonly TableExhibit[], title: string) => tables.find((t) => t.title === title)

/** Table exhibits of an analysis: the ones this section reads (by kind), in exhibit order. */
export const tableIds = (exhibits: readonly ExhibitMeta[]) => exhibits.filter((x) => x.kind === 'table').map((x) => x.id)

const text = (c: Cell) => (c == null ? '' : String(c))

const UNNAMED_TAG = /^(?:([A-Za-z][A-Za-z0-9]*):)?(?:(\d{1,5})|0x([0-9a-fA-F]{4}))$/
/** A metadata tag that the reader has no name for is recorded by its number ("41733", "ExifIFD:0xa305"): the page prints it as
 *  an unnamed tag with that number, in decimal (the report's `tag_words`). A named tag is printed as recorded. */
export function tagWords(tag: string): string {
  const match = UNNAMED_TAG.exec(tag)
  if (!match) return tag
  return `${match[1] ? `${match[1]}: ` : ''}unnamed tag ${match[2] ? Number(match[2]) : parseInt(match[3], 16)}`
}

export interface TreeField { tag: string; value: string; severity: Severity; rule: string; note: string }
export interface TreeGroup { name: string; fields: TreeField[]; worst: Severity }

const RANK: Record<Severity, number> = { info: 0, notable: 1, anomaly: 2, documentary: 3 }
const asSeverity = (c: Cell): Severity => (c === 'notable' || c === 'anomaly' || c === 'documentary' ? c : 'info')

/** Groups of a metadata tree table in first-seen order. A table without a severity column (EXIF tags, JPEG structure) becomes one
 *  group of info fields: absence and ordinary values count for nothing (UX §7.3 rule 5). */
export function treeGroups(table: Pick<TableExhibit, 'columns' | 'rows' | 'title'>): TreeGroup[] {
  const col = (name: string) => table.columns.indexOf(name)
  const [g, t, v, s, r, n] = ['group', 'tag', 'value', 'severity', 'rule', 'note'].map(col)
  const k = t >= 0 ? t : Math.max(col('field'), 0)
  const val = v >= 0 ? v : Math.max(col('value'), 1)
  const groups = new Map<string, TreeGroup>()
  for (const row of table.rows) {
    const name = g >= 0 ? text(row[g]) || 'Other' : table.title
    const field: TreeField = { tag: t >= 0 ? tagWords(text(row[k])) : text(row[k]), value: text(row[val]), severity: s >= 0 ? asSeverity(row[s]) : 'info', rule: r >= 0 ? text(row[r]) : '', note: n >= 0 ? text(row[n]) : '' }
    const group = groups.get(name) ?? { name, fields: [], worst: 'info' as Severity }
    group.fields.push(field)
    if (RANK[field.severity] > RANK[group.worst]) group.worst = field.severity
    groups.set(name, group)
  }
  return [...groups.values()]
}

export function severityCounts(groups: readonly TreeGroup[]): Record<Severity, number> {
  const out: Record<Severity, number> = { info: 0, notable: 0, anomaly: 0, documentary: 0 }
  groups.forEach((gr) => gr.fields.forEach((f) => { out[f.severity] += 1 }))
  return out
}

export type C2paState = 'absent' | 'valid' | 'trusted' | 'invalid' | null

/** The H3 assessment's `c2pa_state` (null when no Content Credentials check ran or a remote manifest was not fetched). */
export function c2paState(analysis: Pick<AnalysisDetail, 'assessments'>): C2paState {
  const h3 = (analysis.assessments ?? []).find((a) => a.hypothesis === 'H3_provenance_integrity')
  const s = h3?.c2pa_state
  return s === 'absent' || s === 'valid' || s === 'trusted' || s === 'invalid' ? s : null
}

/** UX §7.6 wording. `trusted` is the signer on the trust list; `valid` is intact with an unverified signer. */
export function c2paWords(state: C2paState, checked: boolean): { line: string; state: string } {
  switch (state) {
    case 'trusted': return { state: 'valid', line: 'Credentials valid · signer on the trust list' }
    case 'valid': return { state: 'untrusted', line: 'Credentials intact, but the signer isn’t on the trust list, so its identity is unverified.' }
    case 'invalid': return { state: 'invalid', line: 'Credentials invalid — the file or its record changed after signing. Disregard what the record says.' }
    case 'absent': return { state: 'absent', line: 'No Content Credentials found. Most files carry none; this is not evidence either way.' }
    default:
      return checked
        ? { state: 'unknown', line: 'The Content Credentials check ran but recorded no validation state (for example, a remote manifest that Veritas does not fetch offline).' }
        : { state: 'not-run', line: 'Content Credentials were not checked for this analysis.' }
  }
}

/** field → value rows of a two-column table. */
export function fields(table: Pick<TableExhibit, 'rows'> | undefined): Map<string, string> {
  return new Map((table?.rows ?? []).map((row) => [text(row[0]), text(row[1])]))
}

export interface ChainNode { depth: number; title: string; relationship: string; signer: string; sourceTypes: string; failures: string; manifest: string }

export function chainNodes(table: Pick<TableExhibit, 'columns' | 'rows'> | undefined): ChainNode[] {
  if (!table) return []
  const col = (name: string) => table.columns.indexOf(name)
  const get = (row: readonly Cell[], name: string) => (col(name) >= 0 ? text(row[col(name)]) : '')
  return table.rows.map((row) => ({
    depth: Number(get(row, 'depth')) || 0, title: get(row, 'title'), relationship: get(row, 'relationship'), signer: get(row, 'signer'),
    sourceTypes: get(row, 'digital source types'), failures: get(row, 'recorded validation failures'), manifest: get(row, 'manifest'),
  }))
}

/** Generative-AI terms in the backend's pinned IPTC vocabulary. `algorithmicMedia` is other_creation.
 * Exported credential rows include the backend tier; chain rows are unqualified terms. Preserve the older `(current)` display,
 * but never upgrade a term explicitly classified as unknown. Commas inside a tier qualifier are not term separators. */
export const declaresAi = (sourceTypes: string) => sourceTypes.split(/[;,](?![^()]*\))/).some((term) =>
  /^(?:trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|compositeSynthetic)(?:\s+\((?:ai(?:,\s*retired term)?|current)\))?$/i.test(term.trim()))

export interface Fact { fact: string; value: string; reference: string }

export function codecFacts(table: Pick<TableExhibit, 'rows'> | undefined): Fact[] {
  return (table?.rows ?? []).map((row) => ({ fact: text(row[0]), value: text(row[1]), reference: text(row[2]) }))
}
