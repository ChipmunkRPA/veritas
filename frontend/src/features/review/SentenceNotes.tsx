import { useId, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { useAnnotations, useCase, useMe, type AnalysisDetail, type Annotation } from '../../lib/api/queries'
import { mayExamine } from '../../lib/api/roles'
import { Button, Eyebrow, IdChip } from '../../components/ui'
import { VisibilityChoice, textareaClass } from './Dispositions'
import { useReviewRefresh } from './ReviewPanel'
import { noteNumbers, reviewGate, sentenceIndex, type Visibility } from './review'

/** The notes of one sentence by their numbers (`N3`), for the manuscript's marks. */
export function sentenceNoteNumbers(annotations: readonly Annotation[]): Map<number, string[]> {
  const numbers = noteNumbers(annotations)
  const bySentence = new Map<number, string[]>()
  for (const a of annotations) {
    const idx = sentenceIndex(a)
    const n = numbers.get(a.id)
    if (idx === null || !n) continue
    bySentence.set(idx, [...(bySentence.get(idx) ?? []), n])
  }
  return bySentence
}

/** Notes on the selected sentence, in the manuscript's rail (N-19a; browser QA "Workflow": the API took a note on a sentence
 *  and the page and the PDF showed it, but the manuscript had no way to add one). The note is an annotation of the sentence
 *  (`target_type: sentence`, its index), numbered with the notes of the review section, where it can be edited. Offered to
 *  who may write the examiner layer of this revision, as the notes of the review section are. */
export default function SentenceNotes({ analysis, sentenceIdx }: { analysis: AnalysisDetail; sentenceIdx: number }) {
  const uid = useId()
  const annotations = useAnnotations(analysis.id)
  const me = useMe()
  const kase = useCase(analysis.case_id ?? '')
  const refresh = useReviewRefresh(analysis.id, analysis.case_id)
  const [body, setBody] = useState('')
  const [visibility, setVisibility] = useState<Visibility>('internal')
  const [invalid, setInvalid] = useState<string | null>(null)
  const writable = reviewGate(analysis).writable && mayExamine(me.data, kase.data?.my_role)
  const numbers = noteNumbers(annotations.data ?? [])
  const notes = (annotations.data ?? []).filter((a) => sentenceIndex(a) === sentenceIdx && numbers.has(a.id))
    .sort((x, y) => Date.parse(x.created_at) - Date.parse(y.created_at))

  const add = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/analyses/{aid}/annotations', {
      params: { path: { aid: analysis.id } },
      body: { target_type: 'sentence', shape: 'sentence', geometry: { sentence_idx: sentenceIdx }, body: body.trim(), visibility },
    })),
    onSuccess: () => { setBody(''); setVisibility('internal'); refresh() },
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!body.trim()) { setInvalid('A note needs text.'); return }
    setInvalid(null)
    add.mutate()
  }

  if (notes.length === 0 && !writable) return null
  return (
    <div className="mt-4 pt-4 border-t border-border" data-testid="sentence-notes">
      <Eyebrow as="h3">Notes on this sentence</Eyebrow>
      {notes.length > 0 && (
        <ol className="mt-2 m-0 p-0 list-none flex flex-col gap-2">
          {notes.map((a) => (
            <li key={a.id} data-testid="sentence-note" className="text-small text-text">
              <IdChip analyst>{numbers.get(a.id)}</IdChip>
              <span className="ml-2 text-caption text-text-muted">{a.visibility === 'in_report' ? 'in report' : 'internal'}</span>
              <span className="block mt-1 whitespace-pre-line break-words">{a.body}</span>
            </li>
          ))}
        </ol>
      )}
      {writable && (
        <form onSubmit={submit} className="mt-3" noValidate data-testid="add-sentence-note">
          <label className="flex flex-col gap-1 text-small text-text-muted">
            Add a note on sentence {sentenceIdx + 1}
            <textarea value={body} onChange={(e) => setBody(e.target.value)} rows={3} maxLength={10000} className={textareaClass} />
          </label>
          <VisibilityChoice name={`${uid}-visibility`} value={visibility} onChange={setVisibility} />
          {(invalid ?? add.error?.message) && <p role="alert" className="mt-2 mb-0 text-small text-danger">{invalid ?? add.error?.message}</p>}
          <Button variant="ghost" type="submit" className="mt-3" disabled={add.isPending}>{add.isPending ? 'Adding…' : 'Add note'}</Button>
        </form>
      )}
    </div>
  )
}
