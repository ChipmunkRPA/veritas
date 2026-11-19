// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest'
import { apiClient } from '../../lib/api/client'
import { compareDraft, draftAvailability, type DraftComparison } from './documentRevisionApi'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function markdownComparison(): DraftComparison {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 34, mime_part: null }
  return {
    schema: 'chipmunk.document-revision-comparison/1', format: 'markdown',
    coordinate_space: 'markdown_extracted_text_unicode_codepoint',
    markdown_fenced_code: 'exclude-commonmark-fenced-code/1',
    earlier: source, later: source, changes: [], scope: 'Extracted Markdown text.', interpretation: 'Source comparison only.',
  }
}

function answer(comparison: unknown) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ comparison }), { status: 200 })))
}

it('preserves both Markdown suffixes from the canonical availability response', async () => {
  vi.spyOn(apiClient, 'GET').mockResolvedValue({ data: { available: true, extension: '.md',
    earlier_extensions: ['.md', '.markdown', '.docx'], paired_check_available: false }, response: new Response() } as never)
  expect(await draftAvailability('draft', new AbortController().signal)).toEqual({
    extension: '.md', earlierExtensions: ['.md', '.markdown', '.docx'],
    pairedCheckAvailable: false, pairedMixedFormatsAvailable: false,
  })
})

it.each(['earlier.md', 'earlier.MARKDOWN'])('accepts a source-bound Markdown response for %s', async filename => {
  const comparison = markdownComparison()
  answer(comparison)
  const result = await compareDraft('draft', new File(['# Draft'], filename), new AbortController().signal)
  expect(result.comparison).toEqual(comparison)
})

it('accepts extracted Markdown positions on one side of a mixed comparison', async () => {
  const comparison = markdownComparison()
  comparison.format = 'mixed'
  comparison.schema = 'chipmunk.document-revision-comparison/3'
  comparison.coordinate_space = 'per_source_unicode_codepoint'
  comparison.earlier = { ...comparison.earlier, format: 'markdown', coordinate_space: 'markdown_extracted_text_unicode_codepoint' }
  comparison.later = { ...comparison.later, format: 'docx', coordinate_space: 'docx_body_unicode_codepoint' }
  answer(comparison)
  expect((await compareDraft('draft', new File(['# Draft'], 'earlier.md'), new AbortController().signal)).comparison).toEqual(comparison)
})

it.each(['coordinate', 'fence_policy', 'paired_check', 'mixed_coordinate'] as const)('rejects an incomplete Markdown %s response', async problem => {
  const comparison = markdownComparison()
  if (problem === 'coordinate') comparison.coordinate_space = 'decoded_source_unicode_codepoint'
  if (problem === 'fence_policy') delete comparison.markdown_fenced_code
  if (problem === 'paired_check') comparison.paired_check = { schema: 'chipmunk.paired-writing-review/1',
    status: 'outside_scope', summary: 'Not checked.', scope: 'No authorship weight.', authorship_confidence_contribution: 0 }
  if (problem === 'mixed_coordinate') {
    comparison.format = 'mixed'
    comparison.coordinate_space = 'per_source_unicode_codepoint'
    comparison.schema = 'chipmunk.document-revision-comparison/3'
    comparison.earlier = { ...comparison.earlier, format: 'markdown', coordinate_space: 'decoded_source_unicode_codepoint' }
  }
  answer(comparison)
  await expect(compareDraft('draft', new File(['# Draft'], 'earlier.md'), new AbortController().signal))
    .rejects.toThrow('The server returned an incomplete draft comparison.')
})

it('keeps the controlled incomplete-response error when the comparison is absent', async () => {
  answer(null)
  await expect(compareDraft('draft', new File(['# Draft'], 'earlier.md'), new AbortController().signal))
    .rejects.toThrow('The server returned an incomplete draft comparison.')
})
