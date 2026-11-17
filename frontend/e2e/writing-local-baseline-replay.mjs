import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const reason = 'Selected by comparing this passage with the writing patterns elsewhere in this file.'
const normalize = (text) => text.replace(/\s+/g, ' ').trim()
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({
    viewport: { width: 1366, height: 980 },
    extraHTTPHeaders: { 'X-API-Key': input.key },
  })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  const checked = []
  page.on('pageerror', error => errors.push(error.message))
  for (const sample of input.cases) {
    await page.goto(`${input.url}/analyses/${sample.analysis_id}?view=plain#document-writing-passage-intervals`)
    const section = page.getByRole('region', { name: 'Possible AI-like writing passages' })
    const status = sample.previews.length ? 'Inspect the highlighted passages in context.'
      : 'No passage was selected. Authorship remains undetermined by this check.'
    await section.getByText(status).waitFor()
    if (sample.reportContext) {
      await section.getByText('This file matched the writing patterns of the economic-report reference set.', { exact: false }).waitFor()
    }
    assert.equal(await section.getByText(reason, { exact: true }).count(), sample.relativeCount ?? sample.previews.length)
    const actual = await section.locator('blockquote').allTextContents()
    assert.equal(actual.length, sample.previews.length)
    for (const [index, preview] of sample.previews.entries()) {
      const shown = normalize(actual[index])
      const expected = normalize(preview)
      assert.ok(shown === expected || (shown.endsWith('…') && normalize(shown.slice(0, -1)) === expected))
    }
    await section.screenshot({ path: path.join(input.output, `${sample.carrier}-evidence.png`), animations: 'disabled' })
    checked.push(sample.carrier)
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ checked, page_errors: errors.length }))
} finally {
  await browser.close()
}
