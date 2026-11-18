import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
assert.equal(new URL(input.url).hostname, '127.0.0.1')
const [example] = input.cases
const browser = await chromium.launch({ channel: 'chrome', headless: true })
try {
  const page = await browser.newPage({ extraHTTPHeaders: { 'X-API-Key': input.key } })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`${input.url}/analyses/${example.analysis_id}?view=plain`)
  const draft = page.getByRole('region', { name: 'Compare with an earlier draft' })
  const chooser = draft.getByLabel('Earlier draft')
  assert.match(await chooser.getAttribute('accept'), /\.odt/)
  await chooser.setInputFiles(example.earlier)
  await draft.getByRole('button', { name: 'Compare drafts' }).click()
  await draft.getByText(/OpenDocument offsets count Unicode code points/).waitFor()
  await draft.getByText(/Paragraph 1 \[/).first().waitFor()
  assert(await draft.locator('mark').count() > 1)
  for (const type of ['JSON', 'PDF']) {
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      draft.getByRole('button', { name: `Download comparison ${type}` }).click(),
    ])
    await download.saveAs(path.join(input.output, `browser-odt-comparison.${type.toLowerCase()}`))
  }
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(input.output, 'browser-odt-comparison.json'), 'utf8')),
    JSON.parse(fs.readFileSync(example.comparison, 'utf8')))
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ json_matches_api: true, paragraph_offsets_visible: true, page_errors: 0 }))
} finally {
  await browser.close()
}
