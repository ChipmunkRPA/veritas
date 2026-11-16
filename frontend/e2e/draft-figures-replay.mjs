// Verify the real earlier-draft comparison and its browser downloads.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const [example] = input.cases
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1020 },
    extraHTTPHeaders: { 'X-API-Key': input.key } })
  await context.route('**/*', route => new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`${input.url}/analyses/${example.analysis_id}?view=plain`)
  const draft = page.getByRole('region', { name: 'Compare with an earlier draft' })
  await draft.getByLabel('Earlier draft').setInputFiles(example.earlier)
  await draft.getByRole('button', { name: 'Compare drafts' }).click()
  const evidence = page.getByRole('region', { name: 'Figures and dates' })
  await evidence.getByText('Date changed · March 4, 2020 → 2020-03-05').waitFor()
  await evidence.getByText('Percentage changed · 4.50 percent → 5.00%').waitFor()
  assert.equal(await evidence.locator('mark').filter({ hasText: '2020-03-05' }).count(), 1)
  assert.equal(await evidence.locator('mark').filter({ hasText: '5.00%' }).count(), 1)
  await evidence.screenshot({ path: path.join(input.output, 'figure-evidence.png'), animations: 'disabled' })
  for (const type of ['JSON', 'PDF']) {
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      draft.getByRole('button', { name: `Download comparison ${type}` }).click(),
    ])
    await download.saveAs(path.join(input.output, `browser-comparison.${type.toLowerCase()}`))
  }
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(input.output, 'browser-comparison.json'), 'utf8')),
    JSON.parse(fs.readFileSync(example.comparison, 'utf8')))
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ changed_figures: 2, json_matches_api: true, page_errors: 0 }))
} finally {
  await browser.close()
}
