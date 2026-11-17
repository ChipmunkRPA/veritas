// Check native PDF wording, locations, and the secured current-page crops.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1020 }, extraHTTPHeaders: { 'X-API-Key': input.key } })
  await context.route('**/*', route => new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = [], cases = []
  page.on('pageerror', error => errors.push(error.message))
  for (const { analysis_id, case: name, inspection, hashes } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain#media-evidence-pdf-saved-wording`)
    const card = page.getByRole('region', { name: 'Changes across saved PDF versions' })
    await card.getByText('The comparison carries no AI confidence weight.', { exact: false }).waitFor()
    const plots = []
    if (inspection.status === 'changed') {
      const latest = inspection.comparisons.at(-1)
      for (const [index, change] of latest.changes.entries()) {
        await card.getByRole('heading', { name: { replace: 'Wording replaced', insert: 'Wording added', delete: 'Wording removed' }[change.kind] }).waitFor()
        if (change.earlier) {
          await card.getByText(`Earlier page ${change.earlier.page}, word positions [${change.earlier.word_range.join(', ')}).`).waitFor()
        }
        if (change.current) {
          await card.getByText(change.current.excerpt, { exact: true }).waitFor()
          for (const [role, label] of [['crop', 'Unmarked'], ['overlay', 'Highlighted']]) {
            const figure = card.getByAltText(`${label} PDF crop for change ${index + 1}`)
            await figure.scrollIntoViewIfNeeded()
            const pixels = await figure.evaluate(async image => {
              await image.decode()
              return [image.naturalWidth, image.naturalHeight]
            })
            assert.deepEqual(pixels, hashes[`change${index + 1}_${role}`])
            plots.push(pixels)
          }
        }
      }
    } else {
      await card.getByText('The page wording agrees', { exact: false }).waitFor()
    }
    assert.ok(!(await card.innerText()).includes('ARCHIVEDTOKENPRIVATE'))
    await card.screenshot({ path: path.join(input.output, `${name}-saved-wording.png`), animations: 'disabled' })
    cases.push({ case: name, status: inspection.status, plots })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
