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
  for (const { analysis_id, case: name, picture, sizes } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain#media-evidence-pdf-saved-pictures`)
    const card = page.getByRole('region', { name: 'Pictures across saved PDF versions' })
    await card.getByText('This check contributes zero weight to the AI assessment.', { exact: false }).waitFor()
    const images = []
    if (picture.status === 'changed') {
      for (const [index, change] of picture.comparisons.at(-1).changes.entries()) {
        const finding = card.getByRole('listitem').nth(index)
        await finding.getByText(`Earlier page ${change.earlier_page} · submitted page ${change.current_page}.`).waitFor()
        await finding.getByText(`Region [${change.region_pixels.join(', ')}] at ${change.render_scale} rendered pixels per PDF point`, { exact: false }).waitFor()
        for (const [role, label] of [
          ['overlay', 'Submitted picture with the measured difference highlighted'],
          ['crop', 'Unmarked crop from the submitted PDF'],
        ]) {
          const image = finding.getByAltText(label, { exact: true })
          await image.scrollIntoViewIfNeeded()
          const geometry = await image.evaluate(async element => {
            await element.decode()
            return [element.naturalWidth, element.naturalHeight]
          })
          assert.deepEqual(geometry, sizes[`picture${index + 1}_${role}`])
          images.push(geometry)
        }
      }
    } else {
      const status = name === 'recompressed' ? 'found no comparable pictures' : 'No local picture differences were found'
      await card.getByText(status, { exact: false }).waitFor()
      assert.equal(await card.getByRole('img').count(), 0)
    }
    await card.screenshot({ path: path.join(input.output, `${name}-saved-pictures.png`), animations: 'disabled' })
    cases.push({ case: name, status: picture.status, images })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
