import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
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
  const replays = []
  page.on('pageerror', error => errors.push(error.message))
  for (const sample of input.cases) {
    await page.goto(`${input.url}/analyses/${sample.analysis_id}?view=plain`)
    const section = page.getByRole('region', { name: 'Compare with an earlier copy' })
    await section.getByLabel('Earlier copy').setInputFiles(sample.earlier)
    const response = page.waitForResponse(reply => reply.request().method() === 'POST' &&
      reply.url().endsWith('/image-reference-comparison'))
    await section.getByRole('button', { name: 'Compare images' }).click()
    const { comparison, pngs, pdf } = await (await response).json()
    assert.equal(comparison.schema, 'chipmunk.image-reference-comparison/24')
    assert.equal(comparison.status, sample.expected)
    assert.equal(comparison.affects_overall_confidence, false)
    assert.equal(comparison.regions.length, sample.regions)
    if (sample.regions) {
      await section.getByText(/between the earlier JPEG and submitted WebP/).waitFor()
      const region = comparison.regions[0]
      for (const [name, artifact] of [
        ['Earlier copy crop for region 1', region.reference_crop],
        ['Submitted image crop for region 1', region.submitted_crop],
      ]) {
        const exhibit = section.getByRole('img', { name, exact: true })
        await exhibit.scrollIntoViewIfNeeded()
        const dimensions = await exhibit.evaluate(async image => {
          await image.decode()
          return [image.naturalWidth, image.naturalHeight]
        })
        assert.deepEqual(dimensions, [artifact.width, artifact.height])
      }
      await section.getByRole('button', { name: 'Highlight difference pixels' }).first().click()
      await section.getByText('Amber marks the selected pixels in this crop.').first().waitFor()
      await section.screenshot({ path: path.join(input.output, 'browser-comparison.png'), animations: 'disabled' })
    } else {
      const message = sample.expected === 'comparable' ? 'No localized differences found' : 'Comparison unassessed'
      await section.getByRole('status').filter({ hasText: message }).waitFor()
    }
    const jsonDownload = page.waitForEvent('download')
    await section.getByRole('button', { name: 'Download comparison JSON' }).click()
    const exportPath = await (await jsonDownload).path()
    assert.deepEqual(JSON.parse(fs.readFileSync(exportPath, 'utf8')), { comparison, pngs })
    const pdfDownload = page.waitForEvent('download')
    await section.getByRole('button', { name: 'Download comparison PDF' }).click()
    assert.deepEqual(fs.readFileSync(await (await pdfDownload).path()), Buffer.from(pdf, 'base64'))
    replays.push({ role: sample.role, status: comparison.status, regions: sample.regions })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ replays, page_errors: errors.length }))
} finally {
  await browser.close()
}
