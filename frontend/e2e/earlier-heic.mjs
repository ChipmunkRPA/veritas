import assert from 'node:assert/strict'
import fs from 'node:fs'
import { chromium } from 'playwright'

const fixture = JSON.parse(fs.readFileSync(0, 'utf8'))
const [item] = fixture.cases
const browser = await chromium.launch()
try {
  const context = await browser.newContext({ extraHTTPHeaders: { 'X-API-Key': fixture.key }, acceptDownloads: true })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === fixture.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`${fixture.url}/analyses/${item.analysis_id}?view=plain#earlier-image-comparison`)
  const section = page.locator('#earlier-image-comparison')
  await section.getByRole('heading', { name: 'Compare with an earlier copy' }).waitFor()
  const input = section.getByLabel('Earlier copy')
  assert.ok((await input.getAttribute('accept')).split(',').includes('.heic'))
  await input.setInputFiles(item.reference)
  await section.getByRole('button', { name: 'Compare images' }).click()
  await section.getByText('The crops use sRGB display colors at the original pixel positions.', { exact: false }).waitFor()
  for (const [role, label] of [['reference_crop', 'Earlier copy'], ['submitted_crop', 'Submitted image']]) {
    const asset = item.comparison.regions[0][role]
    const image = section.getByRole('img', { name: `${label} crop for region 1`, exact: true })
    await image.scrollIntoViewIfNeeded()
    const size = await image.evaluate(async element => {
      await element.decode()
      return [element.naturalWidth, element.naturalHeight]
    })
    assert.deepEqual(size, [asset.width, asset.height])
  }
  await section.getByText('Show the area compared', { exact: true }).click()
  const coverage = section.getByRole('img', { name: 'Compared area in submitted image coordinates' })
  await coverage.waitFor()
  assert.equal(await coverage.evaluate(async image => { await image.decode(); return image.naturalWidth }),
    item.comparison.submitted.width)
  for (const format of ['JSON', 'PDF']) {
    const downloadPromise = page.waitForEvent('download')
    await section.getByRole('button', { name: `Download comparison ${format}` }).click()
    const download = await downloadPromise
    const content = fs.readFileSync(await download.path())
    if (format === 'JSON') assert.deepEqual(JSON.parse(content).comparison, item.comparison)
    else assert.equal(content.subarray(0, 5).toString(), '%PDF-')
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ crop_sizes_verified: 2, coverage_verified: true, json_matches_api: true,
    pdf_downloaded: true, page_errors: errors.length }))
} finally {
  await browser.close()
}
