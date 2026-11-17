// Verify the exact image evidence in the ordinary local analysis view.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({
    viewport: { width: 1366, height: 1050 },
    extraHTTPHeaders: { 'X-API-Key': input.key },
  })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const examples = []
  for (const { name, analysis_id, review } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}?view=plain#image-evidence-jpeg-prior-grid`)
    const evidence = page.getByRole('region', { name: 'Local JPEG compression history' })
    const [x, y, right, bottom] = review.record.region
    await evidence.getByText(review.record.basis).waitFor()
    await evidence.getByText(`Source columns [${x}, ${right}) and rows [${y}, ${bottom}).`, { exact: false }).waitFor()
    const plates = {}
    for (const [role, label] of [
      ['context', 'Highlighted area in the submitted image'],
      ['crop', 'Unmarked crop from the submitted image'],
    ]) {
      const image = evidence.getByRole('img', { name: label })
      await image.scrollIntoViewIfNeeded()
      const size = await image.evaluate(async element => {
        await element.decode()
        return [element.naturalWidth, element.naturalHeight]
      })
      const asset = review.artifacts[role]
      assert.deepEqual(size, [asset.width, asset.height])
      assert.ok((await image.getAttribute('src')).includes(asset.sha256))
      plates[role] = size
    }
    await evidence.screenshot({ path: path.join(input.output, `${name}-ordinary-view.png`), animations: 'disabled' })
    const focus = [x, y, right - x, bottom - y].join(',')
    await evidence.getByRole('link', { name: 'Inspect this position in the original image' }).click()
    await page.waitForURL(route => route.pathname.endsWith('/evidence') && route.searchParams.get('focus') === focus)
    examples.push({ case: name, coordinates: [x, y, right, bottom], decoded_images: plates })
  }
  assert.ok(examples.length > 0)
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases: examples, page_errors: errors.length }))
} finally {
  await browser.close()
}
