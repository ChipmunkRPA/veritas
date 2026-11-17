// Local browser check for the paid HEIC comparison. Credentials arrive over stdin.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
assert.equal(new URL(input.url).hostname, '127.0.0.1')
const browser = await chromium.launch({ channel: 'chrome', headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1050 }, extraHTTPHeaders: { 'X-API-Key': input.key } })
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(input.url)
  const evidence = page.getByRole('region', { name: 'Embedded preview differences' })
  await evidence.waitFor({ timeout: 30000 })
  const words = await evidence.innerText()
  assert.match(words, /when or how the difference arose/)
  let images = 0
  for (const row of input.regions) {
    for (const [role, label] of [['main_image', 'Main image crop'], ['embedded_preview', 'Embedded preview crop'],
      ['measured_footprint', 'Measured difference footprint']]) {
      const image = evidence.getByRole('img', { name: `${label} for region ${row.region}` })
      await image.scrollIntoViewIfNeeded()
      const size = await image.evaluate(async element => {
        await element.decode()
        return [element.naturalWidth, element.naturalHeight]
      })
      assert.deepEqual(size, [row.crops[role].display_width, row.crops[role].display_height])
      const box = row[role === 'measured_footprint' ? 'footprint_grid' : role]
      assert(words.includes(`x ${box.x}, y ${box.y}, ${box.w} × ${box.h} pixels`))
      images += 1
    }
  }
  assert.deepEqual(errors, [])
  await evidence.screenshot({ path: path.join(input.folder, 'ordinary-view.png'), animations: 'disabled' })
  console.log(JSON.stringify({ browser: browser.version(), ordinary_view: true, decoded_images: images }))
} finally {
  await browser.close()
}
