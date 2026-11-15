// Replays a locally prepaid result. The isolated API key is read through stdin.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1050 },
    extraHTTPHeaders: { 'X-API-Key': input.key } })
  await context.route('**/*', route => new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  const checked = []
  page.on('pageerror', error => errors.push(error.message))
  for (const { analysis_id, name, pair } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}?view=plain`)
    const evidence = page.getByRole('region', { name: 'Copy similarity without mirroring' })
    const rotation = pair.transform === 'rotate_90_ccw' ? '90° counterclockwise' : '180°'
    await evidence.getByText(`Area B closely matches Area A after rotating it ${rotation}.`, { exact: true }).waitFor()
    await evidence.getByText('This check carries zero weight in AI authorship confidence.').waitFor()
    await evidence.getByText('Show both areas on the original image').click()
    await evidence.getByText(/The outlines locate Areas A and B/).waitFor()
    for (const [area, label] of [['area_a', 'Area A'], ['area_b', 'Area B']]) {
      const box = pair.regions[area]
      await evidence.getByText(`x ${box.x}, y ${box.y}, ${box.w} × ${box.h} px`, { exact: true }).first().waitFor()
      const link = evidence.getByRole('link', { name: `Inspect ${label} in the original image` })
      const caption = evidence.getByRole('link', { name: `${label} in the original image`, exact: true })
      assert.equal(await link.getAttribute('href'), await caption.getAttribute('href'))
      const crop = evidence.getByRole('img', { name: `${pair.id}: ${label} cropped from the submitted image` })
      await crop.waitFor()
      assert.equal(await crop.evaluate(image => image.decode().then(() => image.naturalWidth)), box.w)
      assert.equal(await crop.evaluate(image => image.naturalHeight), box.h)
      assert.ok((await crop.getAttribute('src')).includes(pair.artifacts[area].sha256))
    }
    assert.doesNotMatch(await evidence.innerText(), /matches Area A exactly|examiner.s conclusion|sign.off/i)
    await evidence.screenshot({ path: path.join(input.output, `${name}-browser.png`), animations: 'disabled' })
    checked.push(name)
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ verified_cases: checked, original_location_overlays: true, original_crop_links: true, page_errors: 0 }))
} finally {
  await browser.close()
}
