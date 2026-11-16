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
  for (const { analysis_id, name, pair, pairs } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}?view=plain`)
    const evidence = page.getByRole('region', { name: 'Copy similarity without mirroring' })
    const items = pairs ?? [pair]
    if (pairs) await evidence.getByText('Two separate pairs of rotated areas share closely matching detail.', { exact: true }).waitFor()
    await evidence.getByText('This check carries zero weight in AI authorship confidence.').waitFor()
    for (const item of items) {
      const rotation = item.transform === 'rotate_90_ccw' ? '90° counterclockwise' : '180°'
      const card = evidence.getByRole('article').filter({ has: page.getByRole('heading', { name: `${item.id} · Rotated JPEG pair` }) })
      if (pairs) await card.getByText(`Rotate Area B ${rotation} to align with Area A.`, { exact: false }).waitFor()
      else await evidence.getByText(`Area B closely matches Area A after rotating it ${rotation}.`, { exact: true }).waitFor()
      await card.getByText('Show both areas on the original image').click()
      await card.getByText(/The outlines locate Areas A and B/).waitFor()
      for (const [area, label] of [['area_a', 'Area A'], ['area_b', 'Area B']]) {
        const box = item.regions[area]
        await card.getByText(`x ${box.x}, y ${box.y}, ${box.w} × ${box.h} px`, { exact: true }).first().waitFor()
        const link = card.getByRole('link', { name: `Inspect ${label} in the original image` })
        const caption = card.getByRole('link', { name: `${label} in the original image`, exact: true })
        assert.equal(await link.getAttribute('href'), await caption.getAttribute('href'))
        const crop = card.getByRole('img', { name: `${item.id}: ${label} cropped from the submitted image` })
        await crop.waitFor()
        assert.equal(await crop.evaluate(image => image.decode().then(() => image.naturalWidth)), box.w)
        assert.equal(await crop.evaluate(image => image.naturalHeight), box.h)
        assert.ok((await crop.getAttribute('src')).includes(item.artifacts[area].sha256))
      }
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
