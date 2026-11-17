import assert from 'node:assert/strict'
import fs from 'node:fs'
import { chromium } from 'playwright'

const fixture = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch()
try {
  const context = await browser.newContext({ extraHTTPHeaders: { 'X-API-Key': fixture.key } })
  await context.route('**/*', route => new URL(route.request().url()).origin === fixture.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  const cases = []
  page.on('pageerror', error => errors.push(error.message))
  for (const item of fixture.cases) {
    await page.goto(`${fixture.url}/analyses/${item.id}/media-evidence?view=plain`)
    const section = page.locator(`#media-evidence-${item.packet}`)
    await section.getByRole('heading', { name: item.title }).waitFor()
    await section.getByText(/compares detail and visual appearance inside the face/).waitFor()
    await section.getByText('Exact window coverage', { exact: true }).click()
    await section.getByText(`Input frame indices: ${item.frames.join(', ')}.`, { exact: false }).waitFor()
    const frames = section.getByRole('img', { name: /exact original video frame/ })
    const crops = section.getByRole('img', { name: /source face crop for context/ })
    for (const images of [frames, crops]) {
      assert.equal(await images.count(), 3)
      for (const image of await images.all()) {
        await image.scrollIntoViewIfNeeded()
        assert.ok(await image.evaluate(async element => { await element.decode(); return element.naturalWidth > 1 }))
      }
    }
    await section.getByRole('button', { name: `Go to frame ${item.target.frame_index} in playback`, exact: true }).click()
    await page.waitForFunction(time => {
      const video = document.querySelector('video[aria-label="Original video evidence playback"]')
      return video?.readyState >= 1 && !video.seeking && Math.abs(video.currentTime - time) < .003
    }, item.target.playback_time_s)
    cases.push({ fixture: item.name, source_frames: 3, face_crops: 3, native_seek_s: item.target.playback_time_s })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
