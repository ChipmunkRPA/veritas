import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const fixture = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch()
try {
  const context = await browser.newContext({ extraHTTPHeaders: { 'X-API-Key': fixture.key } })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === fixture.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  const cases = []
  page.on('pageerror', error => errors.push(error.message))
  for (const item of fixture.cases) {
    await page.goto(`${fixture.url}/analyses/${item.id}/media-evidence?view=plain`)
    await page.getByTestId('video-evidence-content').waitFor()
    const label = 'Face transition check'
    const coverage = page.getByRole('region', { name: label, exact: true })
    const link = coverage.getByRole('link', { name: 'Inspect the transition', exact: true })
    const section = page.locator(`#media-evidence-${item.packet}`)
    if (!item.status) {
      assert.equal(await coverage.count(), 0)
    } else {
      await coverage.getByText('This check carries zero AI confidence weight.', { exact: true }).waitFor()
    }
    if (!item.frames.length) {
      assert.equal(await link.count(), 0)
      assert.equal(await section.count(), 0)
      if (item.status === 'no_match') {
        await coverage.getByText('No qualifying face appearance transition was found in the portion checked.', { exact: true }).waitFor()
      }
      if (item.reason === 'shot_change') {
        await coverage.getByText(/shot change was detected/i).waitFor()
      }
      cases.push({ fixture: item.name, status: item.status, packet_visible: false })
      continue
    }
    assert.equal(await link.getAttribute('href'), `#media-evidence-${item.packet}`)
    await link.click()
    assert.equal(new URL(page.url()).hash, `#media-evidence-${item.packet}`)
    await section.getByRole('heading', { name: /Face appearance transition$/ }).waitFor()
    await section.getByText('Surrounding picture similarity at the transition', { exact: true }).waitFor()
    const background = new Intl.NumberFormat('en', { maximumFractionDigits: 3 }).format(item.background)
    await section.getByText(`${background} correlation`, { exact: true }).waitFor()
    const frames = section.getByRole('img', { name: /exact original video frame/ })
    const crops = section.getByRole('img', { name: /source face crop for context/ })
    for (const images of [frames, crops]) {
      assert.equal(await images.count(), 3)
      for (const image of await images.all()) {
        await image.scrollIntoViewIfNeeded()
        assert.ok(await image.evaluate(async element => {
          await element.decode()
          return element.naturalWidth > 1 && element.naturalHeight > 1
        }))
      }
    }
    for (const frame of item.frames) {
      assert.equal(typeof frame.seek, 'number')
      await section.getByRole('button', { name: `Go to frame ${frame.index} in playback`, exact: true }).click()
      await page.waitForFunction(time => {
        const video = document.querySelector('video[aria-label="Original video evidence playback"]')
        return video?.readyState >= 1 && !video.seeking && Math.abs(video.currentTime - time) < .003
      }, frame.seek)
    }
    if (item.name === 'tandem') {
      await section.screenshot({ path: path.join(fixture.output, 'face-transition-browser.png'), animations: 'disabled' })
    }
    cases.push({ fixture: item.name, status: item.status, packet_visible: true,
      source_frames: 3, face_crops: 3, native_seek_s: item.frames.map(frame => frame.seek) })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
