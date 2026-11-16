import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({ extraHTTPHeaders: { 'X-API-Key': input.key } })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const cases = []
  for (const row of input.cases) {
    await page.goto(`${input.url}/analyses/${row.analysis_id}/media-evidence?view=plain`)
    const coverage = page.getByRole('region', { name: 'Mirrored footage check' })
    await coverage.waitFor()
    const finding = page.locator('#media-evidence-VPR06')
    if (!row.matched) {
      await coverage.getByText('No mirrored repeat was found in the first 225 frames.', { exact: true }).waitFor()
      assert.equal(await finding.count(), 0)
      cases.push({ fixture: row.name, matched: false })
      continue
    }
    await coverage.getByRole('link', { name: 'Inspect both source ranges and frames' }).click()
    await finding.getByRole('heading', { name: 'Similar footage flipped left to right' }).waitFor()
    const textSeconds = value => `${value.toFixed(3)} s`
    for (const range of row.ranges_s) {
      await finding.getByText(range.map(textSeconds).join('–'), { exact: true }).waitFor()
    }
    const images = finding.getByRole('img', { name: /exact original video frame/ })
    assert.equal(await images.count(), 4)
    for (const [index, frame] of row.frames.entries()) {
      const image = images.nth(index)
      await image.scrollIntoViewIfNeeded()
      const dimensions = await image.evaluate(async element => {
        await element.decode()
        return [element.naturalWidth, element.naturalHeight]
      })
      assert.deepEqual(dimensions, [640, 360])
      await finding.getByRole('button', { name: `Go to frame ${frame.frame_index} in playback`, exact: true }).waitFor()
    }
    const anchor = row.frames[2]
    await finding.getByRole('button', { name: `Go to frame ${anchor.frame_index} in playback`, exact: true }).click()
    const player = page.getByLabel('Original video evidence playback')
    await page.waitForFunction(({ selector, time }) => {
      const video = document.querySelector(selector)
      return video && video.readyState >= 1 && !video.seeking && Math.abs(video.currentTime - time) < .002
    }, { selector: 'video[aria-label="Original video evidence playback"]', time: anchor.playback_time_s })
    const seek = await player.evaluate(element => element.currentTime)
    await finding.screenshot({ path: path.join(input.output, `${row.name}-browser.png`), animations: 'disabled' })
    cases.push({ fixture: row.name, matched: true, source_frames: row.frames.map(frame => frame.frame_index), seek_s: seek })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
