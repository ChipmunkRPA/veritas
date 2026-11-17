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
    const cropped = row.mode === 'crop'
    const retimed = row.mode === 'retimed'
    const labels = retimed ? {
      heading: 'Footage repeated at two playback speeds', packet: 'VPR05',
      empty: `No match at ${row.fractional ? 'a 2:1, 3:2, or 4:3' : 'a two-to-one'} speed ratio was found in the first ${row.checked_frames} frames.`,
      link: 'Inspect the faster and slower source ranges', title: 'Footage repeated at two playback speeds',
    } : cropped ? {
      heading: 'Cropped footage check', packet: 'VPR07',
      empty: `No qualifying cropped repeat was found in the first ${row.checked_frames} frames.`,
      link: 'Inspect both time ranges and matching crops', title: 'Similar footage shown with a tighter crop',
    } : {
      heading: 'Mirrored footage check', packet: 'VPR06',
      empty: `No mirrored repeat was found in the first ${row.checked_frames} frames.`,
      link: 'Inspect both source ranges and frames', title: 'Similar footage flipped left to right',
    }
    await page.goto(`${input.url}/analyses/${row.analysis_id}/media-evidence?view=plain`)
    const coverage = page.getByRole('region', { name: labels.heading })
    await coverage.waitFor()
    const finding = page.locator(`#media-evidence-${labels.packet}`)
    if (!row.matched) {
      await coverage.getByText(labels.empty, { exact: true }).waitFor()
      assert.equal(await finding.count(), 0)
      cases.push({ fixture: row.name, matched: false })
      continue
    }
    await coverage.getByRole('link', { name: labels.link }).click()
    await finding.getByRole('heading', { name: labels.title }).waitFor()
    const textSeconds = value => `${value.toFixed(3)} s`
    for (const range of row.ranges_s) {
      await finding.getByText(range.map(textSeconds).join('–'), { exact: true }).waitFor()
    }
    if (row.fractional) {
      const first = row.ranges_s[0].map(textSeconds).join('–')
      const label = row.first_is_faster ? 'Faster occurrence' : 'Slower occurrence'
      await finding.getByText(new RegExp(`^${label} ${first}`)).waitFor()
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
      assert.deepEqual(dimensions, row.dimensions)
      await finding.getByRole('button', { name: `Go to frame ${frame.frame_index} in playback`, exact: true }).waitFor()
    }
    if (cropped) {
      const crops = finding.getByRole('img', { name: /wider source cropped to the matching area/ })
      assert.equal(await crops.count(), 2)
      for (const [index, dimensions] of row.crop_dimensions.entries()) {
        const image = crops.nth(index)
        await image.scrollIntoViewIfNeeded()
        assert.deepEqual(await image.evaluate(async element => {
          await element.decode()
          return [element.naturalWidth, element.naturalHeight]
        }), dimensions)
      }
    }
    let seek = null
    if (row.seek_verified_in_browser) {
      const anchor = row.frames[2]
      await finding.getByRole('button', { name: `Go to frame ${anchor.frame_index} in playback`, exact: true }).click()
      const player = page.getByLabel('Original video evidence playback')
      await page.waitForFunction(({ selector, time }) => {
        const video = document.querySelector(selector)
        return video && video.readyState >= 1 && !video.seeking && Math.abs(video.currentTime - time) < .002
      }, { selector: 'video[aria-label="Original video evidence playback"]', time: anchor.playback_time_s })
      seek = await player.evaluate(element => element.currentTime)
    }
    await finding.screenshot({ path: path.join(input.output, `${row.name}-browser.png`), animations: 'disabled' })
    cases.push({ fixture: row.name, matched: true, source_frames: row.frames.map(frame => frame.frame_index), seek_s: seek,
      ...(cropped ? { crop_images: 2 } : {}) })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
