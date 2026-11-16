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
  const seconds = sample => `${(sample / 16000).toFixed(3)} s`
  for (const { analysis_id, name, review } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain`)
    const section = page.getByRole('region', { name: 'Two-second speech windows' })
    await section.getByRole('status').filter({ hasText: review.reason }).waitFor()
    const verified = []
    for (const region of review.regions) {
      const { start_sample: start, end_sample: end, artifacts } = region
      const span = `${seconds(start)}–${seconds(end)}`
      await section.getByText(`Source PTS [${start}, ${end}) at 1/16000.`).waitFor()
      const waveform = section.getByRole('img', { name: `Waveform for the two-second speech window at ${span}`, exact: true })
      await waveform.scrollIntoViewIfNeeded()
      assert.ok((await waveform.getAttribute('src')).includes(artifacts.waveform.sha256))
      const size = await waveform.evaluate(async image => {
        await image.decode()
        return [image.naturalWidth, image.naturalHeight]
      })
      assert.deepEqual(size, [artifacts.waveform.width, artifacts.waveform.height])
      const player = section.getByLabel(`Two-second speech window at ${span}`, { exact: true })
      assert.ok((await player.getAttribute('src')).includes(artifacts.selected_clip.sha256))
      const duration = await player.evaluate(audio => {
        if (audio.readyState >= 1) return audio.duration
        return new Promise((resolve, reject) => {
          audio.addEventListener('loadedmetadata', () => resolve(audio.duration), { once: true })
          audio.addEventListener('error', reject, { once: true })
          audio.preload = 'metadata'
          audio.load()
        })
      })
      assert.equal(duration, 2)
      verified.push({ start_sample: start, end_sample: end, waveform_pixels: size, playback_s: duration })
    }
    if (verified.length) await section.screenshot({ path: path.join(input.output, `${name}-browser.png`), animations: 'disabled' })
    cases.push({ case: name, verified })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
