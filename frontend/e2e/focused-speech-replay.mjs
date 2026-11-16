// Replay the waveform and playable excerpts from the prepaid audio check.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({
    viewport: { width: 1366, height: 1020 },
    extraHTTPHeaders: { 'X-API-Key': input.key },
  })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const excerpts = []
  for (const { analysis_id, name, review } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain`)
    const card = page.getByRole('region', { name: review.title })
    const { start_sample, end_sample } = review.interval
    await card.getByText(`samples [${start_sample}, ${end_sample})`, { exact: false }).waitFor()
    await card.getByText(`Source PTS [${start_sample}, ${end_sample}) at 1/16000.`).waitFor()
    const waveform = card.getByAltText('Waveform with the focused speech window marked')
    const [width, height] = await waveform.evaluate(async image => {
      await image.decode()
      return [image.naturalWidth, image.naturalHeight]
    })
    assert.deepEqual([width, height], [review.artifacts.waveform.width, review.artifacts.waveform.height])
    const playback = card.getByLabel('Focused speech window', { exact: true })
    assert.ok((await playback.getAttribute('src')).includes(review.artifacts.selected_clip.sha256))
    const duration = await playback.evaluate(audio => {
      if (audio.readyState >= 1) return audio.duration
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Audio metadata did not load')), 10000)
        audio.addEventListener('loadedmetadata', () => {
          clearTimeout(timeout)
          resolve(audio.duration)
        }, { once: true })
        audio.addEventListener('error', () => {
          clearTimeout(timeout)
          reject(new Error('The browser could not decode the speech excerpt'))
        }, { once: true })
        audio.preload = 'metadata'
        audio.load()
      })
    })
    assert.ok(Math.abs(duration - (end_sample - start_sample) / 16000) < .001)
    await card.screenshot({ path: path.join(input.output, `${name}-browser.png`), animations: 'disabled' })
    excerpts.push({ case: name, start_sample, end_sample, playback_seconds: duration, waveform_pixels: [width, height] })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ excerpts, page_errors: errors.length }))
} finally {
  await browser.close()
}
