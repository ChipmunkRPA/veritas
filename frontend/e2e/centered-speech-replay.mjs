import assert from 'node:assert/strict'
import fs from 'node:fs'
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
  for (const { analysis_id, name, review } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain`)
    const card = page.getByRole('region', { name: 'Focused speech edit review' })
    await card.getByText(review.reason, { exact: true }).waitFor()
    await card.getByText(/^Search scope /).waitFor()
    const scope = await card.getByText(/^Search scope /).textContent()
    for (const sample of Object.values(review.checked_scope)) {
      assert.ok(scope.includes((sample / 16000).toFixed(3)))
    }
    const evidence = { case: name, checked_scope: review.checked_scope, status: review.status }
    if (review.interval) {
      const { start_sample: start, end_sample: end } = review.interval
      await card.getByText(`Source PTS [${start}, ${end}) at 1/16000.`).waitFor()
      const image = card.getByRole('img', { name: 'Waveform with the focused speech window marked' })
      await image.scrollIntoViewIfNeeded()
      assert.ok((await image.getAttribute('src')).includes(review.artifacts.waveform.sha256))
      evidence.waveform = await image.evaluate(async element => {
        await element.decode()
        return [element.naturalWidth, element.naturalHeight]
      })
      assert.deepEqual(evidence.waveform, [960, 280])
      const player = card.getByLabel('Focused speech window', { exact: true })
      assert.ok((await player.getAttribute('src')).includes(review.artifacts.selected_clip.sha256))
      evidence.playback_s = await player.evaluate(audio => new Promise((resolve, reject) => {
        if (audio.readyState >= 1) return resolve(audio.duration)
        audio.addEventListener('loadedmetadata', () => resolve(audio.duration), { once: true })
        audio.addEventListener('error', reject, { once: true })
        audio.preload = 'metadata'
        audio.load()
      }))
      assert.equal(evidence.playback_s, (end - start) / 16000)
    } else {
      assert.equal(await card.getByLabel('Focused speech window', { exact: true }).count(), 0)
    }
    cases.push(evidence)
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
