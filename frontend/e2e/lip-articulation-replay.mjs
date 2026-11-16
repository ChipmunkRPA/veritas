// Replay the real prepaid evidence through the plain video results view.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1020 },
    extraHTTPHeaders: { 'X-API-Key': input.key } })
  await context.route('**/*', route => new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const [example] = input.cases
  await page.goto(`${input.url}/analyses/${example.analysis_id}/media-evidence?view=plain`)
  const coverage = page.getByRole('region', { name: 'Speech and mouth comparison' })
  const support = coverage.getByRole('link', { name: 'Inspect source frames and waveform' })
  await support.waitFor()
  assert.equal(await support.getAttribute('href'), '#media-evidence-VLT01')
  const card = page.locator('#media-evidence-VLT01')
  await card.getByRole('heading', { name: /Local speech and mouth mismatch/ }).waitFor()
  const [start, end] = example.witness.audio_source_pts.map(sample => (sample / 16000).toFixed(3))
  await card.getByText(`Audio source time ${start}–${end} s.`, { exact: false }).waitFor()
  for (const [role, index] of Object.entries(example.witness.picture_frames)) {
    await card.getByText(new RegExp(`${role} mismatch range · Frame ${index}`, 'i')).waitFor()
    const crop = card.getByAltText(new RegExp(`VLT01\\.${role}: exact measured mouth region`))
    await crop.waitFor()
    await crop.evaluate(image => image.decode())
  }
  const waveform = card.getByAltText('Waveform during the speech and mouth mismatch')
  await waveform.evaluate(image => image.decode())
  const playback = card.getByLabel('Listen to the soundtrack during the speech and mouth mismatch')
  assert.ok((await playback.getAttribute('src')).includes(example.sound_sha256))
  const audioSeconds = await playback.evaluate(async audio => {
    if (audio.readyState >= 1) return audio.duration
    const loaded = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Audio metadata did not load')), 10000)
      audio.addEventListener('loadedmetadata', () => {
        clearTimeout(timeout)
        resolve(audio.duration)
      }, { once: true })
      audio.addEventListener('error', () => {
        clearTimeout(timeout)
        reject(new Error('The browser could not decode the audio excerpt'))
      }, { once: true })
    })
    audio.preload = 'metadata'
    audio.load()
    return loaded
  })
  assert.ok(Math.abs(audioSeconds - (example.witness.audio_source_pts[1] - example.witness.audio_source_pts[0]) / 16000) < .001)
  await card.screenshot({ path: path.join(input.output, 'mismatch-browser.png'), animations: 'disabled' })
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ original_mouth_crops: 3, matching_audio_range: true, audio_seconds: audioSeconds, page_errors: 0 }))
} finally {
  await browser.close()
}
