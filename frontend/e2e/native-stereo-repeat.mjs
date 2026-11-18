// Check the original stereo audio and waveforms exposed by the local app.
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
  const results = []
  for (const { analysis_id, name, pairs, rate, unavailable, description = 'Times and excerpts use the original 48 kHz stereo recording.' } of input.cases) {
    const anchor = rate ? 'rate-repetition' : 'aligned-repetition'
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain#media-evidence-${anchor}`)
    if (unavailable) {
      const heading = 'Stereo Opus repeat check unavailable'
      const skipped = page.getByRole('region', { name: heading })
      await skipped.getByRole('heading', { name: heading }).waitFor()
      await skipped.getByText(/continuous 3–4 second stereo Ogg Opus/).waitFor()
      assert.equal(await page.getByRole('region', { name: 'Repeated sound after stereo alignment' }).count(), 0)
      results.push({ case: name, unavailable: true })
      continue
    }
    const card = page.getByRole('region', { name: rate ? 'Repeated sound at different speeds' : 'Repeated sound after stereo alignment' })
    await card.getByText(description, { exact: false }).waitFor()
    const clips = []
    for (const pair of pairs) {
      if (rate) {
        const speed = (pair.duration_denominator / pair.duration_numerator).toFixed(2)
        await card.getByRole('heading', { name: `${pair.id} · The ${pair.faster_occurrence} occurrence plays about ${speed}× as fast` }).waitFor()
      }
      for (const side of ['first', 'second']) {
        const label = side === 'first' ? 'First' : 'Second'
        const range = pair[side].source_timestamp_ranges[0]
        await card.getByText(`Source PTS [${range.source_start_pts}, ${range.source_end_pts}) at 1/48000.`, { exact: false }).waitFor()
        const waveform = card.getByAltText(`${pair.id} · ${label} occurrence waveform`)
        const dimensions = await waveform.evaluate(async image => {
          await image.decode()
          return [image.naturalWidth, image.naturalHeight]
        })
        assert.deepEqual(dimensions, [960, 280])
        const playback = card.getByLabel(`${pair.id} · ${label} occurrence in the original recording`)
        await playback.evaluate(audio => {
          audio.preload = 'metadata'
          audio.load()
        })
        await page.waitForFunction(label => {
          const audio = document.querySelector(`audio[aria-label="${label}"]`)
          return audio && audio.readyState >= 1
        }, `${pair.id} · ${label} occurrence in the original recording`)
        const duration = await playback.evaluate(audio => audio.duration)
        const source = pair.artifacts[`${side}_recorded_clip`]
        assert.ok(Math.abs(duration - source.samples / source.sample_rate) < .001)
        clips.push({ time_base: range.time_base, source_pts: [range.source_start_pts, range.source_end_pts],
          playback_seconds: duration, waveform_pixels: dimensions })
      }
    }
    await card.screenshot({ path: path.join(input.output, `${name}-stereo-repeat.png`), animations: 'disabled' })
    results.push({ case: name, clips })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases: results, page_errors: errors.length }))
} finally {
  await browser.close()
}
