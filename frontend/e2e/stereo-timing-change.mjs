// Check the source clips and plots for a temporary stereo timing change.
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
  const errors = [], cases = []
  page.on('pageerror', error => errors.push(error.message))
  for (const { analysis_id, name, clips, video_origin_s } of input.cases) {
    await page.goto(`${input.url}/analyses/${analysis_id}/media-evidence?view=plain#media-evidence-stereo-timing-change`)
    const card = page.getByRole('region', { name: 'Temporary change in stereo timing' })
    await card.getByText('This finding carries no AI confidence weight.', { exact: false }).waitFor()
    const plots = []
    for (const alt of ['Measured delay between the stereo channels', 'Left-channel waveform with the flagged interval highlighted']) {
      const plot = card.getByAltText(alt)
      await plot.scrollIntoViewIfNeeded()
      const dimensions = await plot.evaluate(async image => {
        await image.decode()
        return [image.naturalWidth, image.naturalHeight]
      })
      assert.deepEqual(dimensions, [960, 280])
      plots.push(dimensions)
    }
    const recorded = []
    for (const clip of clips) {
      const range = clip.source_timestamp_ranges[0]
      const source = [range.source_start_pts, range.source_end_pts]
      const times = `Source PTS [${source[0]}, ${source[1]}) at ${range.time_base}.`
      const playback = source.map(pts => (pts / 48000 - video_origin_s).toFixed(3))
      await card.getByText(video_origin_s == null ? times : `${times} Video playback ${playback[0]} s–${playback[1]} s.`).waitFor()
      const role = clip.role[0].toUpperCase() + clip.role.slice(1)
      const player = card.getByLabel(`${role} the change · Original stereo audio`)
      const read = await player.evaluate(async audio => {
        audio.muted = true
        await audio.play()
        const duration = audio.duration
        audio.pause()
        return duration
      })
      assert.ok(Math.abs(read - (clip.end_sample - clip.start_sample) / 48000) < .001)
      recorded.push({ role: clip.role, time_base: range.time_base, source_pts: source, duration_seconds: read })
    }
    await card.screenshot({ path: path.join(input.output, `${name}-stereo-timing.png`), animations: 'disabled' })
    cases.push({ case: name, plots, clips: recorded })
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ cases, page_errors: errors.length }))
} finally {
  await browser.close()
}
