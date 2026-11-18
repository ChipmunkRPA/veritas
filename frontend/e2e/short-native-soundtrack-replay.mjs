import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({
    viewport: { width: 1366, height: 1024 },
    extraHTTPHeaders: { 'X-API-Key': input.key },
  })
  await context.addInitScript(({ origin, key }) => {
    if (location.origin === origin) localStorage.setItem('veritas_api_key', key)
  }, { origin: input.url, key: input.key })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const [sample] = input.cases
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`${input.url}/analyses/${sample.analysis_id}?view=plain`)
  const upload = page.getByRole('region', { name: 'Compare with an earlier video' })
  await upload.getByLabel('Earlier video', { exact: true }).setInputFiles(sample.reference)
  await upload.getByRole('checkbox', { name: 'Include experimental soundtrack comparison' }).check()
  const comparison = page.waitForResponse(response => response.request().method() === 'POST'
    && new URL(response.url()).pathname.endsWith('/video-reference-comparison'))
  await upload.getByRole('button', { name: 'Compare videos' }).click()
  const response = await comparison
  assert.ok(response.ok(), `Soundtrack comparison returned HTTP ${response.status()}`)
  const { comparison: record } = await response.json()
  assert.equal(record.sound_difference?.status, 'localized_difference', record.sound_difference?.reason)
  const proof = upload.getByRole('region', { name: sample.title ?? /^Soundtrack difference/ })
  const presented = sample.presented_s ?? '2.700'
  const [start, end] = sample.range_s ?? ['1.400', '1.720']
  await proof.getByText(`Common presented audio duration ${presented} s.`, { exact: false }).waitFor()
  await proof.getByText(`${start} to ${end} s`, { exact: false }).first().waitFor()
  if (sample.range_s) {
    await proof.getByText(/Sound matched before and after the interval/).waitFor()
    for (const [label, file] of [['Download JSON evidence', 'browser-comparison.json'], ['Download PDF', 'browser-comparison.pdf']]) {
      const [download] = await Promise.all([page.waitForEvent('download'), upload.getByRole('button', { name: label }).click()])
      await download.saveAs(path.join(input.output, file))
    }
  }
  const stills = proof.locator('img')
  assert.equal(await stills.count(), 6)
  for (const still of await stills.all()) await still.scrollIntoViewIfNeeded()
  await Promise.all((await stills.all()).map(still => still.evaluate(img => img.decode())))
  const lengths = await proof.locator('audio').evaluateAll(players => Promise.all(players.map(audio => new Promise((resolve, reject) => {
    audio.addEventListener('loadedmetadata', () => resolve(audio.duration), { once: true })
    audio.addEventListener('error', () => reject(new Error('Soundtrack excerpt did not decode')), { once: true })
    audio.load()
  }))))
  const duration = Number(end) - Number(start)
  assert.ok(lengths.length === 2 && lengths.every(length => Math.abs(length - duration) < .001))
  await proof.screenshot({ path: path.join(input.output, 'native-soundtrack-evidence.png'), animations: 'disabled' })
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ native_frames: 6, excerpt_durations_s: lengths, page_errors: errors.length }))
} finally {
  await browser.close()
}
