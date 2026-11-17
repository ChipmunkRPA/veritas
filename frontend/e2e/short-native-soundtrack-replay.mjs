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
  await upload.getByRole('button', { name: 'Compare videos' }).click()
  const proof = upload.getByRole('region', { name: /^Soundtrack difference/ })
  await proof.getByText(/Common presented audio duration 2\.700 s/).waitFor()
  await proof.getByText(/1\.400 to 1\.720 s/).waitFor()
  const stills = proof.locator('img')
  assert.equal(await stills.count(), 6)
  for (const still of await stills.all()) await still.scrollIntoViewIfNeeded()
  await page.waitForFunction(() => [...document.querySelectorAll('section[aria-label^="Soundtrack difference"] img')].every(img => img.naturalWidth > 0))
  const lengths = await proof.locator('audio').evaluateAll(players => Promise.all(players.map(audio => new Promise((resolve, reject) => {
    audio.addEventListener('loadedmetadata', () => resolve(audio.duration), { once: true })
    audio.addEventListener('error', () => reject(new Error('Soundtrack excerpt did not decode')), { once: true })
    audio.load()
  }))))
  assert.ok(lengths.length === 2 && lengths.every(duration => Math.abs(duration - .320) < .001))
  await proof.screenshot({ path: path.join(input.output, 'native-soundtrack-evidence.png'), animations: 'disabled' })
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ native_frames: 6, excerpt_durations_s: lengths, page_errors: errors.length }))
} finally {
  await browser.close()
}
