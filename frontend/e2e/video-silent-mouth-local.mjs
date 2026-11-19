import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
assert.equal(new URL(input.url).hostname, '127.0.0.1')
const browser = await chromium.launch({ channel: 'chrome', headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1050 }, extraHTTPHeaders: { 'X-API-Key': input.key } })
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const item = input.cases[0]
  const range = item.activity ? 'reduced speech activity' : 'near-silent'
  const title = item.activity ? 'Reduced speech activity during visible mouth movement' : 'Mouth movement with a near-silent soundtrack'
  await page.goto(`${input.url}/analyses/${item.analysis_id}/media-evidence?view=plain#media-evidence-timing`)
  await page.getByRole('tab', { name: /Audio\/picture timing/ }).waitFor()
  assert.equal(await page.getByRole('tab', { name: /Audio\/picture timing/ }).getAttribute('aria-selected'), 'true')
  if (item.shot_count) {
    const coverage = page.getByRole('region', { name: 'Speech and mouth comparison' })
    await coverage.getByText('Mouth movement during a near-silent interval.', { exact: false }).waitFor()
    assert.equal(await coverage.getByRole('listitem').count(), item.shot_count)
    assert.equal(await coverage.getByText('Shot length outside the checked range.', { exact: false }).count(), item.shots_outside_scope)
  }
  const link = page.getByRole('link', { name: 'Inspect source frames and waveform' })
  await link.waitFor({ timeout: 30_000 })
  assert.equal(await link.getAttribute('href'), '#media-evidence-VLT01')
  await link.click()
  const card = page.locator('#media-evidence-VLT01')
  assert((await card.innerText()).includes(title))
  if (item.stereo_levels) {
    for (const name of ['Left channel', 'Right channel', 'Channel difference']) {
      await card.getByText(`${name} sound level, before → during → after`, { exact: true }).waitFor()
    }
    await card.getByText('both contained measurable sound', { exact: false }).waitFor()
  }
  const pictures = []
  for (const [role, index] of Object.entries(item.frames)) {
    assert.match(await card.innerText(), new RegExp(`${role} ${range} range · Frame ${index}`, 'i'))
    for (const name of ['exact original video frame', 'exact measured mouth region']) {
      pictures.push(card.getByRole('img', { name: `VLT01.${role}: ${name}` }))
    }
  }
  pictures.push(card.getByRole('img', { name: item.activity ? 'Waveform around the reduced speech activity interval'
    : 'Waveform around the near-silent soundtrack interval' }))
  for (const picture of pictures) {
    await picture.scrollIntoViewIfNeeded()
    assert(await picture.evaluate(async image => { await image.decode(); return image.naturalWidth > 0 }))
  }
  const audio = card.getByLabel(item.activity ? 'Listen to the original channels mixed during reduced speech activity'
    : 'Listen to the near-silent soundtrack interval')
  await audio.scrollIntoViewIfNeeded()
  await audio.evaluate(element => { element.preload = 'metadata'; element.load() })
  await page.waitForFunction(() => document.querySelector('#media-evidence-VLT01 audio')?.readyState >= 1)
  assert.deepEqual(errors, [])
  await card.screenshot({ path: path.join(input.output, `${item.activity ? 'reduced-activity' : 'near-silent'}-browser.png`), animations: 'disabled' })
  console.log(JSON.stringify({ chromium: browser.version(), ordinary_view: true, decoded_images: pictures.length, audio_ready: true,
    ...(item.stereo_levels ? { stereo_levels_visible: true } : {}) }))
} finally {
  await browser.close()
}
