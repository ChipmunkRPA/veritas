// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { root, analysis_id, title, api_key, url } = JSON.parse(fs.readFileSync(0, 'utf8'))
const { chromium } = require(path.join(root, 'frontend/node_modules/playwright'))

;(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 960 }, reducedMotion: 'reduce' })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.addInitScript(key => localStorage.setItem('veritas_api_key', key), api_key)
    await page.goto(url + '/analyses/' + analysis_id + '/media-evidence?view=plain', { waitUntil: 'domcontentloaded' })
    const review = page.getByRole('region', { name: title, exact: true })
    await review.getByText(/3\.000 s–4\.500 s/).first().waitFor({ timeout: 15000 })
    await review.getByText(/48000, 72000/).first().waitFor()
    await review.getByText(/overlap/i).first().waitFor()
    const waveform = review.getByAltText(/^Waveform for the selected speech range/)
    assert.equal(await waveform.count(), 1)
    await waveform.scrollIntoViewIfNeeded()
    await waveform.evaluate(item => item.decode())
    assert.deepEqual(await waveform.evaluate(item => [item.naturalWidth, item.naturalHeight]), [960, 280])
    const audio = review.locator('audio[aria-label^="Selected speech range"]')
    assert.equal(await audio.count(), 1)
    await audio.evaluate(item => { item.preload = 'metadata'; item.load() })
    await page.waitForFunction(() => {
      const element = document.querySelector('#media-evidence-long-voice-review audio')
      return element?.readyState >= 1 && Math.abs(element.duration - 1.5) < 0.00001
    })
    assert.deepEqual(errors, [])
    console.log(JSON.stringify({ plain_view: true, range_seconds: [3, 4.5], sample_interval: [48000, 72000],
      waveform_dimensions: [960, 280], playback_seconds: 1.5, page_errors: errors.length }))
  } finally {
    await browser.close()
  }
})().catch(error => { console.error(error); process.exitCode = 1 })
