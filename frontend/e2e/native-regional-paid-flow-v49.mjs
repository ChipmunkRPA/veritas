// Local isolated app check. All images, keys and captures remain private.
import { chromium } from 'playwright'
import fs from 'node:fs/promises'
import path from 'node:path'

const folder = process.argv[2]
const input = JSON.parse(await fs.readFile(path.join(folder, 'browser-input.json'), 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 1050 }, extraHTTPHeaders: input.headers })
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await page.goto(`http://127.0.0.1:8164/analyses/${input.analysis_id}?view=plain`, { waitUntil: 'networkidle' })
  await page.getByTestId('v2-result').waitFor({ timeout: 30000 })
  const evidence = page.getByTestId('image-evidence')
  await evidence.scrollIntoViewIfNeeded()
  await page.getByTestId('image-evidence-card').first().waitFor()
  await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="image-evidence"] img')].every(i => i.complete && i.naturalWidth > 0))
  await page.screenshot({ path: path.join(folder, 'app-normal-view.png'), fullPage: true, animations: 'disabled' })
  const card = page.getByTestId('image-evidence-card').first()
  await card.screenshot({ path: path.join(folder, 'app-source-crop.png'), animations: 'disabled' })
  const mask = page.getByRole('button', { name: /^Show measured mask for/ }).first()
  await mask.click()
  await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="image-evidence"] img')].every(i => i.complete && i.naturalWidth > 0))
  await card.screenshot({ path: path.join(folder, 'app-measured-mask.png'), animations: 'disabled' })
  const body = await page.locator('body').innerText()
  if (/RelayFormer|TIDAL|local\.native_regional_v1|loc\.native_regional|examiner.s conclusion|not signed off/i.test(body)) throw Error('Private model identity or examination workflow exposed')
  if (errors.length) throw Error(`Browser errors: ${errors.join('; ')}`)
  await fs.writeFile(path.join(folder, 'browser-result.json'), JSON.stringify({ status: 'complete', ordinaryView: true, loadedSourceCrops: true, toggledMeasuredMask: true, pageErrors: 0, modelNamesHidden: true }, null, 2))
} finally {
  await browser.close()
}
