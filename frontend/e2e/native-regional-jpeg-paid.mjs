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
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`http://127.0.0.1:8164/analyses/${input.analysis_id}?view=plain`, { waitUntil: 'networkidle' })
  await page.getByTestId('v2-result').waitFor({ timeout: 30000 })
  const card = page.getByTestId('image-evidence-card').and(page.locator(`[data-evidence-id="${input.packet_id}"]`))
  await card.scrollIntoViewIfNeeded()

  async function checkImage(label, digest) {
    const asset = card.getByRole('img', { name: label })
    await asset.scrollIntoViewIfNeeded()
    await page.waitForFunction((name) => {
      const image = [...document.images].find(item => item.alt === name)
      return image?.complete && image.naturalWidth > 0
    }, label)
    const url = await asset.getAttribute('src')
    if (!url?.includes(digest)) throw Error(`Rendered image has a different SHA-256 for ${label}`)
  }

  await checkImage(`${input.packet_id}: full source image with the inspection region outlined`, input.context_sha256)
  await checkImage(`${input.packet_id}: unaltered source crop with surrounding context`, input.crop_sha256)
  await card.screenshot({ path: path.join(folder, 'app-source-crop.png'), animations: 'disabled' })
  await card.getByRole('button', { name: `Show measured mask for ${input.packet_id}` }).click()
  await checkImage(`${input.packet_id}: selected connected region on the full measured grid`, input.mask_sha256)
  await card.screenshot({ path: path.join(folder, 'app-measured-mask.png'), animations: 'disabled' })
  const visible = await page.locator('body').innerText()
  if (/RelayFormer|TIDAL|local\.native_regional_v1|loc\.native_regional|examiner.s conclusion|not signed off/i.test(visible)) {
    throw Error('Private model identity or examination workflow exposed')
  }
  if (errors.length) throw Error(`Browser errors: ${errors.join('; ')}`)
  await fs.writeFile(path.join(folder, 'browser-result.json'), JSON.stringify({ modelNamesHidden: true }))
} finally {
  await browser.close()
}
