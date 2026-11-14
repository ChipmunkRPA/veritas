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
  if (input.earlier_png) {
    const comparison = page.getByRole('region', { name: 'Compare with an earlier copy' })
    async function compare(filename) {
      await comparison.getByLabel('Earlier copy').setInputFiles(filename)
      const response = page.waitForResponse(reply => reply.request().method() === 'POST' &&
        reply.url().endsWith('/image-reference-comparison'))
      await comparison.getByRole('button', { name: 'Compare images' }).click()
      const { comparison: record, pdf } = await (await response).json()
      if (record?.schema !== 'chipmunk.image-reference-comparison/18' || record.regions.length ||
          record.affects_overall_confidence !== false || !pdf) throw Error('Earlier-copy response omitted exact WebP evidence')
      return record
    }
    const unassessed = await compare(input.earlier_png)
    if (unassessed.status !== 'webp_framing_unassessed') throw Error('Expected insufficient shared detail')
    await comparison.getByText('Comparison unassessed').waitFor()
    const record = await compare(input.matching_png)
    if (record.status !== 'comparable' || record.exact_webp.changed_pixels !== 0) throw Error('Expected exact RGB identity')
    await comparison.getByText(/All decoded RGB pixels match/).waitFor()
    const received = page.waitForEvent('download')
    await comparison.getByRole('button', { name: 'Download comparison JSON' }).click()
    const download = await received
    const downloaded = JSON.parse(await fs.readFile(await download.path(), 'utf8'))
    if (JSON.stringify(downloaded.comparison) !== JSON.stringify(record)) throw Error('Browser changed the comparison while exporting')
    await comparison.screenshot({ path: path.join(folder, 'app-earlier-copy.png'), animations: 'disabled' })
  }
  const visible = await page.locator('body').innerText()
  if (/RelayFormer|TIDAL|local\.native_regional_v1|loc\.native_regional|examiner.s conclusion|not signed off/i.test(visible)) {
    throw Error('Private model identity or examination workflow exposed')
  }
  if (errors.length) throw Error(`Browser errors: ${errors.join('; ')}`)
  await fs.writeFile(path.join(folder, 'browser-result.json'), JSON.stringify({ modelNamesHidden: true }))
} finally {
  await browser.close()
}
