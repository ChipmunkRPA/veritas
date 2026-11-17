import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const input = JSON.parse(fs.readFileSync(0, 'utf8'))
const browser = await chromium.launch({ headless: true })
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    extraHTTPHeaders: { 'X-API-Key': input.key },
  })
  await context.route('**/*', route =>
    new URL(route.request().url()).origin === input.url ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []
  const checked = []
  page.on('pageerror', error => errors.push(error.message))
  for (const sample of input.cases) {
    await page.goto(`${input.url}/analyses/${sample.analysis_id}?view=plain#document-published-wording`)
    const section = page.getByRole('region', { name: 'Earlier published wording' })
    await section.getByRole('link', { name: `U.S. GAO report ${sample.reference}` }).first().waitFor()
    assert.equal(await section.getByText(`Published ${sample.date}`, { exact: false }).count(), sample.sections ?? 1)
    assert.equal(await section.getByText('Submitted wording', { exact: true }).count(), sample.changes)
    for (const sentence of sample.sentences ?? []) {
      await section.getByRole('heading', { name: `Sentence ${sentence.sentence}`, exact: true }).waitFor()
      for (const number of sentence.differences) {
        const target = `#published-wording-difference-${number}`
        assert.ok(await section.getByRole('link', { name: `Difference ${number}`, exact: true }).count())
        await section.locator(target).getByText('Submitted wording', { exact: true }).waitFor()
      }
    }
    if (sample.sentences?.length) {
      const link = section.getByRole('link', { name: `Difference ${sample.sentences[0].differences[0]}`, exact: true }).first()
      const href = await link.getAttribute('href')
      await link.click()
      await page.waitForURL(url => url.hash === href)
    }
    if (sample.sections) {
      assert.equal(await section.getByRole('heading', { name: /^Original PDF page / }).count(), sample.sections)
      for (const [sourcePage, lo, hi] of sample.locations) {
        const location = `Original PDF page ${sourcePage} · Extracted characters [${lo}, ${hi})`
        assert.ok(await section.getByText(location, { exact: false }).count())
      }
    }
    if (sample.cleared) {
      const style = page.getByRole('region', { name: 'Possible AI-like writing passages' })
      assert.equal(await style.locator('blockquote').count(), 0)
      assert.equal(await style.getByRole('link', { name: 'source comparison' }).getAttribute('href'), '#document-published-wording')
    }
    const image = path.join(input.output, `${sample.case}-evidence.png`)
    await section.screenshot({ path: image, animations: 'disabled' })
    checked.push(sample.case)
  }
  assert.deepEqual(errors, [])
  process.stdout.write(JSON.stringify({ checked, page_errors: errors.length }))
} finally {
  await browser.close()
}
