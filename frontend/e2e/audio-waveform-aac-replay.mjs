import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { chromium } from 'playwright'

const fixture = JSON.parse(readFileSync(0, 'utf8'))
const browser = await chromium.launch()
try {
  const context = await browser.newContext({ extraHTTPHeaders: { 'X-API-Key': fixture.key } })
  await context.route('**/*', route => new URL(route.request().url()).origin === fixture.url ? route.continue() : route.abort())
  const page = await context.newPage()
  for (const row of fixture.cases) {
    const expected = JSON.parse(readFileSync(row.expected, 'utf8'))
    const waveform = expected.comparison.matching_waveform
    const flagged = row.status === 'localized_difference'
    await page.goto(`${fixture.url}/analyses/${row.analysis_id}/media-evidence?view=plain`)
    await page.getByLabel('Earlier audio', { exact: true }).setInputFiles(row.earlier)
    await page.getByRole('checkbox', { name: 'The earlier file is a clean copy of the same recording' }).check()
    await page.getByText(/waveform check also supports six-second 48 kHz mono AAC M4A/).waitFor()
    await page.getByRole('button', { name: 'Compare recordings' }).click()
    const region = page.getByRole('region', { name: 'Matching speech waveform' })
    await region.getByText(waveform.summary, { exact: true }).waitFor()
    await region.getByText(/6\.000 s · 48 kHz mono source/).waitFor()
    assert.equal(await region.getByRole('img', { name: /matching waveform excerpt waveform/ }).count(), flagged ? 2 : 0)
    if (flagged) {
      const [start, end] = waveform.range.submitted.native_source_pts
      await region.getByText(`AAC presentation [${start}, ${end}) on the 48 kHz clock.`).waitFor()
    }
    for (const label of ['Download JSON evidence', 'Download PDF']) {
      const download = page.waitForEvent('download')
      await page.getByRole('button', { name: label, exact: true }).last().click()
      const result = await download
      const bytes = readFileSync(await result.path())
      if (label.endsWith('PDF')) assert.equal(bytes.subarray(0, 5).toString(), '%PDF-')
      else assert.deepEqual(JSON.parse(bytes.toString()), expected)
    }
  }
  process.stdout.write(JSON.stringify({ results: fixture.cases.length, pdf_downloads: fixture.cases.length,
    json_downloads_exact: fixture.cases.length }))
} finally {
  await browser.close()
}
