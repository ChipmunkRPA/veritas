// SPDX-License-Identifier: Apache-2.0
// Capture source-bound photo-rich and flat browser layouts at their native size.
const fs = require('node:fs/promises')
const path = require('node:path')
const { chromium } = require('../../frontend/node_modules/playwright')

const labels = {
  gallery: 'Archive photo · Previous · Next',
  'photo-editor': 'Exposure · Crop · Tone · Export',
  'street-map': 'Trail map · View next marker',
  article: 'Local park guide · Trail report',
  'dense-plot': 'Monthly data dashboard',
  calendar: 'Community room booking calendar',
}
const content = {
  gallery: '<section class="bar low"><b>Field journal, item 42</b> &nbsp; September archive</section>',
  'photo-editor': '<aside><b>Light balance</b><hr>Exposure <input type="range"><hr>Contrast <input type="range"><hr>Highlights <input type="range"></aside>',
  'street-map': '<svg viewBox="0 0 1200 768"><path d="M90 700 L350 400 L680 460 L1080 120" stroke="#ffffaf" stroke-width="9" fill="none"/><circle cx="680" cy="460" r="20" fill="#36a1f3"/></svg><section class="bar low">Trail access · Open</section>',
  article: '<aside class="article"><b>Planning an afternoon walk</b><p>The district publishes four routes. Read this week’s notes before leaving the visitor center.</p><p>The photograph comes from the public reference archive.</p></aside>',
  'dense-plot': `<div class="chart">${Array.from({ length: 90 }, (_, i) => `<div style="height:${90 + ((i * 67) % 420)}px"></div>`).join('')}</div>`,
  calendar: `<div class="days">${Array.from({ length: 35 }, (_, i) => `<div>${(i % 31) + 1}</div>`).join('')}</div>`,
}
const style = `
body{margin:0;background:#eef3f7;color:#26323c;font:20px Arial,sans-serif}
img,svg{position:absolute;inset:0;width:100%;height:100%}
.bar,aside{position:absolute;background:#fcfcfddd;border-radius:9px;padding:22px}
.bar{top:12px;left:18px}.low{top:auto;bottom:14px}aside{right:18px;top:98px;width:170px}.article{width:240px}
.chart{position:absolute;inset:68px 0 0;background:linear-gradient(30deg,#112329,#386fad,#aac4b6);display:flex;align-items:flex-end;gap:3px;padding:28px 50px}
.chart div{width:18px;background:linear-gradient(#86dedb,#153981)}
.days{margin:120px 70px 0;display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.days div{background:#fff;padding:22px;height:60px}
`

async function capture() {
  const directory = path.resolve(process.argv[2])
  const plan = JSON.parse(await fs.readFile(path.join(directory, 'plan.json'), 'utf8'))
  const browser = await chromium.launch({ channel: 'chrome', headless: true })
  try {
    for (const row of plan.cases) {
      const [width, height] = row.dimensions
      const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
      let photo = ''
      if (row.layout !== 'dense-plot' && row.layout !== 'calendar') {
        const bytes = await fs.readFile(path.join(directory, row.photo))
        photo = `<img src="data:image/png;base64,${bytes.toString('base64')}">`
      }
      const header = `<section class="bar">${labels[row.layout]}</section>`
      await page.setContent(`<html><style>${style}</style>${photo}${content[row.layout]}${header}</html>`, { waitUntil: 'load' })
      await page.screenshot({ path: path.join(directory, row.screen), type: 'png' })
      await page.close()
    }
    await fs.writeFile(path.join(directory, 'browser.json'), JSON.stringify({ version: browser.version() }) + '\n', { flag: 'wx' })
  } finally { await browser.close() }
}

capture().catch(error => { console.error(error); process.exitCode = 1 })
