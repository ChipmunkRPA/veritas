// SPDX-License-Identifier: Apache-2.0
// Capture project-authored layouts around unchanged, attributed photographs.
const fs = require('node:fs/promises')
const path = require('node:path')
const { chromium } = require('../../frontend/node_modules/playwright')

const layouts = {
  film: '<footer><span>Photo archive</span><span>Frame 08 · View full size</span></footer>',
  reader: '<aside><h2>Trail record</h2><p>Archived on Tuesday</p><p>Local reference copy</p><button>Read the route notes</button></aside>',
  cart: '<section><b>Selected item</b><p>Unframed print · In stock</p><button>Return to collection</button></section>',
  table: '<main><h1>City room schedules</h1><table><tbody>' +
    Array.from({ length: 12 }, (_, i) => '<tr><td>Meeting space ' + (i + 1) +
      '</td><td>Available</td><td>Daily timetable</td></tr>').join('') + '</tbody></table></main>',
  inspector: '<header><span>Archive viewer</span><span>Pan · Zoom · Measure</span></header><aside><h2>Layers</h2><p>Background image</p><p>Visible and locked</p></aside>',
  small: '<footer class="small">Field archive · View item 52</footer>',
}

async function capture(directory) {
  const plan = JSON.parse(await fs.readFile(path.join(directory, 'capture-plan.json'), 'utf8'))
  const browser = await chromium.launch({ channel: 'chrome', headless: true })
  try {
    for (const { id, layout, size, photo } of plan) {
      const [width, height] = size
      const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
      try {
        const data = await fs.readFile(path.join(directory, photo), 'base64')
        const image = layout === 'table' ? '' : '<img src="data:image/png;base64,' + data + '">'
        const ink = width === 1280 ? '#ebf2fa' : '#192b39'
        const fill = width === 1280 ? '#152735' : '#f4f8fa'
        await page.setContent(`<!doctype html><html lang="en"><meta charset="utf-8"><style>
          *{box-sizing:border-box}body{margin:0;background:${fill};color:${ink};font:18px Arial,sans-serif}
          img{position:absolute;inset:0;width:100%;height:100%}header,footer,aside,section,main{position:absolute;background:${fill}}
          header,footer{width:100%;padding:10px 26px;display:flex;justify-content:space-between}
          header{top:0}footer{bottom:0}aside{right:35px;top:120px;padding:18px 24px;border:1px solid currentColor}
          section{left:40px;bottom:40px;padding:20px}button{padding:7px 10px;font:inherit}
          .small{bottom:15px;left:20px;width:auto;padding:8px 12px;font-size:14px}
          main{width:100%;height:100%;padding:50px}table{border-collapse:collapse;width:95%}td{padding:13px 18px;border:1px solid currentColor}
          </style>${image}${layouts[layout]}`, { waitUntil: 'load' })
        await page.screenshot({ path: path.join(directory, id + '.png'), type: 'png' })
      } finally {
        await page.close()
      }
    }
    await fs.writeFile(path.join(directory, 'capture-browser.json'), JSON.stringify({ version: browser.version() }) + '\n', { flag: 'wx' })
  } finally {
    await browser.close()
  }
}

capture(path.resolve(process.argv[2])).catch(error => { console.error(error); process.exitCode = 1 })
