// SPDX-License-Identifier: Apache-2.0
// Render original non-generative browser controls at native viewport resolution.
const fs = require('node:fs/promises')
const path = require('node:path')
const { chromium } = require('../../frontend/node_modules/playwright')

const layouts = [
  ['reading', '<h1>Neighborhood archive</h1><p>Meeting notes, volunteer schedules, and a guide to the new reading room.</p><article><h2>November open hours</h2><p>Tuesday to Saturday. The reference collection opens at nine.</p><h2>Member requests</h2><p>Sign up at the front desk for a short orientation.</p></article>'],
  ['dashboard', '<h1>Workshop schedule</h1><p>Weekly registration totals</p><div class="bars">'+[3, 8, 5, 7, 4, 9].map((n, i) => `<div style="height:${n * 28}px" aria-label="Week ${i + 1}"></div>`).join('')+'</div><p>All totals represent completed registrations.</p>'],
  ['form', '<h1>Room booking</h1><p>Select a date and tell us what you will need.</p><article><label>Meeting title <input value="Friday planning"></label><label>Room <select><option>Library meeting room</option></select></label><label>Notes <textarea>Chairs for 12 people</textarea></label><button>Save reservation</button></article>'],
  ['pattern', '<h1>Product material samples</h1><p>Review the current colors and surface patterns.</p><div class="swatches"><b>Paper</b><b>Fabric</b><b>Tile</b><b>Vinyl</b></div><p>Display colors vary with brightness and calibration.</p>'],
]

async function capture() {
  const destination = path.resolve(process.argv[2])
  const browser = await chromium.launch({ channel: 'chrome', headless: true })
  const version = browser.version()
  const paths = []
  try {
    for (const [name, content] of layouts) {
      for (const [theme, ink, paper, surface, accent] of [
        ['light', '#203141', '#e9eff1', '#fff', '#287879'], ['dark', '#e8edf2', '#101c2a', '#263a4a', '#b27d51'],
      ]) {
        const page = await browser.newPage({ viewport: { width: 1360, height: 1080 }, deviceScaleFactor: 1 })
        await page.setContent(`<html lang="en"><meta charset="utf-8"><style>
          *{box-sizing:border-box}body{margin:0;background:${paper};color:${ink};font:19px Arial,sans-serif}nav,main{width:1060px;margin:0 auto}
          header{background:${surface};border-bottom:2px solid ${accent};padding:25px 0}nav{display:flex;justify-content:space-between}
          main{padding:72px 0}article,.bars,.swatches{margin-top:42px;padding:35px;background:${surface};border-radius:15px}
          article>*{display:block;margin:0 0 24px}.bars,.swatches{display:flex;gap:25px;align-items:flex-end}.bars div{width:84px;background:${accent}}
          .swatches b{display:block;padding:98px 31px 19px;background:repeating-linear-gradient(60deg,${surface} 0 12px,${accent} 12px 14px)}
          input,textarea,select,button{display:block;margin-top:10px;padding:12px;font:inherit}h1{font-size:52px}h2{font-size:28px}
          </style><header><nav><b>Community Commons</b><span>Home · Directory · Help</span></nav></header><main>${content}</main></html>`)
        const file = `${name}-${theme}.png`
        await page.screenshot({ path: path.join(destination, file), type: 'png' })
        await page.close()
        paths.push({ file, group: name, theme })
      }
    }
  } finally { await browser.close() }
  await fs.writeFile(path.join(destination, 'screenshots.json'), JSON.stringify({ version, paths }) + '\n', { flag: 'wx' })
}

capture().catch(error => { console.error(error); process.exitCode = 1 })
