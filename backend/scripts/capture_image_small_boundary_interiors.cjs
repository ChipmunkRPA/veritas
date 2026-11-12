// SPDX-License-Identifier: Apache-2.0
// Render fixed photo-heavy browser fixtures without external requests.

const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');

const widths = [704, 768];
const heights = [544, 512];
const types = ['flat', 'translucent', 'patterned'];
const plan = { schema: 'chipmunk.image-small-boundary-interior-captures/1',
  screens: Array.from({ length: 9 }, (_, slot) => ({ slot, type: types[slot % types.length] })),
  viewports: widths.map((width, i) => ({ width, height: heights[i] })) };
const sha = value => createHash('sha256').update(value).digest('hex');
const json = value => `${JSON.stringify(value, null, 2)}\n`;

function layout(slot, type, photo) {
  const colors = ['#f4efdc', '#e4ebfa', '#f7e5dd'];
  const fills = {
    flat: colors[slot % 3],
    translucent: `rgba(243, 249, 255, ${[.55, .65, .73][Math.floor(slot / 3)]})`,
    patterned: 'repeating-linear-gradient(42deg,#f2eedc 0px,#f2eedc 2px,#ddd9c6 2px,#ddd9c6 4px)',
  };
  const labels = ['Gallery', 'Image viewer', 'Place guide'];
  const panel = `<div class="panel"><strong>${['Details', 'Info', 'Layers'][slot % 3]}</strong>
    <span>View ${slot + 1}</span><small>Open</small></div>`;
  return `<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#152335;color:#f5f8ff;font:15px Arial,sans-serif}
header,footer{height:42px;padding:11px 21px;display:flex;justify-content:space-between}
main{height:calc(100vh - 72px);position:relative;background-size:cover;background-position:center;
background-image:url('data:image/jpeg;base64,${photo.toString('base64')}')}
.panel{position:absolute;left:${145 + 23 * (slot % 4)}px;top:${113 + 27 * (slot % 5)}px;
width:${[68, 79, 84, 91][slot % 4]}px;height:${[76, 66, 87, 71][slot % 4]}px;background:${fills[type]};
color:#283143;padding:9px 8px;border:1px solid #e6e9ef;box-shadow:0 2px 4px #18202e40;font-size:13px}
.panel span,.panel small{display:block;margin-top:4px}.panel small{font-size:10px}
footer{font-size:12px;height:30px;padding-top:7px}</style>
<header><strong>${labels[slot % 3]}</strong><span>Open · Inspect · Export</span></header>
<main>${panel}</main><footer><span>Preview</span><span>Fit to window</span></footer>`;
}

async function capture(run) {
  if (await fs.readFile(path.join(run, 'plan.json'), 'utf8') !== json(plan)) throw new Error('Capture plan changed.');
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({ deviceScaleFactor: 1, reducedMotion: 'reduce', serviceWorkers: 'block' });
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    const output = path.join(run, 'captures');
    await fs.mkdir(output);
    const rows = [];
    for (const { slot, type } of plan.screens) {
      const photo = await fs.readFile(path.join(run, 'screen-photos', `${slot}.jpg`));
      const html = layout(slot, type, photo);
      await page.setContent(html, { waitUntil: 'load' });
      for (const [view, size] of plan.viewports.entries()) {
        await page.setViewportSize(size);
        await page.evaluate(async () => {
          await document.fonts.ready;
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        });
        const png = await page.screenshot({ animations: 'disabled' });
        const filename = `${slot}-${view}.png`;
        await fs.writeFile(path.join(output, filename), png, { flag: 'wx' });
        rows.push({ slot, view, type, filename, size: [size.width, size.height], sha256: sha(png),
          html_sha256: sha(html), photo_jpeg_sha256: sha(photo) });
      }
    }
    await fs.writeFile(path.join(run, 'capture.json'), json({ browser: browser.version(), rows }), { flag: 'wx' });
  } finally {
    await browser.close();
  }
}

async function main() {
  const [command, run] = process.argv.slice(2);
  if (command === 'plan' && run) {
    await fs.writeFile(path.join(run, 'plan.json'), json(plan), { flag: 'wx' });
  } else if (command === 'capture' && run) {
    await capture(run);
  } else {
    throw new Error('Use plan or capture followed by a local run directory.');
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
