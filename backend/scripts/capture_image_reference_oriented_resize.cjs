// SPDX-License-Identifier: Apache-2.0
// Capture separate records in the same document interface as hard negatives.
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const fs = require('node:fs/promises');
const path = require('node:path');

const digest = value => createHash('sha256').update(value).digest('hex');
const kinds = ['Invoice preview', 'Service schedule', 'Library catalog', 'Meeting register'];
const tables = [
  ['Consultation', 'Licenses', 'Delivery', 'Training', 'Audit', 'Materials'],
  ['Briefing', 'Operations', 'Planning', 'Workshop', 'Review', 'Support'],
  ['Calendar', 'Travel', 'Preparation', 'Access', 'Archive', 'Notes'],
];

function markup(slot, variant) {
  const tint = ['#265068', '#73562b', '#634f73', '#24685b'][slot % 4];
  const kind = kinds[slot % kinds.length];
  const entries = Array.from({ length: 9 + variant }, (_, index) => {
    const name = tables[(slot + variant) % 3][(index + slot + 2 * variant) % 6];
    const amount = (index * 41 + slot * 37 + variant * 39) % 280;
    return `<tr><td>${name}</td><td>${String(index + 1 + variant).padStart(3, '0')}</td><td>${amount}.00</td></tr>`;
  }).join('');
  return '<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}'
    + 'body{margin:0;background:#d8e1e6;color:#203241;font:14px system-ui}header{height:48px;background:' + tint
    + ';color:white;padding:12px 24px}aside{position:absolute;top:48px;bottom:0;left:0;width:112px;padding:24px 10px;background:#e9eff2}'
    + 'main{margin:22px 28px 0 138px;padding:16px;background:white;min-height:466px}'
    + 'h1{font-size:25px;margin:0 0 12px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:' + (7 + variant)
    + 'px 8px;border-bottom:1px solid #abc2ca}tr:nth-child(2n){background:#eaf0f3}</style>'
    + `<header>${kind} · Export view</header><aside>Files<br><br>History<br><br>Print</aside>`
    + `<main><h1>${kind} ${2024 + variant}</h1><p>Document ${100 + slot * 5 + variant * 8} · Reviewed</p>`
    + '<table><tr><th>Entry</th><th>Reference</th><th>Total</th></tr>' + entries + '</table></main>';
}

async function main() {
  const [output] = process.argv.slice(2);
  if (!output) throw Error('Specify a new local capture folder.');
  await fs.mkdir(output, { mode: 0o700 });
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ deviceScaleFactor: 1, reducedMotion: 'reduce', serviceWorkers: 'block' });
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    const rows = [];
    for (let slot = 0; slot < 12; slot += 1) {
      const [width, height] = [[704, 568], [640, 612], [768, 534], [683, 596]][slot % 4];
      await page.setViewportSize({ width, height });
      for (let variant = 0; variant < 2; variant += 1) {
        const html = markup(slot, variant);
        await page.setContent(html, { waitUntil: 'load' });
        await page.evaluate(async () => { await document.fonts.ready; });
        const png = await page.screenshot({ animations: 'disabled' });
        const name = `${slot}-${variant}.png`;
        await fs.writeFile(path.join(output, name), png, { flag: 'wx' });
        rows.push({ slot, variant, name, size: [width, height], png_sha256: digest(png), html_sha256: digest(html) });
      }
    }
    await fs.writeFile(path.join(output, 'captures.json'), JSON.stringify({ browser: browser.version(), rows }, null, 2) + '\n', { flag: 'wx' });
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
