// SPDX-License-Identifier: Apache-2.0
// Render local documentary interfaces around source-aligned generated edits.
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const fs = require('node:fs/promises');
const path = require('node:path');

const digest = data => createHash('sha256').update(data).digest('hex');
const labels = ['Photo review', 'Article preview', 'Listing dashboard', 'Asset catalog'];

function html(group, role, input) {
  const label = labels[group.slot % labels.length];
  const plain = role === 'template';
  const records = Array.from({ length: 11 }, (_, i) => `<tr><td>Inspection item ${i + 1}</td><td>Current media file</td><td>Entry ${135 + group.slot * 19 + i}</td></tr>`).join('');
  const content = plain ? `<main><h1>Current source records</h1><table>${records}</table></main>`
    : `<img aria-label="Photo preview" src="data:image/png;base64,${input.toString('base64')}" style="left:${group.x}px;top:${group.y}px;width:${group.w}px;height:${group.h}px">`;
  return '<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}body{margin:0;background:#e1e6e9;color:#263c4c;font:14px system-ui}'
    + 'header{height:54px;background:#263d4e;color:#fff;padding:14px 28px;display:flex;justify-content:space-between}aside{position:absolute;'
    + 'top:54px;bottom:0;left:0;width:142px;background:#f5f8fa;padding:20px 13px}p{margin:10px 0}h1{font-size:20px;margin:0 0 18px}'
    + 'main{margin:24px 24px 0 160px;background:#fff;padding:14px;min-height:calc(100vh - 105px)}table{border-collapse:collapse;width:100%}'
    + 'td{border-bottom:1px solid #bacbd2;padding:7px 10px}tr:nth-child(2n){background:#eaf0f4}img{position:absolute;display:block}</style>'
    + `<header><strong>${label}</strong><span>Library · Open preview</span></header><aside><strong>Media workspace</strong><p>Source files</p>`
    + '<p>Display options</p><p>Current review</p><p>View history</p><p>Export copy</p></aside>' + content;
}

async function main() {
  const [run] = process.argv.slice(2);
  const plan = JSON.parse(await fs.readFile(path.join(run, 'layout-plan.json'), 'utf8'));
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ deviceScaleFactor: 1, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await ctx.route('**/*', route => route.abort());
    const page = await ctx.newPage();
    const rows = [];
    await fs.mkdir(path.join(run, 'captures'));
    for (const group of plan) {
      await page.setViewportSize({ width: group.width, height: group.height });
      for (const role of ['template', 'photo', 'edit']) {
        const input = role === 'template' ? null : await fs.readFile(path.join(run, 'photo-fixtures', `${group.slot}-${role}.png`));
        const markup = html(group, role, input);
        await page.setContent(markup, { waitUntil: 'load' });
        await page.evaluate(async () => { await document.fonts.ready; await Promise.all(Array.from(document.images, item => item.decode())); });
        const image = await page.screenshot({ animations: 'disabled' });
        const name = `${group.slot}-${role}.png`;
        await fs.writeFile(path.join(run, 'captures', name), image, { flag: 'wx' });
        rows.push({ slot: group.slot, role, name, png_sha256: digest(image), html_sha256: digest(markup),
          photo_sha256: input ? digest(input) : null });
      }
    }
    await fs.writeFile(path.join(run, 'captures.json'), JSON.stringify({ chromium: browser.version(), rows }, null, 2) + '\n', { flag: 'wx' });
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
