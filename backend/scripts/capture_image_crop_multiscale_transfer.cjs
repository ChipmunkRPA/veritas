// SPDX-License-Identifier: Apache-2.0
// Render four authored interfaces with exact native photo pairs and ordinary controls.
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const fs = require('node:fs/promises');
const path = require('node:path');

const digest = body => createHash('sha256').update(body).digest('hex');
const themes = [
  ['#f8f9f7', '#233945', '#c8d9d8', '#eef2ee', 'Shipment overview', 'Asset, Owner, Status'],
  ['#1e242d', '#eef4ff', '#7854a6', '#2b3949', 'Camera inspection', 'Angle, Image, Recording'],
  ['#ede6db', '#24261e', '#e0ad71', '#fbf7ee', 'Regional dispatch', 'Photo, Editor, Desk'],
  ['#f0effb', '#27325a', '#99aade', '#e1e7fb', 'Evidence collection', 'Reference, Format, Date'],
];

function html(group, role, photo) {
  const [base, ink, accent, panel, name, labels] = themes[group.layout];
  const imageBox = `left:${group.x}px;top:${group.y}px;width:${group.w}px;height:${group.h}px`;
  const rows = Array.from({ length: 10 }, (_, i) => `<div class="row"><b>${labels.split(', ')[i % 3]} ${i + 1}</b><span>Entry ${(group.slot + 2) * (i + 3) + 150}</span><span>${i % 3 ? 'Open' : 'Filed'}</span></div>`).join('');
  const frames = role === 'template' ? `<section class="photo" style="${imageBox}">${rows}</section>`
    : `<img class="photo" alt="Source photo" src="data:image/png;base64,${photo.toString('base64')}" style="${imageBox}">`;
  const details = group.layout === 1 ? 'right:24px;bottom:24px;width:292px' : group.layout === 2
    ? 'right:18px;top:154px;width:178px' : group.layout === 3 ? 'left:12px;top:108px;width:152px' : 'left:14px;top:88px;width:126px';
  const bar = group.layout === 3 ? 'bottom:0;left:0;right:0' : 'top:0;left:0;right:0';
  return `<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}body{margin:0;background:${base};color:${ink};font:14px system-ui}
    header{position:absolute;${bar};height:52px;background:${accent};padding:13px 23px;display:flex;justify-content:space-between}
    section.photo{padding:15px;background:${panel};overflow:hidden;border-radius:${group.layout * 3}px}.photo{position:absolute;display:block}
    .row{display:grid;grid-template-columns:2fr 1fr 1fr;gap:8px;padding:8px 4px;border-bottom:1px solid ${accent}}
    aside{position:absolute;${details};padding:13px;background:${panel};border:1px solid ${accent}}aside p{margin:9px 0}</style>
    <header><b>${name}</b><span>Archive &nbsp; / &nbsp; Local preview</span></header><aside><b>Source details</b><p>Reference files</p><p>Review history</p><p>Export settings</p></aside>${frames}`;
}

async function main() {
  const [run] = process.argv.slice(2);
  const plan = JSON.parse(await fs.readFile(path.join(run, 'layout-plan.json'), 'utf8'));
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ deviceScaleFactor: 1, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await ctx.route('**/*', request => request.abort());
    const page = await ctx.newPage();
    const rows = [];
    await fs.mkdir(path.join(run, 'captures'));
    for (const group of plan) {
      await page.setViewportSize({ width: group.width, height: group.height });
      for (const role of ['template', 'photo', 'edit']) {
        const photo = role === 'template' ? null : await fs.readFile(path.join(run, 'photo-fixtures', `${group.slot}-${role}.png`));
        const markup = html(group, role, photo);
        await page.setContent(markup, { waitUntil: 'load' });
        await page.evaluate(async () => { await document.fonts.ready; await Promise.all(Array.from(document.images, item => item.decode())); });
        const png = await page.screenshot({ animations: 'disabled' });
        await fs.writeFile(path.join(run, 'captures', `${group.slot}-${role}.png`), png, { flag: 'wx' });
        rows.push({ slot: group.slot, role, png_sha256: digest(png), html_sha256: digest(markup),
          photo_sha256: photo ? digest(photo) : null });
      }
    }
    await fs.writeFile(path.join(run, 'captures.json'), JSON.stringify({ chromium: browser.version(), rows }, null, 2) + '\n', { flag: 'wx' });
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
