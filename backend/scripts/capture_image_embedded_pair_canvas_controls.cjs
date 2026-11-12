// SPDX-License-Identifier: Apache-2.0
// Capture ordinary authored interfaces with no requests or user content.
const { createHash } = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const { createRequire } = require('node:module');

const sizes = [[768, 768], [1024, 768], [1280, 720], [1366, 768], [1920, 1080]];
const layouts = ['catalog', 'contact', 'news', 'editor', 'directory', 'gallery', 'calendar', 'preview'];
const labels = ['Survey reports', 'Daily overview', 'Print guide', 'Design collection', 'Weather station',
  'Street maps', 'Historical archive', 'Product cards', 'Recent files'];
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');

const css = `
*{box-sizing:border-box}body{margin:0;color:#1c2734;background:#f1f4f6;font:15px Arial,sans-serif}
header{height:68px;display:flex;justify-content:space-between;align-items:center;padding:0 32px;
background:#202c3a;color:#fff}main{padding:27px 31px}.with-aside{margin-left:177px}aside{position:absolute;
left:0;top:68px;bottom:0;width:177px;background:#dde5ec;padding:18px}.menu{display:block;padding:13px 0}
h1{font-size:27px;margin:0 0 19px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
gap:17px}.card{min-width:0;background:#fff;border:1px solid #bac6d2;padding:13px}.art{height:154px;
border:1px solid #c0cdd6;position:relative;overflow:hidden}.art b{position:absolute;top:20px;left:14px;
background:#f1f4fbdd;padding:8px}.line{height:31px;padding:7px;border-bottom:1px solid #cad2dc}
.editor{display:flex;gap:22px}.pane{position:relative;background:#e9edf2;width:238px;min-width:180px}
.pane img{width:100%;height:auto;display:block;image-rendering:auto}.pane small{position:absolute;bottom:12px;
right:12px;background:#f4f7f0dc;color:#293642;padding:9px}.note{font-size:17px;padding:15px 0}
.news .card:first-child{grid-column:span 2}.calendar .card{min-height:167px}.gallery img{width:100%}
`;

function chart(i, variation) {
  const bars = Array.from({length: 12}, (_, j) => {
    const y = 76 + (i * 37 + j * 19 + variation * 11) % 69;
    return `<rect x="${8+j*16}" y="${y}" width="12" height="${155-y}" fill="${j%2?'#147a8a':'#3a4f9a'}"/>`;
  }).join('');
  const dots = Array.from({length: 9}, (_, j) => `<circle cx="${16+j*22}" cy="${13+(j*13+i*7)%58}"
  r="${4+(j+i)%8}" fill="${j%2?'#df754f':'#e4b453'}"/>`).join('');
  return `<svg viewBox="0 0 216 160" width="100%" height="100%" aria-hidden="true">
    <rect width="216" height="160" fill="#e6eef6"/>${dots}${bars}</svg>`;
}

function authoredArt(i) {
  const circles = Array.from({length: 35}, (_, j) => `<circle cx="${(j*53+i*17)%280}" cy="${(j*47+i*53)%249}"
  r="${5+(j*13)%35}" fill="hsl(${(j*21+i*51)%360} 41% ${40+(j*11)%40}%)"/>`).join('');
  return 'data:image/svg+xml;base64,' + Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 249">
    <rect width="280" height="249" fill="#d9e5f0"/>${circles}</svg>`).toString('base64');
}

function page(layout, id) {
  const head = `<header><strong>Public reference library</strong><span>Collection / ${layout} / Preferences</span></header>`;
  const menu = `<aside>${labels.slice(0,6).map(label => `<span class="menu">${label}</span>`).join('')}</aside>`;
  const common = [1, 3, 4, 7].includes(id);
  let body = '';
  if (id === 3 || id === 7) {
    const images = id === 3 ? [authoredArt(id), authoredArt(id + 1)] : [authoredArt(id), authoredArt(id), authoredArt(id)];
    body = `<div class="editor">${images.map((asset, i) => `<div class="pane"><img alt="" src="${asset}">
    ${i === 2 ? '<small>Draft settings</small>' : ''}</div>`).join('')}</div>
    <p class="note">${id === 3 ? 'Separate diagrams and display panels' : 'Multiple previews of the same layout'}</p>`;
  } else if (id === 5) {
    body = `<div class="grid">${labels.slice(0,6).map((label, i) => `<article class="card gallery">
      <img alt="" src="${authoredArt(i)}"><h3>${label}</h3><span>Browse group</span></article>`).join('')}</div>`;
  } else if (id === 1 || id === 4) {
    body = Array.from({length: 18}, (_, i) => `<div class="line">${String(i+1).padStart(2,'0')} ·
      ${labels[(i+id)%labels.length]} · Group ${(i*7)%21+1} · ${i%2?'Available':'Published'}</div>`).join('');
  } else {
    body = `<div class="grid ${layout}">${labels.map((label,i) => `<article class="card">
      <div class="art">${chart(i, id)}${i%3===1?'<b>Published summary</b>':''}</div>
      <h3>${label}</h3><span>Preview ${i+1}</span></article>`).join('')}</div>`;
  }
  return `<!doctype html><meta charset="utf-8"><style>${css}</style>${head}${common?menu:''}
    <main class="${common?'with-aside':''}"><h1>${layout[0].toUpperCase()+layout.slice(1)}</h1>${body}</main>`;
}

async function main() {
  const folder = process.argv[2];
  const loadFrontend = createRequire(path.resolve('frontend/package.json'));
  const { chromium } = loadFrontend('playwright');
  const browser = await chromium.launch({headless: true, ...(process.env.CHROMIUM_EXECUTABLE
    ? {executablePath: process.env.CHROMIUM_EXECUTABLE} : {})});
  const rows = [];
  try {
    const context = await browser.newContext({deviceScaleFactor: 1, locale:'en-US', reducedMotion:'reduce'});
    await context.route('**/*', request => request.abort());
    const tab = await context.newPage();
    for (const [width, height] of sizes) {
      await tab.setViewportSize({width, height});
      for (const [slot, name] of layouts.entries()) {
        const html = page(name, slot);
        await tab.setContent(html, {waitUntil: 'load'});
        await tab.evaluate(() => document.fonts.ready);
        const filename = `${width}x${height}-${name}.png`;
        const png = await tab.screenshot({animations:'disabled', scale:'css', path:path.join(folder, filename)});
        rows.push({filename, size:[width,height], role:name, html_sha256:hash(html), png_sha256:hash(png)});
      }
    }
    await fs.writeFile(path.join(folder,'screenshots.json'), JSON.stringify({browser:browser.version(),rows},null,2)+'\n',
      {flag:'wx'});
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode=1; });
