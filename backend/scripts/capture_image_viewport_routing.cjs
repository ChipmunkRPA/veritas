// SPDX-License-Identifier: Apache-2.0
// Render independently named ordinary interfaces and native edited photo previews.
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const fs = require('node:fs/promises');
const path = require('node:path');
const hash = data => createHash('sha256').update(data).digest('hex');
const roles = ['template', 'photo', 'sharp', 'feather', 'ambiguous'];
const themes = [
  ['#f8f9f7', '#243642', '#dce6e7', 'Archive review', 'Source records'],
  ['#222938', '#edf3fa', '#3c4962', 'Inventory desk', 'Item details'],
  ['#f0e9de', '#312a28', '#dccbbd', 'Photo dispatch', 'Media catalog'],
  ['#f1f4fb', '#273154', '#dae1f8', 'Document index', 'Review history'],
];

function html(group, role, bytes) {
  const { x, y, w, h, layout, slot } = group;
  const [background, ink, trim, title, topic] = themes[layout];
  const position = `left:${x}px;top:${y}px;width:${w}px;height:${h}px`;
  const nav = [`left:12px;top:${y+9}px;width:130px`, `left:${x+w+16}px;top:${y+9}px;width:144px`,
    `left:22px;top:${y+h+12}px;width:340px`, 'left:21px;top:53px;width:680px'][layout];
  const rows = Array.from({length: 8}, (_, index) => `<p><b>${topic} ${index + 1}</b><span>Entry ${155 + slot*17 + index}</span></p>`).join('');
  const panel = role === 'template' ? `<section class="content table" style="${position}">${rows}</section>`
    : `<img class="content" aria-label="Source photo" style="${position}" src="data:image/png;base64,${bytes.toString('base64')}">`;
  const ambiguity = role === 'ambiguous' ? `<section class="obscured" style="left:${Math.max(18, x-39)}px;top:${y+64}px;width:196px;height:222px"><b>Display details</b><p>Alternate preview</p><p>Settings options</p><p>Open source</p></section>` : '';
  const line = ['Source files', 'Review details', 'Export history', 'Display settings', 'Related reports'];
  return `<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}body{margin:0;background:${background};color:${ink};font:15px system-ui}
  header{height:44px;background:${trim};display:flex;justify-content:space-between;padding:11px 20px}
  aside,.content,.obscured{position:absolute}aside{line-height:18px}aside p{margin:1px 0;font-size:14px;white-space:nowrap}
  aside p:nth-child(n+4){display:${layout === 3 ? 'none' : 'block'}}.table{padding:22px;background:${trim}}.table p{margin:0;padding:11px 4px;border-bottom:1px solid ${ink};display:flex;justify-content:space-between}
  .obscured{padding:16px;color:#142132;background:repeating-linear-gradient(45deg,#d8e7f6d9,#d8e7f6d9 2px,#f2f8ffc9 2px,#f2f8ffc9 4px);box-shadow:0 0 0 2px #63809b}
  </style><header><b>${title}</b><span>Local preview · ${240 + slot}</span></header><aside style="${nav}">${line.map(t=>`<p>${t}</p>`).join('')}</aside>${panel}${ambiguity}`;
}

async function main() {
  const [run] = process.argv.slice(2);
  const plan = JSON.parse(await fs.readFile(path.join(run, 'layout-plan.json'), 'utf8'));
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ deviceScaleFactor: 1, serviceWorkers: 'block', reducedMotion: 'reduce', viewport: {width:768,height:768} });
    await ctx.route('**/*', request => request.abort());
    const page = await ctx.newPage();
    await fs.mkdir(path.join(run, 'captures'));
    const rows = [];
    for (const group of plan) {
      for (const role of roles) {
        const asset = role === 'ambiguous' ? 'photo' : role;
        const bytes = role === 'template' ? null : await fs.readFile(path.join(run, 'photo-fixtures', `${group.slot}-${asset}.png`));
        const markup = html(group, role, bytes);
        await page.setContent(markup, { waitUntil: 'load' });
        await page.evaluate(async () => { await document.fonts.ready; await Promise.all(Array.from(document.images, x=>x.decode())); });
        const image = await page.screenshot({animations:'disabled'});
        await fs.writeFile(path.join(run,'captures',`${group.slot}-${role}.png`), image, {flag:'wx'});
        rows.push({slot:group.slot, role, html_sha256:hash(markup), png_sha256:hash(image), photo_sha256:bytes && hash(bytes)});
      }
    }
    await fs.writeFile(path.join(run,'captures.json'), JSON.stringify({chromium:browser.version(),rows}, null, 2)+'\n',{flag:'wx'});
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
