// SPDX-License-Identifier: Apache-2.0
// Capture ordinary photo layouts with two strong rectangular regions.

const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const fs = require('node:fs/promises');
const path = require('node:path');

const sizes = [[704, 544], [640, 608], [768, 512], [512, 704]];
const digest = value => createHash('sha256').update(value).digest('hex');
const json = value => JSON.stringify(value, null, 2) + '\n';
const plan = { schema: 'chipmunk.image-boundary-multiple-captures/1', groups: Array.from({ length: 16 }, (_, slot) => {
  const [width, height] = sizes[slot % sizes.length];
  const fraction = [1 / 6, 1 / 4, 1 / 3];
  const sizesFor = index => [Math.round(width * fraction[index]), Math.round(height * fraction[(index + slot) % 3])];
  const [leftWidth, leftHeight] = sizesFor(slot % 3);
  const [rightWidth, rightHeight] = sizesFor((slot + Math.floor(slot / 4)) % 3);
  return { slot, width, height, type: ['reading', 'directory', 'comparison', 'preview'][slot % 4], panels: [
    [52 + 3 * slot, 109 + 2 * slot, leftWidth, leftHeight],
    [width - 50 - rightWidth - 2 * slot, height - 61 - rightHeight - slot, rightWidth, rightHeight],
  ] };
}) };

function markup(group, variant, photo) {
  const [paper, ink, alternate] = [
    ['#f7f1db', '#30392d', '#d1e9ed'], ['#edf3f9', '#1a2d42', '#fff0cf'],
    ['#121a23', '#f8fafc', '#263e5f'], ['#fff7ef', '#292839', '#e0e7fb'],
  ][group.slot % 4];
  const photoUrl = 'url(data:image/jpeg;base64,' + photo.toString('base64') + ')';
  const labels = { reading: ['Summary', 'Related reading'], directory: ['Collection', 'Saved items'],
    comparison: ['First view', 'Second view'], preview: ['Print preview', 'Comments'] }[group.type];
  const cards = group.panels.map(([x, y, w, h], index) => {
    const body = variant !== 'overlay'
      ? `<div class="tile ${index ? 'other' : ''}"></div><small>${labels[index]}</small>`
      : `<h2>${labels[index]}</h2><p>Open the collection to view details.</p><ul><li>Overview</li><li>Current selection</li></ul>`;
    return `<article style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;background:${index ? alternate : paper}">${body}</article>`;
  }).join('');
  const extra = variant === 'photos' ? 'article{border:none;padding:0}.tile{height:100%}small{display:none}' : '';
  return '<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}'
    + 'body{margin:0;color:' + ink + ';font:14px system-ui;background:' + paper + '}'
    + 'main{position:fixed;inset:0;background-position:center;background-size:cover;'
    + 'background-image:' + (variant === 'overlay' ? photoUrl : 'linear-gradient(145deg,#d0dbe3,#e1e3d0)') + '}'
    + 'header{position:relative;background:' + paper + ';height:43px;padding:11px 24px;display:flex;justify-content:space-between}'
    + 'article{position:absolute;border:1px solid ' + ink + ';overflow:hidden;padding:8px}'
    + 'p,h2,ul{margin:0 0 5px}h2{font-size:14px}.tile{height:calc(100% - 20px);background-image:' + photoUrl
    + ';background-position:30% 25%;background-size:110% auto}.other{background-position:85% 65%;background-size:165% auto}'
    + extra + '</style>'
    + '<main>' + cards + '</main><header><strong>Workspace</strong><span>View · Export</span></header>';
}

async function capture(run) {
  if (await fs.readFile(path.join(run, 'capture-plan.json'), 'utf8') !== json(plan)) throw Error('Capture plan differs.');
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ deviceScaleFactor: 1, reducedMotion: 'reduce', serviceWorkers: 'block' });
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    const rows = [];
    await fs.mkdir(path.join(run, 'captures'));
    for (const group of plan.groups) {
      const photo = await fs.readFile(path.join(run, 'screen-photos', `${group.slot}.jpg`));
      await page.setViewportSize({ width: group.width, height: group.height });
      for (const variant of ['overlay', 'tiles', 'photos']) {
        const html = markup(group, variant, photo);
        await page.setContent(html, { waitUntil: 'load' });
        await page.evaluate(async () => { await document.fonts.ready; });
        const png = await page.screenshot({ animations: 'disabled' });
        const filename = `${group.slot}-${variant}.png`;
        await fs.writeFile(path.join(run, 'captures', filename), png, { flag: 'wx' });
        rows.push({ slot: group.slot, variant, filename, html_sha256: digest(html),
          source_sha256: digest(photo), png_sha256: digest(png) });
      }
    }
    await fs.writeFile(path.join(run, 'captures.json'), json({ chromium: browser.version(), rows }), { flag: 'wx' });
  } finally {
    await browser.close();
  }
}

async function main() {
  const [stage, run] = process.argv.slice(2);
  if (stage === 'plan' && run) await fs.writeFile(path.join(run, 'capture-plan.json'), json(plan), { flag: 'wx' });
  else if (stage === 'capture' && run) await capture(run);
  else throw Error('Use plan or capture followed by a local run directory.');
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
