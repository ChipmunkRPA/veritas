// SPDX-License-Identifier: Apache-2.0
// Freeze local interface panels with opaque, repeated and translucent interiors.

const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const fs = require('node:fs/promises');
const path = require('node:path');

const types = ['opaque', 'alpha', 'periodic'];
const variants = [[704,544],[640,608],[768,512],[512,704]];
const plan = { schema: 'chipmunk.boundary-context-captures/1', groups: Array.from({length:12}, (_, slot) => ({
  slot, size: variants[slot % variants.length], views: types.map((type, index) => ({
    type, caption: (slot + index) % 2 === 0, width: 66 + (slot * 7) % 28, height: 62 + (slot * 11) % 31,
    x: 134 + (slot * 19) % 138, y: 121 + (slot * 23) % 168,
  })),
})) };
const digest = data => createHash('sha256').update(data).digest('hex');
const json = data => JSON.stringify(data, null, 2) + '\n';

function markup(group, view, photo) {
  const palette = [['#faf1d4','#e7dcba'],['#e2edf9','#c3d3e5'],['#f2e4ef','#d4bed0']][group.slot % 3];
  const pattern = [
    'repeating-linear-gradient(35deg,C1 0px,C1 2.5px,C2 2.5px,C2 5px)',
    'repeating-conic-gradient(C1 0% 25%, C2 0% 50%) 0 0/9px 9px',
    'repeating-linear-gradient(72deg,C1 0px,C1 4px,C2 4px,C2 8px)',
  ][group.slot % 3].replaceAll('C1',palette[0]).replaceAll('C2',palette[1]);
  const background = {
    opaque: palette[0],
    alpha: 'rgba(238,247,250,' + [.34,.51,.67][group.slot % 3] + ')',
    periodic: pattern,
  }[view.type];
  const caption = view.caption ? '<strong>Preview</strong><small>Open item</small>' : '';
  return '<!doctype html><meta charset="utf-8"><style>'
    + '*{box-sizing:border-box}body{margin:0;font:14px Arial;color:#27313a;background:#28384b}'
    + 'header,footer{height:38px;color:#eef2f7;padding:10px 24px;display:flex;justify-content:space-between}'
    + 'main{position:relative;height:calc(100vh - 76px);background-position:center;background-size:cover;'
    + 'background-image:url(data:image/jpeg;base64,' + photo.toString('base64') + ')}'
    + 'aside{position:absolute;left:' + view.x + 'px;top:' + view.y + 'px;width:' + view.width + 'px;height:' + view.height
    + 'px;background:' + background + ';border:1px solid #dae2ee;box-shadow:0 2px 4px #18243840;padding:8px}'
    + 'small{display:block;margin-top:4px}</style><header><strong>Image viewer</strong><span>Open · Review</span></header>'
    + '<main><aside aria-label="Image panel">' + caption + '</aside></main><footer><span>Preview</span><span>Fit to window</span></footer>';
}

async function capture(run) {
  if (await fs.readFile(path.join(run, 'capture-plan.json'), 'utf8') !== json(plan)) throw Error('Capture plan changed.');
  const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ deviceScaleFactor:1, reducedMotion:'reduce', serviceWorkers:'block' });
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    const rows = [];
    await fs.mkdir(path.join(run,'captures'));
    for (const group of plan.groups) {
      const [width,height] = group.size;
      const photo = await fs.readFile(path.join(run, 'screen-photos', group.slot + '.jpg'));
      await page.setViewportSize({width,height});
      for (const view of group.views) {
        const html = markup(group, view, photo);
        await page.setContent(html,{waitUntil:'load'});
        await page.evaluate(async () => {
          await document.fonts.ready;
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        });
        const bytes = await page.screenshot({animations:'disabled'});
        const filename = group.slot + '-' + view.type + '.png';
        await fs.writeFile(path.join(run,'captures',filename),bytes,{flag:'wx'});
        rows.push({slot:group.slot,type:view.type,caption:view.caption,filename,size:group.size,
          rect:[view.x,view.y+38,view.x+view.width,view.y+view.height+38],png_sha256:digest(bytes),
          html_sha256:digest(html),photo_sha256:digest(photo)});
      }
    }
    await fs.writeFile(path.join(run,'captures.json'),json({browser:browser.version(),rows}),{flag:'wx'});
  } finally { await browser.close(); }
}

async function main() {
  const [command, run] = process.argv.slice(2);
  if (command === 'plan' && run) await fs.writeFile(path.join(run, 'capture-plan.json'),json(plan),{flag:'wx'});
  else if (command === 'capture' && run) await capture(run);
  else throw Error('Use plan or capture followed by a local run directory.');
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
