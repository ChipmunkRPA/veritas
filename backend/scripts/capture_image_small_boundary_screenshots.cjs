// SPDX-License-Identifier: Apache-2.0
// Capture a predeclared set of published browser layouts and local photo composites.

const { createHash } = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const { createRequire } = require('node:module');

const bootstrap = 'https://getbootstrap.com/docs/5.3/examples/';
const govuk = 'https://design-system.service.gov.uk/components/';
const uswds = 'https://designsystem.digital.gov/components/';
const lists = [
  ['bootstrap', bootstrap, 'https://getbootstrap.com/docs/5.3/about/license/', [
    'album', 'pricing', 'checkout', 'product', 'cover', 'carousel', 'blog', 'dashboard', 'sign-in',
    'sticky-footer', 'jumbotron', 'grid', 'navbars', 'sidebars', 'features', 'list-groups',
  ]],
  ['govuk', govuk, 'https://design-system.service.gov.uk/components/', [
    'accordion', 'checkboxes', 'date-input', 'details', 'error-summary', 'fieldset', 'file-upload', 'inset-text',
    'notification-banner', 'pagination', 'panel', 'radios', 'select', 'summary-list', 'table', 'task-list',
  ]],
  ['uswds', uswds, 'https://designsystem.digital.gov/about/contribute/#licenses-and-attribution', [
    'accordion', 'alert', 'banner', 'breadcrumb', 'button-group', 'character-count', 'checkbox', 'combo-box',
    'date-picker', 'file-input', 'form', 'in-page-navigation', 'process-list', 'step-indicator', 'summary-box', 'table',
  ]],
];
const layouts = ['news', 'editor', 'listing', 'gallery'];
const sourceGroups = [
  ...lists.flatMap(([family, origin, terms, slugs]) => slugs.map(slug => ({
    id: `${family}-${slug}`, family, terms, source: `${origin}${slug}/`,
  }))),
  ...Array.from({ length: 12 }, (_, slot) => ({
    id: `composite-${String(slot).padStart(2, '0')}`, family: 'composite', slot,
    layout: layouts[slot % layouts.length],
  })),
];
const variants = [{ width: 704, height: 544, scroll: 0 }, { width: 768, height: 512, scroll: 220 }];
const plan = { schema: 'chipmunk.small-boundary-screenshot-roster/1', groups: sourceGroups, variants };
const reserve = { ...plan, groups: [
  ...['headers', 'footers', 'dropdowns', 'modals', 'badges', 'breadcrumbs', 'buttons', 'jumbotrons', 'navbar-static',
    'navbar-fixed', 'navbar-bottom', 'offcanvas-navbar', 'cheatsheet', 'heroes'].map(slug => ({
    id: `bootstrap-${slug}`, family: 'bootstrap', terms: lists[0][2], source: `${bootstrap}${slug}/`,
  })),
  ...lists[1][3].map(slug => ({ id: `govuk-example-${slug}`, family: 'govuk', lineage: `govuk-${slug}`,
    terms: lists[1][2], source: `${govuk}${slug}/default/index.html`, focused: true })),
  ...lists[2][3].map(slug => ({ id: `uswds-example-${slug}`, family: 'uswds', lineage: `uswds-${slug}`,
    terms: lists[2][2], source: `${uswds}${slug}/`, focus: `#${slug}-preview`, focused: true })),
] };

const compositeCSS = `
*{box-sizing:border-box}body{margin:0;color:#17212d;font:15px Arial,sans-serif}header{height:48px;background:#203347;
color:white;padding:15px 24px;display:flex;justify-content:space-between}main{position:relative;height:calc(100vh - 48px);
background-position:center;background-size:cover}main.editor{height:calc(100vh - 76px);margin-top:28px}
.panel{background:#faf8ef;color:#27313a;position:absolute;padding:10px;border:1px solid #e4e9ee;box-shadow:0 4px 12px #1423334d}
.headline{position:absolute;left:20px;bottom:24px;padding:18px;background:#192834e8;color:white;font-size:28px;font-weight:bold}
.thumb{position:absolute;right:28px;bottom:28px;width:86px;height:66px;border:2px solid white;object-fit:cover}
.toolbar{background:#e3e8ef;color:#27313a;padding:5px 17px;position:absolute;top:-28px;width:100%;height:28px}
.panel span,.panel b{display:block;margin-bottom:4px}.panel small{display:block;margin-top:4px}
`;

function compositePage(group, bytes) {
  const encoded = bytes.toString('base64');
  const photo = `data:image/jpeg;base64,${encoded}`;
  const x = 144 + 17 * (group.slot % 5);
  const y = 104 + 21 * (group.slot % 6);
  const w = [66, 84, 96, 78][group.slot % 4];
  const h = [74, 62, 76, 91][group.slot % 4];
  const labels = {
    news: ['Local Briefing', 'Latest', '12 updates', 'Open story'],
    editor: ['Studio', 'Adjust', 'Exposure', 'Auto levels'],
    listing: ['Field Guide', 'Visit', 'Open daily', 'Read more'],
    gallery: ['Image Archive', 'Image 06', 'Full size', 'View details'],
  }[group.layout];
  const panel = `<article class="panel" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px">
  <b>${labels[1]}</b><small>${labels[2]}</small></article>`;
  const extra = {
    news: `<div class="headline">${labels[3]}</div>`,
    editor: `<div class="toolbar">Open · Crop · Adjust · Export</div>`,
    listing: `<div class="headline">${labels[3]}</div><div class="panel" style="right:26px;top:54px">Map</div>`,
    gallery: `<img class="thumb" alt="" src="${photo}">`,
  }[group.layout];
  return `<!doctype html><meta charset="utf-8"><style>${compositeCSS}</style>
  <header><strong>${labels[0]}</strong><span>Browse · Overview</span></header>
  <main class="${group.layout}" style="background-image:url('${photo}')">${panel}${extra}</main>`;
}

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = data => `${JSON.stringify(data, null, 2)}\n`;
const viewport = ({ width, height }) => ({ width, height });

async function capture(directory, currentPlan) {
  const expected = await fs.readFile(path.join(directory, 'screenshot-plan.json'), 'utf8');
  if (expected !== json(currentPlan)) throw new Error('The screenshot roster changed after freeze.');
  const loadFrontend = createRequire(path.resolve('frontend/package.json'));
  const { chromium } = loadFrontend('playwright');
  const browser = await chromium.launch({ headless: true });
  const output = path.join(directory, 'captures');
  await fs.mkdir(output);
  const collected = [];
  try {
    const context = await browser.newContext({ deviceScaleFactor: 1, locale: 'en-US', colorScheme: 'light',
      reducedMotion: 'reduce', serviceWorkers: 'block' });
    await context.route('**/*', route => {
      const request = route.request();
      const url = new URL(request.url());
      const permitted = ['getbootstrap.com', 'design-system.service.gov.uk', 'designsystem.digital.gov'].includes(url.hostname)
        || (url.hostname === 'cdn.jsdelivr.net' && /^\/npm\/(bootstrap|@docsearch)@/.test(url.pathname));
      const kind = request.resourceType();
      return request.method() === 'GET' && permitted && !['xhr', 'fetch', 'websocket'].includes(kind)
        ? route.continue() : route.abort();
    });
    async function queue(groups) {
      const page = await context.newPage();
      try {
        for (const group of groups) {
          const result = { id: group.id, family: group.family, views: [] };
          try {
            await page.setViewportSize(viewport(variants[0]));
            if (group.source) {
              const response = await page.goto(group.source, { waitUntil: 'load', timeout: 18000 });
              if (!response?.ok() || new URL(page.url()).origin !== new URL(group.source).origin) {
                throw new Error(`Capture failed with status ${response?.status() ?? 'unavailable'}.`);
              }
              const body = await page.locator('body').innerText({ timeout: 6000 });
              if (body.length < (group.focused ? 18 : 120) || !await page.title()) {
                throw new Error('The page did not render enough content.');
              }
              result.html_sha256 = hash(await response.body());
              result.final_url = page.url();
            } else {
              await page.goto('about:blank');
              const photo = await fs.readFile(path.join(directory, 'composite-photos', `${group.slot}.jpg`));
              const html = compositePage(group, photo);
              await page.setContent(html, { waitUntil: 'load' });
              result.html_sha256 = hash(html);
            }
            for (const [view, variant] of variants.entries()) {
              await page.setViewportSize(viewport(variant));
              const scroll = await page.evaluate(async ({ y, focus }) => {
                await document.fonts.ready;
                const element = focus ? document.querySelector(focus) : null;
                if (focus && !element) throw new Error('The focused example section was unavailable.');
                window.scrollTo(0, y + (element ? element.getBoundingClientRect().top + window.scrollY : 0));
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                return window.scrollY;
              }, { y: group.source ? (group.focused ? variant.scroll / 4 : variant.scroll) : 0, focus: group.focus });
              const png = await page.screenshot({ animations: 'disabled', caret: 'hide', timeout: 10000 });
              const filename = `${group.id}-${view}.png`;
              await fs.writeFile(path.join(output, filename), png, { flag: 'wx' });
              result.views.push({ view, filename, scroll, width: variant.width, height: variant.height, sha256: hash(png) });
            }
          } catch (error) {
            result.error = String(error.message).split('\n')[0];
          }
          collected.push(result);
        }
      } finally {
        await page.close();
      }
    }
    await Promise.all(['bootstrap', 'govuk', 'uswds', 'composite'].map(family => queue(currentPlan.groups.filter(row => row.family === family))));
    const results = { schema: currentPlan.schema, browser: browser.version(), groups: collected.sort((a, b) => a.id.localeCompare(b.id)) };
    await fs.writeFile(path.join(directory, 'screenshot-captures.json'), json(results), { flag: 'wx' });
    console.log(`${results.groups.reduce((total, group) => total + group.views.length, 0)} captured screenshot views`);
  } finally {
    await browser.close();
  }
}

async function main() {
  const [command, directory] = process.argv.slice(2);
  const current = command?.startsWith('reserve-') ? reserve : plan;
  if (['plan', 'reserve-plan'].includes(command) && directory) {
    await fs.writeFile(path.join(directory, 'screenshot-plan.json'), json(current), { flag: 'wx' });
  } else if (['capture', 'reserve-capture'].includes(command) && directory) {
    await capture(directory, current);
  } else {
    throw new Error('Use plan, capture, reserve-plan or reserve-capture followed by a local run directory.');
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
