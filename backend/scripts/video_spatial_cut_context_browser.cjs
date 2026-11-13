// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('../../frontend/node_modules/playwright');

const [folder, base, keyfile = path.join(folder, 'browser-key.txt')] = process.argv.slice(2);

async function run() {
  const fixture = JSON.parse(fs.readFileSync(path.join(folder, 'browser-input.json'), 'utf8'));
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    await page.addInitScript(key => localStorage.setItem('veritas_api_key', key), fs.readFileSync(keyfile, 'utf8').trim());
    const anchor = '#media-evidence-' + fixture.packet_id;
    await page.goto(`${base}/analyses/${fixture.analysis_id}/media-evidence?view=plain${anchor}`);
    const card = page.locator(anchor);
    const context = card.locator('section').filter({ has: page.getByRole('heading', { name: 'Picture-cut context', exact: true }) });
    await context.waitFor({ timeout: 30000 });
    assert.match(await card.innerText(), /Picture-cut context for a face change/);
    assert.match(await card.innerText(), /zero weight in overall confidence|no weight in overall AI confidence/);
    const words = await context.innerText();
    assert.match(words, /2\.100 s–2\.467 s/);
    assert.match(words, /exclusive end/);
    assert.match(words, /cut between shots is one plausible explanation/i);
    for (const [role, index] of [['before', 62], ['during', 68], ['after', 74]]) {
      assert.match(words, new RegExp(`${role} picture-cut range · Frame ${index}`, 'i'));
    }
    const images = await context.getByRole('img').all();
    assert.equal(images.length, 6);
    const dimensions = [];
    for (const image of images) {
      await image.scrollIntoViewIfNeeded();
      dimensions.push(await image.evaluate(async element => {
        await element.decode();
        return [element.naturalWidth, element.naturalHeight];
      }));
    }
    assert.equal(dimensions.filter(([w, h]) => w === 1280 && h === 720).length, 3);
    assert(dimensions.every(([w, h]) => w > 0 && h > 0));
    await context.getByRole('button', { name: 'Go to frame 68 in playback' }).click();
    assert.match(await page.getByText(/Selected frame 68/).innerText(), /source 2\.267 s/);
    const navigator = page.getByRole('region', { name: 'Selected video moments' });
    const moment = navigator.getByRole('button', { name: `Inspect ${fixture.packet_id}.C2 at source 2.267 s` });
    assert.equal(await moment.getAttribute('aria-pressed'), 'true');
    assert.match(await navigator.innerText(), /whole picture|full source picture|surrounding picture/i);
    await context.screenshot({ path: path.join(folder, 'browser-evidence.png') });
    return { chrome: browser.version(), plain_evidence: true, verified_images: 6, playback_seek_frame: 68, dimensions };
  } finally {
    await browser.close();
  }
}

run().then(result => console.log(JSON.stringify(result))).catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
