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
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(key => localStorage.setItem('veritas_api_key', key), fs.readFileSync(keyfile, 'utf8').trim());
    await page.goto(`${base}/analyses/${fixture.analysis_id}/media-evidence?view=plain`);
    const navigator = page.getByRole('region', { name: 'Selected video moments' });
    const original = page.getByLabel('Original video evidence playback');
    const coverage = page.getByRole('region', { name: 'Overlapping face-change windows' });
    await coverage.waitFor({ timeout: 30000 });
    const coverageWords = await coverage.innerText();
    for (const [start, end] of [['0.000', '4.000'], ['2.000', '6.000'], ['4.000', '8.000']]) {
      assert(coverageWords.includes(`Window ${start}–${end} s`));
    }
    assert.match(coverageWords, /zero AI confidence weight/);
    const receipts = [];
    for (const entry of fixture.packets) {
      const card = page.locator(`#media-evidence-${entry.packet_id}`);
      await card.waitFor({ timeout: 30000 });
      const words = await card.innerText();
      assert.match(words, /Brief face-change range to review/);
      assert.match(words, /with an exclusive end/);
      assert.match(words, /no weight in overall AI confidence/);
      assert(words.includes(entry.times_s.map(value => `${value.toFixed(3)} s`).join('–')));
      for (const [position, role] of ['before', 'during', 'after'].entries()) {
        assert.match(words, new RegExp(`${role} flagged range · Frame ${entry.indices[position]}`, 'i'));
        if (entry.anchor_times_s) {
          assert(words.includes(`Source ${entry.anchor_times_s[position].toFixed(3)} s`));
        }
      }
      const dimensions = [];
      for (const frame of entry.frames) {
        for (const name of ['exact original video frame', 'source face crop for context']) {
          const image = card.getByRole('img', { name: `${frame}: ${name}` });
          await image.scrollIntoViewIfNeeded();
          dimensions.push(await image.evaluate(async element => {
            await element.decode();
            return [element.naturalWidth, element.naturalHeight];
          }));
        }
      }
      assert.equal(dimensions.filter(([w, h]) => w === 1280 && h === 720).length, 3);
      assert(dimensions.every(([w, h]) => w > 0 && h > 0));
      const index = entry.indices[1];
      const second = (entry.anchor_times_s?.[1] ?? index / 30).toFixed(3);
      await card.getByRole('button', { name: `Go to frame ${index} in playback` }).click();
      const selected = page.getByText(`Selected frame ${index}`, { exact: false });
      assert((await selected.innerText()).includes(`source ${second} s`));
      const moment = navigator.getByRole('button', { name: `Inspect ${entry.frames[1]} at source ${second} s` });
      assert.equal(await moment.getAttribute('aria-pressed'), 'true');
      if (fixture.verify_browser_seek) {
        await original.waitFor({ timeout: 15000 });
        await page.waitForFunction(time => {
          const video = document.querySelector('video[aria-label="Original video evidence playback"]');
          return video?.readyState >= 1 && Math.abs(video.currentTime - time) < .05;
        }, Number(second), { timeout: 15000 });
      }
      receipts.push({ source_times_s: entry.times_s, images: dimensions, playback_selected_frame: index });
    }
    assert.equal(receipts.length, 2);
    const findings = page.getByTestId('findings');
    for (const entry of fixture.packets) {
      const link = findings.getByTestId('finding-source-support').filter({ hasText: `${entry.packet_id} · Brief face-change range` });
      assert.match(await link.first().getAttribute('href'), new RegExp(`#media-evidence-${entry.packet_id}$`));
      assert((await link.first().innerText()).includes(entry.times_s.map(value => `${value.toFixed(3)} s`).join('–')));
    }
    assert.deepEqual(errors, []);
    return { chrome: browser.version(), plain_evidence: true, coverage_windows: 3, exact_finding_links: 2,
      ...(fixture.verify_browser_seek ? { actual_original_playback_seeks_verified: 2 } : {}), windows: receipts };
  } finally {
    await browser.close();
  }
}

run().then(result => console.log(JSON.stringify(result))).catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
