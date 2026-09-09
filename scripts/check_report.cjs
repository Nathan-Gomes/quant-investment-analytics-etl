const { chromium } = require('playwright');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const browser = await chromium.launch({headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? {channel: process.env.PLAYWRIGHT_CHANNEL} : {})});
  try {
    for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
      const page = await browser.newPage({viewport: {width, height}});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(pathToFileURL(path.resolve('output/report.html')).href);
      await page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot .main-svg').length >= 7);
      const result = await page.evaluate(() => ({
        overflow: document.documentElement.scrollWidth > innerWidth + 1,
        charts: document.querySelectorAll('.js-plotly-plot').length,
        paths: document.querySelectorAll('.scatterlayer path.js-line').length,
      }));
      if (result.overflow || result.charts !== 7 || result.paths < 10 || errors.length) {
        throw new Error(JSON.stringify({name, result, errors}));
      }
      await page.screenshot({path: `output/${name}-report.png`, fullPage: true});
      console.log(name, result);
      await page.close();
    }
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
