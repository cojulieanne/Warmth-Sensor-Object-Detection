const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const outputDir = __dirname;
  const source = path.join(outputDir, 'analysis_designs.html');
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1200 },
    deviceScaleFactor: 2,
  });
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto(`file://${source}`, { waitUntil: 'networkidle' });

  const frame = page.frames().find((item) => item !== page.mainFrame());
  if (!frame) throw new Error('Visualization iframe not found.');
  await frame.waitForSelector('#thermal-analysis-atlas svg');
  await frame.waitForTimeout(1000);

  const contentHeight = await frame.evaluate(() => document.documentElement.scrollHeight);
  await page.locator('iframe').evaluate((element, height) => {
    element.style.height = `${height}px`;
  }, contentHeight);
  await frame.waitForTimeout(300);

  const root = frame.locator('#thermal-analysis-atlas');
  await root.screenshot({
    path: path.join(outputDir, '00_all_representative_analysis_designs.png'),
  });

  const names = [
    '01_raw_input_comparison.png',
    '02_temperature_effect.png',
    '03_thermal_feature_vs_effusivity.png',
    '04_material_signal_vs_repetition_noise.png',
    '05_esn_feature_space_pca.png',
    '06_xgboost_shap_importance.png',
    '07_oof_prediction_and_residual_diagnosis.png',
  ];
  const panels = frame.locator('.atlas-panel');
  const count = await panels.count();
  if (count !== names.length) {
    throw new Error(`Expected ${names.length} panels, found ${count}.`);
  }
  for (let index = 0; index < count; index += 1) {
    await panels.nth(index).screenshot({ path: path.join(outputDir, names[index]) });
  }

  await browser.close();
})();
