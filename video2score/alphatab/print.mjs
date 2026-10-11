// 用无界面 Chrome 把排好的网页打印成 PDF（video2score/engrave.py 调用）。
// 用法：node print.mjs <输入.html> <输出.pdf> <浏览器可执行文件> <配置目录>
// 浏览器通常是包里自带的 chrome-headless-shell；从仓库运行时也可能是本机的 Chrome。
// 纸张大小、页边距按网页里的 @page（A4、边距 0）来，页码和出处网页里已经印好了，不要浏览器自己的页眉页脚。
// 配置目录由调用方给（放在输出 PDF 旁边的临时文件夹里），不往系统临时目录写。
import path from 'path';
import { pathToFileURL } from 'url';
import puppeteer from 'puppeteer-core';

const [input, output, executablePath, userDataDir] = process.argv.slice(2);
// chrome-headless-shell 只有老的无界面模式（puppeteer 叫 'shell'）；完整的 Chrome 用新的无界面模式。
const shell = path.basename(executablePath).toLowerCase().startsWith('chrome-headless-shell');
const browser = await puppeteer.launch({
  executablePath,
  headless: shell ? 'shell' : true,
  userDataDir,
  args: ['--no-first-run', '--no-default-browser-check', '--disable-extensions', '--disable-sync', '--disable-background-networking', '--disable-gpu'],
});
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(input).href, { waitUntil: 'load', timeout: 120000 });
  // 等嵌在网页里的 alphaTab 字体加载完，再按页宽缩一遍过长的标题（网页里的脚本在 load 前已经跑过）。
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({ path: output, preferCSSPageSize: true, printBackground: true, displayHeaderFooter: false, timeout: 120000 });
} finally {
  await browser.close();
}
