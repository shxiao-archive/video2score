// 用 alphaTab 把一个 Guitar Pro 文件排成五线谱 + TAB（Guitar Pro 那种版式），输出每一块（标题、每一行谱）的 SVG。
// 用法：node render.mjs <输入.gp> <输出.json> [宽度像素] [每行小节数] [字宽.json]
// 输出 {width, height, copyright, chunks: [{y, height, svg}]}；分页由 video2score/engrave.py 做（一行谱不拆到两页），
// 版权一栏（出处）也由它印在每页页脚，这里不画。
//
// 只用 alphaTab 公开的设置和接口，不改它的代码：
// - 连奏滑音那一拍写上 "sl."（GP 的谱上有，alphaTab 不画），用的是拍上的文字；
// - 字宽：Node 里 alphaTab 量不了，按每字 8 像素估；engrave.py 用本机字体量好了传进来。
import fs from 'fs';
import * as alphaTab from '@coderline/alphatab';

const [input, output, widthText, barsText, fontsFile] = process.argv.slice(2);
if (fontsFile && fs.existsSync(fontsFile)) {
  const fonts = JSON.parse(fs.readFileSync(fontsFile, 'utf8'));
  for (const [family, data] of Object.entries(fonts)) {
    alphaTab.platform.FontSizes.FontSizeLookupTables.set(family, new alphaTab.platform.FontSizeDefinition(data.widths, data.height));
  }
}
const settings = new alphaTab.Settings();
settings.core.engine = 'svg';
settings.core.enableLazyLoading = false;
settings.display.layoutMode = alphaTab.LayoutMode.Page;
settings.display.staveProfile = alphaTab.StaveProfile.ScoreTab;
settings.display.barsPerRow = Number(barsText || 4);
settings.display.justifyLastSystem = false;
// GP 的小节号是黑的（alphaTab 默认红色）。
settings.display.resources.barNumberColor = new alphaTab.model.Color(0, 0, 0, 255);
settings.notation.elements.set(alphaTab.NotationElement.ScoreCopyright, false);

const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(new Uint8Array(fs.readFileSync(input)), settings);
// 连着的空小节合成一个多小节休止（GP 导出的 PDF 也是这样）。
score.stylesheet.multiTrackMultiBarRest = true;
score.stylesheet.perTrackMultiBarRest = new Set([0]);
// 连奏滑音的那一拍写上 "sl."。
for (const track of score.tracks) {
  for (const staff of track.staves) {
    for (const bar of staff.bars) {
      for (const voice of bar.voices) {
        for (const beat of voice.beats) {
          if (!beat.text && beat.notes.some((note) => note.slideOutType === alphaTab.model.SlideOutType.Legato)) {
            beat.text = 'sl.';
          }
        }
      }
    }
  }
}
const renderer = new alphaTab.rendering.ScoreRenderer(settings);
renderer.width = Number(widthText || 950);
const chunks = [];
renderer.partialRenderFinished.on((result) => {
  if (result.renderResult) {
    chunks.push({ y: result.y, height: result.height, svg: result.renderResult });
  }
});
renderer.error.on((error) => {
  console.error(String(error));
  process.exit(1);
});
renderer.renderFinished.on((result) => {
  chunks.sort((a, b) => a.y - b.y);
  fs.writeFileSync(output, JSON.stringify({ width: result.totalWidth, height: result.totalHeight, copyright: score.copyright, chunks }));
});
renderer.renderScore(score, [0]);
