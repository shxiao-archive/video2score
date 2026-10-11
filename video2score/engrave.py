"""把识别结果重新排版成 PDF（五线谱 + TAB，Guitar Pro 那种版式）。

先导出成 Guitar Pro 文件（gpfile.py），再用 alphaTab（照 Guitar Pro 版式排谱的开源库，Node 里跑 video2score/alphatab/render.mjs）
排成一块块 SVG：标题一块、每行谱一块。这里按 A4 分页（一行谱不拆到两页），页脚写"页码/总页数"，再由 Node 里的 Puppeteer
（video2score/alphatab/print.mjs）控制无界面 Chrome 打印成 PDF（MuPDF 画不了带字体的 SVG）。每页页脚左边印出处（谱的版权一栏：视频作者和链接）。

打包好的 video2score 自带 Node、alphaTab、Puppeteer 和固定版本的 chrome-headless-shell，每台电脑出来的 PDF 都一样。
从仓库运行时，第一次用之前在 video2score/alphatab 里运行一次 `npm install`（要装 Node.js）；浏览器用
`python packaging/build.py --browser` 下载同一版本的 chrome-headless-shell，没下载就用本机的 Chrome。
中间文件（连同浏览器的配置目录）放在输出 PDF 旁边的临时文件夹里，用完就删，不写系统临时目录。
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from video2score.paths import ALPHATAB, BUNDLED_BROWSER, RESOURCES, node

RENDERER = ALPHATAB
FONT_WIDTHS = RESOURCES / "font_widths.json"
_FONT = RENDERER / "node_modules" / "@coderline" / "alphatab" / "dist" / "font" / "Bravura.woff2"
# alphaTab 排版的宽度（像素）和每行小节数；按 A4（96 dpi 下 794 × 1123 像素）缩放，左右各留 15 毫米。
RENDER_WIDTH = 900
BARS_PER_ROW = 4
_PAGE_WIDTH, _PAGE_HEIGHT = 794, 1123
_MARGIN_X, _MARGIN_TOP, _MARGIN_BOTTOM = 57, 40, 55
# 从仓库运行、又没下载 chrome-headless-shell 时，用本机的 Chrome（也可能只装给当前用户）。
# Edge 不行：Windows 上 154 起 msedge.exe 只是个启动器，拉起真正的浏览器就退出，Puppeteer 接不上。
_BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
_BROWSER_NAMES = ("chrome", "google-chrome", "chromium", "chromium-browser")


def find_browser() -> str | None:
    """打印用的浏览器：环境变量 SCORE_BROWSER 优先，再用自带的 chrome-headless-shell，最后找本机的 Chrome。"""
    configured = os.environ.get("SCORE_BROWSER")
    if configured and Path(configured).is_file():
        return configured
    if BUNDLED_BROWSER.is_file():
        return str(BUNDLED_BROWSER)
    for path in _BROWSERS:
        path = os.path.expandvars(path)
        if Path(path).is_file():
            return path
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


# alphaTab 排版用到的字体（谱上的文字、TAB 数字）和本机的字体文件。
_FONT_FILES = {
    "Arial": ("arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"),
    "Georgia": ("georgia.ttf", "Georgia.ttf", "DejaVuSerif.ttf"),
    "Times New Roman": ("times.ttf", "Times New Roman.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"),
}


def font_widths() -> dict[str, dict]:
    """常用字体每个字符（0x20~0xFE）在 11 像素字号下的宽度，交给 alphaTab。

    alphaTab 在 Node 里量不了字宽，默认每个字都按 8 像素算：括号、X 这些字算得太宽，"(12)" 后面的滑音线就离得很远。
    这里用本机的字体文件按大字号量准了再缩回去；找不到字体的就不给，alphaTab 照旧估。
    """
    from PIL import ImageFont

    folders = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
        Path("/usr/share/fonts"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
    ]
    size = 1100
    out: dict[str, dict] = {}
    for family, names in _FONT_FILES.items():
        font = None
        for name in names:
            for folder in folders:
                candidates = [folder / name] + (list(folder.rglob(name)) if folder.is_dir() and not (folder / name).is_file() else [])
                found = next((path for path in candidates if path.is_file()), None)
                if found is not None:
                    font = ImageFont.truetype(str(found), size)
                    break
            if font is not None:
                break
        if font is None:
            continue
        ascent, descent = font.getmetrics()
        out[family] = {
            "widths": [round(font.getlength(chr(code)) * 11 / size, 3) for code in range(0x20, 0xFF)],
            "height": round((ascent + descent) / size, 3),
        }
    return out


def frozen_font_widths() -> dict[str, dict]:
    """存下来的字宽（resources/font_widths.json，`python -m video2score glyphs` 生成）；没有就用本机字体现量。"""
    if FONT_WIDTHS.is_file():
        return json.loads(FONT_WIDTHS.read_text(encoding="utf-8"))
    return font_widths()


def _scratch(folder: Path) -> tempfile.TemporaryDirectory:
    """放在 folder 里的临时文件夹（不用系统临时目录），用完删掉。"""
    folder.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=folder, prefix="tmp-", ignore_cleanup_errors=True)


def render_chunks(gp_file: Path, width: int = RENDER_WIDTH, bars_per_row: int = BARS_PER_ROW) -> dict:
    """调 alphaTab 排版：{width, height, chunks: [{y, height, svg}]}，按从上到下排好。"""
    runner = node()
    if runner is None:
        raise RuntimeError("导出 PDF 要用 Node.js 跑 alphaTab：没找到 node，请先安装 Node.js。")
    if not _FONT.is_file():
        raise RuntimeError(f"还没装 alphaTab：先在 {RENDERER} 里运行一次 npm install。")
    gp_file = Path(gp_file).resolve()
    with _scratch(gp_file.parent) as folder:
        out = Path(folder) / "chunks.json"
        widths = Path(folder) / "fonts.json"
        widths.write_text(json.dumps(frozen_font_widths()), encoding="utf-8")
        subprocess.run(
            [runner, str(RENDERER / "render.mjs"), str(gp_file), str(out), str(width), str(bars_per_row), str(widths)],
            check=True,
            timeout=300,
            cwd=RENDERER,
            capture_output=True,
        )
        return json.loads(out.read_text(encoding="utf-8"))


def paginate(chunks: list[dict], scale: float) -> list[list[dict]]:
    """按 A4 分页：一块一块往下排，放不下就换一页。一块（一行谱）不拆开；比一页还高的单独占一页。"""
    room = _PAGE_HEIGHT - _MARGIN_TOP - _MARGIN_BOTTOM
    pages: list[list[dict]] = [[]]
    used = 0.0
    for chunk in chunks:
        height = chunk["height"] * scale
        if pages[-1] and used + height > room:
            pages.append([])
            used = 0.0
        pages[-1].append(chunk)
        used += height
    return [page for page in pages if page]


def page_html(chunks: list[dict], width: int, credit: str = "") -> str:
    """每页一个 A4 大小的块：谱按比例缩到版心宽，页脚右边写页码/总页数（和 GP 导出的一样），左边写出处（链接可点）。"""
    scale = (_PAGE_WIDTH - 2 * _MARGIN_X) / width
    pages = paginate(chunks, scale)
    font = base64.b64encode(_FONT.read_bytes()).decode("ascii")
    body = []
    for number, page in enumerate(pages, start=1):
        blocks = "".join(_scaled_svg(chunk, width, scale) for chunk in page)
        source = f'<div class="source">{_linked(credit)}</div>' if credit else ""
        body.append(f'<section class="page"><div class="score">{blocks}</div>{source}<div class="footer">{number}/{len(pages)}</div></section>')
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        f"@font-face {{ font-family: 'alphaTab'; src: url(data:font/woff2;base64,{font}) format('woff2'); }}"
        "@page { size: A4; margin: 0 } html, body { margin: 0; background: white }"
        f".page {{ position: relative; width: {_PAGE_WIDTH}px; height: {_PAGE_HEIGHT}px; overflow: hidden; page-break-after: always }}"
        ".page:last-child { page-break-after: auto }"
        f".score {{ position: absolute; left: {_MARGIN_X}px; top: {_MARGIN_TOP}px }}"
        ".score svg { display: block; overflow: visible }"
        ".at { font-family: 'alphaTab'; font-size: 34px; font-style: normal; font-weight: normal; line-height: 1 }"
        f".footer {{ position: absolute; right: {_MARGIN_X}px; bottom: 30px; font: 11px Georgia, 'Times New Roman', serif }}"
        f".source {{ position: absolute; left: {_MARGIN_X}px; right: {_MARGIN_X + 60}px; bottom: 30px; font: 9px Arial, 'Microsoft YaHei', sans-serif; color: #555;"
        " white-space: nowrap; overflow: hidden; text-overflow: ellipsis }"
        ".source a { color: inherit; text-decoration: none }"
        "</style></head><body>" + "".join(body) + _FIT_TEXT + "</body></html>"
    )


# alphaTab 不给标题换行：比谱面还宽的一行字（长标题）打印前按比例缩小字号，免得超出纸边。
_FIT_TEXT = """<script>
for (const svg of document.querySelectorAll('.score svg')) {
  const width = svg.viewBox.baseVal.width;
  for (const text of svg.querySelectorAll('text')) {
    const box = text.getBBox();
    if (box.width > width) {
      text.style.fontSize = (parseFloat(getComputedStyle(text).fontSize) * width / box.width * 0.98) + 'px';
    }
  }
}
</script>"""


def _linked(text: str) -> str:
    """转义成 HTML，里面的网址做成链接。"""
    parts = re.split(r"(https?://\S+)", text)
    return "".join(
        f'<a href="{html.escape(part)}">{html.escape(part)}</a>' if index % 2 else html.escape(part) for index, part in enumerate(parts)
    )


def _scaled_svg(chunk: dict, width: int, scale: float) -> str:
    """alphaTab 的每块 SVG 是按排版宽度的像素大小：加上 viewBox，按版心宽缩小（不用 CSS transform，打印时会被裁掉）。"""
    svg = chunk["svg"]
    height = float(chunk["height"])
    start = svg.index("<svg")
    end = svg.index(">", start)
    head = svg[start:end]
    for name in ("width", "height"):
        head = re.sub(rf'\s{name}="[^"]*"', "", head)
    head += f' viewBox="0 0 {width} {height:g}" width="{width * scale:.2f}" height="{height * scale:.2f}"'
    return svg[:start] + head + svg[end:]


def gp_to_pdf(gp_file: Path, pdf: Path, browser: str | None = None) -> Path:
    """把 Guitar Pro 文件（.gp / .gpx / .gp5 都行）排版并打印成 PDF。"""
    browser = browser or find_browser()
    if browser is None:
        raise RuntimeError(
            "导出 PDF 要用无界面 Chrome 打印：没找到浏览器。请运行 python packaging/build.py --browser 下载，"
            "或安装 Chrome，或用环境变量 SCORE_BROWSER 指定。"
        )
    runner = node()
    if runner is None:
        raise RuntimeError("导出 PDF 要用 Node.js：没找到 node，请先安装 Node.js。")
    if not (RENDERER / "node_modules" / "puppeteer-core").is_dir():
        raise RuntimeError(f"还没装 Puppeteer：先在 {RENDERER} 里运行一次 npm install。")
    rendered = render_chunks(gp_file)
    document = page_html(rendered["chunks"], int(rendered["width"]) or RENDER_WIDTH, str(rendered.get("copyright") or ""))
    pdf = pdf.resolve()
    pdf.unlink(missing_ok=True)
    with _scratch(pdf.parent) as folder:
        page = Path(folder) / "score.html"
        page.write_text(document, encoding="utf-8")
        result = subprocess.run(
            [runner, str(RENDERER / "print.mjs"), str(page), str(pdf), browser, str(Path(folder) / "profile")],
            cwd=RENDERER,
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0 or not pdf.is_file():
            output = (result.stdout + result.stderr).decode("utf-8", errors="replace").strip().splitlines()[-15:]
            raise RuntimeError("浏览器打印 PDF 失败。" + ("\n" + "\n".join(output) if output else ""))
    return pdf


def musicxml_to_pdf(musicxml: Path, pdf: Path, browser: str | None = None) -> Path:
    """MusicXML 先转成 .gp（和导出的 Guitar Pro 文件一样），再排版成 PDF。"""
    from video2score.gpfile import read_score, write_gp

    pdf = Path(pdf).resolve()
    with _scratch(pdf.parent) as folder:
        gp_file = Path(folder) / "score.gp"
        write_gp(gp_file, read_score(musicxml))
        return gp_to_pdf(gp_file, pdf, browser)
