# PyInstaller 打包配置（onedir）。由 packaging/build.py 调用，不要直接跑：它先准备好 build/node 里的 Node、build/chrome 里的浏览器。
#
# 包里的布局（paths.py 按这个找）：
#   _internal/（mac 是 video2score.app/Contents/Frameworks 与 Resources）
#     resources/            glyphs.npz、font_widths.json、gp_template/
#     web/                  界面的 index.html、app.js、style.css
#     alphatab/             render.mjs、print.mjs、node_modules（alphaTab、Puppeteer）
#     node/node(.exe)       自带的 Node
#     chrome/               打印 PDF 用的 chrome-headless-shell（固定版本；mac 上不经过 PyInstaller，
#                           由 build.py 原样拷进 video2score.app/Contents/Resources/chrome/）
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
PACKAGE = ROOT / "video2score"
NODE = ROOT / "build" / "node" / ("node.exe" if sys.platform == "win32" else "node")
CHROME = ROOT / "build" / "chrome"
ICONS = ROOT / "packaging" / "icons"

datas = [
    (str(PACKAGE / "resources" / "glyphs.npz"), "resources"),
    (str(PACKAGE / "resources" / "font_widths.json"), "resources"),
    (str(PACKAGE / "gp_template"), "resources/gp_template"),
    (str(PACKAGE / "app" / "web"), "web"),
    (str(PACKAGE / "alphatab" / "render.mjs"), "alphatab"),
    (str(PACKAGE / "alphatab" / "print.mjs"), "alphatab"),
    (str(PACKAGE / "alphatab" / "package.json"), "alphatab"),
    (str(PACKAGE / "alphatab" / "node_modules"), "alphatab/node_modules"),
]
if sys.platform == "win32":
    # 整个文件夹原样放进去：可执行文件要和旁边的 .pak、icudtl.dat、动态库在一起。
    # mac 上 PyInstaller 会去改动态库的装载路径（Google 的库不让改，改了签名也就坏了），所以 mac 另外拷。
    datas.append((str(CHROME), "chrome"))
binaries = [(str(NODE), "node")]
hiddenimports = collect_submodules("video2score") + collect_submodules("yt_dlp")
# 这几个库没有现成的 PyInstaller hook：把它们的数据文件和动态库都带上。
for package in ("curl_cffi", "pymupdf", "certifi", "qrcode", "yt_dlp_ejs"):
    found = collect_all(package)
    datas += found[0]
    binaries += found[1]
    hiddenimports += found[2]

analysis = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "onnxruntime", "scipy", "pandas", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="video2score",
    icon=str(ICONS / ("video2score.ico" if sys.platform == "win32" else "video2score.icns")),
    # Windows 留一个控制台窗口：显示地址和日志，关掉它就退出。mac 的 .app 没有窗口，在界面上点"退出"。
    console=sys.platform != "darwin",
    upx=False,
)
collected = COLLECT(exe, analysis.binaries, analysis.datas, name="video2score", upx=False)

if sys.platform == "darwin":
    from video2score import __version__

    app = BUNDLE(
        collected,
        name="video2score.app",
        icon=str(ICONS / "video2score.icns"),
        bundle_identifier="io.github.shxiao-archive.video2score",
        version=__version__,
        info_plist={
            "CFBundleDisplayName": "video2score",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
