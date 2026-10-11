"""文件都放在哪。

- 从仓库运行（开发）：作业在 jobs/，B 站 Cookie 在仓库根目录，和以前一样；界面用的上传、回收站、日志放在 jobs/ 下的 _uploads、_trash、_logs。
- 打包成 video2score 以后：所有用户数据都在 video2score/data/ 里（jobs、uploads、trash、config、logs），不往别处写；
  程序自带的资源（模板、alphaTab、网页、Node、打印 PDF 用的 chrome-headless-shell）在包里。删掉 video2score/ 就卸干净了。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "video2score"
FROZEN = bool(getattr(sys, "frozen", False))
PACKAGE = Path(__file__).resolve().parent


def _app_root() -> Path:
    if not FROZEN:
        return PACKAGE.parent
    executable = Path(sys.executable).resolve()
    # mac：video2score/video2score.app/Contents/MacOS/video2score → video2score/
    if executable.parent.name == "MacOS" and executable.parents[1].name == "Contents":
        return executable.parents[3]
    return executable.parent


APP_ROOT = _app_root()

if FROZEN:
    BUNDLE = Path(getattr(sys, "_MEIPASS", APP_ROOT))
    DATA_DIR = APP_ROOT / "data"
    JOBS_DIR = DATA_DIR / "jobs"
    UPLOADS_DIR = DATA_DIR / "uploads"
    TRASH_DIR = DATA_DIR / "trash"
    CONFIG_DIR = DATA_DIR / "config"
    LOGS_DIR = DATA_DIR / "logs"
    COOKIE_PATH = CONFIG_DIR / "bilibili_cookie.txt"
    RESOURCES = BUNDLE / "resources"
    GP_TEMPLATE = RESOURCES / "gp_template"
    ALPHATAB = BUNDLE / "alphatab"
    WEB = BUNDLE / "web"
    BUNDLED_NODE = BUNDLE / "node" / ("node.exe" if os.name == "nt" else "node")
    if sys.platform == "darwin":
        # mac 上浏览器原样放在 video2score.app/Contents/Resources/chrome/（见 packaging/build.py）。
        BUNDLED_BROWSER = Path(sys.executable).resolve().parents[1] / "Resources" / "chrome" / "chrome-headless-shell"
    else:
        BUNDLED_BROWSER = BUNDLE / "chrome" / "chrome-headless-shell.exe"
else:
    BUNDLE = PACKAGE
    DATA_DIR = APP_ROOT / "jobs"
    JOBS_DIR = DATA_DIR
    UPLOADS_DIR = DATA_DIR / "_uploads"
    TRASH_DIR = DATA_DIR / "_trash"
    CONFIG_DIR = APP_ROOT
    LOGS_DIR = DATA_DIR / "_logs"
    COOKIE_PATH = APP_ROOT / "bilibili_cookie.txt"
    RESOURCES = PACKAGE / "resources"
    GP_TEMPLATE = PACKAGE / "gp_template"
    ALPHATAB = PACKAGE / "alphatab"
    WEB = PACKAGE / "app" / "web"
    BUNDLED_NODE = PACKAGE / "alphatab" / "node" / ("node.exe" if os.name == "nt" else "node")
    # 开发时 python packaging/build.py --browser 下载到这里（不进仓库），和打包用的是同一个版本。
    BUNDLED_BROWSER = PACKAGE / "alphatab" / "chrome" / ("chrome-headless-shell.exe" if os.name == "nt" else "chrome-headless-shell")


def node() -> str | None:
    """跑 alphaTab 用的 Node：先用包里自带的，再找 PATH 里的。"""
    if BUNDLED_NODE.is_file():
        return str(BUNDLED_NODE)
    return shutil.which("node")


def translocated() -> bool:
    """mac 把未签名、带隔离标记的 .app 挪到随机的只读位置运行时，旁边的 data/ 找不到。"""
    return FROZEN and sys.platform == "darwin" and "/AppTranslocation/" in str(Path(sys.executable).resolve())
