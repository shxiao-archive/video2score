"""把用户粘贴的东西变成本地视频文件：本地路径、B 站链接或 BV 号、YouTube 和其他视频网站链接（yt-dlp）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

_BVID_RE = re.compile(r"BV[0-9A-Za-z]{10}")


def classify_source(source: str) -> tuple[str, str]:
    """返回 (类型, 引用)。类型是 local、bilibili 或 url。"""
    text = source.strip().strip('"')
    if not text:
        raise ValueError("缺少视频来源。")
    path = Path(text)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"不是文件：{path}")
        return "local", str(path.resolve())
    lowered = text.lower()
    if _BVID_RE.fullmatch(text) or "bilibili.com" in lowered or "b23.tv" in lowered:
        return "bilibili", text
    if lowered.startswith(("http://", "https://")):
        return "url", text
    raise FileNotFoundError(f"找不到视频文件，也不是链接：{text}")


def fetch_video(source: str, kind: str, download_dir: Path, progress: Callable[[str], None] | None = None) -> Path:
    """本地文件直接用；B 站走 fetch.py；YouTube 和其他网站交给 yt-dlp（progress 收到"取视频…… 35%"这样的下载进度）。"""
    if kind == "local":
        return Path(source.strip().strip('"')).resolve()
    if kind == "bilibili":
        from video2score.fetch import download_bilibili

        return download_bilibili(_expand_short_link(source.strip()), download_dir / "video.mp4")
    return _download_with_ytdlp(source.strip(), download_dir, progress)


def ytdlp_options() -> dict[str, Any]:
    """yt-dlp 的公共设置：不打印进度条；YouTube 要用 JavaScript 运行环境解签名，用包里带的 Node（没有就找系统的）。
    代理照系统设置（环境变量 HTTP(S)_PROXY，或 Windows / macOS 的系统代理）。"""
    from video2score.paths import node

    options: dict[str, Any] = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # YouTube 对一次拉完的大文件常中途回 403：分成 10 MB 一段请求，出错重试。
        "http_chunk_size": 10 * 1024 * 1024,
        "retries": 10,
        "fragment_retries": 10,
    }
    runner = node()
    if runner is not None:
        options["js_runtimes"] = {"node": {"path": runner}}
    return options


def _expand_short_link(source: str) -> str:
    """b23.tv 短链里没有 BV 号，先跟一次跳转。"""
    if _BVID_RE.search(source) or "b23.tv" not in source.lower():
        return source
    from curl_cffi import requests

    response = requests.get(source, impersonate="chrome124", allow_redirects=True, timeout=20)
    return str(response.url)


def _download_with_ytdlp(url: str, download_dir: Path, progress: Callable[[str], None] | None = None) -> Path:
    """只要视频轨。优先 H.264，OpenCV 读 AV1 和部分 VP9 会失败。"""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("缺少 yt-dlp。请执行 pip install -r requirements.txt。") from exc
    download_dir.mkdir(parents=True, exist_ok=True)
    options = {
        **ytdlp_options(),
        "format": (
            "bestvideo[vcodec^=avc1][height<=1080]/"
            "bestvideo[height<=1080]/best[height<=1080]/best"
        ),
        "outtmpl": str(download_dir / "video.%(ext)s"),
    }
    if progress is not None:
        last = [-1]

        def hook(status: dict[str, Any]) -> None:
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if status.get("status") != "downloading" or not total:
                return
            percent = int(100 * (status.get("downloaded_bytes") or 0) / total)
            # 每 5% 报一次，免得命令行刷屏。
            if percent >= last[0] + 5 or (percent == 100 and last[0] != 100):
                last[0] = percent
                progress(f"取视频…… {percent}%")

        options["progress_hooks"] = [hook]
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        path = Path(downloader.prepare_filename(info))
    if not path.is_file():
        found = sorted(download_dir.glob("video.*"))
        if not found:
            raise RuntimeError(f"yt-dlp 没有下载到视频：{url}")
        path = found[0]
    return path
